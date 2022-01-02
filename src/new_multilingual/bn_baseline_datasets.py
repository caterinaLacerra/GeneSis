import argparse
from typing import Dict, Set, List, Iterator, Tuple, Optional

import numpy as np
import torch
import tqdm
import transformers

from sklearn.metrics.pairwise import cosine_similarity

from src.utils import file_len, flatten, extract_word_embedding
from src.utils_wsd import read_from_raganato, WSDInstance
from src.wsd.utils.utils import LexSubInstance


def load_candidates(candidates_path: str) -> Dict[str, Set[str]]:
    candidates = {}
    for line in open(candidates_path):
        bn_id, *substitutes = line.strip().split('\t')
        if bn_id not in candidates:
            candidates[bn_id] = set(substitutes)
    return candidates


def replace_target_with_substitutes(current_idx: int, token: WSDInstance, text_sentence: str, candidates: Dict[str, Set[str]]) -> \
        Tuple[str, Optional[str], List[int], List[str], List[str], List[List[int]]]:

    lexeme = f"{token.annotated_token.lemma}.{token.annotated_token.pos}"
    instance_id = token.instance_id
    target_idx = [x for x in range(current_idx, current_idx + len(token.annotated_token.text.split()))]

    assert " ".join([text_sentence.split()[x] for x in target_idx]) == token.annotated_token.text

    associated_candidates = [x for bn_id in token.labels for x in candidates[bn_id]
                             if bn_id in candidates
                             if token.annotated_token.lemma != x]

    replaced_sentences = [" ".join(text_sentence.split()[:target_idx[0]]
                                   + [substitute]
                                   + text_sentence.split()[target_idx[-1] + 1:])
                          for substitute in associated_candidates]

    replaced_indexes = [[x for x in range(target_idx[0], target_idx[0] + len(substitute.split()))]
                        for substitute in associated_candidates]

    stack_sentences = [text_sentence] + replaced_sentences
    stack_indexes = [target_idx] + replaced_indexes

    return lexeme, instance_id, target_idx, associated_candidates, stack_sentences, stack_indexes


def get_sorted_substitutes(
        batch_sentences: List[List[str]],
        batch_indexes: List[List[List[int]]],
        batch_candidates: List[List[str]],
        batch_instances: List[LexSubInstance],
        tokenizer: transformers.AutoTokenizer.from_pretrained,
        embedder: transformers.AutoModel.from_pretrained,
        device: torch.device,
        config: transformers.AutoConfig.from_pretrained,
        special_chars: str
) -> Iterator[LexSubInstance]:

    hidden_states, target_bpes = extract_word_embedding(flatten(batch_sentences),
                                                        flatten(batch_indexes),
                                                        tokenizer, embedder, device,
                                                        config, special_chars)

    # target, substitutes vectors
    vectors = torch.zeros((hidden_states.shape[0], hidden_states.shape[-1]))
    assert  len(target_bpes) == len(flatten(batch_sentences)), print(len(vectors), len(target_bpes), len(flatten(batch_sentences)))

    for i in range(len(flatten(batch_sentences))):
        if target_bpes[i] is None:
            continue

        vectors[i] = torch.mean(torch.stack([hidden_states[i][bpe] for bpe in target_bpes[i]]),
                                dim=0).cpu()

    flattened_idx = 0
    for i in range(len(batch_sentences)):

        # skip instances without a match between target and bpes
        if target_bpes[flattened_idx] is None:
            continue

        if len(batch_candidates[i]) == 0:
            continue

        # recover target-substitutes from flattened lists
        control_indexes = [x for x in range(flattened_idx, flattened_idx + len(batch_sentences[i]))]

        # if i only have one substitute:
        if len(batch_candidates[i]) == 1:
            assert len(control_indexes) == 2
            cos_sim = cosine_similarity(vectors[control_indexes[0]].reshape(1, -1),
                                        vectors[control_indexes[-1]].reshape(1, -1))[0]

        else:
            assert len(batch_candidates[i]) == len([x for x in range(control_indexes[0] + 1, control_indexes[-1] + 1)])
            cos_sim = cosine_similarity(vectors[control_indexes[0]].reshape(1, -1),
                                        vectors[control_indexes[0] + 1: control_indexes[-1] + 1])[0]

        sorted_indexes = np.argsort(cos_sim)[::-1]
        sorted_substitutes = [(batch_candidates[i][x], str(int(cos_sim[x]*100))) for x in sorted_indexes]
        flattened_idx += len(batch_sentences[i])

        yield LexSubInstance(batch_instances[i].target, batch_instances[i].instance_id, batch_instances[i].target_idx,
                             batch_instances[i].sentence, mask=None, gold = {subst: score
                                                                             for subst, score in sorted_substitutes})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_xml', required=True)
    parser.add_argument('--gold_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--candidates_path', required=True)
    parser.add_argument('--model_name', required=False, default='xlm-roberta-large')
    parser.add_argument('--cpu', action="store_true", default=False)
    parser.add_argument('--max_tokens_batch', type=int, default=25000)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    SPECIAL_CHARS = {"xlm-roberta-large": "▁"}

    candidates = load_candidates(args.candidates_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    embedder = transformers.AutoModel.from_pretrained(args.model_name)
    model_config = transformers.AutoConfig.from_pretrained(args.model_name)

    if not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tot_instances = 0
    instances_with_candidates = 0

    with open(args.output_path, 'w') as out, tqdm.tqdm(total=file_len(args.gold_path)) as pbar:

        batch_instances, batch_sentences, batch_indexes, batch_candidates = [], [], [], []

        for _, _, sentence in read_from_raganato(args.input_xml, args.gold_path):

            text_sentence = " ".join([token.annotated_token.text for token in sentence])
            
            current_idx = 0
            current_batch_len = 0

            for token in sentence:

                if token.labels is None:
                    current_idx += len(token.annotated_token.text.split())
                    continue

                lexeme, instance_id, target_idx, \
                associated_candidates, stack_sentences, stack_indexes = replace_target_with_substitutes(
                    current_idx, token, text_sentence, candidates
                )

                if len(associated_candidates) ==  0:
                    tot_instances += 1
                    current_idx += len(token.annotated_token.text.split())
                    pbar.update()
                    continue


                # tokenize batch
                if len(batch_sentences) > 0:
                    tokenized_batch = tokenizer.batch_encode_plus(flatten(batch_sentences),
                                                                  padding=True,
                                                                  truncation=True)['input_ids']

                    current_batch_len = len(tokenized_batch) * len(tokenized_batch[0])


                tokenized_next_batch = tokenizer.batch_encode_plus(stack_sentences,
                                                                   padding=True,
                                                                   truncation=True)['input_ids']

                next_batch_len = len(tokenized_next_batch) * len(tokenized_next_batch[0])

                if current_batch_len + next_batch_len < args.max_tokens_batch:
                    batch_instances.append(LexSubInstance(lexeme, instance_id, target_idx, text_sentence))
                    batch_sentences.append(stack_sentences)
                    batch_indexes.append(stack_indexes)
                    batch_candidates.append(associated_candidates)

                else:
                    for lexsub_instance in get_sorted_substitutes(batch_sentences, batch_indexes, batch_candidates,
                                                                  batch_instances, tokenizer, embedder, device,
                                                                  model_config, SPECIAL_CHARS[args.model_name]):
                        instances_with_candidates += 1
                        out.write(f"{repr(lexsub_instance)}\n")

                    batch_sentences = []
                    batch_instances = []
                    batch_indexes = []
                    batch_candidates = []

                    batch_instances.append(LexSubInstance(lexeme, instance_id, target_idx, text_sentence))
                    batch_sentences.append(stack_sentences)
                    batch_indexes.append(stack_indexes)
                    batch_candidates.append(associated_candidates)

                pbar.update()
                tot_instances += 1
                current_idx += len(token.annotated_token.text.split())

    print(f"tot instances: {tot_instances}, with candidates: {instances_with_candidates}")