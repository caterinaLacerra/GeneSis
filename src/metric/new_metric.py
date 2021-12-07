from typing import List, Dict, Tuple

import numpy as np
import torch
import tqdm
import transformers

from src.new_multilingual.bn_baseline_datasets import get_sorted_substitutes
from src.utils import read_from_input_file, file_len, flatten, extract_word_embedding
from sklearn.metrics.pairwise import cosine_similarity

SPECIAL_CHARS = {"xlm-roberta-large": "▁",
                 "bert-large-cased": "##"}


def replace_target(sentences: List[str], target_indexes: List[List[int]], substitutes: List[List[str]]) -> Tuple[
    List[List[List[int]]], List[List[str]]]:

    new_indexes, new_sentences = [], []

    for i in range(len(target_indexes)):

        left_context = sentences[i].split()[:target_indexes[i][0]]
        right_context = sentences[i].split()[target_indexes[i][-1] + 1:]

        i_th_indexes = []
        i_th_sentences = []

        for substitute in substitutes[i]:
            sep = "_" if "_" in substitute else " "
            new_target = substitute.split(sep)
            i_th_sentences.append(" ".join(left_context + new_target + right_context))
            i_th_indexes.append([x for x in range(target_indexes[i][0], target_indexes[i][0] + len(new_target))])

        new_indexes.append(i_th_indexes)
        new_sentences.append(i_th_sentences)

    return new_indexes, new_sentences


def compute_vectors(input_contexts: List[str],
                    instance_ids: List[str],
                    substitutes: List[List[str]],
                    target_index: List[List[int]],
                    tokenizer: transformers.AutoTokenizer.from_pretrained,
                    embedder: transformers.AutoModelForMaskedLM.from_pretrained,
                    device: torch.device,
                    model_config: transformers.AutoConfig) -> Dict[str, dict]:

    output_dict = {}

    hidden_states, target_bpes_indexes = extract_word_embedding(input_contexts,
                                                                target_index,
                                                                tokenizer, embedder,
                                                                device, model_config)

    sentence_avg_vectors = torch.zeros((len(input_contexts), model_config.hidden_size), device=device)
    # target_vectors = torch.zeros((len(input_contexts), model_config.hidden_size), device=device)

    for j in range(len(hidden_states)):
        # exclude target occurrences
        sentence_avg_vectors[j] = torch.mean(torch.stack([hidden_states[j][x] for x in range(len(hidden_states[j]))
                                                          if x not in target_bpes_indexes[j]]), dim=0)

        # consider only (original) target occurrence
        # target_vectors[j] = torch.mean(torch.stack([hidden_states[j][x] for x in range(len(hidden_states[j]))
        #                                                 if x in target_bpes_indexes[j]]))

        output_dict[instance_ids[j]] = {"original_sentence_vec": sentence_avg_vectors[j]}

    # extract substitutes indexes and replace targets with substitutes
    substitutes_indexes, substitutes_sentences = replace_target(input_contexts, target_index, substitutes)

    # embed sentences with substitutes
    for i, (indexes, substitute_s) in enumerate(zip(substitutes_indexes, substitutes_sentences)):
        hidden_states_substitute, subst_bpes_indexes = extract_word_embedding(substitute_s,
                                                                              indexes, tokenizer, embedder,
                                                                              device, model_config)
        for k in range(len(hidden_states_substitute)):
            subst_vec = torch.mean(torch.stack([hidden_states_substitute[k][x]
                                                for x in range(len(hidden_states_substitute[k]))
                                                if x in subst_bpes_indexes[k]]), dim=0)
            output_dict[instance_ids[i]][substitutes[i][k]] = subst_vec

    return output_dict


def sort_with_metric(input_path: str, model_name: str, batch_size: int):

    model_config = transformers.AutoConfig.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    embedder = transformers.AutoModel.from_pretrained(model_name)
    device = torch.device("cuda")

    batch = []
    instances_output = {}
    for instance in tqdm.tqdm(read_from_input_file(input_path), total=file_len(input_path)):
        batch.append(instance)

        if len(batch) > batch_size:
            output_vecs = compute_vectors([x.sentence for x in batch],
                                          [f"{x.target} {x.instance_id}" for x in batch],
                                          [list(x.gold) for x in batch],
                                          [x.target_idx for x in batch],
                                          tokenizer, embedder, device, model_config)
            instances_output.update(output_vecs)
            batch = []

    if len(batch) != 0:
        output_vecs = compute_vectors([x.sentence for x in batch],
                                      [f"{x.target} {x.instance_id}" for x in batch],
                                      [list(x.gold) for x in batch],
                                      [x.target_idx for x in batch],
                                      tokenizer, embedder, device, model_config)
        instances_output.update(output_vecs)

    sorted_results = {}
    for instance_id in instances_output:
        target = instances_output[instance_id]["original_sentence_vec"].cpu()
        substitute_keys = [k for k in instances_output[instance_id] if k != "original_sentence_vec"]

        substitutes_vec = torch.stack([instances_output[instance_id][s] for s in substitute_keys]).cpu()
        similarity = cosine_similarity(target.reshape(1, -1), substitutes_vec)[0]
        sorted_indexes = np.argsort(similarity)[::-1]
        sorted_results[instance_id] = [(substitute_keys[i], similarity[i]) for i in sorted_indexes]
    return sorted_results


def sort_with_cos_sim(input_path: str, model_name: str, max_tokens_batch: int, special_chars: str, device: torch.device):

    output_dict = {}

    model_config = transformers.AutoConfig.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    embedder = transformers.AutoModel.from_pretrained(model_name)


    batch_sentences = []
    batch_instances = []
    batch_indexes = []
    batch_candidates = []

    current_batch_len = 0

    with tqdm.tqdm(total=file_len(input_path)) as pbar:

        for instance in read_from_input_file(input_path):

            replaced_sentences = [" ".join(instance.sentence.split()[:instance.target_idx[0]]
                                           + [substitute]
                                           + instance.sentence.split()[instance.target_idx[-1] + 1:])
                                  for substitute in instance.gold]

            replaced_indexes = [[x for x in range(instance.target_idx[0], instance.target_idx[0] + len(substitute.split()))]
                                for substitute in instance.gold]

            stack_sentences = [instance.sentence] + replaced_sentences
            stack_indexes = [instance.target_idx] + replaced_indexes

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

            if current_batch_len + next_batch_len < max_tokens_batch:
                batch_instances.append(instance)
                batch_sentences.append(stack_sentences)
                batch_indexes.append(stack_indexes)
                batch_candidates.append(list(instance.gold.keys()))

            else:
                for lexsub_instance in get_sorted_substitutes(batch_sentences, batch_indexes, batch_candidates,
                                                              batch_instances, tokenizer, embedder, device,
                                                              model_config, special_chars):
                    output_dict[lexsub_instance.instance_id] = lexsub_instance

                pbar.update(len(batch_sentences))

                batch_sentences = []
                batch_instances = []
                batch_indexes = []
                batch_candidates = []

                batch_instances.append(instance)
                batch_sentences.append(stack_sentences)
                batch_indexes.append(stack_indexes)
                batch_candidates.append(list(instance.gold.keys()))

    return output_dict

if __name__ == '__main__':

    input_path = "data/lst_candidates.tsv"
    output_path = "data/lst_candidates.cossim_sorted.tsv"
    model_name = "bert-large-cased"
    max_tokens_batch = 10_000
    special_chars = SPECIAL_CHARS[model_name]

    device = torch.device("cpu")
    output_dict = sort_with_cos_sim(input_path, model_name, max_tokens_batch, special_chars, device)

    with open(output_path, 'w') as out:
        for instance_id, instance in output_dict.items():
            out.write(repr(instance) + '\n')
