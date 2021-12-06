import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import tqdm
import transformers

from src.utils import recover_bpes, read_from_input_file, file_len
from sklearn.metrics.pairwise import cosine_similarity

def get_occurrences_indexes(input_sentence: str, target_word: str) -> List[List[int]]:

    separator = "_" if "_" in target_word else " "
    target_words = target_word.lower().split(separator)

    if len(target_words) == 1:
        target_indexes = [[i] for i, word in enumerate(input_sentence.lower().split()) if word == target_words[0]]
        return target_indexes

    i = 0
    target_indexes = []
    while i < len(input_sentence.split()):
        if i + len(target_words) < len(input_sentence.split()):
            window = input_sentence.lower().split()[i: i + len(target_words)]

            if "_".join(window) == "_".join(target_words):
                target_indexes.append([x for x in range(i, i + len(target_words))])

        i += len(target_words)

    return target_indexes

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


def extract_word_embedding(input_sentences: List[str], target_indexes: List[List[int]],
                           tokenizer: transformers.AutoTokenizer.from_pretrained,
                           embedder: transformers.AutoModelForMaskedLM.from_pretrained,
                           device: torch.device,
                           model_config: transformers.AutoConfig):

    idx_to_token = {v: k for k, v in tokenizer.get_vocab().items()}

    # embed sentences as the average of the last four layers
    tokenized = tokenizer.batch_encode_plus(input_sentences, return_tensors='pt', padding=True, truncation=True)
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    layer_indexes = [model_config.num_hidden_layers - 4, model_config.num_hidden_layers]

    embedder.to(device)
    embedder.eval()
    with torch.no_grad():
        hidden_states = embedder(input_ids, attention_mask, output_hidden_states=True)["hidden_states"]

    hidden_states = torch.mean(torch.stack(hidden_states[layer_indexes[0]:layer_indexes[-1] + 1]), dim=0)
    words = [[x for x in sentence.split(' ') if x != ""] for sentence in input_sentences]
    bpes = [[idx_to_token[idx.item()] for idx in sentence] for sentence in input_ids]

    target_bpes = []

    for j in range(len(input_ids)):
        # for each sentence, consider all the occurrences of the target and retrieve the corresponding bpes

        for target_list_idx in target_indexes[j]:
            bpes_idx = recover_bpes(bpes[j], words[j], target_list_idx, tokenizer)
            if bpes_idx is None:
                continue

            reconstruct = ''.join(bpes[j][bpes_idx[0]:bpes_idx[-1] + 1]).replace('##', '')
            target = words[j][target_list_idx]

            if target != reconstruct:
                continue

            target_bpes.append(bpes_idx)

    return hidden_states, target_bpes

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