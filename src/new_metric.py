from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import transformers

from src.utils import recover_bpes


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
    List[List[List[int]]], List[str]]:

    new_indexes, new_sentences = [], []

    for i in range(len(target_indexes)):

        left_context = sentences[i].split()[:target_indexes[i][0]]
        right_context = sentences[i].split()[target_indexes[i][-1] + 1:]

        i_th_indexes = []

        for substitute in substitutes[i]:
            sep = "_" if "_" in substitute else " "
            new_target = substitute.split(sep)
            new_sentences.append(" ".join(left_context + new_target + right_context))
            i_th_indexes.append([x for x in range(target_indexes[i][0], target_indexes[i][0] + len(new_target))])

        new_indexes.append(i_th_indexes)

    return new_indexes, new_sentences

def compute_vectors(input_contexts: List[str], target_words: List[str],
                     substitutes: List[List[str]], target_index: List[List[int]],
                     tokenizer: transformers.AutoTokenizer.from_pretrained,
                     embedder: transformers.AutoModelForMaskedLM.from_pretrained,
                     device: torch.device,
                     model_config: transformers.AutoConfig) -> List[Dict[str, Any]]:

    idx_to_token = {v: k for k, v in tokenizer.get_vocab().items()}

    # get the indexes for all the occurrences of the target
    occurrences_idx = [get_occurrences_indexes(input_context, target_word)
                       for input_context, target_word in zip(input_contexts, target_words)]

    # embed sentences as the average of the last four layers
    tokenized = tokenizer.batch_encode_plus(input_contexts, return_tensors='pt', padding=True, truncation=True)
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    layer_indexes = [model_config.num_hidden_layers - 4, model_config.num_hidden_layers]

    embedder.to(device)
    with torch.no_grad():
        hidden_states = embedder(input_ids, attention_mask, output_hidden_states=True)["hidden_states"]

    hidden_states = torch.mean(torch.stack(hidden_states[layer_indexes[0]:layer_indexes[-1] + 1]), dim=0)
    words = [[x for x in sentence.split(' ') if x != ""] for sentence in input_contexts]
    bpes = [[idx_to_token[idx.item()] for idx in sentence] for sentence in input_ids]

    sentence_avg_vectors = torch.zeros((len(input_contexts), model_config.hidden_size), device=device)
    target_substitutes = torch.zeros((len(input_contexts), model_config.hidden_size), device=device)

    for j in range(len(input_ids)):
        # for each sentence, consider all the occurrences of the target and retrieve the corresponding bpes
        exclude_bpes = []
        original_target_idx = []

        for target_list_idx in occurrences_idx[j]:
            for tix in target_list_idx:
                bpes_idx = recover_bpes(bpes[j], words[j], tix, tokenizer)
                if bpes_idx is None:
                    continue

                reconstruct = ''.join(bpes[j][bpes_idx[0]:bpes_idx[-1] + 1]).replace('##', '')
                target = words[j][tix]

                if target != reconstruct:
                    continue

                exclude_bpes.extend(bpes_idx)
                if tix in target_index[j]:
                    original_target_idx.extend(bpes_idx)

        # exclude target occurrences
        sentence_avg_vectors[j] = torch.mean(torch.stack([hidden_states[j][x] for x in range(len(hidden_states[j]))
                                                          if x not in exclude_bpes]), dim=0)

        # consider only (original) target occurrence
        target_substitutes[j] = torch.mean(torch.stack([hidden_states[j][x] for x in range(len(hidden_states[j]))
                                                        if x in original_target_idx]))

    substitutes_indexes, substitutes_sentences = replace_target(input_contexts, target_index, substitutes)

    # todo: embed substitutes sentences and retrieve only the substitutes vectors
    # todo: return as dictionary
    """ 
    # embed sentences as the average of the last four layers
    tokenized = tokenizer.batch_encode_plus(input_contexts, return_tensors='pt', padding=True, truncation=True)
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    layer_indexes = [model_config.num_hidden_layers - 4, model_config.num_hidden_layers]

    embedder.to(device)
    with torch.no_grad():
        hidden_states = embedder(input_ids, attention_mask, output_hidden_states=True)["hidden_states"]

    hidden_states = torch.mean(torch.stack(hidden_states[layer_indexes[0]:layer_indexes[-1] + 1]), dim=0)
    words = [[x for x in sentence.split(' ') if x != ""] for sentence in input_contexts]
    bpes = [[idx_to_token[idx.item()] for idx in sentence] for sentence in input_ids]

    sentence_avg_vectors = torch.zeros((len(input_contexts), model_config.hidden_size), device=device)
    target_substitutes = torch.zeros((len(input_contexts), model_config.hidden_size), device=device)
"""

if __name__ == '__main__':

    sentences = ["The cat is on the table", "We bought a new table"]
    target_indexes = [[0, 1], [4]]
    substitutes = [["animal", "feline animal"], ["furniture"]]

    new_indexes, new_sentences = replace_target(sentences, target_indexes, substitutes)

    i = 0
    for list_indexes in new_indexes:
        for j in range(len(list_indexes)):
            print(list_indexes[j], new_sentences[i])
            i+=1
        print("*****")