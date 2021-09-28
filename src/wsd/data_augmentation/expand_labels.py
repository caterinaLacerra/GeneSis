import argparse
import os
import random

import lemminflect
import tqdm
from nltk.corpus import wordnet as wn
from typing import List, Dict

import stanza

from src.utils import yield_batch, get_target_index_list, universal_to_wn_pos, file_len, WSDInstance
from src.wordnet_utils import synset_from_sensekey


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--semcor_path', required=True)
    parser.add_argument('--semcor_keys', required=True)
    parser.add_argument('--output_folder', required=True)
    return parser.parse_args()


def clean_substitutes(target: str, substitutes: List[str]) -> List[str]:

    *lemma, pos = target.split('.')
    clean_list = []

    target_synsets = wn.synsets("_".join(".".join(lemma).split()), pos=universal_to_wn_pos(pos))
    target_lemmas = set([l for s in target_synsets for l in s.lemmas()])

    for substitute in substitutes:

        sub = lemminflect.getLemma(substitute, upos=pos)[0]
        sub = "_".join(sub.split())

        if sub == "_".join(lemma):
            continue

        # check if word in wordnet (with target pos)
        synsets = wn.synsets(sub, universal_to_wn_pos(pos))

        if len(synsets) == 0:
            continue

        if len(substitute) == 1:
            continue

        sub_antonyms_lemmas = set()
        for s in synsets:
            for l in s.lemmas():
                sub_antonyms_lemmas.update(l.antonyms())

        # remove antonyms from substitutes
        if len(sub_antonyms_lemmas.intersection(target_lemmas)) != 0:
            continue

        clean_list.append(substitute)

    return clean_list


def expand_label(target: str, target_sense: List[str], substitutes: List[str]) -> Dict[str, dict]:
    *lemma, pos = target.split('.')
    sub_to_sense = {}

    for sub in substitutes:
        lemma_sub = lemminflect.getLemma(sub, upos=pos)[0]
        sub = "_".join(lemma_sub.split())

        synsets = wn.synsets(sub, universal_to_wn_pos(pos))

        # monosemous substitutes
        if len(synsets) == 1:
            lemmas = synsets[0].lemmas()
            sub_to_sense[sub] = {l.key(): "monosemous" for l in lemmas if l.key().startswith(sub+'%')}

        else:
            # pick the target synset and select the closest sense among those available for the substitute
            target_synset = synset_from_sensekey(random.choice(target_sense))

            paths = []
            for s in synsets:
                n_edges = target_synset.shortest_path_distance(s)
                if n_edges is None:
                    n_edges = 1000
                paths.append((n_edges, s, s.definition()))

            sorted_paths = sorted(paths, key=lambda x:x[0])
            try:
                selected_synset = sorted_paths[0][1]

            except:
                print(synsets)
                print(sub)
                exit()

            lemmas = selected_synset.lemmas()
            sub_to_sense[sub] = {l.key(): sorted_paths[0][0] for l in lemmas if l.key().startswith(sub+'%')}

    return sub_to_sense


if __name__ == '__main__':
    args = parse_args()

    semcor_keys = {line.strip().split()[0]:line.strip().split()[1:]
                   for line in open(args.semcor_keys)}

    monosemous_additional_instances, possible_additional, non_null_additional_instances = 0, 0, 0

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    output_path = os.path.join(args.output_folder, 'data_augmentation_hr.txt')
    monos_path = os.path.join(args.output_folder, 'monosemous_augmentation.tsv')
    full_path = os.path.join(args.output_folder, 'full_augmentation.tsv')
    non_null_path = os.path.join(args.output_folder, 'non_null_augmentation.tsv')

    with open(output_path, 'w') as out, tqdm.tqdm(total=file_len(args.semcor_keys)) as pbar, \
        open(monos_path, 'w') as mono, open(full_path, 'w') as full, open(non_null_path, 'w') as non_null:

        for batch in yield_batch(args.semcor_path, separator='#########\n'):

            target, instance_id, *target_index = batch[0].strip().split()
            *lemma, pos = target.split('.')
            lemma = '.'.join(lemma)


            if lemma in ["person", "organization", "location", "group"]:
                pbar.update()
                continue

            sentence = batch[1].strip()
            substitutes = set()

            for line in batch[2: -1]:
                substitutes.update(line.strip().split(", "))

            substitutes = list(substitutes)

            target_index = get_target_index_list(" ".join(target_index))
            filtered_subst = clean_substitutes(target, substitutes)


            if len(filtered_subst) > 0:

                out.write(f"{target}\t{sentence}\n")

                sub_to_selected_senses = expand_label(target, semcor_keys[instance_id], filtered_subst)
                out.write(f"gold sense:\t{semcor_keys[instance_id]}\t"
                          f"{synset_from_sensekey(random.choice(semcor_keys[instance_id])).definition()}\n\n")

                for k, values in sub_to_selected_senses.items():
                    for sensekey in values:
                        out.write(f"{k}\t{sensekey}\t{values[sensekey]}\t{synset_from_sensekey(sensekey).definition()}\n")

                        if values[sensekey] == "monosemous":
                            monosemous_additional_instances += 1
                            # todo: write to the output files
                            #word = lemminflect.getInflection()
                            # todo: replace " " with "_" before creating the instance
                            #instance = WSDInstance(k, pos, word, new_sentence, sensekey)
                        elif values[sensekey] != 1000:
                            non_null_additional_instances += 1

                        possible_additional += 1

                out.write("=========================\n")
            pbar.update()

    print(f"Monosemous additional: {monosemous_additional_instances}")
    print(f"Non-null (+ monosemous) additional: {non_null_additional_instances}")
    print(f"Maximum possible additional: {possible_additional}")