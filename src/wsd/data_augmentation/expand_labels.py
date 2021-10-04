import argparse
import json
import os
import random

import lemminflect
import spacy
import tqdm
from nltk.corpus import wordnet as wn
from typing import List, Dict

from src.wsd.utils.utils import yield_batch, get_target_index_list, file_len
from src.wsd.utils.wordnet_utils import synset_from_sensekey, universal_to_wn_pos
from src.wsd.utils.utils_edo import RaganatoBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--semcor_path', required=True)
    parser.add_argument('--semcor_keys', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--mappings_path', default='data/mappings/semcor.json')
    parser.add_argument('--output_xml', required=True)
    parser.add_argument('--labels_output', required=True)
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
            selected_synset = sorted_paths[0][1]

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

    nlp = spacy.load("en_core_web_sm")

    output_path = os.path.join(args.output_folder, 'data_augmentation_hr.txt')
    monos_path = os.path.join(args.output_folder, 'monosemous_augmentation.tsv')
    non_null_path = os.path.join(args.output_folder, 'non_null_augmentation.tsv')

    builder = RaganatoBuilder()

    json_dict = json.load(open(args.mappings_path))

    text_id, sentence_id, token_id = 0, 0, 0

    builder.open_text_section(f"d{str(text_id).zfill(3)}")

    with open(output_path, 'w') as out, tqdm.tqdm(total=file_len(args.semcor_keys)) as pbar, \
        open(monos_path, 'w') as mono, open(non_null_path, 'w') as non_null:

        for batch in yield_batch(args.semcor_path, separator='#########\n'):

            target, instance_id, *target_index = batch[0].strip().split()
            *lemma, pos = target.split('.')
            lemma = '.'.join(lemma)
            sent_id = '.'.join(instance_id.split('.')[:-1])

            if lemma in ["person", "organization", "location", "group"]:
                pbar.update()
                continue

            sentence = " ".join(["_".join(x["text"].split()) for x in json_dict[sent_id]])
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

                doc = nlp(sentence)

                flag_multiword_target = False

                # add substitutes
                for subst, values in sub_to_selected_senses.items():

                    if len(values) == 0:
                        continue

                    if sentence_id <= 999:
                        builder.open_sentence_section(sentence_id=f"s{str(sentence_id).zfill(3)}")
                        sentence_id += 1

                    else:
                        text_id += 1
                        sentence_id = 0
                        builder.open_text_section(f"d{str(text_id).zfill(3)}")
                        builder.open_sentence_section(sentence_id=f"s{str(sentence_id).zfill(3)}")

                    for word_idx, word in enumerate(sentence.split()):

                        if word_idx not in target_index:
                            token_lemma = json_dict[sent_id][word_idx]['lemma']
                            token_pos = json_dict[sent_id][word_idx]['pos']

                            # unannotated token
                            builder.add_annotated_token(word.replace("_", " "), token_lemma, token_pos)

                        else:
                            if len(target_index) == 1 or not flag_multiword_target:

                                sense = list(values.keys())

                                complete_token_id = f"d{str(text_id).zfill(3)}.s{str(sentence_id).zfill(3)}.t{str(token_id).zfill(3)}"

                                if list(values.values())[0] == "monosemous" or list(values.values())[0] != 1000:
                                    if "_" in subst:
                                        builder.add_annotated_token(subst.replace("_", " "), subst, pos,
                                                                    complete_token_id, sense)

                                    else:
                                        original_pos = doc[word_idx].tag_
                                        try:
                                            inflected_substitute = lemminflect.getInflection(subst, original_pos)[0]
                                            builder.add_annotated_token(inflected_substitute, subst, pos,
                                                                        complete_token_id, sense)
                                        except IndexError:
                                            builder.add_annotated_token(subst, subst, pos,
                                                                        complete_token_id, sense)

                                    if token_id < 999:
                                        token_id += 1

                                    else:
                                        sentence_id += 1
                                        token_id = 0

                                    flag_multiword_target = True

                                else:
                                    token_lemma = json_dict[sent_id][word_idx]['lemma']
                                    token_pos = json_dict[sent_id][word_idx]['pos']

                                    # unnnotated token
                                    builder.add_annotated_token(word.replace("_", " "), token_lemma, token_pos)

                    for sensekey in values:
                        out.write(f"{subst}\t{sensekey}\t{values[sensekey]}\t{synset_from_sensekey(sensekey).definition()}\n")

                        if values[sensekey] == "monosemous":
                            monosemous_additional_instances += 1
                            mono.write(
                                f"{subst}\t{sensekey}\t{values[sensekey]}\t{synset_from_sensekey(sensekey).definition()}\n")

                        elif values[sensekey] != 1000:
                            non_null_additional_instances += 1
                            non_null.write(
                                f"{subst}\t{sensekey}\t{values[sensekey]}\t{synset_from_sensekey(sensekey).definition()}\n")

                        possible_additional += 1

                out.write("=========================\n")
            pbar.update()

    print(f"Monosemous additional: {monosemous_additional_instances}")
    print(f"Non-null (+ monosemous) additional: {non_null_additional_instances}")
    print(f"Maximum possible additional: {possible_additional}")

    builder.store(data_output_path=args.output_xml, labels_output_path=args.labels_output)