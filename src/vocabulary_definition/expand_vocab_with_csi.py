import os
from typing import List, Dict

import tqdm
from nltk.corpus import wordnet as wn

from src.utils import read_from_input_file, universal_to_wn_pos
from src.vocabulary_definition.create_vocab_from_wn import get_related_lemmas
from src.wordnet_utils import get_csi_keys, get_synset_from_str_offset


def load_csi(csi_path: str):
    synset_to_labels = {}
    for line in open(csi_path):
        line = line.strip().split('\t')
        synset, *labels = line
        synset_to_labels[synset] = [x for x in labels]
    return synset_to_labels


def write_to_folders(lexemes_list: List[str], root_output_folder: str, csi: Dict[str, List[str]]) -> None:

    reverse_csi = {}
    for key in csi:
        for label in csi[key]:
            if label not in reverse_csi:
                reverse_csi[label] = set()
            reverse_csi[label].add(key)

    if not os.path.exists(root_output_folder):
        os.makedirs(root_output_folder)

    missing, total = 0, 0
    for lexeme in tqdm.tqdm(lexemes_list):
        *lemma, pos = lexeme.split('.')
        if pos == 'ADV':
            lemma = '.'.join(lemma)
            synsets = wn.synsets(lemma, universal_to_wn_pos(pos))
            if not any(get_csi_keys(synset, pos) in csi for synset in synsets):
                missing += 1

            keys = [get_csi_keys(synset, pos) for synset in synsets]
            labels = set([l for k in keys if k in csi for l in csi[k]])
            related_synsets = set()

            for l in labels:
                related_synsets.update([offset for offset in reverse_csi[l] if offset.endswith('r')])

            substitutes = set()
            for related in related_synsets:
                s = get_synset_from_str_offset(related)
                substitutes.update(set([lemma.name() for lemma in s.lemmas()]))

            with open(os.path.join(root_output_folder, lexeme), 'w') as out:
                for substitute in substitutes:
                    out.write(substitute + '\n')
            total += 1

        else:
            substitutes = get_related_lemmas(lexeme)
            with open(os.path.join(root_output_folder, lexeme), 'w') as out:
                for substitute in substitutes:
                    out.write(substitute + '\n')
    print(f'Missing: {missing} over {total}')


if __name__ == '__main__':
    csi_path = 'data/csi_data/wn_synset2csi.txt'
    test_list = [x.target for x in read_from_input_file('data/lst/lst_test.tsv')]
    lexemes_list = list(set(test_list))
    csi = load_csi(csi_path)
    write_to_folders(lexemes_list, 'vocab/wordnet_vocab_expanded_csi', csi)
