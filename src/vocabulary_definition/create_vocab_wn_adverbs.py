import os
from typing import List, Dict

import tqdm
from nltk.corpus import wordnet as wn

from src.utils import read_from_input_file, universal_to_wn_pos
from src.vocabulary_definition.create_vocab_from_wn import get_related_lemmas
from src.wordnet_utils import get_csi_keys, get_synset_from_str_offset, convert_to_wn_pos


def load_csi(csi_path: str):
    synset_to_labels = {}
    for line in open(csi_path):
        line = line.strip().split('\t')
        synset, *labels = line
        synset_to_labels[synset] = [x for x in labels]
    return synset_to_labels


def write_to_folders(lexemes_list: List[str], root_output_folder: str) -> None:

    if not os.path.exists(root_output_folder):
        os.makedirs(root_output_folder)

    all_adverbs = set()
    for synset in wn.all_synsets(convert_to_wn_pos('ADV')):
        lemmas = set([l.name() for l in synset.lemmas()])
        all_adverbs.update(lemmas)

    for lexeme in tqdm.tqdm(lexemes_list):
        *lemma, pos = lexeme.split('.')
        if pos == 'ADV':
            lemma = '.'.join(lemma)
            substitutes = [x for x in all_adverbs if x!=lemma]
            with open(os.path.join(root_output_folder, lexeme), 'w') as out:
                for substitute in substitutes:
                    out.write(substitute + '\n')

        else:
            substitutes = get_related_lemmas(lexeme)
            with open(os.path.join(root_output_folder, lexeme), 'w') as out:
                for substitute in substitutes:
                    out.write(substitute + '\n')


if __name__ == '__main__':
    test_list = [x.target for x in read_from_input_file('data/lst/lst_test.tsv')]
    dev_list = [x.target for x in read_from_input_file('data/coinco_twsi/coinco_twsi_dev.tsv')]
    lexemes_list = list(set(test_list).union(dev_list))
    write_to_folders(lexemes_list, 'vocab/wordnet_vocab_expanded_adv')
