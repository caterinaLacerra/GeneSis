import os
from typing import List, Set, Dict, Tuple

import nltk
import numpy as np
import tqdm
from nltk.corpus import wordnet as wn

from src.utils import read_from_input_file, universal_to_wn_pos
from src.wordnet_utils import convert_to_wn_pos


def get_all_related_lemmas(synset: nltk.corpus.reader.wordnet) -> Set[str]:
    related = set()
    related.update(set([l.name() for l in synset.lemmas()]))

    for hyper in synset.hypernyms():
        related.update(set([l.name() for l in hyper.lemmas()]))

    for inst_hyper in synset.instance_hypernyms():
        related.update(set([l.name() for l in inst_hyper.lemmas()]))

    for hypo in synset.hyponyms():
        related.update(set([l.name() for l in hypo.lemmas()]))

    for inst_hypo in synset.instance_hyponyms():
        related.update(set([l.name() for l in inst_hypo.lemmas()]))

    for hol in synset.member_holonyms():
        related.update(set([l.name() for l in hol.lemmas()]))

    for hol in synset.substance_holonyms():
        related.update(set([l.name() for l in hol.lemmas()]))

    for hol in synset.part_holonyms():
        related.update(set([l.name() for l in hol.lemmas()]))

    for mer in synset.member_meronyms():
        related.update(set([l.name() for l in mer.lemmas()]))

    for mer in synset.substance_meronyms():
        related.update(set([l.name() for l in mer.lemmas()]))

    for mer in synset.part_meronyms():
        related.update(set([l.name() for l in mer.lemmas()]))

    for attribute in synset.attributes():
        related.update(set([l.name() for l in attribute.lemmas()]))

    for entailment in synset.entailments():
        related.update(set([l.name() for l in entailment.lemmas()]))

    for cause in synset.causes():
        related.update(set([l.name() for l in cause.lemmas()]))

    for also_see in synset.also_sees():
        related.update(set([l.name() for l in also_see.lemmas()]))

    for verb_group in synset.verb_groups():
        related.update(set([l.name() for l in verb_group.lemmas()]))

    for similar in synset.similar_tos():
        related.update(set([l.name() for l in similar.lemmas()]))

    return related


def get_related_lemmas(lexeme: str) -> Set[str]:
    *lemma, pos = lexeme.split('.')
    lemma = '.'.join(lemma)
    related = set()
    synsets = wn.synsets(lemma, universal_to_wn_pos(pos))

    for synset in synsets:
        # include all neighbours (distance 1)
        related.update(get_all_related_lemmas(synset))

        for also_see in synset.also_sees():
            related.update(get_all_related_lemmas(also_see))

        for similar in synset.similar_tos():
            related.update(get_all_related_lemmas(similar))

        for hypo in synset.hyponyms():
            related.update(get_all_related_lemmas(hypo))

        for hyper in synset.hypernyms():
            related.update(get_all_related_lemmas(hyper))

    return related


def load_lemma_freq(input_path: str) -> Dict[str, float]:
    tot = 0
    word_to_freq = {}
    for line in open(input_path):
        word, count = line.strip().split('\t')
        count = float(count)
        word_to_freq[word] = count
        tot += count

    for k, v in word_to_freq.items():
        word_to_freq[k] = v/tot

    return word_to_freq


def load_pairs_freq(input_path: str) -> Dict[str, float]:
    pair_to_freq = {}
    tot = 0
    for line in open(input_path):
        word_1, word_2, count = line.strip().split('\t')
        tuple_key = '###'.join([word_1, word_2])
        count = float(count)
        pair_to_freq[tuple_key] = count
        tot += count

    for k, v in pair_to_freq.items():
        pair_to_freq[k] = v / tot

    return pair_to_freq

def get_pmi(lexeme_a: str, lexeme_b: str, lexemes_freq: Dict[str, float], pairs_freq: Dict[str, float]) -> float:
    if lexeme_a in lexemes_freq and lexeme_b in lexemes_freq:
        pair_key = '###'.join(sorted([lexeme_a, lexeme_b]))
        if pair_key in pairs_freq:
            pmi = np.log(pairs_freq[pair_key] / (lexemes_freq[lexeme_a]*lexemes_freq[lexeme_b]))
            return pmi
    return -100


def write_to_folders(lexemes_list: List[str], root_output_folder: str, lemma_path: str, pairs_path: str, threshold: int) -> None:
    if not os.path.exists(root_output_folder):
        os.makedirs(root_output_folder)

    print(f'Loading word dictionary ...')
    word_to_freq = load_lemma_freq(lemma_path)
    pair_to_freq = load_pairs_freq(pairs_path)
    print('...Done!')
    pos_words = {}
    pos_list = ['VERB', 'NOUN', 'ADJ', 'ADV']

    for pos in pos_list:
        pos_words[pos] = set()
        for synset in list(wn.all_synsets(convert_to_wn_pos(pos))):
            names = set([l.name() for l in synset.lemmas()])
            pos_words[pos].update(names)

    all_words = set([x for p in pos_list for x in pos_words[p]])

    for lexeme in tqdm.tqdm(lexemes_list):
        *lemma, pos = lexeme.split('.')

        if lexeme in word_to_freq:

            substitutes = [word for word, score in get_more_freq_pairs(lexeme, pos_words[pos],
                                                                       word_to_freq, pair_to_freq, pos) if score > -100]

        else:
            substitutes = [word for word, score in get_more_freq_pairs(lexeme, all_words, word_to_freq,
                                                                       pair_to_freq, pos) if score > -100]


        substitutes = substitutes[:threshold]

        with open(os.path.join(root_output_folder, lexeme), 'w') as out:
            for substitute in substitutes:
                if substitute != '_'.join(lemma):
                    out.write(substitute + '\n')


def get_more_freq_pairs(lexeme: str, candidates: Set[str], word_freq: Dict[str, float], pair_freq: Dict[str, float], pos: str) -> List[Tuple[str, float]]:
    ranked_candidates = []
    for candidate in candidates:
        candidate_lexeme = f'{candidate}.{pos}'
        pmi = get_pmi(candidate_lexeme, lexeme, word_freq, pair_freq)
        ranked_candidates.append((candidate, pmi))

    substitutes = sorted(ranked_candidates, key=lambda x: x[1], reverse=True)
    return substitutes


if __name__ == '__main__':
    test_list = [x.target for x in read_from_input_file('data/lst/lst_test.tsv')]
    dev_list = [x.target for x in read_from_input_file('data/coinco_twsi/coinco_twsi_dev.tsv')]
    lexemes_list = list(set(test_list).union(set(dev_list)))

    threshold = 50

    lemma_path = '/home/caterina/PycharmProjects/commonsense/pmi/lemma_counter.txt'
    pairs_path = '/home/caterina/PycharmProjects/commonsense/pmi/pairs_counter.txt'

    output_path = f'vocab/all_wordnet_vocab_{threshold}'

    write_to_folders(lexemes_list, output_path, lemma_path, pairs_path, threshold)
