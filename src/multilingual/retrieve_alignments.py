import argparse
import json
import numpy as np
import os
from typing import List, Dict, Any, Union, Optional

import stanza as stanza
import tqdm

from src.wsd.utils.utils import multipos_to_pos, file_len
from src.wsd.utils.wordnet_utils import map_to_wn_pos


def get_target_index_from_tokenized(tokenized_sentence: str, word: str, original_idx: List[int]) -> Optional[
    List[Union[int, List[int]]]]:

    tokens = tokenized_sentence.split()

    # case a: single word
    if len(original_idx) == 1:
        idx_occurrences = [i for i, x in enumerate(tokens) if x == word]
        distances = [abs(i - original_idx[0]) for i in idx_occurrences]
        if distances == []:
            return None
        return [idx_occurrences[np.argmin(distances)]]

    # case b: multi-word
    else:
        range_original_idx = [n for n in range(original_idx[0], original_idx[-1] + 1)]
        idx_occurrences = [i for i in range(len(tokens))
                           if i + len(range_original_idx) < len(tokens) and
                           " ".join(tokens[i: i + len(range_original_idx)]) == word]

        distances = [abs(i - original_idx[0]) for i in idx_occurrences]

        if distances == []:
            return None

        return [idx_occurrences[np.argmin(distances)], idx_occurrences[np.argmin(distances)] + len(range_original_idx) - 1]


def process_target_sentence(en_sentence: str, original_target_word: str, en_target_idx: List[int],
                            alignment_mapping: Dict[int, Dict],
                            translation_dict: Dict[str, Any], key: str):

    # retrieve target index in en_sentence
    updated_idx = get_target_index_from_tokenized(en_sentence, original_target_word, en_target_idx)

    if updated_idx:
        updated_range = [x for x in range(updated_idx[0], updated_idx[-1] + 1)]
        if updated_idx != en_target_idx:
            assert " ".join([en_sentence.split()[x] for x in updated_range]) == original_target_word

        # get translated target
        translation_indexes, alignment_scores = [], []

        for x in updated_range:
            if x in alignment_mapping:
                translation_indexes.extend(list(alignment_mapping[x]))
                alignment_scores.extend(list(alignment_mapping[x].values()))

        # if the target is aligned
        if translation_indexes != []:
            target_score = np.mean(alignment_scores)

            # get corresponding translations (target, sentence)
            translated_target = " ".join([it_sentence.split()[x] for x in translation_indexes])

            translation_dict[key].update({"target": translated_target,
                                          "idx": translation_indexes,
                                          "target_score": target_score})



def process_substitute_sentence(substitute: str, en_sentence: str, it_sentence: str, en_target_idx: List[int],
                                alignment_mapping: Dict[int, Dict],
                                translation_dict: Dict[str, Any], key: str):


    if len(substitute.split()) > 1:
        substitute_idx = [en_target_idx[0], en_target_idx[0] + len(substitute.split())]

    else:
        substitute_idx = [en_target_idx[0]]

    # retrieve substitute index in en_sentence
    updated_idx = get_target_index_from_tokenized(en_sentence, substitute, substitute_idx)

    if updated_idx:
        updated_range = [x for x in range(updated_idx[0], updated_idx[-1] + 1)]
        if updated_idx != substitute_idx:
            assert "".join([en_sentence.split()[x] for x in updated_range]) == substitute

        # get translated substitute idx
        translation_indexes = []
        alignment_scores = []

        for x in updated_range:
            if x in alignment_mapping:
                translation_indexes.extend(list(alignment_mapping[x]))
                alignment_scores.extend(list(alignment_mapping[x].values()))

        # get translated substitute
        translated_subst = " ".join([it_sentence.split()[x] for x in translation_indexes])

        if translated_subst != "":
            # compute substitute score
            original_score = info_dict['substitutes'][substitute]
            score = original_score * np.mean(alignment_scores)

            if translated_subst not in translation_dict[key]["substitutes"]:
                translation_dict[key]["substitutes"][translated_subst] = score


def retrieve_sentence_alignment(id_idx: str, info_str: str, tokenized_str: str, alignment_idx_str: str,
                                alignment_probs: str, translation_dict: Dict[str, Dict]):

    alignment_mapping = {}

    for pair, scores in zip(alignment_idx_str.split(), alignment_probs.split()):
        src, tgt = pair.split('-')
        score = float(scores)
        if int(src) not in alignment_mapping:
            alignment_mapping[int(src)] = {}
        alignment_mapping[int(src)][int(tgt)] = score


    idx, info = info_str.strip().split('\t')
    info_dict = json.loads(info)

    en_sentence, it_sentence = tokenized_str.strip().split(' ||| ')
    key = id_idx

    if "replaced_sentence" not in info_dict:

        translation_dict[key] = {"substitutes": {},
                                 "instance_id": info_dict["instance_id"],
                                 "sentence": it_sentence,
                                 "en_sentence": en_sentence}

    original_target_word = info_dict['target']
    en_target_idx = info_dict['idx']

    # case 1: translating the original sentence
    if 'replaced_sentence' not in info_dict:
        process_target_sentence(en_sentence, original_target_word, en_target_idx, alignment_mapping,
                                translation_dict, key)


    # case 2: translating a substitute
    else:
        for i, substitute in enumerate(info_dict['substitutes']):
            if substitute in en_sentence.split():
                process_substitute_sentence(substitute, en_sentence, it_sentence, en_target_idx, alignment_mapping,
                                            translation_dict, key)
                break


def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser()
    parser.add_argument('--translation_folder', required=False, default='data/translation')
    parser.add_argument('--laser_folder', required=False, default='data/translation/laser_embeddings')
    parser.add_argument('--language', required=True)
    parser.add_argument('--dataset', required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    english_info_path = os.path.join(args.translation_folder, f'{args.dataset}.id.txt')
    tokenized_parallel = os.path.join(args.translation_folder, f'{args.dataset}.{args.language}.tokenized.txt')
    alignment_indexes = os.path.join(args.translation_folder, f'{args.dataset}.{args.language}.aligned.txt')
    alignment_probs = os.path.join(args.translation_folder, f'{args.dataset}.{args.language}.align_prob.txt')

    embeddings_folder = args.laser_folder
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    translation_dict = {}

    for idx, (info, tokenize, indexes, probs) in tqdm.tqdm(enumerate(zip(open(english_info_path),
                                                                         open(tokenized_parallel),
                                                                         open(alignment_indexes),
                                                                         open(alignment_probs))),
                                                           total=file_len(tokenized_parallel)):

        id_idx, infol = info.strip().split('\t')
        info_dict = json.loads(infol)
        en_sentence, it_sentence = tokenize.strip().split(' ||| ')

        retrieve_sentence_alignment(id_idx, info, tokenize, indexes, probs, translation_dict)

    nlp = stanza.Pipeline(lang=args.language, processors='tokenize,mwt,pos,lemma', tokenize_no_ssplit=True,
                          tokenize_pretokenized=True)

    print(f'Translated and aligned sentences: {len(translation_dict)}')

    counter_no_target = 0
    chunks_count = 0

    to_write_en, to_write_lang, to_write_tot = [], [], []

    for instance_idx, idx in tqdm.tqdm(enumerate(translation_dict)):

        sentence = translation_dict[idx]['sentence']
        # extract lemma and pos for the target word
        words = [w for w in nlp(sentence).sentences[0].words]
        try:
            index = translation_dict[idx]['idx']

        except KeyError:
            counter_no_target += 1
            continue

        if any(x.lemma is None for i, x in enumerate(words) if i in index):
            continue

        lemma = "_".join([w.lemma for i, w in enumerate(words) if i in index])
        pos_list = [word.upos for i, word in enumerate(words) if i in index]

        if len(pos_list) == 0:
            continue

        pos_tag = multipos_to_pos(pos_list)
        pos = map_to_wn_pos(pos_tag)

        if pos is None:
            continue

        lexeme = f'{lemma}.{pos}'

        clean_substitutes = {substitute: score
                              for substitute, score in translation_dict[idx]['substitutes'].items()
                              if substitute != lemma and substitute != translation_dict[idx]['target']}

        sorted_substitutes = sorted([(sub.replace(' ', '_'), score) for sub, score in clean_substitutes.items()],
                                    key=lambda x:x[1], reverse=True)

        if len(sorted_substitutes) > 0:
            subst_str = " ".join([f'{sub}::{score}' for sub, score in sorted_substitutes])

            to_write_tot.append(f'{lexeme}\t{translation_dict[idx]["instance_id"]}\t'
                                f'{translation_dict[idx]["idx"]}\t{sentence}\t---\t{subst_str}')

            to_write_lang.append(sentence)
            to_write_en.append(translation_dict[idx]["en_sentence"])

        if instance_idx % 100000 == 0 and instance_idx > 0:

            sentences_en = os.path.join(embeddings_folder, f'{args.dataset}.laser.en.{chunks_count}.txt')
            sentences_lang = os.path.join(embeddings_folder,
                                          f'{args.dataset}.laser.{args.language}.{chunks_count}.txt')

            output_file = os.path.join(embeddings_folder,
                                       f'{args.dataset}.formatted.{chunks_count}.txt')

            with open(sentences_en, 'w') as out_en, open(sentences_lang, 'w') as out_lang, open(output_file, 'w') as out:
                out_en.write("\n".join(to_write_en))
                out_lang.write("\n".join(to_write_lang))
                out.write("\n".join(to_write_tot))

            to_write_lang, to_write_tot, to_write_en = [], [], []
            chunks_count += 1

    if to_write_en != []:
        sentences_en = os.path.join(embeddings_folder, f'{args.dataset}.laser.en.{chunks_count}.txt')
        sentences_lang = os.path.join(embeddings_folder,
                                      f'{args.dataset}.laser.{args.language}.{chunks_count}.txt')

        output_file = os.path.join(embeddings_folder,
                                   f'{args.dataset}.formatted.{chunks_count}.txt')

        print(f'Saving output to {output_file}')

        with open(sentences_en, 'w') as out_en, open(sentences_lang, 'w') as out_lang, open(output_file, 'w') as out:
            out_en.write("\n".join(to_write_en))
            out_lang.write("\n".join(to_write_lang))
            out.write("\n".join(to_write_tot))


    print(f'No idx found: {counter_no_target}')