import argparse
import string
from typing import Optional, List

import nltk
import numpy
import stanza
import tqdm
import wordfreq

from src.wsd.utils.utils import multipos_to_pos, LexSubInstance, get_target_index_list, file_len, convert_to_universal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--lang', required=True, help="BCP 47 or ISO 639 code of the language to use, such as 'en'."
                                                      " Check supported languages on https://pypi.org/project/wordfreq/")
    return parser.parse_args()


def clean_substitute(substitute: str, lang: str, stopwords: List[str]) -> Optional[str]:

    if substitute.endswith(':'):
        substitute = substitute.replace(':', '')

    words = substitute.split('_')

    # compute combined freq of non-stopwords
    freqs = [wordfreq.zipf_frequency(w, lang, wordlist='best', minimum=0.0) for w in words if w not in stopwords]
    combined_freq = numpy.prod(freqs)
    # not existing multiwords
    if combined_freq == 0:
        return None

    if all(x in stopwords for x in words):
        return None

    return substitute


def valid_word(word: str) -> bool:
    punct = [x for x in string.punctuation if x!="."]
    for idx, char in enumerate(word):
        if char in punct:
            if char == "'" and idx != 0:
                continue
            else:
                return False
    return True


def clean_dataset(input: str, output: str, lang: str, pipeline: stanza.Pipeline):
    full_lang_mapping = {'it': 'italian', 'en': 'english', 'de': 'german'}
    stopwords = nltk.corpus.stopwords.words(full_lang_mapping[lang])
    punctuation = set([x for x in string.punctuation])

    with open(output, 'w') as out:
        for l, line in tqdm.tqdm(enumerate(open(input)), total=file_len(input)):
            substitutes = line.strip().split('\t')[-1].split()

            gold = {}
            for sub in substitutes:
                if ':::' in sub:
                    continue
                word, score = sub.split('::')

                if word not in punctuation and word not in stopwords:
                    if "'" in word:
                        word = "".join(word.split("'")[1:])
                    gold[word] = float(score)

            if len(gold) == 0:
                continue

            try:
                target, instance_id, target_idx, sentence, mask, _ = line.strip().split('\t')
            except:
                print(line)
                continue
            target_idx = get_target_index_list(target_idx)
            instance = LexSubInstance(target, instance_id, target_idx, sentence, gold=gold)

            instance.target_idx = sorted(instance.target_idx)
            instance.target = instance.target.lower()
            words = instance.sentence.split()
            sentences = [instance.sentence]
            sorted_substitutes = [instance.target.split('_')]

            for substitute in instance.gold:
                new_sentence = words[:instance.target_idx[0]] + substitute.split('_') + words[instance.target_idx[-1] + 1:]
                sentences.append(" ".join(new_sentence))
                sorted_substitutes.append(substitute.split('_'))

            doc = pipeline("\n\n".join(sentences))
            start_idx = instance.target_idx[0]

            cleaned_substitutes = {}

            target_pos_tag = convert_to_universal(instance.target.split('.')[-1].upper())


            for i, sent in enumerate(doc.sentences):
                end_idx = start_idx + len(sorted_substitutes[i])
                relevant_words = [word for j, word in enumerate(sent.words)
                                  if j >= start_idx and j < end_idx]
                if i == 0:
                    target_lemma = "_".join([w.lemma for w in relevant_words if w.lemma]).lower()
                    target_word = "_".join([w.text for w in relevant_words]).lower()

                else:
                    pos_list = [w.upos for w in relevant_words if w.upos]
                    if pos_list == []:
                        continue

                    postag = multipos_to_pos(pos_list)
                    if postag not in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                        continue

                    # remove substitutes with different POS than target
                    if postag == target_pos_tag:
                        lemmatized_substitute = "_".join([w.lemma for w in relevant_words]).lower()

                        # remove single words that already are in the sentence (noise)
                        if len(lemmatized_substitute.split('_')) == 1 and lemmatized_substitute in words:
                            continue

                        # remove target from substitutes
                        if lemmatized_substitute != target_lemma and lemmatized_substitute != target_word:

                            if lemmatized_substitute not in cleaned_substitutes:
                                cleaned_substitutes[lemmatized_substitute] = []
                            cleaned_substitutes[lemmatized_substitute].append(instance.gold["_".join(sorted_substitutes[i])])


            if len(cleaned_substitutes) > 0:
                instance.gold = {k: numpy.mean(v) for k, v in cleaned_substitutes.items()}
                instance.mask = ['---']
                out.write(str(instance) + '\n')



if __name__ == '__main__':
    args = parse_args()

    nlp = stanza.Pipeline(lang=args.lang, processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True,
                          tokenize_no_ssplit=True)

    clean_dataset(args.input_path, args.output_path, args.lang, nlp)
