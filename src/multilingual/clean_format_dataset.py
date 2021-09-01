import argparse
import string
from typing import Optional, List

import nltk
import numpy
import wordfreq

from src.utils import read_from_input_file


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


def clean_dataset(input: str, output: str, lang: str):
    # stopwords = nltk.corpus.stopwords.words('italian')

    with open(output, 'w') as out:
        try:
            for instance in read_from_input_file(input):
                instance.target_idx = sorted(instance.target_idx)
                instance.target = instance.target.lower()
                if len(instance.target_idx) > 1:
                    continue
                else:
                    # if instance.target_idx[0] == instance.target_idx[-1]:
                    #     instance.target_idx = [instance.target_idx[0]]
                    # wrong alignments
                    # if len([x for x in range(instance.target_idx[0], instance.target_idx[-1] + 1)]) != len(instance.target.split('_')):
                    #     continue
                    # else:
                    #     *word, pos = instance.target.split('.')
                    #     words = '.'.join(word).split('_')
                    #     compute combined freq of non-stopwords
                        # freqs = [wordfreq.zipf_frequency(w, lang, wordlist='best', minimum=0.0) for w in words if w not in stopwords]
                        # combined_freq = numpy.prod(freqs)
                        # not existing multiwords
                        # if combined_freq == 0:
                        #     continue
                        #
                        # else:
                        #     if not valid_word(" ".join(words)):
                        #         continue
                        #
                        #     else:
                        #         if any(clean_substitute(s, lang, stopwords) for s in instance.gold):
                        *target, pos = instance.target.split('.')
                        target = '.'.join(target)
                        reduced_substitutes = [s for s in instance.gold if s not in target]
                        if len(reduced_substitutes) > 0:
                            instance.gold = {k.lower(): v for k, v in instance.gold.items() if k in reduced_substitutes}
                        out.write(str(instance) + '\n')
        except:
            pass


if __name__ == '__main__':
    args = parse_args()
    clean_dataset(args.input_path, args.output_path, args.lang)