import json
import string
from typing import Dict
from wordfreq import zipf_frequency


def read_from_json_format(json_path: str):
    with open(json_path, 'r') as inp:
        for line in inp:
            instance = json.loads(line)
            yield instance

def ends_with_punct(word: str):
    punctuation = set([x for x in string.punctuation])
    for p in punctuation:
        if word.endswith(p):
            return True
    return False

def get_clean_generated_substitutes(generation: str, target_word: str) -> Dict[str, int]:
    beams = generation.split('\n')

    set_clean_words = {}

    for beam in beams:
        words = beam.split(', ')

        # remove single char words, target word and check if word ends with punctuation
        clean_words = set([w for w in words if len(w) > 1 and
                           w.lower() != target_word.lower() and
                           not ends_with_punct(w) and
                           w.lower().replace(' ', '_') != target_word.lower() and
                           target_word not in w])


        for w in clean_words:

            # check zipf frequency of the word
            if zipf_frequency(w, 'en') > 0:
                if w not in set_clean_words:

                    set_clean_words[w] = 0

                set_clean_words[w] += 1


    return set_clean_words

