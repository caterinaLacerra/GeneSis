import json
import string
from typing import List, Set, Dict
from wordfreq import zipf_frequency

from src.utils import yield_batch, get_target_index_list


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

def substitutes_to_json_file(substitutes_path: str):

    instances = set()

    for batch in yield_batch(substitutes_path, separator='#########\n'):
        target_lexeme, instance_id, *target_index = batch[0].strip().split()
        target_index = ' '.join(target_index)
        target_index = get_target_index_list(target_index)
        original_context = batch[1].strip()
        substitutes = batch[-1].split('# set candidates: ')[1].strip().split(';')
        instance = AugmentedWSDInstance(target_lexeme, target_index, instance_id, original_context, substitutes)
        instances.add(instance)

    json_file_path = substitutes_path.replace('.txt','.json')
    with open(json_file_path, 'w', encoding='utf-8') as out:
        for instance in instances:
            out.write(json.dumps(instance.__dict__, indent=4) + '\n')

class AugmentedWSDInstance:

    def __init__(self, target_lexeme: str, target_index: List[int],
                 instance_id: str, context: str, substitutes: Set[str]):

        self.target = target_lexeme
        self.instance_id = instance_id
        self.context = context
        self.target_index = target_index
        self.substitutes = substitutes

        words = context.split()
        init_words = words[:target_index[0]]
        end_words = words[target_index[-1] + 1:]

        substitutes_contexts, final_target_indexes = [], []

        for substitute in substitutes:
            # todo: inflect the substitutes
            substitutes_contexts.append(" ".join(init_words + [substitute] + end_words))
            if len(substitute.split()) == 1:
                final_target_indexes.append([target_index[0]])

            else:
                final_target_indexes.append([target_index[0], target_index[0] + len(substitute.split()) - 1])

        self.augmented_contexts = substitutes_contexts
        self.augmented_target_indexes = final_target_indexes

