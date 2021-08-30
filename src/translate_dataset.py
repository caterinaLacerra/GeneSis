import argparse
import json
from typing import Dict

from src.utils import get_target_index_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    return parser.parse_args()


def format_sentence(input_line: str) -> Dict:
    lexeme, instance_id, instance_index, sentence, mask, substitutes = input_line.strip().split('\t')
    output_dict = {}
    *lemma, pos = lexeme.split('.')
    lemma = '.'.join(lemma)
    output_dict['lemma'] = lemma
    output_dict['pos'] = pos
    output_dict['lexeme'] = lexeme
    output_dict['idx'] = get_target_index_list(instance_index)
    output_dict['instance_id'] = instance_id
    output_dict['sentence'] = sentence
    if len(output_dict['instance_id']) == 1:
        target = sentence.split()[output_dict['idx'][0]]
    else:
        target = sentence.split()[output_dict['idx'][0]:output_dict['idx'][-1] + 1]
    output_dict['target'] = ' '.join(target)
    dict_substitutes = {x.split('::')[0].replace('_', ' '): float(x.split('::')[1]) for x in substitutes.split()}
    output_dict['substitutes'] = dict_substitutes

    return output_dict

if __name__ == '__main__':

    args = parse_args()
    assert args.output_path.endswith('.txt'), print('output path should end with .txt')

    id_num = 0
    with open(args.output_path, 'w') as out, open(args.output_path.replace('.txt', '.id.txt'), 'w') as id_out:
        for line in open(args.input_path):

            input_dict = format_sentence(line)
            # format to single line
            target_sentence = f'{input_dict["sentence"]}'

            # write to output file
            out.write(f'{target_sentence}\n')

            # write to output id file
            js_dict = json.dumps(input_dict)
            id_out.write(f'id:{id_num}\t{js_dict}\n')

            id_num += 1

            for sub in input_dict["substitutes"]:

                # build sentence with substitute
                sentence = input_dict["sentence"]
                replaced_sentence = " ".join(sentence.split()[:input_dict["idx"][0]] + [sub] + sentence.split()[input_dict["idx"][-1] + 1:])
                subst_sentence = f'{replaced_sentence}'

                # write to output files
                out.write(f'{subst_sentence}\n')
                id_out.write(f'id: {id_num}\t{js_dict}\n')

                id_num += 1