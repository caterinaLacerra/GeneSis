import argparse
from typing import Dict, Set
import urllib.request

from src.utils import read_from_input_file

def parse_input_file(input_path: str) -> Dict[str, Set[str]]:

    pos_to_targets = {}
    for instance in read_from_input_file(input_path):
        target = instance.target
        *lemma, pos = target.split('.')
        if pos not in pos_to_targets:
            pos_to_targets[pos] = set()
        pos_to_targets[pos].add(".".join(lemma))

    return pos_to_targets

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--lang", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pos_to_lemma = parse_input_file(args.input_path)


    with open(args.output_path, 'w') as out:
        for pos, lemmas in pos_to_lemma.items():
            for lemma in lemmas:
                out.write(f"{lemma}\t{pos}\n")