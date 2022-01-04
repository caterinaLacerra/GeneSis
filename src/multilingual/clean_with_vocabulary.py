import argparse
from typing import Set

import tqdm

from src.utils import read_from_input_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def filter_substitutes(input_path: str, output_path: str, vocab_words: Set[str]):
    with open(output_path, 'w') as out:
        for instance in tqdm.tqdm(read_from_input_file(input_path)):
            instance.gold = {k: v for k, v in instance.gold.items() if k.lower() in vocab_words}
            if len(instance.gold) > 0:
                out.write(instance.__repr__() + '\n')


if __name__ == '__main__':
    args = parse_args()
    vocabulary = set([x.strip() for x in open(args.vocab_path)])
    filter_substitutes(args.input_path, args.output_path, vocabulary)