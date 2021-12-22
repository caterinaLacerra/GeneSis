import argparse
import collections

from src.utils import read_from_input_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    return parser.parse_args()


def get_input_distribution(input_path: str, output_path: str):

    substitutes = []
    for instance in read_from_input_file(input_path):
        substitutes.extend(list(instance.gold.keys()))

    c = collections.Counter(substitutes).most_common(100)
    with open(output_path, 'w') as out:
        for word, counts in c:
            out.write(f"{word} {counts} {counts/len(substitutes)}\n")


if __name__ == '__main__':

    args = parse_args()
    get_input_distribution(args.input_path, args.output_path)