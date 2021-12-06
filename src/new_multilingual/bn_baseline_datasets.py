import argparse

from src.utils_wsd import read_from_raganato


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_xml', required=True)
    parser.add_argument('--gold_path', required=True)
    parser.add_argument('--output_path', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    for _, _, sentence in read_from_raganato(args.input_xml, args.gold_path):
        for token in sentence:
            if token.labels is not None:
                print(token)