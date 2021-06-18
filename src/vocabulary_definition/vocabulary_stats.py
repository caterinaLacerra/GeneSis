import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', required=True)
    parser.add_argument('--test_path', required=True)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    lexemes = set([line.strip().split('\t')[0] for line in open(args.test_path).readlines()])

    pos_vocab = {}
    pos_total_words = {}

    for file_name in os.listdir(args.vocab_path):
        if file_name not in lexemes:
            continue
        file = os.path.join(args.vocab_path, file_name)
        *_, pos = file_name.split('.')
        if pos not in pos_vocab:
            pos_vocab[pos] = 0
            pos_total_words[pos] = 0

        pos_total_words[pos] += 1
        substitutes = set([x for x in open(file).readlines()])
        pos_vocab[pos] += len(substitutes)

    for pos in pos_vocab:
        print(f'{pos} avg substitutes: {pos_vocab[pos]/pos_total_words[pos]}')