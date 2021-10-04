import argparse
import os

from src.wsd.utils.utils import read_from_input_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', required=True)
    parser.add_argument('--test_path', required=True)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    lexemes = set([line.strip().split('\t')[0] for line in open(args.test_path).readlines()])

    substitutes = set()


    for file_name in os.listdir(args.vocab_path):
        if file_name not in lexemes:
            continue
        file = os.path.join(args.vocab_path, file_name)
        substitutes.update(set([x.strip() for x in open(file).readlines()]))

    print(f"Substitutes in vocab: {len(substitutes)}")

    test_words = set()
    for instance in read_from_input_file(args.test_path):
        test_words.update(set(instance.gold.keys()))

    print(f"Substitutes in test: {len(test_words)}")

    print(f"Intersection: {len(test_words.intersection(substitutes))}")

    out_from_test = [x for x in test_words if x not in substitutes]

    print(f"Missing in vocab from test: {len(out_from_test)}")
    print(f"Missing in test from vocab: {len([x for x in substitutes if x not in test_words])}")

    uncovered_targets = set()
    compl_uncovered = set()

    for instance in read_from_input_file(args.test_path):
        for s in instance.gold:
            if s in out_from_test:
                uncovered_targets.add(instance.target)
        if all(s in out_from_test for s in instance.gold):
            compl_uncovered.add(instance.target)

    print(f"Targets with at least one substitute not available: {len(uncovered_targets)}")

    pos_distr = {}
    for lexeme in uncovered_targets:
        *lemma, pos = lexeme.split('.')
        if pos not in pos_distr:
            pos_distr[pos] = []

        pos_distr[pos].append(".".join(lemma))

    for pos in pos_distr:
        print(pos, len(pos_distr[pos]))

    print(f"Targets with all substitutes not available: {len(compl_uncovered)}")
    pos_distr = {}
    for lexeme in compl_uncovered:
        *lemma, pos = lexeme.split('.')
        if pos not in pos_distr:
            pos_distr[pos] = []

        pos_distr[pos].append(".".join(lemma))

    for pos in pos_distr:
        print(pos, len(pos_distr[pos]), pos_distr[pos])
