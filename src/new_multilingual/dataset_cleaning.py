import argparse
import string

import tqdm
import wordfreq

from src.utils import read_from_input_file, file_len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--lang", required=True)
    return parser.parse_args()


def clean_targets(input_path: str, output_path: str,  language_code: str):

    targets = set([instance.target for instance in read_from_input_file(input_path)])

    reduced_targets = set()
    for target in targets:
        *lemma, pos = target.split(".")
        lemma = ".".join(lemma)
        if wordfreq.zipf_frequency(lemma, language_code, wordlist='best') > 0:
            # exclude non-existing targets
            reduced_targets.add(target)

    with open(output_path, 'w') as out:
        for instance in tqdm.tqdm(read_from_input_file(input_path), total=file_len(input_path)):
            if instance.target in reduced_targets:
                words = set([word for word in instance.sentence.split() if word not in string.punctuation])

                # exclude sentences with non-existing words
                if all(wordfreq.zipf_frequency(word, language_code, wordlist="best") > 0 for word in words):
                    out.write(f"{repr(instance)}\n")

    print(f"Initial instances: {file_len(input_path)}\n"
          f"Cleaned targets instances: {file_len(output_path)}")

if __name__ == '__main__':

    args = parse_args()
    clean_targets(args.input_path, args.output_path, language_code=args.lang)
