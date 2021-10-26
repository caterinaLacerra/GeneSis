import argparse

import numpy as np
import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import save_reduced_numberbatch, load_numberbatch, read_from_input_file, find_numberbatch_keys, file_len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--numberbatch", required=True)
    parser.add_argument("--language")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    numb_vectors = load_numberbatch(args.numberbatch)

    with open(args.output_path, 'w') as out:

        for instance in tqdm.tqdm(read_from_input_file(args.input_path), total=file_len(args.input_path)):
            target_lemma = '.'.join(instance.target.split('.')[:-1])
            target_keys = find_numberbatch_keys(target_lemma, args.language, numb_vectors)

            if target_keys is None:
                continue

            target_vector = np.mean([numb_vectors[k] for k in target_keys], axis=0)

            clean_substitutes = []
            for substitute in instance.gold:
                substitute_keys = find_numberbatch_keys(substitute, args.language, numb_vectors)

                if substitute_keys is None:
                    continue

                substitute_vec = np.mean([numb_vectors[k] for k in substitute_keys], axis=0)
                cos_sim = cosine_similarity(target_vector.reshape(1, -1), substitute_vec.reshape(1, -1))
                if cos_sim > args.threshold:
                    clean_substitutes.append(substitute)

            if clean_substitutes != []:
                instance.gold = {k: v for k, v in instance.gold.items() if k in clean_substitutes}
                out.write(instance.__repr__() + '\n')