import argparse
import numpy as np
import os

from sklearn import metrics
import tqdm

from src.utils import file_len


def load_laser_embs(embs_path: str) -> np.array:
    dim = 1024
    X = np.fromfile(embs_path, dtype=np.float32, count=-1)
    X.resize(X.shape[0] // dim, dim)
    return X


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', required=False, default='data/translation/')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    output_path = os.path.join(args.data_folder, f'semcor_0.7_train.it.clean.txt')
    check_removed = os.path.join(args.data_folder, f'semcor_removed.txt')

    count = 0
    with open(output_path, 'w') as out, open(check_removed, 'w') as out_check:

        for split in ['head', 'tail']:

            k_counters = {}

            en_embs = os.path.join(args.data_folder, f'semcor_0.7_train.laser.en.{split}.embs')
            it_embs = os.path.join(args.data_folder, f'semcor_0.7_train.laser.it.{split}.embs')

            en_txt = open(os.path.join(args.data_folder, f'semcor_0.7_train.en.{split}.txt')).readlines()
            it_txt = open(os.path.join(args.data_folder, f'semcor_0.7_train.it.{split}.txt')).readlines()

            for sent in it_txt:
                if sent.strip() not in k_counters:
                    k_counters[sent.strip()] = 0
                k_counters[sent.strip()] += 1

            initial_input_file = os.path.join(args.data_folder, f'semcor_0.7_train.it.formatted.{split}.txt')

            en_matrix = load_laser_embs(en_embs)
            it_matrix = load_laser_embs(it_embs)

            cos_sim = metrics.pairwise.cosine_similarity(en_matrix, it_matrix)

            for idx, line in tqdm.tqdm(enumerate(open(initial_input_file)), total=file_len(initial_input_file)):
                it_sent = line.split('\t')[3]
                k = k_counters[it_sent] * 2
                # top closest sentences
                top_k_idx = np.argpartition(cos_sim[idx], -k)[-k:]
                sorted_idx = top_k_idx[np.argsort(cos_sim[idx][top_k_idx])]

                # check if the EN sentence closest to the IT one is its translation
                if idx in top_k_idx:
                    out.write(line)

                else:
                    count += 1
                    out_check.write(str(idx) + '\t' + it_sent +
                                    '\n\n' +
                                    "".join([f"{x} {cos_sim[idx][x]} {en_txt[x]}" for x in reversed(sorted_idx)]) +
                                    en_txt[idx] +
                                    '------\n')

        print(f'No matching sentences: {count}')
