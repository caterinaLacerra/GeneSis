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
    parser.add_argument('--laser_folder', required=False, default='data/translation/laser_embeddings')
    parser.add_argument('--embeddings_folder', required=False, default='data/translation/embeddings')
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--language', required=True)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    count = 0
    with open(args.output_path, 'w') as out:

        lang_embs = [path for path in os.listdir(args.embeddings_folder)
                   if path.endswith('.embs') and f'.{args.language}.' in path]

        for path in lang_embs:
            en_path = path.replace(f".{args.language}.", ".en.")
            lang_emb = os.path.join(args.embeddings_folder, path)
            en_emb = os.path.join(args.embeddings_folder, en_path)

            k_counters = {}

            en_txt = open(os.path.join(args.laser_folder, en_path.replace("train", "train.laser").replace(".embs", ".txt"))).readlines()
            lang_txt = open(os.path.join(args.laser_folder, path.replace("train", "train.laser").replace(".embs", ".txt"))).readlines()

            for sent in lang_txt:
                if sent.strip() not in k_counters:
                    k_counters[sent.strip()] = 0
                k_counters[sent.strip()] += 1

            initial_input_file = os.path.join(args.laser_folder, en_path.replace('en', "formatted").replace("embs", "txt"))

            en_matrix = load_laser_embs(en_emb)
            it_matrix = load_laser_embs(lang_emb)

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

        print(f'No matching sentences: {count}')
