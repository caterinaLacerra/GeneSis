import argparse

from src.utils import read_from_input_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--output_folder', required=True)
    return parser.parse_args()


def format_for_translation(input_path: str, output_path: str) -> None:

    with open(output_path, 'w') as out:
        for instance in read_from_input_file(input_path):
            out.write(instance.sentence+'\n')


"""PYTHONPATH=nlp_utils/ python nlp_utils/mt/hf.py \
    Helsinki-NLP/opus-mt-en-it \
    -f data/sample/en.txt \
    -o data/sample/it.marian.txt \
    --l1 en --l2 it \
    -n 1 --gen-params num_beams=5 --token-batch-size 800 \
    --cuda-device 0"""


if __name__ == '__main__':
    args = parse_args()