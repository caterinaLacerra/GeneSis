import argparse
import os

from src.utils import yield_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, help='generation output path, as generated with test_on_wsd.py script.')
    parser.add_argument('--output_folder', required=True, help='output folder where produced files will be saved.')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    dataset_name = args.input_path.split('_')[-1].split('.')[0]

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with open(os.path.join(args.output_folder, f'{dataset_name}.substitutes.txt'), 'w') as substitutes,\
        open(os.path.join(args.output_folder, f'{dataset_name}.info.txt'), 'w') as info:

        for batch in yield_batch(args.input_path, separator='#########\n'):
            target, instance, *index = batch[0].strip().split()
            sentence = batch[1].strip()
            candidates = [b for b in batch if b.startswith('# candidates: ')][0].split('# candidates: ')
            substitutes.write(f'{instance}\t')

            if len(candidates) > 0:
                candidates = candidates[-1].split(';')
                substitutes.write('\t'.join([c.split(': ')[0] for c in candidates]) + '\n')

            info.write(f'{instance}\t{target}\t{sentence}\n')