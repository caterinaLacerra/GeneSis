import argparse
import os

from src.utils import read_from_input_file, convert_to_lst_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_folder', required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    output_path = os.path.join(args.output_folder, args.input_path.split('/')[-1].replace('.tsv', '_gold.txt'))
    print(f'Writing output path in {output_path}')

    with open(output_path, 'w') as out:
        for instance in read_from_input_file(args.input_path):
            lst_target = convert_to_lst_target(instance.target)

            gold = ";".join([f'{word} {int(score*10)}' for word, score in instance.gold.items()])
            out.write(f'{lst_target} {instance.instance_id} :: {gold}\n')