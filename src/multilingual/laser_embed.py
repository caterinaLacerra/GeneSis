import argparse
import os
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--laser_folder', required=True)
    parser.add_argument('--embeddings_folder', required=True)
    return parser.parse_args()


if __name__ == '__main__':

    os.environ["LASER"] = "/home/caterina/LASER"

    args = parse_args()

    if not os.path.exists(args.embeddings_folder):
        os.makedirs(args.embeddings_folder)

    for file in os.listdir(args.laser_folder):
        if 'formatted' in file:
            continue

        else:
            dataset_name = file.split('laser')[0]
            short_name = file.split('laser')[1].split('.txt')[0]

            _, language, chunk_id = short_name.split('.')
            output_path = os.path.join(args.embeddings_folder, f'{dataset_name}{language}.{chunk_id}.embs')
            command = ['zsh', os.path.abspath("/home/caterina/LASER/tasks/embed/embed.sh"),
                       os.path.abspath(os.path.join(args.laser_folder, file)),
                       language,
                       os.path.abspath(output_path)]

            result = subprocess.run(command, check=True)
