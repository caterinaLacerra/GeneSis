import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_folder")
    parser.add_argument("--cleaned_vocab_folder")
    return parser.parse_args()

def cut_vocab_len(old_folder: str, new_folder: str, max_len: int):

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for file in os.listdir(old_folder):
        with open(os.path.join(new_folder, file), 'w') as out:
            for line in open(os.path.join(old_folder, file)):
                word = line.strip()
                if len(word) <= max_len and len(word) > 0:
                    out.write(line)

if __name__ == '__main__':

    args = parse_args()

    tot_words, avg, max = 0, 0, 0
    for file in os.listdir(args.vocab_folder):
        for line in open(os.path.join(args.vocab_folder, file)):
            word = line.strip()
            if len(word) > max:
                max = len(word)
            avg += len(word)
            tot_words += 1
    print(f"Avg: {avg/tot_words}")
    print(f"Max len: {max}")
    print(f"Tot words: {tot_words}")

    cut_vocab_len(args.vocab_folder, args.cleaned_vocab_folder, max_len=20)


