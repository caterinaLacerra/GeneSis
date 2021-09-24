import argparse
from typing import List

import stanza
import tqdm

from src.utils import file_len


def tokenize_batch(sentences: List[str]) -> List[str]:

    to_write = []

    doc = '\n\n'.join(sentences)
    doc = nlp(doc)

    for s in doc.sentences:
        to_write.append(" ".join([f'{token.text}' for token in s.tokens]))

    return to_write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lang', type=str, default="en", help="Source language")
    parser.add_argument('--tgt_lang', type=str, required=True, help="Target language")
    parser.add_argument('--input_path', type=str, required=True, help="Input .txt file, "
                                                                      "as produced by the nlp-utils "
                                                                      "translation script")
    parser.add_argument('--output_path', type=str, required=True, help="Output .txt file path")

    parser.add_argument('--input_sep', type=str, default=" ||| ")
    parser.add_argument('--output_sep', type=str, default=" ||| ")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    nlp = stanza.Pipeline(lang=args.src_lang, processors='tokenize', tokenize_no_ssplit=True)
    nlp_it = stanza.Pipeline(lang=args.tgt_lang, processors='tokenize', tokenize_no_ssplit=True)


    with open(args.output_path, 'w') as out, tqdm.tqdm(total=file_len(args.input_path)) as pbar:

        joint_sentences = []

        for i, sentence in enumerate(open(args.input_path)):

            joint_sentences.append(sentence.strip())

            if i > 0 and i % 2000 == 0:

                # process
                en_sentences, it_sentences = [], []
                for s in joint_sentences:
                    en_sentences.append(s.split(args.input_sep)[0])
                    it_sentences.append(s.split(args.input_sep)[1])

                to_write_en = tokenize_batch(en_sentences)
                to_write_it = tokenize_batch(it_sentences)

                # write
                for en_sent, it_sent in zip(to_write_en, to_write_it):
                    out.write(f'{en_sent}{args.output_sep}{it_sent}\n')
                    pbar.update()

                # update batch
                joint_sentences = []

        if joint_sentences != []:

            en_sentences, it_sentences = [], []

            for s in joint_sentences:
                en_sentences.append(s.split(args.input_sep)[0])
                it_sentences.append(s.split(args.input_sep)[1])

            to_write_en = tokenize_batch(en_sentences)
            to_write_it = tokenize_batch(it_sentences)

            # write
            for en_sent, it_sent in zip(to_write_en, to_write_it):
                out.write(f'{en_sent}{args.output_sep}{it_sent}\n')
                pbar.update()

