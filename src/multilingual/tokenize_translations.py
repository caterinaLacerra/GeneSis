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


if __name__ == '__main__':

    nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)
    nlp_it = stanza.Pipeline(lang='it', processors='tokenize', tokenize_no_ssplit=True)

    input_file = 'data/translation/semcor_0.7_train.it.txt'
    output_file = 'data/translation/semcor_0.7_train.it.tokenized.txt'

    with open(output_file, 'w') as out, tqdm.tqdm(total=file_len(input_file)) as pbar:

        joint_sentences = []

        for i, sentence in enumerate(open(input_file)):

            joint_sentences.append(sentence.strip())

            if i > 0 and i % 2000 == 0:

                # process
                en_sentences, it_sentences = [], []
                for s in joint_sentences:
                    en_sentences.append(s.split('\t')[0])
                    it_sentences.append(s.split('\t')[1])

                to_write_en = tokenize_batch(en_sentences)
                to_write_it = tokenize_batch(it_sentences)

                # write
                for en_sent, it_sent in zip(to_write_en, to_write_it):
                    out.write(f'{en_sent} ||| {it_sent}\n')
                    pbar.update()

                # update batch
                joint_sentences = []

        if joint_sentences != []:

            en_sentences, it_sentences = [], []

            for s in joint_sentences:
                en_sentences.append(s.split('\t')[0])
                it_sentences.append(s.split('\t')[1])

            to_write_en = tokenize_batch(en_sentences)
            to_write_it = tokenize_batch(it_sentences)

            # write
            for en_sent, it_sent in zip(to_write_en, to_write_it):
                out.write(f'{en_sent} ||| {it_sent}\n')
                pbar.update()

