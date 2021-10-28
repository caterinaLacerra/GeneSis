import string

import tqdm
from datasets import load_dataset
from nltk.corpus import stopwords

from src.utils import contains_punctuation, contains_number

if __name__ == '__main__':
    dataset = load_dataset('wikipedia', '20200501.de', split='train')

    vocab = set()
    excluded_words = set([x for x in string.punctuation]).union(set([x for x in stopwords.words("german")]))
    punct = set([x for x in string.punctuation]).union(["“", "–", "’", "„"])

    for element in tqdm.tqdm(dataset):
        paragraphs = element['text'].split("\n\n")
        for paragraph in paragraphs:
            for sentence in paragraph.split("\n"):

                words = set([x.lower() for x in sentence.split()
                             if len(x.lower()) > 1 and
                             x.lower() not in excluded_words and
                             not contains_punctuation(x.lower(), punct) and
                             not contains_number(x.lower())])

                vocab.update(words)

    with open("data/de_vocab.txt", "w") as out:
        for word in vocab:
            out.write(word + '\n')
