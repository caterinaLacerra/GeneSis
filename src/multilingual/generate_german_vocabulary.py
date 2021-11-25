import string

import tqdm
from datasets import load_dataset
from nltk.corpus import stopwords

if __name__ == '__main__':
    dataset = load_dataset('wikipedia', '20200501.de', split='train')

    vocab = set()
    excluded_words = set([x for x in string.punctuation]).union(set([x for x in stopwords.words("german")]))

    for element in tqdm.tqdm(dataset):
        paragraphs = element['text'].split("\n\n")
        for paragraph in paragraphs:
            for sentence in paragraph.split("\n"):
                words = set([x.lower() for x in sentence.split() if x.lower() not in excluded_words])
                vocab.update(words)

    with open("data/de_vocab.txt", "w") as out:
        for word in vocab:
            out.write(word + '\n')
