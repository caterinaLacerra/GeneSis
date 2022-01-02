import argparse
import os
from typing import Set, Dict, Tuple, Iterator, List

import requests
import stanza
import torch
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel, BertTokenizer, BertForMaskedLM

from multimirror.multimirror.scripts.battleship_predict import Predictor


def get_sentence_and_target(input_sent: str) -> Tuple[str, str]:
    words = input_sent.split()
    # get clean target word in context
    target_word = " ".join([w for w in words if w.startswith('**')
                            or w.endswith('**')]) \
        .replace("**", "") \
        .replace(",", "") \
        .replace(".", ""). \
        replace(")", "")
    sentence = input_sent.replace("**", "")
    return target_word, sentence


def load_cached_dictionary(cache_path: str) -> Dict[str, Set[str]]:
    word_to_candidates = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            word, *candidates = line.strip().split("\t")
            if word not in word_to_candidates:
                word_to_candidates[word] = set()
            word_to_candidates[word].update(candidates)

    return word_to_candidates


def precompute_candidates(lexemes: Set[Tuple[str, str]], lang_code: str, cached_dict: Dict[str, Set[str]],
                          cache_path: str) -> Dict[str, Set[str]]:
    lexeme_to_candidates = {}

    for (lemma, pos) in tqdm(lexemes):
        candidates = get_candidates(lemma, pos, cached_dict, lang_code, cache_path)
        dict_key = f"{lemma}.{pos}"
        lexeme_to_candidates[dict_key] = candidates

    return lexeme_to_candidates


def get_input_batch(input_path: str, batch_size: int) -> Iterator[Tuple]:
    target_words, input_text = [], []
    original_sentences = []

    input_lines = []
    for line in open(input_path):

        if len(input_lines) > batch_size:
            yield target_words, input_text, original_sentences, input_lines
            target_words, input_text, original_sentences, input_lines = [], [], [], []

        bn_id, sensekey, main_lemma, gloss, context, gold_subst = line.strip().split('\t')
        target, sentence = get_sentence_and_target(context)
        target_words.append(target)
        input_text.append(sentence)
        original_sentences.append(sentence)
        input_lines.append(line.strip())

    if len(input_lines) != 0:
        yield target_words, input_text, original_sentences, input_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--ranking', required=False, action="store_true", default=False)
    parser.add_argument('--wrong_path', required=True)
    parser.add_argument('--batch_size', required=False, type=int, default=200)
    parser.add_argument('--mm_ckpt', required=False,
                        default="/home/caterina/PycharmProjects/multimirror/experiments/training/"
                                "mbert_battleship_our-dataset_en-it_flip-langs_optimus-mul_6L/"
                                "epoch=275-val_f1=0.96.ckpt",
                        help="path to multimirror ckpt")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    print(f"Loading models...")

    device = torch.device("cuda")
    multimirror_device = 0

    translation_model_name = "Helsinki-NLP/opus-mt-en-it"
    # ranking_model_name = "bert-base-multilingual-cased"

    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name)
    translation_model.to(device)

    en_pipeline = stanza.Pipeline(lang='en', processors="tokenize,pos,lemma", tokenize_no_ssplit=True)
    it_pipeline = stanza.Pipeline(lang='it', processors="tokenize", tokenize_no_ssplit=True)
    lemmatization_pipe = stanza.Pipeline(lang='it', processors="tokenize,pos,lemma",
                                         tokenize_no_ssplit=True, lemma_model_path="/home/caterina/stanza_resources/"
                                                                                   "it/lemma/isdt_customized.pt")

    predictor = Predictor(args.mm_ckpt, multimirror_device)
    print(f"...Done.")

    correct = 0
    retrieved = 0
    total = 0

    out_dir = f"data/translated_datasets/tokenization"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.out_path, 'w') as output, open(args.wrong_path, 'w') as wrong:

        for target_words, input_text, original_sentences, input_lines in get_input_batch(args.path,
                                                                                         args.batch_size):
            tot_batch, correct_batch, retrieved_batch = 0, 0, 0

            doc = en_pipeline("\n\n".join(input_text))

            pos_tags = [None] * len(target_words)
            lemmas = [None] * len(target_words)

            for i, sent in enumerate(doc.sentences):
                for x in sent.words:
                    if x.text == target_words[i]:
                        pos_tags[i] = x.upos
                        lemmas[i] = x.lemma
                        break

            print(f"Translating...")
            original_decoded = translate(original_sentences, device, translation_model, translation_tokenizer)

            # tokenize before alignment
            input_data_file = tokenize("\n\n".join(original_sentences),
                                       "\n\n".join(original_decoded),
                                       original_sentences, out_dir, en_pipeline, it_pipeline)

            # align and get translations
            for i, (src, tgt, tg_word) in tqdm(enumerate(zip(original_sentences, original_decoded, target_words)),
                                               total=len(original_sentences)):
                total += 1
                tot_batch += 1

                if lemmas[i] is None or pos_tags[i] is None:
                    continue

                aligned = align_multimirror(src, tgt, predictor, tg_word)
                if aligned != "":
                    retrieved += 1
                    retrieved_batch += 1

                doc = lemmatization_pipe(tgt)
                aligned_tokens = aligned.split()
                corresponding_lemmas = [x.lemma for sent in doc.sentences
                                        for x in sent.words if x.text in aligned_tokens
                                        if x.upos != "AUX" and x.upos != "PRON"]

                lemmatized_align = ' '.join(corresponding_lemmas)

                pool_candidates = get_candidates(lemmas[i], pos_tags[i], lang_code="IT", cached_dict=cached_dictionary,
                                                 cache_path=cache_path)

                bn_id, sensekey, _, gloss, context, gold_words = input_lines[i].split('\t')
                gold = gold_words.split("; ")
                added = []
                if pos_tags[i] == "VERB":
                    for x in gold:
                        if x.endswith("si"):
                            gold.append(x[:-2] + "e")
                            added.append(x[:-2] + "e")

                output.write(f"{bn_id} {sensekey} {context}\n"
                             f"gloss: {gloss}\n"
                             f"gold: {gold_words} ++ {' '.join(added)}\n"
                             f"candidates for {lemmas[i]}.{pos_tags[i]}: {';'.join(pool_candidates)}\n"
                             f"original translation: {original_decoded[i]}\n"
                             f"aligned: {aligned} -- {lemmatized_align}\n")

                if aligned in gold or lemmatized_align in gold:
                    correct += 1
                    correct_batch += 1
                    output.write(f"correct alignment: {aligned} -- {lemmatized_align}\n\n")

                else:
                    wrong.write(f"{bn_id} {sensekey} {context}\n"
                                f"gloss: {gloss}\n"
                                f"gold: {gold_words} + {' '.join(added)}\n"
                                f"candidates for {lemmas[i]}.{pos_tags[i]}: {';'.join(pool_candidates)}\n"
                                f"translation: {original_decoded[i]}\n"
                                f"aligned: {aligned} -- {lemmatized_align}\n")

                    output.write(f"wrong alignment: {aligned} -- {lemmatized_align}\n\n")
                    wrong.write(f"wrong alignment: {aligned} -- {lemmatized_align}\n\n")

            if retrieved_batch != 0:
                print(f"Precision batch: {correct_batch/retrieved_batch}")
            else:
                print(f"No sentences retrieved")
            print(f"Recall: {correct_batch / tot_batch}")
            torch.cuda.empty_cache()

    if retrieved != 0:
        print(f"Precision: {correct / retrieved}")

    else:
        print(f"No sentences retrieved")

    print(f"Recall: {correct / total}")
