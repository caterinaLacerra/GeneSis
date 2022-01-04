import argparse
import os
import string
from collections import defaultdict
from typing import Tuple, Dict, Any, Iterator, List, Set

import nltk
import pytorch_lightning as pl
import stanza
import torch
import tqdm
import transformers
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
from wordfreq import zipf_frequency

from src.model import BartModel
from src.utils import read_from_input_file, align_multimirror, file_len
from multimirror.multimirror.scripts.battleship_predict import Predictor


def get_mapping_to_en(alignment_path: str, input_path: str, predictor, mm_tokenizer) -> Iterator[Dict[
    Tuple[Any, Any], Dict[str, Any]]]:

    sentence_to_target_idx = defaultdict(list)

    for instance in read_from_input_file(input_path):
        sentence_to_target_idx[instance.sentence].append(instance)

    mapping_dict = {}

    # from it/de sentence, get en sentence with target
    for line in open(alignment_path):
        line = line.strip()
        original, translated, *alignments = line.split('\t')

        if translated in sentence_to_target_idx:
            for instance in sentence_to_target_idx[translated]:
                # recover en alignments for each target
                temp = align_multimirror(translated, original, predictor,
                                          instance.target_idx, mm_tokenizer)

                if temp is not None:
                    alignment, alignment_indexes = temp
                    mapping_dict[(instance.target, translated)] = {
                        "src-indexes": instance.target_idx,
                        "aligned-translation": alignment,
                        "alignment-indexes": alignment_indexes,
                        "translated-sentence": original,
                        "original-substitutes": instance.gold,
                        "instance-id": instance.instance_id
                    }
                    yield (instance.target, translated, mapping_dict[(instance.target, translated)])


def get_en_substitutes(
        en_sentence: str, en_target: str, target_index: List[int],
        model: pl.LightningModule,
        exclude_words: Set[str],
        device: torch.device,
        start_token: str = "<t>",
        end_token: str = "</t>"
) -> List[str]:

    # from en sentence + target, get en substitutes
    format_sentence = f"{' '.join(en_sentence.split()[:target_index[0]])} {start_token}{en_target}{end_token} " \
                      f"{' '.join(en_sentence.split()[target_index[-1] + 1:])}"

    input_ids = model.tokenizer.prepare_seq2seq_batch([format_sentence], return_tensors='pt')['input_ids'].to(device)
    input_padding_mask = input_ids != model.tokenizer.pad_token_id

    batch = {
            "source": input_ids,
            "source_padding_mask": input_padding_mask,
        }

    str_input, str_generation, generation = model.generate(batch)
    clean_replacements = clean_substitutes(str_generation, exclude_words)
    return clean_replacements


def clean_substitutes(batch: List[str], exclude_words: Set[str]) -> List[str]:

    # include check on multiwords, single words and de-biasing
    substitutes = [[y.strip(',').lower() for y in x.strip().split(', ')] for x in batch]

    clean_substitutes = []

    for sub_line in substitutes:
        for word in sub_line:
            words = word.split('\n')
            for w in words:
                w = w.strip(',')
                if w not in clean_substitutes:
                    if w.lower().strip() not in exclude_words and \
                            not any(x in exclude_words for x in w.lower().strip().split()):
                        clean_substitutes.append(w)

    return clean_substitutes

def translate_substitutes(substitutes: List[str], en_target_sentence: str, en_target_index: List[int],
                          source_sentence: str, source_indexes: List[int],
                          src_pipeline: stanza.Pipeline) -> List[Tuple[str, List[int], Any]]:

    # compose sentences with substitutes
    original_tokens = en_target_sentence.split()
    new_sentences = [en_target_sentence]
    new_indexes = [en_target_index]
    for sub in substitutes:
        new_sentences.append(f"{' '.join(original_tokens[:en_target_index[0]])} {sub} "
                             f"{' '.join(original_tokens[en_target_index[-1] + 1:])}")

        if len(sub.split()) == 1:
            new_indexes.append([en_target_index[0]])

        else:
            new_indexes.append([x for x in range(en_target_index[0], en_target_index[0] + len(sub.split()))])

    # translate batch
    tokenizer_output = translation_tokenizer(new_sentences,
                                             return_tensors="pt",
                                             padding=True,
                                             truncation=True,
                                             max_length=1024)

    translated = translation_model.generate(input_ids=tokenizer_output["input_ids"].to(device),
                                            attention_mask=tokenizer_output["attention_mask"].to(device))

    translated_sentences = [translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    # align substitutes
    aligned_substitutes = []
    for translation in translated_sentences:
        src_doc = src_pipeline("\n\n".join([translation]))

        for i, it_sent in enumerate(src_doc.sentences):
            tokenized_translation = " ".join([token.text for token in it_sent.tokens])
            try:
                alignment = align_multimirror(en_target_sentence, tokenized_translation, predictor,
                                           source_indexes, mm_tokenizer)
            except IndexError:
                alignment = None

            if alignment is not None:
                aligned_word, aligned_index = alignment
                aligned_substitutes.append((aligned_word, aligned_index, tokenized_translation))

    return aligned_substitutes

def clean_translations(aligned_translations: List[str], target: str,
                       src_pipeline: stanza.Pipeline, tgt_sentences: List[str],
                       tgt_indexes: List[List[int]], lang: str,
                       exclude_words: Set[str]):

    # todo: de-biasing in tgt language
    *target_words, target_pos = target.split(".")
    target = ".".join(target_words)

    src_doc = src_pipeline("\n\n".join(tgt_sentences))

    clean_subst = []
    for i, (index, it_sent, subst) in enumerate(zip(tgt_indexes, src_doc.sentences, aligned_translations)):
        lemmatized_translation = [word.lemma for word in it_sent.words]
        lemma = " ".join([lemmatized_translation[x] for x in index])

        # remove original target and its lemmatized forms, check if words exist
        if lemma != target and subst != target and zipf_frequency(subst, lang) > 0:

            # remove duplicates and multiwords
            if lemma not in clean_subst and len(lemma.split()) == 1 and lemma not in exclude_words:
                clean_subst.append(lemma.replace(" ", "_"))

    return clean_subst


def load_best_ckpt(map_location: str):
    ckpt_path = 'weights/semcor_148/best.ckpt'
    model = BartModel.load_from_checkpoint(ckpt_path, strict=False, map_location=map_location)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--multimirror_folder", required=False, default="/media/ssd1/caterina/multimirror/")

    parser.add_argument('--mm_ckpt', required=False,
                        default="/home/caterina/PycharmProjects/multimirror/experiments/training/"
                                "mbert_battleship_our-dataset_en-it_flip-langs_optimus-mul_6L/"
                                "epoch=275-val_f1=0.96.ckpt",
                        help="path to multimirror ckpt")

    parser.add_argument("--translation_name", required=False, default="Helsinki-NLP/opus-mt-en-it")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--lang", required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)


    language_mapping = {'it': 'italian', 'de': 'german'}
    multimirror_device = 0

    exclude_words_src = set([x for x in stopwords.words(language_mapping[args.lang])]).union(set(string.punctuation))

    if not args.cpu:
        device = torch.device('cuda')
        map_location = 'cuda:0'

    else:
        device = torch.device('cpu')
        map_location = 'cpu'

    translation_model_name = f"Helsinki-NLP/opus-mt-en-{args.lang}"

    generative_model = load_best_ckpt(map_location=map_location)
    generative_model.to(device)

    translation_tokenizer = transformers.AutoTokenizer.from_pretrained(translation_model_name)
    translation_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)
    translation_model.to(device)

    stanza_pipeline = stanza.Pipeline(lang=args.lang, processors="tokenize", tokenize_no_ssplit=True)
    it_pipeline = stanza.Pipeline(lang=args.lang, processors="tokenize,pos,lemma", tokenize_no_ssplit=True)

    original_dataset_name = args.input_path.split("/")[-1].split(".")[1]
    lang = args.input_path.split("/")[-1].split(".")[2].split("_")[0]

    predictor = Predictor(args.mm_ckpt, multimirror_device)
    mm_tokenizer = AutoTokenizer.from_pretrained(predictor.module.hparams.transformer_model, use_fast=True)

    if original_dataset_name == "semcor" or original_dataset_name == "wngt":
        associated_mm_path = f"{args.multimirror_folder}" \
                             f"{original_dataset_name}-translated-{lang}-ours-loosebn/" \
                             f"proj.output.tsv"

        output_path = os.path.join(args.output_folder, f"{original_dataset_name}.{lang}_train2.tsv")

        with open(output_path, 'w') as out:

            for i, (target, sentence, mapping_dict) in tqdm.tqdm(enumerate(get_mapping_to_en(associated_mm_path, args.input_path,
                                                                    predictor, mm_tokenizer)),
                                                            total=file_len(associated_mm_path)):

                en_indexes = mapping_dict["alignment-indexes"]
                en_target = " ".join([mapping_dict["translated-sentence"].split()[x] for x in en_indexes])
                en_sentence = mapping_dict["translated-sentence"]
                exclude_words = {en_target}

                scored_substitutes = get_en_substitutes(en_sentence, en_target, en_indexes, generative_model,
                                                        exclude_words, device=device)

                source_indexes = mapping_dict["alignment-indexes"]
                aligned_translations = translate_substitutes(scored_substitutes, en_sentence, en_indexes,
                                                             sentence, source_indexes, stanza_pipeline)

                words, indexes, translations = [], [], []
                for w, idx, translate in aligned_translations:
                    words.append(w)
                    indexes.append(idx)
                    translations.append(translate)

                final_translations = clean_translations(words, target, it_pipeline, translations, indexes, lang, exclude_words_src)
                if len(final_translations) > 0:

                    str_translations = " ".join([f"{w}::1" for w in final_translations])
                    out.write(f"{target}\t{mapping_dict['instance-id']}\t{mapping_dict['src-indexes']}"
                              f"\t{sentence}\t---\t{str_translations}\n")

    else:
        print(f"No corresponding multimirror path for {original_dataset_name}")
        exit()

