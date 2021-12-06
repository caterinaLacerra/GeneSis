import argparse
from typing import List, Iterator

import torch
import tqdm
import transformers
from transformers import MarianTokenizer, MarianMTModel

from src.cross_lingual.utils import TranslationInstance, load_instances_from_json
from src.utils import file_len


def process_batch(batch: List[TranslationInstance], translation_model: transformers.AutoModel.from_pretrained,
                  tokenizer_model: transformers.AutoTokenizer.from_pretrained, device: torch.device) -> List[TranslationInstance]:

    sentences = [inst.inflected_sentence if inst.inflected_sentence is not None else inst.original_sentence
                 for inst in batch]
    decoded = []

    with torch.no_grad():
        translated = translation_model.generate(
            **tokenizer_model(sentences, return_tensors="pt", padding=True, max_length=1024).to(device))

        decoded.extend([tokenizer_model.decode(t, skip_special_tokens=True) for t in translated])

    for j, inst in enumerate(batch):
        inst.translated_sentence = decoded[j]

    return batch


def translate_instances(input_jsonl: str, translation_model: transformers.AutoModel.from_pretrained,
                        tokenizer_model: transformers.AutoTokenizer.from_pretrained,
                        batch_size: int, device: torch.device) -> Iterator[List[TranslationInstance]]:

    batch = []
    for instance in tqdm.tqdm(load_instances_from_json(input_jsonl), total=file_len(input_jsonl)):

        if len(batch) + 1 < batch_size:
            batch.append(instance)

        else:
            batch_w_translation = process_batch(batch, translation_model, tokenizer_model, device)
            yield batch_w_translation
            batch = []
            batch.append(instance)

    if len(batch) != 0:
        batch_w_translation = process_batch(batch, translation_model, tokenizer_model, device)
        yield batch_w_translation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name", required=False, default="Helsinki-NLP/opus-mt-en-it")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=300)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    translation_tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    translation_model = MarianMTModel.from_pretrained(args.model_name)
    translation_model.to(device)

    with open(args.output_path, "a") as out:
        for translated_batch in translate_instances(args.input_path, translation_model,
                                                    translation_tokenizer, args.batch_size, device):
            for instance in translated_batch:
                out.write(instance.toJSON() + '\n')
