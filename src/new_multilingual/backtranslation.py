import argparse
from typing import Dict, Any, Iterator, Tuple

import torch
import tqdm
import transformers

from src.utils import read_from_input_file


def build_sentences_associations(input_path_sentences: str, input_path_instances: str) -> Dict[str, Any]:

    translated_to_original = {}
    for line in open(input_path_sentences):
        original, translated = line.strip().split('\t')
        translated_to_original[translated] = {"original_sentence": original}

    for instance in read_from_input_file(input_path_instances):
        translated = instance.sentence
        if translated in translated_to_original:
            translated_to_original[translated].update({"instance": instance})

    return translated_to_original

def translate(
        model: transformers.AutoModel.from_pretrained,
        tokenizer: transformers.AutoTokenizer.from_pretrained,
        device: torch.device,
        input_dictionary: Dict[str, Any],
        batch_size: int
) -> Iterator[Tuple[str, Dict[str, Any]]]:

    dict_keys = list(input_dictionary)
    for i in tqdm.tqdm(range(0, len(dict_keys), batch_size)):

        batch_sentences = dict_keys[i: i + batch_size]

        with torch.no_grad():
            input_tokens = tokenizer(batch_sentences,
                                     return_tensors="pt",
                                     padding=True,
                                     max_length=1024).to(device)

            translated = model.generate(**input_tokens)
            decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        for j, k in enumerate(batch_sentences):
            input_dictionary[k]["translated_to_en"] = decoded[j]

            yield (k, input_dictionary[k])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances_path", required=True)
    parser.add_argument("--aligned_path", required=True)
    parser.add_argument("--output_path", required=True)
    #parser.add_argument("--discarded_path", required=True)
    parser.add_argument("--device", required=False, default="cuda")
    parser.add_argument("--model_name", required=False, default="Helsinki-NLP/opus-mt-it-en")
    parser.add_argument("--batch_size", required=False, type=int, default=250)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    model = transformers.MarianMTModel.from_pretrained(args.model_name)
    tokenizer = transformers.MarianTokenizer.from_pretrained(args.model_name)
    device = torch.device(args.device)
    model.to(device)

    alignments_dict = build_sentences_associations(args.aligned_path, args.instances_path)
    with open(args.output_path, 'w') as out:
        for translated, dict_translation in translate(model, tokenizer, device, alignments_dict, args.batch_size):
            if 'instance' in dict_translation:
                out.write(f"{dict_translation['original_sentence']}\t{dict_translation['translated_to_en']}"
                      f"\t{translated}\t{dict_translation['instance'].instance_id}\n")

