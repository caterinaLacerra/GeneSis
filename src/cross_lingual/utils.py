import json
from typing import Optional, List, Any, Dict, Iterator

import torch
import tqdm
import transformers

from src.wsd.utils.utils import get_target_index_list, file_len


class TranslationInstance:

    def __init__(self, original_sentence: str, instance_id: str, lexeme: str, target_index: str,
                 is_substitute: bool, tokenized_original: Optional[str] = None,
                 tokenized_translated: Optional[str] = None, translated_sentence: Optional[str] = None,
                 inflected_sentence: Optional[str] = None, postag_info: Optional[List[Any]] = None,
                 lemma: Optional[str] = None, pos: Optional[str] = None):

        self.original_sentence = original_sentence
        format_idx_target = get_target_index_list(target_index)

        if pos is not None:
            self.pos = pos

        else:
            self.pos = lexeme.split(".")[-1]

        if is_substitute:
            self.instance_id = f"subst.{instance_id}"
            self.lemma = "_".join([original_sentence.split()[x] for x in format_idx_target])

        else:
            self.instance_id = instance_id
            self.lemma = ".".join(lexeme.split(".")[:-1])

        if lemma is not None:
            self.lemma = lemma

        self.lexeme =  f"{self.lemma}.{self.pos}"
        self.target_idx = format_idx_target
        self.is_substitute = is_substitute
        self.tokenized_original = tokenized_original
        self.tokenized_translated = tokenized_translated
        self.translated_sentence = translated_sentence
        self.postag_info = postag_info
        self.inflected_sentence = inflected_sentence

    def __repr__(self):

        repr = [f"{self.lexeme}\t{self.instance_id}\t{self.target_idx}\t{self.original_sentence}"]

        if self.inflected_sentence is not None:
            repr.append("\n\t\t\t" + self.inflected_sentence)
        if self.translated_sentence is not None:
            repr.append("\n" + f"translation # {self.translated_sentence}")

        if self.tokenized_original is not None:
            repr.append("\n" + f"tok origin # {self.tokenized_original}")

        if self.tokenized_translated is not None:
            repr.append("\n" + f"tok translated # {self.tokenized_translated}")

        return "".join(repr)

    def toJSON(self) -> str:

        clean_data_dict = {k: v for k, v in self.__dict__.items() if k != "target_idx"}
        clean_data_dict["target_index"] = str(self.__dict__["target_idx"])

        return json.dumps(clean_data_dict, sort_keys=True)


def load_instances_from_lexsub_line(input_line: str) -> List[TranslationInstance]:
    lexeme, instance_id, instance_index, sentence, mask, substitutes = input_line.strip().split('\t')

    instances = []

    # original word
    tr_instance = TranslationInstance(sentence, instance_id, lexeme, instance_index, is_substitute=False)
    instances.append(tr_instance)

    dict_substitutes = {x.split('::')[0].replace('_', ' '): float(x.split('::')[1]) for x in substitutes.split()}
    pos_tag = lexeme.split(".")[-1]

    target_idx = get_target_index_list(instance_index)

    for substitute in dict_substitutes:
        if substitute == "":
            continue

        new_lexeme = f"{substitute}.{pos_tag}"
        new_sentence = " ".join(sentence.split()[:target_idx[0]] +
                                [substitute] +
                                sentence.split()[target_idx[-1] + 1:])

        new_index = str([x for x in range(target_idx[0], target_idx[0] + len(substitute.split()))])

        sub_instance = TranslationInstance(new_sentence, instance_id, new_lexeme, new_index,
                                           is_substitute=True)

        instances.append(sub_instance)

    return instances


def load_instances_from_json(json_path: str) -> Iterator[TranslationInstance]:
    for i, line in tqdm.tqdm(enumerate(open(json_path)), desc="Loading instances from file", total=file_len(json_path)):
        data_dict = json.loads(line)
        instance = TranslationInstance(**data_dict)
        yield instance

