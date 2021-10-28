import argparse
from typing import Dict
import xml.etree.ElementTree as ET

from src.utils import convert_to_universal_target
from src.wsd.utils.utils import LexSubInstance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_path', required=True)
    parser.add_argument('--gold_path', required=True)
    parser.add_argument('--output_path', required=True)
    return parser.parse_args()

def is_multiword(word: str) -> bool:
    return len(word.split()) > 1 or len(word.split('_')) > 1

def multiword_len(word: str) -> int:
    if "_" in word:
        return len(word.split("_"))
    return len(word.split())



def get_gold_dictionary(gold_path: str) -> Dict[str, Dict[str, int]]:
    instance_to_gold = {}
    for line in open(gold_path):
        target, instance_id = line.split(' :: ')[0].split()
        substitutes = line.strip().split(' :: ')[1].split(';')
        instance_to_gold[instance_id] = {}
        for substitute in substitutes:
                *words, score = substitute.split(' ')
                clean_word = " ".join([x for x in words if x != ""])
                if clean_word != "":
                    instance_to_gold[instance_id][clean_word] = int(score)
    return instance_to_gold


def read_germeval_xml(xml_path: str) -> Dict[str, LexSubInstance]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    id_to_instance = {}

    for element in root.iter():
        if element.tag == "lexelt":
            for instance in element.iter():
                if instance.tag == "instance":
                    target = element.attrib["item"]
                    lexelt = convert_to_universal_target(target)
                    instance_id = instance.attrib["id"]

                    for context in instance.iter(tag="context"):
                        left_context = context.text.strip()
                        head = list(context.iter(tag="head"))[0]
                        target_word = head.text.strip()
                        right_context = head.tail.strip()
                        sentence = left_context + " " + target_word + right_context

                        if is_multiword(target_word):
                            target_idx = [x for x in range(len(left_context.split()),
                                                           len(left_context.split()) + multiword_len(target_word))]
                        else:
                            target_idx = [len(left_context.split())]

                        instance = LexSubInstance(lexelt, instance_id, target_idx, sentence)
                        id_to_instance[instance_id] = instance

    return id_to_instance


if __name__ == '__main__':
    args = parse_args()

    instance_to_gold = get_gold_dictionary(args.gold_path)
    id_to_instance = read_germeval_xml(args.xml_path)

    for instance_id, instance in id_to_instance.items():
        instance.gold = instance_to_gold[instance_id]

    with open(args.output_path, 'w') as out:
        for id, instance in id_to_instance.items():
            out.write(str(instance) + '\n')