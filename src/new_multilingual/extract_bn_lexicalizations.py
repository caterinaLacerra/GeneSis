import argparse
import os
from typing import Set, Dict

import requests
import tqdm

from src.utils_wsd import read_from_raganato


def dump_lexicalizations(lexicalizations: Dict[str, Set[str]], output_path: str):
    with open(output_path, 'w') as out:
        for bn_id, candidates in lexicalizations.items():
            str_candidates = "\t".join(lexicalizations[bn_id])
            out.write(f"{bn_id}\t{str_candidates}\n")


def get_lexicalizations(bn_ids: Set[str], lexicalizations: Dict[str, Set[str]], output_path: str, lang_code: str, bn_key: str) -> Dict[str, Set[str]]:

    if os.path.exists(output_path):
        for line in open(output_path):
            bn_id, *candidates = line.strip().split("\t")
            if bn_id not in lexicalizations:
                lexicalizations[bn_id] = set()
            lexicalizations[bn_id].update(candidates)

    for bn_id in tqdm.tqdm(bn_ids):
        if bn_id not in lexicalizations:
            url = f"https://babelnet.io/v6/getSynset?id={bn_id}&targetLang={lang_code}&key={bn_key}"
            lexicalizations[bn_id] = set()
            try:
                response = requests.get(url)
                resp_data = response.json()
                for sense in resp_data["senses"]:
                    if sense["properties"]["language"] == lang_code:
                        if sense["properties"]["lemma"]["type"] == "HIGH_QUALITY":
                            lemma_cand = sense["properties"]["simpleLemma"]
                            lexicalizations[bn_id].add(lemma_cand)
            except:
                return lexicalizations

    return lexicalizations

def extract_bn_ids_from_xml(xml_path: str, gold_path: str, output_path: str):

    bn_ids = set()

    for last_seen_document_id, last_seen_sentence_id, sentence in tqdm.tqdm(read_from_raganato(xml_path, gold_path)):
        for token in sentence:
            if token.labels is not None:
                labels = set(token.labels)
                bn_ids.update(labels)

    with open(output_path, 'w') as out:
        out.write("\n".join(list(bn_ids)))

    print(f"Tot senses: {len(bn_ids)}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_path', required=True)
    parser.add_argument('--gold_path', required=True)
    parser.add_argument('--bn_ids_path', required=True)
    parser.add_argument('--language_code', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    extract_bn_ids_from_xml(args.xml_path, args.gold_path, args.bn_ids_path)

    output_path = args.bn_ids_path.replace(".txt", "_mapping.txt")
    lexicalizations = {}
    bn_ids = set([x.strip() for x in open(args.bn_ids_path)])

    print(f"Writing output in {output_path}")

    while len(lexicalizations) < len(bn_ids):
        lexicalizations = get_lexicalizations(
            bn_ids,
            lexicalizations,
            output_path,
            lang_code=args.language_code,
            bn_key="e4157eb7-d2c6-4689-99a8-794f3aecb8cf"
        )
        dump_lexicalizations(lexicalizations, output_path)

    dump_lexicalizations(lexicalizations, output_path)

