import argparse
import os
from typing import Set

import requests
import tqdm

from src.utils import read_from_input_file


def get_candidates(lexemes: Set[str], target_lang_code: str, source_lang_code: str, cache_path: str, bn_key: str):
    candidates_dict = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            lexeme, *candidates = line.strip().split("\t")
            candidates_dict[lexeme] = candidates

    for lexeme in tqdm.tqdm(lexemes):
        if lexeme not in candidates_dict:
            *lemma, pos = lexeme.split(".")
            lemma = ".".join(lemma)

            url = f"https://babelnet.io/v6/getSynsetIds?lemma={lemma}&searchLang=" \
                  f"{source_lang_code.upper()}&pos={pos}&key={bn_key}"
            try:
                response = requests.get(url)
                resp_data = response.json()

                # get all BN ids for the lexeme
                bn_ids = [bn_data["id"] for bn_data in resp_data]
                candidates_dict[lexeme] = set()

                # get lexicalizations for each synset
                for synset_id in bn_ids:
                    url = f"https://babelnet.io/v6/getSynset?id={synset_id}&" \
                          f"targetLang={target_lang_code}&key={bn_key}"
                    try:
                        response = requests.get(url)
                        resp_data = response.json()
                        for sense in resp_data["senses"]:
                            if sense["properties"]["language"] == target_lang_code.upper():
                                if sense["properties"]["lemma"]["type"] == "HIGH_QUALITY":
                                    candidates_dict[lexeme].add(sense["properties"]["simpleLemma"])

                    except:
                        with open(cache_path, 'w') as out:
                            for lexeme, candidates in candidates_dict.items():
                                str_candidates = "\t".join(candidates)
                                out.write(f"{lexeme}\t{str_candidates}\n")
                        exit()

            except:
                with open(cache_path, 'w') as out:
                    for lexeme, candidates in candidates_dict.items():
                        str_candidates = "\t".join(candidates)
                        out.write(f"{lexeme}\t{str_candidates}\n")
                exit()

    with open(cache_path, 'w') as out:
        for lexeme, candidates in candidates_dict.items():
            str_candidates = "\t".join(candidates)
            out.write(f"{lexeme}\t{str_candidates}\n")

def get_neighbours_candidates(lexemes: Set[str], target_lang_code: str, source_lang_code: str, cache_path: str, bn_key: str):
    candidates_dict = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            lexeme, *candidates = line.strip().split("\t")
            candidates_dict[lexeme] = candidates

    for lexeme in tqdm.tqdm(lexemes):
        if lexeme not in candidates_dict:
            *lemma, pos = lexeme.split(".")
            lemma = ".".join(lemma)

            url = f"https://babelnet.io/v6/getSynsetIds?lemma={lemma}&searchLang=" \
                  f"{source_lang_code.upper()}&pos={pos}&key={bn_key}"
            try:
                response = requests.get(url)
                resp_data = response.json()

                # get all BN ids for the lexeme
                bn_ids = set([bn_data["id"] for bn_data in resp_data])

                # get all outgoing edges
                for bn_id in bn_ids:
                    new_url = f"https://babelnet.io/v6/getOutgoingEdges?id={bn_id}&key={bn_key}"
                    try:
                        response = requests.get(new_url)
                        resp_data = response.json()
                        for element in resp_data:
                            bn_ids.add(element["target"])

                    except:
                        with open(cache_path, 'w') as out:
                            for lexeme, candidates in candidates_dict.items():
                                str_candidates = "\t".join(candidates)
                                out.write(f"{lexeme}\t{str_candidates}\n")
                        exit()

                candidates_dict[lexeme] = set()

                # get lexicalizations for each synset
                for synset_id in bn_ids:
                    url = f"https://babelnet.io/v6/getSynset?id={synset_id}&" \
                          f"targetLang={target_lang_code}&key={bn_key}"
                    try:
                        response = requests.get(url)
                        resp_data = response.json()
                        for sense in resp_data["senses"]:
                            if sense["properties"]["language"] == target_lang_code.upper():
                                if sense["properties"]["lemma"]["type"] == "HIGH_QUALITY":
                                    candidates_dict[lexeme].add(sense["properties"]["simpleLemma"])

                    except:
                        with open(cache_path, 'w') as out:
                            for lexeme, candidates in candidates_dict.items():
                                str_candidates = "\t".join(candidates)
                                out.write(f"{lexeme}\t{str_candidates}\n")
                        exit()

            except:
                with open(cache_path, 'w') as out:
                    for lexeme, candidates in candidates_dict.items():
                        str_candidates = "\t".join(candidates)
                        out.write(f"{lexeme}\t{str_candidates}\n")
                exit()

    with open(cache_path, 'w') as out:
        for lexeme, candidates in candidates_dict.items():
            str_candidates = "\t".join(candidates)
            out.write(f"{lexeme}\t{str_candidates}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--src_language", required=True)
    parser.add_argument("--tgt_language", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    BN_KEY = "e4157eb7-d2c6-4689-99a8-794f3aecb8cf"

    if os.path.exists(args.output_path):
        init_candidates = {line.strip().split('\t')[0]: set([c for c in line.strip().split('\t')[1:]])
                           for line in open(args.output_path)}
    else:
        init_candidates = {}

    lexemes_list = set([x.target for x in read_from_input_file(args.test_path)])

    while (len(init_candidates)) < len(lexemes_list):
        get_candidates(lexemes_list, target_lang_code=args.tgt_language,
                       source_lang_code=args.src_language,
                       cache_path=args.output_path, bn_key=BN_KEY)
        init_candidates = {line.strip().split('\t')[0]: set([c for c in line.strip().split('\t')[1:]])
                           for line in open(args.output_path)}
