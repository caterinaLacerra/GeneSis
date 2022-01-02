import argparse
from collections import defaultdict
from typing import List

from src.utils import read_from_input_file


def get_substitutes(input_path: str):
    avg_substitutes = []
    avg_substitutes_pos = defaultdict(list)
    for instance in read_from_input_file(input_path):
        avg_substitutes.append(len(instance.gold))
        postag = instance.target.split(".")[-1]
        avg_substitutes_pos[postag].append(len(instance.gold))

    print(f"Average substitutes: {sum(avg_substitutes) / len(avg_substitutes)}")
    for pos in avg_substitutes_pos:
        print(f"Average substitutes per {pos}: {sum(avg_substitutes_pos[pos]) / len(avg_substitutes_pos[pos])}")


def get_target_words(input_path: str):
    targets = set([instance.target for instance in read_from_input_file(input_path)])
    print(f"Total target words: {len(targets)}")

    targets_per_pos = defaultdict(set)
    for instance in read_from_input_file(input_path):
        postag = instance.target.split(".")[-1]
        targets_per_pos[postag].add(instance.target)

    for pos in targets_per_pos:
        print(f"Targets per {pos}: {len(targets_per_pos[pos])}")


def get_senses(input_path: str, gold_path: str):
    key_to_sense = {}
    for line in open(gold_path):
        key, *sense = line.strip().split()
        key_to_sense[key] = sense

    all_senses = set()
    senses_per_pos = defaultdict(set)
    senses_per_target = defaultdict(set)
    targets_per_pos = defaultdict(list)

    for instance in read_from_input_file(input_path):
        all_senses.update(key_to_sense[instance.instance_id])
        postag = instance.target.split(".")[-1]
        senses_per_pos[postag].update(key_to_sense[instance.instance_id])
        senses_per_target[instance.target].update(key_to_sense[instance.instance_id])

    for target in senses_per_target:
        postag = target.split(".")[-1]
        targets_per_pos[postag].append(senses_per_target[target])

    print(f"Tot senses: {len(all_senses)}")
    for pos in senses_per_pos:
        print(f"Senses per {pos}: {len(senses_per_pos[pos])}")

    print(
        f"avg senses "
        f"per target: {sum([len(senses_per_target[t]) for t in senses_per_target]) / len(senses_per_target)}")

    for pos in targets_per_pos:
        print(
            f"avg senses "
            f"per {pos}: {sum([len(list_senses) for list_senses in targets_per_pos[pos]]) / len(targets_per_pos[pos])}")


def recover_original_datasets(input_paths: List[str], suffixes: List[str], input_gold: List[str], merged_path: str,
                              new_output_path: str, new_gold_path: str):
    first = {instance.sentence: instance for instance in read_from_input_file(input_paths[0])}
    first_gold = {line.strip().split()[0]: line.strip().split()[1:] for line in open(input_gold[0])}
    second_gold = {line.strip().split()[0]: line.strip().split()[1:] for line in open(input_gold[1])}

    updated_gold = {}

    with open(new_output_path, 'w') as out:
        for instance in read_from_input_file(merged_path):
            if instance.sentence in first:
                new_instance_id = f"{suffixes[0]}.{instance.instance_id}"
                updated_gold[new_instance_id] = first_gold[instance.instance_id]

            else:
                new_instance_id = f"{suffixes[1]}.{instance.instance_id}"
                updated_gold[new_instance_id] = second_gold[instance.instance_id]

            instance.instance_id = new_instance_id
            out.write(repr(instance) + '\n')

    with open(new_gold_path, 'w') as oug:
        for instance, gold in updated_gold.items():
            str_gold = " ".join(gold)
            oug.write(f"{instance} {str_gold}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", required=True)
    # parser.add_argument("--gold_path", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    get_substitutes(args.input_tsv)
    print()
    get_target_words(args.input_tsv)
    print()
    # get_senses(args.input_tsv, args.gold_path)
