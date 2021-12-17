import argparse
from typing import Union, List

from src.utils import read_from_input_file


def avg_substitutes_and_targets(input_path: str):
    pos_dict = {}
    target_dict = {}
    total_instances = 0
    total_substitutes = 0
    targets = set()
    for instance in read_from_input_file(input_path):
        targets.add(instance.target)
        pos_tag = instance.target.split(".")[-1]
        if pos_tag not in pos_dict:
            pos_dict[pos_tag] = []
            target_dict[pos_tag] = set()
        target_dict[pos_tag].add(instance.target)
        pos_dict[pos_tag].append(list(instance.gold))
        total_substitutes += len(instance.gold)
        total_instances += 1

    print(f"Avg substitutes per POS:")
    for pos in pos_dict:
        print(f"{pos}: {sum([len(x) for x in pos_dict[pos]])/len(pos_dict[pos])}")

    print(f"Avg substitutes: {total_substitutes/total_instances}")
    print()
    print(f"Targets per POS:")
    for pos in target_dict:
        print(f"{pos}: {len(target_dict[pos])}")
    print(f"Tot targets: {len(targets)}")


def get_senses(input_path: str, gold_path: Union[str, List[str]]):

    id_to_synset = {line.strip().split()[0]: line.strip().split()[1]
                    for line in open(gold_path)}

    senses_per_pos = {}
    senses = set()

    for instance in read_from_input_file(input_path):
        senses.add(id_to_synset[instance.instance_id])

        pos = instance.target.split(".")[-1]
        if pos not in senses_per_pos:
            senses_per_pos[pos] = set()
        senses_per_pos[pos].add(id_to_synset[instance.instance_id])
    print()
    print(f"Senses per POS:")
    for pos in senses_per_pos:
        print(f"{pos}: {len(senses_per_pos[pos])}")
    print(f"Tot senses: {len(senses)}")


def merge_paths(input_paths: List[str], suffixes: List[str], gold_paths: List[str], merged_path: str, output_path: str, output_gold: str):

    ds_to_instances = {}
    ds_to_gold = {}
    for i, (path, gold_path) in enumerate(zip(input_paths, gold_paths)):
        ds = suffixes[i]
        ds_to_instances[ds] = {}
        ds_to_gold[ds] = {line.strip().split()[0]: line.strip().split()[1] for line in open(gold_path)}

        for instance in read_from_input_file(path):
            ds_to_instances[ds][instance.instance_id] = instance

    with open(output_path, 'w') as out, open(output_gold, 'w') as out_gold:
        for instance in read_from_input_file(merged_path):
            for ds in ds_to_instances:
                if instance.instance_id in ds_to_instances[ds] and ds_to_instances[ds][instance.instance_id].sentence == instance.sentence:
                    new_instance_id = f"{ds}.{instance.instance_id}"
                    out_gold.write(f"{new_instance_id} {ds_to_gold[ds][instance.instance_id]}\n")
                    instance.instance_id = new_instance_id
                    out.write(repr(instance) + '\n')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gold", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # semcor_path = "data/translated_datasets/multimirror.it.clean_targets_instances_train.tsv"
    # wngt_path = "data/translated_datasets/multimirror.wngt.it.clean_targets_train.tsv"
    #
    # semcor_gold = "/root/multimirror/semcor-translated-it-ours-loosebn/dataset/dataset.gold.key.txt"
    # wngt_gold = "/root/multimirror/wngt-translated-it-ours-loosebn/dataset/dataset.gold.key.txt"
    #
    # merged_path = "data/translated_datasets/multimirror.semcor_wngt.it.clean_targets_train.tsv"
    #
    # input_paths = [semcor_path, wngt_path]
    # gold_paths = [semcor_gold, wngt_gold]
    # suffixes = ["semcor", "wngt"]
    # output_path = "data/translated_datasets/semcor_wngt.it.clean_instances_train.tsv"
    # out_gold = "/root/multimirror/merged_semcor_wngt.gold.txt"

    # merge_paths(input_paths, suffixes, gold_paths, merged_path, output_path, out_gold)
    avg_substitutes_and_targets(args.input)
    get_senses(args.input, args.gold)