import argparse
import os
import pickle
import random
import subprocess

import numpy as np

from src.metric.new_metric import sort_with_metric
from src.utils import read_from_input_file, convert_to_lst_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--scoring_path", required=False, default="scoring_scripts")
    parser.add_argument("--candidates_path", required=False, default="data/lst_candidates.tsv")
    parser.add_argument("--gold_path", required=False, default="scoring_scripts/lst_gold.txt")
    parser.add_argument("--threshold", type=float, required=False, default=0.0)
    parser.add_argument("--random", action="store_true")

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    # sorted_output = sort_with_metric(args.candidates_path, model_name="bert-base-cased", batch_size=250)
    #
    # pickle.dump(sorted_output, open("data/lst_metric_candidates.pkl", "wb"))

    sorted_output = pickle.load(open("data/lst_metric_candidates.pkl", "rb"))
    with open(args.output_path, 'w') as out:
        for instance_id in sorted_output:
            lexeme, instance_num = instance_id.split()
            *target, pos = lexeme.split(".")
            lexeme = convert_to_lst_target(lexeme)
            if args.random:
                random.shuffle(sorted_output[instance_id])
            output_str = "\t".join([f"{word} {np.round(score, 3)}"
                                    for word, score in sorted_output[instance_id] if
                                    score > args.threshold])
            out.write(f"RESULT\t{lexeme} {instance_num}\t{output_str}\n")

    gap_script_abs_path = os.path.abspath(os.path.join(args.scoring_path, 'lst_gap.py'))
    output_gap_scores_path = args.output_path.replace(".tsv", ".scores.tsv")

    subprocess.check_output(f'python {gap_script_abs_path} {args.gold_path} {args.output_path} '
                            f'{output_gap_scores_path} no-mwe', shell=True)

    lines = open(output_gap_scores_path).readlines()[-5:]
    for l in lines:
        print('\t'.join(l.strip().split('\t')))
