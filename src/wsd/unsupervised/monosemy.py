import argparse
import os.path

from nltk.corpus import wordnet as wn

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--substitutes_folder', required=False, default="/media/ssd1/caterina/genesis_checkpoints/"
                                                                        "bart_313_pt_semcor_0.7_drop_0.1_enc_lyd_0.6_dec_lyd_0.2/"
                                                                        "beams_15_return_15/wsd_output_files/substitutes/")

    parser.add_argument('--wsd_framework_folder', required=False, default="/home/caterina/PycharmProjects/wsd/data/"
                                                                          "WSD_Evaluation_Framework/Evaluation_Datasets/")
    return parser.parse_args()


def map_posnum_to_upos(pos_num: str) -> str:
    if pos_num == "1":
        return "n"
    elif pos_num == "2":
        return "v"
    elif pos_num == "3":
        return "a"
    elif pos_num == "5":
        return "s"
    else:
        return "r"


if __name__ == '__main__':
    args = parse_args()

    input_substitutes = os.path.join(args.substitutes_folder, f"{args.dataset}.substitutes.txt")
    input_keys = os.path.join(args.wsd_framework_folder, args.dataset, f"{args.dataset}.gold.key.txt")

    key_to_postag = {}
    for line in open(input_keys):
        key, *gold_senses = line.strip().split()
        pos_num = gold_senses[0].split('%')[1].split(':')[0]
        key_to_postag[key] = map_posnum_to_upos(pos_num)

    key_to_lemma = {}
    for line in open(input_keys):
        key, *gold_senses = line.strip().split()
        lemma = gold_senses[0].split('%')[0]
        key_to_lemma[key] = lemma

    key_to_substitutes = {}
    for line in open(input_substitutes):
        key, *substitutes = line.strip().split()
        key_to_substitutes[key] = substitutes

    key_to_prediction = {}

    orig_monosemous, incremental_monosemy = 0, 0

    for key in key_to_lemma:
        postag = key_to_postag[key]
        synsets = wn.synsets(key_to_lemma[key], pos=postag)
        if len(synsets) == 1:
            orig_monosemous += 1

        substitutes = key_to_substitutes[key]
        substitutes_synsets = set()

        for sub in substitutes:
            sub_synsets = wn.synsets(sub, postag)
            substitutes_synsets.update([s for s in sub_synsets if s in synsets])

        if len(substitutes_synsets) == 1 and len(synsets) > 1:
            incremental_monosemy += 1
            #key_to_prediction[key] =
            all_lemmas = list(substitutes_synsets)[0].lemmas()
            corresponding_sensekeys = [x.key() for x in all_lemmas if x.name() == key_to_lemma[key]]
            key_to_prediction[key] = corresponding_sensekeys

    print(f"instances originally monosemous: {orig_monosemous}")
    print(f"incremental monosemous instances: {incremental_monosemy}")
    print(f"total instances: {len(key_to_lemma)}")
