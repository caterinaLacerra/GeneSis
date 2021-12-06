import argparse
import json
import os
from typing import List, Optional

import lemminflect
import stanza
import tqdm

from src.cross_lingual.utils import TranslationInstance, load_instances_from_lexsub_line
from src.wsd.utils.utils import file_len, multipos_to_pos


# produce one-line entry for each (target, substitute) pair in the dataset
def write_decoupled_input(input_path: str, output_path: str) -> List[TranslationInstance]:

    tot_instances = file_len(input_path)
    tot_substitutes = 0

    substitutes_per_pos = {}
    total_per_pos = {}

    all_instances = []
    with open(output_path, 'w') as out:
         for line in tqdm.tqdm(open(input_path), total=tot_instances):
            instances = load_instances_from_lexsub_line(line)
            all_instances.extend(instances)
            tot_substitutes += len(instances) - 1

            target_pos = instances[0].pos
            if target_pos not in substitutes_per_pos:
                substitutes_per_pos[target_pos] = 0
            if target_pos not in total_per_pos:
                total_per_pos[target_pos] = 0

            substitutes_per_pos[target_pos] += len(instances) - 1
            total_per_pos[target_pos] += 1

            out.write("\n".join([repr(instance) for instance in instances]) + "\n")

    print(f"Average substitutes per word: {tot_substitutes/tot_instances}")
    for postag in substitutes_per_pos:
        print(f"Avg substitutes per {postag}: {substitutes_per_pos[postag]/total_per_pos[postag]}")

    return all_instances


def postag_batch(batch_instances: List[TranslationInstance], pipeline: stanza.Pipeline):

    batch_sentences = "\n\n".join([x.original_sentence for x in batch_instances])
    doc = pipeline(batch_sentences)

    for i, sent in enumerate(doc.sentences):
        batch_instances[i].postag_info = [(word.text, word.upos, word.xpos) for word in sent.words]


def inflect_input(input_instances: List[TranslationInstance], pipeline: stanza.Pipeline,
                  batch_size: int, inflections_path: str):

    # running postag on original sentences only (excluded adv)
    to_postag = [(idx, instance) for idx, instance in enumerate(input_instances)
                 if not instance.is_substitute and instance.pos != "ADV"]

    with tqdm.tqdm(total=len(to_postag)) as pbar, open(inflections_path, "w") as out:

        for i in range(0, len(to_postag), batch_size):

            sentences_batch = to_postag[i: i + batch_size]

            # postag a batch of original sentences
            postag_batch([x[1] for x in sentences_batch], pipeline)

            # extract corresponding sentences with substitutes to inflect
            associated_instances = [[inst for inst in input_instances if inst.instance_id == f"subst.{sentence.instance_id}"]
                                    for _, sentence in sentences_batch]

            # extract target tokens with corresponding info
            relevant_tokens = [[sentence.postag_info[x] for x in sentence.target_idx] for _, sentence in sentences_batch]

            # associate the right inflection to the input sentence
            for j, tokens in enumerate(relevant_tokens):

                # single target word
                if len(tokens) == 1:
                    xpos = tokens[0][2]
                    if xpos is None:
                        continue

                    for instance in associated_instances[j]:
                        inflected_target = lemminflect.getInflection(instance.lemma, xpos)
                        if len(inflected_target) > 0:
                            inflected_target = inflected_target[0]
                            inflected_sentence = " ".join(
                                instance.original_sentence.split()[:instance.target_idx[0]] + \
                                [inflected_target] + \
                                instance.original_sentence.split()[instance.target_idx[-1] + 1:]
                            )
                            instance.inflected_sentence = inflected_sentence
                        out.write(instance.toJSON() + '\n')

                # merge pos for multiwords
                else:
                    merged_pos = multipos_to_pos([x[1] for x in tokens])
                    xpos = set([t[2] for t in tokens if t[1] == merged_pos])

                    if xpos is None:
                        continue

                    if len(xpos) > 0:
                        for instance in associated_instances[j]:
                            inflected_target = lemminflect.getInflection(instance.lemma, list(xpos)[0])
                            if len(inflected_target) > 0:
                                inflected_target = inflected_target[0]
                                inflected_sentence = " ".join(instance.original_sentence.split()[:instance.target_idx[0]] + \
                                                          [inflected_target] +
                                                          instance.original_sentence.split()[instance.target_idx[-1] + 1:])
                                instance.inflected_sentence = inflected_sentence
                                out.write(instance.toJSON() + '\n')

            pbar.update(batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--batch_size', type=int, default=500)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # produce one sentence for each en substitute (lemmas)
    decoupling_path = os.path.join(args.output_folder,
                                   os.path.split(args.input_path)[-1].replace(".tsv", "_decoupling.tsv"))
    instances = write_decoupled_input(args.input_path, decoupling_path)

    pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos',
                               tokenize_pretokenized=True,
                               tokenize_no_ssplit=True)

    inflections_path = os.path.join(args.output_folder,
                                    os.path.split(args.input_path)[-1].replace(".tsv", "_inflected.jsonl"))
    print(f"Writing to {inflections_path}")

    # inflect en substitutes
    inflect_input(instances, pipeline, args.batch_size, inflections_path)


