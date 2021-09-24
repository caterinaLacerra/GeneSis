import argparse
import collections
import json
import os
import random
import subprocess
from typing import Dict, List, Optional, Any, Tuple

import lemminflect
import numpy as np
import stanza
import torch
import tqdm
import transformers
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity

from src.task_evaluation import get_generated_substitutes
from src.utils import yield_batch, flatten, embed_sentences, get_target_index_list, read_from_input_file
from src.vocabulary_definition.create_vocab_from_wn import get_related_lemmas
from src.wordnet_utils import synset_from_sensekey


def map_xpos(target_xpos: str, available_xpos: List[str]) -> Optional[str]:

    if target_xpos in available_xpos:
        return target_xpos

    else:
        if target_xpos == 'VBD':
            if 'VBN' in available_xpos:
                return 'VBN'
            return None

        elif target_xpos == 'VBN':
            if 'VBD' in available_xpos:
                return 'VBD'
            return None


def get_input_substitutes_ranked_by_frequency(input_path: str, top_k: int,
                                              pipeline: stanza.Pipeline, gold_dict: Dict[str, List[str]]) -> Dict[Any, list]:


    sentence_to_substitutes = {}

    for batch in yield_batch(input_path, separator='#########\n'):

        lexeme, instance_id, *target_index = batch[0].strip().split()
        target_index = ' '.join(target_index)
        input_sentence = batch[1].strip()
        substitutes = flatten([line.strip().split(', ') for line in batch[2:]])
        clean_subst = [word for word, count in collections.Counter(substitutes).most_common(top_k)]
        target_index = get_target_index_list(target_index)

        sensekey = gold_dict[instance_id][0]
        gloss = synset_from_sensekey(sensekey).definition()

        associated_span = " ".join(input_sentence.split()[target_index[0]:target_index[-1] + 1])
        postagged = [(w.text.lower(), w.xpos, w.upos) for s in pipeline(input_sentence).sentences for w in s.words]

        inflected_substitutes = []
        if len(target_index) == 1:
            word, xpos, upos = postagged[target_index[0]]

            if word != input_sentence.lower().split()[target_index[0]]:
                remap_indexes = [i for i, (w, x, u) in enumerate(postagged) if w == word]
                closest_idx = remap_indexes[np.argmin(abs(i - target_index[0]) for i in remap_indexes)]
                word, xpos, upos = postagged[closest_idx]

            for substitute in clean_subst:
                inflect_dict = lemminflect.getAllInflections(substitute, upos)
                mapped_xpos = map_xpos(xpos, available_xpos=list(inflect_dict.keys()))
                if mapped_xpos:
                    inflected_substitutes.append(inflect_dict[mapped_xpos][0])

                else:
                    inflected_substitutes.append(substitute)

        else:
            for substitute in clean_subst:
                inflected_substitutes.append(substitute)

        if input_sentence not in sentence_to_substitutes:
            sentence_to_substitutes[input_sentence] = []

        sentence_to_substitutes[input_sentence].append({'gloss': gloss,
                                                        'instance_id': instance_id,
                                                        'target_index': target_index,
                                                        'substitutes': inflected_substitutes,
                                                        'target_span': associated_span,
                                                        'lexeme': lexeme})

    return sentence_to_substitutes

def produce_modified_sentences(input_path: str, substitutes_dict: Dict[str, List[str]]) -> Dict[str, Dict]:

    input_dict = {}
    for instance in read_from_input_file(input_path):
        input_dict[instance.instance_id] = {}
        if instance.instance_id in substitutes_dict:
            substitutes = substitutes_dict[instance.instance_id]
            target_idx = instance.target_idx
            start_idx = target_idx[0]
            end_idx = target_idx[-1]

            modified_sentences, modified_indexes = [], []
            input_words = instance.sentence.split()
            modified_sentences.append(instance.sentence)
            modified_indexes.append(target_idx)

            for substitute in substitutes:
                modified_sentences.append(" ".join(input_words[:start_idx] + substitute.split() +
                                                   input_words[end_idx + 1:]))
                modified_indexes.append([x for x in range(start_idx, start_idx + len(substitute.split()))])

            input_dict[instance.instance_id]["new_sentences"] = modified_sentences
            input_dict[instance.instance_id]["new_indexes"] = modified_indexes
            input_dict[instance.instance_id]["lexeme"] = instance.target
            input_dict[instance.instance_id]["sentence"] = instance.sentence

    return input_dict


def embed(input_dict: Dict[str, Dict], embedder: transformers.AutoModel.from_pretrained,
          tokenizer: transformers.AutoTokenizer.from_pretrained, hs: int, device: str,
          layer_indexes: List[int], batch_size: int = 100):

    stacked_input_sentences, stacked_target_indexes = [], []
    indexes_mapping = {}

    idx = 0
    for instance_id in input_dict:
        indexes_mapping[instance_id] = {'original': -1, 'substitutes': [],
                                          'sentence': input_dict[instance_id]["sentence"],
                                          'lexeme': input_dict[instance_id]['lexeme']}

        for j, sent in enumerate(input_dict[instance_id]["new_sentences"]):
            if j == 0:
                indexes_mapping[instance_id]["original"] = idx
            else:
                indexes_mapping[instance_id]["substitutes"].append(idx)

            stacked_input_sentences.append(sent)
            stacked_target_indexes.append(input_dict[instance_id]["new_indexes"][j])
            idx += 1

    input_matrix_embed = torch.zeros((len(stacked_input_sentences), hs), device=device)

    for i in tqdm.tqdm(range(0, len(stacked_input_sentences), batch_size), desc='Embedding sentences'):
        batch = stacked_input_sentences[i: i + batch_size]
        batch_indexes = stacked_target_indexes[i: i + batch_size]
        vecs = embed_sentences(embedder, tokenizer, batch_indexes, batch, device, hidden_size=hs,
                               layer_indexes=layer_indexes, sum=True)

        input_matrix_embed[i: i + batch_size] = vecs

    return indexes_mapping, input_matrix_embed


def load_ares(input_path: str) -> Tuple[Dict[Any, int], Optional[np.ndarray]]:

    print('Loading ARES')
    matrix = None
    mapping = {}

    for i, line in enumerate(open(input_path)):
        if i > 0:
            break
        x, y = line.strip().split()
        matrix = np.zeros((int(x), int(y)))

    for i, line in tqdm.tqdm(enumerate(open(input_path))):
        if i > 0:
            sensekey, *vec = line.strip().split()
            mapping[sensekey] = i - 1
            matrix[i-1] = np.asarray([float(x) for x in vec])

    return mapping, matrix

def is_in_wordnet(word: str) -> bool:
    synsets = wn.synsets(word.replace(' ', '_'))
    return len(synsets) > 0

def save_substitute_vectors(output_folder: str, substitutes_dict: Dict[str, List[str]], input_path: str,
                            model_name: str, device: str, dataset_name: str):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sentence_to_data = produce_modified_sentences(input_path, substitutes_dict)

    with open(os.path.join(output_folder, f'{dataset_name}.json'), 'w', encoding="utf-8") as out:
        json.dump(sentence_to_data, out, indent=2, ensure_ascii=False)

    embedder = transformers.AutoModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    hs = transformers.AutoConfig.from_pretrained(model_name).hidden_size
    layer_indexes = [20, 23]

    vecs_folder = os.path.join(output_folder, 'vectors')
    mappings_folder = os.path.join(output_folder, 'mappings')

    if not os.path.exists(vecs_folder):
        os.makedirs(vecs_folder)

    if not os.path.exists(mappings_folder):
        os.makedirs(mappings_folder)

    indexes_mapping, matrix = embed(sentence_to_data, embedder, tokenizer, hs, device, layer_indexes)
    matrix = matrix.cpu().numpy()

    with open(os.path.join(vecs_folder, f'{dataset_name}'), 'wb') as out:
        np.save(out, matrix)

    with open(os.path.join(mappings_folder, f'{dataset_name}.json'), 'w', encoding='utf-8') as out:
        json.dump(indexes_mapping, out, indent=2, ensure_ascii=False)


def convert_to_sensekey(lexeme: str) -> List[str]:
    *lemma, pos = lexeme.split('.')
    lemma = '.'.join(lemma)
    pos_map = {'n': ['1'], 'v': ['2'], 'r': ['4'], 'a': ['3', '5'],
               'NOUN': ['1'], 'VERB': ['2'], 'ADV': ['4'], 'ADJ': ['3', '5']}
    sensekeys = []
    for p in pos_map[pos]:
        sensekeys.append(f'{lemma}%{p}:')
    return sensekeys

def compute_baseline(original_vector: np.array, sense_vectors: np.matrix, top_k: int,
                    possible_senses_indexes: np.matrix, index_to_sensekey: Dict[int, str]) -> List[Tuple[str, float]]:

    if original_vector.shape[0] == 1 and sense_vectors.shape[0] == 1:
        cos_sim =  cosine_similarity(original_vector, sense_vectors)
    else:
        cos_sim =  cosine_similarity(original_vector, sense_vectors).squeeze()

    sorted_idx = cos_sim.squeeze().argsort()[-top_k:][::-1]

    # get indexes and scores
    most_similar_indexes = [possible_senses_indexes[x] for x in sorted_idx]
    most_similar_scores = [cos_sim[x].item() if type(cos_sim[x]) == np.ndarray else cos_sim[x] for x in sorted_idx]

    # convert the indexes to sensekeys
    top_senses = [(index_to_sensekey[i], j) for i, j in zip(most_similar_indexes, most_similar_scores)]

    return top_senses


def weighted_majority_voting(original_vector: np.array, substitutes_vectors: List[np.array], sense_vectors: np.matrix, top_k: int,
                    possible_senses_indexes: np.matrix, index_to_sensekey: Dict[int, str]) -> List[Tuple[str, float]]:

    if substitutes_vectors == []:
        if original_vector.shape[0] == 1 and sense_vectors.shape[0] == 1:
            cos_sim =  cosine_similarity(original_vector, sense_vectors)
        else:
            cos_sim =  cosine_similarity(original_vector, sense_vectors).squeeze()

        sorted_idx = cos_sim.squeeze().argsort()[-top_k:][::-1]

        # get indexes and scores
        most_similar_indexes = [possible_senses_indexes[x] for x in sorted_idx]
        most_similar_scores = [cos_sim[x].item() if type(cos_sim[x]) == np.ndarray else cos_sim[x] for x in sorted_idx]

        # convert the indexes to sensekeys
        top_senses = [(index_to_sensekey[i], j) for i, j in zip(most_similar_indexes, most_similar_scores)]

    else:
        stacked_vectors = np.stack([original_vector] + substitutes_vectors)

        if sense_vectors.shape[0] > 1:
            cos_sim = cosine_similarity(stacked_vectors.squeeze(), sense_vectors)
            idx_to_scores = {}
            closest_senses = []

            for i, subst_row in enumerate(cos_sim):
                sorted_idx = cos_sim[i].argsort()[-1:][::-1]
                # for each substitute (and original word) keep track of closest sense --> (idx, score)
                closest_senses.append((sorted_idx[0], cos_sim[i][sorted_idx[0]]))
                if sorted_idx[0] not in idx_to_scores:
                    idx_to_scores[sorted_idx[0]] = []
                idx_to_scores[sorted_idx[0]].append(cos_sim[i][sorted_idx[0]])

            # keep most common indexes --> (idx, count)
            most_common = collections.Counter([x[0] for x in closest_senses]).most_common(min(top_k, len(closest_senses)))

            # sort senses by their avg score
            top_senses = sorted([(index_to_sensekey[possible_senses_indexes[idx]],
                                  np.mean(idx_to_scores[idx])) for idx, count in most_common],
                                key=lambda x:x[1], reverse=True)

        # monosemous
        else:
            top_senses = [(index_to_sensekey[i], 1) for i in possible_senses_indexes]

    return top_senses



def majority_voting(original_vector: np.array, substitutes_vectors: List[np.array], sense_vectors: np.matrix, top_k: int,
                    possible_senses_indexes: np.matrix, index_to_sensekey: Dict[int, str]) -> List[Tuple[str, float]]:

    if substitutes_vectors == []:
        if original_vector.shape[0] == 1 and sense_vectors.shape[0] == 1:
            cos_sim =  cosine_similarity(original_vector, sense_vectors)
        else:
            cos_sim =  cosine_similarity(original_vector, sense_vectors).squeeze()

        sorted_idx = cos_sim.squeeze().argsort()[-top_k:][::-1]

        # get indexes and scores
        most_similar_indexes = [possible_senses_indexes[x] for x in sorted_idx]
        most_similar_scores = [cos_sim[x].item() if type(cos_sim[x]) == np.ndarray else cos_sim[x] for x in sorted_idx]

        # convert the indexes to sensekeys
        top_senses = [(index_to_sensekey[i], j) for i, j in zip(most_similar_indexes, most_similar_scores)]

    else:
        stacked_vectors = np.stack([original_vector] + substitutes_vectors)

        if sense_vectors.shape[0] > 1:
            cos_sim = cosine_similarity(stacked_vectors.squeeze(), sense_vectors)
            idx_to_scores = {}
            closest_senses = []

            for i, subst_row in enumerate(cos_sim):
                sorted_idx = cos_sim[i].argsort()[-1:][::-1]
                # for each substitute (and original word) keep track of closest sense --> (idx, score)
                closest_senses.append((sorted_idx[0], cos_sim[i][sorted_idx[0]]))
                if sorted_idx[0] not in idx_to_scores:
                    idx_to_scores[sorted_idx[0]] = []
                idx_to_scores[sorted_idx[0]].append(cos_sim[i][sorted_idx[0]])

            # keep most common indexes --> (idx, count)
            most_common = collections.Counter([x[0] for x in closest_senses]).most_common(min(top_k, len(closest_senses)))

            top_senses = [(index_to_sensekey[possible_senses_indexes[idx]],
                           np.mean(idx_to_scores[idx])) for idx, count in most_common]

        # monosemous
        else:
            top_senses = [(index_to_sensekey[i], 1) for i in possible_senses_indexes]

    return top_senses

#todo: homogeneous inputs to get_closest_senses and majority_voting
def get_closest_sense(target_vector: np.array, sense_vectors: np.matrix, top_k: int,
                      possible_senses_indexes: np.matrix, index_to_sensekey: Dict[int, str]) -> List[Tuple[str, float]]:

    if sense_vectors.shape[0] > 1:
        cos_sim = cosine_similarity(target_vector, sense_vectors).squeeze()
        sorted_idx = cos_sim.squeeze().argsort()[-top_k:][::-1]

        # get indexes and scores
        most_similar_indexes = [possible_senses_indexes[x] for x in sorted_idx]
        most_similar_scores = [cos_sim.squeeze()[x] for x in sorted_idx]

        # convert the indexes to sensekeys
        top_senses = [(index_to_sensekey[i], j) for i, j in zip(most_similar_indexes, most_similar_scores)]

    # monosemous
    else:
        top_senses = [(index_to_sensekey[i], 1) for i in possible_senses_indexes]

    return top_senses


def disambiguate(data_folder: str, output_dir: str, dataset_name: str, top_k: int, ares_mapping: Dict,
                 ares_vecs: np.ndarray, gold_dict: Dict[str, Any], substitutes_dict: Dict[str, List[str]],
                 glosses_dict: Dict[str, str], majority: bool, weighted: bool, baseline: bool):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if majority:
        print(f"Using majority voting")
    if weighted:
        print(f"Using weighted majority voting")
    if baseline:
        print(f"Computing baseline")

    vec_mapping_path = os.path.join(data_folder, 'mappings', f'{dataset_name}.json')
    vecs_matrix_path = os.path.join(data_folder, 'vectors', dataset_name)

    vecs_matrix = np.load(vecs_matrix_path)

    with open(vec_mapping_path) as inp:
        instance_to_matrix_idx = json.load(inp)

    index_to_sensekey = {index: sensekey for sensekey, index in ares_mapping.items()}

    with open(os.path.join(output_dir, f'{dataset_name}_predictions.txt'), 'w') as output,\
            open(os.path.join(output_dir, f'{dataset_name}_keys.txt'), 'w') as out_k:

        for instance in tqdm.tqdm(instance_to_matrix_idx):
            lexeme = instance_to_matrix_idx[instance]['lexeme']
            converted_lexeme_keys = convert_to_sensekey(lexeme)
            possible_sensekeys = [k for k in ares_mapping if any(k.startswith(z) for z in converted_lexeme_keys)]
            possible_indexes = [ares_mapping[k] for k in possible_sensekeys]

            original_vec_index = instance_to_matrix_idx[instance]['original']
            substitutes_vec_indexes = instance_to_matrix_idx[instance]['substitutes']

            original_vec = np.expand_dims(np.concatenate((vecs_matrix[original_vec_index],
                                                          vecs_matrix[original_vec_index])),
                                          0)


            possible_sense_vecs =  np.stack([ares_vecs[i] for i in possible_indexes])

            if baseline:
                top_senses_baseline = compute_baseline(original_vec, possible_sense_vecs, top_k,
                                                      possible_indexes, index_to_sensekey)

            elif majority:
                substitutes_vectors = [np.expand_dims(np.concatenate((vecs_matrix[x], vecs_matrix[x])), 0)
                                       for x in substitutes_vec_indexes]

                top_senses = majority_voting(original_vec, substitutes_vectors, possible_sense_vecs, top_k,
                                             possible_indexes, index_to_sensekey)

                top_senses_baseline = majority_voting(original_vec, [], possible_sense_vecs, top_k,
                                             possible_indexes, index_to_sensekey)

            elif weighted:
                substitutes_vectors = [np.expand_dims(np.concatenate((vecs_matrix[x], vecs_matrix[x])), 0)
                                       for x in substitutes_vec_indexes]

                top_senses = weighted_majority_voting(original_vec, substitutes_vectors, possible_sense_vecs, top_k,
                                                      possible_indexes, index_to_sensekey)

                top_senses_baseline = weighted_majority_voting(original_vec, [], possible_sense_vecs, top_k,
                                                               possible_indexes, index_to_sensekey)

            # CENTROID # todo: separate method
            else:
                substitutes_vecs = [original_vec]

                for x in substitutes_vec_indexes:
                    substitutes_vecs.append(np.expand_dims(np.concatenate((vecs_matrix[x], vecs_matrix[x])), 0))

                # centroid of substitutes and original vec
                target_vec = np.mean(substitutes_vecs, axis=0)

                top_senses = get_closest_sense(target_vec, possible_sense_vecs, top_k, possible_indexes, index_to_sensekey)
                top_senses_baseline = get_closest_sense(original_vec, possible_sense_vecs, top_k, possible_indexes, index_to_sensekey)

            if not baseline:
                if top_senses[0][0] not in gold_dict[instance]:
                    output.write('### ')

                output.write(f'{instance}\t{lexeme}\t{instance_to_matrix_idx[instance]["sentence"]}\n')
                gloss = glosses_dict[instance]
                output.write(f'gold gloss: {gloss}\n')
                generated_substitutes = substitutes_dict[instance]
                output.write(f'generated substitutes: {"; ".join(generated_substitutes)}\n')


                for i, (sense, score) in enumerate(top_senses):
                    if i == 0:
                        output.write(f'\n{sense}\t{score}\t{synset_from_sensekey(sense).definition()}\t(pred with substitutes)\n')
                    else:
                        output.write(f'{sense}\t{score}\t{synset_from_sensekey(sense).definition()}\n')

                output.write('---------------\n')
                for i, (sense, score) in enumerate(top_senses_baseline):
                    if i == 0:
                        output.write(f'\n{sense}\t{score}\t{synset_from_sensekey(sense).definition()}\t(pred without substitutes)\n')
                    else:
                        output.write(f'{sense}\t{score}\t{synset_from_sensekey(sense).definition()}\n')

                output.write('==============\n')
                out_k.write(f'{instance} {top_senses[0][0]}\n')

            else:
                if top_senses_baseline[0][0] not in gold_dict[instance]:
                    output.write('### ')

                output.write(f'{instance}\t{lexeme}\t{instance_to_matrix_idx[instance]["sentence"]}\n')
                gloss = glosses_dict[instance]
                output.write(f'gold gloss: {gloss}\n')

                for i, (sense, score) in enumerate(top_senses_baseline):
                    output.write(f'{sense}\t{score}\t{synset_from_sensekey(sense).definition()}\n')

                out_k.write(f'{instance} {top_senses_baseline[0][0]}\n')


def evaluate(dataset: str, input_folder: str, gold_folder: str):
    gold = os.path.join(gold_folder, dataset, f'{dataset}.gold.key.txt')
    input_path = os.path.join(input_folder, f'{dataset}_keys.txt')

    gold_dict = {line.strip().split()[0]: line.strip().split()[1:] for line in open(gold)}
    input_dict = {line.strip().split()[0]: line.strip().split()[1:] for line in open(input_path)}

    correct = 0
    for key in gold_dict:
        if any(x in gold_dict[key] for x in input_dict[key]):
            correct += 1
    precision = correct/len(input_dict)
    recall = correct/len(gold_dict)
    print(f'F1: {np.round(((2*precision*recall) / (precision + recall)), 2)}')


def main(args: argparse.Namespace) -> None:

    ares_mapping, ares_vecs = load_ares(args.ares_path)
    device = 'cpu'

    datasets = ['semeval2007'] if args.dev else ['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015']

    for dataset_name in datasets:
        print(f'Dataset: {dataset_name}')

        input_path = os.path.join(args.input_folder, f'{dataset_name}_test.tsv')

        substitutes_path = os.path.join(args.substitutes_folder, f'wsd_output_{dataset_name}.txt')

        substitutes_dict, glosses_dict = {}, {}

        for batch in yield_batch(substitutes_path, separator="#########\n"):
            target, instance_id, *index = batch[0].strip().split()
            substitutes = [x for line in batch[2:-1] for x in line.strip().split(', ')]
            top_3 = collections.Counter(substitutes).most_common(args.top_substitutes)
            top_substitutes = [x[0] for x in top_3]
            substitutes_dict[instance_id] = top_substitutes

        gold_path = os.path.join(args.eval_framework_folder, dataset_name, f'{dataset_name}.gold.key.txt')
        gold_dict = {line.strip().split()[0]: line.strip().split()[1:] for line in open(gold_path)}

        for instance_id in gold_dict:
            gold_key = random.choice(gold_dict[instance_id])
            gloss = synset_from_sensekey(gold_key).definition()
            glosses_dict[instance_id] = gloss

        save_substitute_vectors(args.output_folder, substitutes_dict, input_path, args.model_name, device,
                                dataset_name)

        output_dir = os.path.join(args.output_folder, 'disambiguated_ares')

        disambiguate(args.output_folder, output_dir, dataset_name, top_k=5,
                     ares_mapping=ares_mapping, ares_vecs=ares_vecs, gold_dict=gold_dict,
                     substitutes_dict=substitutes_dict, glosses_dict=glosses_dict,
                     majority=args.majority, weighted=args.weighted, baseline=args.baseline)

        evaluate(dataset_name, output_dir, args.eval_framework_folder)


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--eval_framework_folder', default="WSD_Evaluation_Framework/Evaluation_Datasets/")
    args.add_argument('--substitutes_folder', default="/media/ssd1/caterina/genesis_checkpoints/"
                                                         "bart_313_pt_semcor_0.7_drop_0.1_enc_lyd_0.6_dec_lyd_0.2/"
                                                         "beams_15_return_15/wsd_output_files/")
    args.add_argument('--output_folder', required=True)
    args.add_argument('--input_folder', default="data/wsd")

    args.add_argument('--scorer_folder', default='experiments/wsd')

    args.add_argument('--model_name', default='bert-large-cased')
    args.add_argument('--ares_path', default='ares_embedding/ares_bert_large.txt')
    args.add_argument('--top_k', type=int, default=3)
    args.add_argument('--top_substitutes', type=int, default=1)
    args.add_argument('--majority', action="store_true", default=False)
    args.add_argument('--weighted', action="store_true", default=False)
    args.add_argument('--baseline', action="store_true", default=False)

    args.add_argument('--dev', action="store_true", default=False)
    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
