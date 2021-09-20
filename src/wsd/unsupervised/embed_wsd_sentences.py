import argparse
import collections
import json
import os
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

def produce_modified_sentences(input_path: str, substitutes_dict: Dict[str, List[str]]) -> Dict[Any, list]:


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

            input_dict[instance.sentence]["new_sentences"] = modified_sentences
            input_dict[instance.sentence]["new_indexes"] = modified_indexes

    return input_dict


def embed(input_dict: Dict[Any, list], embedder: transformers.AutoModel.from_pretrained,
          tokenizer: transformers.AutoTokenizer.from_pretrained, hs: int, device: str,
          layer_indexes: List[int], batch_size: int = 100):

    stacked_input_sentences, stacked_target_indexes = [], []
    indexes_mapping = {}

    idx = 0
    for sentence in input_dict:
        for instance_dict in input_dict[sentence]:
            indexes_mapping[instance_dict["instance_id"]] = {'original': -1, 'substitutes': [],
                                                             'sentence': sentence, 'lexeme': instance_dict['lexeme']}
            for j, sent in enumerate(instance_dict["new_sentences"]):
                if j == 0:
                    indexes_mapping[instance_dict["instance_id"]]["original"] = idx
                else:
                    indexes_mapping[instance_dict["instance_id"]]["substitutes"].append(idx)

                stacked_input_sentences.append(sent)
                stacked_target_indexes.append(instance_dict["new_indexes"][j])
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

def save_substitute_vectors(output_folder: str, substitutes_dict: Dict[str, List[str]], input_path: str, model_name: str, device: str,
                            dataset_name: str):

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


def disambiguate(data_folder: str, output_dir: str, dataset_name: str, top_k: int, ares_mapping: Dict,
                 ares_vecs: np.ndarray, gold_dict: Dict[str, Any]):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vec_mapping_path = os.path.join(data_folder, 'mappings', f'{dataset_name}.json')
    infos_mapping_path = os.path.join(data_folder, f'{dataset_name}.json')
    vecs_matrix_path = os.path.join(data_folder, 'vectors', dataset_name)

    vecs_matrix = np.load(vecs_matrix_path)

    with open(infos_mapping_path) as inp:
        infos_mapping = json.load(inp)

    with open(vec_mapping_path) as inp:
        instance_to_matrix_idx = json.load(inp)

    index_to_sensekey = {index: sensekey for sensekey, index in ares_mapping.items()}

    with open(os.path.join(output_dir, f'{dataset_name}_predictions.txt'), 'w') as output,\
            open(os.path.join(output_dir, f'{dataset_name}_keys.txt'), 'w') as out_k:

        for instance in tqdm.tqdm(instance_to_matrix_idx):
            lexeme = instance_to_matrix_idx[instance]['lexeme']
            converted_lexeme_keys = convert_to_sensekey(lexeme)
            associated_sensekeys = [k for k in ares_mapping if any(k.startswith(z) for z in converted_lexeme_keys)]
            associated_indexes = [ares_mapping[k] for k in associated_sensekeys]

            original_vec_index = instance_to_matrix_idx[instance]['original']
            substitutes_vec_indexes = instance_to_matrix_idx[instance]['substitutes']

            if len(substitutes_vec_indexes) > 0:
                context_vec = np.expand_dims(np.concatenate((vecs_matrix[original_vec_index],
                                                         # vecs_matrix[substitutes_vec_indexes[0]]
                                                         np.mean([vecs_matrix[x] for x in substitutes_vec_indexes],
                                                                 axis=0),
                                                        )), 0)
            else:
                context_vec = np.expand_dims(np.concatenate((vecs_matrix[original_vec_index],
                                                             vecs_matrix[original_vec_index],
                                                             )), 0)

            compare_vecs =  np.stack([ares_vecs[i] for i in associated_indexes])

            if compare_vecs.shape[0] > 1:
                cos_sim = cosine_similarity(context_vec, compare_vecs).squeeze()
                sorted_idx = cos_sim.squeeze().argsort()[-top_k:][::-1]
                most_similar_indexes = [associated_indexes[x] for x in sorted_idx]
                most_similar_scores = [cos_sim.squeeze()[x] for x in sorted_idx]
                top_senses = [(index_to_sensekey[i], j)  for i, j in zip(most_similar_indexes, most_similar_scores)]

            else:
                top_senses = [(index_to_sensekey[i], 1) for i in associated_indexes]

            if top_senses[0][0] not in gold_dict[instance]:
                output.write('### ')

            output.write(f'{instance}\t{lexeme}\t{instance_to_matrix_idx[instance]["sentence"]}\n')
            sentencekey = instance_to_matrix_idx[instance]["sentence"]
            instances = infos_mapping[sentencekey]
            gloss = [inst['gloss'] for inst in instances if inst['instance_id'] == instance][0]
            output.write(f'gold gloss: {gloss}\n')
            generated_substitutes = [inst['substitutes'] for inst in instances if inst['instance_id'] == instance][0]
            output.write(f'generated substitutes: {"; ".join(generated_substitutes)}\n')
            
            for sense, score in top_senses:
                output.write(f'{sense}\t{score}\t{synset_from_sensekey(sense).definition()}\n')

            output.write('\n')
            out_k.write(f'{instance} {top_senses[0][0]}\n')


def evaluate(dataset: str, scorer_folder: str, input_folder: str, gold_folder: str):
    gold = os.path.join(gold_folder, dataset, f'{dataset}.gold.key.txt')
    input_path = os.path.join(input_folder, f'{dataset}_keys.txt')
    command = ['java', '-classpath', scorer_folder, 'Scorer', gold, input_path]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(result.stdout)

def main(args: argparse.Namespace) -> None:

    #ares_mapping, ares_vecs = load_ares(args.ares_path)
    device = 'cpu'

    for dataset_name in ['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015']:
        print(f'Dataset: {dataset_name}')

        input_path = os.path.join(args.input_folder, f'{dataset_name}_test.tsv')

        substitutes_paths = [os.path.join(args.eval_framework_folder,
                                        f'{dataset_name}_substitutes/{dataset_name}.substitutes_{x}.gold.key.txt')
                             for x in ["1st", "2nd"]]
        substitutes_dict = {}
        for path in substitutes_paths:
            for line in open(path):
                if len(line.split()) > 0:
                    key, *substitutes = line.strip().split()
                    if key not in substitutes_dict:
                        substitutes_dict[key] = []
                    substitutes_dict[key].extend(substitutes)

        gold_path = os.path.join(args.eval_framework_folder, dataset_name, f'{dataset_name}.gold.key.txt')

        save_substitute_vectors(args.output_folder, substitutes_dict, input_path, args.model_name, device,
                                 dataset_name)

        output_dir = os.path.join(args.output_folder, 'disambiguated_ares_centroid')

        gold_dict = {line.strip().split()[0]: line.strip().split()[1:] for line in open(gold_path)}

        disambiguate(args.output_folder, output_dir, dataset_name, top_k=5,
                     ares_mapping=ares_mapping, ares_vecs=ares_vecs, gold_dict=gold_dict)

        evaluate(dataset_name, args.scorer_folder, output_dir, args.gold_root_folder)

def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--eval_framework_folder', default="WSD_Evaluation_Framework/Evaluation_Datasets/")
    args.add_argument('--output_folder', required=True)
    args.add_argument('--input_folder', default="data/wsd")

    args.add_argument('--scorer_folder', default='experiments/wsd')

    args.add_argument('--model_name', default='bert-large-cased')
    args.add_argument('--ares_path', default='ares_embedding/ares_bert_large.txt')
    args.add_argument('--top_k', type=int, default=3)
    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
