import os
import string
import subprocess
from typing import Dict, Optional, List, Iterable, Any, Set, Tuple

import numpy as np
import xml.etree.ElementTree as ET

import torch
import tqdm
import transformers

from src.wsd.utils.utils import LexSubInstance

_universal_to_lst = {
    'NOUN': 'n',
    'ADJ': 'a',
    'ADV': 'r',
    'VERB': 'v'
}

def convert_to_universal(pos: str):
    pos = pos.upper()
    if pos in {'NOUN', 'ADJ', 'ADV', 'VERB'}:
        return pos

    if pos.startswith('V'):
        return 'VERB'
    elif pos.startswith('N'):
        return 'NOUN'
    elif pos.startswith('R'):
        return 'ADV'
    elif pos.startswith('J') or pos.startswith('A') or pos.startswith('S'):
        return 'ADJ'
    raise ValueError('Unknown pos tag {}'.format(pos))


def convert_to_universal_target(lexeme: str) -> str:
    *lemma, pos = lexeme.split('.')
    pos = convert_to_universal(pos)
    return ".".join(lemma) + "." + pos


def convert_to_lst_target(target: str) -> str:
    *lemma, pos = target.split('.')
    lemma = '.'.join(lemma)
    pos = _universal_to_lst[pos]
    return f'{lemma}.{pos}'


def read_from_input_file(input_path: str, encoding: str = 'utf-8') -> Iterable[LexSubInstance]:
    for line in open(input_path, encoding=encoding):
        if len(line.strip().split('\t')) == 6:
            target, instance_id, target_idx, sentence, mask, gold = line.strip().split('\t')
            mask = list(set([x for x in mask.split(' ')]))
            gold = {x.split('::')[0]: float(x.split('::')[1]) for x in gold.split(' ')}

        else:
            target, instance_id, target_idx, sentence = line.strip().split('\t')
            mask = None
            gold = None

        *lemma, pos = target.split('.')
        lemma = '.'.join(lemma)
        pos = convert_to_universal(pos)
        target = f'{lemma}.{pos}'
        target_idx = get_target_index_list(target_idx)

        yield LexSubInstance(target=target, instance_id=instance_id, target_idx=target_idx,
                             sentence=sentence, mask=mask, gold=gold)


def flatten(lst: List[list]) -> list:
    return [_e for sub_l in lst for _e in sub_l]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def file_len(file_name):
    return int(subprocess.check_output(f"wc -l {file_name}", shell=True).split()[0])


class WSDInstance:

    def __init__(self, lemma: str, pos: str, word: str, sentence: str, sense: Optional[List[str]] = None,
                 instance_id: Optional[str] = None, target_idx: Optional[int] = None):
        self.instance_id = instance_id
        self.target_idx = target_idx
        self.sentence = sentence
        self.lemma = lemma
        self.pos = pos
        self.word = word
        self.sense = sense

        if ' ' in word:
            len_mw = len(word.split(' '))
            end_idx = target_idx + len_mw -1
            self.target_idx = [target_idx, end_idx]

        else:
            self.target_idx = [self.target_idx]


    def __repr__(self):
        if self.sense:
            return f'{self.sense} {self.instance_id} {self.target_idx} {self.sentence}'

        return f'{self.lemma} {self.pos} {self.sentence}'


def read_from_raganato_format(input_xml: str, input_keys: str) -> Iterable[List[WSDInstance]]:
    sense_keys = {}
    for line in open(input_keys):
        line = line.strip().split(' ')
        key = line[0]
        sense_keys[key] = line[1:]

    root = ET.parse(input_xml).getroot()
    for element in root.iter(tag='corpus'):

        for sentence in element.iter(tag='sentence'):
            sentence_str = ' '.join([x.text for x in sentence.iter() if x.tag in {'wf', 'instance'}])
            instances = []

            instance_index = 0
            for instance in sentence.iter():
                if instance.tag == 'wf':
                    lemma = instance.attrib['lemma']
                    pos = instance.attrib['pos']
                    word = instance.text
                    instances.append(WSDInstance(lemma, pos, word, sentence_str))

                    instance_index += len(word.split(' '))

                elif instance.tag == 'instance':
                    instance_id = instance.attrib['id']
                    lemma = instance.attrib['lemma']
                    pos = instance.attrib['pos']
                    word = instance.text
                    instances.append(WSDInstance(lemma, pos, word, sentence_str, sense_keys[instance_id], instance_id,
                                                 instance_index))

                    instance_index += len(word.split(' '))

            yield instances


def contains(small: List[str], big: List[str]):
    for i in range(len(big) - len(small) + 1):
        for j in range(len(small)):
            if big[i + j] != small[j]:
                break
        else:
            return i, i + len(small)
    return False

def recover_mw_bpes(bpes: List[str], words: List[str], word_idx: List[int], tokenizer):

    target_word = " ".join([words[x] for x in word_idx])
    tokenized = tokenizer.tokenize(target_word)
    if len(tokenized) == 0:
        return None

    start_indexes = [i for i, x in enumerate(bpes) if x == tokenized[0]]

    if not start_indexes:
        return None

    diff = [abs(x - word_idx[0]) for x in start_indexes]

    start_index = np.asarray(start_indexes)[np.argmin(diff)]

    try:
        start, end = contains(tokenized, bpes[start_index:])
        return [x for x in range(start + start_index, end + start_index)]
    except:
        return None


def yield_batch(input_path: str, separator: str):
    lines = []
    for line in open(input_path):
        if line == separator:
            yield lines
            lines = []

        else:
            lines.append(line)

    if lines:
        yield lines


def define_exp_name(config: Dict[str, Any]) -> str:
    exp_name = config['exp_name']
    exp_name = f'{exp_name}_{config["model"]["seed"]}'

    exp_name = f'{exp_name}_pt_{config["datasets"]["pretrain"]}'

    if 'dropout' in config['model'] and config['model']['dropout'] != 0:
        exp_name = f'{exp_name}_drop_{config["model"]["dropout"]}'

    if 'encoder_layerdropout' in config['model'] and config['model']['encoder_layerdropout']!=0:
        if config['model']['encoder_layerdropout']:
            exp_name = f'{exp_name}_enc_lyd_{config["model"]["encoder_layerdropout"]}'

    if 'decoder_layerdropout' in config['model'] and config['model']['decoder_layerdropout']!=0:
        if config['model']['decoder_layerdropout']:
            exp_name = f'{exp_name}_dec_lyd_{config["model"]["decoder_layerdropout"]}'

    return exp_name


def define_generation_out_folder(config: Dict[str, Any]) -> str:
    shorten_gen_keys = config["shorten_gen_keys"]
    out_name = "_".join(sorted([f'{shorten_gen_keys[k]}_{v}'
                                for k, v in config["generation_parameters"].items()
                                if shorten_gen_keys[k] != None and shorten_gen_keys[k]!='None']))

    return out_name


def contains_punctuation(word: str) -> bool:
    punct = set([x for x in string.punctuation])
    return any(char in punct for char in word)


def get_output_dictionary(output_vocab_folder: str) -> Dict[str, Set[str]]:
    output_vocab = {}
    for filename in tqdm.tqdm(os.listdir(output_vocab_folder)):
        target = filename.replace('.txt', '')
        output_vocab[target] = set()
        for line in open(os.path.join(output_vocab_folder, filename)):
            output_vocab[target].add(line.strip().lower())
    return output_vocab


def get_target_index_list(target_idx: str) -> List[int]:

    if '[' in target_idx:
        target_index = target_idx.replace('[', '').replace(']', '').split(', ')
        target_index = [int(x) for x in target_index]

    elif '##' in target_idx:
        target_index = target_idx.split('##')
        target_index = [int(x) for x in target_index]

    else:
        target_index = [int(target_idx)]

    return target_index


def universal_to_wn_pos(upos: str) -> List[str]:
    pos_map = {'NOUN': ['n'], 'VERB': ['v'], 'ADJ': ['a', 's'], 'ADV': ['r']}
    return pos_map[upos]


def multipos_to_pos(pos_s: List[str]) -> str:

    _pos2score = {
        'ADJ': 1,
        'ADP': 0,
        'ADV': 1,
        'CCONJ': 0,
        'DET': 0,
        'INTJ': 0,
        'NOUN': 3,
        'NUM': 0,
        'PART': 0,
        'PRON': 0,
        'PUNCT': 0,
        'SCONJ': 0,
        'SYM': 0,
        'VERB': 2,
        'X': 1
    }

    transformations = {'AUX': 'VERB', 'PROPN': 'NOUN'}
    pos_s = [transformations.get(pos, pos) for pos in pos_s]
    if len(pos_s) == 1 or len(set(pos_s)) == 1:
        return pos_s[0]
    else:
        if 'NOUN' in pos_s:
            return 'NOUN'
        elif 'VERB' in pos_s:
            return 'VERB'
        else:
            return max(pos_s, key=lambda x: _pos2score[x])


def map_to_wn_pos(upos: str) -> str:

    mapping_pos = {'NOUN':'n', 'ADJ':'a', 'ADV': 'r', 'VERB':'v'}

    if upos in mapping_pos:
        return mapping_pos[upos]

    return None

def get_gold_dictionary(gold_path: str) -> Dict[str, str]:

    dict_keys = {}

    for line in open(gold_path):
        lexeme_info, substitutes = line.strip().split('::')
        target, instance_id = lexeme_info.strip().split()

        substitute_list = []
        for substitute_pair in substitutes.split(';'):
            if substitute_pair != '':
                *words, score = substitute_pair.split()
                word = '_'.join(words)
                substitute_list.append((word, score))

        substitute_list = sorted(substitute_list, key=lambda x:x[1], reverse=True)
        dict_keys[instance_id] = " ".join([f'{w}::{s}' for w, s in substitute_list])

    return dict_keys

def save_reduced_numberbatch(input_path: str, output_path: str, languages: List[str]):

    rows, cols = 0, 0
    for i, line in tqdm.tqdm(enumerate(open(input_path))):
        if i == 0:
            cols = int(line.strip().split()[0])
        else:
            word_id, *vector = line.strip().split()
            *_, lang_code, word = word_id.split('/')
            if lang_code in languages:
                rows += 1

    with open(output_path, 'w') as output:
        for i, line in tqdm.tqdm(enumerate(open(input_path))):
            if i == 0:
                output.write(f"{rows} {cols}\n")

            else:
                word_id, *vector = line.strip().split()
                *_, lang_code, word = word_id.split('/')
                if lang_code in languages:
                    output.write(line)

def load_numberbatch(txt_path: str, languages: Optional[List[str]]=None) -> Dict[str, np.matrix]:

    vectors = {}

    if languages is None:

        for i, line in tqdm.tqdm(enumerate(open(txt_path))):
            if i == 0:
                continue
            else:
                word_id, *vector = line.strip().split()
                vector = np.asarray([float(x) for x in vector])
                vectors[word_id] = vector
    else:
        for i, line in tqdm.tqdm(enumerate(open(txt_path))):
            if i == 0:
                continue
            else:
                word_id, *vector = line.strip().split()
                *_, lang_code, word = word_id.split('/')
                if lang_code in languages:
                    vector = np.asarray([float(x) for x in vector])
                    vectors[word_id] = vector

    return vectors


def get_numberbatch_key(target_word: str, language: str) -> str:
    key = f"/c/{language}/{target_word}"
    return key

def find_numberbatch_keys(target_word: str, language: str, keys: Dict[str, np.matrix]) -> Optional[List[str]]:
    target_word = target_word.lower()

    ret_keys = []
    for word in target_word.split("_"):
        key = get_numberbatch_key(word, language)
        if key in keys:
            ret_keys.append(key)

        else:
            # search for the same word in en
            if language != "en":
                key = get_numberbatch_key(word, "en")
                if key in keys:
                    ret_keys.append(key)

                else:
                    prefix = word[:-1]
                    while len(prefix) > 1:
                        avg_indexes = [k for k in keys if k.split("/")[-1] == prefix]
                        if avg_indexes != []:
                            ret_keys.extend(avg_indexes)
                        prefix = prefix[:-1]

            else:
                prefix = word[:-1]
                while len(prefix) > 1:
                    avg_indexes = [k for k in keys if k.split("/")[-1] == prefix]
                    if avg_indexes != []:
                        ret_keys.extend(avg_indexes)
                    prefix = prefix[:-1]

    if ret_keys != []:
        return ret_keys

    return None


def extract_word_embedding(input_sentences: List[str], target_indexes: List[List[int]],
                           tokenizer: transformers.AutoTokenizer.from_pretrained,
                           embedder: transformers.AutoModelForMaskedLM.from_pretrained,
                           device: torch.device,
                           model_config: transformers.AutoConfig,
                           special_chars: str) -> Tuple[torch.Tensor, list]:

    idx_to_token = {v: k for k, v in tokenizer.get_vocab().items()}

    # embed sentences as the average of the last four layers
    tokenized = tokenizer.batch_encode_plus(input_sentences, return_tensors='pt', padding=True, truncation=True,
                                            max_length=1024)

    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    layer_indexes = [model_config.num_hidden_layers - 4, model_config.num_hidden_layers]

    embedder.to(device)
    with torch.no_grad():
        embedder.eval()
        hidden_states = embedder(input_ids, attention_mask, output_hidden_states=True)["hidden_states"]

    # average of the last four layers
    hidden_states = torch.mean(torch.stack(hidden_states[layer_indexes[0]:layer_indexes[-1] + 1]), dim=0)

    words = [[x for x in sentence.split(' ') if x != ""] for sentence in input_sentences]
    bpes = [[idx_to_token[idx.item()] for idx in sentence] for sentence in input_ids]

    target_bpes = []

    for j in range(len(input_ids)):

        # for each sentence, consider the target and retrieve the corresponding bpes
        bpes_idx = recover_mw_bpes(bpes[j], words[j], target_indexes[j], tokenizer)

        if bpes_idx is None:
            target_bpes.append(None)
            continue

        reconstruct = ''.join(bpes[j][bpes_idx[0]:bpes_idx[-1] + 1]).replace(special_chars, '')
        target = "".join([words[j][x] for x in target_indexes[j]])

        if target != reconstruct:
            target_bpes.append(None)
            continue

        target_bpes.append(bpes_idx)

    return hidden_states, target_bpes
