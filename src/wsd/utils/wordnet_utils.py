from typing import List

import nltk

from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset


def get_csi_keys(synset, pos: str):

    return f'wn:{str(synset.offset()).zfill(8)}{universal_to_wn_pos(pos)[0]}'

def get_synset_from_str_offset(str_offset: str) -> nltk.corpus.reader.wordnet.Synset:

    offset = str_offset.replace('wn:', '')
    pos = offset[-1]
    offset = int(offset[:-1])
    synset = wn.synset_from_pos_and_offset(pos, offset)
    return synset

def convert_to_wn_pos(pos: str) -> str:
    mapping = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
    return mapping[pos]

def synset_from_sensekey(sensekey: str) -> Synset:
    patching_data = {
        'ddc%1:06:01::': 'dideoxycytosine.n.01.DDC',
        'ddi%1:06:01::': 'dideoxyinosine.n.01.DDI',
        'earth%1:15:01::': 'earth.n.04.earth',
        'earth%1:17:02::': 'earth.n.01.earth',
        'moon%1:17:03::': 'moon.n.01.moon',
        'sun%1:17:02::': 'sun.n.01.Sun',
        'kb%1:23:01::': 'kilobyte.n.02.kB',
        'kb%1:23:03::': 'kilobyte.n.01.kB',
    }
    if sensekey in patching_data:
        synset_name = patching_data[sensekey]
        lemma, pos, *_ = synset_name.split('.')
        synset = [x for x in wn.synsets(lemma, pos) if x.name() in synset_name]

        assert len(synset) == 1
        return synset[0]

    return wn.lemma_from_key(sensekey).synset()


def universal_to_wn_pos(upos: str) -> List[str]:
    pos_map = {'NOUN': ['n'], 'VERB': ['v'], 'ADJ': ['a', 's'], 'ADV': ['r']}
    return pos_map[upos]


def map_to_wn_pos(upos: str) -> str:

    mapping_pos = {'NOUN':'n', 'ADJ':'a', 'ADV': 'r', 'VERB':'v'}

    if upos in mapping_pos:
        return mapping_pos[upos]

    return None