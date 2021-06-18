import nltk

from nltk.corpus import wordnet as wn

from src.utils import universal_to_wn_pos


def get_csi_keys(synset, pos: str):

    return f'wn:{str(synset.offset()).zfill(8)}{universal_to_wn_pos(pos)[0]}'

def get_synset_from_str_offset(str_offset: str) -> nltk.corpus.reader.wordnet.Synset:

    offset = str_offset.replace('wn:', '')
    pos = offset[-1]
    offset = int(offset[:-1])
    synset = wn.synset_from_pos_and_offset(pos, offset)
    return synset