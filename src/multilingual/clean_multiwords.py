import argparse
import io
import time
import urllib
import json
import gzip

import tqdm

from src.utils import read_from_input_file, file_len


def check_multiwords(service_url: str, key: str, lemma: str, pos: str, lang: str):

    params = {
        'lemma': lemma,
        'searchLang': lang,
        'pos': pos,
        'key': key
    }


    url = service_url + '?' + urllib.parse.urlencode(params)
    try:
        request = urllib.request.Request(url)
        request.add_header('Accept-encoding', 'gzip')
        response = urllib.request.urlopen(request)
    except:
        time.sleep(40)
        request = urllib.request.Request(url)
        request.add_header('Accept-encoding', 'gzip')
        response = urllib.request.urlopen(request)

    ids = []
    if response.info().get('Content-Encoding') == 'gzip':
        buf = io.BytesIO(response.read())
        f = gzip.GzipFile(fileobj=buf)
        data = json.loads(f.read())
        for result in data:
            ids.append(result['id'])

    return ids

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--removed', required=True)
    parser.add_argument('--lang', required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    service_url = 'https://babelnet.io/v6/getSynsetIds'
    key = "c6ffa010-fd3c-4671-a95a-a132ba8b7cac"

    lang = args.lang
    
    removed_words = args.removed
    output_path = args.output_path
    input_path = args.input_path

    set_multiwords = set()

    for instance in tqdm.tqdm(read_from_input_file(input_path), total=file_len(input_path)):
        target = instance.target
        *lemma, pos = target.split('.')
        pos = pos.upper()
        lemma = ".".join(lemma)
        set_multiwords.add(target)

        for substitute, score in instance.gold.items():
            set_multiwords.add(f'{substitute}.{pos}')

    print(f"Vocab dim: {len(set_multiwords)}")

    set_allowed_words = set()
    for lexeme in tqdm.tqdm(set_multiwords):
        *lemma, pos = lexeme.split('.')
        lemma = '.'.join(lemma)
        ids = check_multiwords(service_url, key, lemma, pos, lang)
        if len(ids) > 0:
            set_allowed_words.add(lexeme)

    with open(output_path, 'w') as out, open(removed_words, 'w') as removed:
        for instance in tqdm.tqdm(read_from_input_file(input_path), total=file_len(input_path)):
            target = instance.target
            pos_tag = target.split('.')[-1]

            if target in set_allowed_words:
                new_gold = {}
                for subst, score in instance.gold.items():
                    if f'{subst}.{pos_tag}' in set_allowed_words:
                        new_gold[subst] = score
                    else:
                        removed.write(subst + '\n')

                if len(new_gold) > 0:
                    instance.gold = new_gold
                    out.write(str(instance) + '\n')

            else:
                removed.write(target + '\n')
