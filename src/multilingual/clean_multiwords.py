import argparse
import io
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


    request = urllib.request.Request(url)
    request.add_header('Accept-encoding', 'gzip')
    response = urllib.request.urlopen(request)

    ids = []
    if response.info().get('Content-Encoding') == 'gzip':
        buf = io.BytesIO(response.read())
        f = gzip.GzipFile(fileobj=buf)
        data = json.loads(f.read())
        try:
            for result in data:
                ids.append(result['id'])
        except:
            print(data)
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
    input_path = args.input_path
    removed_words = args.removed
    output_path = args.output_path

    with open(output_path, 'w') as out, open(removed_words, 'w') as rem_out:
        for instance in tqdm.tqdm(read_from_input_file(input_path), total=file_len(input_path)):
            target = instance.target
            *lemma, pos = target.split('.')
            pos = pos.upper()
            lemma = ".".join(lemma)
            synset_ids = check_multiwords(service_url, key, lemma, pos, lang)
            if synset_ids:
                new_gold = {}
                gold = instance.gold

                for substitute, score in gold.items():
                    ids = check_multiwords(service_url, key, substitute, pos, lang)
                    if ids:
                        new_gold[substitute] = score
                    else:
                        rem_out.write(target + '\t' + substitute + '\n')

                if new_gold:
                    instance.gold = new_gold

                    out.write(str(instance) + '\n')
            else:
                rem_out.write(target + '\n')