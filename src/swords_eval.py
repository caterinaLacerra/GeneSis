import argparse
import gzip
import json


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--input_path', required=True)
    args.add_argument('--swords_path', required=True)
    args.add_argument('--output_json', required=True)
    return args.parse_args()


if __name__ == '__main__':

    args = parse_args()
    with gzip.open(args.swords_path, 'r') as f:
        eval_data = json.load(f)

    results = {'substitutes_lemmatized': True, 'substitutes': {}}

    legacy_ids_to_subst = {}
    for line in open(args.input_path):
        instance_id, substitutes = line.strip().split(' :::')
        leg_id = int(instance_id.split()[-1])
        substitutes = substitutes.strip().split(";")
        scores = [1 - (1 + x)/len(substitutes) for x,_ in enumerate(substitutes)]
        legacy_ids_to_subst[leg_id] = [(word, score) for word, score in zip(substitutes, scores)]

    for tid, target in eval_data['targets'].items():
        context = eval_data['contexts'][target['context_id']]
        assert len(context['extra']['legacy_ids']) == 1
        legacy_id = context['extra']['legacy_ids'][0]
        if legacy_id in legacy_ids_to_subst:
            results['substitutes'][tid] = legacy_ids_to_subst[legacy_id]
        else:
            results['substitutes'][tid] = []

    print(f"Writing output for swords evaluation in {args.output_json}")
    with open(args.output_json, 'w') as f:
        f.write(json.dumps(results))
