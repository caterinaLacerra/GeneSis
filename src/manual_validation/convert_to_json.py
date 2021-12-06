import json
import os

from src.utils import read_from_input_file

if __name__ == '__main__':
    input_path = "data/translation/lst.translated_it.tsv"
    output_json_folder = "data/annotations/"
    if not os.path.exists(output_json_folder):
        os.makedirs((output_json_folder))

    annotation_instances = []

    for instance in read_from_input_file(input_path):
        format_sent = ' '.join([word if i != instance.target_idx else f'**{word}**'
                                  for i, word in enumerate(instance.sentence.split())])

        input = f"{instance.instance_id}##{instance.target}\n{format_sent}"

        annotation_instances.append({
            "sentence": input,
            "substitutes": list(instance.gold.keys()),
        })

    with open(os.path.join(output_json_folder, "lst.json"), 'w') as out:
        json.dump(annotation_instances, out)
