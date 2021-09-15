from src.utils import read_from_input_file, convert_to_lst_target

if __name__ == '__main__':
    input_path = 'scoring_scripts/evalita/evalita_cleaned_gold.txt'
    input_path_text = 'data/translation/evalita/evalita_cleaned_test.tsv'

    output_path = 'scoring_scripts/evalita/evalita_cleaned_no_mw_gold.txt'
    output_path_text = 'data/translation/evalita/evalita_cleaned_no_mw_test.tsv'

    count = 0

    with open(output_path, 'w') as out:
        for line in open(input_path):
            if len(line.strip().split(' :: ')) == 2:
                target, substitutes = line.strip().split(' :: ')
                substitutes = substitutes.strip().split(';')
                clean_words = {}

                for element in substitutes:
                    if element == "":
                        continue
                    *words, score = element.split()
                    if len(words) == 1:
                        clean_words["".join(words)] = score

                if len(clean_words) > 0:
                    new_gold = ";".join([" ".join([value, score]) for value, score in clean_words.items()])
                    out.write(f'{target} :: {new_gold}\n')

                else:
                    count += 1
            else:
                count += 1

    print(f'Removed instances: {count}')

    gold_keys = set([line.strip().split(' :: ')[0] for line in open(output_path)])

    with open(output_path_text, 'w') as out:
        for instance in read_from_input_file(input_path_text):
            target = convert_to_lst_target(instance.target)
            key = f'{target} {instance.instance_id}'
            if key in gold_keys:
                out.write(str(instance) + '\n')
