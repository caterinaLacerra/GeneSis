from src.utils import read_from_input_file, convert_to_lst_target

if __name__ == '__main__':
    input_path = "data/translation/germeval.dev.tsv"
    output_path = "scoring_scripts/germeval_dev_gold.txt"

    with open(output_path, 'w') as out:
        for instance in read_from_input_file(input_path):

            formatted_gold = "; ".join([f"{x[0]} {x[1]}" for key, value in instance.gold.items()
                                       for x in sorted([(key, value)], key=lambda x:x[1], reverse=True)])

            out.write(f"{convert_to_lst_target(instance.target)} {instance.instance_id} :: {formatted_gold}\n")


