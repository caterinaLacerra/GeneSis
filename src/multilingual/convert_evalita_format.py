from src.utils import get_gold_dictionary, read_from_evalita_format

if __name__ == '__main__':

    input_xml = 'data/translation/evalita/lexsub_test.xml'
    gold_path = 'data/translation/evalita/gold.test'

    output_path = 'data/translation/evalita/evalita_test.tsv'

    gold_dict = get_gold_dictionary(gold_path)

    empty_instances = set()

    counter = 0
    for key, value in gold_dict.items():
        if value == '':
            empty_instances.add(key)
            counter += 1
    print(f'Empty instances: {counter}')

    with open(output_path, 'w') as out:
        for instance in read_from_evalita_format(input_xml):
            if instance.instance_id not in empty_instances and instance.instance_id in gold_dict:
                gold_str = gold_dict[instance.instance_id]
                instance.sentence = instance.sentence.replace("e'", "è")
                instance.sentence = instance.sentence.replace("o'", "ò")
                instance.sentence = instance.sentence.replace("a'", "à")
                instance.sentence = instance.sentence.replace("i'", "ì")
                instance.sentence = instance.sentence.replace("u'", "ù")

                out.write(f'{instance.target}\t{instance.instance_id}\t{instance.target_idx}\t'
                          f'{instance.sentence}\t---\t{gold_str}\n')
