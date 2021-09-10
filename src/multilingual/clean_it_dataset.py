from src.utils import read_from_input_file

if __name__ == '__main__':

    input_path = 'data/translation/evalita/evalita_test.tsv'
    output_path = 'data/translation/evalita/evalita_cleaned_test.tsv'

    remove_tokens = ["*CI", "*GLI", "*LA", "*LE", "*LI", "*LO", "*MI", "*NE", "*SE", "*SI", "*TI"]

    with open(output_path, 'w') as out:
        for instance in read_from_input_file(input_path):
            instance.target = instance.target.replace("e'", "è").replace("a'", "à").replace("i'", "ì").replace("o'", "ò").replace("u'", "ù")
            words = instance.sentence.split()
            rem_idx = [i for i, w in enumerate(words) if w in remove_tokens]

            start_idx = instance.target_idx[0]
            target_len = 0 if len(instance.target_idx) == 1 else instance.target_idx[-1] - instance.target_idx[0]

            to_remove = [idx for idx in rem_idx if idx < start_idx]

            if len(to_remove):
                updated_sentence = " ".join([w for i, w in enumerate(words) if i not in rem_idx])
                updated_start = start_idx - len(to_remove)

                if target_len == 0:
                    updated_index = [updated_start]

                else:
                    updated_index = [updated_start, updated_start + target_len]

                instance.sentence = updated_sentence
                old_target = [words[xx] for xx in instance.target_idx]
                new_target = [updated_sentence.split()[cc] for cc in updated_index]
                assert old_target == new_target
                instance.target_idx = updated_index

            out.write(str(instance) + '\n')