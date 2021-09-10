if __name__ == '__main__':
    input_path = 'scoring_scripts/evalita/evalita_gold.txt'
    output_path = 'scoring_scripts/evalita/evalita_cleaned_gold.txt'

    count = 0

    with open(output_path, 'w') as out:
        for line in open(input_path):
            line = line.replace("a'", "à").replace("e'", "è").replace("i'", "ì").replace("o'", "ò").replace("u'", "ù")
            if len(line.strip().split(' :: ')) == 2:
                out.write(line)
            else:
                count += 1

    print(f'No gold instances: {count}')