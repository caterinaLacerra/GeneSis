from datasets import load_dataset_builder

if __name__ == '__main__':
    dataset_builder = load_dataset_builder('wikipedia')
    print(dataset_builder.info.features)
    print(dataset_builder.info.splits)
