import random
from typing import List, Dict, Iterator

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from src.wsd.utils.utils import chunks, flatten
from src.wsd.utils import read_from_json_format


class BertDataset(IterableDataset):

    def __init__(self, conf, dataset_path) -> None:

        super().__init__()

        self.config = conf
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.transformer_name, is_fast=True)
        # todo: define sense inventory

        self.input_path = dataset_path
        self.dataset_store = []


    def encode(self, source: str) -> List[int]:
        sample = self.tokenizer([source], return_tensors='pt')
        return sample['input_ids'][0]

    def __init_dataset(self) -> None:

        for instance in read_from_json_format(self.input_path):
            encoded_original_input = self.encode(instance.context)
            self.dataset_store.append((encoded_original_input, instance))

            for substitute_context in instance.substitutes_contexts:
                encoded_input = self.encode(substitute_context)
                self.dataset_store.append((encoded_input, instance))

        self.dataset_store = sorted(self.dataset_store, key=lambda x: len(x[0]) + random.randint(0, 10))

        self.dataset_store = list(chunks(self.dataset_store, self.config.model.chunk_size))
        random.shuffle(self.dataset_store)
        self.dataset_store = flatten(self.dataset_store)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:

        if len(self.dataset_store) == 0:
            self.__init_dataset()

        current_batch = []

        def output_batch() -> Dict[str, torch.Tensor]:

            input_ids = pad_sequence([x[0] for x in current_batch], batch_first=True,
                                     padding_value=self.tokenizer.pad_token_id)

            instances = [x[-1] for x in current_batch]

            batch = {
                "source": input_ids,
                "metadata": instances,
            }

            return batch

        for element in self.dataset_store:


            encoded_source, instance = element

            future_source_tokens = max(
                max([encoded_source.size(0) for encoded_source, *_ in current_batch], default=0),
                len(encoded_source)
            ) * (len(current_batch) + 1)

            if future_source_tokens > self.config.model.max_tokens_per_batch:

                if len(current_batch) == 0:
                    continue

                yield output_batch()
                current_batch = []

            current_batch.append((encoded_source, instance))

        if len(current_batch) != 0:
            yield output_batch()
