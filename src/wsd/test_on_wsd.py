import argparse
import os

import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader

from src.dataset import BartDataset
from src.model import BartModel
from src.wsd.utils.utils import define_generation_out_folder
#from src.wsd.utils import get_clean_generated_substitutes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--beams', type=int, default=0)
    parser.add_argument('--sequences', type=int, default=0)
    parser.add_argument('--cuda_device', type=int, default=None)
    parser.add_argument('--ckpt', required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    cuda_device = args.cuda_device
    configuration = yaml.load(open(args.config_path), Loader=yaml.FullLoader)

    bart_name = configuration['model']['name']
    max_tokens_per_batch = configuration['model']['max_tokens_per_batch']

    data_dir = configuration['paths']['data_dir']
    dataset_name = configuration['datasets']['wsd_dataset']
    map_location = 'cuda' if cuda_device == 0 else 'cpu'

    test_dataset = BartDataset(os.path.join(data_dir, 'wsd', f'{dataset_name}_test.tsv'),
                               bart_name, max_tokens_per_batch)

    model = BartModel.load_from_checkpoint(args.ckpt, strict=False, map_location=map_location)

    test_dataloader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    model.generation_parameters = configuration["generation_parameters"]

    if args.beams != 0:
        model.generation_parameters["num_beams"] = args.beams

    if args.sequences != 0:
        model.generation_parameters["num_return_sequences"] = args.sequences

    trainer = pl.Trainer(gpus=[cuda_device] if cuda_device is not None else None)

    test_dictionary = trainer.test(test_dataloaders=[test_dataloader], model=model)

    out_name = define_generation_out_folder(configuration)

    ckpt_path, ckpt_name = os.path.split(args.ckpt)
    input_folder = os.path.split(os.path.split(ckpt_path)[0])[0]
    output_folder = os.path.join(input_folder, out_name, 'wsd_output_files')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, f'wsd_output_{dataset_name}.txt')

    print(f'Saving output in {output_path}')

    with open(output_path, 'w') as out:

        for idx, element in enumerate(model.generated_batches):
            instance_batch, generated_batch = element

            for instance, generation in zip(instance_batch, generated_batch):

                out.write(f'{instance.target} {instance.instance_id} {instance.target_idx}\n')
                out.write(f'{instance.sentence}\n')

                out.write(generation)
                substitutes_dict = get_clean_generated_substitutes(generation, '.'.join(instance.target.split('.')[:-1]))

                sorted_by_freq = sorted([(k, v) for k, v in substitutes_dict.items()],
                                        key=lambda x: x[1], reverse=True)
                str_sorted = ";".join([f"{k}: {v}" for (k, v) in sorted_by_freq])
                out.write(f'\n# candidates: {str_sorted}')
                out.write('\n#########\n')
