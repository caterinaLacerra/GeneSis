import argparse
import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.dataset import MBartDataset
from src.model import MBartModel
from src.wsd.utils.utils import define_exp_name, define_generation_out_folder


def main(args: argparse.Namespace):

    configuration_path = args.config_path
    cuda_device = args.cuda_device

    # load configuration
    configuration = yaml.load(open(configuration_path), Loader=yaml.FullLoader)

    pl.seed_everything(args.seed)

    configuration['model']['seed'] = args.seed

    exp_name = define_exp_name(configuration)
    out_name = define_generation_out_folder(configuration)

    output_folder = os.path.join(configuration['paths']['output_folder'], exp_name, out_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_dataset_path = os.path.join(output_folder, 'input_training_dataset.tsv')

    os.system(f'cp {configuration_path} {output_folder}')

    data_dir = configuration['paths']['data_dir']
    pre_train_dataset_name = configuration['datasets']['pretrain']
    dev_dataset_name = configuration['datasets']['dev']
    bart_name = configuration['model']['name']

    gold_path = os.path.join(configuration['paths']['scorer_dir'], f"{configuration['datasets']['test']}_gold.txt")
    max_tokens_per_batch = configuration['model']['max_tokens_per_batch']

    train_dataset = MBartDataset(os.path.join(data_dir, f'{pre_train_dataset_name}_train.tsv'), bart_name,
                                 src_lang=configuration['model']['src_lang'],
                                 max_tokens_per_batch=max_tokens_per_batch, gold_path=gold_path,
                                 input_dataset_path=input_dataset_path)

    val_dataset = MBartDataset(os.path.join(data_dir, f'{dev_dataset_name}_dev.tsv'), bart_name,
                               configuration['model']['src_lang'],
                               max_tokens_per_batch)

    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=0)

    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=0)

    model = MBartModel(configuration_path)

    if 'debug' in exp_name:
        logger = None

    else:
        # setup logger
        logger = WandbLogger(
            name=exp_name,
            project=configuration['wandb']['project_name']
        )

        logger.log_hyperparams(configuration)

    checkpoint_conf = configuration['trainer']['checkpoint']

    checkpointer = ModelCheckpoint(
        dirpath=f'{output_folder}/checkpoints',
        filename=checkpoint_conf['filename'],
        monitor=checkpoint_conf['monitor'],
        mode=checkpoint_conf['mode'],
        save_top_k=checkpoint_conf['save_top_k'],
        save_last=checkpoint_conf['save_last'],
        verbose=True
    )

    early_stopper = EarlyStopping(monitor=checkpoint_conf['monitor'],
                                  mode=checkpoint_conf['mode'],
                                  patience=configuration['trainer']['patience'])

    # setup trainer and train
    trainer = pl.Trainer(
        gpus=[cuda_device] if cuda_device >= 0 else None,
        accumulate_grad_batches=configuration['trainer']['gradient']['accumulation'],
        gradient_clip_val=configuration['trainer']['gradient']['clipping'],
        logger=logger,
        precision=16 if configuration['trainer']['use_amp'] and cuda_device >= 0 else 32,
        callbacks=[checkpointer, early_stopper],
        max_epochs=configuration['trainer']['max_epochs']
    )

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,  help='path to the yaml configuration file.')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
