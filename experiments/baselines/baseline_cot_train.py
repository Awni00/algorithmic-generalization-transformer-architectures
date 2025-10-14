import os
import argparse
import yaml
import pickle
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import wandb

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer

from baseline_models import LitCoTRecurrentTransformerModel, get_experiment_name
from Simtransformer.simtransformer.utils import EasyDict

import copy
current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument('--config_dir', type=str)
# parser.add_argument('--config_dir', type=str, default=os.path.join(current_dir, 'configs/CoT_configs/COT-Nodes32-ADD-Baseline-T1L2H16D256_DeBERTa'))
parser.add_argument('--debug', action='store_true', help='Run in debug mode (no logging, no checkpoints).')


class CoTDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

        # load data
        self.data = torch.load(data_path, weights_only=True)
        self.data, self.ds_info = self.data['tokenized_dataset'], self.data['dataset_info']

        # convert to tensors
        for k in self.ds_info.keys():
            self.ds_info[k] = torch.tensor(self.ds_info[k])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        cot_mask, value_mask, var_mask, operation_mask = tuple(self.ds_info[k][idx][:-1] for k in ['cot_mask', 'value_mask', 'var_mask', 'operation_mask'])

        x = self.data[idx][:-1]
        y = self.data[idx][1:]

        return ((x, y), (cot_mask, value_mask, var_mask, operation_mask))


if __name__ == '__main__':

    args = parser.parse_args()

    # set the seed for reproducibility
    seed = np.random.randint(0, 2**32) # randomly sample a seed
    pl.seed_everything(seed) # sets the seed for all random number generators

    # load model, train, and data config
    with open(os.path.join(args.config_dir, 'model_config.yaml')) as f:
        model_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    with open(os.path.join(args.config_dir, 'train_config.yaml')) as f:
        train_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    with open(os.path.join(args.config_dir, 'data_config.yaml')) as f:
        data_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    train_config.seed = seed

    # load tokenizer
    with open(os.path.join(data_config.data_dir, data_config.tokenizer_file_name), 'rb') as f:
        tokenizer = pickle.load(f)

    model_config.vocab_size = len(tokenizer.vocab)

    # set max_seq_len
    model_config.max_seq_len = max(model_config.max_seq_len, data_config.dag_config.max_length, data_config.val_dag_config.max_length)

    # load data

    train_path = os.path.join(data_config.data_dir, data_config.dag_config.data_file_name)
    val_path = os.path.join(data_config.data_dir, data_config.val_dag_config.data_file_name)

    train_ds = CoTDataset(train_path)
    val_ds = CoTDataset(val_path)

    train_dl = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=train_config.batch_size, shuffle=False, num_workers=4)
    # set matmul precision
    if getattr(train_config, 'matmul_precision', None) is not None:
        torch.set_float32_matmul_precision(train_config.matmul_precision)
        print(f'Matmul precision set to {train_config.matmul_precision}.')

    # create the model and Lightning module
    litmodel = LitCoTRecurrentTransformerModel(model_config, data_config, train_config)
    litmodel.factored_tokenizer = tokenizer

    # Group: Data Config - Model Config ... Run name: Seed + Date-Time
    group_name, run_name = get_experiment_name(model_config, data_config, train_config)

    train_config.experiment_run_name = run_name
    train_config.experiment_group = group_name

    experiment_config = EasyDict(dict(train_config=train_config, model_config=model_config))
    experiment_config.chain_of_thought = True


    logger = pl.loggers.WandbLogger(
        entity=train_config.wandb_config.wandb_entity, project=train_config.wandb_config.wandb_project,
        name=train_config.experiment_run_name, group=train_config.experiment_group,
        config=experiment_config, log_model=True) if not args.debug else None

    # watch gradients and model params
    if not args.debug:
        logger.watch(litmodel.model, log='all', log_graph=True)

    # callbacks: checkpoint and lr monitor
    checkpoint_dir = os.path.join(f'checkpoints/cot_baselines/{train_config.experiment_group}-{train_config.experiment_run_name}')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_loss:.4f}',
        monitor='val/loss', # this depends on logging in the LightningModule
        mode='min',
        save_top_k=3,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor, checkpoint_callback]

    if args.debug:
        callbacks = []
        logger = None

    # compile the model
    if getattr(train_config, 'compile', False):
        # litmodel = torch.compile(litmodel)
        litmodel.model = torch.compile(litmodel.model)
        print('Model compiled.')

    trainer_kwargs = dict(
        max_epochs=train_config.max_epochs,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=getattr(train_config, 'val_check_interval', 1.0),
        precision=getattr(train_config, 'precision', '32'),
        accelerator='gpu',
        # devices=1,
    )

    if getattr(trainer_kwargs, 'precision', None) is not None:
        print(f'Precision set to {trainer_kwargs.precision}.')

    trainer = Trainer(**trainer_kwargs)

    # train the model
    trainer.fit(litmodel, train_dl, val_dl)
