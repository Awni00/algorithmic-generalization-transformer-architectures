import os
import argparse
import yaml
import pickle
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer


from baseline_models import LitRecurrentTransformerModel, get_experiment_name
from tokenizers import FactoredVocabTokenizer
from Simtransformer.simtransformer.utils import EasyDict

import copy
current_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, default=os.path.join(current_dir, 'configs/Nodes32-ADD-BaselineT1L4H16D512_DeBERTa'))
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (no logging, no checkpoints).')

    args = parser.parse_args()

    # set the seed for reproducibility
    seed = np.random.randint(0, 2**32) # randomly sample a seed
    pl.seed_everything(seed) # sets the seed for all random number generators

else:
    args = EasyDict(config_dir=os.path.join(current_dir, 'configs/Nodes32-ADD-L2H16D256_DeBERTa'), debug=True)
    seed = 0
    pl.seed_everything(seed)

# load model, train, and data config
with open(os.path.join(args.config_dir, 'model_config.yaml')) as f:
    model_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

with open(os.path.join(args.config_dir, 'train_config.yaml')) as f:
    train_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

with open(os.path.join(args.config_dir, 'data_config.yaml')) as f:
    data_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

train_config.seed = seed

# load tokenizer
# with open(os.path.join(current_dir, data_config.data_dir, data_config.tokenizer_file_name), 'rb') as f:
    # factored_tokenizer = pickle.load(f)

factored_tokenizer = FactoredVocabTokenizer(n_vars=max(data_config.dag_config.num_nodes, data_config.val_dag_config.num_nodes), ops=data_config.dag_config.func_vocab, mod_val=data_config.dag_config.mod_val, max_fan_in=data_config.dag_config.max_fan_in_deg)

model_config.vocab_sizes = factored_tokenizer.vocab_sizes
model_config.factors = factored_tokenizer.factors

class CustomDataset(Dataset):
    def __init__(self, *args):
        self.data = args

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return tuple(data[idx] for data in self.data)

# Load the data module
class DAGDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path,
                 batch_size,
                 factored_tokenizer,
                 num_workers=4,
                 tr_val_test_split=(0.9, 0.05, 0.05)):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tr_val_test_split = tr_val_test_split
        self.factored_tokenizer = factored_tokenizer

    def setup(self, stage=None):
        # Load your dataset here
        if isinstance(self.data_path, str):
            dataset_dict = torch.load(self.data_path, weights_only=True)
            dataset = dataset_dict['tokenized_dataset']
            dataset_info = dataset_dict['dataset_info']

            tr_split, val_split, test_split = self.tr_val_test_split

            num_data_train = int(len(dataset) * tr_split)
            num_data_val = int(len(dataset) * val_split)
            num_data_test = len(dataset) - num_data_train - num_data_val

            self.data_train_label = dataset[:num_data_train]
            self.data_val_label = dataset[num_data_train:num_data_train+num_data_val]
            self.data_test_label = dataset[-num_data_test:]

            self.data_train_info = dataset_info[:num_data_train]
            self.data_val_info = dataset_info[num_data_train:num_data_train+num_data_val]
            self.data_test_info = dataset_info[-num_data_test:]

        elif isinstance(self.data_path, list) or isinstance(self.data_path, tuple):
            if len(self.data_path) == 3:
                data_train_path, data_val_path, data_test_path = self.data_path
            elif len(self.data_path) == 2:
                data_train_path, data_val_path = self.data_path
                data_test_path = None

            # load the data from the paths
            self.data_train = torch.load(data_train_path, weights_only=True)
            self.data_val = torch.load(data_val_path, weights_only=True)
            self.data_test = self.data_val if data_test_path is None else torch.load(data_test_path, weights_only=True)

            self.data_train_label, self.data_train_info = self.data_train['tokenized_dataset'], self.data_train['dataset_info']
            self.data_val_label, self.data_val_info = self.data_val['tokenized_dataset'], self.data_val['dataset_info']
            self.data_test_label, self.data_test_info = self.data_test['tokenized_dataset'], self.data_test['dataset_info']

            if 'spec' in self.data_train:
                self.train_spec = self.data_train['spec']
            if 'spec' in self.data_val:
                self.val_spec = self.data_val['spec']
            if 'spec' in self.data_test:
                self.test_spec = self.data_test['spec']

        # set the variable tokens to empty
        self.data_train = copy.deepcopy(self.data_train_label)
        value_idx = self.data_train[:, :, self.factored_tokenizer.factors.index('SYNTAX')].eq(self.factored_tokenizer.syntax_tok2idx['VARIABLE'])
        data_value = self.data_train[:, :, self.factored_tokenizer.factors.index('VALUE')]
        data_value[value_idx] = self.factored_tokenizer.value_tok2idx['EMPTY']

        self.data_val = copy.deepcopy(self.data_val_label)
        value_idx_val = self.data_val[:, :, self.factored_tokenizer.factors.index('SYNTAX')].eq(self.factored_tokenizer.syntax_tok2idx['VARIABLE'])
        data_value_val = self.data_val[:, :, self.factored_tokenizer.factors.index('VALUE')]
        data_value_val[value_idx_val] = self.factored_tokenizer.value_tok2idx['EMPTY']

        self.data_test = copy.deepcopy(self.data_test_label)
        value_idx_test = self.data_test[:, :, self.factored_tokenizer.factors.index('SYNTAX')].eq(self.factored_tokenizer.syntax_tok2idx['VARIABLE'])
        data_value_test = self.data_test[:, :, self.factored_tokenizer.factors.index('VALUE')]
        data_value_test[value_idx_test] = self.factored_tokenizer.value_tok2idx['EMPTY']

        self.data_dim = self.data_train.shape[-1]
        self.label_dim = self.data_train_label.shape[-1]
        self.info_dim = self.data_train_info.shape[-1]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.concat([self.data_train, self.data_train_label, self.data_train_info], dim=-1),
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.concat([self.data_val, self.data_val_label, self.data_val_info], dim=-1),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.concat([self.data_test, self.data_test_label, self.data_test_info], dim=-1),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# create the data module
data_module = DAGDataModule(
    data_path=[os.path.join(current_dir, data_config.data_dir, data_config.dag_config.data_file_name), os.path.join(current_dir, data_config.data_dir, data_config.val_dag_config.data_file_name)],
    batch_size=train_config.batch_size,
    num_workers=train_config.num_workers if not args.debug else 0,
    factored_tokenizer=factored_tokenizer,)

# add the dimension of data, label, and info to model config
data_module.setup()
model_config.data_dim = data_module.data_dim
model_config.label_dim = data_module.label_dim
model_config.info_dim = data_module.info_dim


# set matmul precision
if getattr(train_config, 'matmul_precision', None) is not None:
    torch.set_float32_matmul_precision(train_config.matmul_precision)
    print(f'Matmul precision set to {train_config.matmul_precision}.')

# create the model and Lightning module
litmodel = LitRecurrentTransformerModel(model_config, data_config, train_config)
litmodel.factored_tokenizer = factored_tokenizer

if __name__ == '__main__':
    # Group: Data Config - Model Config ... Run name: Seed + Date-Time
    group_name, run_name = get_experiment_name(model_config, data_config, train_config)

    train_config.experiment_run_name = run_name
    train_config.experiment_group = group_name

    experiment_config = EasyDict(dict(train_config=train_config, model_config=model_config))


    logger = pl.loggers.WandbLogger(
        entity=train_config.wandb_config.wandb_entity, project=train_config.wandb_config.wandb_project,
        name=train_config.experiment_run_name, group=train_config.experiment_group,
        config=experiment_config) if not args.debug else None

    # watch gradients and model params
    if not args.debug:
        logger.watch(litmodel.model, log='all', log_graph=True)

    # callbacks: checkpoint and lr monitor
    checkpoint_dir = os.path.join(current_dir, f'checkpoints/baselines/{train_config.experiment_group}-{train_config.experiment_run_name}')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_loss:.4f}',
        monitor='val/total_loss', # this depends on logging in the LightningModule
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
    trainer.fit(litmodel, datamodule=data_module)