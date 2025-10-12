import sys, time, copy
sys.path.append("..")
# from simtransformer.model_bank import GPT2Standard
from simtransformer.module_base import ConfigBase, DataModuleBase, PipelineBase
import torch
import torch.nn as nn
from torch.utils.data import random_split
from simtransformer.manager import TrainingManagerBase
from simtransformer.model_bank import GPT2LinearReg
from simtransformer.utils import MRR_fn, EasyDict
from simtransformer.model_base import LinearWithChannel
import numpy as np
from torch.utils.data import DataLoader

probe_pos_len = 1

class Config(ConfigBase):
    def prepare(self):
        self.data_config.input_size = self.model_config.input_size

class TrainingManager(TrainingManagerBase):
    def get_training_name(self):
        training_name = f'LinReg_L{self.model_config.num_layers}H{self.model_config.num_heads}W{self.train_config.weight_decay}T' + time.strftime("%m%d-%H%M%S") # default
        print(f"Current training run: {training_name}")
        return training_name
    
    def config_pipeline(self):
        training_model = GPT2LinearReg(self.model_config, 
                                       input_size=self.model_config.input_size, 
                                        output_size=1)
        loss_p_model = nn.MSELoss()
        loss_n_model = None
        return  {
            "train_config": self.train_config,
            "training_model": training_model,
            "loss_p_model": loss_p_model,
            "loss_n_model": loss_n_model
        }
        
    def config_probepipeline(self):
        training_model = self.pipeline.training_model
        
        added_probe_target = EasyDict({'encoder': {'blocks':{}}})
        block_dict_pointer = added_probe_target.encoder.blocks
        for name, block in training_model.encoder.blocks.items():
            block_dict_pointer.update({
                name: {
                    # "input": None,
                    "attn": {"input": None, "output": None}, 
                    # "attn_res_output": None,
                    "mlp": {"input": None, "output": None}, 
                    "output": None,
                }
            })
        added_probe_target_key = added_probe_target.flatten().keys()
        
        in_channel_size_ls = (len(added_probe_target_key),  probe_pos_len)
        out_channel_size_ls = [1]
        
        probe_layer = LinearWithChannel(
            input_size=self.model_config.hidden_size, 
            output_size=1, 
            in_channel_size_ls=in_channel_size_ls,
            out_channel_size_ls=out_channel_size_ls
        )
        
        added_vis_target = EasyDict({'encoder': {'blocks':{}}})
        vis_dict_pointer = added_vis_target.encoder.blocks
        for name, block in training_model.encoder.blocks.items():
            vis_dict_pointer.update({
                name: {
                    "attn": {
                        "attn_prob": None, 
                        "logits_query_pos": None,
                        "logits_pos_key": None,
                        },
                }
            })
        added_vis_target_key = added_vis_target.flatten().keys()
        
        return {
            "probe_config": self.probe_config,
            "pipeline": self.pipeline,
            "probe_layer": probe_layer,
            "probe_loss_model": nn.MSELoss(reduction='none'),
            "added_probe_target_key": added_probe_target_key,
            "added_vis_target_key": added_vis_target_key
        }

class Pipeline(PipelineBase):
    pass
        
        

class DataModule(DataModuleBase):
    def train_val_test_split(self, data):
        # split data into train, validation, and test sets by ratio 90:5:5
        data_train, data_test = random_split(data, [int(0.9*len(data)), len(data)-int(0.9*len(data))])
        data_train, data_val = random_split(data_train, [int(0.9*len(data_train)), len(data_train)-int(0.9*len(data_train))])
        return data_train, data_val, data_test
    
    def transform_batch(self, batch, dataloader_idx):
        """
        Here, each sample consists of a dictionary with "pos", "sentence" and "reasoning_path" keys.
        """
        L, d = batch[0]['x'].shape
        bs = len(batch)
        
        # create a sentence tensor of shape (bs, 2 * L, d)
        sentence = torch.zeros(bs, 2 * L, d)
        
        # on the even indices (0, 2, 4, ...), put the x values
        sentence[:, ::2, :] = torch.stack([sample['x'] for sample in batch])
        # on the odd indices (1, 3, 5, ...), put the y values to the first coordinate in the last dimension
        sentence[:, 1::2, 0] = torch.stack([sample['y'] for sample in batch])

        x_tensor = sentence[:, :-1, :] # shape (bs, 2 * L - 1, d)
        # pad 0 to the dimension to make it of size self.data_config.input_size
        x_tensor = torch.cat([x_tensor, torch.zeros(bs, 2 * L - 1, self.data_config.input_size - d)], dim=-1)
        
        y_tensor = sentence[:, 1:, 0] # shape (bs, 2 * L - 1)
        msk_tensor = torch.zeros_like(y_tensor, dtype=torch.bool) # shape (bs, 2 * L - 1)
        
        # Method one: set the mask to 0 for the odd indices
        msk_tensor[:, ::2] = True
        # Method two: only enable the last token in the sequence
        # msk_tensor[:, -1] = True
        
        # create a probe label tensor of shape (bs, 1), which is the y for the last token in the sequence
        probe_label = torch.stack([sample['y'][-1] for sample in batch]).unsqueeze(1)
        probe_msk_tensor = torch.zeros_like(y_tensor, dtype=torch.bool)
        probe_msk_tensor[:, -1] = True

        # stack all the beta vectors into a tensor of shape (bs, d)
        beta_tensor = torch.stack([sample['beta'] for sample in batch])
        y_tensor = y_tensor.unsqueeze(-1)
        return EasyDict({
            "prompt": x_tensor,
            "label": y_tensor,
            "mask": msk_tensor,
            "probe_label": probe_label,
            "probe_mask": probe_msk_tensor,
            "batch_info": beta_tensor,
        })

    def train_dataloader(self):
        return DataLoader(self.data_train, 
                          batch_size=self.data_config.batch_size, 
                          collate_fn=lambda x: x,
                          shuffle=True, 
                          num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.data_val, 
                          batch_size=self.data_config.batch_size, 
                          collate_fn=lambda x: x,
                          shuffle=False, 
                          num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, 
                          batch_size=self.data_config.batch_size, 
                          collate_fn=lambda x: x,
                          shuffle=False)