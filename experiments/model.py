import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add parent directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
from Simtransformer.simtransformer.model_base import TransformerEncoder
import lightning.pytorch as pl
import lightning
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from Simtransformer.simtransformer.utils import CosineAnnealingWarmup, EasyDict, clever_load, clever_save, Shampoo, signSGD, MRR_fn
from typing import Any, Optional, final, Union
from datetime import datetime
import numpy as np
import copy
from model_interp.conduct_controlled_exp import *

INVALID_NUM = -1
# CONTROLLED_EXP_STEPS = 50

def vectorized_func(func, *T):
    '''
    func: function to apply to each pair of elements from T1 and T2
    T1, T2: Tensors of the same shape
    '''
    orig_shape = T[0].shape
    # except for the last dimension, flatten the tensors
    T_reshape = [t.reshape(-1, 1) for t in T]
    # apply the function to each pair of elements
    result = torch.vmap(func)(*T_reshape)
    # reshape the result to the original shape
    return result.reshape(orig_shape)


class DAGTeacherModel:
    def __init__(self, model_config, factored_tokenizer):
        super().__init__()

        self.model_config = model_config
        self.factored_tokenizer = factored_tokenizer
        self.mod_val = factored_tokenizer.mod_val

        # factors = ['SYNTAX', 'VARIABLE', 'OPERATION', 'VALUE', 'REGRET', 'MODIFICATION']
        self.SYNTAX_idx = self.factored_tokenizer.factors.index('SYNTAX')
        self.VARIABLE_idx = self.factored_tokenizer.factors.index('VARIABLE')
        self.OPERATION_idx = self.factored_tokenizer.factors.index('OPERATION')
        self.VALUE_idx = self.factored_tokenizer.factors.index('VALUE')
        self.REGRET_idx = self.factored_tokenizer.factors.index('REGRET')
        self.MODIFICATION_idx = self.factored_tokenizer.factors.index('MODIFICATION')

        # add all the attributes of self.factored_tokenizer to this class
        for attr in dir(self.factored_tokenizer):
            if not attr.startswith('__'):
                setattr(self, attr, getattr(self.factored_tokenizer, attr))

        # we need to convert the value (idx) to a number using the value_tok2num function applied to each element of the tensor
        # The vmap function is useful to apply a function to the batch. https://pytorch.org/docs/stable/generated/torch.func.vmap.html

    def value_idx2num(self, tokens):
        value_tok = self.value_idx2tok[int(tokens[self.VALUE_idx])]
        regret_tok = self.regret_idx2tok[int(tokens[self.REGRET_idx])]
        if tokens[self.SYNTAX_idx] == self.syntax_tok2idx['VALUE'] and value_tok != 'N/A' and value_tok != 'EMPTY':
            return int(value_tok)
        elif tokens[self.SYNTAX_idx] == self.syntax_tok2idx['VARIABLE'] and value_tok != 'N/A' and value_tok != 'EMPTY':
            # NOTE: if regret is TRUE, the value is not computable and we will return NaN
            return int(value_tok)
        else:
            return torch.nan

    def value_num2idx(self, num):
        if num == INVALID_NUM:
            return INVALID_NUM
        else:
            return self.value_tok2idx[str(num)]

    def apply_operation(self, op, x, y):
        if op == 'ADD':
            return (x + y) % self.mod_val
        elif op == 'MUL':
            return (x * y) % self.mod_val
        elif op == 'SUB':
            return (x - y) % self.mod_val


    def step(self, factored_tokens, batch_info):
        '''
        We implement this teacher model because it is more flexible than just giving the labels for certain depths.
        '''
        # factored_tokens: Tensor of integers of shape [batch_size, seq_len, n_factors]
        # batch_info: Tensor of size [batch_size, seq_len,  max_fan_in_deg * 2], where the first coordinate is the depth of the node and the remaining are the expressions for computing the node
        # move to device cpu

        factored_tokens_copy = factored_tokens.clone().cpu()
        batch_info_copy = batch_info.clone().cpu()
        with torch.no_grad():
            expressions = batch_info_copy[:, :, 1:]

            batch_size = len(batch_info_copy)

            # extract the values of factored_tokens
            value = factored_tokens_copy[:, :, self.VALUE_idx].float() # shape: [batch_size, seq_len]
            # the value can be [0, ..., mod - 1] and also 'N/A' (for SYNTAX != VALUE or VARIABLE), 'EMPTY' (for variables that are not yet computed).

            for i in range(batch_size):
                for j in range(len(value[i])):
                    value[i, j] = self.value_idx2num(factored_tokens_copy[i, j])
                # NOTE: the value will be set to NaN if the value is not computable, not computed, or the regret is TRUE

            output_value = torch.zeros_like(value, requires_grad=False)
            for i, factored_seq, expr_seq, value_seq in zip(range(batch_size), factored_tokens_copy, expressions, value):
                output_value[i] = self.next_value_per_seq(factored_seq, expr_seq, value_seq)

            # replace the NaN values with -1
            output_value[torch.isnan(output_value)] = INVALID_NUM
            # change to int and move to device
            output_value = output_value.int()

            output_value = output_value.apply_(self.value_num2idx)

        return output_value.to(factored_tokens.device)


    def next_value_per_seq(self, seq, expr, value):
        # seq: Tensor of integers of shape [seq_len, n_factors]
        # expr: Tensor of integers of shape [seq_len, max_fan_in_deg * 2 - 1]
        # value: Tensor of integers of shape [seq_len]
        output_value = torch.zeros_like(value, requires_grad=False)
        for i, (token, e) in enumerate(zip(seq, expr)):
            next_value = self.next_value_per_token(token, e, value)
            output_value[i] = next_value
        return output_value # shape: [seq_len]

    def next_value_per_token(self, token, expr, value):
        '''
        token: Tensor of integers of shape [n_factors]
        dep: Tensor of integers of shape []
        expr: Tensor of integers of shape [max_fan_in_deg * 2 - 1]
        seq: Tensor of integers of shape [seq_len, n_factors]
        value: Tensor of integers of shape [seq_len]
        '''

        # case 1: the token is a VARIABLE on the right-hand side of an expression
        # NOTE: if a token is not on the right-hand side of an expression, it does not have an expression
        if token[self.SYNTAX_idx] == self.syntax_tok2idx['VALUE']:
            output_value = self.value_idx2num(token)
        elif token[self.SYNTAX_idx] == self.syntax_tok2idx['VARIABLE'] and not torch.all(expr == INVALID_NUM): # ensure the token is on the right-hand side of an expression
            # TO compute the value for the variable, we need to:

            # 1. extract the positions of its parents (on the even indices of e)
            parent_positions = expr[::2] # shape: [max_fan_in_deg]
            # kill the -1 values
            parent_positions = parent_positions[parent_positions != INVALID_NUM]

            # 2. extract the operators between the parents (on the odd indices of e)
            operators = expr[1::2] # shape: [max_fan_in_deg - 1]
            # kill the -1 values
            operators = operators[operators != INVALID_NUM]

            parent_values = value[parent_positions]

            output_value = parent_values[0]

            for i in range(len(operators)):
                output_value = self.apply_operation(self.operation_idx2tok[int(operators[i])], output_value, parent_values[i + 1])
        else:
            output_value = torch.nan
        return output_value

class FactoredTokenEmbedder(torch.nn.Module):

    def __init__(self, vocab_sizes, embed_dim, combine_op='sum'):
        super().__init__()

        self.vocab_sizes = vocab_sizes
        self.emb_dim = embed_dim
        self.combine_op = combine_op
        assert combine_op in ['sum', 'concat'], 'combine_op must be one of "sum" or "concat"'

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(vocab_size, embed_dim)
            for vocab_size in vocab_sizes
        ])

    def forward(self, factored_tokens):
        if self.combine_op == 'sum':
            return sum([
                emb(factored_tokens[:, :, i])
                for i, emb in enumerate(self.embeddings)
            ])
        elif self.combine_op == 'concat':
            return torch.cat([
                emb(factored_tokens[:, :, i])
                for i, emb in enumerate(self.embeddings)
            ], dim=-1)


class RecurrentTransformer(torch.nn.Module):

    def __init__(self, model_config):
        super().__init__()

        self.model_config = model_config


        self.token_embedder = FactoredTokenEmbedder(
            vocab_sizes=model_config.vocab_sizes,
            embed_dim=model_config['hidden_size'],
            combine_op='sum'
        )

        self.transformer = TransformerEncoder(model_config)

        self.output_layers = torch.nn.ModuleList([
            torch.nn.Linear(model_config.hidden_size, vocab_size)
            for vocab_size in self.model_config.vocab_sizes
        ])

    def forward(self, x, nointerm=False):
        if nointerm == False:
            embedded = self.token_embedder(x)
            transformer_output = self.transformer(embedded)
            preds = [
                output_layer(transformer_output)
                for output_layer in self.output_layers
            ]
            # shape: list of Tensor[batch_size, seq_len, factor_vocab_size] for factor in factors
            return preds
        else:
            transformer_output = self.transformer(x)
            preds = [
                output_layer(transformer_output)
                for output_layer in self.output_layers
            ]
            # shape: list of Tensor[batch_size, seq_len, factor_vocab_size] for factor in factors
            return transformer_output, preds


    def TF_forward(self, x):
        transformer_output = self.transformer(x)
        preds = [
            output_layer(transformer_output)
            for output_layer in self.output_layers
        ]
        # shape: list of Tensor[batch_size, seq_len, factor_vocab_size] for factor in factors
        return transformer_output, preds



class LitRecurrentTransformerModel(pl.LightningModule):
    def __init__(self, model_config, data_config, train_config, teacher_model=None):
        super().__init__()
        self.model = RecurrentTransformer(model_config)
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config
        self.factor_loss_scales = [1, 1, 1, 1, 0, 0] # TODO: FIXME: figure out how to set these scaling factors; may need to normalize according to vocab_size of each factor>

        # Important NOTE: There are several moving parts in the optimization procedure:
        # 1. Loop over depths, to train all depths in the batch
        # 2. Loop over factors, to train all factors at each depth
        # PyTorch Lightning does not directly support computing the backward pass for each of these inner loops separately
        # I would like to compute the backward pass for each separately and accumulate the gradients, then step the optimizer
        # So, I will need to implement this manually in the training loop
        self.automatic_optimization = False # disable automatic optimization to manually compute the backward pass for each depth and factor
        # Will need to do optimizer.zero_grad(), gradient accumulation, optimizer.step(), etc. manually
        # See: https://lightning.ai/docs/pytorch/stable/model/build_model_advanced.html

        self.criterion = torch.nn.CrossEntropyLoss()
        self.fast_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        # This is the maximum depth to train in the model.
        # If None, train all depths in batch.
        self.max_train_loop = getattr(train_config, 'max_train_loop', None)
        self.max_val_loop = getattr(train_config, 'max_val_loop', None)
        # NOTE / FIXME: we would like to change how this is done, making it more similar to the initial training runs

        self.teacher_model = teacher_model
        self.last_TchrPred_false_global_step = -1
        self.last_TchrPred_true_global_step = -1

        self.train_temperature = getattr(train_config, 'train_temperature', 0.0)
        self.val_temperature = getattr(train_config, 'val_temperature', 0.0)

        self.nointerm = getattr(train_config, 'nointerm', False)

        self.save_hyperparameters() # hyperparameters will be saved to checkpoint under key "hyper_parameters"

        self.last_step = 0

    def load_ckpt(self, checkpoint_path, strict=True, map_location='cuda'):
        if torch.cuda.is_available() and map_location == 'cuda':
            ckpt = torch.load(checkpoint_path, weights_only=False)
        else:
            ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            print(f"ckpt loaded on {map_location}")
        # the keys of the state_dict start with 'model._orig_mod.' due to the torch compiled model
        # let's remove this prefix
        state_dict = {k.replace('model._orig_mod.', ''): v for k, v in ckpt['state_dict'].items()}
        self.model.load_state_dict(state_dict, strict=strict)
        print(f'Loaded model from checkpoint: {checkpoint_path}')

    def forward(self, factored_tokens):
        return self.model(factored_tokens, self.nointerm)

    def SAE_forward(self, batch, log_prefix):
        # batch: Tensor of integers of shape [batch_size, n_depths, seq_len, n_factors]
        batch = self.reduce_batch(batch)
        batch, batch_label, batch_info = batch[..., :self.model_config.data_dim], batch[..., self.model_config.data_dim:self.model_config.data_dim + self.model_config.label_dim], batch[..., self.model_config.data_dim + self.model_config.label_dim:]

        # batch: shape: [batch_size, seq_len, n_factors]
        # batch_label: shape: [batch_size, seq_len, n_factors]
        # batch_info: shape: [batch_size, seq_len, max_fan_in_deg * 2]
        depths = batch_info[..., 0]

        max_depth_data = depths.max() + 1

        batch_ext, value_avail_pos, value_to_compute_pos = self.get_teacher_batch_with_depth(batch, batch_label, depths)
        # batch_ext shape: [n_depths + 1, batch_size, seq_len, n_factors]
        # value_avail_pos shape: [n_depths, batch_size, seq_len], which indicates the positions where the value is available after each loop

        factor_preds = self(batch_ext[:-1, ...].reshape(-1, batch.size(-2), batch.size(-1))) # list of Tensors of shape [n_depths * batch_size, seq_len, factor_vocab_size] for factor in factors
        # factor_decodes = self.generate_factor_decodes(factor_preds, temperature=self.train_temperature if log_prefix == 'train' else self.val_temperature) # list of Tensors of shape [n_depths * batch_size, seq_len] for factor in factors

        factor_preds = [factor_pred.reshape(max_depth_data, len(batch), -1, factor_pred.size(-1)) for factor_pred in factor_preds] # list of Tensors of shape [n_depths, batch_size, seq_len, factor_vocab_size] for factor in factors

        # return factor_preds, factor_decodes
        factor_decodes = self.generate_factor_decodes(factor_preds, temperature=self.train_temperature if log_prefix == 'train' else self.val_temperature) # list of Tensors of shape [n_depths, batch_size, seq_len] for factor in factors

        return factor_preds, factor_decodes

    def teacher_forward(self, factored_tokens, batch_info):
        return self.teacher_model.step(factored_tokens, batch_info)

    def training_step(self, batch, batch_idx):

        # total_train_loss = self.compute_batch_loss(batch, train=True, log_prefix='train', log=True, on_step=True, on_epoch=True)
        if self.nointerm:
            total_train_loss = self.compute_batch_loss_nointerm(batch, log_prefix='train', log=True, on_step=True, on_epoch=True)
        else:
            if self.train_config.get('use_teacher_pred_for_next_loop', False):
                if self.train_config.get('use_fast_batch_forward', False):
                    total_train_loss = self.compute_batch_loss_fast(batch, log_prefix='train', log=True, on_step=True, on_epoch=True)
                else:
                    total_train_loss = self.compute_batch_loss(batch, train=True, log_prefix='train', log=True, on_step=True, on_epoch=True, use_teacher_pred_for_next_loop=True, use_regret_for_correction=False, verbose=False)
            else:
                total_train_loss = self.compute_batch_loss(batch, train=True, log_prefix='train', log=True, on_step=True, on_epoch=True, use_teacher_pred_for_next_loop=False, use_regret_for_correction=False, verbose=False)



        # return total_train_loss

    def validation_step(self, batch, batch_idx):
        # return 0.0 ## FIXME: remove this line. This is for skipping the sanity check
        if self.nointerm:
            total_val_loss = self.compute_batch_loss_nointerm(batch, log_prefix='val', log=True, on_step=False, on_epoch=True)
        else:
            t = torch.rand(1).item() < 0.5
            if t:
                use_teacher_pred_for_next_loop = False
            else:
                use_teacher_pred_for_next_loop = True

            if self.last_TchrPred_false_global_step != self.global_step:
                verbose = True
                self.last_TchrPred_false_global_step = self.global_step
            elif self.last_TchrPred_true_global_step != self.global_step:
                verbose = True
                self.last_TchrPred_true_global_step = self.global_step
            else:
                verbose = False


            total_val_loss = self.compute_batch_loss(batch, train=False, log_prefix='val', log=True, on_step=False, on_epoch=True, use_teacher_pred_for_next_loop=use_teacher_pred_for_next_loop, use_regret_for_correction=False, verbose=False)
            # TODO: add evaluation of full non-teacher-forcing generation (i.e, start-to-finish computation through iteration)

            # NOTE / FIXME: commented this out for now to test (don't have required file for now)
            # # conduct a controlled experiment
            # for controlled_exp_key in ['var_0', 'var_1', 'var_2', 'rhs']:
            #     controlled_experiment_step(controlled_exp_key, self, self.model_config, logger=self.log, wandb_logger=self.logger)



        return total_val_loss


    def test_step(self, batch, batch_idx):
        if self.nointerm:
            total_test_loss = self.compute_batch_loss_nointerm(batch, log_prefix='test', log=True, on_step=False, on_epoch=True)
        else:
            total_test_loss = self.compute_batch_loss(batch, train=False, log_prefix='test', log=True, on_step=False, on_epoch=True)

        return total_test_loss

    def prepare_label(self, teacher_value_preds, depth_batch, factor_decodes):
        with torch.no_grad():

            label = copy.deepcopy(depth_batch)

            # change the value of the label if teacher_value_preds is not -1
            valid_teacher_preds = teacher_value_preds != INVALID_NUM

            # change label in place
            value_label = label[:, :, self.model_config.factors.index('VALUE')]
            # find the positions of variables
            variable_positions = depth_batch[:, :, self.model_config.factors.index('SYNTAX')].eq(self.teacher_model.syntax_tok2idx['VARIABLE'])
            # set the value of the variables to be EMPTY
            value_label[variable_positions] = self.teacher_model.value_tok2idx['EMPTY']
            # set the value of the valid teacher predictions to be the teacher predictions
            value_label[valid_teacher_preds] = teacher_value_preds[valid_teacher_preds].to(value_label.dtype)

            # # find all the positions where value_label does not match the value in factor_preds
            # mismatch = torch.logical_not(label[..., :4].eq(torch.stack(factor_decodes[0:4], dim=-1))).any(dim=-1)
            # # check only the first 4 factors

            # # set the regret token for the mismatched positions to TRUE, while others to be FALSE
            # regret_label = label[:, :, self.model_config.factors.index('REGRET')]
            # regret_label.fill_(self.teacher_model.regret_tok2idx['FALSE'])
            # regret_label[mismatch] = self.teacher_model.regret_tok2idx['TRUE']

        return label

    def get_eq_rhs_position(batch_info):

        d = batch_info.shape[-1]
        # find all the position where batch_info.sum(dim=-1) is not -d

        return batch_info.sum(dim=-1).ne(-d)

    def randomly_corrupt_values(self, batch_depthwise_inputs, value_avail_pos):
        # this function randomly corrupts the values in the batch_depthwise_inputs

        if self.train_config.recorrection_args.corruption_strategy == 'last':
            # value_avail_pos is a boolean mask with True at every value position that was computed already
            # for the 'last' strategy, we only want to corrupt the values computed at the previous iteration
            # we find this mask by computing "value_avail_pos[iter] and not value_avail_pos[iter-1]""
            candidates = torch.concat([value_avail_pos[0].unsqueeze(0), torch.logical_and(value_avail_pos[:-1].logical_not(), value_avail_pos[1:])], dim=0)
        else:
            candidates = value_avail_pos

        # choose a random set of tokens to corrupt
        random_corruption = torch.rand(candidates.size(), device=candidates.device) < self.train_config.recorrection_args.tokenwise_corruption_prob
        corruption_mask = torch.logical_and(random_corruption, candidates)

        # randomly generate a value to replace the corrupted tokens
        value_factor_index = self.teacher_model.factored_tokenizer.factors.index('VALUE')
        value_vocab_size = self.teacher_model.factored_tokenizer.vocab_sizes[value_factor_index]
        candidate_corruption_values = torch.randint(size=corruption_mask.size(), high=value_vocab_size, low=0, device=corruption_mask.device) # this includes random values at all position

        # replace values
        corrupted_batch_depthwise_inputs = batch_depthwise_inputs.clone()
        corrupted_batch_depthwise_inputs[..., value_factor_index] = torch.where(corruption_mask, candidate_corruption_values, batch_depthwise_inputs[..., value_factor_index])

        return corrupted_batch_depthwise_inputs, corruption_mask

    def reduce_batch(self, batch):
        # find all the positions that only have padding tokens
        batch_view = batch[..., :self.model_config.data_dim].eq(self.teacher_model.factored_tokenizer.syntax_tok2idx['<PAD>'])
        # find the positions in the second dimension that all the values are True
        batch_view = batch_view.permute(1, 0, 2).reshape(batch.size(1), -1).all(dim=-1)
        return batch[:, torch.logical_not(batch_view), :]

    def get_padding_mask(self, depth_batch):
        return depth_batch[..., self.teacher_model.factored_tokenizer.factors.index('SYNTAX')].eq(self.teacher_model.factored_tokenizer.syntax_tok2idx['<PAD>'])

    def get_teacher_batch_with_depth(self, batch, batch_label, depths):
        with torch.no_grad():
            max_depth_data = depths.max() + 1
            batch_ext = batch.unsqueeze(0).repeat(max_depth_data, 1, 1, 1) # shape: [n_depths, batch_size, seq_len, n_factors]
            depth_ext = depths.unsqueeze(0).repeat(max_depth_data, 1, 1) # shape: [n_depths, batch_size, seq_len]
            depth_target = torch.arange(max_depth_data, device=batch.device).unsqueeze(-1).unsqueeze(-1).repeat(1, batch.size(0), batch.size(1)) # shape: [n_depths, batch_size, seq_len]
            batch_label_ext = batch_label.unsqueeze(0).repeat(max_depth_data, 1, 1, 1) # shape: [n_depths, batch_size, seq_len, n_factors]

            value_avail_pos = torch.logical_and(depth_ext.le(depth_target), depth_ext.ne(INVALID_NUM))
            value_to_compute_pos = torch.logical_and(depth_ext.eq(depth_target), depth_ext.ne(INVALID_NUM))
            batch_ext[..., self.model_config.factors.index('VALUE')][value_avail_pos] = batch_label_ext[..., self.model_config.factors.index('VALUE')][value_avail_pos]
        return torch.cat([batch.unsqueeze(0), batch_ext], dim=0), value_avail_pos, value_to_compute_pos

    def compute_batch_loss_fast(self,
                                batch,
                                log_prefix=None,
                                log=True,
                                use_pos_focus=False,
                                **log_kwargs):
        """This method always uses the teacher predictions for the next loop for parallelization. No for loop in the forward pass.

        Args:
            batch (_type_): _description_
            log_prefix (_type_, optional): _description_. Defaults to None.
            log (bool, optional): _description_. Defaults to True.
            use_pos_focus (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # batch: Tensor of integers of shape [batch_size, n_depths, seq_len, n_factors]
        batch = self.reduce_batch(copy.deepcopy(batch))
        batch, batch_label, batch_info = batch[..., :self.model_config.data_dim], batch[..., self.model_config.data_dim:self.model_config.data_dim + self.model_config.label_dim], batch[..., self.model_config.data_dim + self.model_config.label_dim:]

        # batch: shape: [batch_size, seq_len, n_factors]
        # batch_label: shape: [batch_size, seq_len, n_factors]
        # batch_info: shape: [batch_size, seq_len, max_fan_in_deg * 2]
        depths = batch_info[..., 0]

        train=log_prefix == 'train'

        if train:
            opt = self.optimizers()
            opt.zero_grad()

        total_loss = 0

        max_depth_data = depths.max() + 1

        batch_ext, value_avail_pos, value_to_compute_pos = self.get_teacher_batch_with_depth(batch, batch_label, depths)
        # batch_ext shape: [n_depths + 1, batch_size, seq_len, n_factors]
        # value_avail_pos shape: [n_depths, batch_size, seq_len], which indicates the positions where the value is available after each loop

        batch_depthwise_inputs = batch_ext[:-1, ...]

        # if recorrection is enabled in train config, values in the input are randomly corrupted to help train the model to recorrect its mistakes
        if hasattr(self.train_config, 'recorrection_args'):
            corruption_coinflip = np.random.random() < self.train_config.recorrection_args.corruption_prob
        
            if train and self.train_config.recorrection_args.enable and corruption_coinflip:
                # randomly corrupt the values in the batch_depthwise_inputs
                batch_depthwise_inputs, corruption_mask = self.randomly_corrupt_values(batch_depthwise_inputs, value_avail_pos)

        batch_depthwise_inputs = batch_depthwise_inputs.reshape(-1, batch.size(-2), batch.size(-1)) # shape: [n_depths * batch_size, seq_len, n_factors]

        factor_preds = self(batch_depthwise_inputs) # list of Tensors of shape [n_depths * batch_size, seq_len, factor_vocab_size] for factor in factors
        factor_decodes = self.generate_factor_decodes(factor_preds, temperature=self.train_temperature if log_prefix == 'train' else self.val_temperature) # list of Tensors of shape [n_depths * batch_size, seq_len] for factor in factors

        factor_labels = batch_ext[1:, ...] # shape: [n_depths, batch_size, seq_len, n_factors]

        total_loss = 0

        for factor, factor_pred in enumerate(factor_preds):
            # factor_pred: Tensor of shape [n_depths * batch_size, seq_len, factor_vocab_size]
            factor_pred = factor_pred.reshape(max_depth_data, len(batch), -1, factor_pred.size(-1)) # shape: [n_depths, batch_size, seq_len, factor_vocab_size]
            factor_decode = factor_decodes[factor].reshape(max_depth_data, len(batch), -1).detach() # shape: [n_depths, batch_size, seq_len]

            factor_name = self.model_config.factors[factor].lower()

            factor_pred_flattened = factor_pred.view(-1, factor_pred.size(-1))  # shape: [n_depths * batch_size * seq_len, factor_vocab_size]

            factor_label_flatterned = factor_labels[..., factor].reshape(-1) # shape: [n_depths * batch_size * seq_len]

            factor_at_depth_loss = self.fast_criterion(factor_pred_flattened, factor_label_flatterned).reshape(max_depth_data, len(batch), -1)
            # shape: [n_depths, batch_size, seq_len]

            # if log_prefix == 'train':
            #     self.manual_backward(factor_at_depth_loss.mean() * self.factor_loss_scales[factor], retain_graph=factor < len(factor_preds) - 1)

            # NOTE: we let the loss pay attention to value_to_compute_pos
            if factor == self.model_config.factors.index('VALUE') and use_pos_focus:
                focus_ratio = value_to_compute_pos.float().mean()
                factor_loss_with_focus = factor_at_depth_loss[value_to_compute_pos].mean() + (factor_at_depth_loss[torch.logical_not(value_to_compute_pos)] * focus_ratio).mean()
                total_loss += factor_loss_with_focus * self.factor_loss_scales[factor]
            else:
                total_loss += factor_at_depth_loss.mean() * self.factor_loss_scales[factor]
            # factor_decode = factor_pred.argmax(dim=-1).detach() # shape: [n_depths, batch_size, seq_len]

            if log:
                # the following is for logging purpose
                correct = factor_decode.eq(factor_labels[..., factor]) # shape: [n_depths, batch_size, seq_len]

                for loop in range(max_depth_data):

                    # log the cross-entropy loss for each factor at each loop
                    self.log(f'{log_prefix}_{factor_name}_loss_TchrPred_True_RegCorr_False/loop_{loop}',
                             factor_at_depth_loss[loop].mean(),
                             **log_kwargs)

                    # log the per token accuracy for each factor at each loop
                    self.log(f'{log_prefix}_{factor_name}_per_token_acc_TchrPred_True_RegCorr_False/loop_{loop}',
                             correct[loop].float().mean(),
                             **log_kwargs)

                    # log the full sequence accuracy for each factor at each loop
                    self.log(f'{log_prefix}_{factor_name}_full_seq_acc_TchrPred_True_RegCorr_False/loop_{loop}',
                             correct[loop].all(dim=-1).float().mean(),
                             **log_kwargs)

                    # # log the MRR for each factor at each loop
                    # self.log(f'{log_prefix}_{factor_name}_MRR_TchrPred_True_RegCorr_False/loop_{loop}',
                    #             MRR_fn(factor_pred[loop].view(-1, factor_pred.size(-1)), factor_labels[loop, :, :, factor].view(-1)),
                    #             **log_kwargs)

                    if factor == self.model_config.factors.index('VALUE'):
                        # log the ratio of newly solved variables compared to the teacher predictions
                        correct = factor_labels[loop, :, :, factor][value_to_compute_pos[loop]].eq(factor_decode[loop, ...][value_to_compute_pos[loop]])
                        correct_ratio = correct.float().mean()
                        self.log(f'{log_prefix}_ratio_solved_dep_eq_loop_TchrPred_True_RegCorr_False/loop_{loop}', correct_ratio, **log_kwargs)

                        # log the ratio of solved variables compared to the teacher predictions
                        correct = factor_labels[loop, :, :, factor][value_avail_pos[loop]].eq(factor_decode[loop, ...][value_avail_pos[loop]])
                        correct_ratio = correct.float().mean()
                        self.log(f'{log_prefix}_ratio_solved_dep_le_loop_TchrPred_True_RegCorr_False/loop_{loop}', correct_ratio, **log_kwargs)

                        if hasattr(self.train_config, 'recorrection_args'):
                            if train and self.train_config.recorrection_args.enable and corruption_coinflip:
                                # log the ratio of corrupted values that are not solved
                                correct = factor_labels[loop, :, :, factor][corruption_mask[loop]].eq(factor_decode[loop, ...][corruption_mask[loop]])
                                correct_ratio = correct.float().mean()
                                self.log(f'{log_prefix}_percent_recorrected/loop_{loop}', correct_ratio, **log_kwargs)

        if log:
            # log the total loss
            self.log(f'total_loss/{log_prefix}', total_loss, **log_kwargs, prog_bar=True)


        if log_prefix == 'train':
            total_loss.backward()
            opt.step()

        return total_loss

    def decompose_batch(self, batch):
        depth_batch, batch_label, batch_info = batch[..., :self.model_config.data_dim], batch[..., self.model_config.data_dim:self.model_config.data_dim + self.model_config.label_dim], batch[..., self.model_config.data_dim + self.model_config.label_dim:]
        return depth_batch, batch_label, batch_info

    def loop_forward(self, batch, num_loops, temperature=0.0):
        batch = self.reduce_batch(copy.deepcopy(batch))
        depth_batch, batch_label, batch_info = batch[..., :self.model_config.data_dim], batch[..., self.model_config.data_dim:self.model_config.data_dim + self.model_config.label_dim], batch[..., self.model_config.data_dim + self.model_config.label_dim:]

        for loop in range(num_loops):
            factor_preds = self(depth_batch)
            factor_decodes = self.generate_factor_decodes(factor_preds, temperature=temperature)

            depth_batch = torch.stack(factor_decodes, dim=-1)
            # disable the regret and modification factors
            depth_batch[..., self.model_config.factors.index('REGRET')] = self.teacher_model.regret_tok2idx['FALSE']
            depth_batch[..., self.model_config.factors.index('MODIFICATION')] = self.teacher_model.modification_tok2idx['FALSE']


    def generate_factor_decodes(self, factor_preds, temperature=0.0):
        assert temperature >= 0.0, 'temperature must be non-negative'
        if temperature == 0.0: # greedy decoding
            return [factor_pred.argmax(dim=-1) for factor_pred in factor_preds]
        else:
            return [torch.nn.functional.gumbel_softmax(factor_pred, tau=temperature, hard=True).argmax(dim=-1) for factor_pred in factor_preds]

    def compute_batch_loss(self,
                           batch,
                           train=False,
                           log_prefix=None,
                           log=True,
                           use_teacher_pred_for_next_loop=True,
                           use_regret_for_correction=False,
                           verbose=False,
                           **log_kwargs):
        # batch: Tensor of integers of shape [batch_size, n_depths, seq_len, n_factors]

        # Compute the loss for each depth.
        # The loss for each depth is the sum of the losses for each factor at that depth.
        # The total loss is the sum of the losses for all depths.
        # If train=True, accumulate gradients for all depths and factors in this batch and step the optimizer.

        # NOTE: when training, we'd like to accumulate gradients over all depths and factors in the batch
        # In a particular depth, we would like to accumulate gradients over all factors
        # Since most of the computation is shared across factors at a depth, we don't need to recompute it for each factor
        # By default however, loss.backward() will delete intermediary results, assuming they won't be needed anymore (which is true in most cases)
        # For us, they are needed because we need to accumulate gradients over all factors at a depth
        # To keep intermediate results, we can use loss.backward(retain_graph=True) to keep the computation graph for the next backward pass (retain_graph=False for last factor)
        # See: https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795
        batch = self.reduce_batch(copy.deepcopy(batch))

        batch, batch_label, batch_info = batch[..., :self.model_config.data_dim], batch[..., self.model_config.data_dim:self.model_config.data_dim + self.model_config.label_dim], batch[..., self.model_config.data_dim + self.model_config.label_dim:]

        if use_teacher_pred_for_next_loop is None:
            use_teacher_pred_for_next_loop = self.train_config.use_teacher_pred_for_next_loop

        # rhs_positions = self.get_eq_rhs_position(batch_info)
        depths = batch_info[..., 0]

        if train:
            opt = self.optimizers()
            opt.zero_grad()
            # self.optimizer.zero_grad()

        total_loss = 0
        # batch = batch.transpose(0, 1)  # shape: [n_depths, batch_size, seq_len, n_factors]

        max_depth_data = depths.max() + 1

        batch_ext, value_avail_pos, value_to_compute_pos = self.get_teacher_batch_with_depth(batch, batch_label, depths)
        # batch_ext shape: [n_depths + 1, batch_size, seq_len, n_factors]
        # value_avail_pos shape: [n_depths, batch_size, seq_len], which indicates the positions where the value is available after each loop
        # value_to_compute_pos shape: [n_depths, batch_size, seq_len], which indicates the positions where the value is to be computed after each loop

        if hasattr(self.train_config, 'recorrection_args'):
            if train and self.train_config.recorrection_args.enable:
                raise NotImplementedError('recorrection-based training is not implemented for compute_batch_loss, but is implemented for compute_batch_loss_fast')

        depth_batch = copy.deepcopy(batch)

        for loop in range(max_depth_data):
            # depth_batch: Tensor of integers of shape [batch_size, seq_len, n_factors]
            # next_depth_factor_label: Tensor of integers of shape [batch_size, seq_len, n_factors]

            # teacher_value_preds = self.teacher_forward(depth_batch, batch_info)
            label = batch_ext[loop+1, ...]

            factor_preds = self(depth_batch)

            factor_decodes = self.generate_factor_decodes(factor_preds, temperature=self.train_temperature if log_prefix == 'train' else self.val_temperature)

            # label = self.prepare_label(teacher_value_preds, depth_batch, factor_decodes)

            # Compute the loss for each factor at this depth.
            depth_loss = 0
            full_seq_acc_for_all_factors = []
            for factor, factor_pred in enumerate(factor_preds):
                # factor_pred: Tensor of shape [batch_size, seq_len, factor_vocab_size]
                # next_depth_factor_label: Tensor of shape [batch_size, seq_len]
                factor_name = self.model_config.factors[factor].lower()

                factor_pred_flattened = factor_pred.view(-1, factor_pred.size(-1)) # shape: [batch_size * seq_len, factor_vocab_size]

                # next_depth_factor_label = next_depth_batch[:, :, factor] # shape: [batch_size, seq_len]
                next_depth_factor_label = label[:, :, factor] # shape: [batch_size, seq_len]

                next_depth_factor_label_flattened = next_depth_factor_label.contiguous().view(-1) # shape: [batch_size * seq_len]

                factor_at_depth_loss = self.criterion(factor_pred_flattened, next_depth_factor_label_flattened)

                if log:
                    self.log(f'{log_prefix}_{factor_name}_loss_TchrPred_{use_teacher_pred_for_next_loop}_RegCorr_{use_regret_for_correction}/loop_{loop}', factor_at_depth_loss, **log_kwargs)

                if loop == 0 and factor == self.model_config.factors.index('VALUE'):
                    scaled_factor_at_depth_loss = self.factor_loss_scales[factor] * factor_at_depth_loss * 2 # scales up the first step's loss to speed up training
                else:
                    scaled_factor_at_depth_loss = self.factor_loss_scales[factor] * factor_at_depth_loss

                # If training, accumulate gradients for this factor at this depth
                if train:
                    # set retain_graph=True for all factors except the last one to keep intermediary results for the next factor's backward pass
                    self.manual_backward(scaled_factor_at_depth_loss, retain_graph=factor < len(factor_preds) - 1)

                depth_loss += scaled_factor_at_depth_loss

                # compute accuracy
                correct = factor_pred.argmax(dim=-1) == next_depth_factor_label # shape: [batch_size, seq_len]
                per_token_acc = correct.float().mean()
                full_seq_acc = correct.all(dim=-1).float().mean()
                full_seq_acc_for_all_factors.append(full_seq_acc)
                
                if log:
                    self.log(f'{log_prefix}_{factor_name}_per_token_acc_TchrPred_{use_teacher_pred_for_next_loop}_RegCorr_{use_regret_for_correction}/loop_{loop}', per_token_acc, **log_kwargs)
                    self.log(f'{log_prefix}_{factor_name}_full_seq_acc_TchrPred_{use_teacher_pred_for_next_loop}_RegCorr_{use_regret_for_correction}/loop_{loop}', full_seq_acc, **log_kwargs)

                    # compute MRR
                    mrr = MRR_fn(factor_pred.view(-1, factor_pred.size(-1)), next_depth_factor_label.contiguous().view(-1))
                    self.log(f'{log_prefix}_{factor_name}_MRR_TchrPred_{use_teacher_pred_for_next_loop}_RegCorr_{use_regret_for_correction}/loop_{loop}', mrr, **log_kwargs)

            # delete the last two factors from the list of factors
            full_seq_acc_for_all_factors = full_seq_acc_for_all_factors[:-2]
            # apply logical AND element-wise
            full_seq_acc_for_all_factors = torch.all(torch.stack(full_seq_acc_for_all_factors, dim=0), dim=0)
            
            if log:
                self.log(f'{log_prefix}_full_seq_acc_TchrPred_{use_teacher_pred_for_next_loop}_RegCorr_{use_regret_for_correction}/loop_{loop}', full_seq_acc_for_all_factors.float().mean(), **log_kwargs)
                self.log(f'{log_prefix}_loss_TchrPred_{use_teacher_pred_for_next_loop}_RegCorr_{use_regret_for_correction}/loop_{loop}', depth_loss, **log_kwargs)

                # log the ratio of solved variables compared to the teacher predictions

                # find the positions where the depth is less than or equal to the loop and is the rhs of an equation
                dep_less_eq_loop_pos = torch.logical_and(depths.le(loop), depths.ne(INVALID_NUM))
                # get the correct values for all the positions
                value_batch_label = batch_label[..., self.model_config.factors.index('VALUE')]
                value_factor_preds = factor_decodes[self.model_config.factors.index('VALUE')] # shape: [batch_size, seq_len]

                value_batch_label_flattened = value_batch_label[dep_less_eq_loop_pos].flatten()
                value_factor_preds_flattened = value_factor_preds[dep_less_eq_loop_pos].flatten()

                # compute the ratio of correct values
                correct_values = value_batch_label_flattened.eq(value_factor_preds_flattened)
                correct_values_ratio = correct_values.float().mean()

                self.log(f'{log_prefix}_ratio_solved_dep_le_loop_TchrPred_{use_teacher_pred_for_next_loop}_RegCorr_{use_regret_for_correction}/loop_{loop}', correct_values_ratio, **log_kwargs)

                dep_eq_loop_pos = torch.logical_and(depths.eq(loop), depths.ne(INVALID_NUM))
                value_batch_label_flattened = value_batch_label[dep_eq_loop_pos].flatten()
                value_factor_preds_flattened = value_factor_preds[dep_eq_loop_pos].flatten()

                # compute the ratio of correct values
                correct_values = value_batch_label_flattened.eq(value_factor_preds_flattened)
                correct_values_ratio = correct_values.float().mean()

                self.log(f'{log_prefix}_ratio_solved_dep_eq_loop_TchrPred_{use_teacher_pred_for_next_loop}_RegCorr_{use_regret_for_correction}/loop_{loop}', correct_values_ratio, **log_kwargs)

            # update the depth_batch with the predictions
            if use_teacher_pred_for_next_loop:
                depth_batch = label
                # disable the regret and modification tokens
                depth_batch[..., self.model_config.factors.index('REGRET')] = self.teacher_model.regret_tok2idx['FALSE']
                depth_batch[..., self.model_config.factors.index('MODIFICATION')] = self.teacher_model.modification_tok2idx['FALSE']
            else:
                if use_regret_for_correction:
                    depth_batch_new = copy.deepcopy(depth_batch)
                    regret_pos = factor_decodes[self.model_config.factors.index('REGRET')].eq(self.teacher_model.regret_tok2idx['TRUE'])
                    for factor, factor_decode in enumerate(factor_decodes):
                        depth_batch_new_factor = depth_batch_new[..., factor]
                        depth_batch_new_factor = factor_decode
                        # NOTE: if regret is TRUE, we reuse the previous input
                        depth_batch_new_factor[regret_pos] = depth_batch[..., factor][regret_pos].clone()
                    # reset the regret factor
                    depth_batch_new[..., self.model_config.factors.index('REGRET')].fill_(self.teacher_model.regret_tok2idx['FALSE'])
                    depth_batch_new[..., self.model_config.factors.index('REGRET')][regret_pos] = self.teacher_model.regret_tok2idx['TRUE']
                    depth_batch = depth_batch_new
                else:
                    depth_batch = torch.stack(factor_decodes, dim=-1)
                    # disable the regret and modification factors
                    depth_batch[..., self.model_config.factors.index('REGRET')] = self.teacher_model.regret_tok2idx['FALSE']
                    depth_batch[..., self.model_config.factors.index('MODIFICATION')] = self.teacher_model.modification_tok2idx['FALSE']

            total_loss += depth_loss

            if verbose and log_prefix == 'val':
                log_file_path = os.path.join(current_dir, 'checkpoints', f'{self.train_config.experiment_group}-{self.train_config.experiment_run_name}', f'logs_step_{self.global_step}_TchrPred_{use_teacher_pred_for_next_loop}.txt')

                # create the directory if it does not exist
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                # create the log file
                with open(log_file_path, 'a') as f:
                    for factor_idx, factor in enumerate(self.model_config.factors):
                        if factor in ['SYNTAX', 'VARIABLE', 'OPERATION', 'VALUE']:
                            f.write(f'label, loop:{loop}, factor: {factor}, step: {self.global_step}, TchrPred: {use_teacher_pred_for_next_loop}\n {label[0, :, factor_idx].detach().cpu().tolist()}\n')
                            f.write(f'factor_decodes, loop:{loop}, factor: {factor}, step: {self.global_step}, TchrPred: {use_teacher_pred_for_next_loop}\n {factor_decodes[factor_idx][0, :].detach().cpu().tolist()}\n')
                            # f.write(f'batch_label, loop:{loop}, factor: {factor}, step: {self.global_step}, TchrPred: {use_teacher_pred_for_next_loop}\n {batch_label[0, :, factor_idx].detach().cpu().tolist()}\n')
                            f.write('\n')
            

            # Break if we have reached the maximum depth to train
            if train and self.max_train_loop is not None and loop == self.max_train_loop - 1:
                break
            if log_prefix == 'val' and self.max_val_loop is not None and loop == self.max_val_loop - 1:
                break
        # check one more thing: proportion of correct values in each depth at the output
        if log:
            for dep in range(depths.max()+1):
                pos_for_dep = depths.eq(dep)
                value_batch_label = batch_label[..., self.model_config.factors.index('VALUE')][pos_for_dep]
                value_factor_decode = factor_decodes[self.model_config.factors.index('VALUE')][pos_for_dep]
                correct_values = value_batch_label.eq(value_factor_decode)
                correct_values_ratio = correct_values.float().mean()
                self.log(f'{log_prefix}_rhs_value_acc_TchrPred_{use_teacher_pred_for_next_loop}_RegCorr_{use_regret_for_correction}/dep_{dep}', correct_values_ratio, **log_kwargs)
        # If training, step the optimizer on accumulated gradients (for all depths and factors in this batch)
        if train:
            opt.step()
            # self.optimizer_step()

        if log:
            self.log(f'total_loss_TchrPred_{use_teacher_pred_for_next_loop}_RegCorr_{use_regret_for_correction}/{log_prefix}', total_loss, **log_kwargs, prog_bar=True)
            self.log(f'total_loss/{log_prefix}', total_loss, **log_kwargs, prog_bar=True)
        
        return total_loss


    def compute_batch_loss_nointerm(self,
                                   batch,
                                   log_prefix=None,
                                   log=True,
                                   **log_kwargs):
        batch = self.reduce_batch(copy.deepcopy(batch))

        batch, batch_label, batch_info = batch[..., :self.model_config.data_dim], batch[..., self.model_config.data_dim:self.model_config.data_dim + self.model_config.label_dim], batch[..., self.model_config.data_dim + self.model_config.label_dim:]

        depths = batch_info[..., 0]
        train=log_prefix == 'train'
        if train:
            opt = self.optimizers()
            opt.zero_grad()
        total_loss = 0
        max_depth_data = depths.max() + 1
        batch_ext, value_avail_pos, value_to_compute_pos = self.get_teacher_batch_with_depth(batch, batch_label, depths)

        depth_batch = self.model.token_embedder(batch)

        for loop in range(max_depth_data):

            label = batch_ext[loop + 1, ...]

            depth_batch, factor_preds = self.model.TF_forward(depth_batch)

            factor_decodes = self.generate_factor_decodes(factor_preds, temperature=self.train_temperature if log_prefix == 'train' else self.val_temperature)

            depth_loss = 0

            full_factor_correct = None

            for factor, factor_pred in enumerate(factor_preds):
                # factor_pred: Tensor of shape [batch_size, seq_len, factor_vocab_size]
                # next_depth_factor_label: Tensor of shape [batch_size, seq_len]
                factor_name = self.model_config.factors[factor].lower()

                factor_pred_flattened = factor_pred.view(-1, factor_pred.size(-1)) # shape: [batch_size * seq_len, factor_vocab_size]

                # next_depth_factor_label = next_depth_batch[:, :, factor] # shape: [batch_size, seq_len]
                next_depth_factor_label = label[:, :, factor] # shape: [batch_size, seq_len]

                next_depth_factor_label_flattened = next_depth_factor_label.contiguous().view(-1) # shape: [batch_size * seq_len]

                factor_at_depth_loss = self.criterion(factor_pred_flattened, next_depth_factor_label_flattened)

                if log:
                    self.log(f'{log_prefix}_{factor_name}_loss_nointerm/loop_{loop}', factor_at_depth_loss, **log_kwargs)

                scaled_factor_at_depth_loss = self.factor_loss_scales[factor] * factor_at_depth_loss

                # If training, accumulate gradients for this factor at this depth
                # if train:
                #     # set retain_graph=True for all factors except the last one to keep intermediary results for the next factor's backward pass
                #     self.manual_backward(scaled_factor_at_depth_loss, retain_graph=factor < len(factor_preds) - 1)

                depth_loss += scaled_factor_at_depth_loss

                # compute accuracy
                correct = factor_pred.argmax(dim=-1) == next_depth_factor_label
                per_token_acc = correct.float().mean()
                full_seq_acc = correct.all(dim=-1).float().mean()

                if factor != self.model_config.factors.index('REGRET') and factor != self.model_config.factors.index('MODIFICATION'):
                    if full_factor_correct is None:
                        full_factor_correct = correct
                    else:
                        full_factor_correct = torch.logical_and(full_factor_correct, correct)

                if log:
                    self.log(f'{log_prefix}_{factor_name}_per_token_acc_nointerm/loop_{loop}', per_token_acc, **log_kwargs)
                    self.log(f'{log_prefix}_{factor_name}_full_seq_acc_nointerm/loop_{loop}', full_seq_acc, **log_kwargs)

                    # compute MRR
                    mrr = MRR_fn(factor_pred.view(-1, factor_pred.size(-1)), next_depth_factor_label.contiguous().view(-1))
                    self.log(f'{log_prefix}_{factor_name}_MRR_nointerm/loop_{loop}', mrr, **log_kwargs)

            if log:
                self.log(f'{log_prefix}_loss_nointerm/loop_{loop}', depth_loss, **log_kwargs)

                # log the ratio of solved variables compared to the teacher predictions

                # find the positions where the depth is less than or equal to the loop and is the rhs of an equation
                dep_less_eq_loop_pos = torch.logical_and(depths.le(loop), depths.ne(INVALID_NUM))
                # get the correct values for all the positions
                value_batch_label = batch_label[..., self.model_config.factors.index('VALUE')]
                value_factor_preds = factor_decodes[self.model_config.factors.index('VALUE')] # shape: [batch_size, seq_len]

                value_batch_label_flattened = value_batch_label[dep_less_eq_loop_pos].flatten()
                value_factor_preds_flattened = value_factor_preds[dep_less_eq_loop_pos].flatten()

                # compute the ratio of correct values
                correct_values = value_batch_label_flattened.eq(value_factor_preds_flattened)
                correct_values_ratio = correct_values.float().mean()

                self.log(f'{log_prefix}_ratio_solved_dep_le_loop_nointerm/loop_{loop}', correct_values_ratio, **log_kwargs)

                dep_eq_loop_pos = torch.logical_and(depths.eq(loop), depths.ne(INVALID_NUM))
                value_batch_label_flattened = value_batch_label[dep_eq_loop_pos].flatten()
                value_factor_preds_flattened = value_factor_preds[dep_eq_loop_pos].flatten()

                # compute the ratio of correct values
                correct_values = value_batch_label_flattened.eq(value_factor_preds_flattened)
                correct_values_ratio = correct_values.float().mean()

                self.log(f'{log_prefix}_ratio_solved_dep_eq_loop_nointerm/loop_{loop}', correct_values_ratio, **log_kwargs)

                # log the full_factor_correct accuracy
                full_factor_correct_ratio = full_factor_correct.all(dim=-1).float().mean()
                print(f"loop_{loop}, full_factor_correct_ratio: {full_factor_correct_ratio}")

            total_loss += depth_loss

            # Break if we have reached the maximum depth to train
            if train and self.max_train_loop is not None and loop == self.max_train_loop - 1:
                break
            if log_prefix == 'val' and self.max_val_loop is not None and loop == self.max_val_loop - 1:
                break

        # check one more thing: proportion of correct values in each depth at the output
        if log:
            for dep in range(depths.max()+1):
                pos_for_dep = depths.eq(dep)
                value_batch_label = batch_label[..., self.model_config.factors.index('VALUE')][pos_for_dep]
                value_factor_decode = factor_decodes[self.model_config.factors.index('VALUE')][pos_for_dep]
                correct_values = value_batch_label.eq(value_factor_decode)
                correct_values_ratio = correct_values.float().mean()
                self.log(f'{log_prefix}_rhs_value_acc_nointerm/dep_{dep}', correct_values_ratio, **log_kwargs)

        # If training, step the optimizer on accumulated gradients (for all depths and factors in this batch)
        if train:
            self.manual_backward(total_loss)
            opt.step()


        if log:
            # self.log(f'total_loss_nointerm/{log_prefix}', total_loss, **log_kwargs, prog_bar=True)
            self.log(f'total_loss/{log_prefix}', total_loss, **log_kwargs, prog_bar=True)
        return total_loss


    def configure_optimizers(self):
        # Configure the optimizer.
        optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
            'Shampoo': Shampoo,
            'signSGD': signSGD,
        }

        optimizer_name = self.train_config.optimizer
        if optimizer_name not in optimizer_dict.keys():
            raise ValueError(f"Optimizer {optimizer_name} is not implemented!")
        else:
            optimizer = optimizer_dict[optimizer_name](
                self.parameters(),
                **self.train_config[f'{optimizer_name}_optimizer_config']
            )

        # Configure the learning rate scheduler.
        if self.train_config.lr_scheduler == "cosine":
            cosine_scheduler_config = self.train_config.cosine_scheduler_config
            scheduler = CosineAnnealingWarmup(
                optimizer=optimizer,
                warmup_steps=cosine_scheduler_config.warmup_steps,
                learning_rate=self.train_config.learning_rate,
                min_lr=cosine_scheduler_config.min_lr,
                lr_decay_steps=cosine_scheduler_config.lr_decay_steps,
            )
        elif self.train_config.lr_scheduler == "step":
            StepLR_config = self.train_config.StepLR_scheduler_config
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=StepLR_config.step_size,
                gamma=StepLR_config.gamma,
            )
        else:
            # use no scheduler
            scheduler = None
        if scheduler is not None:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer

    def lr_scheduler_step(
            self,
            scheduler: LRSchedulerTypeUnion,
            metric: Any,
    ) -> None:
        scheduler.step()

    # def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer):
    #     """
    #     This function is called before the optimizer step.
    #     You can override this function to do something before the optimizer step.

    #     Args:
    #         optimizer (torch.optim.Optimizer): the optimizer
    #     """
    #     norms = lightning.pytorch.utilities.grad_norm(self.model, norm_type=2)
    #     self.log_dict(norms)

def get_experiment_name(model_config, data_config, train_config):
    # Format:
    # Group: Data Config - Model Config
    # Name: Seed + Date-Time
    data_str = f'Nodes{data_config.dag_config.num_nodes}-' + '_'.join(data_config.dag_config.func_vocab)
    model_str = f'L{model_config.num_layers}H{model_config.num_heads}D{model_config.hidden_size}_{model_config.pos_enc_type}'


    if train_config.get('recorrection_args', None) is not None and train_config.recorrection_args.enable:
        model_str +='_recorrection'

    group_name = f'{data_str} - {model_str}'
    run_name = 'seed-' + str(train_config.seed) + ' - ' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    return group_name, run_name

