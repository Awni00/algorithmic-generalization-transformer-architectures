import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add parent directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
from Simtransformer.simtransformer.model_base import TransformerEncoder
from Simtransformer.simtransformer.utils import CosineAnnealingWarmup, EasyDict, clever_load, clever_save, Shampoo, signSGD, MRR_fn

from typing import Any
import lightning.pytorch as pl

from datetime import datetime
import copy

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

class FactoredRecurrentTransformer(torch.nn.Module):
    def __init__(self, model_config: Any):
        super().__init__()
        self.model_config = model_config
        self.token_embedder = FactoredTokenEmbedder(
            vocab_sizes=model_config.vocab_sizes,
            embed_dim=model_config['hidden_size'],
            combine_op='sum'
        )
        self.n_iters = model_config.n_iters
        self.transformer = TransformerEncoder(model_config)
        self.output_layers = torch.nn.ModuleList([
            torch.nn.Linear(model_config.hidden_size, vocab_size)
            for vocab_size in self.model_config.vocab_sizes
        ])

    def forward(self, factored_tokens, n_iters=None):
        if n_iters is None:
            n_iters = self.n_iters

        embedded = self.token_embedder(factored_tokens)
        x = embedded
        for _ in range(n_iters):
            x = self.transformer(x)
        preds = [
            output_layer(x)
            for output_layer in self.output_layers
        ]
        return preds

class RecurrentTransformer(torch.nn.Module):
    def __init__(self, model_config: Any):
        super().__init__()
        self.model_config = model_config
        self.token_embedder = torch.nn.Embedding(model_config.vocab_size, model_config.hidden_size)
        self.n_iters = model_config.n_iters
        self.transformer = TransformerEncoder(model_config)
        self.output_layer = torch.nn.Linear(model_config.hidden_size, model_config.vocab_size)

    def forward(self, tokens, n_iters=None):
        if n_iters is None:
            n_iters = self.n_iters

        embedded = self.token_embedder(tokens)
        x = embedded
        for _ in range(n_iters):
            x = self.transformer(x)
        output_logits = self.output_layer(x)
        return output_logits

class LitRecurrentTransformerModel(pl.LightningModule):
    def __init__(self, model_config, data_config, train_config):
        super().__init__()
        self.model = FactoredRecurrentTransformer(model_config)
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config
        self.factor_loss_scales = [1, 1, 1, 1, 0, 0] # TODO: FIXME: figure out how to set these scaling factors; may need to normalize according to vocab_size of each factor>

        self.criterion = torch.nn.CrossEntropyLoss()
        self.fast_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.baseline = True # set flag to easily filter baseline models on W&B

        self.save_hyperparameters() # hyperparameters will be saved to checkpoint under key "hyper_parameters"

    def load_ckpt(self, checkpoint_path, strict=True):
        ckpt = torch.load(checkpoint_path)
        # the keys of the state_dict start with 'model._orig_mod.' due to the torch compiled model
        # let's remove this prefix
        state_dict = {k.replace('model._orig_mod.', ''): v for k, v in ckpt['state_dict'].items()}
        self.model.load_state_dict(state_dict, strict=strict)
        pass

    def forward(self, factored_tokens):
        return self.model(factored_tokens)

    def training_step(self, batch, batch_idx):

        total_train_loss = self.compute_batch_loss(batch, log_prefix='train', log=True, on_step=True, on_epoch=True)

        return total_train_loss

    def validation_step(self, batch, batch_idx):
        total_val_loss = self.compute_batch_loss(batch, log_prefix='val', log=True, on_step=False, on_epoch=True)

        return total_val_loss

    def test_step(self, batch, batch_idx):
        total_test_loss = self.compute_batch_loss(batch, log_prefix='test', log=True, on_step=False, on_epoch=True)

        return total_test_loss

    def reduce_batch(self, batch):
        # find all the positions that only have padding tokens
        batch_view = batch[..., :self.model_config.data_dim].eq(self.factored_tokenizer.syntax_tok2idx['<PAD>'])
        # find the positions in the second dimension that all the values are True
        batch_view = batch_view.permute(1, 0, 2).reshape(batch.size(1), -1).all(dim=-1)
        return batch[:, torch.logical_not(batch_view), :]

    def get_padding_mask(self, depth_batch):
        return depth_batch[..., self.factored_tokenizer.factors.index('SYNTAX')].eq(self.factored_tokenizer.syntax_tok2idx['<PAD>'])

    def compute_batch_loss(self,
                           batch,
                           log_prefix=None,
                           log=True,
                           **log_kwargs):
        # batch: Tensor of integers of shape [batch_size, n_depths, seq_len, n_factors]

        batch = self.reduce_batch(copy.deepcopy(batch))

        batch, batch_label, batch_info = batch[..., :self.model_config.data_dim], batch[..., self.model_config.data_dim:self.model_config.data_dim + self.model_config.label_dim], batch[..., self.model_config.data_dim + self.model_config.label_dim:]

        factor_preds = self(batch)

        # Compute the loss for each factor at this depth.
        total_loss = 0
        for factor, factor_pred in enumerate(factor_preds):
            # factor_pred: Tensor of shape [batch_size, seq_len, factor_vocab_size]
            factor_name = self.model_config.factors[factor].lower()

            factor_pred_flattened = factor_pred.view(-1, factor_pred.size(-1)) # shape: [batch_size * seq_len, factor_vocab_size]

            factor_label = batch_label[:, :, factor] # shape: [batch_size, seq_len]

            factor_label_flattened = factor_label.contiguous().view(-1) # shape: [batch_size * seq_len]

            factor_at_depth_loss = self.criterion(factor_pred_flattened, factor_label_flattened)

            if log:
                self.log(f'{log_prefix}/loss/{factor_name}', factor_at_depth_loss, **log_kwargs)

            scaled_factor_at_depth_loss = self.factor_loss_scales[factor] * factor_at_depth_loss

            total_loss += scaled_factor_at_depth_loss

            # compute accuracy
            correct = factor_pred.argmax(dim=-1) == factor_label
            per_token_acc = correct.float().mean()
            full_seq_acc = correct.all(dim=-1).float().mean()

            if log:
                self.log(f'{log_prefix}/per_token_acc/{factor_name}', per_token_acc, **log_kwargs)
                self.log(f'{log_prefix}/full_seq_acc/{factor_name}', full_seq_acc, **log_kwargs)

                # compute MRR
                mrr = MRR_fn(factor_pred.view(-1, factor_pred.size(-1)), factor_label.contiguous().view(-1))
                self.log(f'{log_prefix}/MRR/{factor_name}', mrr, **log_kwargs)

        # compute full sequence accuracy over all factors
        correct = torch.stack([factor_pred.argmax(dim=-1) == batch_label[:, :, factor] for factor, factor_pred in enumerate(factor_preds)])
        full_seq_acc = correct.all(dim=0).float().mean()

        if log:
            self.log(f'{log_prefix}/full_seq_acc/all_factors', full_seq_acc, **log_kwargs)
            self.log(f'{log_prefix}/total_loss', total_loss, **log_kwargs, prog_bar=True)

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
            scheduler,
            metric: Any,
    ) -> None:
        scheduler.step()



class LitCoTRecurrentTransformerModel(pl.LightningModule):
    def __init__(self, model_config, data_config, train_config):
        super().__init__()
        self.model = RecurrentTransformer(model_config)
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config

        self.criterion = torch.nn.CrossEntropyLoss()

        self.baseline = True # set flag to easily filter baseline models on W&B

        self.save_hyperparameters() # hyperparameters will be saved to checkpoint under key "hyper_parameters"

    def load_ckpt(self, checkpoint_path, strict=True):
        ckpt = torch.load(checkpoint_path)
        # the keys of the state_dict start with 'model._orig_mod.' due to the torch compiled model
        # let's remove this prefix
        # if compiled, keys will be model._orig_mod. If not compiled, keys will be model.
        state_dict = ckpt['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # NOTE: this is a temporary hacky fix to compensate for a backwards-incompatible change
        # in Simtransformer's implementation of DeBERTA-based attention
        state_dict = {k.replace('pos_relpos_embeddings', 'pos_model.relpos_embeddings'): v for k, v in state_dict.items()}
        state_dict = {k.replace('pos_layer_norm', 'pos_model.layer_norm'): v for k, v in state_dict.items()}

        # change to make it backwarrd-compatible with SinAbPE
        #	Missing key(s) in state_dict: "transformer.pos_model.pe".
        #	Unexpected key(s) in state_dict: "transformer.pos_pe".
        state_dict = {k.replace('transformer.pos_pe', 'transformer.pos_model.pe'): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=strict)
        pass

    def forward(self, factored_tokens):
        return self.model(factored_tokens)

    def training_step(self, batch, batch_idx):

        total_train_loss = self.compute_batch_loss(batch, log_prefix='train', log=True, on_step=True, on_epoch=True)

        return total_train_loss

    def validation_step(self, batch, batch_idx):
        total_val_loss = self.compute_batch_loss(batch, log_prefix='val', log=True, on_step=False, on_epoch=True)

        return total_val_loss

    def test_step(self, batch, batch_idx):
        total_test_loss = self.compute_batch_loss(batch, log_prefix='test', log=True, on_step=False, on_epoch=True)

        return total_test_loss

    def compute_batch_loss(self,
                           batch,
                           log_prefix=None,
                           log=True,
                           **log_kwargs):
        # batch: Tensor of integers of shape [batch_size, n_depths, seq_len, n_factors]

        (x, y), (cot_mask, value_mask, var_mask, operation_mask) = batch

        # x: Tensor of integers of shape [batch_size, seq_len]
        # y: Tensor of integers of shape [batch_size, seq_len]
        # cot_mask: Tensor of booleans of shape [batch_size, seq_len]
        # value_mask: Tensor of booleans of shape [batch_size, seq_len]
        # var_mask: Tensor of booleans of shape [batch_size, seq_len]
        # operation_mask: Tensor of booleans of shape [batch_size, seq_len]

        logits = self(x)
        # logits: Tensor of shape [batch_size, seq_len, vocab_size]

        # compute cross-entropy loss with cot_mask

        loss = self.criterion(logits[torch.where(cot_mask)], y[torch.where(cot_mask)])
        if log:
            self.log(f'{log_prefix}/loss', loss, **log_kwargs)

        # compute metrics
        mask_dict = {'all': cot_mask, 'value': cot_mask & value_mask, 'var': cot_mask & var_mask, 'operation': cot_mask & operation_mask}
        for name, mask in mask_dict.items():
            acc = self.compute_masked_accuracy(logits.argmax(dim=-1), y, mask, type='tokenwise')
            self.log(f'{log_prefix}/tokenwise_accuracy/{name}', acc, **log_kwargs)

            acc = self.compute_masked_accuracy(logits.argmax(dim=-1), y, mask, type='full_seq')
            self.log(f'{log_prefix}/full_seq_accuracy/{name}', acc, **log_kwargs)

        return loss

    def compute_masked_accuracy(self, pred, y, mask, type='tokenwise'):
        if type == 'tokenwise':
            correct = pred[torch.where(mask)] == y[torch.where(mask)]
            acc = correct.sum().float() / correct.shape[0]
        elif type == 'full_seq':
            correct = (pred == y) | ~mask
            full_seq_correct = correct.all(dim=-1)
            acc = full_seq_correct.sum().float() / full_seq_correct.shape[0]
        else:
            raise ValueError(f"Invalid type: {type}")

        return acc

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
            scheduler,
            metric,
    ) -> None:
        scheduler.step()


def get_experiment_name(model_config, data_config, train_config):
    # Format:
    # Group: Data Config - Model Config
    # Name: Seed + Date-Time

    data_str = ''
    if data_config.get('cot', False):
        data_str += 'COT'
        if 'cot_type' in data_config:
            data_str += '_' + data_config.cot_type + '-'
        else:
            raise ValueError('cot_type not found in data_config')
        if data_config.dag_config.get('var_length', False):
            data_str += 'var_length-'
        data_str += '-'

    data_str += f'Nodes{data_config.dag_config.num_nodes}-' + '_'.join(data_config.dag_config.func_vocab)
    model_str = f'Baseline-T{model_config.n_iters}L{model_config.num_layers}H{model_config.num_heads}D{model_config.hidden_size}_{model_config.pos_enc_type}'

    group_name = f'{data_str} - {model_str}'
    run_name = 'seed-' + str(train_config.seed) + ' - ' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    return group_name, run_name

