# prepare the teacher model and value dictionary
# from model import *
import torch
import numpy as np 
import copy 
from tqdm import tqdm
import os

import sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Simtransformer.simtransformer.model_base import MultiHeadAttentionDeBERTa

def process_raw_sentence(raw_sentence, setence_embed, model_config, litmodel, factored_tokenizer, verbose=True):    
    from model import DAGTeacherModel, INVALID_NUM  # local import to avoid circular dependency
    if verbose:
        print(f'raw_sentence: {raw_sentence}')
        print(f'raw_sentence length: {len(raw_sentence)}')
    
    # find the index of the first '<PAD>' token
    pad_idx = raw_sentence.index('<PAD>')
    print(f'first <PAD> token at index {pad_idx}') if verbose else None

    # find all the variables 'x_{i}' in the equation
    used_variables = list(set([word for word in raw_sentence if 'x' in word]))
    unused_variables = [f'x_{i}' for i in range(factored_tokenizer.n_vars) if f'x_{i}' not in used_variables]

    if verbose:
        print(f'number of used variables: {len(used_variables)}')
        print(f'number of unused variables: {len(unused_variables)}')

    # Let us use the teacher model to obtain the variable values    
    batch_old, batch_label_old, batch_info_old = setence_embed[..., :model_config.data_dim], setence_embed[..., model_config.data_dim:model_config.data_dim + model_config.label_dim], setence_embed[..., model_config.data_dim + model_config.label_dim:]

    depths = batch_info_old[..., 0]

    max_depth_data = depths.max() + 1

    batch_ext, value_avail_pos, value_to_compute_pos = litmodel.get_teacher_batch_with_depth(batch_old.unsqueeze(0), batch_label_old.unsqueeze(0), depths.unsqueeze(0))

    teacher_model = DAGTeacherModel(model_config, factored_tokenizer)

    values = [teacher_model.value_idx2num(token) for token in batch_label_old.squeeze(0)] # shape (seq_len,)

    # make a dictionary that stores the variable values
    var_value_dict = {}
    for val, dep, token in zip(values, depths, raw_sentence):
        if dep != INVALID_NUM:
            var_value_dict[token] = val
            
    return batch_ext, pad_idx, used_variables, unused_variables, var_value_dict



def generate_new_eqs(new_vars_pair, old_eqs, old_eqs_batch_ext, var_value_dict, model_config, factored_tokenizer, verbose=True, template=None):
    """
    Returns: 
    [new_eqs, batch_new, new_values]
    """
    
    new_eqs = []
    new_values = []
    
    # NOTE: we copy the teacher prediction at the last depth to the new batch, this is equivalent to the output of the last loop of the litmodel if the model makes correct predictions.
    batch_new = old_eqs_batch_ext[-1, ..., :model_config.data_dim].repeat(len(new_vars_pair), 1, 1)
    pad_idx = old_eqs.index('<PAD>')

    for idx, pair in enumerate(new_vars_pair):
        new_eqs.append(copy.deepcopy(old_eqs))
        
        new_eq_len = len(pair) * 2 # the length of the new equation
        
        # prepare new_eq
        if template is None:
            new_eq = ['<EQ_SEP>']
            for i in range(len(pair) - 2):
                new_eq += [pair[i], 'ADD']
            new_eq += [pair[-2], '=', pair[-1]]
            start_idx = pad_idx 
            end_idx = start_idx + len(new_eq)
        else:
            # check the '<EQ_SEP>' token in the template
            eq_sep_idx = template.index('<EQ_SEP>')
            start_idx = pad_idx - eq_sep_idx
            end_idx = start_idx + len(template)
            new_eq = copy.deepcopy(template)
            for i in range(len(pair)-1):
                for j, token in enumerate(new_eq):
                    if token == f'var_{i}':
                        new_eq[j] = pair[i]
                    elif token == 'rhs':
                        new_eq[j] = pair[-1]
            
        # update the values 
        value = 0
        for i in range(len(pair) - (end_idx - pad_idx) // 2, len(pair) - 1):
            value += var_value_dict[pair[i]]
        value = value % factored_tokenizer.mod_val
        new_values.append(value)
                
        # update the new_eqs with the new equation
        new_eqs[idx][start_idx:end_idx] = new_eq
        
        # prepare the new batch
        factored_new_eq = factored_tokenizer.factor_tokens(new_eq)
        batch_to_add = torch.tensor(factored_tokenizer.encode_factored_tokens(factored_new_eq), dtype=batch_new.dtype, device=batch_new.device)
        
        # NOTE: we don't need to prepare batch_label and batch_info for our purpose
        
        # variable_pos = batch_to_add[..., model_config.factors.index('SYNTAX')].eq(factored_tokenizer.syntax_tok2idx['VARIABLE'])
        # batch_to_add[..., model_config.factors.index('VALUE')][variable_pos] = factored_tokenizer.value_tok2idx['EMPTY']
        
        batch_new[idx, start_idx:end_idx, :model_config.data_dim] = batch_to_add

    # remove the padding tokens
    batch_new = batch_new[..., :pad_idx + new_eq_len, :] 

    if verbose:
        demo_idx = np.random.randint(len(new_eqs))
        demo_eqs = new_eqs[demo_idx][start_idx:end_idx]
        demo_batch = batch_new[demo_idx, start_idx:end_idx, :model_config.data_dim]    
        new_eqs = [new_eq[:pad_idx + new_eq_len] for new_eq in new_eqs]
        print(f'new equations example-idx={demo_idx}:\n {demo_eqs}')
        print(f'tokenized new batch sample-idx={demo_idx}:\n {demo_batch}')
        print(f'number of new equations: {len(new_eqs)}')

        print(f'reduced batch shape: {batch_new.shape}')
    return new_eqs, batch_new, new_values




def transfer_buffer(buffer, buffer_new, template_start, template_end, sequence_mask=None):
    for k, v in buffer_new.items():
        if 'attn_prob' in k:
            if v is None:
                buffer_new[k] = [buffer[k][..., template_start:template_end, :].detach()] 
            else:
                buffer_new[k].append(buffer[k][...,  template_start:template_end, :].detach())
        else:
            if sequence_mask is None:
                sequence_mask = torch.ones(buffer[k].shape[1], dtype=torch.bool, device=buffer[k].device)
            if v is None:
                buffer_new[k] = [buffer[k][:, sequence_mask, ...].detach()] 
            else:
                buffer_new[k].append(buffer[k][:, sequence_mask, ...].detach())

def buffer_to_device(buffer, device):
    for k, v in buffer.items():
        if isinstance(v, list):
            buffer[k] = torch.cat(v, dim=0).to(device)
        else:
            buffer[k] = v.to(device)

def batch_forward(batch_new, litmodel, buffer, template_start, template_end, model_config, batch_size=16, buffer_kwargs='all', verbose=True, train=False, sequence_mask=None, device='cpu'):
    # split batch_new into parts and do forward
    
    if train: 
        litmodel.train()
    else:
        litmodel.eval()
        
    buffer_new = {}
    if buffer_kwargs == 'all':
        buffer_new = {k: None for k in buffer.keys()}
    elif buffer_kwargs == 'none':
        buffer_new = {}
    else:
        for k in buffer_kwargs:
            if k in buffer.keys():
                buffer_new[k] = None
    
    batch_size = batch_size
    factored_preds = []
    for i in tqdm(range(0, len(batch_new), batch_size), disable=not verbose):
        batch_new_part = batch_new[i:i+batch_size]
        with torch.no_grad():
            factored_preds.append(litmodel.model(batch_new_part))
            transfer_buffer(buffer, buffer_new, template_start, template_end, sequence_mask=sequence_mask)

    buffer_to_device(buffer_new, device)

    factored_preds_cat = []
    # concatenate the predictions
    for i in range(len(model_config.factors)):
        factored_preds_cat.append(torch.cat([pred[i] for pred in factored_preds], dim=0))
        
    factored_preds = factored_preds_cat

    factor_decodes = litmodel.generate_factor_decodes(factored_preds, temperature=0.0)
    return factored_preds, factor_decodes, buffer_new


def split_attn_head_output_without_bias(attn_prob, v, 
                                        attn_model: MultiHeadAttentionDeBERTa, ):
    """
    attn_prob: shape (batch_size, num_heads, query_len, key_len)
    v: shape (batch_size, key_len, num_heads, head_dim)
    train: if True, the output will be detached from the computation graph
    """
        
    o = torch.einsum("bhnm,bmhk->bnhk", attn_prob, v) # [batch_size, query_seq_len, num_heads, vo_embed_size_per_head]
    
    # get the output projection weight
    o_proj_weight = attn_model.o_proj.weight # shape (attn_model.o_dim, attn_model._vo_embed_size), 
    num_heads = attn_model._num_heads
    vo_embed_size_per_head = attn_model._vo_embed_size // num_heads
    
    o_proj_weight = attn_model.o_proj.weight.data.detach().view(attn_model.o_dim, num_heads, vo_embed_size_per_head) # shape (attn_model.o_dim, num_heads, vo_embed_size_per_head)
    
    o = torch.einsum("bnhk,ohk->bnho", o, o_proj_weight) # [batch_size, query_seq_len, o_dim]
    return o


def extract_ov_weight(attn_model: MultiHeadAttentionDeBERTa):
    o_proj_weight = attn_model.o_proj.weight # shape (attn_model.o_dim, attn_model._vo_embed_size), 
    num_heads = attn_model._num_heads
    vo_embed_size_per_head = attn_model._vo_embed_size // num_heads
    
    o_proj_weight = attn_model.o_proj.weight.data.detach().view(attn_model.o_dim, num_heads, vo_embed_size_per_head) # shape (attn_model.o_dim, num_heads, vo_embed_size_per_head)
    
    v_proj_weight = attn_model.v_proj.weight # shape (attn_model._vo_embed_size, attn_model._vo_embed_size),
    v_proj_weight = v_proj_weight.data.detach().view(num_heads, vo_embed_size_per_head, attn_model._vo_embed_size) # shape (num_heads, vo_embed_size_per_head, attn_model._vo_embed_size)
    
    o_proj_weight = o_proj_weight.permute(1, 0, 2) # shape (num_heads, attn_model.o_dim, vo_embed_size_per_head)
    
    return o_proj_weight, v_proj_weight



class L1AttnOVDecoder(torch.nn.Module):
    def __init__(self, L1_ov_var0, L1_ov_var1, L1_ov_var2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embedding = torch.cat([L1_ov_var0, L1_ov_var1, L1_ov_var2])
        self.L1_ov_var0 = L1_ov_var0 / L1_ov_var0.norm(dim=-1, keepdim=True)
        self.L1_ov_var1 = L1_ov_var1 / L1_ov_var1.norm(dim=-1, keepdim=True)
        self.L1_ov_var2 = L1_ov_var2 / L1_ov_var2.norm(dim=-1, keepdim=True)
        
        self.group_size = self.L1_ov_var0.shape[0]
        self.decoder = torch.nn.Linear(embedding.shape[1], embedding.shape[0], bias=False)
        self.decoder.weight.data = embedding

    @classmethod
    def load_from_ckpt(cls, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        L1_ov_var0 = checkpoint['L1_ov_var0']
        L1_ov_var1 = checkpoint['L1_ov_var1']
        L1_ov_var2 = checkpoint['L1_ov_var2']
        model = cls(L1_ov_var0, L1_ov_var1, L1_ov_var2)
        return model

    def forward(self, x):
        return self.decoder(x)

    def state_dict(self, *args, **kwargs):
        return {
            'L1_ov_var0': self.L1_ov_var0,
            'L1_ov_var1': self.L1_ov_var1,
            'L1_ov_var2': self.L1_ov_var2,
        }

    def load_state_dict(self, state_dict, strict=True):
        self.L1_ov_var0 = state_dict['L1_ov_var0']
        self.L1_ov_var1 = state_dict['L1_ov_var1']
        self.L1_ov_var2 = state_dict['L1_ov_var2']
        embedding = torch.cat([self.L1_ov_var0, self.L1_ov_var1, self.L1_ov_var2])
        self.decoder = torch.nn.Linear(embedding.shape[1], embedding.shape[0], bias=False)
        self.decoder.weight.data = embedding
        
    def forward(self, x):
        pred = self.decoder(x)
        decode = torch.argmax(pred, dim=-1)
        
        group_idx = decode // self.group_size 
        token_idx = decode % self.group_size
        return (group_idx, token_idx)