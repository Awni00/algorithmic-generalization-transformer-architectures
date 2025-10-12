import torch 
from utils import batch_forward, split_attn_head_output_without_bias
from Simtransformer.simtransformer.utils import EasyDict, attach_hooks
from typing import Optional, Callable
import wandb
import os 
current_dir = os.path.dirname(__file__)
CONTROLLED_EXP_DATA_PATH = os.path.join(current_dir, 'data/controlled_exp.pt')
CONTROLLED_EXP_KEYS = ['var_0', 'var_1', 'var_2', 'rhs']

def get_intermediate_buffer_hook(litmodel):
    HookDict = EasyDict()
    for i in range(2):
        HookDict[f'L{i}_attn_prob'] = f'blocks.layer_{i}.attn.attn_prob'
        HookDict[f'L{i}_attn_v'] = f'blocks.layer_{i}.attn.v'
    
    # attach hooks to the TF model
    buffer, hook_handles = attach_hooks(litmodel.model.transformer, HookDict)
    return buffer, hook_handles
    
def remove_intermediate_buffer_hook(hook_handles, buffer):
    for handle in hook_handles:
        handle.remove()
    del buffer

def compute_attention_variance(litmodel, buffer_new):
    # --------- compute the attention heads' individual output --------- #
    output_by_heads = split_attn_head_output_without_bias(buffer_new['L0_attn_prob'], buffer_new['L0_attn_v'], attn_model=litmodel.model.transformer.blocks['layer_0'].attn)
    # print('output_by_heads:', output_by_heads.shape)

    # --------- compute the variance and total fluctuation --------- #
    variance = (output_by_heads - output_by_heads.mean(dim=0, keepdim=True)).norm(dim=-1, p=2).square().mean(dim=0)
    total_fluc = output_by_heads.norm(dim=-1, p=2).square().mean(dim=0)

    # --------- compute the proportion of variance in the fluctuation --------- #
    portion_variance_in_fluc = variance / total_fluc  # shape (q_len, num_heads)

    return portion_variance_in_fluc, variance, total_fluc, output_by_heads

def controlled_experiment_step(controlled_exp_key, litmodel, model_config, batch_size=64, logger=Optional[Callable], wandb_logger=None):
    """
    controoled_exp_key: str, the key of the controlled experiment
    """
    
    if controlled_exp_key not in CONTROLLED_EXP_KEYS:
        raise ValueError(f'controlled_exp_key must be one of {CONTROLLED_EXP_KEYS}')
    
    # attach hooks to the model
    buffer, hook_handles = get_intermediate_buffer_hook(litmodel)
    
    # load the data and select the data for the controlled experiment
    controlled_exp_data = torch.load(CONTROLLED_EXP_DATA_PATH)
    data_dict = controlled_exp_data[controlled_exp_key]
    
    # unpack the data
    batch_new, new_eqs, new_rhs_values, new_eq_start_pos, new_eq_end_pos = data_dict['batch_new'], data_dict['new_eqs'], data_dict['new_rhs_values'], data_dict['new_eq_start_pos'], data_dict['new_eq_end_pos']

    # move the batch_new tensor to the device
    batch_new = batch_new.to(litmodel.device)
    
    # batch forward
    factored_preds, factor_decodes, buffer_new = batch_forward(batch_new, litmodel, buffer, new_eq_start_pos, new_eq_end_pos, model_config, batch_size=batch_size, buffer_kwargs='all', verbose=False, train=False, sequence_mask=None, device=litmodel.device)
    
    # test the accuracy 
    preds = factor_decodes[model_config.factors.index('VALUE')][..., new_eq_end_pos - 1].tolist()
    correct = sum(1 for i in range(len(preds)) if preds[i] == new_rhs_values[i])
    accuracy = correct / len(preds)
    
    if logger:
        logger(f'ControlledExp_{controlled_exp_key}_accuracy', accuracy)
    
    # compute the fluctuation at each attention head's output 
    portion_variance_in_fluc, variance, total_fluc, output_by_heads = compute_attention_variance(litmodel, buffer_new)
    # portion_variance_in_fluc: shape (q_len, num_heads)
    # variance: shape (q_len, num_heads)
    # total_fluc: shape (q_len, num_heads)
    # output_by_heads: shape (q_len, num_heads, d_model)
    
    if logger:
        for head_idx in range(portion_variance_in_fluc.shape[1]):
            logger(f'ControlledExp_{controlled_exp_key}_L0_attn_output_rel_variance/head_{head_idx}', portion_variance_in_fluc[-1, head_idx])
            logger(f'ControlledExp_{controlled_exp_key}_L0_attn_output_variance/head_{head_idx}', variance[-1, head_idx])
            logger(f'ControlledExp_{controlled_exp_key}_L0_attn_output_norm/head_{head_idx}', total_fluc[-1, head_idx])

    
    # compute the each attention head's probability entropy
    attn_prob = buffer_new['L0_attn_prob'] # shape (batch_size, num_heads, query_len, key_len)
    # pick the query position to be the last position of the equation
    attn_prob = attn_prob[:, :, -1, :] # shape (batch_size, num_heads, key_len)
    # compute the entropy
    entropy = -(attn_prob * torch.log(attn_prob + 1e-10)).sum(dim=-1) # shape (batch_size, num_heads)
    entropy = entropy.mean(dim=0) # shape (num_heads,)
    
    if logger:
        for head_idx in range(entropy.shape[0]):
            logger(f'ControlledExp_{controlled_exp_key}_L0_attn_entropy/head_{head_idx}', entropy[head_idx])
            
            

    # compute the cosine similarity in the attention heads' probability
    cos_sim = torch.nn.functional.cosine_similarity(attn_prob.unsqueeze(1), attn_prob.unsqueeze(2), dim=-1) # shape (batch_size, num_heads, num_heads)
    cos_sim = cos_sim.mean(dim=0) # shape (num_heads, num_heads)
    
    if logger:
        wandb_logger.experiment.log({f'ControlledExp_{controlled_exp_key}_L0_attn_cosine_similarity': wandb.Image(cos_sim)})
    