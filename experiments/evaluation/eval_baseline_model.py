import argparse
import os, sys; sys.path.append(os.path.abspath('../'))

import torch
import numpy as np
import pandas as pd
import plotly.express as px

import yaml
import pickle
import wandb
import wandb.plot

from ansi2html import Ansi2HTMLConverter

from baseline_models import LitRecurrentTransformerModel
from Simtransformer.simtransformer.utils import EasyDict
from tokenizers import FactoredVocabTokenizer

from metric_utils import calc_metrics_across_batches, print_model_steps

from DAG_train import DAGDataModule
from datetime import datetime

parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument('--wandb_project', type=str, default="RecTransformer-DiscIntermRep-Eval")
parser.add_argument('--wandb_entity', type=str, default="transformer-computation-graph")
parser.add_argument('--group_name', type=str, required=True)
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_batches', type=int, default=None)
parser.add_argument('--out_path', type=str, default='results/baselines')
parser.add_argument('--log_wandb', action='store_true')
parser.add_argument('--overwrite_ok', action='store_true')
parser.add_argument('--debug', action='store_true', help="Debug mode (no wandb logging and no saving of results)")

args = parser.parse_args()

if args.debug:
    args.log_wandb = False
    print('Debug mode: no wandb logging and no saving of results')
    args.n_batches = 1

base_dir = os.path.abspath('../')
configs_dir = os.path.join(base_dir, 'configs')

val_ds_path = 'val_datasets'

group_name = args.group_name
run_name = args.run_name

# load model, train, and data config
config_dir = group_name.replace(' ', '') # calc config dir from wandb group name
# if config_dir has '_nointerm', remove it
config_dir = config_dir.replace('_nointerm', '')
# if config_dir has '_interm', remove it

with open(os.path.join(configs_dir, config_dir, 'model_config.yaml')) as f:
    model_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

with open(os.path.join(configs_dir, config_dir, 'train_config.yaml')) as f:
    train_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

with open(os.path.join(configs_dir, config_dir, 'data_config.yaml')) as f:
    data_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

# load factored_tokenizer
# with open(os.path.join(base_dir, data_config.data_dir, data_config.tokenizer_file_name), 'rb') as f:
#     factored_tokenizer = pickle.load(f)
factored_tokenizer = FactoredVocabTokenizer(n_vars=max(data_config.dag_config.num_nodes, data_config.val_dag_config.num_nodes), ops=data_config.dag_config.func_vocab, mod_val=data_config.dag_config.mod_val, max_fan_in=data_config.dag_config.max_fan_in_deg)

# set vocab_size and factors in model_config
model_config.vocab_sizes = factored_tokenizer.vocab_sizes
model_config.factors = factored_tokenizer.factors

# load model
ckpt_path = 'checkpoints/baselines'
ckpt_fname = 'last.ckpt'
model_ckpt_path = os.path.join(base_dir, ckpt_path, f"{group_name}-{run_name}", ckpt_fname) # model checkpoint path from group name
# model_ckpt = torch.load(model_ckpt_path, weights_only=False)

litmodel = LitRecurrentTransformerModel(model_config, train_config, data_config)
litmodel.factored_tokenizer = factored_tokenizer

litmodel.load_ckpt(model_ckpt_path)

# create the data module
func_vocab_val_dir = '_'.join(data_config.dag_config.func_vocab)
# get list of .pt files in val_ds_path/func_vocab_val_dir
val_ds_dir = os.path.join(val_ds_path, func_vocab_val_dir)
val_ds_files = [os.path.join(val_ds_dir, f) for f in os.listdir(val_ds_dir) if f.endswith('.pt')]
# match pattern from test_data_N{a}D{b}.pt and get N and D values
val_sizes = sorted([int(fname.split('_')[-1].split('.')[0].split('N')[1]) for fname in val_ds_files])

# set up wandb logging
if args.log_wandb:
    experiment_config = dict(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        val_sizes=val_sizes,
        val_ds_dir=val_ds_dir,
        val_ds_files=val_ds_files,
        val_ds_path=val_ds_path,
        group_name=group_name,
        run_name=run_name,
        baseline=True
    )
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        group=group_name, name=run_name,
        config=experiment_config)


from metric_utils import compute_attention_metrics, attach_hooks_to_model
import copy
from tqdm import tqdm

def calc_batch_metrics(litmodel, batch, factored_tokenizer, factors_slice=None, verbose=False):

    litmodel.eval()
    buffer, hook_handles = attach_hooks_to_model(litmodel)

    model_config = litmodel.model_config

    if factors_slice is None:
        factors_slice = slice(0, 6) # consider all factors

    # process batch
    # model_config, data_config = litmodel.model_config, litmodel.data_config
    batch = litmodel.reduce_batch(copy.deepcopy(batch))
    batch, batch_label, batch_info = batch[..., :model_config.data_dim], batch[..., model_config.data_dim:model_config.data_dim + model_config.label_dim], batch[..., model_config.data_dim + model_config.label_dim:]
    # note: batch_label is final label

    # infer which tokens are padding, and ignore them in computing accuracy
    pad_encoding = torch.tensor(factored_tokenizer.encode_string(factored_tokenizer.PAD)[0])
    is_pad = torch.eq(batch, pad_encoding).all(dim=-1)

    # get first pad for each sample
    orig_seq_len = batch.size(1)
    seq_lens = (~is_pad).sum(dim=1)
    max_seq_len = seq_lens.max()
    if verbose:
        print('max non-pad length:', max_seq_len)

    is_pad = is_pad.unsqueeze(-1).repeat(1, 1, 6)
    is_pad = is_pad[..., factors_slice]

    # calc metrics up to max_seq_len in batch
    batch = batch[:, :max_seq_len]
    batch_label = batch_label[:, :max_seq_len]
    is_pad = is_pad[:, :max_seq_len]

    # inputs = batch[0]
    logitss = litmodel(batch)
    preds = torch.stack([logits.argmax(-1) for logits in logitss], dim=-1) # argmax prediction for each factor

    metrics =  dict()

    # compute attention metrics
    attn_metrics = compute_attention_metrics(buffer)
    metrics.update(attn_metrics)

    # compute accuracy
    is_correct = (preds == batch_label)[..., factors_slice]

    is_correct = torch.logical_or(is_correct, is_pad)

    full_seq_acc = is_correct.all(axis=(1,2)).float().mean().item()
    factor_acc = is_correct.float().mean(axis=(0,1))
    metrics['final_label_full_acc'] = full_seq_acc
    metrics['final_label_factor_acc'] = factor_acc
    metrics['max_seq_len'] = max_seq_len.float()

    if verbose:
        print('final_label_full_acc', full_seq_acc)
        print('final_label_factor_acc', factor_acc)

    return metrics

def calc_metrics_across_batches(
        litmodel, dataloader, factored_tokenizer,
        n_batches=None, factors_slice=None, verbose=False):

    if factors_slice is None:
        factors_slice = slice(0, 6) # consider all factors

    metrics = dict()

    for i, batch in tqdm(enumerate(dataloader)):
        if n_batches is not None and i >= n_batches:
            break
        batch_size = batch.size(0)

        batch_metrics = calc_batch_metrics(litmodel, batch, factored_tokenizer, factors_slice=factors_slice, verbose=verbose)

        for k, v in batch_metrics.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)

    metrics['final_label_full_acc'] = torch.tensor(metrics['final_label_full_acc'])

    for k, v in metrics.items():
        if isinstance(v, list):
            if isinstance(v[0], torch.Tensor):
                metrics[k] = torch.stack(v, dim=0)

    return metrics

val_metrics_by_size = dict()
val_metrics_by_size_agg = dict()

sample_preds = dict({n: [] for n in val_sizes})

specs = dict()

# evaluate model on different datasets
for n_nodes in val_sizes:

    print()
    print('-'*50)
    print(f"# Nodes: {n_nodes}")
    print(f"Current time: {datetime.now()}")

    ds_file = os.path.join(val_ds_dir, f"testdata_N{n_nodes}.pt")
    print(f"Evaluating model on {ds_file}")

    data_module = DAGDataModule(
        data_path=[ds_file, ds_file],
        batch_size=args.batch_size,
        num_workers=train_config.num_workers,
        factored_tokenizer=factored_tokenizer)


    # add the dimension of data, label, and info to model config
    data_module.setup()
    model_config.data_dim = data_module.data_dim
    model_config.label_dim = data_module.label_dim
    model_config.info_dim = data_module.info_dim

    specs[n_nodes] = data_module.val_spec

    # evaluate model
    # n_steps = None infers the number of steps from the data
    val_metrics = calc_metrics_across_batches(
        litmodel, data_module.val_dataloader(), factored_tokenizer=factored_tokenizer,
        n_batches=args.n_batches, factors_slice=slice(0, 4), verbose=False)

    val_metrics_by_size[n_nodes] = val_metrics

    val_metrics_by_size_agg[n_nodes] = {k: torch.nanmean(v, dim=0) for k, v in val_metrics.items()}

    for k, v in val_metrics_by_size_agg[n_nodes].items():
        print(f"{k}: {v}")


# recursively cast Tensor to np.ndarray
def to_numpy(d):
    if isinstance(d, dict):
        return {k: to_numpy(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.cpu().numpy()
    else:
        return d

val_metrics_by_size_np = to_numpy(val_metrics_by_size)
val_metrics_by_size_np = to_numpy(val_metrics_by_size_agg)

# save parsed results
run_name_ = run_name
if not args.overwrite_ok and os.path.exists(os.path.join(args.out_path, group_name, run_name_)):
    suffix = 1
    while os.path.exists(os.path.join(args.out_path, group_name, f"{run_name}_v{suffix}")):
        suffix += 1
    run_name_ = f"{run_name}_v{suffix}"

out_path = os.path.join(args.out_path, group_name, run_name_)
os.makedirs(out_path, exist_ok=True)

if not args.debug:
    np.save(os.path.join(out_path, 'val_metrics_by_size.npy'), val_metrics_by_size_np)
    np.save(os.path.join(out_path, 'val_metrics_by_size_agg.npy'), val_metrics_by_size_np)
    np.save(os.path.join(out_path, 'sample_preds.npy'), sample_preds)
    np.save(os.path.join(out_path, 'specs.npy'), specs)

# get size of depth_val_metrics_np in megabytes

def calc_size(d):
    if isinstance(d, dict):
        return sum([calc_size(v) for v in d.values()])
    elif isinstance(d, np.ndarray):
        return d.nbytes
    else:
        return 0

val_metrics_by_size_mem = calc_size(val_metrics_by_size_np) / 1e6
val_metrics_by_size_agg_mem = calc_size(val_metrics_by_size_np) / 1e6
sample_preds_mem = sum([sys.getsizeof(s) for s in sample_preds.values()]) / 1e6
specs_mem = sys.getsizeof(specs) / 1e6

print(f"depth_val_metrics_np size: {val_metrics_by_size_mem:.2f} MB")
print(f"depth_val_metrics_agg_np size: {val_metrics_by_size_agg_mem:.2f} MB")
print(f"sample_preds size: {sample_preds_mem:.2f} MB")
print(f"specs size: {specs_mem:.2f} MB")

def add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Max Train # Nodes', line_color='black'):
    fig.add_vline(x=x, line_dash='dash', annotation_text=annotation_text, line_color=line_color, annotation_position='bottom left')

print('Generating plots for results...')

# Final Prediction
print('Final Prediction Metrics')
df = pd.DataFrame({'# Nodes': val_sizes, '% Completely Solved': [val_metrics_by_size_np[d]['final_label_full_acc'] for d in val_sizes]})
for spec_key in ['min_depth', 'mean_depth', 'max_depth', 'min_non_pad_len', 'mean_non_pad_len', 'max_non_pad_len']:
    df.insert(1, spec_key, [specs[d][spec_key] for d in val_sizes])

fig = px.line(df, x='# Nodes', y='% Completely Solved', title='% Completely Solved by Graph Size')
add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Max Train # Nodes')
fig.update_layout(yaxis_tickformat = '.1%')
fig.show()
if args.log_wandb:
    table = wandb.Table(dataframe=df)
    wandb.log({'Summary Metrics/Final Prediction Average Accuracy': np.mean([val_metrics_by_size_np[d]['final_label_full_acc'] for d in val_sizes])})
    wandb.log({'Final Prediction/Final Full Accuracy Table': table})
    wandb.log({'Final Prediction/Final Full Accuracy': fig})
    wandb.log({'Final Prediction/Final Full Accuracy (W&B)': wandb.plot.line(table, x='# Nodes', y='% Completely Solved', title='% Completely Solved by # Nodes')})

# Final Prediction: Factor Accuracy
df = pd.DataFrame(
    {factor: [val_metrics_by_size_np[d]['final_label_factor_acc'][factor_idx] for d in val_sizes] for factor_idx, factor in enumerate(factored_tokenizer.factors[:4])})
df['Graph Size (# Nodes)'] = val_sizes

for spec_key in ['min_depth', 'mean_depth', 'max_depth', 'min_non_pad_len', 'mean_non_pad_len', 'max_non_pad_len']:
    df.insert(1, spec_key, [specs[d][spec_key] for d in val_sizes])

table = wandb.Table(dataframe=df)

fig = px.line(df, x='Graph Size (# Nodes)', y=factored_tokenizer.factors[:4], title='Final Factor Accuracy by Graph Size')
add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Train # Nodes')
fig.update_layout(yaxis_tickformat = '.1%')
fig.show()
if args.log_wandb:
    wandb.log({'Final Prediction/Final Factor Accuracy Table': table})
    wandb.log({'Final Prediction/Final Factor Accuracy': fig})
    wandb.log({'Final Prediction/Final Factor Accuracy (W&B)': wandb.plot.line(table, x='# Nodes', y='% Completely Solved', title='Final Factor Accuracy by # Nodes')})

# attn_score_entropy by graph size
df = pd.DataFrame({'# Nodes': val_sizes, 'Attention Score Entropy': [np.mean(val_metrics_by_size_np[d]['attn_score_entropy']) for d in val_sizes]})
for spec_key in ['min_depth', 'mean_depth', 'max_depth', 'min_non_pad_len', 'mean_non_pad_len', 'max_non_pad_len']:
    df.insert(1, spec_key, [specs[d][spec_key] for d in val_sizes])

fig = px.line(df, x='# Nodes', y='Attention Score Entropy', title='Attention Score Entropy by Graph Size')
add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Max Train # Nodes')
fig.show()
if args.log_wandb:
    table = wandb.Table(dataframe=df)
    wandb.log({'Attention Scores/Entropy Table': table})
    wandb.log({'Attention Scores/Entropy': fig})
    wandb.log({'Attention Scores/Entropy (W&B)': wandb.plot.line(table, x='# Nodes', y='Attention Score Entropy', title='Attention Score Entropy by # Nodes')})

# attn_score_normalized_entropy by graph size
df = pd.DataFrame({'# Nodes': val_sizes, 'Normalized Attention Score Entropy': [np.mean(val_metrics_by_size_np[d]['attn_score_normalized_entropy']) for d in val_sizes]})
for spec_key in ['min_depth', 'mean_depth', 'max_depth', 'min_non_pad_len', 'mean_non_pad_len', 'max_non_pad_len']:
    df.insert(1, spec_key, [specs[d][spec_key] for d in val_sizes])

fig = px.line(df, x='# Nodes', y='Normalized Attention Score Entropy', title='Normalized Attention Score Entropy by Graph Size')
add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Max Train # Nodes')
fig.show()
if args.log_wandb:
    table = wandb.Table(dataframe=df)
    wandb.log({'Attention Scores/Normalized Entropy Table': table})
    wandb.log({'Attention Scores/Normalized Entropy': fig})
    wandb.log({'Attention Scores/Normalized Entropy (W&B)': wandb.plot.line(table, x='# Nodes', y='Normalized Attention Score Entropy', title='Normalized Attention Score Entropy by # Nodes')})

# max_attn_score by graph size
df = pd.DataFrame({'# Nodes': val_sizes, 'Max Attention Score': [np.mean(val_metrics_by_size_np[d]['max_attn_score']) for d in val_sizes]})
for spec_key in ['min_depth', 'mean_depth', 'max_depth', 'min_non_pad_len', 'mean_non_pad_len', 'max_non_pad_len']:
    df.insert(1, spec_key, [specs[d][spec_key] for d in val_sizes])

fig = px.line(df, x='# Nodes', y='Max Attention Score', title='Max Attention Score by Graph Size')
add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Max Train # Nodes')
fig.show()
if args.log_wandb:
    table = wandb.Table(dataframe=df)
    wandb.log({'Attention Scores/Max Score Table': table})
    wandb.log({'Attention Scores/Max Score': fig})
    wandb.log({'Attention Scores/Max Score (W&B)': wandb.plot.line(table, x='# Nodes', y='Max Attention Score', title='Max Attention Score by # Nodes')})


if args.log_wandb:
    wandb.finish()