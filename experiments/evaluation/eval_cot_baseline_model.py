import argparse
import os, sys; sys.path.append(os.path.abspath('../'))

import torch
import numpy as np
import pandas as pd
import plotly.express as px

import yaml
import wandb
import wandb.plot

from tqdm import tqdm
from baseline_models import LitCoTRecurrentTransformerModel, get_experiment_name
from Simtransformer.simtransformer.utils import EasyDict
from tokenizers import CoTTokenizer

from metric_utils import calc_metrics_across_batches, print_model_steps

from torch.utils.data import DataLoader, Dataset
from baseline_cot_train import CoTDataset
from datetime import datetime

parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument('--wandb_project', type=str, default="RecTransformer-DiscIntermRep-Eval")
parser.add_argument('--wandb_entity', type=str, default="transformer-computation-graph")
parser.add_argument('--group_name', type=str, required=True)
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_batches', type=int, default=None)
parser.add_argument('--out_path', type=str, default='results/cot_baselines')
parser.add_argument('--log_wandb', action='store_true')
parser.add_argument('--overwrite_ok', action='store_true', help='Overwrite existing results')
parser.add_argument('--start_graph_size', type=int, default=1, help='Start graph size for evaluation (enables incremental evaluation)')
parser.add_argument('--debug', action='store_true', help='Debug mode (do not log to wandb and do not save results)')

args = parser.parse_args()

if args.debug:
    print('DEBUG MODE')
    args.n_batches = 2

base_dir = os.path.abspath('../')
configs_dir = os.path.join(base_dir, 'configs', 'CoT_configs')

val_ds_path = 'val_datasets/CoT'

group_name = args.group_name
run_name = args.run_name

# load model, train, and data config
config_dir = group_name.replace(' ', '') # calc config dir from wandb group name

with open(os.path.join(configs_dir, config_dir, 'model_config.yaml')) as f:
    model_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

with open(os.path.join(configs_dir, config_dir, 'train_config.yaml')) as f:
    train_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

with open(os.path.join(configs_dir, config_dir, 'data_config.yaml')) as f:
    data_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

# load factored_tokenizer
# with open(os.path.join(base_dir, data_config.data_dir, data_config.tokenizer_file_name), 'rb') as f:
#     factored_tokenizer = pickle.load(f)
tokenizer = CoTTokenizer(
    n_vars=max(data_config.dag_config.num_nodes, data_config.val_dag_config.num_nodes),
    ops=data_config.dag_config.func_vocab, mod_val=data_config.dag_config.mod_val)

# set vocab_size and factors in model_config
model_config.vocab_size = len(tokenizer.vocab)

if data_config.cot_type == 'val' and model_config.max_seq_len < 1344:
    model_config.max_seq_len = 1344
if data_config.cot_type == 'eq-val' and model_config.max_seq_len < 2048:
    model_config.max_seq_len = 2048

# load model
ckpt_path = 'checkpoints/cot_baselines'
ckpt_fname = 'last.ckpt'
model_ckpt_path = os.path.join(base_dir, ckpt_path, f"{group_name}-{run_name}", ckpt_fname) # model checkpoint path from group name
# model_ckpt = torch.load(model_ckpt_path, weights_only=False)

litmodel = LitCoTRecurrentTransformerModel(model_config, train_config, data_config)

litmodel.load_ckpt(model_ckpt_path)

litmodel = litmodel.to('cuda')

# create the data module
# create the data module
func_vocab_val_dir = '_'.join(data_config.dag_config.func_vocab)
cot_type = data_config.cot_type
# get list of .pt files in val_ds_path/func_vocab_val_dir
val_ds_dir = os.path.join(val_ds_path, func_vocab_val_dir, cot_type)

print(f"val_ds_dir: {val_ds_dir}")
if not os.path.exists(val_ds_dir):
    print(f"val_ds_dir does not exist: {val_ds_dir}")
    print(f"Exitting.")
    exit(1)

val_ds_files = [os.path.join(val_ds_dir, f) for f in os.listdir(val_ds_dir) if f.endswith('.pt')]
# match pattern from test_data_N{a}D{b}.pt and get N and D values
val_sizes = sorted([int(fname.split('_')[-1].split('.')[0].split('N')[1]) for fname in val_ds_files])
val_sizes = [n for n in val_sizes if n >= args.start_graph_size] # filter out sizes smaller than start_graph_size

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
        baseline=True,
        cot_baseline=True
    )
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        group=group_name, name=run_name,
        config=experiment_config)

# region eval utils

def autoregressive_generation(litmodel, prompt, max_steps=None, label=None, detect_halt=True, verbose=True, decode_params=None):
    if decode_params is None:
        decode_params = dict(temparature=None, greedy=True)

    prompt = prompt.unsqueeze(0)  # add batch dim
    prompt = prompt.to(litmodel.device)

    if verbose:
        print('PROMPT')
        print(tokenizer.decode_tokens(prompt[0].tolist(), return_string=True))
        print('GENERATED:')

    if max_steps is None and label is not None:
        max_steps = len(label)
    elif max_steps is None:
        max_steps = 1000

    output_tokens = []

    for t in range(len(prompt), max_steps):

        with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
            logits = litmodel.model(prompt)[0, -1]

        if decode_params['greedy']:
            next_token = logits.argmax(-1).unsqueeze(0)
        else:
            logits = logits / decode_params['temperature']
            next_token = torch.multinomial(logits.softmax(-1), 1)

        output_tokens.append(tokenizer.idx2token[next_token.item()])

        prompt = torch.cat([prompt, next_token.unsqueeze(0)], dim=1)

        if detect_halt and next_token == tokenizer.token2idx['<PAD>']:
            if verbose:
                print()
                print('HALTING')
            break

        if verbose:
            print(tokenizer.decode_tokens(next_token.tolist(), return_string=True), end=' ')

    if verbose:
        print()
        print('LABEL:')
        print(tokenizer.decode_tokens(label, return_string=True))

    return output_tokens

def get_example_metrics(pred, label):
    collections = {'var': [], 'value': [], 'operation': [], 'all': []}
    for y, yhat in zip(pred, label):
        collections['all'].append(y == yhat)

        if y in tokenizer.variable_partition:
            collections['var'].append(y == yhat)
        elif y in tokenizer.value_partition:
            collections['value'].append(y == yhat)
        elif y in tokenizer.operation_partition:
            collections['operation'].append(y == yhat)

    metrics = {f'{k}_tokenwise_acc': np.mean(v) for k, v in collections.items()}
    metrics = {**metrics, **{f'{k}_sequence_acc': all(v) for k, v in collections.items()}}
    return metrics

def calc_batch_metrics(litmodel, batch):
    (x, y), (cot_mask, value_mask, var_mask, operation_mask) = batch
    assert x.size(0) == 1, 'batch size must be 1'

    # extract_prompt and cot
    pad_token = tokenizer.token2idx['<PAD>']
    cot_token = tokenizer.token2idx['<COT>']

    cot_mask_ = cot_mask[x!=pad_token]

    x_ = x[x!=pad_token]
    y_ = y[x!=pad_token]

    prompt = x_[~cot_mask_.bool() | (x_ == cot_token)]
    cot = x_[cot_mask_.bool() & (x_ != cot_token)]

    # cot = cot[cot != pad_token] # mask out padding

    prompt_cot = x_ # prompt + cot
    prompt_cot_label = y_ # labels (next-token) for prompt + cot

    with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
        teacher_forcing_preds = litmodel.model(prompt_cot.unsqueeze(0).cuda()).argmax(-1)[0]

    decoded_pred = tokenizer.decode_tokens(teacher_forcing_preds[cot_mask_.bool()].tolist(), return_string=False)
    decoded_label = tokenizer.decode_tokens(prompt_cot_label[cot_mask_.bool()].tolist(), return_string=False)
    teacherforcing_metrics = get_example_metrics(decoded_pred, decoded_label)

    decoded_pred = autoregressive_generation(litmodel, prompt.cuda(), label=prompt_cot_label[cot_mask_.bool()].tolist(), verbose=False, detect_halt=False, decode_params=dict(temperature=None, greedy=True))
    decoded_label = tokenizer.decode_tokens(prompt_cot_label[cot_mask_.bool()].tolist(), return_string=False)
    aut_metrics = get_example_metrics(decoded_pred, decoded_label)


    metrics = {**{f'tf_{k}': v for k, v in teacherforcing_metrics.items()}, **aut_metrics}

    return metrics

def calc_metrics_across_batches(litmodel, dataloader, n_batches=None, verbose=False):

    metrics = dict()

    for i, batch in tqdm(enumerate(dataloader)):
        if n_batches is not None and i >= n_batches:
            break

        batch_metrics = calc_batch_metrics(litmodel, batch)

        for k, v in batch_metrics.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)

    for k, v in metrics.items():
        if isinstance(v, list):
            if isinstance(v[0], torch.Tensor):
                metrics[k] = torch.stack(v, dim=0)

    return metrics


# endregion

# region eval

# infer output path from group name and run name
# if output already saved, increment run name to avoid overwriting

run_name_ = run_name

if not args.overwrite_ok and os.path.exists(os.path.join(args.out_path, group_name, run_name_)):
    suffix = 1
    while os.path.exists(os.path.join(args.out_path, group_name, f"{run_name}_v{suffix}")):
        suffix += 1
    run_name_ = f"{run_name}_v{suffix}"

out_path = os.path.join(args.out_path, group_name, run_name_)
os.makedirs(out_path, exist_ok=True)

# recursively cast Tensor to np.ndarray
def to_numpy(d):
    if isinstance(d, dict):
        return {k: to_numpy(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.cpu().numpy()
    else:
        return d

def save_results(val_metrics_by_size, val_metrics_by_size_agg):
    if not args.debug:
        val_metrics_by_size_np = to_numpy(val_metrics_by_size)
        val_metrics_by_size_np = to_numpy(val_metrics_by_size_agg)
        np.save(os.path.join(out_path, 'val_metrics_by_size.npy'), val_metrics_by_size_np)
        np.save(os.path.join(out_path, 'val_metrics_by_size_agg.npy'), val_metrics_by_size_np)
        print('Saved results to', out_path)

val_metrics_by_size = dict()
val_metrics_by_size_agg = dict()

# evaluate model on different datasets
if args.debug:
    val_sizes = val_sizes[::4] # choose representative subset

start_time = datetime.now()
for n_nodes in val_sizes:

    print()
    print('-'*50)
    print(f"# Nodes: {n_nodes}")
    print(f"Current time: {datetime.now()}")
    print(f'Elapsed time: {datetime.now() - start_time}')

    ds_file = os.path.join(val_ds_dir, f"testdata_N{n_nodes}.pt")
    print(f"Evaluating model on {ds_file}")

    val_ds = CoTDataset(ds_file)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4)

    # evaluate model
    val_metrics = calc_metrics_across_batches(
        litmodel, val_dl, n_batches=args.n_batches, verbose=False)

    val_metrics_by_size[n_nodes] = val_metrics

    val_metrics_by_size_agg[n_nodes] = {k: torch.nanmean(torch.tensor(v).float(), dim=0) for k, v in val_metrics.items()}

    for k, v in val_metrics_by_size_agg[n_nodes].items():
        print(f"{k}: {v}")

    # save (intermediate) results
    save_results(val_metrics_by_size, val_metrics_by_size_agg)

# save final results
save_results(val_metrics_by_size, val_metrics_by_size_agg)

# get size of depth_val_metrics_np in megabytes

def calc_size(d):
    if isinstance(d, dict):
        return sum([calc_size(v) for v in d.values()])
    elif isinstance(d, np.ndarray):
        return d.nbytes
    else:
        return 0


val_metrics_by_size_np = to_numpy(val_metrics_by_size)
val_metrics_by_size_agg_np = to_numpy(val_metrics_by_size_agg)

val_metrics_by_size_mem = calc_size(val_metrics_by_size_np) / 1e6
val_metrics_by_size_agg_mem = calc_size(val_metrics_by_size_np) / 1e6

print(f"depth_val_metrics_np size: {val_metrics_by_size_mem:.2f} MB")
print(f"depth_val_metrics_agg_np size: {val_metrics_by_size_agg_mem:.2f} MB")

# def add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Max Train # Nodes', line_color='black'):
#     fig.add_vline(x=x, line_dash='dash', annotation_text=annotation_text, line_color=line_color, annotation_position='bottom left')

# print('Generating plots for results...')

print('DONE')