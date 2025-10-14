"""
Generate Chain-of-Thought depth generalization datasets for evaluation.

This script creates Chain-of-Thought evaluation datasets at different graph sizes
to test out-of-distribution (OoD) algorithmic generalization capabilities with
intermediate supervision.
"""

import argparse
import yaml
import os
import sys
import pickle
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.abspath('..'))

from baselines.generate_cot_data import generate_cot_dataset
from Simtransformer.simtransformer.utils import EasyDict


def main():
    parser = argparse.ArgumentParser(description="Generate Chain-of-Thought evaluation datasets at different scales")
    parser.add_argument('--data_config_path', type=str, required=True,
                        help='Path to the Chain-of-Thought data configuration file')
    parser.add_argument('--min_nodes', type=int, default=8,
                        help='Minimum number of nodes for evaluation datasets')
    parser.add_argument('--max_nodes', type=int, default=128,
                        help='Maximum number of nodes for evaluation datasets')
    parser.add_argument('--increment', type=int, default=8,
                        help='Step size for increasing node counts')
    parser.add_argument('--num_samples', type=int, default=1024,
                        help='Number of samples per evaluation dataset')
    parser.add_argument('--cot_type', type=str, default='eq-val',
                        help='Chain-of-Thought supervision type (e.g., eq-val, step-by-step)')
    parser.add_argument('--output_dir', type=str, default='val_datasets/CoT',
                        help='Output directory for generated datasets')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum sequence length for tokenized expressions')
    parser.add_argument('--func_vocab', type=str, nargs='+', default=['ADD'],
                        help='Function vocabulary to use')

    args = parser.parse_args()

    # Load base configuration
    print(f"Loading configuration from: {args.data_config_path}")
    with open(args.data_config_path, 'r') as f:
        base_data_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # Load tokenizer
    tokenizer_path = os.path.join('..', base_data_config.data_dir, base_data_config.tokenizer_file_name)
    print(f"Loading tokenizer from: {tokenizer_path}")

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Base DAG configuration template
    base_dag_config = dict(
        data_file_name=None,  # will be filled per dataset
        max_comp_depth=None,  # will be filled per dataset
        num_nodes=None,       # will be filled per dataset
        max_depth=None,       # will be filled per dataset
        func_vocab=None,      # will be filled per dataset

        mod_val=23,
        min_fan_in_deg=1,
        max_fan_in_deg=3,
        fix_graph=False,
        num_leaf_nodes=6,
        shuffle_node_name=True,

        max_length=args.max_length,
        num_samples=args.num_samples,
        verbose=False,
    )

    # Use function vocabulary from args or config
    if args.func_vocab != ['ADD']:
        func_vocab = args.func_vocab
    else:
        func_vocab = base_data_config.dag_config.func_vocab if hasattr(base_data_config.dag_config, 'func_vocab') else ['ADD']

    # Generate range of node counts
    n_nodes_list = list(range(args.min_nodes, args.max_nodes + 1, args.increment))

    print(f"Generating CoT evaluation datasets for node counts: {n_nodes_list}")
    print(f"Function vocabulary: {func_vocab}")
    print(f"CoT type: {args.cot_type}")
    print(f"Samples per dataset: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")

    for n_nodes in tqdm(n_nodes_list, desc="Generating CoT datasets"):
        print('-' * 50)
        print(f'Generating data for N = {n_nodes}')

        # Create configuration for this node count
        dag_config = base_dag_config.copy()
        max_depth = n_nodes  # Don't limit max_depth

        dag_config['num_nodes'] = n_nodes
        dag_config['max_comp_depth'] = max_depth
        dag_config['max_depth'] = max_depth
        dag_config['func_vocab'] = func_vocab
        dag_config['data_file_name'] = f'data_N{n_nodes}.pt'

        dag_config = EasyDict(dag_config)

        # Create output directory and path
        out_dir = os.path.join(args.output_dir, '_'.join(func_vocab), args.cot_type)
        out_path = os.path.join(out_dir, f'testdata_N{n_nodes}.pt')
        os.makedirs(out_dir, exist_ok=True)

        print(f"Configuration: {dag_config}")

        # Generate the dataset
        try:
            tokenized_dataset, dataset_info = generate_cot_dataset(dag_config, tokenizer, cot_type=args.cot_type)
        except Exception as e:
            print(f"Error generating dataset for N={n_nodes}: {e}")
            continue

        # Calculate sequence length statistics
        pad_encoding = torch.tensor(tokenizer.encode_string(tokenizer.PAD)[0])
        is_pad = torch.eq(tokenized_dataset, pad_encoding)

        orig_seq_len = tokenized_dataset.size(1)
        seq_lens = (~is_pad).sum(dim=1)

        print(f'min/avg/max non-pad length: {seq_lens.min()}/{seq_lens.float().mean():.0f}/{seq_lens.max()}; original length: {orig_seq_len}')

        # Create dataset specification
        spec = dict(
            min_non_pad_len=seq_lens.min().item(),
            mean_non_pad_len=seq_lens.float().mean().item(),
            max_non_pad_len=seq_lens.max().item(),
            orig_seq_len=orig_seq_len,
            cot_type=args.cot_type,
            func_vocab=func_vocab,
        )

        print(f"Saving to: {out_path}")

        # Save the dataset
        torch.save({
            'tokenized_dataset': tokenized_dataset,
            'dataset_info': dataset_info,
            'spec': spec
        }, out_path)

        print(f"Dataset N={n_nodes} saved successfully")
        print()

    print("\n" + "=" * 50)
    print("All CoT evaluation datasets generated successfully!")
    print(f"Output directory: {args.output_dir}")
    print(f"Generated {len(n_nodes_list)} datasets with CoT type: {args.cot_type}")


if __name__ == "__main__":
    main()
