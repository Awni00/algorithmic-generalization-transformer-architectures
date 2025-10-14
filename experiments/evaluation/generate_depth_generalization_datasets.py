"""
Generate depth generalization datasets for evaluation.

This script creates validation datasets at different graph sizes to test
out-of-distribution (OoD) algorithmic generalization capabilities.
"""

import argparse
import yaml
import os, sys; sys.path.append(os.path.abspath('..'))
import pickle
import torch
from tqdm import tqdm

from generate_data import generate_dataset

from Simtransformer.simtransformer.utils import EasyDict
from tokenizers import FactoredVocabTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate depth generalization datasets")
    parser.add_argument('--data_config_path', type=str, required=True,
                        help='Path to the data configuration YAML file')
    parser.add_argument('--min_nodes', type=int, default=8,
                        help='Minimum number of nodes for evaluation datasets')
    parser.add_argument('--increment', type=int, default=8,
                        help='Node increment step for generating datasets')
    parser.add_argument('--num_samples', type=int, default=1024,
                        help='Number of samples per dataset')

    args = parser.parse_args()

    # Load base data configuration
    with open(args.data_config_path, 'r') as f:
        base_data_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    
    func_vocab = base_data_config.dag_config.func_vocab

    # Create factored tokenizer
    factored_tokenizer = FactoredVocabTokenizer(
        n_vars=max(base_data_config.dag_config.num_nodes, base_data_config.val_dag_config.num_nodes),
        ops=func_vocab,
        mod_val=base_data_config.dag_config.mod_val,
        max_fan_in=base_data_config.dag_config.max_fan_in_deg
    )

    # Base DAG configuration template
    base_dag_config = dict(
        data_file_name=None,  # will be filled
        max_comp_depth=None,  # will be filled
        num_nodes=None,  # will be filled
        max_depth=None,  # will be filled
        func_vocab=func_vocab,  # will be filled

        mod_val=23,
        min_fan_in_deg=1,
        max_fan_in_deg=3,
        fix_graph=False,
        num_leaf_nodes=6,
        shuffle_node_name=True,

        max_length=max(base_data_config.dag_config.max_length, base_data_config.val_dag_config.max_length),
        num_samples=args.num_samples,

        verbose=False,
    )

    # Create output directory path
    depth_gen_ds_paths = f'val_datasets/Tr{base_data_config.dag_config.num_nodes}Test{base_data_config.val_dag_config.num_nodes}'
    print(f"Output directory: {depth_gen_ds_paths}")

    # Generate datasets for different node counts
    max_nodes = max(base_data_config.dag_config.num_nodes, base_data_config.val_dag_config.num_nodes)
    n_nodess = list(range(args.min_nodes, max_nodes + 1, args.increment))

    print(f"Generating datasets for node counts: {n_nodess}")
    print(f"Function vocabulary: {func_vocab}")

    for n_nodes in tqdm(n_nodess, desc="Generating datasets"):
        print('-' * 50)
        print(f'Generating data for N = {n_nodes}')

        dag_config = base_dag_config.copy()

        # Set configuration for this node count
        max_depth = n_nodes
        dag_config['num_nodes'] = n_nodes
        dag_config['max_comp_depth'] = max_depth
        dag_config['max_depth'] = max_depth
        dag_config['data_file_name'] = f'data_N{n_nodes}.pt'

        dag_config = EasyDict(dag_config)

        # Create output paths
        out_dir = os.path.join(depth_gen_ds_paths, '_'.join(args.func_vocab))
        out_path = os.path.join(out_dir, f'testdata_N{n_nodes}.pt')
        os.makedirs(out_dir, exist_ok=True)

        # Generate the dataset
        print(f"Config: {dag_config}")
        tokenized_dataset, dataset_info = generate_dataset(dag_config, factored_tokenizer)

        # Calculate depth statistics
        depths = dataset_info[..., 0].max(dim=-1).values
        min_depth, mean_depth, max_depth = depths.min(), depths.float().mean(), depths.max()
        print(f'Min depth: {min_depth}; Mean depth: {mean_depth:.1f}; Max depth: {max_depth}')

        # Calculate sequence length statistics
        pad_encoding = torch.tensor(factored_tokenizer.encode_string(factored_tokenizer.PAD)[0])
        is_pad = torch.eq(tokenized_dataset, pad_encoding).all(dim=-1)

        orig_seq_len = tokenized_dataset.size(1)
        seq_lens = (~is_pad).sum(dim=1)

        print(f'min/avg/max non-pad length: {seq_lens.min()}/{seq_lens.float().mean():.0f}/{seq_lens.max()}; original length: {orig_seq_len}')

        # Create specification dictionary
        spec = dict(
            min_depth=min_depth.item(),
            mean_depth=mean_depth.item(),
            max_depth=max_depth.item(),
            min_non_pad_len=seq_lens.min().item(),
            mean_non_pad_len=seq_lens.float().mean().item(),
            max_non_pad_len=seq_lens.max().item(),
            orig_seq_len=orig_seq_len,
        )

        # Save the dataset
        torch.save({
            'tokenized_dataset': tokenized_dataset,
            'dataset_info': dataset_info,
            'spec': spec
        }, out_path)
        print(f'Saved to {out_path}')
        print()

    print(f"Dataset generation complete! All datasets saved in: {depth_gen_ds_paths}")


if __name__ == "__main__":
    main()
