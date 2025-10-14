import os
import yaml
import argparse
import pickle
from functools import reduce
import copy
from tqdm import tqdm, trange
import numpy as np
import torch
from dag_generator import generate_abs_dag
from Simtransformer.simtransformer.utils import EasyDict


from tokenizers import CoTTokenizer
from generate_data import replace_variable_name, pad_truncate_seq

current_dir = os.path.dirname(os.path.realpath(__file__))

def generate_cot(equations, values_dict, tokenizer, cot_type):

    cot = []

    for eq in equations:
        lhs, rhs = eq.split(' = ')
        lhs_symbols = lhs.split()

        # CoT is of the form RHS = VAL, for each equation's RHS
        if cot_type == 'val' or (len(lhs_symbols) == 1 and lhs_symbols[0] in tokenizer.value_partition):
            eq_cot = ' '.join([rhs, tokenizer.EQUAL, str(values_dict[rhs])])

        # CoT is of the form RHS = EQUATION = VAL, for each equation's RHS, where EQUATION is the LHS
        elif cot_type == 'eq-val':
            eq_cot = ' '.join([rhs, tokenizer.EQUAL, *lhs_symbols, tokenizer.EQUAL, str(values_dict[rhs])])
        else:
            raise ValueError(f'cot_type {cot_type} is not supported')

        cot.append(eq_cot)

    cot = f' {tokenizer.EQ_SEP} '.join(cot)

    return cot

def generate_cot_dataset(dag_config, tokenizer, cot_type):
    # dag_config: dictionary containing the configuration for the DAG
    # tokenizer: CoTTokenizer object
    # cot_type: type of CoT to generate.
    #   'val' for CoT of the form RHS = VAL,
    #   'eq-val' for CoT of the form RHS = EQUATION = VAL

    num_samples = dag_config['num_samples']
    max_length = dag_config['max_length']

    tokenized_dataset = []

    dataset_info = {'cot_mask': [], 'value_mask': [], 'var_mask': [], 'operation_mask': [], 'num_nodes': []}
    # the first coordinate is the depth of the token, the second to the last are the expressions for computing the token, where the variables are replaced by the first occurrence position of that variable in the equation.

    for num_idx in trange(num_samples, desc="Generating dataset"):

        if dag_config.get('var_length', False) and np.random.random() < dag_config['var_length_prob']:
            # randomly and uniformly sample the number of nodes between 8 and dag_config['num_nodes']
            dag_config_ = copy.deepcopy(dag_config)
            dag_config_['num_nodes'] = np.random.randint(8, dag_config['num_nodes'] + 1)
            dag = generate_abs_dag(dag_config_)
            dataset_info['num_nodes'].append(dag_config_['num_nodes'])
        elif not dag_config['fix_graph']:
            dag = generate_abs_dag(dag_config)
            dataset_info['num_nodes'].append(dag_config['num_nodes'])
        else:
            dataset_info['num_nodes'].append(dag_config['num_nodes'])

        equations, original_indices, depths, values, opers, parents_dict, depth_dict, values_dict = dag.generate_data(shuffle=False, to_string=False)

        # replace the variable names with the sampled nodes
        equations, parents_dict, depth_dict, values_dict = replace_variable_name(tokenizer, equations, parents_dict, depth_dict, values_dict)

        # generate CoT string for the example's equations
        cot_str = generate_cot(equations, values_dict, tokenizer, cot_type)

        # generate input string by concatenating the equations with EQ_SEP
        input_string = f' {tokenizer.EQ_SEP} '.join(equations)

        # concatenate the input string with the CoT string, separated by special COT token
        example_string = f'{input_string} {tokenizer.COT} {cot_str}'

        # split the example string into tokens
        example_tokens = example_string.split()

        # pad
        if len(example_tokens) > max_length:
            raise ValueError(f'Example length {len(example_tokens)} is greater than max_length {max_length}')
        example_tokens = pad_truncate_seq(example_tokens, max_length=max_length, pad_token=tokenizer.PAD)

        # encode the tokens
        tokenized_example = tokenizer.encode_tokens(example_tokens)
        tokenized_dataset.append(tokenized_example)

        # generate masks
        # mask is True for tokens in CoT. We will use this mask to calculate the loss only for the CoT portion, not the input portion.
        cot_mask = [1 if t >= example_tokens.index(tokenizer.COT) else 0 for t, token in enumerate(example_tokens)]
        var_mask = [1 if token in tokenizer.variable_partition else 0 for token in example_tokens]
        value_mask = [1 if token in tokenizer.value_partition else 0 for token in example_tokens]
        operation_mask = [1 if token in tokenizer.operation_partition else 0 for token in example_tokens]

        dataset_info['cot_mask'].append(cot_mask)
        dataset_info['var_mask'].append(var_mask)
        dataset_info['value_mask'].append(value_mask)
        dataset_info['operation_mask'].append(operation_mask)

    tokenized_dataset = torch.tensor(tokenized_dataset)
    print(tokenized_dataset.shape)
    for key in dataset_info:
        dataset_info[key] = torch.tensor(dataset_info[key])

    return tokenized_dataset, dataset_info

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config_path', type=str)#, default=os.path.join(current_dir, 'configs/Nodes64-ADD-L2H16D256_DeBERTa/data_config.yaml'))

    args = parser.parse_args()

    with open(args.data_config_path, 'r') as f:
        data_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    dag_config = data_config['dag_config']
    val_dag_config = data_config['val_dag_config']
    cot_type = data_config['cot_type']

    tokenizer = CoTTokenizer(
        n_vars=max(dag_config['num_nodes'], val_dag_config['num_nodes']),
        ops=dag_config['func_vocab'],
        mod_val=dag_config['mod_val'])

    # serialize and pickle the tokenizer
    os.makedirs(os.path.join(current_dir, data_config['data_dir']), exist_ok=True)

    with open(os.path.join(current_dir, data_config['data_dir'], data_config['tokenizer_file_name']), 'wb') as f:
        pickle.dump(tokenizer, f)


    print('GENERATING TRAINING DATASET')
    print('=' * 50)
    print(f'num_samples: {dag_config["num_samples"]}')
    print(f'num_nodes: {dag_config["num_nodes"]}')
    print(f'var_length: {dag_config.get("var_length", False)}')
    print(f'var_length_prob: {dag_config.get("var_length_prob", 0)}')
    print(f'cot_type: {cot_type}')
    print(f'func_vocab: {dag_config["func_vocab"]}')
    print(f'mod_val: {dag_config["mod_val"]}')
    print('-' * 50)
    print('Generating training dataset...')
    # generate the training dataset
    tokenized_dataset, dataset_info = generate_cot_dataset(dag_config, tokenizer, cot_type)

    print('Value Counts of num_nodes:')
    # Convert tensor to numpy array
    num_nodes = dataset_info['num_nodes'].numpy()
    value_counts = np.bincount(num_nodes) / len(num_nodes)
    for i, count in enumerate(value_counts):
        if count == 0:
            continue
        print(f'num_nodes {i}: {count:.4f}')
    print('-' * 50)
    # print samples of first 5 examples
    print('Printing samples of first 5 examples:')
    for i in range(5):
        print(f"\tExample {i} (num_nodes = {dataset_info['num_nodes'][i]}):")
        print(tokenizer.decode_tokens(tokenized_dataset[i].tolist(), return_string=True))

    print('-' * 50)
    print('Saving training dataset...')
    torch.save({'tokenized_dataset': tokenized_dataset,
                'dataset_info': dataset_info},
                os.path.join(current_dir, data_config['data_dir'], dag_config['data_file_name']))

    print('Training dataset saved!')

    print('GENERATING VALIDATION DATASET')
    print('=' * 50)
    print(f'num_samples: {val_dag_config["num_samples"]}')
    print(f'num_nodes: {val_dag_config["num_nodes"]}')
    print(f'var_length: {val_dag_config.get("var_length", False)}')
    print(f'var_length_prob: {val_dag_config.get("var_length_prob", 0)}')
    print(f'cot_type: {cot_type}')
    print(f'func_vocab: {val_dag_config["func_vocab"]}')
    print(f'mod_val: {val_dag_config["mod_val"]}')
    print('-' * 50)

    print('Generating validation dataset...')
    # generate the validation dataset
    tokenized_dataset, dataset_info = generate_cot_dataset(data_config.val_dag_config, tokenizer, cot_type)

    print('Value Counts of num_nodes:')
    num_nodes = dataset_info['num_nodes'].numpy()
    value_counts = np.bincount(num_nodes) / len(num_nodes)
    for i, count in enumerate(value_counts):
        if count == 0:
            continue
        print(f'num_nodes {i}: {count:.4f}')
    print('-' * 50)

    # print samples of first 5 examples
    print('Printing samples of first 5 examples:')
    for i in range(5):
        print(f"Example {i} (num_nodes = {dataset_info['num_nodes'][i]}):")
        print(tokenizer.decode_tokens(tokenized_dataset[i].tolist(), return_string=True))
    print('-' * 50)

    print('Saving validation dataset...')
    # save the validation dataset
    torch.save({'tokenized_dataset': tokenized_dataset,
                'dataset_info': dataset_info},
                os.path.join(current_dir, data_config['data_dir'], val_dag_config['data_file_name']))

    print('Validation dataset saved!')
    print('Done!')