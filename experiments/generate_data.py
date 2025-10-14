import os
import yaml
import argparse
import pickle
from functools import reduce
import copy
from tqdm import tqdm, trange
from dag_generator import generate_abs_dag
import torch
from Simtransformer.simtransformer.utils import EasyDict

from tokenizers import FactoredVocabTokenizer

# get the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))

def compute_out_eq_at_each_depth(factored_in_eqs,
    depths, values, factored_tokenizer,
    max_depth=None, tokenize=False):

    VALUE_FACTOR_IDX = factored_tokenizer.factors.index('VALUE')

    if max_depth is None:
        max_depth = max(depths)

    factored_eqs_per_depth = [] # list of shape [max_depth+1, n_eqs, n_factors]

    for comp_depth in range(1):
        factored_out_eqs_depth = copy.deepcopy(factored_in_eqs)

        # NOTE: we are not preparing according to depth, the information will be calculated by the teacher model when training. Deprecating the following code.

        for out_eq, eq_depth, val in zip(factored_out_eqs_depth, depths, values):
        #     # in out_eq, 'EMPTY' is replaced with the value of the variable if eq_depth < comp_depth

        #     # if eq_depth < comp_depth:

        #     # eq has form of '1 = x_1' or 'x_1 ADD x_2 = x_3'; so [-1] extracts the "result" token
        #     # this has factors of form ['VARIABLE', x_i, 'N/A', 'EMPTY']
        #     # we replace 'EMPTY' with the value of the variable
            out_eq[-1][VALUE_FACTOR_IDX] = str(val)

        if tokenize:
            factored_out_eqs_depth = [factored_tokenizer.encode_factored_tokens(eq) for eq in factored_out_eqs_depth]
        factored_eqs_per_depth.append(factored_out_eqs_depth)

    return factored_eqs_per_depth[0] # (n_eqs, len_of_each_eq, n_factors)

def collate_factored_eqs_by_depth(factored_eqs_per_depth, factored_tokenizer, tokenize=False):
    if tokenize:
        eq_separator_factored_token = factored_tokenizer.encode_string(factored_tokenizer.EQ_SEP)
    else:
        eq_separator_factored_token = [factored_tokenizer.factor_token(factored_tokenizer.EQ_SEP)]
    # collated = [reduce(lambda x, y: x + eq_separator_factored_token + y, factored_eqs_per_depth[depth]) for depth in range(len(factored_eqs_per_depth))]
    collated = [reduce(lambda x, y: x + eq_separator_factored_token + y, factored_eqs_per_depth)]
    return collated[0]

def pad_truncate_seq(seq, max_length, pad_token):
    if len(seq) < max_length:
        return seq + [pad_token] * (max_length - len(seq))
    else:
        return seq[:max_length]

def pad_truncate_seqs(seqs, max_length, pad_token):
    return [pad_truncate_seq(seq, max_length, pad_token) for seq in seqs]

def index_token_with_variable(variable, factored_eqs, factored_tokenizer):
    for i in range(len(factored_eqs)):
        if factored_eqs[i][factored_tokenizer.factors.index('VARIABLE')] == variable:
            return i

def locate_expressions_in_factored_eqs(token, factored_eqs, factored_tokenizer):
    '''
    Input the tokenized_eqs_by_depth that is not encoded
    '''
    if token[factored_tokenizer.factors.index('SYNTAX')] != 'VARIABLE':
        return [-1, -1]

    # search for the first position in factored_eqs that this token appears
    idx = index_token_with_variable(token[factored_tokenizer.factors.index('VARIABLE')], factored_eqs, factored_tokenizer)
    idx = idx - 1 # the index for the equal sign

    # search for the last occurence of eq_sep token
    eq_separator_factored_token = factored_tokenizer.factor_token(factored_tokenizer.EQ_SEP)
    eq_len = (factored_eqs[idx - 1::-1] + [eq_separator_factored_token]).index(eq_separator_factored_token)

    return [idx - eq_len, idx]

import random
def replace_variable_name(factorized_tokenizer, equations, parents_dict, depth_dict, values_dict):
    original_nodes = list(values_dict.keys())
    candidate_nodes = factorized_tokenizer.variable_partition
    # from candidate_nodes, we sample the same number of nodes as the original_nodes without replacement
    sampled_nodes = random.sample(candidate_nodes, len(original_nodes))
    # we build a dictionary that maps the original nodes to the sampled nodes
    sampled_nodes_dict = dict(zip(original_nodes, sampled_nodes))

    equation_split = [eq.split(' ') for eq in equations]
    for i, eq in enumerate(equation_split):
        for j, token in enumerate(eq):
            if token in sampled_nodes_dict:
                equation_split[i][j] = sampled_nodes_dict[token]
    equations = [' '.join(eq) for eq in equation_split]

    # also replace the keys in parents_dict, depth_dict, and values_dict
    parents_dict = {sampled_nodes_dict[k]: [sampled_nodes_dict[v] for v in vs] for k, vs in parents_dict.items()}
    depth_dict = {sampled_nodes_dict[k]: v for k, v in depth_dict.items()}
    values_dict = {sampled_nodes_dict[k]: v for k, v in values_dict.items()}
    return equations, parents_dict, depth_dict, values_dict

## Generate dataset

def generate_dataset(dag_config, factored_tokenizer):
    num_samples = dag_config['num_samples']
    max_length = dag_config['max_length']
    max_comp_depth = dag_config['max_comp_depth']
    max_fan_in_deg = dag_config['max_fan_in_deg']

    dataset = []
    tokenized_dataset = []

    dataset_info = torch.ones([num_samples, max_length, max_fan_in_deg * 2], dtype=torch.long) * (-1)
    # the first coordinate is the depth of the token, the second to the last are the expressions for computing the token, where the variables are replaced by the first occurrence position of that variable in the equation.

    for num_idx in trange(num_samples, desc="Generating dataset"):
        if not dag_config['fix_graph']:
            dag = generate_abs_dag(dag_config)
        equations, original_indices, depths, values, opers, parents_dict, depth_dict, values_dict = dag.generate_data(shuffle=False, to_string=False)

        # replace the variable names with the sampled nodes
        equations, parents_dict, depth_dict, values_dict = replace_variable_name(factored_tokenizer, equations, parents_dict, depth_dict, values_dict)

        factored_in_eqs = [factored_tokenizer.factor_string(eq) for eq in equations]

        factored_eqs_per_depth = compute_out_eq_at_each_depth(factored_in_eqs, depths, values, factored_tokenizer, max_depth=max_comp_depth, tokenize=False) # The depth is not playing a role here
        # (max_depth+1, n_eqs, len_of_each_eq, n_factors)

        tokenized_eqs_by_depth = collate_factored_eqs_by_depth(factored_eqs_per_depth, factored_tokenizer, tokenize=False)
        # (n_eqs * (len_of_each_eq + 1), n_factors), where the added 1 is for the EQ_SEP token

        # NOTE: we are preparing an additional list that will record the parents' position of each token in the above list. We also record the depth of each token.

        for i, token in enumerate(tokenized_eqs_by_depth):
            if token[factored_tokenizer.factors.index('SYNTAX')] == 'VARIABLE' and token[factored_tokenizer.factors.index('VALUE')] != 'EMPTY': # if the token is a variable and is on the right hand side of the equation
                dataset_info[num_idx, i, 0] = depth_dict[token[factored_tokenizer.factors.index('VARIABLE')]]
                a, b = locate_expressions_in_factored_eqs(token, tokenized_eqs_by_depth, factored_tokenizer) # the first and last position + 1 of the expression that computes the token

                # Next, we will index the first occurrence of the variables in the expression
                expression = tokenized_eqs_by_depth[a:b]
                for j, token in enumerate(expression):
                    if token[factored_tokenizer.factors.index('SYNTAX')] == 'VARIABLE':
                        # If the token is a variable, we put its first occurrence position in the expression
                        dataset_info[num_idx, i, j + 1] = index_token_with_variable(token[factored_tokenizer.factors.index('VARIABLE')], tokenized_eqs_by_depth, factored_tokenizer)
                    if token[factored_tokenizer.factors.index('SYNTAX')] == 'VALUE':
                        # If the token is a value, we put the position of the token in the expression
                        dataset_info[num_idx, i, j + 1] = i - 2 # the position of the value token in the expression
                    elif token[factored_tokenizer.factors.index('SYNTAX')] == 'OPERATION':
                        # If the token is an operation, we put the operation token in the expression
                        dataset_info[num_idx, i, j + 1] = factored_tokenizer.encode_factored_token(token)[factored_tokenizer.factors.index('OPERATION')]

        PAD_TOKEN = factored_tokenizer.factor_token(factored_tokenizer.PAD)
        tokenized_eqs_by_depth = pad_truncate_seq(tokenized_eqs_by_depth, max_length=max_length, pad_token=PAD_TOKEN)

        # encode
        tokenized_eqs_by_depth = factored_tokenizer.encode_factored_tokens(tokenized_eqs_by_depth)

        tokenized_dataset.append(tokenized_eqs_by_depth)

    tokenized_dataset = torch.tensor(tokenized_dataset)
    print(tokenized_dataset.shape)

    return tokenized_dataset, dataset_info



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config_path', type=str)

    args = parser.parse_args()

    with open(args.data_config_path, 'r') as f:
        data_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    dag_config = data_config['dag_config']
    val_dag_config = data_config['val_dag_config']

    factored_tokenizer = FactoredVocabTokenizer(
        n_vars=max(dag_config['num_nodes'], val_dag_config['num_nodes']),
        ops=dag_config['func_vocab'],
        mod_val=dag_config['mod_val'],
        max_fan_in=dag_config['max_fan_in_deg'])

    # serialize and pickle the tokenizer
    os.makedirs(os.path.join(current_dir, data_config['data_dir']), exist_ok=True)

    with open(os.path.join(current_dir, data_config['data_dir'], data_config['tokenizer_file_name']), 'wb') as f:
        pickle.dump(factored_tokenizer, f)

    # generate the training dataset
    tokenized_dataset, dataset_info = generate_dataset(dag_config, factored_tokenizer)

    torch.save({'tokenized_dataset': tokenized_dataset,
                'dataset_info': dataset_info},
                os.path.join(current_dir, data_config['data_dir'], dag_config['data_file_name']))

    # generate the validation dataset
    tokenized_dataset, dataset_info = generate_dataset(data_config.val_dag_config, factored_tokenizer)
    torch.save({'tokenized_dataset': tokenized_dataset,
                'dataset_info': dataset_info},
                os.path.join(current_dir, data_config['data_dir'], val_dag_config['data_file_name']))