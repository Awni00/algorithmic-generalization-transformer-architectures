from collections.abc import MutableMapping
import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
import yaml, json, os, csv
from lightning.pytorch.utilities.parsing import AttributeDict
import random
from typing import Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections.abc import Mapping
import operator
from scipy.linalg import logm
import warnings

# Suppress specific warnings related to logm
warnings.filterwarnings("ignore", message="logm result may be inaccurate")


def shuffle_with_indices(data: list, indices: Union[range, list]):
    combined = list(zip(data, indices))
    random.shuffle(combined)
    new_data, original_indices = zip(*combined)
    original_indices = sorted(range(len(original_indices)), key=lambda k: original_indices[k])
    return new_data, original_indices

def flatten_dict(dictionary, prefix='', separator='.'):
    """
    Flattens a nested dictionary by concatenating keys.

    Args:
        dictionary (dict): The dictionary to flatten.
        prefix (str, optional): A prefix to add to the keys. Defaults to ''.
        separator (str, optional): The separator to use between concatenated keys. Defaults to '.'.

    Returns:
        dict: A new dictionary with flattened keys.
    """
    items = []
    for key, value in dictionary.items():
        key_prefix = prefix + separator + key if prefix else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, key_prefix, separator=separator).items())
        else:
            items.append((key_prefix, value))
    return dict(items)

class CosineAnnealingWarmup(_LRScheduler):
    """
    Implements a learning rate scheduler that combines a warmup phase with a cosine annealing decay.
    The learning rate increases linearly from 0 to the specified learning rate during the warmup phase.
    After the warmup phase, the learning rate follows a cosine decay schedule down to a minimum learning rate.
    Attributes:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for the warmup phase.
        learning_rate (float): Maximum learning rate after warmup.
        min_lr (float): Minimum learning rate after decay.
        lr_decay_steps (int): Total number of steps for the decay phase.
        verbose (bool): If True, prints a message to stdout for each update.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int,
            learning_rate: Union[float, list[float]],
            min_lr: float,
            lr_decay_steps: int,
            verbose: bool = False,
    ):
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        if isinstance(self.learning_rate, float):
            self.learning_rate = [self.learning_rate]
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        
         # Try to pass verbose parameter, fall back if not supported
        try:
            super().__init__(optimizer=optimizer, last_epoch=-1, verbose=verbose)
        except TypeError:
            # Fallback for older PyTorch versions that don't support verbose
            super().__init__(optimizer=optimizer, last_epoch=-1)
        self._step_count = 0

    def get_warmup_lr(self) -> list[float]:
        return [lr * self._step_count / self.warmup_steps for lr in self.learning_rate] 
    
    def get_decay_lr(self, coeff: float) -> list[float]:
        """
        coeff is the cosine coefficient.
        """
        return [self.min_lr + coeff * (lr - self.min_lr) for lr in self.learning_rate]
    
    def get_min_lr(self) -> list[float]:
        return [self.min_lr for _ in range(len(self.learning_rate))] 
    
    def get_lr(self) -> list[float]:
        """
        We need to override the default behavior of the `get_lr` method to support warmup and decay.
        
        We also handle two cases: 
        1. parameter group having a single 'lr' value; 
        2. parameter group having multiple learning rates under the 'group_lrs' key. 
        """
        # Case 1: warmup phase
        if self._step_count < self.warmup_steps:
            return self.get_warmup_lr()
            # return [self.learning_rate * self._step_count / self.warmup_steps
            #         for group in self.optimizer.param_groups]
        
        # Case 2: Post-decay phase
        elif self._step_count > self.lr_decay_steps:
            return self.get_min_lr() 
        
        decay_ratio = (
            (self._step_count - self.warmup_steps)
            / (self.lr_decay_steps - self.warmup_steps)
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.get_decay_lr(coeff)
        
        
def MRR_fn(
    y_hat: torch.Tensor,
    y: torch.Tensor,
): 
    """
    Compute mean reciprocal rank, which is the average of the reciprocal ranks of the true labels.
    
    Args:
    - y_hat: tensor of shape (batch_size, num_classes)
    - y: tensor of shape (batch_size, )
    
    Returns:
    - mrr: tensor of shape (1, )
    """
    _, indices = torch.topk(y_hat, k=y_hat.shape[1]) # shape: (batch_size, num_classes), indices of top k values
    ranks = (indices == y.unsqueeze(1)).float() # indices == y.unsqueeze(1) will give a true value at the ranking position for the true label
    ranks = torch.argmax(ranks, dim=1) + 1 # get the rank of the true label
    return torch.mean(1.0 / ranks.float())


class EasyDict(AttributeDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = EasyDict(v)
            else:
                self[k] = v

    def update(self, other_dict, recursive=True):
        """This function overwrites the original update method of the dictionary class. It updates the dictionary with another dictionary. The hierarchical construction of EasyDict is called recursively.
            
        Args:
            other_dict: a dictionary to update the current dictionary with.
            recursive: a boolean to indicate whether to recursively update the dictionary. Defaults to True.
            If false, this function will rewrite the direct key-value pairs of the current dictionary without checking for nested dictionaries (and they will be overwritten).
            If true, this function is more like adding the key-value pairs (if we consider the flattened form of the dictionary).
        """
        # if other_dict is empty, return
        if not other_dict:
            return
        for k, v in other_dict.items():
            if isinstance(v, dict):
                if recursive == False or k not in self.keys():
                    self[k] = EasyDict(v)
                else:
                    if not isinstance(self[k], EasyDict):
                        self[k] = EasyDict(self[k])
                    self[k].update(v, recursive=True)
            else:
                self[k] = v
                
    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, EasyDict):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out
    
    def update_from_file(self, path):
        """This function updates the dictionary with the contents of a file using the `update` method.

        Args:
            path (_type_): path to the file to load the dictionary from.

        Raises:
            NotImplementedError: if the file extension is not supported.
        """
        self.update(clever_load(path)) # update the current dictionary with the loaded dictionary
        
        
    def save_to_file(self, path):
        out = self.to_dict()
        clever_save(out, path)
    
    def flatten(self, prefix='', separator='.'):
        """Return a flattened version of the dictionary.

        Args:
            prefix (str, optional): the prefix to the be added to each key. Defaults to ''.
            separator (str, optional): Defaults to '.'.

        Returns:
            EasyDict: a flattened version of the dictionary.
        """
        return AttributeDict(flatten_dict(self, prefix, separator))
    
    def setattr_with_string(self, key, value):
        keys = key.split(".")
        current = self
        for i, k in enumerate(keys[:-1]):
            if isinstance(current, EasyDict):
                if k not in current.keys():
                    current[k] = EasyDict()
                current = current[k]
            else:
                raise ValueError(f"Cannot set attribute {'.'.join(keys[:i])} as {'.'.join(keys[:i-1])} has value that is not an EasyDict.")
        k = keys[-1]
        if isinstance(current, EasyDict):
            current.update({k: value})
        else:
            raise ValueError(f"Cannot set attribute {key} as {'.'.join(keys[:-1])} has value that is not an EasyDict.")
        
    def popfirst(self):
        """Pop the first key-value pair from the dictionary.

        Returns:
            tuple: the first key-value pair.
        """
        key = list(self.keys())[0]
        value = self[key]
        del self[key]
        return key, value
        

def clever_load(file_path):
    """Support loading from both .yaml, .json, and .pth files.

    Args:
        file_path (str): path to the file to load.
    """
    if file_path.endswith(".yaml"):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_path.endswith(".pth"):
        return torch.load(file_path)
    elif file_path.endswith(".csv"):
        # with open(file_path, "r", newline="") as f:
        #     reader = csv.reader(f)
        #     obj =[row for row in reader]
        #     return obj
        df = pd.read_csv(file_path)
        return df
    elif file_path.endswith(".txt"):
        with open(file_path, 'r') as file:
            vocab_list = [line.strip() for line in file]
        return vocab_list
                
    else:
        raise NotImplementedError(f"File extension {file_path.split('.')[-1]} is not supported!")
    
def clever_save(obj, file_path):
    """Support saving to both .yaml, .json, and .pth files.

    Args:
        obj (_type_): the object to save.
        file_path (str): path to save the object to.
    """
    # check if the file directory exists
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if file_path.endswith(".yaml"):
        with open(file_path, "w") as f:
            yaml.dump(obj, f)
    elif file_path.endswith(".json"):
        with open(file_path, "w") as f:
            json.dump(obj, f)
    elif file_path.endswith(".pth"):
        torch.save(obj, file_path)
    elif file_path.endswith(".csv"):
        # with open(file_path, "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(obj)
        obj.to_csv(file_path, index=False)
    else:
        raise NotImplementedError(f"File extension {file_path.split('.')[-1]} is not supported!")
    
    
    
def estimate_mfu(self, fwdbwd_per_iter, dt, num_layers, num_heads, dim_per_head, seq_len):
    """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS 
    
    first estimate the number of flops we do per iteration.
    see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    """
    N = self.get_num_params()
    flops_per_token = 6 * N + 12 * num_layers * num_heads * dim_per_head * seq_len
    flops_per_fwdbwd = flops_per_token * seq_len
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0 / dt)  # per second
    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu

def token_accuracy(y_hat, y):
    """
    Computes the token prediction accuracy.

    Parameters:
    y_hat (torch.Tensor): Predictions from the model of shape (batch_size, num_classes).
    y (torch.Tensor): Ground truth labels of shape (batch_size,).

    Returns:
    float: The accuracy of token predictions.
    """
    # Get the predicted class by taking the argmax over the class dimension
    predicted = torch.argmax(y_hat, dim=1)
    
    # Compare predictions with the ground truth labels
    correct_predictions = (predicted == y).sum().item()
    
    # Compute accuracy
    accuracy = correct_predictions / y.size(0)
    
    return accuracy

import copy
def check_cosine_similarity(embedding, 
                            target_embedding=None, 
                            verbose=False, 
                            emb_label=None, 
                            target_label=None, 
                            max_size=16,
                            title=None, 
                            diag_only=False, 
                            figsize=(5, 3), 
                            return_tensor=False):
    # check the cosine similarity between the embeddings
    if not diag_only:
        if isinstance(embedding, torch.Tensor):
            embedding_np = embedding.cpu().detach().numpy()
        else:
            embedding_np = embedding
        if target_embedding is None:
            cos_sim = cosine_similarity(embedding_np, embedding_np)
        else:
            target_embedding_np = target_embedding.cpu().detach().numpy()
            cos_sim = cosine_similarity(embedding_np, target_embedding_np)
        if verbose:
            if target_embedding is None:
                plt.figure(figsize=(max_size + 2, max_size))
                sns.heatmap(cos_sim, annot=False)
                if emb_label is not None:
                    fontsize = max(6, 12 - len(emb_label) // 10)
                    plt.xticks(ticks=np.arange(len(emb_label)), labels=emb_label, rotation=90, fontsize=fontsize)
                    plt.yticks(ticks=np.arange(len(emb_label)), labels=emb_label, rotation=0, fontsize=fontsize)
                if title is not None:
                    plt.title(title)
                plt.show()
            else:
                # select a size according to the relative length of embedding and target_embedding
                size = (len(embedding_np), len(target_embedding_np))
                # control the maximum size of the heatmap to be 16 while keeping the aspect ratio
                if size[0] > max_size:
                    size = (max_size, int(max_size * size[1] / size[0]))
                elif size[1] > max_size:
                    size = (int(max_size * size[0] / size[1]), max_size)
                size = (size[1] + 2, size[0])
                plt.figure(figsize=size)
                sns.heatmap(cos_sim, annot=False)
                if emb_label is not None:
                    fontsize = max(6, 12 - len(emb_label) // 10)
                    plt.xticks(ticks=np.arange(len(emb_label)), labels=emb_label, rotation=90, fontsize=fontsize)
                if target_label is not None:
                    fontsize = max(6, 12 - len(target_label) // 10)
                    plt.yticks(ticks=np.arange(len(target_label)), labels=target_label, rotation=0, fontsize=fontsize)
                if title is not None:
                    plt.title(title)
                plt.show()
        return cos_sim if not return_tensor else torch.tensor(cos_sim, dtype=torch.float32)
    else:
        if isinstance(embedding, torch.Tensor):
            embedding_np = embedding.cpu().detach().numpy()
        else:
            embedding_np = copy.deepcopy(embedding)
            embedding = torch.tensor(embedding, dtype=torch.float32)
        if isinstance(target_embedding, torch.Tensor):
            target_embedding_np = target_embedding.cpu().detach().numpy()
        elif target_embedding is None:
            raise ValueError("target_embedding cannot be None when diag_only is True.")
        else:
            target_embedding_np = copy.deepcopy(target_embedding)
            target_embedding = torch.tensor(target_embedding, dtype=torch.float32)
        inner_product = torch.sum(embedding * target_embedding, dim=-1).cpu().detach().numpy()
        norm_1 = torch.norm(embedding, dim=-1).cpu().detach().numpy()
        norm_2 = torch.norm(target_embedding, dim=-1).cpu().detach().numpy()
        cos_sim = inner_product / (norm_1 * norm_2)
        if verbose:
            # use histogram to show the distribution of cosine similarity
            plt.figure(figsize=figsize)
            sns.histplot(cos_sim, bins='auto')
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Frequency")
            if title is not None:
                plt.title(title)
            plt.show()
        return cos_sim if not return_tensor else torch.tensor(cos_sim, dtype=torch.float32)

def calculate_l2_similarity(input, 
                            target, 
                            verbose=False, 
                            title=None, 
                            figsize=(5, 3), 
                            return_tensor=False):
    l2_dist = torch.norm(input - target, dim=-1)
    norm_1 = torch.norm(input, dim=-1)
    norm_2 = torch.norm(target, dim=-1)
    l2_dist_similarity = (l2_dist / torch.sqrt(norm_1 * norm_2)).squeeze()
    if verbose:
        # histogram of the l2_dist_similarity
        plt.figure(figsize=figsize)
        sns.histplot(l2_dist_similarity.detach().cpu().numpy(), bins='auto')
        plt.xlabel("L2 Distance Similarity")
        plt.ylabel("Frequency")
        if title is not None:
            plt.title(title)
        plt.show()
    return l2_dist_similarity.cpu().detach().numpy() if not return_tensor else l2_dist_similarity

# check the largest off-diagonal element
def check_largest_off_diag(cos_sim, verbose=True):
    cos_sim_off_diag = cos_sim - torch.diag(torch.diag(cos_sim))
    largest_off_diag = cos_sim_off_diag.max()
    largest_off_diag_indices = (cos_sim_off_diag == largest_off_diag).nonzero(as_tuple=True)
    if verbose:
        print(f'largest off-diagonal element: {largest_off_diag}')
        print(f'indices of largest off-diagonal element: {largest_off_diag_indices}')

    return largest_off_diag, largest_off_diag_indices

def check_diag_average(cos_sim, verbose=True):
    cos_sim_diag = torch.diag(cos_sim)
    average_diag = cos_sim_diag.mean()
    if verbose:
        print(f'average of the diagonal: {average_diag}')
    return average_diag

import torch
import torch.jit as jit
import concurrent.futures

def extract_diagonals(tensor, row_idx, diag_idx, dim1=-2, dim2=-1):
    """
    Extracts the specified diagonals from a tensor for selected rows.

    Args:
        tensor (torch.Tensor): The input tensor, assumed to be at least 2D.
        row_idx (list): The list of row indices to select from each diagonal.
        diag_idx (list): The list of diagonal indices to extract.
        dim1 (int): The first dimension to use for extracting diagonals.
        dim2 (int): The second dimension to use for extracting diagonals.

    Returns:
        torch.Tensor: A tensor of shape (..., len(row_idx), len(diag_idx)).
    """
    # Initialize the output tensor to store the selected diagonals.
    result_shape = (*tensor.shape[:-2], len(row_idx), len(diag_idx))
    result = torch.empty(result_shape, dtype=tensor.dtype, device=tensor.device)

    row_idx = torch.tensor(row_idx, device=tensor.device)

    def extract_single_diagonal(j, diag):
        diagonal = tensor.diagonal(offset=diag, dim1=dim1, dim2=dim2)
        if diag < 0:
            # If the diagonal is below the main diagonal, we need to shift the row indices accordingly.
            row_idx_shifted = row_idx + diag
        else:
            row_idx_shifted = row_idx
        return j, diagonal[..., row_idx_shifted]

    # Use ThreadPoolExecutor to parallelize diagonal extraction.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_single_diagonal, j, diag) for j, diag in enumerate(diag_idx)]
        for future in concurrent.futures.as_completed(futures):
            j, selected_diagonal = future.result()
            result[..., :, j] = selected_diagonal

    return result


def _matrix_power(matrix, power):
    """Compute the matrix to the given power using SVD."""
    # Use CPU for SVD to speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)

import torch.optim as optim
class Shampoo(optim.Optimizer):
    r"""Implements the Shampoo optimizer algorithm.

    Shampoo: Preconditioned Stochastic Tensor Optimization.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate (default: 1e-1).
        momentum (float): Momentum factor (default: 0).
        weight_decay (float): Weight decay factor (default: 0).
        epsilon (float): Epsilon for numerical stability (default: 1e-4).
        update_freq (int): Update frequency for computing the matrix inverse (default: 1).
    """

    def __init__(
        self,
        params,
        lr=1e-1,
        momentum=0,
        weight_decay=0,
        epsilon=1e-4,
        update_freq=1,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if update_freq < 1:
            raise ValueError(f"Invalid update_freq value: {update_freq}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
        )
        super(Shampoo, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            Optional[float]: The loss if a closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        state[f"precond_{dim_id}"] = group["epsilon"] * torch.eye(dim, out=grad.new(dim, dim))
                        state[f"inv_precond_{dim_id}"] = grad.new(dim, dim).zero_()

                # Apply momentum
                if momentum > 0:
                    grad.mul_(1 - momentum).add_(state["momentum_buffer"], alpha=momentum)

                # Apply weight decay
                if weight_decay > 0:
                    grad.add_(p.data, alpha=weight_decay)

                # Preconditioning update
                for dim_id, dim in enumerate(grad.size()):
                    precond = state[f"precond_{dim_id}"]
                    inv_precond = state[f"inv_precond_{dim_id}"]

                    # Reshape gradient for matrix multiplication
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    precond.add_(grad @ grad_t)
                    if state["step"] % group["update_freq"] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / order))

                    if dim_id == order - 1:
                        # Final preconditioned gradient
                        grad = grad_t @ inv_precond
                        grad = grad.view(original_size)
                    else:
                        grad = inv_precond @ grad
                        grad = grad.view(transposed_size)

                # Update step
                state["step"] += 1
                state["momentum_buffer"] = grad
                p.data.add_(grad, alpha=-group["lr"])

        return loss
    
    
class signSGD(optim.Optimizer):

    def __init__(self, params, lr=0.01, rand_zero=True):
        defaults = dict(lr=lr)
        self.rand_zero = rand_zero
        super(signSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # take sign of gradient
                grad = torch.sign(p.grad)

                # randomise zero gradients to ±1
                if self.rand_zero:
                    grad[grad==0] = torch.randint_like(grad[grad==0], low=0, high=2)*2 - 1
                    assert not (grad==0).any()
                
                # make update
                p.data -= group['lr'] * grad

        return loss
    
class normSGD(optim.Optimizer):
    def __init__(self, params, lr=0.0001):
        defaults = dict(lr=lr
                          )
        super(normSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Normalize the gradient
                grad = p.grad / (p.grad.norm() + 1e-8)

                # Update the parameters
                p.data -= group['lr'] * grad

        return loss

class NormalizeSGD(optim.Optimizer):
    def __init__(self, 
                 params, 
                 lr=0.01,
                 weight_decay=0.0,
                 ):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(NormalizeSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                lr = group['lr']
                weight_decay = group['weight_decay']
                gradient = p.grad / (p.grad.norm(dim=-1, keepdim=True) + 1e-8)
                dim = p.data.shape[-1]
                gradient = gradient * math.sqrt(dim)
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)
                p.data -= lr * gradient
        return loss


    
from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl

class EpochCheckpointCallback(pl.Callback):
    def __init__(self, ckpt_epochs, dirpath):
        super().__init__()
        self.ckpt_epochs = ckpt_epochs
        self.dirpath = dirpath

    def on_train_epoch_end(self, *args, **kwargs):
        trainer, pl_module = args[0], args[1]
        if trainer.current_epoch in self.ckpt_epochs:
            trainer.save_checkpoint(os.path.join(self.dirpath, f'epoch={trainer.current_epoch:02d}.ckpt'))

class  StepCheckpointCallback(pl.Callback):
    def __init__(self, ckpt_steps, dirpath):
        super().__init__()
        self.ckpt_steps = ckpt_steps
        self.dirpath = dirpath

    def on_train_batch_end(self, *args, **kwargs):
        trainer, pl_module = args[0], args[1]
        if trainer.global_step in self.ckpt_steps:
            trainer.save_checkpoint(os.path.join(self.dirpath, f'step={trainer.global_step}.ckpt'))

def dominance_metrics(tensor, dim, metrics_to_use=None):
    """
    Calculate dominance metrics along a specified dimension of a tensor using PyTorch.
    
    Parameters:
        tensor (torch.Tensor): The input tensor.
        dim (int): The dimension along which to compute the metrics.
        metrics_to_use (list): A list of metrics to compute. Options include:
                               "Dominance Index", "Top-to-Mean Ratio",
                               "Z-Score", "Entropy".
                               If None, all metrics will be computed.
    
    Returns:
        dict: A dictionary with the selected metrics, where each metric's value is 
              a tensor computed along the specified dimension.
    """
    # Ensure tensor is a PyTorch tensor
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    
    # Components along the specified dimension
    max_vals, _ = torch.max(tensor, dim=dim, keepdim=True)
    sum_vals = torch.sum(tensor, dim=dim, keepdim=True)
    mean_vals = torch.mean(tensor, dim=dim, keepdim=True)
    std_devs = torch.std(tensor, dim=dim, keepdim=True, unbiased=False)
    proportions = tensor / sum_vals

    # Available metrics
    all_metrics = {
        "Dominance Index": max_vals / sum_vals,
        "Top-to-Mean Ratio": max_vals / mean_vals,
        "Z-Score": (max_vals - mean_vals) / (std_devs + 1e-8),  # Add epsilon to prevent division by zero
        "Entropy": -torch.sum(proportions * torch.log(proportions + 1e-8), dim=dim) / torch.log(torch.tensor(tensor.size(dim), dtype=torch.float32))
    }

    # Filter metrics to compute
    if metrics_to_use is None:
        metrics_to_use = all_metrics.keys()
    
    # if only a string is provided
    if isinstance(metrics_to_use, str):
        return all_metrics[metrics_to_use]
    else:
        return {metric: torch.squeeze(all_metrics[metric], dim=dim) for metric in metrics_to_use if metric in all_metrics}


def Calinski_Harabasz_score(x, dim=0):
    # x: shape (max_buffer_vis_size, *channel_size_ls, hidden_size)
    
    # make the dim of interest the first dimension
    x = x.transpose(0, dim)
    
    # Step 1: find a threshold for the pre-activation, which is computed as 0.5 times the largest value in the first dimension
    thres = 0.5 * x.max(dim=0)[0] # shape: (*channel_size_ls, hidden_size)
    cluster_ge_thres = (x > thres)
    cluster_le_thres = torch.logical_not(cluster_ge_thres)

    # Step 2: compute the mean for each cluster
    mean_ge_thres = x[cluster_ge_thres].mean()
    mean_le_thres = x[cluster_le_thres].mean()
    mean = x.flatten().mean()
    
    # Step 3: compute the variance for each cluster
    std_ge_thres = x[cluster_ge_thres].std()
    std_le_thres = x[cluster_le_thres].std()
    
    # Step 4: compute the between-cluster variance
    between_cluster_variance = (mean_ge_thres - mean)**2 * cluster_ge_thres.float().sum() + (mean_le_thres - mean)**2 * cluster_le_thres.float().sum()
    
    within_cluster_variance = (std_ge_thres**2 * cluster_ge_thres.float().sum() + std_le_thres**2 * cluster_le_thres.float().sum())
    
    return between_cluster_variance / within_cluster_variance


def neuron_sorting(neuron_pattern, mode='max'):
    if mode == 'min':
        neuron_pattern = -neuron_pattern
    # for each row of neuron_pattern, find the index of the maximum value
    if isinstance(neuron_pattern, torch.Tensor):
        _, indices = torch.sort(neuron_pattern, dim=1, descending=True)
    else:
        _, indices = torch.sort(torch.tensor(neuron_pattern), dim=1, descending=True)
    # indices: shape: (num_neurons, num_patterns)
    # NOTE: indices indicate the order of the patterns for each neuron
    
    # reorder the rows of neuron_pattern according to the first column of indices, e.g., the first column of indices is [2, 0, 1, 1], then the first row of the reordered neuron_pattern is the second row of the original neuron_pattern, the second and third rows of the reordered neuron_pattern are the last two rows of the original neuron_pattern, and the last row of the reordered neuron_pattern is the first row of the original neuron_pattern
    
    _, row_order = torch.sort(indices[:, 0], descending=False) # indices = [2, 0, 1, 1] -> row_order = [1, 2, 3, 0]
    
    reordered_neuron_pattern = neuron_pattern[row_order]
    reordered_indices = indices[row_order]
    
    if mode == 'min':
        reordered_neuron_pattern = -reordered_neuron_pattern
    return reordered_neuron_pattern, row_order


def create_hook_fn(keyword: str,
                    tensor_to_hook_str: str, 
                    storage_dict: EasyDict):
    """return a hook function that can be used to hook a tensor from a model and store it in a storage_dict.

    Args:
    - keyword: the key to store the tensor in the storage_dict
    - tensor_to_hook_str: the string representation of the tensor to hook
    - storage_dict: the storage_dict to store the tensor
    """
    def hook_fn(module, input, output):
        ## --------- change the probe model input here --------- ##
        if isinstance(output, tuple):
            direct_output, intermediate_dict = output
        else:
            direct_output = output
            intermediate_dict = None
        # combine model_to_hook_str and tensor_to_hook_str with a dot
        # keyword = f"{model_to_hook_str}.{tensor_to_hook_str}"
        if tensor_to_hook_str == "output":
            storage_dict.setattr_with_string(keyword, direct_output)
        elif tensor_to_hook_str == "input":
            if isinstance(input, tuple):
                storage_dict.setattr_with_string(keyword, input[0])
            else: 
                storage_dict.setattr_with_string(keyword, input)
        else:
            storage_dict.setattr_with_string(keyword, intermediate_dict[tensor_to_hook_str])
    return hook_fn
    
def attach_hooks(TF_model, HookDict):
    buffer = copy.deepcopy(HookDict)
    hook_handles = copy.deepcopy(HookDict)
    # set the value to None
    for key in buffer.keys():
        buffer[key] = None
        hook_handles[key] = None
    # attach hooks
    for key in HookDict.keys():
        parts = HookDict[key].split('.')
        model_name = '.'.join(parts[:-1])
        tensor_name = parts[-1]
        model_to_hook = operator.attrgetter(model_name)(TF_model)
        hook_handles[key] = model_to_hook.register_forward_hook(create_hook_fn(key, tensor_name, buffer))
    return buffer, hook_handles


class AlignmentLoss(torch.nn.Module):
    def __init__(self, dim=-1, eps=1e-8, reduction='mean', normalize=True):
        super().__init__()
        if normalize:
            self.cos_sim = torch.nn.CosineSimilarity(dim=dim, eps=eps)
        self.reduction = reduction
        self.normalize = normalize
        self.eps = eps
        
    def forward(self, input, target):
        """
        Compute the cosine similarity between W_enc and feat.
        """
        if self.normalize:
            if self.reduction == 'mean':
                return 1.0 - self.cos_sim(input, target).mean()
            elif self.reduction == 'sum':
                return (1.0 - self.cos_sim(input, target)).sum()
            else:
                return (1.0 - self.cos_sim(input, target))
        else:
            # if self.reduction == 'mean':
            #     return - (input * target / (target.norm(dim=-1, keepdim=True) + self.eps)).sum(dim=-1).mean()
            # elif self.reduction == 'sum':
            #     return - (input * target / (target.norm(dim=-1, keepdim=True) + self.eps)).sum(dim=-1)
            # else:
            #     return - (input * target / (target.norm(dim=-1, keepdim=True) + self.eps))
            if self.reduction == 'mean':
                return - (input * target).sum(dim=-1).mean()
            elif self.reduction == 'sum':
                return - (input * target).sum(dim=-1)
            else:
                return - (input * target)


def clean_config(config):
    """
    Recursively clean nested objects (e.g., EasyDict) and return a plain dictionary.
    """
    if isinstance(config, Mapping):  # Check if it's a dict-like object
        return {k: clean_config(v) for k, v in config.items()}
    elif isinstance(config, list):  # Check for lists
        return [clean_config(i) for i in config]
    else:
        return config  # Return primitive values as-is


def construct_gram_matrix(matrix, dim=-1):
    """
    Construct a gram matrix from a matrix, the type of the matrix is nparray
    matrix: shape (..., n, m)
    dim: the dimension to normalize the matrix along
    """
    # normalized_matrix = matrix / torch.norm(matrix, dim=dim, keepdim=True) # shape (..., n, m)
    normalized_matrix = matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)
    gram_matrix = normalized_matrix @ normalized_matrix.T
    return gram_matrix
    
def matrix_entropy(matrix):
    """
    Compute the entropy of a positive-definite matrix.
    matrix: shape (dim, dim)
    dim: the dimension to normalize the matrix along
    """
    # compute the entropy
    assert matrix.shape[0] == matrix.shape[1]
    dim = matrix.shape[0]
    matrix = matrix / dim
    log_matrix, _ = logm(matrix, disp=False)
    # get the trace of the matrix
    return -np.trace(matrix @ log_matrix)


def MIR(matrix_1, matrix_2, dim_1=-1, dim_2=-1):
    """
    Compute the matrix mutual information ratio of two matrices.
    """
    if type(matrix_1) == torch.Tensor:
        matrix_1 = matrix_1.detach().cpu().numpy()
    if type(matrix_2) == torch.Tensor:
        matrix_2 = matrix_2.detach().cpu().numpy()
    gram_matrix_1 = construct_gram_matrix(matrix_1, dim_1) #
    gram_matrix_2 = construct_gram_matrix(matrix_2, dim_2) 
    hardamard_product = gram_matrix_1 * gram_matrix_2 # shape (..., n, n)
    matrix_mutual_information = matrix_entropy(gram_matrix_1) + matrix_entropy(gram_matrix_2) - matrix_entropy(hardamard_product)

    return matrix_mutual_information/min(matrix_entropy(gram_matrix_1), matrix_entropy(gram_matrix_2))


def HDR(matrix_1, matrix_2, dim_1=-1, dim_2=-1):
    """
    compute the matrix entropy ratio of two matrices.
    """
    if type(matrix_1) == torch.Tensor:
        matrix_1 = matrix_1.detach().cpu().numpy()
    if type(matrix_2) == torch.Tensor:
        matrix_2 = matrix_2.detach().cpu().numpy()
    gram_matrix_1 = construct_gram_matrix(matrix_1, dim_1) # 
    gram_matrix_2 = construct_gram_matrix(matrix_2, dim_2) # 
    # HDR = |H(A) - H(B)| / max(H(A), H(B))
    return np.abs(matrix_entropy(gram_matrix_1) - matrix_entropy(gram_matrix_2))/max(matrix_entropy(gram_matrix_1), matrix_entropy(gram_matrix_2))


def get_cos_threshold(features, angle_division_factor):
    """
    Get the cosine similarity threshold
    features: shape (num, feature_dim)
    angle_division_factor: Factor to divide the maximum angle by to create a stricter threshold
    """
    import math
    cos_sim = check_cosine_similarity(features, features, return_tensor=True, verbose=False)
    print(cos_sim.shape)
    # get the max cosine similarity for each distinct feature
    off_diag_cos_sim = cos_sim[np.triu(np.ones(cos_sim.shape), k=1).astype(bool)]
    max_cos_sim = off_diag_cos_sim.max()
    angle = math.acos(max_cos_sim)
    divide_angle = angle / angle_division_factor
    threshold = math.cos(divide_angle)
    return threshold