from .model_base import TransformerEncoder, ReadIn, ReadOut, nnModule, TransformerEncoderOnlyAttn
from .utils import EasyDict
from torch import Tensor
from typing import Optional
import torch.nn as nn
import math

class GPT2Standard(nnModule):
    def __init__(self, config: EasyDict, vocab_size: int, weight_tie: bool = False):
        super().__init__()
        self.readin = ReadIn(vocab_size, config.hidden_size)
        self.encoder = TransformerEncoder(config)
        self.readout = ReadOut(config.hidden_size, vocab_size)
        
        self.model_config = config
        # ## ----------  initialization ---------- ##
        self.apply(self._init_weights)
        
        # for layer n, initialize the MLP proj weights with mean 0 and std 0.02 / sqrt(2 * n)
        # per-GPT2 paper, scale intialization of output projection and last layer of mlp
        # apply special n_layer-scaled initialization to layers that add to the residual stream
        # (output projection of attention and last layer of mlp)
        # this ensures that, at initialization, adding to the residual stream does not cause things to blow up
        for pn, p in self.named_parameters():
            if pn.endswith(f'mlp.proj.weight') or pn.endswith('o_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.model_config.num_layers))
                
        # add weight tieing for the input and output embeddings
        if weight_tie:
            self.readout.emb2idx.weight = self.readin.idx2emb.weight

    def forward(self, input_token_ids: Tensor, mask: Optional[Tensor] = None):
        x = self.readin(input_token_ids)
        x = self.encoder(x, mask)
        output = self.readout(x)
        return output
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

class GPT2LinearReg(nnModule):
    def __init__(self, config: EasyDict, input_size: int, output_size: int):
        super().__init__()
        self.readin = nn.Linear(input_size, config.hidden_size)
        self.encoder = TransformerEncoder(config)
        self.readout = nn.Linear(config.hidden_size, output_size)

    def forward(self, input_tokens: Tensor, mask: Optional[Tensor] = None):
        x = self.readin(input_tokens)
        x = self.encoder(x, mask)
        output = self.readout(x)
        return output
    

class GPT2OnlyAttention(GPT2Standard):
    """
    GPT2 model with only attention mechanism.
    """
    def __init__(self, config: EasyDict, input_size: int, output_size: int):
        super().__init__(config, input_size, output_size)
        self.encoder = TransformerEncoderOnlyAttn(config)
