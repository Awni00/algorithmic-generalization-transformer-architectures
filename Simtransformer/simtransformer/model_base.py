from typing import Optional, Union, Literal, Any
import torch
from torch import nn, Tensor
import math
from .utils import EasyDict

def make_relposition(query_seq_len: int, 
                      key_seq_len: int, 
                      relpos_win: tuple):
    """
    This function generates a relative position matrix for the given query and key sequence lengths, clamped within the specified relative position window.
    The function is a tool for the MultiHeadAttentionDeBERTa module to generate relative positional encoding.

    Args:
        query_seq_len (int): Length of the query sequence.
        key_seq_len (int): Length of the key sequence.
        relpos_win (tuple): A tuple specifying the minimum and maximum relative positions.

    Returns:
        torch.Tensor: A tensor of shape [query_seq_len, key_seq_len] containing the relative positions,
                      clamped within the specified window and adjusted to start from 0.
    """
    query_pos = torch.arange(query_seq_len)
    key_pos = torch.arange(key_seq_len)
    relpos = (key_pos[None, :] - query_pos[:, None]).clamp(min=relpos_win[0], max=relpos_win[1]) # [query_seq_len, key_seq_len], clamped to the window
    relpos = relpos - relpos_win[0] # make the relative position start from 0
    return relpos


class nnModule(nn.Module):        
    def __view__(self):
        """
        This method provides a view of the internal weights of the nnModule module by recursively calling the __view__ method of the children (which includes both submodules and nn.Parameters).
        If a child does not have the __view__ method, then the child is a leaf and its named_parameters() are returned in an EasyDict.
        
        Returns:
            dict: An EasyDict containing the self.named_parameters() of the module.
        """
        # enumerate over all submodules
        params_dict = EasyDict({})
        for name, module in self.named_children():
            # check if the module has the __view__ method
            if hasattr(module, "__view__"):
                params_dict[name] = module.__view__()
            else:
                params_dict[name] = EasyDict(module.named_parameters())
        return params_dict

class nnModuleDict(nn.ModuleDict):
    def __view__(self):
        """
        This method provides a view of the internal weights of the modules in the nn.ModuleDict model's dictionary.
        
        Returns:
            dict: An EasyDict containing the self.named_parameters() of the module.
        """
        params_dict = EasyDict({})
        for name, module in self.items():
            # check if the module has the __view__ method
            if hasattr(module, "__view__"):
                params_dict[name] = module.__view__()
            else:
                params_dict[name] = EasyDict(module.named_parameters())
        return params_dict

class AbsolutePositionalEmbeddings(nnModule):
    """
    This class implements the absolute positional embeddings for the transformer model with a forward method that can be used to add positional embeddings to the input tensor.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        use_pos_layer_norm: bool = True,
    ):
        """
        Initialize the AbsolutePositionalEmbeddings module.

        Args:
            dim (int): The embedding dimension of the model.
            max_position_embeddings (int): The maximum number of positions to embed.
            use_pos_layer_norm (bool, optional): Whether to use layer normalization after adding the positional embeddings. Defaults to True.
        """
        super().__init__()
        self.abspos_embeddings = nn.Embedding(max_position_embeddings, dim)
        if use_pos_layer_norm:
            self.layer_norm = nn.LayerNorm(dim)
        self.use_pos_layer_norm = use_pos_layer_norm

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Apply the positional embeddings to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, seq_len, hidden_size].
            position_ids (torch.Tensor): The position ids tensor of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: The output tensor after applying the positional embeddings.
        """
        if input_pos is None:
            input_pos = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        position_embeddings = self.abspos_embeddings(input_pos)
        if self.use_pos_layer_norm:
            return self.layer_norm(x + position_embeddings)
        else:
            return x + position_embeddings

class RotaryPositionalEmbeddings(nnModule):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init. 
    See https://pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: Tensor, *, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

class DeBERTaPositionalEmbeddings(nnModule):
    """
    This class implements the positional embeddings for DeBERTa with relative positional embeddings.
    """

    def __init__(
        self,
        dim: int,
        relpos_win_size: int,
        relpos_shift: int = 0, 
        relpos_q_k_enabled: Union[list, tuple] = [True, True],
        use_pos_layer_norm: bool = True,
    ):
        """
        Initialize the DeBERTaPositionalEmbeddings module.

        Args:
            dim (int): The embedding dimension of the model.
            relpos_win_size (int): The window size for relative positional encoding.
            relpos_shift (int, optional): The shift for relative positional encoding. Defaults to 0. By default, the relative positional window is [-relpos_win_size + 1, 0] for the key_id - query_id.

        """
        super().__init__()
        assert relpos_win_size > 0, f"Relative positional window size must be greater than 0, but got {relpos_win_size}!"
        if relpos_q_k_enabled[0] == False and relpos_q_k_enabled[1] == False:
            raise ValueError("At least one of the relative positional encoding for queries and keys must be enabled!")
        
        self.relpos_embeddings = nn.Embedding(relpos_win_size, dim)
        self.relpos_shift = relpos_shift
        self.relpos_q_k_enabled = relpos_q_k_enabled
        self.relpos_win_size = relpos_win_size
        
        if use_pos_layer_norm:
            self.layer_norm = nn.LayerNorm(dim)
        self.use_pos_layer_norm = use_pos_layer_norm
        self.dim = dim
        

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                relpos_q_proj: nn.Module,
                relpos_k_proj: nn.Module,
                pos_dropout: nn.Module
                ) -> torch.Tensor:
        """
        Apply the positional embeddings to the input tensor.

        Args:
            q (torch.Tensor): The input tensor of shape [batch_size, seq_len, num_heads, qk_embed_size_per_head].
            k (torch.Tensor): The input tensor of shape [batch_size, seq_len, num_heads, qk_embed_size_per_head].
            relpos_q_proj (nn.module): The relative positional projection module for queries.
            relpos_k_proj (nn.module): The relative positional projection module for keys.
            pos_dropout (nn.module): The positional dropout module.

        Returns:
            Tuple[torch.Tensor]: The output tensor after applying the positional embeddings.
            Tuple[torch.Tensor]: The relative positional embeddings for queries and keys.
        """


        batch_size = q.shape[0]
        q_seq_len = q.shape[-3]
        k_seq_len = k.shape[-3]
        num_heads = q.shape[-2]
        qk_embed_size_per_head = q.shape[-1]
        
        # apply position dropouts
        relpos_embeddings = self.relpos_embeddings(torch.arange(self.relpos_win_size, device=q.device))
        if self.use_pos_layer_norm:
            relpos_embeddings = self.layer_norm(relpos_embeddings)
        
        relpos_embeddings = pos_dropout(relpos_embeddings)

        # compute the positional indexing matrix
        if self.relpos_q_k_enabled[0] or self.relpos_q_k_enabled[1]:
            # the window for relative PE
            relpos_win = [-self.relpos_win_size + 1 + self.relpos_shift, 0 + self.relpos_shift]
            # create the table for key_idx - query_idx
            relpos = make_relposition(q_seq_len, k_seq_len, relpos_win) #[q_seq_len, k_seq_len]
            # expand
            relpos = relpos.to(q.device)[None, None, :, :].expand(batch_size, num_heads, q_seq_len, k_seq_len)

        if self.relpos_q_k_enabled[0]:
            # compute the query-position scores

            relpos_k = relpos_k_proj(relpos_embeddings).view(-1, num_heads, qk_embed_size_per_head) # [relpos_win_size, num_heads, qk_embed_size_per_head]

            # compute the query-position scores
            query_pos_scores = torch.einsum("bnhx,khx->bhnk", q, relpos_k) # [batch_size, num_heads, query_seq_len, relpos_win_size]

            # put the scores on the correct position
            logits_query_pos = torch.gather(query_pos_scores, 3, relpos)
        else:
            logits_query_pos = 0

        if self.relpos_q_k_enabled[1]:
            # compute the key-position scores

            relpos_q = relpos_q_proj(relpos_embeddings).view(-1, num_heads, qk_embed_size_per_head) # [relpos_win_size, num_heads, qk_embed_size_per_head]

            # compute the key-position scores
            key_pos_scores = torch.einsum("bnhx,khx->bhkn", k, relpos_q) # [batch_size, num_heads, relpos_win_size,  key_seq_len]

            # put the scores on the correct position
            logits_pos_key = torch.gather(key_pos_scores, 2, relpos)
        else:
            logits_pos_key = 0
        
        scale_factor = 1 + self.relpos_q_k_enabled[0] + self.relpos_q_k_enabled[1]
        return (logits_pos_key, logits_query_pos, scale_factor), (relpos_q, relpos_k)


class MultiHeadAttentionDeBERTa(nnModule):
    """
    Implementation of multi-head self attention with relative positional embedding for DeBERTa. https://github.com/microsoft/DeBERTa/tree/master 
    """

    def __init__(
            self,
            num_heads: int,
            hidden_size: int,
            attention_type: Literal["softmax", "relu", "linear"] = "softmax",
            use_bias: bool = True,
            attn_pdrop: float = 0.20,
            q_k_v_o_proj_enabled: list = [True, True, True, True],
            relpos_q_k_enabled: bool = [True, True],
            relpos_embed_size: int = 64,
            causal_attn: bool = True,
            **kwargs,
    ):
        """
        Initialize the MultiHeadAttentionDeBERTa module.

        Args:
            num_heads (int): Number of attention heads.
            hidden_size (int): Hidden size of the model.
            attention_type (Literal["softmax", "relu", "linear"], optional): Type of attention activation. Defaults to "softmax".
            use_bias (bool, optional): Whether to use bias in the projection layers. Defaults to False.
            attn_pdrop (float, optional): Dropout rate for attention weights. Defaults to 0.0.
            q_k_v_o_proj_enabled (list, optional): List of booleans indicating whether to enable projection layers for queries, keys, values, and outputs. Defaults to [True, True, True, True].
            relpos_q_k_enabled (list, optional): List of booleans indicating whether to enable relative positional encoding for queries and keys. Defaults to [True, True].
            relpos_embed_size (int, optional): Embedding size for relative positional encoding. Defaults to 64.
            causal_attn (bool, optional): Whether to use causal attention. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self._num_heads = num_heads
        self.q_k_v_o_proj_enabled = q_k_v_o_proj_enabled
        # causal attention
        self.causal_attn = causal_attn

        # Set the size of for queries, keys and values and outputs.
        self.q_dim = self.k_dim = self.v_dim = self.o_dim = hidden_size

        # find the maximum of q_dim, k_dim, v_dim and o_dim
        max_dim = max(self.q_dim, self.k_dim)

        # Initialization of embedding sizes per head.
        qk_embed_size_per_head, vo_embed_size_per_head = None, None
        self._qk_embed_size_per_head = int(math.ceil(max_dim / self._num_heads))
        self._qk_embed_size = self._qk_embed_size_per_head * self._num_heads
        
        self._vo_embed_size_per_head = int(math.ceil(self.v_dim / self._num_heads))
        self._vo_embed_size = self._vo_embed_size_per_head * self._num_heads
        
        


        # Initialization of attention activation.
        if attention_type == 'softmax':
            self.attention_activation = nn.Softmax(dim=-1)
        elif attention_type == 'relu':
            self.attention_activation = nn.ReLU()
        elif attention_type == 'linear':
            self.attention_activation = nn.Identity()
        else:
            raise NotImplementedError(
                f"Attention type {attention_type} is not implemented!"
            )
        self.attention_type = attention_type


        # initialize the q_proj, k_proj, v_proj and o_proj layers for each head
        if q_k_v_o_proj_enabled[0]:
            self.q_proj = nn.Linear(
                in_features=self.q_dim,
                out_features=self._qk_embed_size,
                bias=use_bias,
            )  
        else:
            if self._qk_embed_size == self.q_dim:
                self.q_proj = nn.Identity()
            else:
                raise ValueError(
                    f"q_proj must be enabled for q_dim {self.q_dim} and qk_embed_size {self._qk_embed_size}!"
                )
        
        if q_k_v_o_proj_enabled[1]:
            self.k_proj = nn.Linear(
                in_features=self.k_dim,
                out_features=self._qk_embed_size,
                bias=use_bias,
            )
        else:
            if self._qk_embed_size == self.k_dim:
                self.k_proj = nn.Identity()
            else:
                raise ValueError(
                    f"k_proj must be enabled for k_dim {self.k_dim} and qk_embed_size {self._qk_embed_size}!"
                )
        
        if q_k_v_o_proj_enabled[2]:
            self.v_proj = nn.Linear(
                in_features=self.v_dim,
                out_features=self._vo_embed_size, 
                bias=use_bias,
            )
        else:
            if self._vo_embed_size == self.v_dim:
                self.v_proj = nn.Identity()
            else:
                raise ValueError(
                    f"v_proj must be enabled for v_dim {self.v_dim} and vo_embed_size {self._vo_embed_size}!"
                )
        
        if q_k_v_o_proj_enabled[3]:
            self.o_proj = nn.Linear(
                in_features=self._vo_embed_size,
                out_features=self.o_dim,
                bias=use_bias,
            )
        else:
            if self._vo_embed_size == self.o_dim:
                self.o_proj = nn.Identity()
            else:
                raise ValueError(
                    f"o_proj must be enabled for o_dim {self.o_dim} and vo_embed_size {self._vo_embed_size}!"
                )

        # Initialization of dropout layer.
        self.dropout = nn.Dropout(p=attn_pdrop)
        
        # relative positional encoding
        self.relpos_q_k_enabled = relpos_q_k_enabled
        self.relpos_embed_size = relpos_embed_size
        
        self.relpos_k_proj = nn.Linear(in_features=self.relpos_embed_size, out_features=self._qk_embed_size, bias=use_bias) if self.relpos_q_k_enabled[0] else nn.Identity()
        self.relpos_q_proj = nn.Linear(in_features=self.relpos_embed_size, out_features=self._qk_embed_size, bias=use_bias) if self.relpos_q_k_enabled[1] else nn.Identity()
        self.pos_dropout = nn.Dropout(p=attn_pdrop)
        
        self.use_bias = use_bias
        

    def forward(
            self,
            x: torch.Tensor,
            pos_model: Optional[nn.Module] = None,
            mask: Optional[Union[torch.Tensor, None]] = None,
            head_mask: Optional[torch.Tensor] = None,
            logits_shift: Optional[torch.Tensor] = None,
    ):
        """
        Apply a forward pass of attention.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
            pos_model (nn.Module, optional): The positional model to apply. Defaults to None.
            mask (Union[torch.Tensor, None], optional): The mask tensor. Defaults to None.
            head_mask (torch.Tensor, optional): The head mask tensor. Defaults to None.
            logits_shift (torch.Tensor, optional): The shift tensor for logits. Defaults to None.

        Returns:
            o (torch.Tensor): Output tensor after applying multi-head attention of shape [batch_size, seq_len, hidden_size].
            present (Tuple[torch.Tensor]): Tuple of query, key, value tensors for the next layer.
            weights (torch.Tensor): Attention probabilities of shape [batch_size, num_heads, query_seq_len, key_seq_len].
        """
        # if x does not have batch dimension, raise an error
        if len(x.shape) != 3:
            raise ValueError(f"Attention input tensor x must have 3 dimensions (batch_size, seq_len, hidden_size), but got a tensor of shape {x.shape}!")

        batch_size = x.shape[0]

        # Linear projections for queries, keys and values.
        q = self.q_proj(x) # [batch_size, query_seq_len, qk_embed_size]
        k = self.k_proj(x)  # [batch_size, key_seq_len, qk_embed_size]
        v = self.v_proj(x)  # [batch_size, key_seq_len, vo_embed_size]

        q_seq_len = q.shape[-2]
        k_seq_len = k.shape[-2]

        # Reshape to 4D tensors of shape
        # [batch_size, seq_len, num_heads, qkv_size_per_head].
        q = q.reshape(-1, q.shape[1], self._num_heads, self._qk_embed_size_per_head)
        k = k.reshape(-1, k.shape[1], self._num_heads, self._qk_embed_size_per_head)
        v = v.reshape(-1, v.shape[1], self._num_heads, self._vo_embed_size_per_head)

        # Apply the rotary positional encoding
        if pos_model is not None:
            if isinstance(pos_model, RotaryPositionalEmbeddings):
                q = pos_model(q)
                k = pos_model(k)
                logits_pos_key = logits_query_pos = 0.0
                relpos_q = relpos_k = None
                scale_factor = 1.0
            elif isinstance(pos_model, DeBERTaPositionalEmbeddings):
                output, relpos_qk = pos_model(q, k, self.relpos_q_proj, self.relpos_k_proj, self.pos_dropout)
                logits_pos_key, logits_query_pos, scale_factor = output
                relpos_q, relpos_k = relpos_qk
        else:
            logits_query_pos = logits_pos_key = 0.0
            scale_factor = 1.0
        
        logits_query_key = torch.einsum("bnhk,bmhk->bhnm", q, k) 
        # Compute attention weights.
        logits = (logits_query_key + logits_query_pos + logits_pos_key) * (self._qk_embed_size_per_head * scale_factor) ** (-0.5) # [batch_size, num_heads, query_seq_len, key_seq_len]
        
        if logits_shift is not None:
            logits = logits + logits_shift
        
        
        # Apply mask to the logits.
        total_mask = torch.ones((q_seq_len, k_seq_len), device=logits.device, dtype=bool)
        if mask is not None:
            total_mask = torch.logical_and(total_mask, mask.bool())
        if self.causal_attn:
            mask = torch.tril(torch.ones_like(logits).bool(), diagonal=0)
            total_mask = torch.logical_and(total_mask, mask)
        
        logits = logits.masked_fill(torch.logical_not(total_mask), float('-inf'))
        attn_prob = self.attention_activation(logits) # [batch_size, num_heads, query_seq_len, key_seq_len]
        
        if self.attention_type != 'softmax':
            attn_prob = attn_prob.masked_fill(torch.logical_not(total_mask), 0.0)

        # Apply attention attn_prob dropout.
        attn_prob = self.dropout(attn_prob)

        o = torch.einsum("bhnm,bmhk->bnhk", attn_prob, v) # [batch_size, query_seq_len, num_heads, vo_embed_size_per_head]

        # apply head mask
        if head_mask is not None:
            head_mask = head_mask.squeeze(0).to(dtype=o.dtype, device=o.device)
            assert head_mask.shape == (self._num_heads,), f"Head mask shape {head_mask.shape} does not match the number of heads {self._num_heads}!"
            head_mask = head_mask[None, None, :, None]
            o = o * head_mask
        
        # Reshape to 3D tensor.
        o = torch.reshape(o, (-1, o.shape[1], self._vo_embed_size_per_head * self._num_heads)) # [batch_size, query_seq_len, vo_embed_size]

        # Linear projection for outputs.
        o = self.o_proj(o) # [batch_size, query_seq_len, o_dim]

        # return intermediate values
        intermediate = EasyDict({
            'input': x,
            'q': q if 'q' in locals() else None,
            'k': k if 'k' in locals() else None,
            'v': v if 'v' in locals() else None,
            'relpos_k': relpos_k if 'relpos_k' in locals() else None,
            'relpos_q': relpos_q if 'relpos_q' in locals() else None,
            'logits_query_pos': logits_query_pos if 'logits_query_pos' in locals() else None,
            'logits_pos_key': logits_pos_key if 'logits_pos_key' in locals() else None,
            'logits_query_key': logits_query_key if 'logits_query_key' in locals() else None,
            'attn_prob': attn_prob if 'attn_prob' in locals() else None,
            'output': o
        })
        return o, intermediate
    

    
    def __view__(self):
        """
        This method provides a view of the internal weights of the MultiHeadAttentionDeBERTa module.
        
        Returns:
            dict: A dictionary containing the following keys:
            - "kq_effect_weights": The effective weights for the key-query projections.
            - "ov_effect_weights": The effective weights for the output-value projections.
            - "q_proj_weights": The weights for the query projections.
            - "k_proj_weights": The weights for the key projections.
            - "v_proj_weights": The weights for the value projections.
            - "o_proj_weights": The weights for the output projections.
        """
        params_dict = EasyDict({})
        
        # Linear projections for queries and keys.
        k_proj_weights = self.k_proj.weight.data if self.q_k_v_o_proj_enabled[1] else torch.eye(self.k_dim)
        q_proj_weights = self.q_proj.weight.data if self.q_k_v_o_proj_enabled[0] else torch.eye(self.q_dim)
        k_proj_bias = self.k_proj.bias.data if self.q_k_v_o_proj_enabled[1] and self.use_bias else torch.zeros(self.k_dim)
        q_proj_bias = self.q_proj.bias.data if self.q_k_v_o_proj_enabled[0] and self.use_bias else torch.zeros(self.q_dim)

        # split the weights into num_heads using torch.view method
        k_proj_weights = k_proj_weights.view(self._num_heads, self._qk_embed_size_per_head, self.k_dim) # shape: (num_heads, qk_size_split, k_dim)
        q_proj_weights = q_proj_weights.view(self._num_heads, self._qk_embed_size_per_head, self.q_dim) # shape: (num_heads, qk_size_split, q_dim)
        k_proj_bias = k_proj_bias.view(self._num_heads, 1, self._qk_embed_size_per_head) # shape: (num_heads, 1, k_dim)
        q_proj_bias = q_proj_bias.view(self._num_heads, 1, self._qk_embed_size_per_head) # shape: (num_heads, 1, q_dim)

        # compute the attention weights
        kq_effect_weights = torch.einsum("hdk,hdq->hkq", k_proj_weights, q_proj_weights) * self._qk_embed_size_per_head ** -.5 # shape: (num_heads, k_dim, q_dim)

        v_proj_weights = self.v_proj.weight.data if self.q_k_v_o_proj_enabled[2] else torch.eye(self.v_dim)
        o_proj_weights = self.o_proj.weight.data if self.q_k_v_o_proj_enabled[3] else torch.eye(self.o_dim)
        v_proj_bias = self.v_proj.bias.data if self.q_k_v_o_proj_enabled[2] and self.use_bias else torch.zeros(self.v_dim)
        o_proj_bias = self.o_proj.bias.data if self.q_k_v_o_proj_enabled[3] and self.use_bias else torch.zeros(self.o_dim)

        # split the weights into num_heads
        v_proj_weights = v_proj_weights.view(self._num_heads, self._vo_embed_size_per_head, self.v_dim)  # shape: (num_heads, vo_size_per_head, v_dim)
        o_proj_weights = o_proj_weights.view(self.o_dim, self._num_heads, self._vo_embed_size_per_head).transpose(1, 0)  # shape: (num_heads, o_dim, vo_size_per_head)
        v_proj_bias = v_proj_bias.view(self._num_heads, 1, self._vo_embed_size_per_head)  # shape: (num_heads, 1, v_dim)
        o_proj_bias = o_proj_bias.view(self._num_heads, 1, self._vo_embed_size_per_head)  # shape: (num_heads, 1, o_dim)

        # compute the output weights
        ov_effect_weights = torch.einsum("hod,hdv->hov", o_proj_weights, v_proj_weights) # shape: (num_heads, o_dim, v_dim)
        
        if self.q_k_v_o_proj_enabled[0]:
            tmp_dict = EasyDict({"weight": q_proj_weights})
            if self.use_bias:
                tmp_dict.update({"bias": q_proj_bias})
            params_dict.q_proj = tmp_dict
        if self.q_k_v_o_proj_enabled[1]:
            tmp_dict = EasyDict({"weight": k_proj_weights})
            if self.use_bias:
                tmp_dict.update({"bias": k_proj_bias})
            params_dict.k_proj = tmp_dict
        if self.q_k_v_o_proj_enabled[2]:
            tmp_dict = EasyDict({"weight": v_proj_weights})
            if self.use_bias:
                tmp_dict.update({"bias": v_proj_bias})
            params_dict.v_proj = tmp_dict
        if self.q_k_v_o_proj_enabled[3]:
            tmp_dict = EasyDict({"weight": o_proj_weights})
            if self.use_bias:
                tmp_dict.update({"bias": o_proj_bias})
            params_dict.o_proj = tmp_dict
            
        params_dict.kq_effect = EasyDict({"weight": kq_effect_weights})
        params_dict.ov_effect = EasyDict({"weight": ov_effect_weights})
        
        if self.relpos_q_k_enabled[1]:
            tmp_dict = EasyDict({"weight": self.relpos_q_proj.weight.data.view(self._num_heads, self._qk_embed_size_per_head, self.relpos_embed_size)})
            if self.use_bias:
                tmp_dict.update({"bias": self.relpos_q_proj.bias.data.view(self._num_heads, 1, self._qk_embed_size_per_head)})
            params_dict.relpos_q_proj = tmp_dict
        if self.relpos_q_k_enabled[0]:
            tmp_dict = EasyDict({"weight": self.relpos_k_proj.weight.data.view(self._num_heads, self._qk_embed_size_per_head, self.relpos_embed_size)})
            if self.use_bias:
                tmp_dict.update({"bias": self.relpos_k_proj.bias.data.view(self._num_heads, 1, self._qk_embed_size_per_head)})
            params_dict.relpos_k_proj = tmp_dict
            
        
        return params_dict

class SigmoidLU(nnModule):
    def __init__(self, beta=1.0):
        super(SigmoidLU, self).__init__()
        self.beta = beta
        
    def forward(self, x):
        return torch.log(1.0 + torch.exp(self.beta * x)) / self.beta

class PowerReLU(nnModule):
    def __init__(self, power=2.0):
        """
        PowerReLU activation function.
        
        Args:
            p (float): The power to which the positive part of the input is raised. Default is 2.0.
        """
        super(PowerReLU, self).__init__()
        self.power = power

    def forward(self, x):
        # Apply the PowerReLU function
        # return torch.where(x > 0, x ** self.power, torch.zeros_like(x))
        return nn.functional.relu(x) ** self.power

class JumpReLU(nnModule):
    def __init__(self, theta):
        """
        JumpReLU activation function.
        
        Args:
            theta (tensor) the shape of theta should be the same as the input x(or the last dimension of x)
        """
        super(JumpReLU, self).__init__()
        self.theta = theta

    def forward(self, x):
        # Apply the JumpReLU function
        return torch.where(x > self.theta, x, torch.zeros_like(x))

class ThresholdReLU(nnModule):
    def __init__(self):
        """
        ThresholdReLU activation function. The difference between ReLU and ThresholdReLU is that in thresholdReLU, if a neuron is activated, the postactivation will substract the bias of the neuron.
        """
        super(ThresholdReLU, self).__init__()
    
    def forward(self, x, neg_bias):
        # Apply the ThresholdReLU function
        return torch.where(x > neg_bias, x, torch.zeros_like(x))

class MLP(nnModule):
    def __init__(self, hidden_size, intermediate_size, resid_pdrop, **kwargs):
        super(MLP, self).__init__()
        self.fc = nn.Linear(hidden_size, intermediate_size, bias=True)
        activation = kwargs.get('activation', 'relu')
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'sigmoidlu':
            sigmoidlu_beta = kwargs.get('sigmoidlu_beta', 1.0)
            self.act = SigmoidLU(beta=sigmoidlu_beta)
        self.proj = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x, neuron_mask=None):
        pre_activation = self.fc(x)
        post_activation = self.act(pre_activation)
        if neuron_mask is not None:
            post_activation = post_activation * neuron_mask
        output = self.proj(post_activation)
        output = self.dropout(output)
        intermediate = EasyDict({
            "input": x,
            "pre_activation": pre_activation, 
            "post_activation": post_activation, 
            "output": output})
        return output, intermediate



class TransformerBlock(nnModule):
    """Implementation of a Transformer block.
    """

    def __init__(
            self, model_config: EasyDict, layer_idx: int=None,
    ):
        """
        Initialize the TransformerBlock module.

        Args:
        """
        super().__init__()
        
        hidden_size = model_config.hidden_size
        num_heads = model_config.num_heads
        self.relpos_shift = model_config.relpos_shift
        # check if n_inner is defined in the model_configuration
        inner_size = model_config.n_inner if hasattr(model_config, "n_inner") else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=model_config.layer_norm_eps) if model_config.use_layer_norm else nn.Identity()
        self.attn = MultiHeadAttentionDeBERTa(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_type=model_config.attention_type,
            use_bias=model_config.use_bias,
            attn_pdrop=model_config.attn_pdrop,
            q_k_v_o_proj_enabled=model_config.q_k_v_o_proj_enabled,
            relpos_q_k_enabled=model_config.relpos_q_k_enabled,
            relpos_embed_size=model_config.relpos_embed_size if hasattr(model_config, "relpos_embed_size") else hidden_size, 
            causal_attn=model_config.causal_attn,
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=model_config.layer_norm_eps) if model_config.use_layer_norm else nn.Identity()
        self.mlp = MLP(hidden_size, inner_size, model_config.resid_pdrop)

        # store hyperparameters
        self.hidden_size = hidden_size
        self.inner_size = inner_size

    def forward(
        self, 
        in_x: torch.Tensor,
        pos_model: Optional[nn.Module] = None,
        mask: Optional[Union[torch.Tensor, None]] = None,
        head_mask: Optional[torch.Tensor] = None,
        logits_shift: Optional[torch.Tensor] = None,
    ):
        """
        Apply a Transformer block to the input tensor.

        Args:
            in_x (torch.Tensor): A 3D tensor of shape [batch_size, seq_len, input_size].
            mask (Optional[Union[torch.Tensor, None]]): Optional mask tensor 
                of shape [batch_size, 1, seq_len, seq_len].
            head_mask (Optional[torch.Tensor]): Optional head mask tensor.

        Returns:
            torch.Tensor: A 3D tensor of shape [batch_size, seq_len, hidden_size].
        """

        ##-----------------Attention-----------------##
        residual_attn = in_x

        # layer-normalization
        x = self.ln_1(in_x)
        # Apply multi-head self-attention.
        attn_outputs = self.attn(
            x, 
            pos_model=pos_model,
            mask=mask,
            head_mask=head_mask,
            logits_shift=logits_shift,
        )
        attn_output, _ = attn_outputs # the `intermediate` is not used in the forward pass, and we will handle it by forward hook.

        # residual connection
        attn_res_output = attn_output + residual_attn
        
        # if model is in the evaluation mode, store the attention output

        ##-----------------Feed-Forward Network-----------------##
        residual_mlp = attn_res_output

        # layer-normalization
        x = self.ln_2(attn_res_output)
        # Apply position-wise feed-forward network.
        mlp_outputs = self.mlp(x)
        mlp_output, _ = mlp_outputs

        # residual connection
        output = mlp_output + residual_mlp
        
        # if model is in the evaluation mode, store the mlp output
        
        return output, EasyDict({
            'input': in_x,
            'attn_res_output': attn_res_output,
            'output': output,
            })

class MultiHeadAttentionBlock(nnModule):
    """Implementation of a Transformer Block without MLP layer.
    """

    def __init__(
            self, model_config: EasyDict, layer_idx: int=None,
    ):
        """
        Initialize the TransformerBlock module.
        """
        super().__init__()
        
        hidden_size = model_config.hidden_size
        num_heads = model_config.num_heads
        self.relpos_shift = model_config.relpos_shift
        # check if n_inner is defined in the model_configuration
        inner_size = model_config.n_inner if hasattr(model_config, "n_inner") else 4 * hidden_size

        self.ln = nn.LayerNorm(hidden_size, eps=model_config.layer_norm_eps) if model_config.use_layer_norm else nn.Identity()
        self.attn = MultiHeadAttentionDeBERTa(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_type=model_config.attention_type,
            use_bias=model_config.use_bias,
            attn_pdrop=model_config.attn_pdrop,
            q_k_v_o_proj_enabled=model_config.q_k_v_o_proj_enabled,
            relpos_q_k_enabled=model_config.relpos_q_k_enabled,
            relpos_embed_size=model_config.relpos_embed_size if hasattr(model_config, "relpos_embed_size") else hidden_size, 
            causal_attn=model_config.causal_attn,
        )
        # self.ln_2 = nn.LayerNorm(hidden_size, eps=model_config.layer_norm_eps) if model_config.use_layer_norm else nn.Identity()
        # self.mlp = MLP(hidden_size, inner_size, model_config.resid_pdrop)

        # store hyperparameters
        self.hidden_size = hidden_size
        self.inner_size = inner_size

    def forward(
        self, 
        in_x: torch.Tensor,
        pos_model: Optional[nn.Module] = None,
        mask: Optional[Union[torch.Tensor, None]] = None,
        head_mask: Optional[torch.Tensor] = None,
        logits_shift: Optional[torch.Tensor] = None,
    ):
        """
        Apply a Transformer block to the input tensor.

        Args:
            in_x (torch.Tensor): A 3D tensor of shape [batch_size, seq_len, input_size].
            mask (Optional[Union[torch.Tensor, None]]): Optional mask tensor 
                of shape [batch_size, 1, seq_len, seq_len].
            head_mask (Optional[torch.Tensor]): Optional head mask tensor.

        Returns:
            torch.Tensor: A 3D tensor of shape [batch_size, seq_len, hidden_size].
        """

        ##-----------------Attention-----------------##
        residual_attn = in_x

        # layer-normalization
        x = self.ln(in_x)
        # Apply multi-head self-attention.
        attn_outputs = self.attn(
            x, 
            pos_model=pos_model,
            mask=mask,
            head_mask=head_mask,
            logits_shift=logits_shift,
        )
        attn_output, _ = attn_outputs # the `intermediate` is not used in the forward pass, and we will handle it by forward hook.

        # residual connection
        output = attn_output + residual_attn
        
        # if model is in the evaluation mode, store the attention output
        
        return output, EasyDict({
            'input': in_x,
            'attn_output': attn_output,
            'output': output,
            })

class ReadOut(nnModule):
    def __init__(self, 
                 hidden_size: int, 
                 vocab_size: int, 
                 readin: Optional[nn.Module]=None
                 ):
        super().__init__()
        self.emb2idx = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x: torch.Tensor):
        logits = self.emb2idx(x)  # [batch_size, seq_len, vocab_size]
        return logits

class ReadIn(nnModule):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.idx2emb = nn.Embedding(vocab_size, hidden_size)
        
    def forward(self, input_token_ids: torch.Tensor):
        return self.idx2emb(input_token_ids)
    
class TransformerEncoder(nnModule):
    """Implementation of Transformer encoder."""

    def __init__(
            self, 
            model_config: EasyDict,
            **kwargs: Any,
    ):
        """
        Initialize the TransformerEncoder module.

        Args:
            model_config (EasyDict): The model_configuration of the model.
        """
        super().__init__()

        num_layers = model_config.num_layers
        hidden_size = model_config.hidden_size
        num_heads = model_config.num_heads

        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}!"

        ## ----------  positional embedding ---------- ##
        pos_emb_type = model_config.pos_enc_type
        max_seq_len = model_config.max_seq_len if hasattr(model_config, "max_seq_len") else 512
        relpos_win_size = model_config.relpos_win_size if hasattr(model_config, "relpos_win_size") else max_seq_len
        
        if pos_emb_type == "NoPE": # no positional encoding
            self.pos_model = None
        elif pos_emb_type == "AbPE": # absolute positional embedding
            self.pos_model = AbsolutePositionalEmbeddings(
                dim=hidden_size,
                max_position_embeddings=max_seq_len,
                use_pos_layer_norm=model_config.use_pos_layer_norm,
            )
        elif pos_emb_type == "RoPE": # rotary positional encoding
            self.pos_model = RotaryPositionalEmbeddings(
                dim=hidden_size//num_heads, 
                max_seq_len=max_seq_len,
                )
        elif pos_emb_type == "DeBERTa": # DeBERTa positional embedding
            self.pos_model = DeBERTaPositionalEmbeddings(
                dim=model_config.relpos_embed_size if hasattr(model_config, "relpos_embed_size") else hidden_size,
                relpos_win_size=relpos_win_size,
                relpos_shift=model_config.relpos_shift,
                relpos_q_k_enabled=model_config.relpos_q_k_enabled, 
                use_pos_layer_norm=model_config.use_pos_layer_norm,
                )
        else:
            raise ValueError(f"pos_enc_type {pos_emb_type} is not supported!")
        
        self.pos_emb_type = pos_emb_type
        
        # ## ----------  token embedding ---------- ##
        # self.readin = ReadIn(model_config.vocab_size, hidden_size)
        
        self.dropout = nn.Dropout(model_config.resid_pdrop)
        
        
        ## ----------  transformer blocks ---------- ##
        self.blocks = nnModuleDict(
            {
                f"layer_{layer_id}": TransformerBlock(
                    model_config=model_config,
                    layer_idx=layer_id,
                )
                for layer_id in range(num_layers)
            }
        )

        ## ----------  layer normalization ---------- ##
        if model_config.use_layer_norm:
            self.ln_final = nn.LayerNorm(hidden_size, eps=model_config.layer_norm_eps)
        self.use_layer_norm = model_config.use_layer_norm

        
    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        mask: Optional[Union[torch.Tensor, None]] = None,
    ):
        """
        Apply the Transformer encoder to the input tensor.
        
        Args:
            x (Optional[torch.Tensor], optional): The input tensor. Defaults to None.
            mask (Optional[Union[torch.Tensor, None]], optional): The mask tensor. Defaults to None.
            
        Returns:
            torch.Tensor: The output tensor after applying the Transformer encoder.
        """
        # ## ----------  token embedding ---------- ##
        # x = self.readin(input_token_ids)
        
        ## ----------  absolute positional embedding ---------- ##
        if self.pos_emb_type == "AbPE":
            x = self.pos_model(x)
        
        ## ----------  dropout ---------- ##
        x = self.dropout(x)
        
        ## ----------  transformer blocks ---------- ##
        # DeBERTa positional embedding and rotary positional encoding are applied in the transformer block.
        for name, block in self.blocks.items():
            pos_model = self.pos_model if self.pos_emb_type == "DeBERTa" or self.pos_emb_type == "RoPE" else None
            x, _ = block(x, 
                      pos_model=pos_model,
                      mask=mask)
        
        ## ----------  layer normalization ---------- ##
        if self.use_layer_norm:
            x = self.ln_final(x)
        
        # ## ----------  readout layer ---------- ##
        # if self.use_readout_proj:
        #     output = self.readout(x)
        # else:
        #     # return the all token embeddings and multiple the last token embedding by the weight matrix
        #     output = x
        #     logits = torch.matmul(output, self.readin.idx2emb.weight.T)
        #     output = nn.functional.softmax(logits, dim=-1)
        return x
        
class TransformerEncoderOnlyAttn(TransformerEncoder):
    """Implementation of Transformer Encoder with only attention blocks."""

    def __init__(
            self, 
            model_config: EasyDict,
            **kwargs: Any,
    ):
        """
        Initialize the TransformerEncoder module.

        Args:
            model_config (EasyDict): The model_configuration of the model.
        """
        super().__init__(model_config=model_config, **kwargs)
        ## ----------  transformer blocks ---------- ##
        self.blocks = nnModuleDict(
            {
                f"layer_{layer_id}": MultiHeadAttentionBlock(
                    model_config=model_config,
                    layer_idx=layer_id,
                )
                for layer_id in range(model_config.num_layers)
            }
        )
        ## ----------  layer normalization ---------- ##
        if model_config.use_layer_norm:
            self.ln_final = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)
        self.use_layer_norm = model_config.use_layer_norm
    
class LinearWithChannel(nnModule):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 in_channel_size_ls: Union[list, tuple, int], 
                 out_channel_size_ls: Union[list, tuple, int],
                 ):
        super(LinearWithChannel, self).__init__()
        if isinstance(in_channel_size_ls, int):
            in_channel_size_ls = [in_channel_size_ls]
        if isinstance(out_channel_size_ls, int):
            out_channel_size_ls = [out_channel_size_ls]
        
        #initialize weights
        self.weight = torch.nn.Parameter(torch.randn(*in_channel_size_ls, *out_channel_size_ls, output_size, input_size))
        # shape of weight: (*in_channel_size_ls, *out_channel_size_ls, output_size, input_size)
        
        self.bias = torch.nn.Parameter(torch.randn(*in_channel_size_ls, *out_channel_size_ls, output_size))
        # shape of bias: (*in_channel_size_ls, *out_channel_size_ls, output_size)
        
        self.in_channel_size_ls = in_channel_size_ls
        self.out_channel_size_ls = out_channel_size_ls
    
    def forward(self, x: torch.Tensor):
        """
        Args:
        - x: tensor of shape (batch_size, *in_channel_size_ls, input_size)

        Returns:
        - output: tensor of shape (batch_size, *in_channel_size_ls, *out_channel_size_ls, output_size)
        """
        
        # expand x to shape (batch_size, *in_channel_size_ls, *out_channel_size_ls, input_size)
        
        # Step 1: add len(out_channel_size_ls) dimensions to x
        for _ in range(len(self.out_channel_size_ls)):
            x = x.unsqueeze(-2)
        # x.shape: (batch_size, *in_channel_size_ls, 1, ...1, input_size)
        # append a dimension to the end of x
        x = x.unsqueeze(-1)
        # x.shape: (batch_size, *in_channel_size_ls, 1, ...1, input_size, 1)
            
        # Step 2: perform matrix multiplication with weight
        output = torch.matmul(self.weight, x)
        # output.shape: (batch_size, *in_channel_size_ls, *out_channel_size_ls, output_size, 1)
        
        # Step 3: remove the last dimension
        output = output.squeeze(-1)
        # output.shape: (batch_size, *in_channel_size_ls, *out_channel_size_ls, output_size)
        
        # Step 4: add bias
        output = output + self.bias
        
        return output
        
        
class SparseAutoEncoder(nnModule):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 activation: str='relu',
                 **kwargs,
                 ):
        super(SparseAutoEncoder, self).__init__()
        # weight = nn.Parameter(torch.randn(hidden_size, input_size))
        # self.encoder = nn.Linear(input_size, self.hidden_size, bias=True)
        # self.decoder = nn.Linear(self.hidden_size, input_size, bias=False)
        self.hidden_size = int(hidden_size)
        self.act = Activation(activation, **kwargs)
        self.W = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_size))
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))

        # weight tying
        # self.decoder.weight.data = self.encoder.weight.data.T
                
        # initialize bias
        # self.encoder.bias.data.fill_(0.0)
        
        # initialize the encoder weight
        # nn.init.kaiming_uniform_(self.encoder.weight.data, a=math.sqrt(5))
        # nn.init.zeros_(self.encoder.bias.data)
        nn.init.zeros_(self.encoder_bias.data)
        nn.init.kaiming_uniform_(self.W.data, a=math.sqrt(5))
        nn.init.zeros_(self.decoder_bias.data)
        
    def forward(self, x: torch.Tensor):
        """
        Args:
        - x: tensor of shape (batch_size, input_size)
        """
        # # Step 1: encode
        # pre_act = self.encoder(x)
        # post_act = self.act(pre_act)
        
        # # Step 2: decode
        # x = self.decoder(post_act)
        pre_act = x @ self.W.t() + self.encoder_bias
        post_act = self.act(pre_act)
        x = post_act @ self.W + self.decoder_bias
        return x, pre_act

class SparseAutoEncoderWithThreshold(nnModule):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 **kwargs,
                 ):
        
        super(SparseAutoEncoderWithThreshold, self).__init__()
        # weight = nn.Parameter(torch.randn(hidden_size, input_size))
        # self.encoder = nn.Linear(input_size, self.hidden_size, bias=True)
        # self.decoder = nn.Linear(self.hidden_size, input_size, bias=False)
        self.hidden_size = int(hidden_size)
        self.threshold = torch.zeros(hidden_size) * -0.5
        self.W = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))

        # weight tying
        # self.decoder.weight.data = self.encoder.weight.data.T
                
        # initialize bias
        # self.encoder.bias.data.fill_(0.0)
        
        # initialize the encoder weight
        # nn.init.kaiming_uniform_(self.encoder.weight.data, a=math.sqrt(5))
        # nn.init.zeros_(self.encoder.bias.data)
        nn.init.kaiming_uniform_(self.W.data, a=math.sqrt(5))
        nn.init.zeros_(self.decoder_bias.data)
    
    def set_threshold(self, threshold: torch.Tensor):
        assert threshold.shape == (self.hidden_size,), f"Threshold shape {threshold.shape} does not match the hidden size {self.hidden_size}!"
        self.threshold = threshold #

    def forward(self, x: torch.Tensor):
        """
        Args:
        - x: tensor of shape (batch_size, input_size)
        """

        pre_act = x @ self.W.t()
        threshold = self.threshold.unsqueeze(0) # (1, hidden_size)
        post_act = pre_act.clamp_(min=threshold)
        x = post_act @ self.W + self.decoder_bias
        return x, pre_act



class TopKSparseAutoEncoder(nnModule):
    def __init__(self, 
                input_size, 
                hidden_size, 
                k: int,  # Number of top activations to retain
                activation: str='relu',
                **kwargs,
                ):
        super(TopKSparseAutoEncoder, self).__init__()
        self.hidden_size = int(hidden_size)
        self.k = k  # Top-K activations
        self.act = Activation(activation, **kwargs)
        
        self.W = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_size))
        
        nn.init.zeros_(self.encoder_bias.data)
        nn.init.kaiming_uniform_(self.W.data, a=math.sqrt(5))


    def forward(self, x: torch.Tensor):
        """
        Args:
        - x: tensor of shape (batch_size, input_size)
        Returns:
        - x_recon: tensor of shape (batch_size, input_size)
        - pre_act: tensor of shape (batch_size, hidden_size)
        - post_act: tensor of shape (batch_size, hidden

        """

        # Encoding step
        pre_act = x @ self.W.t() + self.encoder_bias
        post_act = self.act(pre_act)

        # Top-K Sparsification
        if self.k < self.hidden_size:
            threshold = torch.topk(post_act, self.k, dim=1, largest=True, sorted=False).values[:, -1:]
            mask = post_act >= threshold  # Retain only top-K activations
            post_act = post_act * mask  # Zero out other activations

        # Decoding step
        x_recon = post_act @ self.W
        return x_recon, EasyDict({
            'post_act': post_act,
            'pre_act': pre_act,
        })
        
import re

class Activation(nnModule):
    def __init__(self, activation: str, **kwargs):
        super(Activation, self).__init__()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'sigmoidlu':
            self.act = SigmoidLU()
        elif activation == 'jumprelu':
            self.act = JumpReLU(kwargs.get('theta', torch.tensor(0.0)))
        elif activation == 'thresholdrelu':
            self.act = ThresholdReLU()
            
        elif 'powerrelu' in activation:
            # suppose the activation name is something like 'powerrelu2.0', 'powerrelu-2.0', 'powerrelu2', 'powerrelu - 2' so on, find the power value in the string
            power = re.search(r'powerrelu\s*[-]?\s*(\d+\.?\d*)', activation)
            if power:
                power = float(power.group(1))
            else:
                raise ValueError(f"Please provide the power value in the activation name {activation}!")
            self.act = PowerReLU(power)
        else:
            raise ValueError(f"Activation {activation} is not supported!")
    
    def forward(self, 
                x: torch.Tensor, 
                topk: Optional[int]=None, 
                bias: Optional[torch.Tensor]=None):
        if isinstance(self.act, ThresholdReLU):
            post_act = self.act(x, bias)
        else:
            post_act = self.act(x)
        if topk is not None:
            threshold = torch.topk(post_act, topk, dim=-1, largest=True, sorted=False).values[..., -1:]
            mask = post_act >= threshold
            post_act = post_act * mask
        return post_act
    
        
class SAEWithChannel(nnModule):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 channel_size_ls: Union[list, tuple, int],
                 activation: str='relu',
                 use_neuron_weight: bool=False,
                 **kwargs,
                 ):
        super(SAEWithChannel, self).__init__()
        self.hidden_size = int(hidden_size)
        
        if 'group_indices' in kwargs: 
            self.group_indices = kwargs['group_indices']
            self._W_enc = nn.ParameterList([
                nn.Parameter(torch.randn(*channel_size_ls, self.group_indices[0], input_size))
            ])
            self._b_enc = nn.ParameterList([
                nn.Parameter(torch.zeros(*channel_size_ls, self.group_indices[0]))
            ])
            for i in range(1, len(self.group_indices)):
                self._W_enc.append(
                    nn.Parameter(torch.randn(*channel_size_ls, self.group_indices[i] - self.group_indices[i-1], input_size))
                )
                self._b_enc.append(
                    nn.Parameter(torch.zeros(*channel_size_ls, self.group_indices[i] - self.group_indices[i-1]))
                )
        else:
            self._W_enc = nn.Parameter(torch.randn(*channel_size_ls, hidden_size, input_size))
            self._b_enc = nn.Parameter(torch.zeros(*channel_size_ls, hidden_size))
            
        self.b_dec = nn.Parameter(torch.randn(*channel_size_ls, input_size))
        self.act = Activation(activation, **kwargs)
        
        if use_neuron_weight:
            self.neuron_weight = nn.Parameter(torch.ones(*channel_size_ls, hidden_size))
        
        # initialize the encoder weight
        nn.init.kaiming_uniform_(self.W_enc.data, a=math.sqrt(5))
        # initialize the encoder bias
        nn.init.zeros_(self.b_enc.data)
        nn.init.zeros_(self.b_dec.data)
    
    @property 
    def W_enc(self):
        if isinstance(self._W_enc, nn.ParameterList):
            return torch.cat(list(self._W_enc), dim=-2)
        else:
            return self._W_enc
    
    @property
    def b_enc(self):
        if isinstance(self._b_enc, nn.ParameterList):
            return torch.cat(list(self._b_enc), dim=-1)
        else:
            return self._b_enc
        
    def init_neuron_weight(self, b_enc_value: float=1.0):
        self.neuron_weight = nn.Parameter(torch.ones_like(self.b_enc) * b_enc_value)
    
    @property
    def W(self):
        return self.W_enc
    
    @property
    def encoder_bias(self):
        return self.b_enc
    
    def prune_neurons(self, neuron_mask: torch.Tensor, verbose: bool=False):
        # Apply neuron mask and delete the neurons with False mask
        self.hidden_size = neuron_mask.sum().item()
        self.W_enc = nn.Parameter(self.W_enc[..., neuron_mask, :])
        self.b_enc = nn.Parameter(self.b_enc[..., neuron_mask])
        if hasattr(self, 'neuron_weight'):
            self.neuron_weight = nn.Parameter(self.neuron_weight[..., neuron_mask])
            
        if verbose:
            print(f"Pruned {neuron_mask.size(0) - self.hidden_size} neurons!")
        
    
    def forward(self, 
                x: torch.Tensor, 
                neuron_mask: Optional[torch.Tensor]=None,
                topk: Optional[int]=None,):
        """
        Args:
        - x: tensor of shape (batch_size, *channel_size_ls, input_size)
        """
        x_centered = x - self.b_dec
        
        pre_act = torch.einsum('...ij,...j->...i', self.W_enc, x_centered) + self.b_enc
        # pre_act = torch.matmul(self.W_enc, x_centered.unsqueeze(-1)).squeeze(1) + self.b_enc # shape: (batch_size, *channel_size_ls, hidden_size)
        
        if neuron_mask is not None:
            # assert neuron_mask.shape == self.b_enc.shape, f"neuron_mask shape {neuron_mask.shape} does not match the hidden size {self.hidden_size}!"
            pre_act = pre_act * neuron_mask.float() # Apply neuron mask
        
        if isinstance(self.act.act, ThresholdReLU):
            post_act = self.act(pre_act, bias=self.b_enc)
        else:
            post_act = self.act(pre_act, topk) # shape: (batch_size, *channel_size_ls, hidden_size)
        
        
        if hasattr(self, 'neuron_weight'):
            post_act = post_act * self.neuron_weight
        
        
        x_reconstructed = torch.einsum('...ij,...i->...j', self.W_enc, post_act) + self.b_dec
        # x_reconstructed = torch.matmul(self.W_enc.transpose(-1, -2), post_act.unsqueeze(-1)).squeeze(1) + self.b_dec
        
        # reconstruct_loss = nn.functional.mse_loss(x_reconstructed, x, reduction='mean')
        
        # l1_loss = self.l1_penalty * post_act * torch.norm(self.W_enc, p=2, dim=-1).unsqueeze(0)
        
        return x_reconstructed, EasyDict({
            'post_act': post_act,
            'pre_act': pre_act,
        })
        
    # def save_state_dict(self) -> dict:
    #     """
    #     Saves the state of the SAEWithChannel model.
        
    #     Returns:
    #         dict: A dictionary containing the model's state including:
    #             - W_enc: The encoder weights
    #             - b_enc: The encoder biases
    #             - b_dec: The decoder biases
    #             - neuron_weight: The neuron weights (if they exist)
    #     """
    #     state_dict = {
    #         'W_enc': self.W_enc.data,
    #         'b_enc': self.b_enc.data,
    #         'b_dec': self.b_dec.data,
    #     }
        
    #     # Only save neuron_weight if it exists
    #     if hasattr(self, 'neuron_weight'):
    #         state_dict['neuron_weight'] = self.neuron_weight.data
            
    #     return state_dict
    
    # def load_state_dict(self, state_dict: dict):
    #     """
    #     Loads the state of the SAEWithChannel model from a state dictionary.
        
    #     Args:
    #         state_dict (dict): A dictionary containing the model's state including:
    #             - W_enc: The encoder weights
    #             - b_enc: The encoder biases
    #             - b_dec: The decoder biases
    #             - neuron_weight: The neuron weights (if they exist)
    #     """
    #     # Load encoder weights
    #     if isinstance(self._W_enc, nn.ParameterList):
    #         # Handle grouped weights case
    #         total_size = 0
    #         for i, w in enumerate(self._W_enc):
    #             group_size = w.shape[-2]  # Get the size of this group
    #             self._W_enc[i].data = state_dict['W_enc'][..., total_size:total_size + group_size, :]
    #             total_size += group_size
    #     else:
    #         # Handle single tensor case
    #         self._W_enc.data = state_dict['W_enc']
        
    #     # Load encoder biases
    #     if isinstance(self._b_enc, nn.ParameterList):
    #         # Handle grouped biases case
    #         total_size = 0
    #         for i, b in enumerate(self._b_enc):
    #             group_size = b.shape[-1]  # Get the size of this group
    #             self._b_enc[i].data = state_dict['b_enc'][..., total_size:total_size + group_size]
    #             total_size += group_size
    #     else:
    #         # Handle single tensor case
    #         self._b_enc.data = state_dict['b_enc']
        
    #     # Load decoder biases
    #     self.b_dec.data = state_dict['b_dec']
        
    #     # Load neuron weights if they exist in both the model and state_dict
    #     if hasattr(self, 'neuron_weight') and 'neuron_weight' in state_dict:
    #         self.neuron_weight.data = state_dict['neuron_weight']
        

# add a intermediate model where the gradient backpropagation is scaled by a factor
# NOTE: should use apply method for autograd function https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
class GradRescaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(torch.tensor([scale]))
        return input

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.saved_tensors[0]
        return grad_output * scale.item(), None
    
class LayerWithGradRescale(nn.Module):
    def __init__(self):
        super(LayerWithGradRescale, self).__init__()
        self.fn = GradRescaler.apply
    
    def forward(self, x, scale):
        return self.fn(x, scale)