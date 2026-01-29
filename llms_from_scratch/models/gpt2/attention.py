import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
        max_seq_len: int | None = None,
        window_size: int | None = None,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len or context_length
        self.window_size = window_size or self.max_seq_len
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        b, num_tokens, _d_in = x.shape
        if use_cache:
            assert num_tokens <= self.window_size, (
                f"Input chunk size ({num_tokens}) exceeds KV cache window size ({self.window_size})."
            )
        keys_new = self.W_key(x)
        values_new = self.W_value(x)
        queries = self.W_query(x)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)
        queries = queries.transpose(1, 2)
        if use_cache:
            if self.cache_k is None or self.cache_k.size(0) != b:
                self.cache_k = torch.zeros(
                    b, self.num_heads, self.window_size, self.head_dim, device=x.device
                )
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_cur = 0
            # handle overflow by shifting cache
            if self.ptr_cur + num_tokens > self.window_size:
                overflow = self.ptr_cur + num_tokens - self.window_size
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()
                self.ptr_cur -= overflow
            self.cache_k[:, :, self.ptr_cur : self.ptr_cur + num_tokens, :] = keys_new
            self.cache_v[:, :, self.ptr_cur : self.ptr_cur + num_tokens, :] = values_new
            self.ptr_cur += num_tokens
            keys = self.cache_k[:, :, : self.ptr_cur, :]
            values = self.cache_v[:, :, : self.ptr_cur, :]
        else:
            keys, values = keys_new, values_new
            self.ptr_cur = 0
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        k_len = attn_scores.size(-1)
        if num_tokens == k_len:
            causal_mask = torch.triu(
                torch.ones(num_tokens, k_len, device=x.device, dtype=torch.bool), diagonal=1
            )
        else:
            # cached case: only mask future positions for query tokens
            offset = k_len - num_tokens
            row_idx = torch.arange(num_tokens, device=x.device).unsqueeze(1)
            col_idx = torch.arange(k_len, device=x.device).unsqueeze(0)
            causal_mask = row_idx + offset < col_idx
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None
