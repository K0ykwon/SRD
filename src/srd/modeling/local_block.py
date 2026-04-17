"""Local-only sequence block used for regular token updates in SRD."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LocalBlock(nn.Module):
    """Applies causal local attention without any path to the long-memory bank."""

    def __init__(self, d_model: int, num_heads: int, window_size: int, dropout_p: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = d_model // num_heads

        self.norm_1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_p)

        self.norm_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def _local_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Builds a small diagnostic mask for return_attention mode."""
        positions = torch.arange(seq_len, device=device)
        distance = positions[:, None] - positions[None, :]
        return (distance >= 0) & (distance <= self.window_size)

    def _reshape_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Updates all positions using only a bounded causal local window."""
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states
        normed = self.norm_1(hidden_states)
        qkv = self.qkv(normed)
        query, key, value = qkv.chunk(3, dim=-1)

        query = self._reshape_heads(query, batch_size, seq_len)
        key = self._reshape_heads(key, batch_size, seq_len)
        value = self._reshape_heads(value, batch_size, seq_len)

        mask = self._local_causal_mask(seq_len, hidden_states.device)
        if return_attention:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
            attention = torch.softmax(scores, dim=-1)
            attention = self.dropout(attention)
            attended = torch.matmul(attention, value)
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        hidden_states = residual + self.out_proj(attended)
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        if return_attention:
            return hidden_states, attention
        return hidden_states

    def prefill_cache(self, hidden_states: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Runs one full local block forward and returns a bounded KV cache for incremental decode."""
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states
        normed = self.norm_1(hidden_states)
        qkv = self.qkv(normed)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self._reshape_heads(query, batch_size, seq_len)
        key = self._reshape_heads(key, batch_size, seq_len)
        value = self._reshape_heads(value, batch_size, seq_len)

        mask = self._local_causal_mask(seq_len, hidden_states.device)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        hidden_states = residual + self.out_proj(attended)
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))

        cache_len = min(self.window_size, key.size(2))
        if cache_len > 0:
            cached_key = key[:, :, -cache_len:, :].detach()
            cached_value = value[:, :, -cache_len:, :].detach()
        else:
            cached_key = key[:, :, :0, :].detach()
            cached_value = value[:, :, :0, :].detach()
        return hidden_states, {"key": cached_key, "value": cached_value}

    def forward_step(self, hidden_states: Tensor, cache: dict[str, Tensor] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        """Runs one incremental local-attention step and updates the bounded KV cache."""
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len != 1:
            raise ValueError("forward_step expects a single new token")
        residual = hidden_states
        normed = self.norm_1(hidden_states)
        qkv = self.qkv(normed)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self._reshape_heads(query, batch_size, seq_len)
        key = self._reshape_heads(key, batch_size, seq_len)
        value = self._reshape_heads(value, batch_size, seq_len)

        if cache is None:
            cached_key = key
            cached_value = value
        else:
            cached_key = torch.cat([cache["key"], key], dim=2)
            cached_value = torch.cat([cache["value"], value], dim=2)

        attended = F.scaled_dot_product_attention(
            query,
            cached_key,
            cached_value,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        hidden_states = residual + self.out_proj(attended)
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))

        cache_len = min(self.window_size, cached_key.size(2))
        if cache_len > 0:
            next_key = cached_key[:, :, -cache_len:, :].detach()
            next_value = cached_value[:, :, -cache_len:, :].detach()
        else:
            next_key = cached_key[:, :, :0, :].detach()
            next_value = cached_value[:, :, :0, :].detach()
        return hidden_states, {"key": next_key, "value": next_value}
