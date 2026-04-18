"""Full causal attention block for conventional decoder Transformer baselines."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FullBlock(nn.Module):
    """Applies standard causal self-attention over the full visible prefix."""

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
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
        self._mask_cache: dict[tuple[int, str], Tensor] = {}

    def _causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Builds the standard decoder-style causal attention mask."""
        device_key = f"{device.type}:{device.index if device.index is not None else -1}"
        cache_key = (seq_len, device_key)
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached
        positions = torch.arange(seq_len, device=device)
        mask = positions[:, None] >= positions[None, :]
        self._mask_cache[cache_key] = mask
        return mask

    def forward(self, hidden_states: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Updates all positions using full causal self-attention."""
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states
        normed = self.norm_1(hidden_states)
        qkv = self.qkv(normed)
        query, key, value = qkv.chunk(3, dim=-1)

        def reshape_heads(x: Tensor) -> Tensor:
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        query = reshape_heads(query)
        key = reshape_heads(key)
        value = reshape_heads(value)

        if return_attention:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = self._causal_mask(seq_len, hidden_states.device)
            scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
            attention = torch.softmax(scores, dim=-1)
            attention = self.dropout(attention)
            attended = torch.matmul(attention, value)
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        hidden_states = residual + self.out_proj(attended)
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        if return_attention:
            return hidden_states, attention
        return hidden_states

    def prefill_cache(self, hidden_states: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Runs a full forward pass for one layer and returns layer KV cache."""
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states
        normed = self.norm_1(hidden_states)
        qkv = self.qkv(normed)
        query, key, value = qkv.chunk(3, dim=-1)

        def reshape_heads(x: Tensor) -> Tensor:
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        query = reshape_heads(query)
        key = reshape_heads(key)
        value = reshape_heads(value)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        hidden_states = residual + self.out_proj(attended)
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        return hidden_states, {"key": key.detach(), "value": value.detach()}

    def forward_step(self, hidden_states: Tensor, cache: dict[str, Tensor] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        """Runs one incremental causal step and updates layer KV cache."""
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len != 1:
            raise ValueError("forward_step expects a single new token")
        residual = hidden_states
        normed = self.norm_1(hidden_states)
        qkv = self.qkv(normed)
        query, key, value = qkv.chunk(3, dim=-1)

        def reshape_heads(x: Tensor) -> Tensor:
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        query = reshape_heads(query)
        key = reshape_heads(key)
        value = reshape_heads(value)

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
        return hidden_states, {"key": cached_key.detach(), "value": cached_value.detach()}
