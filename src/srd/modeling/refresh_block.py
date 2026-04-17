"""Refresh-state block that is allowed to consume long-memory bank context."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RefreshBlock(nn.Module):
    """Updates refresh states by attending over the shared long-memory bank."""

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.refresh_norm = nn.LayerNorm(d_model)
        self.bank_norm = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_p)

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, refresh_states: Tensor, bank_states: Tensor) -> tuple[Tensor, dict]:
        """Returns updated refresh states plus a small access trace."""
        if bank_states.size(1) == 0:
            return refresh_states, {
                "bank_used": False,
                "bank_read_slots": 0,
                "attention": None,
            }

        batch_size, refresh_len, _ = refresh_states.shape
        bank_len = bank_states.size(1)

        refresh_residual = refresh_states
        query = self.query_proj(self.refresh_norm(refresh_states))
        key = self.key_proj(self.bank_norm(bank_states))
        value = self.value_proj(self.bank_norm(bank_states))

        def reshape_heads(x: Tensor, length: int) -> Tensor:
            return x.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)

        query = reshape_heads(query, refresh_len)
        key = reshape_heads(key, bank_len)
        value = reshape_heads(value, bank_len)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, refresh_len, self.d_model)

        refreshed = refresh_residual + self.out_proj(attended)
        refreshed = refreshed + self.mlp(self.mlp_norm(refreshed))
        return refreshed, {
            "bank_used": True,
            "bank_read_slots": refresh_len,
            "attention": None,
        }
