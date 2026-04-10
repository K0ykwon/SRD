"""Token-level bank reader used only for the summary-memory baseline."""

import math

import torch
from torch import Tensor, nn


class TokenMemoryBlock(nn.Module):
    """Lets ordinary token states cross-attend directly to a shared summary bank."""

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.token_norm = nn.LayerNorm(d_model)
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

    def forward(self, hidden_states: Tensor, bank_states: Tensor) -> tuple[Tensor, dict]:
        """Returns token states updated by direct summary-bank access."""
        if bank_states.size(1) == 0:
            return hidden_states, {
                "bank_used": False,
                "bank_read_slots": 0,
                "token_bank_access_count": 0,
                "attention": None,
            }

        batch_size, token_len, _ = hidden_states.shape
        bank_len = bank_states.size(1)

        residual = hidden_states
        query = self.query_proj(self.token_norm(hidden_states))
        key = self.key_proj(self.bank_norm(bank_states))
        value = self.value_proj(self.bank_norm(bank_states))

        def reshape_heads(x: Tensor, length: int) -> Tensor:
            return x.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)

        query = reshape_heads(query, token_len)
        key = reshape_heads(key, bank_len)
        value = reshape_heads(value, bank_len)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, token_len, self.d_model)

        hidden_states = residual + self.out_proj(attended)
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states, {
            "bank_used": True,
            "bank_read_slots": token_len,
            "token_bank_access_count": token_len,
            "attention": attention,
        }
