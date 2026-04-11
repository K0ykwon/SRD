"""Additional conventional baselines for the final SRD comparison pass."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor, nn

from srd.config import SRDConfig
from srd.modeling.local_block import LocalBlock
from srd.modeling.refresh_block import RefreshBlock
from srd.modeling.token_memory_block import TokenMemoryBlock


class TransformerXLStyleMemoryModel(nn.Module):
    """Block-recurrent baseline where tokens read a bounded cache of past token states."""

    def __init__(self, config: SRDConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.local_blocks = nn.ModuleList(
            [
                LocalBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    window_size=config.local_window,
                    dropout_p=config.dropout_p,
                )
                for _ in range(max(1, config.num_layers - 1))
            ]
        )
        self.memory_block = TokenMemoryBlock(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout_p=config.dropout_p,
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _block_ranges(self, seq_len: int) -> List[tuple[int, int]]:
        block_size = self.config.effective_block_size()
        return [
            (start, min(start + block_size, seq_len))
            for start in range(0, seq_len, block_size)
        ]

    def _truncate_memory(self, memory_states: Tensor) -> Tensor:
        max_tokens = self.config.memory_blocks * self.config.effective_block_size()
        if memory_states.size(1) <= max_tokens:
            return memory_states
        return memory_states[:, -max_tokens:, :]

    def forward(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> Dict[str, Tensor]:
        batch_size, seq_len = input_ids.shape
        embeddings = self.embedding(input_ids)
        memory_states = embeddings[:, :0, :]

        output_blocks = []
        token_bank_access_count = 0
        bank_read_slots = 0
        block_ranges = self._block_ranges(seq_len)

        for start, end in block_ranges:
            hidden_states = embeddings[:, start:end, :]
            for block in self.local_blocks:
                hidden_states = block(hidden_states)
            hidden_states, memory_trace = self.memory_block(hidden_states, memory_states)
            output_blocks.append(hidden_states)
            token_bank_access_count += memory_trace["token_bank_access_count"]
            bank_read_slots += memory_trace["bank_read_slots"]
            memory_states = self._truncate_memory(torch.cat([memory_states, hidden_states.detach()], dim=1))

        hidden_states = torch.cat(output_blocks, dim=1)
        logits = self.lm_head(self.final_norm(hidden_states))
        empty = hidden_states[:, :0, :]
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "refresh_states": empty,
            "bank_states": memory_states,
            "predicted_summary": empty,
            "target_summary": empty,
            "debug": {
                "segment_count": len(block_ranges),
                "bank_read_segments": sum(1 for _ in block_ranges),
                "bank_read_slots": bank_read_slots,
                "token_bank_access_count": token_bank_access_count,
                "refresh_bank_access_count": 0,
                "memory_blocks": self.config.memory_blocks,
            },
        }


class PerceiverLatentModel(nn.Module):
    """Baseline where all blocks interact through a small shared latent array."""

    def __init__(self, config: SRDConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.latents = nn.Parameter(torch.randn(config.latent_slots, config.d_model) * 0.02)
        self.local_blocks = nn.ModuleList(
            [
                LocalBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    window_size=config.local_window,
                    dropout_p=config.dropout_p,
                )
                for _ in range(max(1, config.num_layers - 1))
            ]
        )
        self.token_to_latent = TokenMemoryBlock(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout_p=config.dropout_p,
        )
        self.latent_update = RefreshBlock(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout_p=config.dropout_p,
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _block_ranges(self, seq_len: int) -> List[tuple[int, int]]:
        block_size = self.config.effective_block_size()
        return [
            (start, min(start + block_size, seq_len))
            for start in range(0, seq_len, block_size)
        ]

    def forward(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> Dict[str, Tensor]:
        batch_size, seq_len = input_ids.shape
        embeddings = self.embedding(input_ids)
        latent_states = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        output_blocks = []
        token_bank_access_count = 0
        bank_read_slots = 0
        block_ranges = self._block_ranges(seq_len)

        for start, end in block_ranges:
            hidden_states = embeddings[:, start:end, :]
            for block in self.local_blocks:
                hidden_states = block(hidden_states)
            hidden_states, token_trace = self.token_to_latent(hidden_states, latent_states)
            latent_states, latent_trace = self.latent_update(latent_states, hidden_states)
            output_blocks.append(hidden_states)
            token_bank_access_count += token_trace["token_bank_access_count"]
            bank_read_slots += token_trace["bank_read_slots"] + latent_trace["bank_read_slots"]

        hidden_states = torch.cat(output_blocks, dim=1)
        logits = self.lm_head(self.final_norm(hidden_states))
        empty = hidden_states[:, :0, :]
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "refresh_states": empty,
            "bank_states": latent_states,
            "predicted_summary": empty,
            "target_summary": empty,
            "debug": {
                "segment_count": len(block_ranges),
                "bank_read_segments": len(block_ranges),
                "bank_read_slots": bank_read_slots,
                "token_bank_access_count": token_bank_access_count,
                "refresh_bank_access_count": 0,
                "latent_slots": self.config.latent_slots,
            },
        }
