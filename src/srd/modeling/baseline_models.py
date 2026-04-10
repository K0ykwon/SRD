"""Conventional decoder-style baselines for external SRD comparison."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor, nn

from srd.config import SRDConfig
from srd.modeling.full_block import FullBlock
from srd.modeling.local_block import LocalBlock
from srd.modeling.long_bank import LongMemoryBank
from srd.modeling.token_memory_block import TokenMemoryBlock


class TransformerLocalModel(nn.Module):
    """Standard local-window causal Transformer without any long-memory path."""

    def __init__(self, config: SRDConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                LocalBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    window_size=config.local_window,
                    dropout_p=config.dropout_p,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> Dict[str, Tensor]:
        """Runs a standard local-only decoder baseline."""
        hidden_states = self.embedding(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)

        logits = self.lm_head(self.final_norm(hidden_states))
        empty = hidden_states[:, :0, :]
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "refresh_states": empty,
            "bank_states": empty,
            "predicted_summary": empty,
            "target_summary": empty,
            "debug": {
                "segment_count": len(range(0, input_ids.size(1), self.config.segment_length)),
                "bank_read_segments": 0,
                "bank_read_slots": 0,
                "token_bank_access_count": 0,
                "refresh_bank_access_count": 0,
            },
        }


class TransformerFullModel(nn.Module):
    """Standard full-attention causal Transformer baseline."""

    def __init__(self, config: SRDConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                FullBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    dropout_p=config.dropout_p,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> Dict[str, Tensor]:
        """Runs a standard full causal Transformer baseline."""
        hidden_states = self.embedding(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)

        logits = self.lm_head(self.final_norm(hidden_states))
        empty = hidden_states[:, :0, :]
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "refresh_states": empty,
            "bank_states": empty,
            "predicted_summary": empty,
            "target_summary": empty,
            "debug": {
                "segment_count": len(range(0, input_ids.size(1), self.config.segment_length)),
                "bank_read_segments": 0,
                "bank_read_slots": 0,
                "token_bank_access_count": 0,
                "refresh_bank_access_count": 0,
            },
        }


class SummaryMemoryModel(nn.Module):
    """Baseline with direct token-path access to a shared summary bank."""

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
        self.long_bank = LongMemoryBank(d_model=config.d_model, max_slots=config.bank_size)
        self.segment_to_summary = nn.Linear(2 * config.d_model, config.d_model)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _segment_ranges(self, seq_len: int) -> List[tuple[int, int]]:
        """Returns segment boundaries used for bank writes and reads."""
        return [
            (start, min(start + self.config.segment_length, seq_len))
            for start in range(0, seq_len, self.config.segment_length)
        ]

    def forward(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> Dict[str, Tensor]:
        """Runs a summary-memory baseline where token states can read the bank directly."""
        batch_size, seq_len = input_ids.shape
        embeddings = self.embedding(input_ids)
        bank_states = initial_bank_states if initial_bank_states is not None else self.long_bank.empty(batch_size, input_ids.device)

        output_segments = []
        bank_read_segments = 0
        bank_read_slots = 0
        token_bank_access_count = 0
        segment_ranges = self._segment_ranges(seq_len)

        for start, end in segment_ranges:
            hidden_states = embeddings[:, start:end, :]
            for block in self.local_blocks:
                hidden_states = block(hidden_states)
            hidden_states, memory_trace = self.memory_block(hidden_states, self.long_bank.read(bank_states))
            bank_read_segments += int(memory_trace["bank_used"])
            bank_read_slots += memory_trace["bank_read_slots"]
            token_bank_access_count += memory_trace["token_bank_access_count"]
            output_segments.append(hidden_states)

            pooled = hidden_states.mean(dim=1)
            boundary = hidden_states[:, -1, :]
            bank_entry = self.segment_to_summary(torch.cat([pooled, boundary], dim=-1)).unsqueeze(1)
            bank_states = self.long_bank.write(bank_states, bank_entry)

        hidden_states = torch.cat(output_segments, dim=1)
        logits = self.lm_head(self.final_norm(hidden_states))
        empty = hidden_states[:, :0, :]
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "refresh_states": empty,
            "bank_states": bank_states,
            "predicted_summary": empty,
            "target_summary": empty,
            "debug": {
                "segment_count": len(segment_ranges),
                "bank_read_segments": bank_read_segments,
                "bank_read_slots": bank_read_slots,
                "token_bank_access_count": token_bank_access_count,
                "refresh_bank_access_count": 0,
            },
        }
