"""Minimal SRD model wiring segment-local token processing and refresh-only memory."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor, nn

from srd.config import SRDConfig
from srd.modeling.local_block import LocalBlock
from srd.modeling.long_bank import LongMemoryBank
from srd.modeling.refresh_block import RefreshBlock


class SRDModel(nn.Module):
    """Implements segment-wise SRD routing without token-level bank access."""

    def __init__(self, config: SRDConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.refresh_seed = nn.Parameter(torch.randn(config.refresh_count, config.d_model) * 0.02)
        self.pre_blocks = nn.ModuleList(
            [
                LocalBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    window_size=config.local_window,
                    dropout_p=config.dropout_p,
                )
                for _ in range(config.num_local_layers_pre)
            ]
        )
        self.refresh_block = RefreshBlock(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout_p=config.dropout_p,
        )
        self.post_blocks = nn.ModuleList(
            [
                LocalBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    window_size=config.local_window,
                    dropout_p=config.dropout_p,
                )
                for _ in range(config.num_local_layers_post)
            ]
        )
        self.long_bank = LongMemoryBank(d_model=config.d_model, max_slots=config.bank_size)
        self.segment_to_refresh = nn.Linear(2 * config.d_model, config.refresh_count * config.d_model)
        self.bank_entry_proj = nn.Linear(config.d_model, config.d_model)
        self.carry_to_pre = nn.Linear(config.d_model, config.d_model)
        self.carry_to_post = nn.Linear(config.d_model, config.d_model)
        self.sufficiency_head = nn.Linear(config.d_model, config.d_model)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _segment_ranges(self, seq_len: int) -> List[tuple[int, int]]:
        """Returns the closed-open segment ranges used for scheduled refresh."""
        return [
            (start, min(start + self.config.segment_length, seq_len))
            for start in range(0, seq_len, self.config.segment_length)
        ]

    def _build_refresh_inputs(self, segment_hidden: Tensor) -> Tensor:
        """Generates one or more refresh states from segment-boundary pooled states."""
        pooled = segment_hidden.mean(dim=1)
        boundary = segment_hidden[:, -1, :]
        summary = torch.cat([pooled, boundary], dim=-1)
        refresh_inputs = self.segment_to_refresh(summary)
        refresh_inputs = refresh_inputs.view(segment_hidden.size(0), self.config.refresh_count, self.config.d_model)
        return refresh_inputs + self.refresh_seed.unsqueeze(0)

    def _segment_summary_target(self, segment_ids: Tensor) -> Tensor:
        """Returns a detached target summary for the next-segment sufficiency loss."""
        return self.embedding(segment_ids).mean(dim=1).detach()

    def forward(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> Dict[str, Tensor]:
        """Runs SRD segment processing with refresh-only bank access."""
        batch_size, seq_len = input_ids.shape
        embeddings = self.embedding(input_ids)
        bank_states = initial_bank_states if initial_bank_states is not None else self.long_bank.empty(batch_size, input_ids.device)
        carry_context = None

        output_segments = []
        refresh_segments = []
        sufficiency_predictions = []
        sufficiency_targets = []
        bank_read_segments = 0
        bank_read_slots = 0
        segment_ranges = self._segment_ranges(seq_len)

        for segment_index, (start, end) in enumerate(segment_ranges):
            hidden_states = embeddings[:, start:end, :]
            if carry_context is not None and not self.config.upper_layer_only_refresh:
                hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)

            for block in self.pre_blocks:
                hidden_states = block(hidden_states)

            if carry_context is not None:
                hidden_states = hidden_states + self.carry_to_post(carry_context).unsqueeze(1)

            for block in self.post_blocks:
                hidden_states = block(hidden_states)

            output_segments.append(hidden_states)

            if self.config.use_refresh:
                refresh_inputs = self._build_refresh_inputs(hidden_states)
                refreshed, refresh_trace = self.refresh_block(refresh_inputs, self.long_bank.read(bank_states))
                refresh_segments.append(refreshed)
                bank_read_slots += refresh_trace["bank_read_slots"]
                bank_read_segments += int(refresh_trace["bank_used"])
                carry_context = refreshed.mean(dim=1)
                bank_entry = self.bank_entry_proj(carry_context).unsqueeze(1)
                bank_states = self.long_bank.write(bank_states, bank_entry)

                if segment_index + 1 < len(segment_ranges):
                    next_start, next_end = segment_ranges[segment_index + 1]
                    next_ids = input_ids[:, next_start:next_end]
                    sufficiency_predictions.append(self.sufficiency_head(carry_context))
                    sufficiency_targets.append(self._segment_summary_target(next_ids))
            else:
                carry_context = None

        hidden_states = torch.cat(output_segments, dim=1)
        logits = self.lm_head(self.final_norm(hidden_states))
        if refresh_segments:
            refresh_states = torch.cat(refresh_segments, dim=1)
        else:
            refresh_states = hidden_states[:, :0, :]
        if sufficiency_predictions:
            predicted_summary = torch.stack(sufficiency_predictions, dim=1)
            target_summary = torch.stack(sufficiency_targets, dim=1)
        else:
            predicted_summary = hidden_states[:, :0, :]
            target_summary = hidden_states[:, :0, :]
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "refresh_states": refresh_states,
            "bank_states": bank_states,
            "predicted_summary": predicted_summary,
            "target_summary": target_summary,
            "debug": {
                "segment_count": len(segment_ranges),
                "bank_read_segments": bank_read_segments,
                "bank_read_slots": bank_read_slots,
                "token_bank_access_count": 0,
                "refresh_bank_access_count": bank_read_slots,
            },
        }
