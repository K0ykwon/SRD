"""Block-refresh SRD with a tiny bounded detail-memory retrieval path."""

from __future__ import annotations

import math
from typing import Dict, List

import torch
from torch import Tensor, nn

from srd.config import SRDConfig
from srd.modeling.block_refresh_model import BlockRefreshModel


class BlockRefreshDetailModel(BlockRefreshModel):
    """Extends block-refresh SRD with sparse retrieval over tiny per-block detail slots."""

    def __init__(self, config: SRDConfig):
        super().__init__(config)
        self.saliency_scorer = nn.Linear(config.d_model, 1)
        self.detail_query_proj = nn.Linear(config.d_model, config.d_model)
        self.detail_key_proj = nn.Linear(config.d_model, config.d_model)
        self.detail_value_proj = nn.Linear(config.d_model, config.d_model)
        self.detail_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Linear(config.d_model, 1),
        )

    def _select_detail_slots(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Selects a tiny set of anchor and salient token states from one block."""
        batch_size, block_len, d_model = hidden_states.shape
        candidate_positions: list[int] = []
        if self.config.detail_anchor_first and block_len > 0:
            candidate_positions.append(0)
        if self.config.detail_anchor_last and block_len > 0:
            candidate_positions.append(block_len - 1)

        unique_positions: list[int] = []
        for position in candidate_positions:
            if position not in unique_positions:
                unique_positions.append(position)

        remaining_capacity = max(0, self.config.detail_slots - len(unique_positions))
        saliency_count = min(
            self.config.detail_saliency_slots,
            remaining_capacity,
            max(0, block_len - len(unique_positions)),
        )

        selected_tensors = []
        selected_positions = []
        if unique_positions:
            anchor_positions = torch.tensor(unique_positions, device=hidden_states.device, dtype=torch.long)
            anchor_index = anchor_positions.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, d_model)
            selected_tensors.append(hidden_states.gather(1, anchor_index))
            selected_positions.append(anchor_positions.unsqueeze(0).expand(batch_size, -1))

        if saliency_count > 0:
            saliency_scores = self.saliency_scorer(hidden_states).squeeze(-1)
            if unique_positions:
                saliency_scores[:, unique_positions] = float("-inf")
            top_positions = saliency_scores.topk(k=saliency_count, dim=1).indices
            top_index = top_positions.unsqueeze(-1).expand(batch_size, -1, d_model)
            selected_tensors.append(hidden_states.gather(1, top_index))
            selected_positions.append(top_positions)

        if not selected_tensors:
            empty_states = hidden_states[:, :0, :]
            empty_positions = torch.empty(batch_size, 0, dtype=torch.long, device=hidden_states.device)
            return empty_states, empty_positions

        slots = torch.cat(selected_tensors, dim=1)
        positions = torch.cat(selected_positions, dim=1)
        if slots.size(1) > self.config.detail_slots:
            slots = slots[:, : self.config.detail_slots, :]
            positions = positions[:, : self.config.detail_slots]
        return slots, positions

    def _retrieve_detail_context(
        self,
        block_hidden: Tensor,
        past_detail_keys: Tensor,
        past_detail_values: Tensor,
    ) -> tuple[Tensor, dict]:
        """Retrieves a bounded top-k detail context from prior blocks."""
        batch_size = block_hidden.size(0)
        pooled = block_hidden.mean(dim=1)
        if past_detail_keys.size(1) == 0:
            return pooled.new_zeros(batch_size, self.config.d_model), {
                "detail_used": False,
                "detail_candidate_count": 0,
                "detail_topk_used": 0,
            }

        query = self.detail_query_proj(pooled).unsqueeze(1)
        scores = torch.matmul(query, past_detail_keys.transpose(-2, -1)).squeeze(1) / math.sqrt(self.config.d_model)
        topk = min(self.config.detail_topk, past_detail_keys.size(1))
        top_scores, top_indices = scores.topk(k=topk, dim=1)
        weights = torch.softmax(top_scores, dim=-1)
        gather_index = top_indices.unsqueeze(-1).expand(batch_size, topk, self.config.d_model)
        selected_values = past_detail_values.gather(1, gather_index)
        detail_context = torch.sum(weights.unsqueeze(-1) * selected_values, dim=1)
        return detail_context, {
            "detail_used": True,
            "detail_candidate_count": int(past_detail_keys.size(1)),
            "detail_topk_used": int(topk),
        }

    def _fuse_long_context(self, refresh_context: Tensor | None, detail_context: Tensor, block_hidden: Tensor) -> tuple[Tensor | None, Tensor]:
        """Fuses refresh carry and detail retrieval with a simple scalar gate."""
        pooled = block_hidden.mean(dim=1)
        if refresh_context is None and detail_context.numel() == 0:
            return None, pooled.new_zeros(pooled.size(0), 1)
        if refresh_context is None:
            return detail_context, pooled.new_zeros(pooled.size(0), 1)
        if detail_context.numel() == 0:
            return refresh_context, pooled.new_ones(pooled.size(0), 1)
        if self.config.detail_gate_enabled:
            gate = torch.sigmoid(self.detail_gate(pooled))
            fused = gate * refresh_context + (1.0 - gate) * detail_context
            return fused, gate
        fused = 0.5 * (refresh_context + detail_context)
        return fused, pooled.new_full((pooled.size(0), 1), 0.5)

    def forward(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> Dict[str, Tensor]:
        """Runs block-refresh SRD plus sparse retrieval over tiny detail memories."""
        batch_size, seq_len = input_ids.shape
        embeddings = self.embedding(input_ids)
        bank_states = initial_bank_states if initial_bank_states is not None else self.long_bank.empty(batch_size, input_ids.device)
        carry_context = None
        past_detail_keys = embeddings[:, :0, :]
        past_detail_values = embeddings[:, :0, :]

        output_blocks = []
        refresh_blocks = []
        detail_blocks = []
        sufficiency_predictions = []
        sufficiency_targets = []
        bank_read_blocks = 0
        bank_read_slots = 0
        refresh_norms = []
        detail_selected_counts = []
        detail_topk_used = []
        detail_candidate_counts = []
        detail_gate_means = []
        block_ranges = self._block_ranges(seq_len)

        for block_index, (start, end) in enumerate(block_ranges):
            hidden_states = embeddings[:, start:end, :]
            if carry_context is not None and not self.config.upper_layer_only_refresh:
                hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)

            for block in self.pre_blocks:
                hidden_states = block(hidden_states)

            detail_context = hidden_states.new_zeros(batch_size, self.config.d_model)
            detail_trace = {
                "detail_used": False,
                "detail_candidate_count": 0,
                "detail_topk_used": 0,
            }
            if self.config.detail_enabled:
                detail_context, detail_trace = self._retrieve_detail_context(hidden_states, past_detail_keys, past_detail_values)

            fused_context, gate = self._fuse_long_context(carry_context, detail_context, hidden_states)
            if fused_context is not None:
                hidden_states = hidden_states + self.carry_to_post(fused_context).unsqueeze(1)

            for block in self.post_blocks:
                hidden_states = block(hidden_states)

            output_blocks.append(hidden_states)
            detail_gate_means.append(float(gate.mean().detach().cpu()))
            detail_candidate_counts.append(detail_trace["detail_candidate_count"])
            detail_topk_used.append(detail_trace["detail_topk_used"])

            if self.config.refresh_enabled:
                refresh_inputs = self._build_refresh_inputs(hidden_states)
                refreshed, refresh_trace = self.refresh_block(refresh_inputs, self.long_bank.read(bank_states))
                refresh_blocks.append(refreshed)
                bank_read_slots += refresh_trace["bank_read_slots"]
                bank_read_blocks += int(refresh_trace["bank_used"])
                carry_context = refreshed.mean(dim=1)
                refresh_norms.append(carry_context.norm(dim=-1).mean())
                bank_entry = self.bank_entry_proj(carry_context).unsqueeze(1)
                bank_states = self.long_bank.write(bank_states, bank_entry)

                if block_index + 1 < len(block_ranges):
                    next_start, next_end = block_ranges[block_index + 1]
                    next_ids = input_ids[:, next_start:next_end]
                    sufficiency_predictions.append(self.sufficiency_head(carry_context))
                    sufficiency_targets.append(self._next_block_target(next_ids))
            else:
                carry_context = None

            if self.config.detail_enabled:
                detail_slots, detail_positions = self._select_detail_slots(hidden_states)
                detail_blocks.append(detail_slots)
                detail_selected_counts.append(int(detail_slots.size(1)))
                if detail_slots.size(1) > 0:
                    detail_keys = self.detail_key_proj(detail_slots)
                    detail_values = self.detail_value_proj(detail_slots)
                    past_detail_keys = torch.cat([past_detail_keys, detail_keys], dim=1)
                    past_detail_values = torch.cat([past_detail_values, detail_values], dim=1)
            else:
                detail_selected_counts.append(0)

        hidden_states = torch.cat(output_blocks, dim=1)
        logits = self.lm_head(self.final_norm(hidden_states))
        refresh_states = torch.cat(refresh_blocks, dim=1) if refresh_blocks else hidden_states[:, :0, :]
        detail_states = torch.cat(detail_blocks, dim=1) if detail_blocks else hidden_states[:, :0, :]
        if sufficiency_predictions:
            predicted_summary = torch.stack(sufficiency_predictions, dim=1)
            target_summary = torch.stack(sufficiency_targets, dim=1)
        else:
            predicted_summary = hidden_states[:, :0, :]
            target_summary = hidden_states[:, :0, :]

        refresh_norm_mean = 0.0
        if refresh_norms:
            refresh_norm_mean = float(torch.stack(refresh_norms).mean().detach().cpu())

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "refresh_states": refresh_states,
            "detail_states": detail_states,
            "bank_states": bank_states,
            "predicted_summary": predicted_summary,
            "target_summary": target_summary,
            "debug": {
                "block_count": len(block_ranges),
                "segment_count": len(block_ranges),
                "block_size": self.config.effective_block_size(),
                "refresh_slots": self.config.effective_refresh_slots(),
                "refresh_enabled": self.config.refresh_enabled,
                "detail_enabled": self.config.detail_enabled,
                "detail_slots": self.config.detail_slots,
                "detail_topk": self.config.detail_topk,
                "detail_state_shape": tuple(detail_states.shape),
                "refresh_state_shape": tuple(refresh_states.shape),
                "bank_read_blocks": bank_read_blocks,
                "bank_read_segments": bank_read_blocks,
                "bank_read_slots": bank_read_slots,
                "token_bank_access_count": 0,
                "refresh_bank_access_count": bank_read_slots,
                "detail_access_count": sum(detail_topk_used),
                "detail_candidate_count_mean": float(sum(detail_candidate_counts) / max(len(detail_candidate_counts), 1)),
                "detail_selected_count_mean": float(sum(detail_selected_counts) / max(len(detail_selected_counts), 1)),
                "detail_topk_used_mean": float(sum(detail_topk_used) / max(len(detail_topk_used), 1)),
                "detail_gate_mean": float(sum(detail_gate_means) / max(len(detail_gate_means), 1)),
                "refresh_norm_mean": refresh_norm_mean,
            },
        }
