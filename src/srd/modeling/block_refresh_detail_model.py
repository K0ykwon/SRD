"""Block-refresh SRD with a tiny bounded detail-memory retrieval path."""

from __future__ import annotations

import math
from typing import Any, Dict, List

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
        if config.detail_scan_carry_mode == "affine":
            self.scan_carry_gate = nn.Sequential(
                nn.Linear(2 * config.d_model, config.d_model),
                nn.Tanh(),
                nn.Linear(config.d_model, config.d_model),
            )
        else:
            self.scan_carry_gate = None

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

    def _detail_grouping_enabled(self, candidate_count: int) -> bool:
        """Returns whether grouped coarse-to-fine detail routing is active for this block."""
        return (
            self.config.detail_coarse_group_size > 0
            and self.config.detail_coarse_topk_groups > 0
            and candidate_count > self.config.detail_coarse_group_size
        )

    def _select_detail_slots_parallel(self, hidden_blocks: Tensor) -> tuple[Tensor, Tensor]:
        """Selects per-block detail slots in one flattened pass over the block axis."""
        batch_size, block_count, block_size, d_model = hidden_blocks.shape
        if not self.config.detail_enabled:
            empty_slots = hidden_blocks.new_zeros(batch_size, block_count, 0, d_model)
            empty_mask = torch.zeros(batch_size, block_count, 0, 1, device=hidden_blocks.device, dtype=torch.bool)
            return empty_slots, empty_mask
        flat_hidden = hidden_blocks.view(batch_size * block_count, block_size, d_model)
        flat_slots, _ = self._select_detail_slots(flat_hidden)
        slot_count = flat_slots.size(1)
        if slot_count == 0:
            empty_slots = hidden_blocks.new_zeros(batch_size, block_count, 0, d_model)
            empty_mask = torch.zeros(batch_size, block_count, 0, 1, device=hidden_blocks.device, dtype=torch.bool)
            return empty_slots, empty_mask
        slot_mask = torch.ones(
            batch_size,
            block_count,
            slot_count,
            1,
            device=hidden_blocks.device,
            dtype=torch.bool,
        )
        return flat_slots.view(batch_size, block_count, slot_count, d_model), slot_mask

    def _refresh_due_mask(self, block_count: int, device: torch.device) -> Tensor:
        """Returns the fixed block schedule where refresh writes are emitted."""
        block_ids = torch.arange(1, block_count + 1, device=device)
        return self.config.refresh_enabled & (block_ids % self.config.refresh_interval_blocks == 0)

    def _scan_carry_traces_from_refresh_writes(
        self,
        refresh_write_states: Tensor,
        refresh_write_mask: Tensor,
        initial_carry: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Reconstructs both prefix and updated carry traces from explicit refresh writes."""
        batch_size, block_count, _ = refresh_write_states.shape
        prefix_carry_states = torch.zeros(batch_size, block_count, self.config.d_model, device=refresh_write_states.device)
        prefix_carry_mask = torch.zeros(batch_size, block_count, 1, device=refresh_write_states.device, dtype=torch.bool)
        updated_carry_states = torch.zeros(batch_size, block_count, self.config.d_model, device=refresh_write_states.device)
        updated_carry_mask = torch.zeros(batch_size, block_count, 1, device=refresh_write_states.device, dtype=torch.bool)
        carry_context = initial_carry
        for block_index in range(block_count):
            if carry_context is not None:
                prefix_carry_states[:, block_index, :] = carry_context
                prefix_carry_mask[:, block_index, :] = True
            if bool(refresh_write_mask[:, block_index, :].any().item()):
                refresh_carry = refresh_write_states[:, block_index, :]
                carry_context = self._compose_scan_carry_state(carry_context, refresh_carry)
            if carry_context is not None:
                updated_carry_states[:, block_index, :] = carry_context
                updated_carry_mask[:, block_index, :] = True
        return prefix_carry_states, prefix_carry_mask, updated_carry_states, updated_carry_mask

    def _build_detail_group_summaries(
        self,
        detail_keys: Tensor,
        detail_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Pools fixed-size detail groups into summary keys/values for coarse routing."""
        group_size = self.config.detail_coarse_group_size
        batch_size, candidate_count, d_model = detail_keys.shape
        group_count = math.ceil(candidate_count / group_size)
        padded_count = group_count * group_size
        if padded_count != candidate_count:
            pad = padded_count - candidate_count
            detail_keys = torch.cat([detail_keys, detail_keys.new_zeros(batch_size, pad, d_model)], dim=1)
            detail_values = torch.cat([detail_values, detail_values.new_zeros(batch_size, pad, d_model)], dim=1)
        grouped_keys = detail_keys.view(batch_size, group_count, group_size, d_model)
        grouped_values = detail_values.view(batch_size, group_count, group_size, d_model)
        valid = (
            torch.arange(padded_count, device=detail_keys.device)
            .view(1, group_count, group_size, 1)
            < candidate_count
        )
        counts = valid.sum(dim=2).clamp_min(1).to(grouped_keys.dtype)
        summary_keys = (grouped_keys * valid).sum(dim=2) / counts
        summary_values = (grouped_values * valid).sum(dim=2) / counts
        return summary_keys, summary_values

    def _select_detail_groups(
        self,
        query: Tensor,
        summary_keys: Tensor,
    ) -> tuple[Tensor, dict[str, int]]:
        """Selects a small set of summary groups before fine detail retrieval."""
        scores = torch.matmul(query, summary_keys.transpose(-2, -1)).squeeze(1) / math.sqrt(self.config.d_model)
        group_topk = min(self.config.detail_coarse_topk_groups, summary_keys.size(1))
        group_indices = scores.topk(k=group_topk, dim=1).indices
        return group_indices, {
            "detail_group_count": int(summary_keys.size(1)),
            "detail_group_topk_used": int(group_topk),
        }

    def _gather_grouped_detail_candidates(
        self,
        detail_keys: Tensor,
        detail_values: Tensor,
        group_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Materializes only the fine candidates inside selected coarse detail groups."""
        batch_size, candidate_count, d_model = detail_keys.shape
        group_size = self.config.detail_coarse_group_size
        offsets = torch.arange(group_size, device=detail_keys.device).view(1, 1, group_size)
        candidate_indices = group_indices.unsqueeze(-1) * group_size + offsets
        valid = candidate_indices < candidate_count
        safe_indices = candidate_indices.clamp_max(max(candidate_count - 1, 0))
        gather_index = safe_indices.view(batch_size, -1).unsqueeze(-1).expand(batch_size, -1, d_model)
        grouped_keys = detail_keys.gather(1, gather_index)
        grouped_values = detail_values.gather(1, gather_index)
        return grouped_keys, grouped_values, valid.view(batch_size, -1)

    def _retrieve_detail_context(
        self,
        pooled_hidden: Tensor,
        past_detail_keys: Tensor,
        past_detail_values: Tensor,
    ) -> tuple[Tensor | None, dict]:
        """Retrieves a bounded top-k detail context from prior blocks."""
        batch_size = pooled_hidden.size(0)
        if past_detail_keys.size(1) == 0 or self.config.detail_topk <= 0:
            return None, {
                "detail_used": False,
                "detail_candidate_count": 0,
                "detail_fine_candidate_count": 0,
                "detail_group_count": 0,
                "detail_group_topk_used": 0,
                "detail_topk_used": 0,
            }

        query = self.detail_query_proj(pooled_hidden).unsqueeze(1)
        detail_keys = past_detail_keys
        detail_values = past_detail_values
        detail_candidate_mask = None
        detail_group_count = 0
        detail_group_topk_used = 0
        fine_candidate_count = int(past_detail_keys.size(1))
        if self._detail_grouping_enabled(past_detail_keys.size(1)):
            summary_keys, _ = self._build_detail_group_summaries(past_detail_keys, past_detail_values)
            group_indices, group_trace = self._select_detail_groups(query, summary_keys)
            detail_group_count = group_trace["detail_group_count"]
            detail_group_topk_used = group_trace["detail_group_topk_used"]
            detail_keys, detail_values, detail_candidate_mask = self._gather_grouped_detail_candidates(
                past_detail_keys,
                past_detail_values,
                group_indices,
            )
            fine_candidate_count = int(detail_candidate_mask.sum(dim=1).max().item())

        scores = torch.matmul(query, detail_keys.transpose(-2, -1)).squeeze(1) / math.sqrt(self.config.d_model)
        if detail_candidate_mask is not None:
            scores = scores.masked_fill(~detail_candidate_mask, float("-inf"))
        topk = min(self.config.detail_topk, max(fine_candidate_count, 1))
        top_scores, top_indices = scores.topk(k=topk, dim=1)
        weights = torch.softmax(top_scores, dim=-1)
        gather_index = top_indices.unsqueeze(-1).expand(batch_size, topk, self.config.d_model)
        selected_values = detail_values.gather(1, gather_index)
        detail_context = torch.sum(weights.unsqueeze(-1) * selected_values, dim=1)
        return detail_context, {
            "detail_used": True,
            "detail_candidate_count": int(past_detail_keys.size(1)),
            "detail_fine_candidate_count": fine_candidate_count,
            "detail_group_count": detail_group_count,
            "detail_group_topk_used": detail_group_topk_used,
            "detail_topk_used": int(topk),
        }

    def _fuse_long_context(
        self,
        refresh_context: Tensor | None,
        detail_context: Tensor | None,
        pooled_hidden: Tensor,
    ) -> tuple[Tensor | None, Tensor]:
        """Fuses refresh carry and detail retrieval with a simple scalar gate."""
        if refresh_context is None and detail_context is None:
            return None, pooled_hidden.new_zeros(pooled_hidden.size(0), 1)
        if refresh_context is None:
            if detail_context is None:
                return None, pooled_hidden.new_zeros(pooled_hidden.size(0), 1)
            return detail_context, pooled_hidden.new_zeros(pooled_hidden.size(0), 1)
        if detail_context is None:
            return refresh_context, pooled_hidden.new_ones(pooled_hidden.size(0), 1)
        if self.config.detail_gate_enabled:
            gate = torch.sigmoid(self.detail_gate(pooled_hidden))
            fused = gate * refresh_context + (1.0 - gate) * detail_context
            return fused, gate
        fused = 0.5 * (refresh_context + detail_context)
        return fused, pooled_hidden.new_full((pooled_hidden.size(0), 1), 0.5)

    def _process_token_block_with_detail(
        self,
        block_ids: Tensor,
        carry_context: Tensor | None,
        detail_key_blocks: Tensor,
        detail_value_blocks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Runs one open block using cached detail memories from completed blocks only."""
        batch_size = block_ids.size(0)
        hidden_states = self.embedding(block_ids)
        for block in self.pre_blocks:
            hidden_states = block(hidden_states)
        if carry_context is not None and not self.config.upper_layer_only_refresh:
            hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)

        pooled_hidden = hidden_states.mean(dim=1)
        detail_context = None
        if self.config.detail_enabled and detail_key_blocks.size(1) > 0:
            detail_context, _ = self._retrieve_detail_context(
                pooled_hidden,
                detail_key_blocks,
                detail_value_blocks,
            )

        fused_context, _ = self._fuse_long_context(carry_context, detail_context, pooled_hidden)
        hidden_states = self._apply_detail_post_stack(hidden_states, fused_context)
        if detail_context is None:
            detail_context = hidden_states.new_zeros(batch_size, self.config.d_model)
        return hidden_states, detail_context

    def _encode_blocks_parallel(self, block_embeddings: Tensor) -> Tensor:
        """Runs the block-local pre stack in one block-parallel stage."""
        return self._apply_local_stack_parallel(block_embeddings, self.pre_blocks)

    def _slice_past_detail_kv(
        self,
        detail_key_blocks: Tensor,
        detail_value_blocks: Tensor,
        detail_write_index: int,
    ) -> tuple[Tensor, Tensor]:
        """Returns only the completed detail slots visible to the current block."""
        return (
            detail_key_blocks[:, :detail_write_index, :],
            detail_value_blocks[:, :detail_write_index, :],
        )

    def _scan_block_state(
        self,
        pre_hidden_states: Tensor,
        carry_context: Tensor | None,
        detail_key_blocks: Tensor,
        detail_value_blocks: Tensor,
        detail_write_index: int,
        pooled_hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor, dict]:
        """Computes the current block's compact long-range state under sequential semantics."""
        hidden_states, pooled_hidden = self._materialize_coarse_block_state(
            pre_hidden_states,
            carry_context,
            pooled_hidden,
        )
        past_detail_keys, past_detail_values = self._slice_past_detail_kv(
            detail_key_blocks,
            detail_value_blocks,
            detail_write_index,
        )
        detail_context, detail_trace = self._retrieve_detail_context(
            pooled_hidden,
            past_detail_keys,
            past_detail_values,
        )
        fused_context, gate = self._fuse_long_context(carry_context, detail_context, pooled_hidden)
        return hidden_states, pooled_hidden, detail_context, fused_context, gate, detail_trace

    def _materialize_coarse_block_state(
        self,
        pre_hidden_states: Tensor,
        carry_context: Tensor | None,
        pooled_hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Builds the coarse block state before any detail refinement."""
        hidden_states = pre_hidden_states
        if carry_context is not None and not self.config.upper_layer_only_refresh:
            hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)
        if pooled_hidden is None:
            pooled_hidden = hidden_states.mean(dim=1)
        return hidden_states, pooled_hidden

    def _refine_block_with_detail(
        self,
        pooled_hidden: Tensor,
        detail_key_blocks: Tensor,
        detail_value_blocks: Tensor,
        detail_write_index: int,
    ) -> tuple[Tensor | None, dict]:
        """Retrieves bounded detail context as a refinement on top of the coarse block state."""
        past_detail_keys, past_detail_values = self._slice_past_detail_kv(
            detail_key_blocks,
            detail_value_blocks,
            detail_write_index,
        )
        return self._retrieve_detail_context(
            pooled_hidden,
            past_detail_keys,
            past_detail_values,
        )

    def _compose_detail_refined_context(
        self,
        carry_context: Tensor | None,
        detail_context: Tensor | None,
        pooled_hidden: Tensor,
    ) -> tuple[Tensor | None, Tensor]:
        """Combines the coarse carry path with optional detail refinement."""
        return self._fuse_long_context(carry_context, detail_context, pooled_hidden)

    def _apply_conditioned_post_blocks(
        self,
        hidden_states: Tensor,
        fused_context: Tensor | None,
    ) -> Tensor:
        """Runs the conditioned post stack as an explicit final stage."""
        return self._apply_detail_post_stack(hidden_states, fused_context)

    def _compose_scan_carry_state(
        self,
        previous_carry: Tensor | None,
        refresh_carry: Tensor,
    ) -> Tensor:
        """Builds the bounded inter-block carry state from the previous state and refresh write."""
        if self.config.detail_scan_carry_mode == "legacy" or previous_carry is None:
            return refresh_carry
        if self.scan_carry_gate is None:
            raise RuntimeError("affine detail scan carry mode requires scan_carry_gate")
        gate_input = torch.cat([previous_carry, refresh_carry], dim=-1)
        mix = torch.sigmoid(self.scan_carry_gate(gate_input))
        return mix * previous_carry + (1.0 - mix) * refresh_carry

    def _scan_carry_sequence_from_refresh_writes(
        self,
        refresh_write_states: Tensor,
        refresh_write_mask: Tensor,
        initial_carry: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Reconstructs prefix carry states from explicit per-block refresh writes."""
        prefix_carry_states, prefix_carry_mask, _, _ = self._scan_carry_traces_from_refresh_writes(
            refresh_write_states,
            refresh_write_mask,
            initial_carry=initial_carry,
        )
        return prefix_carry_states, prefix_carry_mask

    def _build_parallel_refresh_traces(
        self,
        pre_hidden_blocks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Builds refresh proposals for every due block in one block-parallel pass."""
        batch_size, block_count, block_size, d_model = pre_hidden_blocks.shape
        refresh_write_states = torch.zeros(batch_size, block_count, d_model, device=pre_hidden_blocks.device)
        refresh_write_mask = torch.zeros(batch_size, block_count, 1, device=pre_hidden_blocks.device, dtype=torch.bool)
        due_mask = self._refresh_due_mask(block_count, pre_hidden_blocks.device)
        due_indices = due_mask.nonzero(as_tuple=False).squeeze(-1)
        if due_indices.numel() == 0:
            empty_refresh = pre_hidden_blocks.new_zeros(batch_size, 0, d_model)
            return refresh_write_states, refresh_write_mask, empty_refresh
        due_hidden = pre_hidden_blocks[:, due_indices, :, :].contiguous().view(batch_size * due_indices.numel(), block_size, d_model)
        refresh_inputs = self._build_refresh_inputs(due_hidden)
        refresh_carry = self._write_bank_entry(refresh_inputs)
        if self.config.refresh_write_gate_enabled:
            if self.bank_write_importance is None:
                raise RuntimeError("refresh write gating requires bank_write_importance module")
            gate = torch.sigmoid(self.bank_write_importance(refresh_inputs).mean(dim=1))
            refresh_carry = gate * refresh_carry
        refresh_write_states[:, due_indices, :] = refresh_carry.view(batch_size, due_indices.numel(), d_model)
        refresh_write_mask[:, due_indices, :] = True
        refresh_blocks = refresh_inputs.view(
            batch_size,
            due_indices.numel() * self.config.effective_refresh_slots(),
            d_model,
        )
        return refresh_write_states, refresh_write_mask, refresh_blocks

    def _retrieve_detail_context_parallel(
        self,
        pooled_hidden: Tensor,
        detail_states: Tensor,
        detail_keys: Tensor,
        detail_values: Tensor,
        detail_slot_mask: Tensor,
    ) -> tuple[Tensor, dict[str, list[int] | float]]:
        """Retrieves full-history detail in one vectorized pass across the block axis."""
        batch_size, block_count, _, d_model = detail_states.shape
        total_slots = detail_states.size(1) * detail_states.size(2)
        if total_slots == 0 or self.config.detail_topk <= 0:
            empty_context = pooled_hidden.new_zeros(batch_size, block_count, d_model)
            zero_stats = [0] * block_count
            return empty_context, {
                "detail_candidate_counts": zero_stats,
                "detail_fine_candidate_counts": zero_stats,
                "detail_group_counts": zero_stats,
                "detail_group_topk_used": zero_stats,
                "detail_topk_used": zero_stats,
            }
        if self.config.detail_coarse_group_size > 0 or self.config.detail_coarse_topk_groups > 0:
            raise ValueError("parallel_scan detail forward mode does not support grouped coarse retrieval")

        flat_keys = detail_keys.view(batch_size, total_slots, d_model)
        flat_values = detail_values.view(batch_size, total_slots, d_model)
        flat_slot_mask = detail_slot_mask.view(batch_size, total_slots)
        slot_block_ids = (
            torch.arange(block_count, device=pooled_hidden.device)
            .view(block_count, 1)
            .expand(block_count, detail_states.size(2))
            .reshape(total_slots)
        )
        block_ids = torch.arange(block_count, device=pooled_hidden.device)
        prefix_mask = slot_block_ids.view(1, 1, total_slots) < block_ids.view(1, block_count, 1)
        valid_mask = flat_slot_mask.unsqueeze(1) & prefix_mask

        query = self.detail_query_proj(pooled_hidden)
        scores = torch.einsum("btd,bsd->bts", query, flat_keys) / math.sqrt(d_model)
        scores = scores.masked_fill(~valid_mask, float("-inf"))

        topk = min(self.config.detail_topk, total_slots)
        top_scores, top_indices = scores.topk(k=max(topk, 1), dim=-1)
        gather_index = top_indices.unsqueeze(-1).expand(batch_size, block_count, max(topk, 1), d_model)
        expanded_values = flat_values.unsqueeze(1).expand(batch_size, block_count, total_slots, d_model)
        selected_values = expanded_values.gather(2, gather_index)
        valid_top = top_scores.isfinite()
        safe_scores = top_scores.masked_fill(~valid_top, 0.0)
        weights = torch.softmax(safe_scores, dim=-1) * valid_top.to(safe_scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        detail_context = torch.sum(weights.unsqueeze(-1) * selected_values, dim=2)
        has_any = valid_mask.any(dim=-1, keepdim=True)
        detail_context = detail_context * has_any.to(detail_context.dtype)

        candidate_counts = valid_mask.sum(dim=-1)
        topk_used = candidate_counts.clamp_max(self.config.detail_topk)
        candidate_count_mean = candidate_counts.to(torch.float32).mean(dim=0).tolist()
        topk_used_mean = topk_used.to(torch.float32).mean(dim=0).tolist()
        zero_stats = [0] * block_count
        return detail_context, {
            "detail_candidate_counts": [int(round(value)) for value in candidate_count_mean],
            "detail_fine_candidate_counts": [int(round(value)) for value in candidate_count_mean],
            "detail_group_counts": zero_stats,
            "detail_group_topk_used": zero_stats,
            "detail_topk_used": [int(round(value)) for value in topk_used_mean],
        }

    def _forward_parallel_scan(
        self,
        input_ids: Tensor,
        initial_bank_states: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        """Runs an experimental block-parallel forward path over the detail model."""
        if not self.config.upper_layer_only_refresh:
            raise ValueError("parallel_scan detail forward mode requires upper_layer_only_refresh=True")
        batch_size, seq_len = input_ids.shape
        embeddings = self.embedding(input_ids)
        block_embeddings, block_count, _ = self._reshape_blocks(embeddings)
        pre_hidden_blocks = self._encode_blocks_parallel(block_embeddings)
        refresh_write_states, refresh_write_mask, refresh_blocks = self._build_parallel_refresh_traces(pre_hidden_blocks)
        bank_states = initial_bank_states if initial_bank_states is not None else self.long_bank.empty(batch_size, input_ids.device)
        initial_carry = bank_states[:, -1, :] if bank_states.size(1) > 0 else None
        prefix_carry_states, prefix_carry_mask, updated_carry_states, _ = self._scan_carry_traces_from_refresh_writes(
            refresh_write_states,
            refresh_write_mask,
            initial_carry=initial_carry,
        )

        pooled_hidden = pre_hidden_blocks.mean(dim=2)
        detail_states, detail_slot_mask = self._select_detail_slots_parallel(pre_hidden_blocks)
        detail_keys = self.detail_key_proj(detail_states)
        detail_values = self.detail_value_proj(detail_states)
        detail_context, detail_trace = self._retrieve_detail_context_parallel(
            pooled_hidden,
            detail_states,
            detail_keys,
            detail_values,
            detail_slot_mask,
        )

        flat_pooled = pooled_hidden.view(batch_size * block_count, self.config.d_model)
        flat_carry = prefix_carry_states.view(batch_size * block_count, self.config.d_model)
        flat_detail = detail_context.view(batch_size * block_count, self.config.d_model)
        flat_fused, flat_gate = self._fuse_long_context(flat_carry, flat_detail, flat_pooled)
        fused_context = flat_fused.view(batch_size, block_count, self.config.d_model)
        gate = flat_gate.view(batch_size, block_count, 1)

        post_inputs = pre_hidden_blocks + self.carry_to_post(fused_context).unsqueeze(2)
        output_blocks = self._apply_local_stack_parallel(post_inputs, self.post_blocks)
        hidden_states = output_blocks.view(batch_size, seq_len, self.config.d_model)
        logits = self.lm_head(self.final_norm(hidden_states))

        due_indices = self._refresh_due_mask(block_count, input_ids.device).nonzero(as_tuple=False).squeeze(-1)
        for block_index in due_indices.tolist():
            carry_after_write = updated_carry_states[:, block_index, :]
            bank_entry = self.bank_entry_proj(carry_after_write).unsqueeze(1)
            bank_states = self.long_bank.write(bank_states, bank_entry)

        precomputed_targets = self._precompute_next_block_targets(input_ids)
        valid_pred_indices = due_indices[due_indices + 1 < block_count]
        empty = self._empty_state(batch_size, input_ids.device)
        if valid_pred_indices.numel() > 0:
            predicted_summary = self.sufficiency_head(updated_carry_states[:, valid_pred_indices, :])
            target_summary = precomputed_targets[:, valid_pred_indices + 1, :]
        else:
            predicted_summary = empty
            target_summary = empty

        detail_states_flat = detail_states.view(batch_size, -1, self.config.d_model).detach()
        refresh_states = refresh_blocks.detach() if refresh_blocks.size(1) > 0 else empty
        detail_states_out = detail_states_flat if detail_states_flat.size(1) > 0 else empty
        refresh_norm_mask = refresh_write_mask.expand_as(updated_carry_states)
        refresh_norm_mean = 0.0
        if refresh_norm_mask.any():
            refresh_norm_mean = float(updated_carry_states[refresh_norm_mask].view(-1, self.config.d_model).norm(dim=-1).mean().cpu().item())

        return {
            "logits": logits,
            "hidden_states": empty,
            "refresh_states": refresh_states,
            "detail_states": detail_states_out,
            "bank_states": bank_states.detach(),
            "predicted_summary": predicted_summary,
            "target_summary": target_summary,
            "debug": {
                "block_count": block_count,
                "segment_count": block_count,
                "block_size": self.config.effective_block_size(),
                "refresh_slots": self.config.effective_refresh_slots(),
                "refresh_enabled": self.config.refresh_enabled,
                "refresh_role_scheme": self.config.refresh_role_scheme,
                "bank_write_policy": self.config.bank_write_policy,
                "bank_merge_policy": self.config.bank_merge_policy,
                "detail_enabled": self.config.detail_enabled,
                "detail_slots": self.config.detail_slots,
                "detail_topk": self.config.detail_topk,
                "detail_scan_carry_mode": self.config.detail_scan_carry_mode,
                "detail_forward_mode": self.config.detail_forward_mode,
                "detail_coarse_group_size": self.config.detail_coarse_group_size,
                "detail_coarse_topk_groups": self.config.detail_coarse_topk_groups,
                "detail_state_shape": tuple(detail_states_out.shape),
                "refresh_state_shape": tuple(refresh_states.shape),
                "prefix_carry_state_shape": tuple(prefix_carry_states.shape),
                "fused_context_state_shape": tuple(fused_context.shape),
                "bank_read_blocks": 0,
                "bank_read_segments": 0,
                "bank_read_slots": 0,
                "token_bank_access_count": 0,
                "refresh_bank_access_count": 0,
                "detail_access_count": sum(detail_trace["detail_topk_used"]),
                "detail_candidate_count_mean": float(sum(detail_trace["detail_candidate_counts"]) / max(len(detail_trace["detail_candidate_counts"]), 1)),
                "detail_fine_candidate_count_mean": float(sum(detail_trace["detail_fine_candidate_counts"]) / max(len(detail_trace["detail_fine_candidate_counts"]), 1)),
                "detail_group_count_mean": float(sum(detail_trace["detail_group_counts"]) / max(len(detail_trace["detail_group_counts"]), 1)),
                "detail_group_topk_used_mean": float(sum(detail_trace["detail_group_topk_used"]) / max(len(detail_trace["detail_group_topk_used"]), 1)),
                "detail_selected_count_mean": float(detail_states.size(2)),
                "detail_topk_used_mean": float(sum(detail_trace["detail_topk_used"]) / max(len(detail_trace["detail_topk_used"]), 1)),
                "detail_gate_mean": float(gate.mean().detach().cpu().item()),
                "refresh_norm_mean": refresh_norm_mean,
            },
        }

    def _allocate_block_context_trace(
        self,
        batch_size: int,
        block_count: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Allocates explicit bounded per-block long-range state traces."""
        prefix_carry_states = torch.empty(batch_size, block_count, self.config.d_model, device=device)
        prefix_carry_mask = torch.zeros(batch_size, block_count, 1, device=device, dtype=torch.bool)
        fused_context_states = torch.empty(batch_size, block_count, self.config.d_model, device=device)
        fused_context_mask = torch.zeros(batch_size, block_count, 1, device=device, dtype=torch.bool)
        return prefix_carry_states, prefix_carry_mask, fused_context_states, fused_context_mask

    def _record_prefix_carry_trace(
        self,
        carry_context: Tensor | None,
        prefix_carry_states: Tensor,
        prefix_carry_mask: Tensor,
        block_index: int,
    ) -> None:
        """Records the online prefix carry visible before processing one block."""
        if carry_context is None:
            return
        prefix_carry_states[:, block_index, :] = carry_context.detach()
        prefix_carry_mask[:, block_index, :] = True

    def _allocate_block_refresh_write_trace(
        self,
        batch_size: int,
        block_count: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Allocates per-block refresh-write traces for later carry scans."""
        refresh_write_states = torch.empty(batch_size, block_count, self.config.d_model, device=device)
        refresh_write_mask = torch.zeros(batch_size, block_count, 1, device=device, dtype=torch.bool)
        return refresh_write_states, refresh_write_mask

    def _materialize_online_detail_block_step(
        self,
        pre_hidden_states: Tensor,
        carry_context: Tensor | None,
        detail_key_blocks: Tensor,
        detail_value_blocks: Tensor,
        detail_write_index: int,
    ) -> dict[str, Tensor | dict]:
        """Materializes one block's conditioned hidden states before refresh/detail writes."""
        coarse_hidden_states, pooled_hidden = self._materialize_coarse_block_state(
            pre_hidden_states,
            carry_context,
        )
        if torch.is_grad_enabled() and detail_write_index > 0:
            past_detail_keys, past_detail_values = self._slice_past_detail_kv(
                detail_key_blocks,
                detail_value_blocks,
                detail_write_index,
            )
            past_detail_keys = past_detail_keys.clone()
            past_detail_values = past_detail_values.clone()
            detail_context, detail_trace = self._retrieve_detail_context(
                pooled_hidden,
                past_detail_keys,
                past_detail_values,
            )
        else:
            detail_context, detail_trace = self._refine_block_with_detail(
                pooled_hidden,
                detail_key_blocks,
                detail_value_blocks,
                detail_write_index,
            )
        fused_context, gate = self._compose_detail_refined_context(
            carry_context,
            detail_context,
            pooled_hidden,
        )
        hidden_states = self._apply_conditioned_post_blocks(coarse_hidden_states, fused_context)
        return {
            "coarse_hidden_states": coarse_hidden_states,
            "pooled_hidden": pooled_hidden,
            "detail_context": detail_context,
            "hidden_states": hidden_states,
            "fused_context": fused_context,
            "gate": gate,
            "detail_trace": detail_trace,
        }

    def _materialize_online_detail_block_pass(
        self,
        pre_hidden_blocks: Tensor,
        initial_bank_states: Tensor,
        detail_key_blocks: Tensor,
        detail_value_blocks: Tensor,
        detail_write_index: int,
        precomputed_targets: Tensor | None = None,
    ) -> dict[str, Any]:
        """Runs the current sequential block pass and materializes explicit intermediate traces."""
        batch_size, block_count, block_size, _ = pre_hidden_blocks.shape
        output_blocks = pre_hidden_blocks.new_empty(batch_size, block_count, block_size, self.config.d_model)
        refresh_slot_count = sum(
            1 for block_index in range(block_count)
            if self.config.refresh_enabled and (block_index + 1) % self.config.refresh_interval_blocks == 0
        ) * self.config.effective_refresh_slots()
        refresh_blocks = pre_hidden_blocks.new_empty(batch_size, refresh_slot_count, self.config.d_model)
        detail_blocks = (
            pre_hidden_blocks.new_empty(batch_size, max(detail_key_blocks.size(1), 0), self.config.d_model)
            if detail_key_blocks.size(1) > 0
            else pre_hidden_blocks[:, :0, :0, :].reshape(batch_size, 0, self.config.d_model)
        )
        predicted_summary = pre_hidden_blocks.new_empty(batch_size, max(block_count - 1, 0), self.config.d_model)
        target_summary = pre_hidden_blocks.new_empty(batch_size, max(block_count - 1, 0), self.config.d_model)
        online_prefix_carry_states, online_prefix_carry_mask, fused_context_states, fused_context_mask = self._allocate_block_context_trace(
            batch_size,
            block_count,
            pre_hidden_blocks.device,
        )
        refresh_write_states, refresh_write_mask = self._allocate_block_refresh_write_trace(
            batch_size,
            block_count,
            pre_hidden_blocks.device,
        )

        carry_context = None
        bank_states = initial_bank_states
        bank_read_blocks = 0
        bank_read_slots = 0
        refresh_norm_sum = pre_hidden_blocks.new_zeros(())
        refresh_norm_count = 0
        detail_selected_counts = []
        detail_topk_used = []
        detail_candidate_counts = []
        detail_fine_candidate_counts = []
        detail_group_counts = []
        detail_group_topk_used = []
        detail_gate_sum = pre_hidden_blocks.new_zeros(())
        detail_gate_count = 0
        refresh_write_index = 0
        output_detail_write_index = detail_write_index
        sufficiency_index = 0
        next_logits = None

        for block_index in range(block_count):
            self._record_prefix_carry_trace(
                carry_context,
                online_prefix_carry_states,
                online_prefix_carry_mask,
                block_index,
            )
            pre_hidden_states = pre_hidden_blocks[:, block_index, :, :]
            step_outputs = self._materialize_online_detail_block_step(
                pre_hidden_states,
                carry_context,
                detail_key_blocks,
                detail_value_blocks,
                output_detail_write_index,
            )
            hidden_states = step_outputs["hidden_states"]
            fused_context = step_outputs["fused_context"]
            gate = step_outputs["gate"]
            detail_trace = step_outputs["detail_trace"]
            if fused_context is not None:
                fused_context_states[:, block_index, :] = fused_context.detach()
                fused_context_mask[:, block_index, :] = True
            output_blocks[:, block_index, :, :] = hidden_states
            next_logits = self._next_token_logits(hidden_states)

            detail_gate_sum = detail_gate_sum + gate.mean().detach()
            detail_gate_count += 1
            detail_candidate_counts.append(detail_trace["detail_candidate_count"])
            detail_fine_candidate_counts.append(detail_trace["detail_fine_candidate_count"])
            detail_group_counts.append(detail_trace["detail_group_count"])
            detail_group_topk_used.append(detail_trace["detail_group_topk_used"])
            detail_topk_used.append(detail_trace["detail_topk_used"])

            carry_context, bank_states, refresh_carry, refreshed, refresh_trace = self._update_detail_refresh_state(
                hidden_states,
                carry_context,
                bank_states,
                block_index + 1,
            )
            if refresh_carry is not None:
                next_refresh_index = refresh_write_index + self.config.effective_refresh_slots()
                if refreshed is None:
                    raise RuntimeError("refresh write produced carry without refreshed slots")
                refresh_blocks[:, refresh_write_index:next_refresh_index, :] = refreshed
                refresh_write_index = next_refresh_index
                refresh_write_states[:, block_index, :] = refresh_carry.detach()
                refresh_write_mask[:, block_index, :] = True
                bank_read_slots += refresh_trace["bank_read_slots"]
                bank_read_blocks += int(refresh_trace["bank_used"])
                if carry_context is not None:
                    refresh_norm_sum = refresh_norm_sum + carry_context.norm(dim=-1).mean().detach()
                    refresh_norm_count += 1
                if precomputed_targets is not None and block_index + 1 < block_count and carry_context is not None:
                    predicted_summary[:, sufficiency_index, :] = self.sufficiency_head(carry_context)
                    target_summary[:, sufficiency_index, :] = precomputed_targets[:, block_index + 1, :]
                    sufficiency_index += 1

            if self.config.detail_enabled:
                detail_slots, _ = self._select_detail_slots(hidden_states)
                detail_selected_counts.append(int(detail_slots.size(1)))
                if detail_slots.size(1) > 0:
                    next_detail_index = output_detail_write_index + detail_slots.size(1)
                    if detail_blocks.size(1) < next_detail_index:
                        next_capacity = max(next_detail_index, max(1, detail_blocks.size(1) * 2, self.config.detail_slots))
                        next_detail_blocks = pre_hidden_blocks.new_empty(batch_size, next_capacity, self.config.d_model)
                        if output_detail_write_index > 0:
                            next_detail_blocks[:, :output_detail_write_index, :] = detail_blocks[:, :output_detail_write_index, :]
                        detail_blocks = next_detail_blocks
                    detail_blocks[:, output_detail_write_index:next_detail_index, :] = detail_slots.detach()
                    with torch.no_grad():
                        detail_keys = self.detail_key_proj(detail_slots.detach())
                        detail_values = self.detail_value_proj(detail_slots.detach())
                    detail_key_blocks, detail_value_blocks = self._ensure_detail_kv_cache_capacity(
                        detail_key_blocks,
                        detail_value_blocks,
                        output_detail_write_index,
                        next_detail_index,
                    )
                    detail_key_blocks[:, output_detail_write_index:next_detail_index, :] = detail_keys
                    detail_value_blocks[:, output_detail_write_index:next_detail_index, :] = detail_values
                    output_detail_write_index = next_detail_index
            else:
                detail_selected_counts.append(0)

        return {
            "output_blocks": output_blocks,
            "refresh_blocks": refresh_blocks[:, :refresh_write_index, :],
            "detail_blocks": detail_blocks[:, :output_detail_write_index, :],
            "detail_key_blocks": detail_key_blocks,
            "detail_value_blocks": detail_value_blocks,
            "detail_write_index": output_detail_write_index,
            "bank_states": bank_states,
            "carry_context": carry_context,
            "next_logits": next_logits,
            "predicted_summary": predicted_summary[:, :sufficiency_index, :] if precomputed_targets is not None else predicted_summary[:, :0, :],
            "target_summary": target_summary[:, :sufficiency_index, :] if precomputed_targets is not None else target_summary[:, :0, :],
            "online_prefix_carry_states": online_prefix_carry_states,
            "online_prefix_carry_mask": online_prefix_carry_mask,
            "fused_context_states": fused_context_states,
            "fused_context_mask": fused_context_mask,
            "refresh_write_states": refresh_write_states,
            "refresh_write_mask": refresh_write_mask,
            "bank_read_blocks": bank_read_blocks,
            "bank_read_slots": bank_read_slots,
            "detail_candidate_counts": detail_candidate_counts,
            "detail_fine_candidate_counts": detail_fine_candidate_counts,
            "detail_group_counts": detail_group_counts,
            "detail_group_topk_used": detail_group_topk_used,
            "detail_selected_counts": detail_selected_counts,
            "detail_topk_used": detail_topk_used,
            "detail_gate_sum": detail_gate_sum,
            "detail_gate_count": detail_gate_count,
            "refresh_norm_sum": refresh_norm_sum,
            "refresh_norm_count": refresh_norm_count,
        }

    def _scan_detail_block_sequence(
        self,
        pre_hidden_blocks: Tensor,
        precomputed_targets: Tensor,
        initial_bank_states: Tensor,
    ) -> dict[str, Any]:
        """Runs the sequential online pass, then reconstructs prefix carry via the scan helper."""
        batch_size, block_count, _, _ = pre_hidden_blocks.shape
        max_detail_slots = block_count * self.config.detail_slots if self.config.detail_enabled else 0
        detail_key_blocks = (
            pre_hidden_blocks.new_empty(batch_size, max_detail_slots, self.config.d_model)
            if max_detail_slots > 0
            else pre_hidden_blocks[:, :0, :0, :].reshape(batch_size, 0, self.config.d_model)
        )
        detail_value_blocks = (
            pre_hidden_blocks.new_empty(batch_size, max_detail_slots, self.config.d_model)
            if max_detail_slots > 0
            else detail_key_blocks
        )
        online_outputs = self._materialize_online_detail_block_pass(
            pre_hidden_blocks,
            initial_bank_states,
            detail_key_blocks,
            detail_value_blocks,
            0,
            precomputed_targets,
        )
        prefix_carry_states, prefix_carry_mask = self._scan_carry_sequence_from_refresh_writes(
            online_outputs["refresh_write_states"],
            online_outputs["refresh_write_mask"],
        )
        online_outputs["prefix_carry_states"] = prefix_carry_states
        online_outputs["prefix_carry_mask"] = prefix_carry_mask
        return online_outputs

    def _update_detail_refresh_state(
        self,
        hidden_states: Tensor,
        carry_context: Tensor | None,
        bank_states: Tensor,
        completed_blocks: int,
    ) -> tuple[Tensor | None, Tensor, Tensor | None, Tensor | None, dict[str, int]]:
        """Updates carry/bank state for the detail model, optionally using an explicit carry recurrence."""
        refresh_due = self.config.refresh_enabled and completed_blocks % self.config.refresh_interval_blocks == 0
        if not refresh_due:
            if not self.config.refresh_enabled:
                carry_context = None
            return carry_context, bank_states, None, None, {"bank_read_slots": 0, "bank_used": 0}

        refresh_inputs = self._build_refresh_inputs(hidden_states)
        refreshed, refresh_trace = self.refresh_block(refresh_inputs, self.long_bank.read(bank_states))
        refresh_carry = self._write_bank_entry(refreshed)
        carry_context = self._compose_scan_carry_state(carry_context, refresh_carry)
        if self.config.refresh_write_gate_enabled:
            if self.bank_write_importance is None:
                raise RuntimeError("refresh write gating requires bank_write_importance module")
            gate_score = torch.sigmoid(self.bank_write_importance(refreshed).mean(dim=1))
            carry_context = gate_score * carry_context
        bank_entry = self.bank_entry_proj(carry_context).unsqueeze(1)
        bank_states = self.long_bank.write(bank_states, bank_entry)
        return carry_context, bank_states, refresh_carry, refreshed, refresh_trace

    def _scan_completed_blocks_prefill(
        self,
        completed_pre_hidden_blocks: Tensor,
        initial_bank_states: Tensor,
        detail_key_blocks: Tensor,
        detail_value_blocks: Tensor,
        detail_write_index: int,
    ) -> dict[str, Any]:
        """Runs the completed-block prefix pass through the shared online materialization helper."""
        online_outputs = self._materialize_online_detail_block_pass(
            completed_pre_hidden_blocks,
            initial_bank_states,
            detail_key_blocks,
            detail_value_blocks,
            detail_write_index,
            precomputed_targets=None,
        )
        prefix_carry_states, prefix_carry_mask = self._scan_carry_sequence_from_refresh_writes(
            online_outputs["refresh_write_states"],
            online_outputs["refresh_write_mask"],
        )
        return {
            "carry_context": online_outputs["carry_context"],
            "bank_states": online_outputs["bank_states"],
            "detail_key_blocks": online_outputs["detail_key_blocks"],
            "detail_value_blocks": online_outputs["detail_value_blocks"],
            "detail_write_index": online_outputs["detail_write_index"],
            "next_logits": online_outputs["next_logits"],
            "prefix_carry_states": prefix_carry_states,
            "prefix_carry_mask": prefix_carry_mask,
            "online_prefix_carry_states": online_outputs["online_prefix_carry_states"],
            "online_prefix_carry_mask": online_outputs["online_prefix_carry_mask"],
            "fused_context_states": online_outputs["fused_context_states"],
            "fused_context_mask": online_outputs["fused_context_mask"],
            "refresh_write_states": online_outputs["refresh_write_states"],
            "refresh_write_mask": online_outputs["refresh_write_mask"],
        }

    def _apply_detail_post_stack(
        self,
        hidden_states: Tensor,
        fused_context: Tensor | None,
    ) -> Tensor:
        """Applies the upper local stack after adding the fused long-range context."""
        if fused_context is not None:
            hidden_states = hidden_states + self.carry_to_post(fused_context).unsqueeze(1)
        for block in self.post_blocks:
            hidden_states = block(hidden_states)
        return hidden_states

    def _detail_suffix_len(self, block_len: int) -> int:
        """Returns the exact local receptive-field suffix needed for one decode step."""
        if block_len <= 0:
            return 0
        if not self.post_blocks:
            return 1
        receptive_field = 1 + len(self.post_blocks) * self.config.local_window
        return min(block_len, receptive_field)

    def _next_token_logits(self, hidden_states: Tensor) -> Tensor:
        """Projects only the last token when decode/prefill needs next-token logits."""
        last_hidden = hidden_states[:, -1:, :]
        return self.lm_head(self.final_norm(last_hidden))[:, -1, :]

    def _allocate_detail_kv_cache(
        self,
        batch_size: int,
        capacity: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        if capacity <= 0:
            empty = torch.empty(batch_size, 0, self.config.d_model, device=device)
            return empty, empty
        detail_key_blocks = torch.empty(batch_size, capacity, self.config.d_model, device=device)
        detail_value_blocks = torch.empty(batch_size, capacity, self.config.d_model, device=device)
        return detail_key_blocks, detail_value_blocks

    def _allocate_detail_cache(
        self,
        batch_size: int,
        capacity: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        detail_key_blocks, detail_value_blocks = self._allocate_detail_kv_cache(batch_size, capacity, device)
        detail_states = (
            torch.empty(batch_size, capacity, self.config.d_model, device=device)
            if capacity > 0
            else detail_key_blocks
        )
        return detail_states, detail_key_blocks, detail_value_blocks

    def _ensure_detail_kv_cache_capacity(
        self,
        detail_key_blocks: Tensor,
        detail_value_blocks: Tensor,
        write_index: int,
        required_capacity: int,
    ) -> tuple[Tensor, Tensor]:
        current_capacity = detail_key_blocks.size(1)
        if required_capacity <= current_capacity:
            return detail_key_blocks, detail_value_blocks
        next_capacity = max(required_capacity, max(1, current_capacity * 2, self.config.detail_slots))
        next_keys, next_values = self._allocate_detail_kv_cache(
            detail_key_blocks.size(0),
            next_capacity,
            detail_key_blocks.device,
        )
        if write_index > 0:
            next_keys[:, :write_index, :] = detail_key_blocks[:, :write_index, :]
            next_values[:, :write_index, :] = detail_value_blocks[:, :write_index, :]
        return next_keys, next_values

    def _ensure_detail_cache_capacity(
        self,
        detail_states: Tensor,
        detail_key_blocks: Tensor,
        detail_value_blocks: Tensor,
        write_index: int,
        required_capacity: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        current_capacity = detail_states.size(1)
        if required_capacity <= current_capacity:
            return detail_states, detail_key_blocks, detail_value_blocks
        next_capacity = max(required_capacity, max(1, current_capacity * 2, self.config.detail_slots))
        next_states = torch.empty(detail_states.size(0), next_capacity, self.config.d_model, device=detail_states.device)
        next_keys, next_values = self._allocate_detail_kv_cache(
            detail_states.size(0),
            next_capacity,
            detail_states.device,
        )
        if write_index > 0:
            next_states[:, :write_index, :] = detail_states[:, :write_index, :]
            next_keys[:, :write_index, :] = detail_key_blocks[:, :write_index, :]
            next_values[:, :write_index, :] = detail_value_blocks[:, :write_index, :]
        return next_states, next_keys, next_values

    def _write_detail_kv_cache(
        self,
        hidden_states: Tensor,
        detail_key_blocks: Tensor,
        detail_value_blocks: Tensor,
        detail_write_index: int,
    ) -> tuple[Tensor, Tensor, int]:
        """Writes detached detail KV caches for decode/prefill state."""
        if not self.config.detail_enabled:
            return detail_key_blocks, detail_value_blocks, detail_write_index
        detail_slots, _ = self._select_detail_slots(hidden_states)
        if detail_slots.size(1) == 0:
            return detail_key_blocks, detail_value_blocks, detail_write_index
        next_detail_index = detail_write_index + detail_slots.size(1)
        detail_key_blocks, detail_value_blocks = self._ensure_detail_kv_cache_capacity(
            detail_key_blocks,
            detail_value_blocks,
            detail_write_index,
            next_detail_index,
        )
        with torch.no_grad():
            detail_key_blocks[:, detail_write_index:next_detail_index, :] = self.detail_key_proj(detail_slots.detach())
            detail_value_blocks[:, detail_write_index:next_detail_index, :] = self.detail_value_proj(detail_slots.detach())
        return detail_key_blocks, detail_value_blocks, next_detail_index

    def _write_detail_cache(
        self,
        hidden_states: Tensor,
        detail_states: Tensor,
        detail_key_blocks: Tensor,
        detail_value_blocks: Tensor,
        detail_write_index: int,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        """Writes selected detail slots into reusable cache buffers."""
        if not self.config.detail_enabled:
            return detail_states, detail_key_blocks, detail_value_blocks, detail_write_index
        detail_slots, _ = self._select_detail_slots(hidden_states)
        if detail_slots.size(1) == 0:
            return detail_states, detail_key_blocks, detail_value_blocks, detail_write_index
        next_detail_index = detail_write_index + detail_slots.size(1)
        detail_states, detail_key_blocks, detail_value_blocks = self._ensure_detail_cache_capacity(
            detail_states,
            detail_key_blocks,
            detail_value_blocks,
            detail_write_index,
            next_detail_index,
        )
        detail_states[:, detail_write_index:next_detail_index, :] = detail_slots.detach()
        with torch.no_grad():
            detail_key_blocks[:, detail_write_index:next_detail_index, :] = self.detail_key_proj(detail_slots.detach())
            detail_value_blocks[:, detail_write_index:next_detail_index, :] = self.detail_value_proj(detail_slots.detach())
        return detail_states, detail_key_blocks, detail_value_blocks, next_detail_index

    def prefill(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> dict[str, Any]:
        """Builds decode cache state for the detail variant from a prefix."""
        batch_size, seq_len = input_ids.shape
        block_size = self.config.effective_block_size()
        completed_blocks = seq_len // block_size
        prefix_len = completed_blocks * block_size
        bank_states = initial_bank_states if initial_bank_states is not None else self.long_bank.empty(batch_size, input_ids.device)
        carry_context = None
        initial_detail_capacity = completed_blocks * self.config.detail_slots if self.config.detail_enabled else 0
        detail_key_blocks, detail_value_blocks = self._allocate_detail_kv_cache(
            batch_size,
            initial_detail_capacity,
            input_ids.device,
        )
        detail_write_index = 0
        next_logits = None
        open_post_caches = self._empty_local_stack_caches(self.post_blocks)
        completed_pre_hidden_blocks = None
        completed_prefix_carry_states = torch.empty(batch_size, completed_blocks, self.config.d_model, device=input_ids.device)
        completed_prefix_carry_mask = torch.zeros(batch_size, completed_blocks, 1, device=input_ids.device, dtype=torch.bool)
        completed_fused_context_states = torch.empty(batch_size, completed_blocks, self.config.d_model, device=input_ids.device)
        completed_fused_context_mask = torch.zeros(batch_size, completed_blocks, 1, device=input_ids.device, dtype=torch.bool)
        if completed_blocks > 0:
            completed_embeddings = self.embedding(input_ids[:, :prefix_len])
            completed_block_embeddings, _, _ = self._reshape_blocks(completed_embeddings)
            completed_pre_hidden_blocks = self._encode_blocks_parallel(completed_block_embeddings)
            completed_outputs = self._scan_completed_blocks_prefill(
                completed_pre_hidden_blocks,
                bank_states,
                detail_key_blocks,
                detail_value_blocks,
                detail_write_index,
            )
            carry_context = completed_outputs["carry_context"]
            bank_states = completed_outputs["bank_states"]
            detail_key_blocks = completed_outputs["detail_key_blocks"]
            detail_value_blocks = completed_outputs["detail_value_blocks"]
            detail_write_index = completed_outputs["detail_write_index"]
            next_logits = completed_outputs["next_logits"]
            completed_prefix_carry_states = completed_outputs["prefix_carry_states"]
            completed_prefix_carry_mask = completed_outputs["prefix_carry_mask"]
            completed_fused_context_states = completed_outputs["fused_context_states"]
            completed_fused_context_mask = completed_outputs["fused_context_mask"]

        open_block_ids = input_ids[:, prefix_len:]
        open_pre_hidden = torch.empty(batch_size, block_size, self.config.d_model, device=input_ids.device)
        open_pre_sum = torch.zeros(batch_size, self.config.d_model, device=input_ids.device)
        open_pre_caches = self._empty_local_stack_caches(self.pre_blocks)
        open_block_hidden = torch.empty(batch_size, block_size, self.config.d_model, device=input_ids.device)
        open_block_len = 0
        if open_block_ids.size(1) > 0:
            hidden_states = self.embedding(open_block_ids)
            hidden_states, open_pre_caches = self._prefill_local_stack_cache(hidden_states, self.pre_blocks)
            open_block_len = int(hidden_states.size(1))
            open_pre_hidden[:, :open_block_len, :] = hidden_states.detach()
            open_pre_sum = hidden_states.detach().sum(dim=1)
            hidden_states, _, detail_context, fused_context, _, _ = self._scan_block_state(
                hidden_states,
                carry_context,
                detail_key_blocks,
                detail_value_blocks,
                detail_write_index,
                pooled_hidden=open_pre_sum / max(open_block_len, 1),
            )
            if detail_context is None:
                if fused_context is not None:
                    hidden_states = hidden_states + self.carry_to_post(fused_context).unsqueeze(1)
                hidden_states, open_post_caches = self._prefill_local_stack_cache(hidden_states, self.post_blocks)
            else:
                hidden_states = self._apply_conditioned_post_blocks(hidden_states, fused_context)
            next_logits = self._next_token_logits(hidden_states)
            open_block_hidden[:, :open_block_len, :] = hidden_states.detach()

        if next_logits is None:
            raise ValueError("prefill requires at least one input token")

        return {
            "open_pre_hidden": open_pre_hidden,
            "open_pre_sum": open_pre_sum,
            "open_pre_caches": open_pre_caches,
            "open_post_caches": open_post_caches,
            "open_block_hidden": open_block_hidden,
            "open_block_len": open_block_len,
            "completed_blocks": completed_blocks,
            "carry_context": carry_context.detach() if carry_context is not None else None,
            "bank_states": bank_states.detach(),
            "detail_key_blocks": detail_key_blocks,
            "detail_value_blocks": detail_value_blocks,
            "detail_write_index": detail_write_index,
            "completed_prefix_carry_states": completed_prefix_carry_states,
            "completed_prefix_carry_mask": completed_prefix_carry_mask,
            "completed_fused_context_states": completed_fused_context_states,
            "completed_fused_context_mask": completed_fused_context_mask,
            "next_logits": next_logits.detach(),
        }

    def decode_step(self, next_input_ids: Tensor, state: dict[str, Any]) -> dict[str, Any]:
        """Appends one token and recomputes only the current open block for the detail variant."""
        if next_input_ids.dim() == 1:
            next_input_ids = next_input_ids.unsqueeze(1)
        carry_context = state["carry_context"]
        bank_states = state["bank_states"]
        completed_blocks = int(state["completed_blocks"])
        detail_key_blocks = state["detail_key_blocks"]
        detail_value_blocks = state["detail_value_blocks"]
        open_pre_hidden = state["open_pre_hidden"]
        open_pre_sum = state["open_pre_sum"]
        open_block_len = int(state["open_block_len"])
        detail_write_index = int(state.get("detail_write_index", detail_key_blocks.size(1)))

        hidden_states = self.embedding(next_input_ids)
        hidden_states, open_pre_caches = self._decode_local_stack_step(
            hidden_states,
            self.pre_blocks,
            state.get("open_pre_caches"),
        )
        open_pre_hidden[:, open_block_len : open_block_len + 1, :] = hidden_states.detach()
        open_pre_sum = open_pre_sum + hidden_states.detach().squeeze(1)
        current_block_len = open_block_len + 1

        pre_hidden = open_pre_hidden[:, :current_block_len, :]
        pre_hidden, _, detail_context, fused_context, _, _ = self._scan_block_state(
            pre_hidden,
            carry_context,
            detail_key_blocks,
            detail_value_blocks,
            detail_write_index,
            pooled_hidden=open_pre_sum / current_block_len,
        )
        open_block_hidden = state["open_block_hidden"]
        open_post_caches = self._empty_local_stack_caches(self.post_blocks)
        used_suffix_recompute = detail_context is not None
        if not used_suffix_recompute:
            token_hidden = pre_hidden[:, -1:, :]
            if fused_context is not None:
                token_hidden = token_hidden + self.carry_to_post(fused_context).unsqueeze(1)
            token_hidden, open_post_caches = self._decode_local_stack_step(
                token_hidden,
                self.post_blocks,
                state.get("open_post_caches"),
            )
            next_logits = self._next_token_logits(token_hidden).detach()
            open_block_hidden[:, open_block_len:current_block_len, :] = token_hidden.detach()
        else:
            suffix_len = self._detail_suffix_len(current_block_len)
            suffix_start = current_block_len - suffix_len
            hidden_states = pre_hidden[:, suffix_start:, :]
            hidden_states = self._apply_detail_post_stack(hidden_states, fused_context)
            next_logits = self._next_token_logits(hidden_states).detach()
            open_block_hidden[:, suffix_start:current_block_len, :] = hidden_states.detach()
        open_block_hidden = state["open_block_hidden"]
        open_block_len = current_block_len

        if open_block_len == self.config.effective_block_size():
            if used_suffix_recompute:
                completed_hidden = pre_hidden
                completed_hidden = self._apply_conditioned_post_blocks(completed_hidden, fused_context)
                open_block_hidden[:, :open_block_len, :] = completed_hidden.detach()
                next_logits = self._next_token_logits(completed_hidden).detach()
            else:
                completed_hidden = open_block_hidden[:, :open_block_len, :]
            completed_blocks += 1
            carry_context, bank_states, _, _, _ = self._update_detail_refresh_state(
                completed_hidden,
                carry_context,
                bank_states,
                completed_blocks,
            )
            detail_key_blocks, detail_value_blocks, detail_write_index = self._write_detail_kv_cache(
                completed_hidden,
                detail_key_blocks,
                detail_value_blocks,
                detail_write_index,
            )
            open_pre_sum = torch.zeros_like(open_pre_sum)
            open_block_len = 0
            open_pre_caches = self._empty_local_stack_caches(self.pre_blocks)
            open_post_caches = self._empty_local_stack_caches(self.post_blocks)

        return {
            "open_pre_hidden": open_pre_hidden,
            "open_pre_sum": open_pre_sum,
            "open_pre_caches": open_pre_caches,
            "open_post_caches": open_post_caches,
            "open_block_hidden": open_block_hidden,
            "open_block_len": open_block_len,
            "completed_blocks": completed_blocks,
            "carry_context": carry_context.detach() if carry_context is not None else None,
            "bank_states": bank_states.detach(),
            "detail_key_blocks": detail_key_blocks,
            "detail_value_blocks": detail_value_blocks,
            "detail_write_index": detail_write_index,
            "next_logits": next_logits,
        }

    def forward(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> Dict[str, Tensor]:
        """Runs block-refresh SRD plus sparse retrieval over tiny detail memories."""
        if self.config.detail_forward_mode == "parallel_scan":
            return self._forward_parallel_scan(input_ids, initial_bank_states)
        batch_size, seq_len = input_ids.shape
        embeddings = self.embedding(input_ids)
        block_embeddings, block_count, _ = self._reshape_blocks(embeddings)
        precomputed_targets = self._precompute_next_block_targets(input_ids)
        bank_states = initial_bank_states if initial_bank_states is not None else self.long_bank.empty(batch_size, input_ids.device)
        pre_hidden_blocks = self._encode_blocks_parallel(block_embeddings)
        scan_outputs = self._scan_detail_block_sequence(
            pre_hidden_blocks,
            precomputed_targets,
            bank_states,
        )

        hidden_states = scan_outputs["output_blocks"].view(batch_size, seq_len, self.config.d_model)
        logits = self.lm_head(self.final_norm(hidden_states))
        empty = self._empty_state(batch_size, input_ids.device)
        refresh_states = scan_outputs["refresh_blocks"].detach() if scan_outputs["refresh_blocks"].size(1) > 0 else empty
        detail_states = scan_outputs["detail_blocks"].detach() if scan_outputs["detail_blocks"].size(1) > 0 else empty
        predicted_summary = (
            scan_outputs["predicted_summary"]
            if scan_outputs["predicted_summary"].size(1) > 0
            else empty
        )
        target_summary = (
            scan_outputs["target_summary"]
            if scan_outputs["target_summary"].size(1) > 0
            else empty
        )
        refresh_norm_mean = 0.0
        if scan_outputs["refresh_norm_count"] > 0:
            refresh_norm_mean = float((scan_outputs["refresh_norm_sum"] / scan_outputs["refresh_norm_count"]).cpu().item())
        detail_gate_mean = 0.0
        if scan_outputs["detail_gate_count"] > 0:
            detail_gate_mean = float((scan_outputs["detail_gate_sum"] / scan_outputs["detail_gate_count"]).cpu().item())

        return {
            "logits": logits,
            "hidden_states": empty,
            "refresh_states": refresh_states,
            "detail_states": detail_states,
            "bank_states": scan_outputs["bank_states"].detach(),
            "predicted_summary": predicted_summary,
            "target_summary": target_summary,
            "debug": {
                "block_count": block_count,
                "segment_count": block_count,
                "block_size": self.config.effective_block_size(),
                "refresh_slots": self.config.effective_refresh_slots(),
                "refresh_enabled": self.config.refresh_enabled,
                "refresh_role_scheme": self.config.refresh_role_scheme,
                "bank_write_policy": self.config.bank_write_policy,
                "bank_merge_policy": self.config.bank_merge_policy,
                "detail_enabled": self.config.detail_enabled,
                "detail_slots": self.config.detail_slots,
                "detail_topk": self.config.detail_topk,
                "detail_scan_carry_mode": self.config.detail_scan_carry_mode,
                "detail_forward_mode": self.config.detail_forward_mode,
                "detail_coarse_group_size": self.config.detail_coarse_group_size,
                "detail_coarse_topk_groups": self.config.detail_coarse_topk_groups,
                "detail_state_shape": tuple(detail_states.shape),
                "refresh_state_shape": tuple(refresh_states.shape),
                "prefix_carry_state_shape": tuple(scan_outputs["prefix_carry_states"].shape),
                "fused_context_state_shape": tuple(scan_outputs["fused_context_states"].shape),
                "bank_read_blocks": scan_outputs["bank_read_blocks"],
                "bank_read_segments": scan_outputs["bank_read_blocks"],
                "bank_read_slots": scan_outputs["bank_read_slots"],
                "token_bank_access_count": 0,
                "refresh_bank_access_count": scan_outputs["bank_read_slots"],
                "detail_access_count": sum(scan_outputs["detail_topk_used"]),
                "detail_candidate_count_mean": float(sum(scan_outputs["detail_candidate_counts"]) / max(len(scan_outputs["detail_candidate_counts"]), 1)),
                "detail_fine_candidate_count_mean": float(sum(scan_outputs["detail_fine_candidate_counts"]) / max(len(scan_outputs["detail_fine_candidate_counts"]), 1)),
                "detail_group_count_mean": float(sum(scan_outputs["detail_group_counts"]) / max(len(scan_outputs["detail_group_counts"]), 1)),
                "detail_group_topk_used_mean": float(sum(scan_outputs["detail_group_topk_used"]) / max(len(scan_outputs["detail_group_topk_used"]), 1)),
                "detail_selected_count_mean": float(sum(scan_outputs["detail_selected_counts"]) / max(len(scan_outputs["detail_selected_counts"]), 1)),
                "detail_topk_used_mean": float(sum(scan_outputs["detail_topk_used"]) / max(len(scan_outputs["detail_topk_used"]), 1)),
                "detail_gate_mean": detail_gate_mean,
                "refresh_norm_mean": refresh_norm_mean,
            },
        }
