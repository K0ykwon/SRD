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
                "detail_topk_used": 0,
            }

        query = self.detail_query_proj(pooled_hidden).unsqueeze(1)
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
        if fused_context is not None:
            hidden_states = hidden_states + self.carry_to_post(fused_context).unsqueeze(1)

        for block in self.post_blocks:
            hidden_states = block(hidden_states)
        if detail_context is None:
            detail_context = hidden_states.new_zeros(batch_size, self.config.d_model)
        return hidden_states, detail_context

    def _allocate_detail_cache(
        self,
        batch_size: int,
        capacity: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if capacity <= 0:
            empty = torch.empty(batch_size, 0, self.config.d_model, device=device)
            return empty, empty, empty
        detail_states = torch.empty(batch_size, capacity, self.config.d_model, device=device)
        detail_key_blocks = torch.empty(batch_size, capacity, self.config.d_model, device=device)
        detail_value_blocks = torch.empty(batch_size, capacity, self.config.d_model, device=device)
        return detail_states, detail_key_blocks, detail_value_blocks

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
        next_states, next_keys, next_values = self._allocate_detail_cache(
            detail_states.size(0),
            next_capacity,
            detail_states.device,
        )
        if write_index > 0:
            next_states[:, :write_index, :] = detail_states[:, :write_index, :]
            next_keys[:, :write_index, :] = detail_key_blocks[:, :write_index, :]
            next_values[:, :write_index, :] = detail_value_blocks[:, :write_index, :]
        return next_states, next_keys, next_values

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
        detail_key_blocks[:, detail_write_index:next_detail_index, :] = self.detail_key_proj(detail_slots).detach()
        detail_value_blocks[:, detail_write_index:next_detail_index, :] = self.detail_value_proj(detail_slots).detach()
        return detail_states, detail_key_blocks, detail_value_blocks, next_detail_index

    def _append_detail_cache(
        self,
        hidden_states: Tensor,
        detail_states: Tensor,
        detail_key_blocks: Tensor,
        detail_value_blocks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Appends selected detail slots from a completed block to the cached detail memory."""
        if not self.config.detail_enabled:
            return detail_states, detail_key_blocks, detail_value_blocks
        detail_slots, _ = self._select_detail_slots(hidden_states)
        if detail_slots.size(1) == 0:
            return detail_states, detail_key_blocks, detail_value_blocks
        detail_states = torch.cat([detail_states, detail_slots.detach()], dim=1)
        detail_key_blocks = torch.cat([detail_key_blocks, self.detail_key_proj(detail_slots).detach()], dim=1)
        detail_value_blocks = torch.cat([detail_value_blocks, self.detail_value_proj(detail_slots).detach()], dim=1)
        return detail_states, detail_key_blocks, detail_value_blocks

    def prefill(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> dict[str, Any]:
        """Builds decode cache state for the detail variant from a prefix."""
        batch_size, seq_len = input_ids.shape
        block_size = self.config.effective_block_size()
        completed_blocks = seq_len // block_size
        prefix_len = completed_blocks * block_size
        bank_states = initial_bank_states if initial_bank_states is not None else self.long_bank.empty(batch_size, input_ids.device)
        carry_context = None
        initial_detail_capacity = completed_blocks * self.config.detail_slots if self.config.detail_enabled else 0
        detail_states, detail_key_blocks, detail_value_blocks = self._allocate_detail_cache(
            batch_size,
            initial_detail_capacity,
            input_ids.device,
        )
        detail_write_index = 0
        next_logits = None

        for block_index in range(completed_blocks):
            block_ids = input_ids[:, block_index * block_size : (block_index + 1) * block_size]
            hidden_states, _ = self._process_token_block_with_detail(
                block_ids,
                carry_context,
                detail_key_blocks[:, :detail_write_index, :],
                detail_value_blocks[:, :detail_write_index, :],
            )
            logits = self.lm_head(self.final_norm(hidden_states))
            next_logits = logits[:, -1, :]
            carry_context, bank_states = self._update_refresh_state(
                hidden_states,
                carry_context,
                bank_states,
                block_index + 1,
            )
            detail_states, detail_key_blocks, detail_value_blocks, detail_write_index = self._write_detail_cache(
                hidden_states,
                detail_states,
                detail_key_blocks,
                detail_value_blocks,
                detail_write_index,
            )

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
            if carry_context is not None and not self.config.upper_layer_only_refresh:
                hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)
            pooled_hidden = open_pre_sum / max(open_block_len, 1)
            detail_context, _ = self._retrieve_detail_context(
                pooled_hidden,
                detail_key_blocks[:, :detail_write_index, :],
                detail_value_blocks[:, :detail_write_index, :],
            )
            fused_context, _ = self._fuse_long_context(carry_context, detail_context, pooled_hidden)
            if fused_context is not None:
                hidden_states = hidden_states + self.carry_to_post(fused_context).unsqueeze(1)
            for block in self.post_blocks:
                hidden_states = block(hidden_states)
            logits = self.lm_head(self.final_norm(hidden_states))
            next_logits = logits[:, -1, :]
            open_block_hidden[:, :open_block_len, :] = hidden_states.detach()

        if next_logits is None:
            raise ValueError("prefill requires at least one input token")

        return {
            "open_pre_hidden": open_pre_hidden,
            "open_pre_sum": open_pre_sum,
            "open_pre_caches": open_pre_caches,
            "open_block_hidden": open_block_hidden,
            "open_block_len": open_block_len,
            "completed_blocks": completed_blocks,
            "carry_context": carry_context.detach() if carry_context is not None else None,
            "bank_states": bank_states.detach(),
            "detail_states": detail_states,
            "detail_key_blocks": detail_key_blocks,
            "detail_value_blocks": detail_value_blocks,
            "detail_write_index": detail_write_index,
            "next_logits": next_logits.detach(),
        }

    def decode_step(self, next_input_ids: Tensor, state: dict[str, Any]) -> dict[str, Any]:
        """Appends one token and recomputes only the current open block for the detail variant."""
        if next_input_ids.dim() == 1:
            next_input_ids = next_input_ids.unsqueeze(1)
        carry_context = state["carry_context"]
        bank_states = state["bank_states"]
        completed_blocks = int(state["completed_blocks"])
        detail_states = state["detail_states"]
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
        if carry_context is not None and not self.config.upper_layer_only_refresh:
            pre_hidden = pre_hidden + self.carry_to_pre(carry_context).unsqueeze(1)
        pooled_hidden = open_pre_sum / current_block_len
        detail_context, _ = self._retrieve_detail_context(
            pooled_hidden,
            detail_key_blocks[:, :detail_write_index, :],
            detail_value_blocks[:, :detail_write_index, :],
        )
        fused_context, _ = self._fuse_long_context(carry_context, detail_context, pooled_hidden)
        hidden_states = pre_hidden
        if fused_context is not None:
            hidden_states = hidden_states + self.carry_to_post(fused_context).unsqueeze(1)
        for block in self.post_blocks:
            hidden_states = block(hidden_states)
        logits = self.lm_head(self.final_norm(hidden_states))
        next_logits = logits[:, -1, :].detach()
        open_block_hidden = state["open_block_hidden"]
        open_block_hidden[:, :current_block_len, :] = hidden_states.detach()
        open_block_len = current_block_len

        if open_block_len == self.config.effective_block_size():
            completed_blocks += 1
            carry_context, bank_states = self._update_refresh_state(
                open_block_hidden[:, :open_block_len, :],
                carry_context,
                bank_states,
                completed_blocks,
            )
            detail_states, detail_key_blocks, detail_value_blocks, detail_write_index = self._write_detail_cache(
                open_block_hidden[:, :open_block_len, :],
                detail_states,
                detail_key_blocks,
                detail_value_blocks,
                detail_write_index,
            )
            open_pre_sum = torch.zeros_like(open_pre_sum)
            open_block_len = 0
            open_pre_caches = self._empty_local_stack_caches(self.pre_blocks)

        return {
            "open_pre_hidden": open_pre_hidden,
            "open_pre_sum": open_pre_sum,
            "open_pre_caches": open_pre_caches,
            "open_block_hidden": open_block_hidden,
            "open_block_len": open_block_len,
            "completed_blocks": completed_blocks,
            "carry_context": carry_context.detach() if carry_context is not None else None,
            "bank_states": bank_states.detach(),
            "detail_states": detail_states,
            "detail_key_blocks": detail_key_blocks,
            "detail_value_blocks": detail_value_blocks,
            "detail_write_index": detail_write_index,
            "next_logits": next_logits,
        }

    def forward(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> Dict[str, Tensor]:
        """Runs block-refresh SRD plus sparse retrieval over tiny detail memories."""
        batch_size, seq_len = input_ids.shape
        embeddings = self.embedding(input_ids)
        block_embeddings, block_count, _ = self._reshape_blocks(embeddings)
        precomputed_targets = self._precompute_next_block_targets(input_ids)
        bank_states = initial_bank_states if initial_bank_states is not None else self.long_bank.empty(batch_size, input_ids.device)
        carry_context = None

        output_blocks = embeddings.new_empty(batch_size, block_count, self.config.effective_block_size(), self.config.d_model)
        refresh_slot_count = sum(
            1 for block_index in range(block_count)
            if self.config.refresh_enabled and (block_index + 1) % self.config.refresh_interval_blocks == 0
        ) * self.config.effective_refresh_slots()
        refresh_blocks = embeddings.new_empty(batch_size, refresh_slot_count, self.config.d_model)
        max_detail_slots = block_count * self.config.detail_slots if self.config.detail_enabled else 0
        detail_blocks = embeddings.new_empty(batch_size, max_detail_slots, self.config.d_model) if max_detail_slots > 0 else embeddings[:, :0, :]
        detail_key_blocks = embeddings.new_empty(batch_size, max_detail_slots, self.config.d_model) if max_detail_slots > 0 else embeddings[:, :0, :]
        detail_value_blocks = embeddings.new_empty(batch_size, max_detail_slots, self.config.d_model) if max_detail_slots > 0 else embeddings[:, :0, :]
        sufficiency_predictions = embeddings.new_empty(batch_size, max(block_count - 1, 0), self.config.d_model)
        sufficiency_targets = embeddings.new_empty(batch_size, max(block_count - 1, 0), self.config.d_model)
        bank_read_blocks = 0
        bank_read_slots = 0
        refresh_norm_sum = embeddings.new_zeros(())
        refresh_norm_count = 0
        detail_selected_counts = []
        detail_topk_used = []
        detail_candidate_counts = []
        detail_gate_sum = embeddings.new_zeros(())
        detail_gate_count = 0
        refresh_write_index = 0
        detail_write_index = 0
        sufficiency_index = 0
        pre_hidden_blocks = self._apply_local_stack_parallel(block_embeddings, self.pre_blocks)

        for block_index in range(block_count):
            hidden_states = pre_hidden_blocks[:, block_index, :, :]
            if carry_context is not None and not self.config.upper_layer_only_refresh:
                hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)

            detail_context = hidden_states.new_zeros(batch_size, self.config.d_model)
            detail_trace = {
                "detail_used": False,
                "detail_candidate_count": 0,
                "detail_topk_used": 0,
            }
            if self.config.detail_enabled:
                pooled_hidden = hidden_states.mean(dim=1)
                past_detail_keys = detail_key_blocks[:, :detail_write_index, :]
                past_detail_values = detail_value_blocks[:, :detail_write_index, :]
                if torch.is_grad_enabled():
                    past_detail_keys = past_detail_keys.clone()
                    past_detail_values = past_detail_values.clone()
                detail_context, detail_trace = self._retrieve_detail_context(
                    pooled_hidden,
                    past_detail_keys,
                    past_detail_values,
                )
            else:
                pooled_hidden = hidden_states.mean(dim=1)

            fused_context, gate = self._fuse_long_context(carry_context, detail_context, pooled_hidden)
            if fused_context is not None:
                hidden_states = hidden_states + self.carry_to_post(fused_context).unsqueeze(1)

            for block in self.post_blocks:
                hidden_states = block(hidden_states)

            output_blocks[:, block_index, :, :] = hidden_states
            detail_gate_sum = detail_gate_sum + gate.mean().detach()
            detail_gate_count += 1
            detail_candidate_counts.append(detail_trace["detail_candidate_count"])
            detail_topk_used.append(detail_trace["detail_topk_used"])

            refresh_due = self.config.refresh_enabled and (block_index + 1) % self.config.refresh_interval_blocks == 0
            if refresh_due:
                refresh_inputs = self._build_refresh_inputs(hidden_states)
                refreshed, refresh_trace = self.refresh_block(refresh_inputs, self.long_bank.read(bank_states))
                next_refresh_index = refresh_write_index + refreshed.size(1)
                refresh_blocks[:, refresh_write_index:next_refresh_index, :] = refreshed
                refresh_write_index = next_refresh_index
                bank_read_slots += refresh_trace["bank_read_slots"]
                bank_read_blocks += int(refresh_trace["bank_used"])
                carry_context = self._write_bank_entry(refreshed)
                if self.config.refresh_write_gate_enabled:
                    if self.bank_write_importance is None:
                        raise RuntimeError("refresh write gating requires bank_write_importance module")
                    gate_score = torch.sigmoid(self.bank_write_importance(refreshed).mean(dim=1))
                    carry_context = gate_score * carry_context
                refresh_norm_sum = refresh_norm_sum + carry_context.norm(dim=-1).mean().detach()
                refresh_norm_count += 1
                bank_entry = self.bank_entry_proj(carry_context).unsqueeze(1)
                bank_states = self.long_bank.write(bank_states, bank_entry)

                if block_index + 1 < block_count:
                    sufficiency_predictions[:, sufficiency_index, :] = self.sufficiency_head(carry_context)
                    sufficiency_targets[:, sufficiency_index, :] = precomputed_targets[:, block_index + 1, :]
                    sufficiency_index += 1
            elif not self.config.refresh_enabled:
                carry_context = None

            if self.config.detail_enabled:
                detail_slots, detail_positions = self._select_detail_slots(hidden_states)
                detail_selected_counts.append(int(detail_slots.size(1)))
                if detail_slots.size(1) > 0:
                    next_detail_index = detail_write_index + detail_slots.size(1)
                    detail_blocks[:, detail_write_index:next_detail_index, :] = detail_slots.detach()
                    detail_keys = self.detail_key_proj(detail_slots).detach()
                    detail_values = self.detail_value_proj(detail_slots).detach()
                    detail_key_blocks[:, detail_write_index:next_detail_index, :] = detail_keys
                    detail_value_blocks[:, detail_write_index:next_detail_index, :] = detail_values
                    detail_write_index = next_detail_index
            else:
                detail_selected_counts.append(0)

        hidden_states = output_blocks.view(batch_size, seq_len, self.config.d_model)
        logits = self.lm_head(self.final_norm(hidden_states))
        empty = self._empty_state(batch_size, input_ids.device)
        refresh_states = refresh_blocks[:, :refresh_write_index, :].detach() if refresh_write_index > 0 else empty
        detail_states = detail_blocks[:, :detail_write_index, :].detach() if detail_write_index > 0 else empty
        predicted_summary = (
            sufficiency_predictions[:, :sufficiency_index, :]
            if sufficiency_index > 0
            else empty
        )
        target_summary = (
            sufficiency_targets[:, :sufficiency_index, :]
            if sufficiency_index > 0
            else empty
        )
        refresh_norm_mean = 0.0
        if refresh_norm_count > 0:
            refresh_norm_mean = float((refresh_norm_sum / refresh_norm_count).cpu().item())
        detail_gate_mean = 0.0
        if detail_gate_count > 0:
            detail_gate_mean = float((detail_gate_sum / detail_gate_count).cpu().item())

        return {
            "logits": logits,
            "hidden_states": empty,
            "refresh_states": refresh_states,
            "detail_states": detail_states,
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
                "detail_gate_mean": detail_gate_mean,
                "refresh_norm_mean": refresh_norm_mean,
            },
        }
