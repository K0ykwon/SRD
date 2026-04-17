"""Adaptive-slot SRD with fixed-shape learned-capacity refresh slots."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from srd.config import SRDConfig
from srd.modeling.block_refresh_model import BlockRefreshModel


class LearnedQueryCrossAttentionPool(nn.Module):
    """Pools a sequence into a fixed learned query set with SDPA."""

    def __init__(self, d_model: int, num_heads: int, query_count: int, dropout_p: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query_count = query_count

        self.learned_queries = nn.Parameter(torch.randn(query_count, d_model) * 0.02)
        self.query_norm = nn.LayerNorm(d_model)
        self.source_norm = nn.LayerNorm(d_model)
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

    def _reshape_heads(self, states: Tensor, batch_size: int, seq_len: int) -> Tensor:
        return states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, source_states: Tensor) -> Tensor:
        """Returns `[batch, query_count, d_model]` pooled slots."""
        if source_states.size(1) == 0:
            return source_states.new_empty(source_states.size(0), self.query_count, self.d_model)

        batch_size = source_states.size(0)
        queries = self.query_norm(self.learned_queries).unsqueeze(0).expand(batch_size, -1, -1)
        source_states = self.source_norm(source_states)

        query = self._reshape_heads(self.query_proj(queries), batch_size, self.query_count)
        key = self._reshape_heads(self.key_proj(source_states), batch_size, source_states.size(1))
        value = self._reshape_heads(self.value_proj(source_states), batch_size, source_states.size(1))

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.query_count, self.d_model)
        pooled = queries + self.out_proj(attended)
        return pooled + self.mlp(self.mlp_norm(pooled))


class AdaptiveSlotSRDModel(BlockRefreshModel):
    """SRD variant with fixed physical refresh slots and learned soft capacity."""

    def __init__(self, config: SRDConfig):
        super().__init__(config)
        query_count = config.effective_refresh_slots()
        memory_query_count = min(2, query_count)
        del self.refresh_block
        del self.block_to_refresh
        if self.summary_to_refresh is not None:
            del self.summary_to_refresh
            self.summary_to_refresh = None
        if self.key_to_refresh is not None:
            del self.key_to_refresh
            self.key_to_refresh = None
        if self.value_to_refresh is not None:
            del self.value_to_refresh
            self.value_to_refresh = None
        if self.rule_to_refresh is not None:
            del self.rule_to_refresh
            self.rule_to_refresh = None
        if self.refresh_saliency is not None:
            del self.refresh_saliency
            self.refresh_saliency = None

        self.refresh_slot_pool = LearnedQueryCrossAttentionPool(
            d_model=config.d_model,
            num_heads=config.num_heads,
            query_count=query_count,
            dropout_p=config.dropout_p,
        )
        self.memory_summary_pool = (
            None
            if config.memory_read_mode == "pooled"
            else LearnedQueryCrossAttentionPool(
                d_model=config.d_model,
                num_heads=config.num_heads,
                query_count=memory_query_count,
                dropout_p=config.dropout_p,
            )
        )
        gate_hidden = max(4, config.d_model // 2)
        self.refresh_gate_norm = nn.LayerNorm(config.d_model)
        self.refresh_gate_mlp = nn.Sequential(
            nn.Linear(config.d_model, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.memory_summary_proj = nn.Linear(config.d_model, config.d_model)

    def _should_apply_pre_context(self) -> bool:
        return not self.config.upper_layer_only_refresh and self.config.memory_read_every_n_layers <= 1

    def _should_apply_post_context(self) -> bool:
        return self.config.memory_read_every_n_layers <= 2

    def _memory_context_active(self) -> bool:
        return self._should_apply_pre_context() or self._should_apply_post_context()

    def _use_incremental_pooled_summary(self) -> bool:
        return self.config.memory_read_mode == "pooled" and self.config.memory_keep_last_n_segments > 0

    def _memory_summary_from_sum(self, slot_sum: Tensor, slot_count: int) -> Tensor:
        return self.memory_summary_proj(slot_sum / max(slot_count, 1))

    def _initialize_memory_state(self, bank_states: Tensor) -> tuple[Tensor | None, int, Tensor | None]:
        if bank_states.size(1) == 0 or not self.config.refresh_enabled:
            return None, 0, None
        if self._use_incremental_pooled_summary():
            slot_sum = bank_states.sum(dim=1)
            return self._memory_summary_from_sum(slot_sum, int(bank_states.size(1))), int(bank_states.size(1)), slot_sum
        carry_context, slot_count = self._read_memory_summary(bank_states)
        return carry_context, slot_count, None

    def _read_memory_summary(self, bank_states: Tensor) -> tuple[Tensor | None, int]:
        """Reads a compact segment-level summary from the dense memory bank."""
        if bank_states.size(1) == 0 or not self.config.refresh_enabled:
            return None, 0
        if self.config.memory_read_mode == "pooled":
            summary = bank_states.mean(dim=1)
        else:
            if self.memory_summary_pool is None:
                raise RuntimeError("slot_query_summary mode requires memory_summary_pool")
            summary_slots = self.memory_summary_pool(bank_states)
            summary = summary_slots.mean(dim=1)
        return self.memory_summary_proj(summary), int(bank_states.size(1))

    def _build_refresh_inputs(self, block_hidden: Tensor) -> Tensor:
        """Pools fixed-shape learned slot queries over the current block."""
        return self.refresh_slot_pool(block_hidden) + self.refresh_seed.unsqueeze(0)

    def _summarize_refresh_slots(self, refresh_slots: Tensor) -> Tensor:
        """Builds the explicit sufficiency path summary from gated refresh slots."""
        if self.config.bank_write_policy == "importance_weighted":
            if self.bank_write_importance is None:
                raise RuntimeError("importance-weighted bank writes require bank_write_importance module")
            weights = torch.softmax(self.bank_write_importance(refresh_slots).squeeze(-1), dim=1)
            return torch.sum(weights.unsqueeze(-1) * refresh_slots, dim=1)
        return refresh_slots.mean(dim=1)

    def _compute_gates(self, raw_refresh: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns gate logits, soft gates, effective gates, and hard masks."""
        gate_logits = self.refresh_gate_mlp(self.refresh_gate_norm(raw_refresh)).squeeze(-1)
        soft_gates = torch.sigmoid(gate_logits / self.config.refresh_gate_temperature)
        if not self.config.refresh_gate_hard:
            return gate_logits, soft_gates, soft_gates, torch.zeros_like(soft_gates)

        if self.config.refresh_gate_topk > 0:
            topk = min(self.config.refresh_gate_topk, soft_gates.size(1))
            topk_indices = soft_gates.topk(topk, dim=1).indices
            hard_mask = torch.zeros_like(soft_gates)
            hard_mask.scatter_(1, topk_indices, 1.0)
        else:
            hard_mask = (soft_gates >= 0.5).to(soft_gates.dtype)
        effective_gates = hard_mask + soft_gates - soft_gates.detach() if self.training else hard_mask
        return gate_logits, soft_gates, effective_gates, hard_mask

    def _write_refresh_slots(
        self,
        bank_states: Tensor,
        gated_refresh: Tensor,
        memory_slot_sum: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, int]:
        """Appends dense refresh slots and updates pooled-summary state when possible."""
        projected_slots = self.bank_entry_proj(gated_refresh).detach()
        if projected_slots.size(1) == 0:
            return bank_states, memory_slot_sum, int(bank_states.size(1))
        if self.config.memory_keep_last_n_segments > 0:
            keep_slots = self.config.memory_keep_last_n_segments * self.config.effective_refresh_slots()
            incoming_slots = min(projected_slots.size(1), keep_slots)
            if bank_states.size(1) == 0 or incoming_slots >= keep_slots:
                updated = projected_slots[:, -keep_slots:, :].contiguous()
                updated_sum = updated.sum(dim=1) if self._use_incremental_pooled_summary() else None
                return updated, updated_sum, int(updated.size(1))
            if bank_states.size(1) == keep_slots:
                updated = bank_states.clone()
                evicted = bank_states[:, :incoming_slots, :]
                updated[:, :-incoming_slots, :] = bank_states[:, incoming_slots:, :]
                updated[:, -incoming_slots:, :] = projected_slots[:, -incoming_slots:, :]
                if self._use_incremental_pooled_summary():
                    if memory_slot_sum is None:
                        memory_slot_sum = bank_states.sum(dim=1)
                    updated_sum = memory_slot_sum - evicted.sum(dim=1) + projected_slots[:, -incoming_slots:, :].sum(dim=1)
                else:
                    updated_sum = None
                return updated, updated_sum, keep_slots
            updated = torch.cat([bank_states, projected_slots], dim=1)[:, -keep_slots:, :].contiguous()
            updated_sum = updated.sum(dim=1) if self._use_incremental_pooled_summary() else None
            return updated, updated_sum, int(updated.size(1))
        updated = self.long_bank.write(bank_states, projected_slots)
        updated_sum = updated.sum(dim=1) if self.config.memory_read_mode == "pooled" and updated.size(1) > 0 else None
        return updated, updated_sum, int(updated.size(1))

    def _process_token_block(self, block_ids: Tensor, carry_context: Tensor | None) -> Tensor:
        """Runs one block with compact memory-summary injection only at configured stages."""
        hidden_states = self.embedding(block_ids)
        for block in self.pre_blocks:
            hidden_states = block(hidden_states)
        if carry_context is not None and self._should_apply_pre_context():
            hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)
        if carry_context is not None and self._should_apply_post_context():
            hidden_states = hidden_states + self.carry_to_post(carry_context).unsqueeze(1)
        for block in self.post_blocks:
            hidden_states = block(hidden_states)
        return hidden_states

    def _update_refresh_state(
        self,
        hidden_states: Tensor,
        carry_context: Tensor | None,
        bank_states: Tensor,
        completed_blocks: int,
        carry_bank_slots: int,
        memory_slot_sum: Tensor | None,
    ) -> tuple[Tensor | None, Tensor, int, Tensor | None]:
        """Writes gated refresh slots and reads the next compact bank summary."""
        refresh_due = self.config.refresh_enabled and completed_blocks % self.config.refresh_interval_blocks == 0
        if refresh_due:
            raw_refresh = self._build_refresh_inputs(hidden_states)
            _, _, effective_gates, _ = self._compute_gates(raw_refresh)
            gated_refresh = effective_gates.unsqueeze(-1) * raw_refresh
            bank_states, memory_slot_sum, carry_bank_slots = self._write_refresh_slots(bank_states, gated_refresh, memory_slot_sum)
            if self._use_incremental_pooled_summary():
                if memory_slot_sum is None:
                    carry_context = None
                    carry_bank_slots = 0
                else:
                    carry_context = self._memory_summary_from_sum(memory_slot_sum, carry_bank_slots)
            else:
                carry_context, carry_bank_slots = self._read_memory_summary(bank_states)
        elif not self.config.refresh_enabled:
            carry_context = None
            carry_bank_slots = 0
            memory_slot_sum = None
        return carry_context, bank_states, carry_bank_slots, memory_slot_sum

    def prefill(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> dict[str, Any]:
        """Builds decode state while keeping bank reads at one compact summary per block."""
        batch_size, seq_len = input_ids.shape
        block_size = self.config.effective_block_size()
        completed_blocks = seq_len // block_size
        prefix_len = completed_blocks * block_size
        bank_states = initial_bank_states if initial_bank_states is not None else self.long_bank.empty(batch_size, input_ids.device)
        carry_context, carry_bank_slots, memory_slot_sum = self._initialize_memory_state(bank_states)
        next_logits = None

        for block_index in range(completed_blocks):
            block_ids = input_ids[:, block_index * block_size : (block_index + 1) * block_size]
            hidden_states = self._process_token_block(block_ids, carry_context)
            logits = self.lm_head(self.final_norm(hidden_states))
            next_logits = logits[:, -1, :]
            carry_context, bank_states, carry_bank_slots, memory_slot_sum = self._update_refresh_state(
                hidden_states,
                carry_context,
                bank_states,
                block_index + 1,
                carry_bank_slots,
                memory_slot_sum,
            )

        open_block_ids = input_ids[:, prefix_len:]
        open_block_hidden = torch.zeros(
            batch_size,
            block_size,
            self.config.d_model,
            device=input_ids.device,
        )
        open_pre_caches = self._empty_local_stack_caches(self.pre_blocks)
        open_post_caches = self._empty_local_stack_caches(self.post_blocks)
        open_block_len = int(open_block_ids.size(1))
        if open_block_ids.size(1) > 0:
            hidden_states = self.embedding(open_block_ids)
            hidden_states, open_pre_caches = self._prefill_local_stack_cache(hidden_states, self.pre_blocks)
            if carry_context is not None and self._should_apply_pre_context():
                hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)
            if carry_context is not None and self._should_apply_post_context():
                hidden_states = hidden_states + self.carry_to_post(carry_context).unsqueeze(1)
            hidden_states, open_post_caches = self._prefill_local_stack_cache(hidden_states, self.post_blocks)
            logits = self.lm_head(self.final_norm(hidden_states))
            next_logits = logits[:, -1, :]
            open_block_hidden[:, :open_block_len, :] = hidden_states.detach()

        if next_logits is None:
            raise ValueError("prefill requires at least one input token")

        return {
            "open_block_hidden": open_block_hidden,
            "open_block_len": open_block_len,
            "open_pre_caches": open_pre_caches,
            "open_post_caches": open_post_caches,
            "completed_blocks": completed_blocks,
            "carry_context": carry_context.detach() if carry_context is not None else None,
            "bank_states": bank_states.detach(),
            "next_logits": next_logits.detach(),
        }

    def decode_step(self, next_input_ids: Tensor, state: dict[str, Any]) -> dict[str, Any]:
        """Appends one token while preserving compact memory-summary conditioning."""
        if next_input_ids.dim() == 1:
            next_input_ids = next_input_ids.unsqueeze(1)
        carry_context = state["carry_context"]
        bank_states = state["bank_states"]
        completed_blocks = int(state["completed_blocks"])
        open_block_hidden = state["open_block_hidden"]
        open_block_len = int(state["open_block_len"])
        carry_bank_slots = int(state.get("carry_bank_slots", bank_states.size(1)))
        memory_slot_sum = state.get("memory_slot_sum")

        hidden_states = self.embedding(next_input_ids)
        hidden_states, open_pre_caches = self._decode_local_stack_step(
            hidden_states,
            self.pre_blocks,
            state.get("open_pre_caches"),
        )
        if carry_context is not None and self._should_apply_pre_context():
            hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)
        if carry_context is not None and self._should_apply_post_context():
            hidden_states = hidden_states + self.carry_to_post(carry_context).unsqueeze(1)
        hidden_states, open_post_caches = self._decode_local_stack_step(
            hidden_states,
            self.post_blocks,
            state.get("open_post_caches"),
        )
        logits = self.lm_head(self.final_norm(hidden_states))
        next_logits = logits[:, -1, :].detach()
        next_open_block_hidden = open_block_hidden.clone()
        next_open_block_hidden[:, open_block_len : open_block_len + 1, :] = hidden_states.detach()
        open_block_len += 1

        if open_block_len == self.config.effective_block_size():
            completed_blocks += 1
            carry_context, bank_states, carry_bank_slots, memory_slot_sum = self._update_refresh_state(
                next_open_block_hidden,
                carry_context,
                bank_states,
                completed_blocks,
                carry_bank_slots,
                memory_slot_sum,
            )
            next_open_block_hidden = torch.zeros_like(next_open_block_hidden)
            open_block_len = 0
            open_pre_caches = self._empty_local_stack_caches(self.pre_blocks)
            open_post_caches = self._empty_local_stack_caches(self.post_blocks)

        return {
            "open_block_hidden": next_open_block_hidden,
            "open_block_len": open_block_len,
            "open_pre_caches": open_pre_caches,
            "open_post_caches": open_post_caches,
            "completed_blocks": completed_blocks,
            "carry_context": carry_context.detach() if carry_context is not None else None,
            "carry_bank_slots": carry_bank_slots,
            "bank_states": bank_states.detach(),
            "memory_slot_sum": memory_slot_sum.detach() if memory_slot_sum is not None else None,
            "next_logits": next_logits,
        }

    def forward(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> Dict[str, Tensor]:
        """Runs adaptive fixed-shape refresh with soft logical slot capacity."""
        batch_size, seq_len = input_ids.shape
        embeddings = self.embedding(input_ids)
        block_embeddings, block_count, block_size = self._reshape_blocks(embeddings)
        precomputed_targets = self._precompute_next_block_targets(input_ids)
        bank_states = initial_bank_states if initial_bank_states is not None else self.long_bank.empty(batch_size, input_ids.device)
        carry_context, carry_bank_slots, memory_slot_sum = self._initialize_memory_state(bank_states)

        output_blocks = embeddings.new_empty(batch_size, block_count, block_size, self.config.d_model)
        trace_base = embeddings.detach()
        refresh_event_count = sum(
            1
            for block_index in range(block_count)
            if self.config.refresh_enabled and (block_index + 1) % self.config.refresh_interval_blocks == 0
        )
        refresh_slots = self.config.effective_refresh_slots()
        refresh_blocks = trace_base.new_empty(batch_size, refresh_event_count * refresh_slots, self.config.d_model)
        gate_logits = trace_base.new_empty(batch_size, refresh_event_count, refresh_slots)
        soft_gates = trace_base.new_empty(batch_size, refresh_event_count, refresh_slots)
        hard_gates = trace_base.new_empty(batch_size, refresh_event_count, refresh_slots)
        sufficiency_predictions = embeddings.new_empty(batch_size, max(block_count - 1, 0), self.config.d_model)
        sufficiency_targets = embeddings.new_empty(batch_size, max(block_count - 1, 0), self.config.d_model)

        bank_read_blocks = 0
        bank_read_slots = 0
        peak_memory_bank_slots = int(bank_states.size(1))
        refresh_norm_sum = embeddings.new_zeros(())
        refresh_norm_count = 0
        budget_loss_sum = embeddings.new_zeros(())
        gate_entropy_loss_sum = embeddings.new_zeros(())
        slot_utilization_sum = trace_base.new_zeros(refresh_slots)
        hard_slot_count_sum = 0.0
        refresh_write_index = 0
        refresh_event_index = 0
        sufficiency_index = 0

        pre_hidden_blocks = self._apply_local_stack_parallel(block_embeddings, self.pre_blocks)

        for block_index in range(block_count):
            hidden_states = pre_hidden_blocks[:, block_index, :, :]
            if carry_context is not None and self._memory_context_active() and not self._use_incremental_pooled_summary():
                bank_read_blocks += 1
                bank_read_slots += carry_bank_slots
            if carry_context is not None and self._should_apply_pre_context():
                hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)
            if carry_context is not None and self._should_apply_post_context():
                hidden_states = hidden_states + self.carry_to_post(carry_context).unsqueeze(1)
            for block in self.post_blocks:
                hidden_states = block(hidden_states)
            output_blocks[:, block_index, :, :] = hidden_states

            refresh_due = self.config.refresh_enabled and (block_index + 1) % self.config.refresh_interval_blocks == 0
            if refresh_due:
                raw_refresh = self._build_refresh_inputs(hidden_states)
                block_gate_logits, block_soft_gates, block_effective_gates, block_hard_gates = self._compute_gates(raw_refresh)
                clipped_gates = block_soft_gates.clamp(1e-6, 1.0 - 1e-6)
                gate_entropy = -(
                    clipped_gates * clipped_gates.log()
                    + (1.0 - clipped_gates) * (1.0 - clipped_gates).log()
                )
                budget_loss_sum = budget_loss_sum + block_soft_gates.sum(dim=-1).mean()
                gate_entropy_loss_sum = gate_entropy_loss_sum - gate_entropy.mean()
                slot_utilization_sum = slot_utilization_sum + block_soft_gates.detach().mean(dim=0)
                hard_slot_count_sum += float(block_hard_gates.detach().sum(dim=-1).mean().cpu())
                gated_refresh = block_effective_gates.unsqueeze(-1) * raw_refresh
                next_refresh_index = refresh_write_index + gated_refresh.size(1)
                refresh_blocks[:, refresh_write_index:next_refresh_index, :] = gated_refresh.detach()
                gate_logits[:, refresh_event_index, :] = block_gate_logits.detach()
                soft_gates[:, refresh_event_index, :] = block_soft_gates.detach()
                hard_gates[:, refresh_event_index, :] = block_hard_gates.detach()
                refresh_write_index = next_refresh_index
                refresh_event_index += 1

                summary = self._summarize_refresh_slots(gated_refresh)
                refresh_norm_sum = refresh_norm_sum + summary.norm(dim=-1).mean().detach()
                refresh_norm_count += 1

                bank_states, memory_slot_sum, carry_bank_slots = self._write_refresh_slots(
                    bank_states,
                    gated_refresh,
                    memory_slot_sum,
                )
                peak_memory_bank_slots = max(peak_memory_bank_slots, int(bank_states.size(1)))
                if self._use_incremental_pooled_summary():
                    if memory_slot_sum is None:
                        carry_context = None
                        carry_bank_slots = 0
                    else:
                        carry_context = self._memory_summary_from_sum(memory_slot_sum, carry_bank_slots)
                else:
                    carry_context, carry_bank_slots = self._read_memory_summary(bank_states)

                if block_index + 1 < block_count:
                    sufficiency_predictions[:, sufficiency_index, :] = self.sufficiency_head(summary)
                    sufficiency_targets[:, sufficiency_index, :] = precomputed_targets[:, block_index + 1, :]
                    sufficiency_index += 1
            elif not self.config.refresh_enabled:
                carry_context = None
                carry_bank_slots = 0

        hidden_states = output_blocks.view(batch_size, seq_len, self.config.d_model)
        logits = self.lm_head(self.final_norm(hidden_states))
        empty_state = self._empty_state(batch_size, input_ids.device)
        empty_gate_state = trace_base.new_empty(batch_size, 0, refresh_slots)
        refresh_states = refresh_blocks[:, :refresh_write_index, :].detach() if refresh_write_index > 0 else empty_state
        predicted_summary = sufficiency_predictions[:, :sufficiency_index, :] if sufficiency_index > 0 else empty_state
        target_summary = sufficiency_targets[:, :sufficiency_index, :] if sufficiency_index > 0 else empty_state
        soft_gate_state = soft_gates[:, :refresh_event_index, :] if refresh_event_index > 0 else empty_gate_state
        hard_gate_state = hard_gates[:, :refresh_event_index, :] if refresh_event_index > 0 else empty_gate_state
        gate_logit_state = gate_logits[:, :refresh_event_index, :] if refresh_event_index > 0 else empty_gate_state

        average_active_soft_slots = 0.0
        average_gate_value = 0.0
        average_active_hard_slots = 0.0
        slot_histogram: list[float] = []
        refresh_budget_loss = embeddings.new_zeros(())
        refresh_gate_entropy_loss = embeddings.new_zeros(())
        if refresh_event_index > 0:
            refresh_budget_loss = budget_loss_sum / refresh_event_index
            refresh_gate_entropy_loss = gate_entropy_loss_sum / refresh_event_index
            average_active_soft_slots = float(refresh_budget_loss.detach().cpu())
            average_gate_value = float((slot_utilization_sum / refresh_event_index).mean().cpu())
            average_active_hard_slots = hard_slot_count_sum / refresh_event_index
            slot_histogram = (slot_utilization_sum / refresh_event_index).cpu().tolist()
        refresh_norm_mean = 0.0
        if refresh_norm_count > 0:
            refresh_norm_mean = float((refresh_norm_sum / refresh_norm_count).cpu().item())

        debug: dict[str, Any] = {
            "block_count": block_count,
            "segment_count": block_count,
            "block_size": self.config.effective_block_size(),
            "refresh_slots": refresh_slots,
            "refresh_slots_max": refresh_slots,
            "refresh_enabled": self.config.refresh_enabled,
            "refresh_state_shape": tuple(refresh_states.shape),
            "memory_read_mode": self.config.memory_read_mode,
            "memory_read_every_n_layers": self.config.memory_read_every_n_layers,
            "memory_keep_last_n_segments": self.config.memory_keep_last_n_segments,
            "refresh_gate_hard": self.config.refresh_gate_hard,
            "refresh_gate_topk": self.config.refresh_gate_topk,
            "bank_read_blocks": bank_read_blocks,
            "bank_read_segments": bank_read_blocks,
            "bank_read_slots": bank_read_slots,
            "memory_bank_slots_used": float(bank_states.size(1)),
            "peak_memory_bank_slots_used": float(peak_memory_bank_slots),
            "token_bank_access_count": 0,
            "refresh_bank_access_count": bank_read_slots,
            "refresh_norm_mean": refresh_norm_mean,
            "average_active_soft_slots_per_segment": average_active_soft_slots,
            "average_gate_value": average_gate_value,
            "average_active_hard_slots": average_active_hard_slots,
            "slot_utilization_histogram": slot_histogram,
        }
        for slot_index, slot_value in enumerate(slot_histogram):
            debug[f"slot_utilization_{slot_index}"] = slot_value

        return {
            "logits": logits,
            "hidden_states": empty_state,
            "refresh_states": refresh_states,
            "refresh_gates": soft_gate_state,
            "soft_refresh_gates": soft_gate_state,
            "hard_refresh_gates": hard_gate_state,
            "refresh_gate_logits": gate_logit_state,
            "refresh_budget_loss": refresh_budget_loss,
            "refresh_gate_entropy_loss": refresh_gate_entropy_loss,
            "bank_states": bank_states.detach(),
            "predicted_summary": predicted_summary,
            "target_summary": target_summary,
            "debug": debug,
        }
