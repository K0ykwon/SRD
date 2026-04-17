"""Block-based SRD variant with refresh-only global access and an explicit sufficiency path."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from srd.config import SRDConfig
from srd.modeling.local_block import LocalBlock
from srd.modeling.long_bank import LongMemoryBank
from srd.modeling.refresh_block import RefreshBlock


class BlockRefreshModel(nn.Module):
    """Implements the paper-facing block-refresh SRD variant."""

    def __init__(self, config: SRDConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.refresh_seed = nn.Parameter(torch.randn(config.effective_refresh_slots(), config.d_model) * 0.02)
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
        self.long_bank = LongMemoryBank(
            d_model=config.d_model,
            max_slots=config.bank_size,
            merge_policy=config.bank_merge_policy,
        )
        self.block_to_refresh = nn.Linear(2 * config.d_model, config.effective_refresh_slots() * config.d_model)
        if config.refresh_role_scheme == "typed":
            self.summary_to_refresh = nn.Linear(2 * config.d_model, config.d_model)
            self.key_to_refresh = nn.Linear(config.d_model, config.d_model) if config.refresh_key_slot else None
            self.value_to_refresh = nn.Linear(config.d_model, config.d_model) if config.refresh_value_slot else None
            self.rule_to_refresh = nn.Linear(config.d_model, config.d_model) if config.refresh_rule_slot else None
            self.refresh_saliency = nn.Linear(config.d_model, 1)
        else:
            self.summary_to_refresh = None
            self.key_to_refresh = None
            self.value_to_refresh = None
            self.rule_to_refresh = None
            self.refresh_saliency = None
        self.bank_entry_proj = nn.Linear(config.d_model, config.d_model)
        if config.bank_write_policy == "importance_weighted" or config.refresh_write_gate_enabled:
            self.bank_write_importance = nn.Linear(config.d_model, 1)
        else:
            self.bank_write_importance = None
        self.carry_to_pre = nn.Linear(config.d_model, config.d_model)
        self.carry_to_post = nn.Linear(config.d_model, config.d_model)
        self.sufficiency_head = nn.Linear(config.d_model, config.d_model)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _empty_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.empty(batch_size, 0, self.config.d_model, device=device)

    def _block_ranges(self, seq_len: int) -> List[tuple[int, int]]:
        """Returns the contiguous blocks used for scheduled refresh."""
        block_size = self.config.effective_block_size()
        return [
            (start, min(start + block_size, seq_len))
            for start in range(0, seq_len, block_size)
        ]

    def _reshape_blocks(self, hidden_states: Tensor) -> tuple[Tensor, int, int]:
        """Reshapes `[batch, seq, dim]` into `[batch, blocks, block, dim]` for block-parallel work."""
        batch_size, seq_len, d_model = hidden_states.shape
        block_size = self.config.effective_block_size()
        if seq_len % block_size != 0:
            raise ValueError("BlockRefreshModel expects sequence length divisible by block size")
        block_count = seq_len // block_size
        return hidden_states.view(batch_size, block_count, block_size, d_model), block_count, block_size

    def _reshape_block_ids(self, input_ids: Tensor) -> tuple[Tensor, int, int]:
        """Reshapes `[batch, seq]` token ids into `[batch, blocks, block]`."""
        batch_size, seq_len = input_ids.shape
        block_size = self.config.effective_block_size()
        if seq_len % block_size != 0:
            raise ValueError("BlockRefreshModel expects sequence length divisible by block size")
        block_count = seq_len // block_size
        return input_ids.view(batch_size, block_count, block_size), block_count, block_size

    def _apply_local_stack_parallel(self, block_states: Tensor, blocks: nn.ModuleList) -> Tensor:
        """Runs a local block stack over all blocks in parallel by flattening the block axis."""
        if not blocks:
            return block_states
        batch_size, block_count, block_size, d_model = block_states.shape
        flat = block_states.view(batch_size * block_count, block_size, d_model)
        for block in blocks:
            flat = block(flat)
        return flat.view(batch_size, block_count, block_size, d_model)

    def _empty_local_stack_caches(self, blocks: nn.ModuleList) -> list[dict[str, Tensor] | None]:
        """Returns an empty incremental-cache list for a local stack."""
        return [None] * len(blocks)

    def _prefill_local_stack_cache(
        self,
        hidden_states: Tensor,
        blocks: nn.ModuleList,
    ) -> tuple[Tensor, list[dict[str, Tensor] | None]]:
        """Runs a local stack once and captures incremental caches for each layer."""
        if not blocks:
            return hidden_states, []
        caches: list[dict[str, Tensor] | None] = []
        for block in blocks:
            hidden_states, cache = block.prefill_cache(hidden_states)
            caches.append(cache)
        return hidden_states, caches

    def _decode_local_stack_step(
        self,
        hidden_states: Tensor,
        blocks: nn.ModuleList,
        caches: list[dict[str, Tensor] | None] | None,
    ) -> tuple[Tensor, list[dict[str, Tensor] | None]]:
        """Runs one incremental token through a cached local stack."""
        if not blocks:
            return hidden_states, []
        if caches is None or len(caches) != len(blocks):
            caches = self._empty_local_stack_caches(blocks)
        next_caches: list[dict[str, Tensor] | None] = []
        for block, cache in zip(blocks, caches):
            hidden_states, next_cache = block.forward_step(hidden_states, cache)
            next_caches.append(next_cache)
        return hidden_states, next_caches

    def _build_refresh_inputs(self, block_hidden: Tensor) -> Tensor:
        """Builds one or more refresh slots from the current block summary."""
        pooled = block_hidden.mean(dim=1)
        boundary = block_hidden[:, -1, :]
        summary = torch.cat([pooled, boundary], dim=-1)
        if self.config.refresh_role_scheme == "typed":
            if self.summary_to_refresh is None or self.refresh_saliency is None:
                raise RuntimeError("typed refresh role scheme requires typed refresh modules")
            slots = [self.summary_to_refresh(summary)]
            saliency_scores = self.refresh_saliency(block_hidden).squeeze(-1)
            key_like_index = saliency_scores.argmax(dim=1)
            gather_index = key_like_index.view(-1, 1, 1).expand(block_hidden.size(0), 1, self.config.d_model)
            key_like = block_hidden.gather(1, gather_index).squeeze(1)
            if self.config.refresh_key_slot and self.key_to_refresh is not None:
                slots.append(self.key_to_refresh(key_like))
            if self.config.refresh_value_slot and self.value_to_refresh is not None:
                slots.append(self.value_to_refresh(boundary))
            if self.config.refresh_rule_slot and self.rule_to_refresh is not None:
                slots.append(self.rule_to_refresh(pooled))
            while len(slots) < self.config.effective_refresh_slots():
                slots.append(self.summary_to_refresh(summary))
            refresh_inputs = torch.stack(slots[: self.config.effective_refresh_slots()], dim=1)
            return refresh_inputs + self.refresh_seed.unsqueeze(0)
        refresh_inputs = self.block_to_refresh(summary)
        refresh_inputs = refresh_inputs.view(
            block_hidden.size(0),
            self.config.effective_refresh_slots(),
            self.config.d_model,
        )
        return refresh_inputs + self.refresh_seed.unsqueeze(0)

    def _write_bank_entry(self, refreshed: Tensor) -> Tensor:
        """Builds the carried context from refresh slots."""
        if self.config.bank_write_policy == "importance_weighted":
            if self.bank_write_importance is None:
                raise RuntimeError("importance-weighted bank writes require bank_write_importance module")
            weights = torch.softmax(self.bank_write_importance(refreshed).squeeze(-1), dim=1)
            carry_context = torch.sum(weights.unsqueeze(-1) * refreshed, dim=1)
        else:
            carry_context = refreshed.mean(dim=1)
        return carry_context

    def _next_block_target(self, block_ids: Tensor) -> Tensor:
        """Returns the detached next-block summary target for sufficiency training."""
        return self.embedding(block_ids).mean(dim=1).detach()

    def _precompute_next_block_targets(self, input_ids: Tensor) -> Tensor:
        """Returns detached next-block summary targets for every block position."""
        block_ids, _, _ = self._reshape_block_ids(input_ids)
        target_embeddings = self.embedding(block_ids.reshape(input_ids.size(0), -1)).view(
            input_ids.size(0),
            block_ids.size(1),
            block_ids.size(2),
            self.config.d_model,
        )
        return target_embeddings.mean(dim=2).detach()

    def _process_token_block(self, block_ids: Tensor, carry_context: Tensor | None) -> Tensor:
        """Runs one token block without mutating the long-memory state."""
        hidden_states = self.embedding(block_ids)
        for block in self.pre_blocks:
            hidden_states = block(hidden_states)
        if carry_context is not None and not self.config.upper_layer_only_refresh:
            hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)
        if carry_context is not None:
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
    ) -> tuple[Tensor | None, Tensor]:
        """Updates carry and bank state only when a completed block reaches a refresh boundary."""
        refresh_due = self.config.refresh_enabled and completed_blocks % self.config.refresh_interval_blocks == 0
        if refresh_due:
            refresh_inputs = self._build_refresh_inputs(hidden_states)
            refreshed, _ = self.refresh_block(refresh_inputs, self.long_bank.read(bank_states))
            carry_context = self._write_bank_entry(refreshed)
            if self.config.refresh_write_gate_enabled:
                if self.bank_write_importance is None:
                    raise RuntimeError("refresh write gating requires bank_write_importance module")
                gate = torch.sigmoid(self.bank_write_importance(refreshed).mean(dim=1))
                carry_context = gate * carry_context
            bank_entry = self.bank_entry_proj(carry_context).unsqueeze(1)
            bank_states = self.long_bank.write(bank_states, bank_entry)
        elif not self.config.refresh_enabled:
            carry_context = None
        return carry_context, bank_states

    def prefill(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> dict[str, Any]:
        """Builds decode cache state from a prefix without re-running completed blocks per token."""
        batch_size, seq_len = input_ids.shape
        block_size = self.config.effective_block_size()
        completed_blocks = seq_len // block_size
        prefix_len = completed_blocks * block_size
        bank_states = initial_bank_states if initial_bank_states is not None else self.long_bank.empty(batch_size, input_ids.device)
        carry_context = None
        next_logits = None

        for block_index in range(completed_blocks):
            block_ids = input_ids[:, block_index * block_size : (block_index + 1) * block_size]
            hidden_states = self._process_token_block(block_ids, carry_context)
            logits = self.lm_head(self.final_norm(hidden_states))
            next_logits = logits[:, -1, :]
            carry_context, bank_states = self._update_refresh_state(
                hidden_states,
                carry_context,
                bank_states,
                block_index + 1,
            )

        open_block_ids = input_ids[:, prefix_len:]
        open_block_hidden = self._empty_state(batch_size, input_ids.device)
        open_pre_caches = self._empty_local_stack_caches(self.pre_blocks)
        open_post_caches = self._empty_local_stack_caches(self.post_blocks)
        if open_block_ids.size(1) > 0:
            hidden_states = self.embedding(open_block_ids)
            hidden_states, open_pre_caches = self._prefill_local_stack_cache(hidden_states, self.pre_blocks)
            if carry_context is not None and not self.config.upper_layer_only_refresh:
                hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)
            if carry_context is not None:
                hidden_states = hidden_states + self.carry_to_post(carry_context).unsqueeze(1)
            hidden_states, open_post_caches = self._prefill_local_stack_cache(hidden_states, self.post_blocks)
            logits = self.lm_head(self.final_norm(hidden_states))
            next_logits = logits[:, -1, :]
            open_block_hidden = hidden_states.detach()

        if next_logits is None:
            raise ValueError("prefill requires at least one input token")

        return {
            "open_block_hidden": open_block_hidden,
            "open_block_len": int(open_block_hidden.size(1)),
            "open_pre_caches": open_pre_caches,
            "open_post_caches": open_post_caches,
            "completed_blocks": completed_blocks,
            "carry_context": carry_context.detach() if carry_context is not None else None,
            "bank_states": bank_states.detach(),
            "next_logits": next_logits.detach(),
        }

    def decode_step(self, next_input_ids: Tensor, state: dict[str, Any]) -> dict[str, Any]:
        """Appends one token and recomputes only the current open block."""
        if next_input_ids.dim() == 1:
            next_input_ids = next_input_ids.unsqueeze(1)
        carry_context = state["carry_context"]
        bank_states = state["bank_states"]
        completed_blocks = int(state["completed_blocks"])
        open_block_hidden = state["open_block_hidden"]
        open_block_len = int(state["open_block_len"])

        hidden_states = self.embedding(next_input_ids)
        hidden_states, open_pre_caches = self._decode_local_stack_step(
            hidden_states,
            self.pre_blocks,
            state.get("open_pre_caches"),
        )
        if carry_context is not None and not self.config.upper_layer_only_refresh:
            hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)
        if carry_context is not None:
            hidden_states = hidden_states + self.carry_to_post(carry_context).unsqueeze(1)
        hidden_states, open_post_caches = self._decode_local_stack_step(
            hidden_states,
            self.post_blocks,
            state.get("open_post_caches"),
        )
        logits = self.lm_head(self.final_norm(hidden_states))
        next_logits = logits[:, -1, :].detach()
        open_block_hidden = (
            torch.cat([open_block_hidden, hidden_states.detach()], dim=1)
            if open_block_len > 0
            else hidden_states.detach()
        )
        open_block_len += 1

        if open_block_len == self.config.effective_block_size():
            completed_blocks += 1
            carry_context, bank_states = self._update_refresh_state(
                open_block_hidden,
                carry_context,
                bank_states,
                completed_blocks,
            )
            open_block_hidden = self._empty_state(next_input_ids.size(0), next_input_ids.device)
            open_block_len = 0
            open_pre_caches = self._empty_local_stack_caches(self.pre_blocks)
            open_post_caches = self._empty_local_stack_caches(self.post_blocks)

        return {
            "open_block_hidden": open_block_hidden,
            "open_block_len": open_block_len,
            "open_pre_caches": open_pre_caches,
            "open_post_caches": open_post_caches,
            "completed_blocks": completed_blocks,
            "carry_context": carry_context.detach() if carry_context is not None else None,
            "bank_states": bank_states.detach(),
            "next_logits": next_logits,
        }

    def forward(self, input_ids: Tensor, initial_bank_states: Tensor | None = None) -> Dict[str, Tensor]:
        """Runs block-local token processing plus refresh-only global access."""
        batch_size, seq_len = input_ids.shape
        embeddings = self.embedding(input_ids)
        block_embeddings, block_count, block_size = self._reshape_blocks(embeddings)
        precomputed_targets = self._precompute_next_block_targets(input_ids)
        bank_states = initial_bank_states if initial_bank_states is not None else self.long_bank.empty(batch_size, input_ids.device)
        carry_context = None

        output_blocks = embeddings.new_empty(batch_size, block_count, block_size, self.config.d_model)
        refresh_slot_count = sum(
            1 for block_index in range(block_count)
            if self.config.refresh_enabled and (block_index + 1) % self.config.refresh_interval_blocks == 0
        ) * self.config.effective_refresh_slots()
        refresh_blocks = embeddings.new_empty(batch_size, refresh_slot_count, self.config.d_model)
        sufficiency_predictions = embeddings.new_empty(batch_size, max(block_count - 1, 0), self.config.d_model)
        sufficiency_targets = embeddings.new_empty(batch_size, max(block_count - 1, 0), self.config.d_model)
        bank_read_blocks = 0
        bank_read_slots = 0
        refresh_norm_sum = embeddings.new_zeros(())
        refresh_norm_count = 0
        refresh_write_index = 0
        sufficiency_index = 0
        final_carry_context = None
        pre_hidden_blocks = self._apply_local_stack_parallel(block_embeddings, self.pre_blocks)

        for block_index in range(block_count):
            hidden_states = pre_hidden_blocks[:, block_index, :, :]
            if carry_context is not None and not self.config.upper_layer_only_refresh:
                hidden_states = hidden_states + self.carry_to_pre(carry_context).unsqueeze(1)

            if carry_context is not None:
                hidden_states = hidden_states + self.carry_to_post(carry_context).unsqueeze(1)

            for block in self.post_blocks:
                hidden_states = block(hidden_states)

            output_blocks[:, block_index, :, :] = hidden_states

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
                    gate = torch.sigmoid(self.bank_write_importance(refreshed).mean(dim=1))
                    carry_context = gate * carry_context
                final_carry_context = carry_context
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

        hidden_states = output_blocks.view(batch_size, seq_len, self.config.d_model)
        logits = self.lm_head(self.final_norm(hidden_states))
        empty = self._empty_state(batch_size, input_ids.device)
        refresh_states = refresh_blocks[:, :refresh_write_index, :].detach() if refresh_write_index > 0 else empty
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

        return {
            "logits": logits,
            "hidden_states": empty,
            "refresh_states": refresh_states,
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
                "refresh_state_shape": tuple(refresh_states.shape),
                "bank_read_blocks": bank_read_blocks,
                "bank_read_segments": bank_read_blocks,
                "bank_read_slots": bank_read_slots,
                "token_bank_access_count": 0,
                "refresh_bank_access_count": bank_read_slots,
                "refresh_norm_mean": refresh_norm_mean,
            },
        }
