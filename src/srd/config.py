"""Configuration utilities for the first end-to-end SRD experimental path."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SRDConfig:
    """Holds the minimal SRD settings needed for model, train, and eval runs."""

    model_family: str = ""
    size_name: str = ""
    target_parameter_count: int = 0
    model_type: str = "srd"
    vocab_size: int = 64
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: float = 4.0
    max_seq_len: int = 0
    block_size: int = 8
    segment_length: int = 8
    local_window: int = 4
    refresh_interval_blocks: int = 1
    refresh_slots: int = 2
    refresh_slots_max: int = 2
    refresh_count: int = 2
    refresh_dim: int | None = None
    refresh_gate_temperature: float = 1.0
    refresh_gate_hard: bool = False
    refresh_gate_topk: int = 0
    refresh_role_scheme: str = "shared"
    refresh_key_slot: bool = False
    refresh_value_slot: bool = False
    refresh_rule_slot: bool = False
    refresh_write_gate_enabled: bool = False
    detail_enabled: bool = False
    detail_slots: int = 4
    detail_topk: int = 2
    detail_scoring: str = "dot"
    detail_gate_enabled: bool = True
    detail_anchor_first: bool = True
    detail_anchor_last: bool = True
    detail_saliency_slots: int = 2
    memory_blocks: int = 4
    latent_slots: int = 8
    bank_size: int = 4
    num_local_layers_pre: int = 1
    num_local_layers_post: int = 1
    upper_layer_only_refresh: bool = True
    refresh_enabled: bool = True
    use_refresh: bool = True
    sufficiency_loss_weight: float = 0.0
    beta_budget: float = 0.0
    gamma_gate_entropy: float = 0.0
    memory_keep_last_n_segments: int = 0
    memory_read_mode: str = "slot_query_summary"
    memory_read_every_n_layers: int = 1
    bank_write_policy: str = "mean"
    bank_merge_policy: str = "oldest_pair"
    dropout_p: float = 0.0
    causal: bool = True
    debug_mode: bool = False

    def validate(self) -> None:
        """Checks core shape and routing assumptions."""
        valid_model_types = {
            "srd",
            "srd_block_refresh",
            "srd_block_refresh_detail",
            "adaptive_slot_srd",
            "transformer_local",
            "transformer_full",
            "transformer_xl_style",
            "perceiver_latent",
            "summary_memory",
        }
        if self.model_type not in valid_model_types:
            raise ValueError(f"model_type must be one of {sorted(valid_model_types)}")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.segment_length <= 0:
            raise ValueError("segment_length must be positive")
        if self.local_window <= 0:
            raise ValueError("local_window must be positive")
        if self.refresh_interval_blocks <= 0:
            raise ValueError("refresh_interval_blocks must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.refresh_slots <= 0:
            raise ValueError("refresh_slots must be positive")
        if self.refresh_slots_max <= 0:
            raise ValueError("refresh_slots_max must be positive")
        if self.refresh_count <= 0:
            raise ValueError("refresh_count must be positive")
        if self.refresh_gate_temperature <= 0:
            raise ValueError("refresh_gate_temperature must be positive")
        if self.refresh_gate_topk < 0:
            raise ValueError("refresh_gate_topk must be non-negative")
        if self.refresh_role_scheme not in {"shared", "typed"}:
            raise ValueError("refresh_role_scheme must be 'shared' or 'typed'")
        if self.memory_keep_last_n_segments < 0:
            raise ValueError("memory_keep_last_n_segments must be non-negative")
        if self.memory_read_mode not in {"slot_query_summary", "pooled"}:
            raise ValueError("memory_read_mode must be 'slot_query_summary' or 'pooled'")
        if self.memory_read_every_n_layers <= 0:
            raise ValueError("memory_read_every_n_layers must be positive")
        if self.bank_write_policy not in {"mean", "importance_weighted"}:
            raise ValueError("bank_write_policy must be 'mean' or 'importance_weighted'")
        if self.bank_merge_policy not in {"oldest_pair", "lowest_importance_pair"}:
            raise ValueError("bank_merge_policy must be 'oldest_pair' or 'lowest_importance_pair'")
        if self.detail_slots <= 0:
            raise ValueError("detail_slots must be positive")
        if self.detail_topk < 0:
            raise ValueError("detail_topk must be non-negative")
        if self.detail_saliency_slots < 0:
            raise ValueError("detail_saliency_slots must be non-negative")
        if self.memory_blocks <= 0:
            raise ValueError("memory_blocks must be positive")
        if self.latent_slots <= 0:
            raise ValueError("latent_slots must be positive")
        if self.bank_size <= 0:
            raise ValueError("bank_size must be positive")
        if self.num_local_layers_pre < 0 or self.num_local_layers_post < 0:
            raise ValueError("local layer counts must be non-negative")
        if self.refresh_dim is not None and self.refresh_dim != self.d_model:
            raise ValueError("refresh_dim must be None or equal to d_model in the first implementation")

    def effective_block_size(self) -> int:
        """Returns the block size used by the configured model."""
        if self.model_type in {"srd_block_refresh", "srd_block_refresh_detail", "adaptive_slot_srd"}:
            return self.block_size
        return self.segment_length

    def effective_refresh_slots(self) -> int:
        """Returns the active refresh-slot count for the configured model."""
        if self.model_type == "adaptive_slot_srd":
            return self.refresh_slots_max
        if self.model_type in {"srd_block_refresh", "srd_block_refresh_detail"}:
            return self.refresh_slots
        return self.refresh_count

    def to_dict(self) -> dict:
        """Converts the config into a JSON-serializable dictionary."""
        return asdict(self)

    def experiment_name(self) -> str:
        """Returns a compact label for the current baseline or SRD variant."""
        if self.model_type == "srd_block_refresh":
            if not self.refresh_enabled:
                return "local_only"
            if self.sufficiency_loss_weight > 0:
                return "refresh_with_sufficiency"
            return "refresh_no_sufficiency"
        if self.model_type == "adaptive_slot_srd":
            return "adaptive_slot_srd"
        if self.model_type == "srd_block_refresh_detail":
            if self.sufficiency_loss_weight > 0:
                return "refresh_with_detail"
            return "refresh_detail_no_sufficiency"
        if self.model_type != "srd":
            return self.model_type
        if not self.use_refresh:
            return "local_only"
        if self.sufficiency_loss_weight > 0:
            return "srd_with_sufficiency"
        return "srd_without_sufficiency"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SRDConfig":
        """Builds a config from a plain dictionary."""
        values = dict(values)
        if "block_size" not in values:
            values["block_size"] = values.get("segment_length", cls.block_size)
        if "segment_length" not in values:
            values["segment_length"] = values.get("block_size", cls.segment_length)
        if "refresh_slots" not in values:
            values["refresh_slots"] = values.get("refresh_slots_max", values.get("refresh_count", cls.refresh_slots))
        if "refresh_slots_max" not in values:
            values["refresh_slots_max"] = values.get("refresh_slots", cls.refresh_slots_max)
        if "refresh_count" not in values:
            values["refresh_count"] = values.get("refresh_slots", cls.refresh_count)
        if "refresh_enabled" not in values:
            values["refresh_enabled"] = values.get("use_refresh", cls.refresh_enabled)
        if "use_refresh" not in values:
            values["use_refresh"] = values.get("refresh_enabled", cls.use_refresh)
        return cls(**values)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "SRDConfig":
        """Loads a config from a JSON file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def preset(cls, name: str) -> "SRDConfig":
        """Returns a named tiny preset used by the initial experiments."""
        presets = {
            "local_tiny": cls(
                model_type="transformer_local",
                block_size=8,
                use_refresh=False,
                refresh_enabled=False,
                sufficiency_loss_weight=0.0,
                upper_layer_only_refresh=True,
                num_layers=4,
            ),
            "srd_tiny": cls(
                model_type="srd",
                block_size=8,
                use_refresh=True,
                refresh_enabled=True,
                sufficiency_loss_weight=0.0,
                upper_layer_only_refresh=True,
            ),
            "srd_suf_tiny": cls(
                model_type="srd",
                block_size=8,
                use_refresh=True,
                refresh_enabled=True,
                sufficiency_loss_weight=0.25,
                upper_layer_only_refresh=True,
            ),
            "srd_all_layers_tiny": cls(
                model_type="srd",
                block_size=8,
                use_refresh=True,
                refresh_enabled=True,
                sufficiency_loss_weight=0.25,
                upper_layer_only_refresh=False,
            ),
            "transformer_local_tiny": cls(
                model_type="transformer_local",
                block_size=8,
                use_refresh=False,
                refresh_enabled=False,
                sufficiency_loss_weight=0.0,
                upper_layer_only_refresh=True,
                num_layers=4,
            ),
            "transformer_full_tiny": cls(
                model_type="transformer_full",
                block_size=8,
                use_refresh=False,
                refresh_enabled=False,
                sufficiency_loss_weight=0.0,
                upper_layer_only_refresh=True,
                num_layers=4,
            ),
            "summary_memory_tiny": cls(
                model_type="summary_memory",
                block_size=8,
                use_refresh=False,
                refresh_enabled=False,
                sufficiency_loss_weight=0.0,
                upper_layer_only_refresh=True,
                num_layers=4,
            ),
            "transformer_xl_style_tiny": cls(
                model_type="transformer_xl_style",
                block_size=8,
                segment_length=8,
                use_refresh=False,
                refresh_enabled=False,
                sufficiency_loss_weight=0.0,
                upper_layer_only_refresh=True,
                num_layers=4,
                memory_blocks=4,
            ),
            "perceiver_latent_tiny": cls(
                model_type="perceiver_latent",
                block_size=8,
                segment_length=8,
                use_refresh=False,
                refresh_enabled=False,
                sufficiency_loss_weight=0.0,
                upper_layer_only_refresh=True,
                num_layers=4,
                latent_slots=8,
            ),
            "block_refresh_local_tiny": cls(
                model_type="srd_block_refresh",
                block_size=8,
                segment_length=8,
                refresh_slots=2,
                refresh_count=2,
                refresh_enabled=False,
                use_refresh=False,
                sufficiency_loss_weight=0.0,
                upper_layer_only_refresh=True,
            ),
            "block_refresh_tiny": cls(
                model_type="srd_block_refresh",
                block_size=8,
                segment_length=8,
                refresh_slots=2,
                refresh_count=2,
                refresh_enabled=True,
                use_refresh=True,
                sufficiency_loss_weight=0.0,
                upper_layer_only_refresh=True,
            ),
            "block_refresh_suf_tiny": cls(
                model_type="srd_block_refresh",
                block_size=8,
                segment_length=8,
                refresh_slots=2,
                refresh_count=2,
                refresh_enabled=True,
                use_refresh=True,
                sufficiency_loss_weight=0.25,
                upper_layer_only_refresh=True,
            ),
            "block_refresh_detail_tiny": cls(
                model_type="srd_block_refresh_detail",
                block_size=8,
                segment_length=8,
                refresh_slots=2,
                refresh_count=2,
                refresh_enabled=True,
                use_refresh=True,
                detail_enabled=True,
                detail_slots=4,
                detail_topk=2,
                detail_saliency_slots=2,
                sufficiency_loss_weight=0.25,
                upper_layer_only_refresh=True,
            ),
            "block_refresh_detail_no_suf_tiny": cls(
                model_type="srd_block_refresh_detail",
                block_size=8,
                segment_length=8,
                refresh_slots=2,
                refresh_count=2,
                refresh_enabled=True,
                use_refresh=True,
                detail_enabled=True,
                detail_slots=4,
                detail_topk=2,
                detail_saliency_slots=2,
                sufficiency_loss_weight=0.0,
                upper_layer_only_refresh=True,
            ),
            "adaptive_slot_srd_tiny": cls(
                model_type="adaptive_slot_srd",
                block_size=8,
                segment_length=8,
                refresh_slots=4,
                refresh_slots_max=4,
                refresh_count=4,
                bank_size=16,
                refresh_enabled=True,
                use_refresh=True,
                sufficiency_loss_weight=0.25,
                beta_budget=0.01,
                gamma_gate_entropy=0.001,
                refresh_gate_temperature=1.0,
                refresh_gate_hard=False,
                refresh_gate_topk=0,
                memory_keep_last_n_segments=4,
                memory_read_mode="pooled",
                memory_read_every_n_layers=2,
                upper_layer_only_refresh=True,
            ),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}")
        return presets[name]
