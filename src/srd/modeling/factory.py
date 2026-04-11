"""Factory for building SRD and conventional baseline model variants."""

from __future__ import annotations

from torch import nn

from srd.config import SRDConfig
from srd.modeling.advanced_baselines import PerceiverLatentModel, TransformerXLStyleMemoryModel
from srd.modeling.baseline_models import SummaryMemoryModel, TransformerFullModel, TransformerLocalModel
from srd.modeling.block_refresh_detail_model import BlockRefreshDetailModel
from srd.modeling.block_refresh_model import BlockRefreshModel
from srd.modeling.srd_model import SRDModel


def build_model(config: SRDConfig) -> nn.Module:
    """Builds the configured SRD or external baseline model."""
    if config.model_type == "srd":
        return SRDModel(config)
    if config.model_type == "srd_block_refresh":
        return BlockRefreshModel(config)
    if config.model_type == "srd_block_refresh_detail":
        return BlockRefreshDetailModel(config)
    if config.model_type == "transformer_local":
        return TransformerLocalModel(config)
    if config.model_type == "transformer_full":
        return TransformerFullModel(config)
    if config.model_type == "summary_memory":
        return SummaryMemoryModel(config)
    if config.model_type == "transformer_xl_style":
        return TransformerXLStyleMemoryModel(config)
    if config.model_type == "perceiver_latent":
        return PerceiverLatentModel(config)
    raise ValueError(f"Unknown model_type: {config.model_type}")
