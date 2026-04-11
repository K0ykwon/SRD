"""Tests for the additional final-comparison baseline variants."""

import torch

from srd.config import SRDConfig
from srd.data.synthetic_benchmarks import SyntheticBenchmarkConfig
from srd.eval.benchmark_runner import build_model_config, canonical_variant_name
from srd.modeling.advanced_baselines import PerceiverLatentModel, TransformerXLStyleMemoryModel
from srd.modeling.block_refresh_model import BlockRefreshModel


def test_transformer_xl_style_outputs_expected_shapes() -> None:
    config = SRDConfig(
        model_type="transformer_xl_style",
        vocab_size=32,
        d_model=16,
        num_heads=4,
        num_layers=4,
        segment_length=4,
        local_window=2,
        memory_blocks=2,
    )
    model = TransformerXLStyleMemoryModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["refresh_states"].numel() == 0
    assert outputs["debug"]["bank_read_slots"] >= 0
    assert outputs["debug"]["memory_blocks"] == 2


def test_perceiver_latent_outputs_expected_shapes() -> None:
    config = SRDConfig(
        model_type="perceiver_latent",
        vocab_size=32,
        d_model=16,
        num_heads=4,
        num_layers=4,
        segment_length=4,
        local_window=2,
        latent_slots=6,
    )
    model = PerceiverLatentModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["refresh_states"].numel() == 0
    assert outputs["bank_states"].shape == (2, 6, config.d_model)
    assert outputs["debug"]["latent_slots"] == 6


def test_build_model_config_supports_new_variants() -> None:
    benchmark_config = SyntheticBenchmarkConfig(
        family="delayed_kv",
        vocab_size=32,
        segment_length=4,
        context_segments=4,
        gap_segments=2,
        local_window=2,
    )
    xl_config = build_model_config("transformer_xl_style", benchmark_config)
    perceiver_config = build_model_config("perceiver_latent", benchmark_config)

    assert xl_config.model_type == "transformer_xl_style"
    assert xl_config.refresh_enabled is False
    assert perceiver_config.model_type == "perceiver_latent"
    assert perceiver_config.refresh_enabled is False


def test_scaled_variant_aliases_map_to_base_variants() -> None:
    assert canonical_variant_name("transformer_full_large") == "transformer_full"
    assert canonical_variant_name("refresh_with_sufficiency_medium") == "refresh_with_sufficiency"
    assert canonical_variant_name("refresh_with_detail_small") == "refresh_with_detail"


def test_srd_block_refresh_still_preserves_refresh_only_constraint() -> None:
    config = SRDConfig(
        model_type="srd_block_refresh",
        vocab_size=32,
        d_model=16,
        num_heads=4,
        num_layers=4,
        segment_length=4,
        refresh_slots=2,
        bank_size=2,
        refresh_enabled=True,
        use_refresh=True,
    )
    model = BlockRefreshModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    outputs = model(input_ids)

    assert outputs["debug"]["token_bank_access_count"] == 0
    assert outputs["debug"]["refresh_bank_access_count"] >= 0
