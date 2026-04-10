"""Tests for the paper-facing block-refresh SRD variant."""

import torch

from srd.config import SRDConfig
from srd.modeling.block_refresh_model import BlockRefreshModel
from srd.training.losses import compute_srd_loss


def test_block_refresh_model_outputs_expected_shapes() -> None:
    config = SRDConfig(
        model_type="srd_block_refresh",
        vocab_size=32,
        d_model=16,
        num_heads=4,
        block_size=4,
        segment_length=4,
        refresh_slots=2,
        refresh_count=2,
        refresh_enabled=True,
        use_refresh=True,
    )
    model = BlockRefreshModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["refresh_states"].shape == (2, 4, config.d_model)
    assert outputs["debug"]["block_count"] == 2
    assert outputs["debug"]["block_size"] == 4
    assert outputs["debug"]["refresh_slots"] == 2


def test_block_refresh_enforces_refresh_only_memory_access() -> None:
    config = SRDConfig(
        model_type="srd_block_refresh",
        vocab_size=32,
        d_model=16,
        num_heads=4,
        block_size=4,
        segment_length=4,
        refresh_slots=2,
        refresh_count=2,
        refresh_enabled=True,
        use_refresh=True,
    )
    model = BlockRefreshModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    initial_bank_a = torch.zeros(1, 2, config.d_model)
    initial_bank_b = torch.randn(1, 2, config.d_model)

    outputs_a = model(input_ids, initial_bank_states=initial_bank_a)
    outputs_b = model(input_ids, initial_bank_states=initial_bank_b)

    assert torch.allclose(outputs_a["logits"][:, : config.block_size, :], outputs_b["logits"][:, : config.block_size, :])
    assert not torch.allclose(outputs_a["logits"][:, config.block_size :, :], outputs_b["logits"][:, config.block_size :, :])
    assert outputs_a["debug"]["token_bank_access_count"] == 0
    assert outputs_a["debug"]["refresh_bank_access_count"] > 0


def test_block_refresh_local_only_variant_keeps_params_and_disables_refresh_reads() -> None:
    local_config = SRDConfig.preset("block_refresh_local_tiny")
    refresh_config = SRDConfig.preset("block_refresh_tiny")
    local_model = BlockRefreshModel(local_config)
    refresh_model = BlockRefreshModel(refresh_config)
    input_ids = torch.randint(0, local_config.vocab_size, (1, 8))
    outputs = local_model(input_ids)

    local_params = sum(parameter.numel() for parameter in local_model.parameters())
    refresh_params = sum(parameter.numel() for parameter in refresh_model.parameters())

    assert local_params == refresh_params
    assert outputs["refresh_states"].numel() == 0
    assert outputs["debug"]["refresh_enabled"] is False
    assert outputs["debug"]["refresh_bank_access_count"] == 0


def test_block_refresh_sufficiency_loss_is_non_trivial() -> None:
    config = SRDConfig.preset("block_refresh_suf_tiny")
    model = BlockRefreshModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    outputs = model(input_ids)
    losses = compute_srd_loss(outputs, input_ids, config)
    losses["loss"].backward()

    assert losses["sufficiency_loss"].item() >= 0.0
    assert model.sufficiency_head.weight.grad is not None
