"""Tests for the block-refresh detail-memory SRD variant."""

import torch

from srd.config import SRDConfig
from srd.modeling.block_refresh_detail_model import BlockRefreshDetailModel
from srd.modeling.block_refresh_model import BlockRefreshModel
from srd.training.losses import compute_srd_loss


def test_detail_model_outputs_expected_shapes() -> None:
    config = SRDConfig(
        model_type="srd_block_refresh_detail",
        vocab_size=32,
        d_model=16,
        num_heads=4,
        block_size=4,
        segment_length=4,
        refresh_slots=2,
        refresh_count=2,
        detail_enabled=True,
        detail_slots=4,
        detail_topk=2,
        refresh_enabled=True,
        use_refresh=True,
    )
    model = BlockRefreshDetailModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["refresh_states"].shape == (2, 4, config.d_model)
    assert outputs["detail_states"].shape[2] == config.d_model
    assert outputs["debug"]["detail_enabled"] is True


def test_detail_slots_selection_respects_capacity() -> None:
    config = SRDConfig(
        model_type="srd_block_refresh_detail",
        d_model=8,
        num_heads=2,
        detail_enabled=True,
        detail_slots=3,
        detail_topk=2,
        detail_saliency_slots=2,
    )
    model = BlockRefreshDetailModel(config)
    hidden_states = torch.randn(2, 5, config.d_model)
    detail_slots, positions = model._select_detail_slots(hidden_states)

    assert detail_slots.shape[1] <= 3
    assert positions.shape[1] == detail_slots.shape[1]


def test_detail_variant_preserves_base_refresh_param_count_plus_small_overhead() -> None:
    base = BlockRefreshModel(SRDConfig.preset("block_refresh_suf_tiny"))
    detail = BlockRefreshDetailModel(SRDConfig.preset("block_refresh_detail_tiny"))
    base_params = sum(parameter.numel() for parameter in base.parameters())
    detail_params = sum(parameter.numel() for parameter in detail.parameters())

    assert detail_params > base_params
    assert detail_params - base_params < 25000


def test_detail_retrieval_affects_later_blocks_only() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    model = BlockRefreshDetailModel(config)
    input_a = torch.randint(0, config.vocab_size, (1, 16))
    input_b = input_a.clone()
    input_b[:, : config.block_size] = (input_b[:, : config.block_size] + 1) % config.vocab_size

    outputs_a = model(input_a)
    outputs_b = model(input_b)

    assert not torch.allclose(outputs_a["logits"][:, config.block_size :, :], outputs_b["logits"][:, config.block_size :, :])
    assert outputs_a["debug"]["detail_access_count"] >= 0


def test_detail_variant_sufficiency_loss_backpropagates() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    model = BlockRefreshDetailModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    outputs = model(input_ids)
    losses = compute_srd_loss(outputs, input_ids, config)
    losses["loss"].backward()

    assert losses["sufficiency_loss"].item() >= 0.0
    assert model.sufficiency_head.weight.grad is not None
