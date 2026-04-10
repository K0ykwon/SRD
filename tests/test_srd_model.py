"""Tests for end-to-end SRD model wiring and routing invariants."""

import torch

from srd.config import SRDConfig
from srd.modeling.srd_model import SRDModel


def test_srd_model_outputs_expected_shapes() -> None:
    config = SRDConfig(vocab_size=32, d_model=16, num_heads=4, segment_length=4, refresh_count=2)
    model = SRDModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["refresh_states"].shape == (2, 4, config.d_model)
    assert outputs["bank_states"].shape[1] <= config.bank_size
    assert outputs["debug"]["segment_count"] == 2


def test_regular_tokens_do_not_directly_read_bank() -> None:
    config = SRDConfig(vocab_size=32, d_model=16, num_heads=4, segment_length=4, refresh_count=2)
    model = SRDModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    initial_bank_a = torch.zeros(1, 2, config.d_model)
    initial_bank_b = torch.randn(1, 2, config.d_model)

    outputs_a = model(input_ids, initial_bank_states=initial_bank_a)
    outputs_b = model(input_ids, initial_bank_states=initial_bank_b)

    assert torch.allclose(outputs_a["logits"][:, : config.segment_length, :], outputs_b["logits"][:, : config.segment_length, :])
    assert not torch.allclose(outputs_a["logits"][:, config.segment_length :, :], outputs_b["logits"][:, config.segment_length :, :])
    assert outputs_a["debug"]["token_bank_access_count"] == 0


def test_local_only_baseline_skips_bank_access() -> None:
    config = SRDConfig(use_refresh=False, sufficiency_loss_weight=0.0)
    model = SRDModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    outputs = model(input_ids)
    assert outputs["refresh_states"].numel() == 0
    assert outputs["debug"]["refresh_bank_access_count"] == 0
