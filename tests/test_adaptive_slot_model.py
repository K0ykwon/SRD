"""Tests for the adaptive fixed-shape refresh-slot SRD variant."""

import torch

from srd.config import SRDConfig
from srd.modeling.adaptive_slot_model import AdaptiveSlotSRDModel
from srd.training.losses import compute_srd_loss


def _adaptive_config(**overrides: object) -> SRDConfig:
    values = {
        "model_type": "adaptive_slot_srd",
        "vocab_size": 32,
        "d_model": 16,
        "num_heads": 4,
        "num_layers": 4,
        "block_size": 4,
        "segment_length": 4,
        "local_window": 2,
        "refresh_slots": 4,
        "refresh_slots_max": 4,
        "refresh_count": 4,
        "bank_size": 16,
        "refresh_enabled": True,
        "use_refresh": True,
        "sufficiency_loss_weight": 0.25,
        "beta_budget": 0.1,
        "gamma_gate_entropy": 0.01,
        "memory_keep_last_n_segments": 2,
        "memory_read_mode": "slot_query_summary",
        "memory_read_every_n_layers": 2,
        "upper_layer_only_refresh": True,
    }
    values.update(overrides)
    return SRDConfig(**values)


def test_adaptive_slot_model_outputs_expected_shapes_and_metrics() -> None:
    config = _adaptive_config()
    model = AdaptiveSlotSRDModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["refresh_states"].shape == (2, 8, config.d_model)
    assert outputs["soft_refresh_gates"].shape == (2, 2, config.refresh_slots_max)
    assert outputs["bank_states"].shape == (2, 8, config.d_model)
    assert outputs["debug"]["token_bank_access_count"] == 0
    assert outputs["debug"]["refresh_slots_max"] == config.refresh_slots_max
    assert outputs["debug"]["memory_bank_slots_used"] == 8.0
    assert len(outputs["debug"]["slot_utilization_histogram"]) == config.refresh_slots_max


def test_adaptive_slot_model_smoke_forward_backward() -> None:
    config = _adaptive_config()
    model = AdaptiveSlotSRDModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    outputs = model(input_ids)
    losses = compute_srd_loss(outputs, input_ids, config)
    losses["loss"].backward()

    gate_grad = model.refresh_gate_mlp[0].weight.grad
    sufficiency_grad = model.sufficiency_head.weight.grad
    assert gate_grad is not None
    assert gate_grad.norm().item() > 0.0
    assert sufficiency_grad is not None
    assert losses["budget_loss"].item() >= 0.0


def test_adaptive_slot_hard_topk_gating_emits_fixed_active_count() -> None:
    config = _adaptive_config(refresh_gate_hard=True, refresh_gate_topk=2)
    model = AdaptiveSlotSRDModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    outputs = model(input_ids)
    hard_gates = outputs["hard_refresh_gates"]

    assert torch.equal(hard_gates.sum(dim=-1), torch.full((1, 2), 2.0))
    assert outputs["debug"]["average_active_hard_slots"] == 2.0


def test_adaptive_slot_prefill_decode_matches_completed_block_forward() -> None:
    config = _adaptive_config(memory_read_mode="pooled")
    model = AdaptiveSlotSRDModel(config)
    model.eval()
    partial_prefix = torch.randint(0, config.vocab_size, (1, config.block_size - 1))
    partial_state = model.prefill(partial_prefix)
    next_token = torch.randint(0, config.vocab_size, (1, 1))
    stepped = model.decode_step(next_token, partial_state)
    completed = torch.cat([partial_prefix, next_token], dim=1)
    completed_outputs = model(completed)

    assert torch.allclose(stepped["next_logits"], completed_outputs["logits"][:, -1, :], atol=1e-5, rtol=1e-5)
