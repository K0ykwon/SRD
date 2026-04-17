"""Tests for conventional baseline variants and shared reporting behavior."""

import torch

from srd.config import SRDConfig
from srd.data.synthetic_benchmarks import SyntheticBenchmarkConfig
from srd.eval.benchmark_runner import build_model_config, run_benchmark_experiment
from srd.modeling.baseline_models import SummaryMemoryModel, TransformerFullModel, TransformerLocalModel


def test_transformer_local_outputs_expected_shapes() -> None:
    config = SRDConfig(model_type="transformer_local", vocab_size=32, d_model=16, num_heads=4, num_layers=3)
    model = TransformerLocalModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["refresh_states"].numel() == 0
    assert outputs["debug"]["token_bank_access_count"] == 0


def test_transformer_full_outputs_expected_shapes() -> None:
    config = SRDConfig(model_type="transformer_full", vocab_size=32, d_model=16, num_heads=4, num_layers=3)
    model = TransformerFullModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["refresh_states"].numel() == 0
    assert outputs["debug"]["bank_read_slots"] == 0


def test_transformer_full_prefill_decode_matches_forward_logits() -> None:
    config = SRDConfig(model_type="transformer_full", vocab_size=32, d_model=16, num_heads=4, num_layers=2)
    model = TransformerFullModel(config)
    model.eval()
    prefix = torch.randint(0, config.vocab_size, (1, 5))

    full_outputs = model(prefix)
    state = model.prefill(prefix)
    assert "layer_caches" in state
    assert len(state["layer_caches"]) == config.num_layers
    assert torch.allclose(state["next_logits"], full_outputs["logits"][:, -1, :], atol=1e-5, rtol=1e-5)

    next_token = torch.randint(0, config.vocab_size, (1, 1))
    stepped = model.decode_step(next_token, state)
    extended = torch.cat([prefix, next_token], dim=1)
    extended_outputs = model(extended)
    assert torch.allclose(stepped["next_logits"], extended_outputs["logits"][:, -1, :], atol=1e-5, rtol=1e-5)


def test_summary_memory_allows_direct_token_bank_access() -> None:
    config = SRDConfig(
        model_type="summary_memory",
        vocab_size=32,
        d_model=16,
        num_heads=4,
        num_layers=4,
        segment_length=4,
        bank_size=2,
    )
    model = SummaryMemoryModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    initial_bank_a = torch.zeros(1, 2, config.d_model)
    initial_bank_b = torch.randn(1, 2, config.d_model)

    outputs_a = model(input_ids, initial_bank_states=initial_bank_a)
    outputs_b = model(input_ids, initial_bank_states=initial_bank_b)

    assert not torch.allclose(outputs_a["logits"], outputs_b["logits"])
    assert outputs_a["debug"]["token_bank_access_count"] == input_ids.size(1)
    assert outputs_a["debug"]["refresh_bank_access_count"] == 0


def test_parameter_count_reporting_matches_model_parameters() -> None:
    benchmark_config = SyntheticBenchmarkConfig(
        family="delayed_kv",
        vocab_size=32,
        segment_length=4,
        context_segments=4,
        gap_segments=2,
        local_window=2,
    )
    model_config = build_model_config("transformer_local", benchmark_config)
    result = run_benchmark_experiment(
        benchmark_config=benchmark_config,
        model_config=model_config,
        train_steps=2,
        eval_batches=1,
        batch_size=2,
        learning_rate=1e-3,
    )
    manual_count = sum(parameter.numel() for parameter in TransformerLocalModel(model_config).parameters())

    assert result["variant"] == "transformer_local"
    assert result["parameter_count"] == manual_count
    assert result["trainable_parameter_count"] == manual_count
