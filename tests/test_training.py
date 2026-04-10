"""Tests for SRD losses and tiny train stability."""

import torch

from srd.config import SRDConfig
from srd.data.synthetic_benchmarks import SyntheticBenchmarkConfig, make_synthetic_dataset
from srd.modeling.factory import build_model
from srd.modeling.srd_model import SRDModel
from srd.training.losses import compute_answer_loss, compute_srd_loss
from srd.training.train import run_tiny_train


def test_sufficiency_loss_computes_and_backpropagates() -> None:
    config = SRDConfig(sufficiency_loss_weight=0.25)
    model = SRDModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    outputs = model(input_ids)
    losses = compute_srd_loss(outputs, input_ids, config)
    losses["loss"].backward()
    grad_norm = model.sufficiency_head.weight.grad.norm().item()
    assert losses["sufficiency_loss"].item() >= 0.0
    assert grad_norm > 0.0


def test_tiny_train_runs_stably_and_loss_does_not_explode() -> None:
    metrics = run_tiny_train(config=SRDConfig.preset("srd_suf_tiny"), steps=6, batch_size=2)
    assert metrics["stable"] is True
    assert metrics["final_loss"] <= metrics["initial_loss"] + 0.5


def test_answer_weighted_loss_reports_answer_loss() -> None:
    config = SRDConfig()
    model = SRDModel(config)
    benchmark = make_synthetic_dataset(SyntheticBenchmarkConfig(family="delayed_copy", seed=1))
    batch = benchmark.make_batch(0, 2, torch.device("cpu"))
    token_weights = batch["loss_weights"].clone()
    token_weights[token_weights > 1.0] = 8.0
    outputs = model(batch["input_ids"])
    losses = compute_srd_loss(outputs, batch["labels"], config, token_weights=token_weights)
    assert losses["answer_loss"].item() >= 0.0


def test_compute_answer_loss_uses_answer_positions() -> None:
    logits = torch.zeros(1, 6, 10)
    logits[0, 2, 7] = 10.0
    answer_positions = torch.tensor([[3]])
    answer_tokens = torch.tensor([[7]])
    loss = compute_answer_loss(logits, answer_positions, answer_tokens)
    assert loss.item() < 0.01


def test_block_refresh_tiny_train_reports_matching_parameter_counts() -> None:
    local_metrics = run_tiny_train(config=SRDConfig.preset("block_refresh_local_tiny"), steps=4, batch_size=2)
    refresh_metrics = run_tiny_train(config=SRDConfig.preset("block_refresh_tiny"), steps=4, batch_size=2)
    suf_metrics = run_tiny_train(config=SRDConfig.preset("block_refresh_suf_tiny"), steps=4, batch_size=2)

    assert local_metrics["stable"] is True
    assert refresh_metrics["stable"] is True
    assert suf_metrics["stable"] is True
    assert local_metrics["parameter_count"] == refresh_metrics["parameter_count"] == suf_metrics["parameter_count"]
    assert local_metrics["refresh_enabled"] is False
    assert refresh_metrics["refresh_enabled"] is True
    assert suf_metrics["sufficiency_enabled"] is True


def test_block_refresh_detail_tiny_train_runs_stably() -> None:
    metrics = run_tiny_train(config=SRDConfig.preset("block_refresh_detail_tiny"), steps=4, batch_size=2)
    assert metrics["stable"] is True
    assert metrics["refresh_enabled"] is True
    assert metrics["sufficiency_enabled"] is True
