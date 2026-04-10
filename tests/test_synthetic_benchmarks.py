"""Tests for synthetic long-context benchmark generation and scoring."""

from pathlib import Path

import torch

from srd.data.synthetic_benchmarks import SyntheticBenchmarkConfig, make_synthetic_dataset


def test_delayed_kv_is_reproducible_with_fixed_seed() -> None:
    config = SyntheticBenchmarkConfig(family="delayed_kv", seed=5)
    dataset_a = make_synthetic_dataset(config)
    dataset_b = make_synthetic_dataset(config)
    assert dataset_a.sample(0).input_ids == dataset_b.sample(0).input_ids
    assert dataset_a.sample(0).answer_tokens == dataset_b.sample(0).answer_tokens


def test_delayed_copy_labels_match_early_source_pattern() -> None:
    config = SyntheticBenchmarkConfig(family="delayed_copy", seed=9, pattern_length=3)
    sample = make_synthetic_dataset(config).sample(0)
    source_pattern = sample.input_ids[1 : 1 + config.pattern_length]
    assert sample.answer_tokens == source_pattern


def test_needles_use_accuracy_metric() -> None:
    config = SyntheticBenchmarkConfig(family="needle", seed=3)
    sample = make_synthetic_dataset(config).sample(0)
    assert sample.metric_name == "accuracy"
    assert len(sample.answer_tokens) == 1


def test_delayed_copy_target_depends_on_early_context() -> None:
    config = SyntheticBenchmarkConfig(family="delayed_copy", seed=4, pattern_length=3)
    dataset = make_synthetic_dataset(config)
    sample_a = dataset.sample(0)
    sample_b = dataset.sample(1)
    query_start_a = sample_a.answer_positions[0] - 2
    query_start_b = sample_b.answer_positions[0] - 2
    query_prefix_a = sample_a.input_ids[query_start_a : query_start_a + 2]
    query_prefix_b = sample_b.input_ids[query_start_b : query_start_b + 2]
    assert query_prefix_a == query_prefix_b
    assert sample_a.answer_tokens != sample_b.answer_tokens


def test_batch_scoring_returns_exact_match_for_copy_task() -> None:
    config = SyntheticBenchmarkConfig(family="delayed_copy", seed=2, pattern_length=2)
    dataset = make_synthetic_dataset(config)
    batch = dataset.make_batch(0, 2, torch.device("cpu"))
    logits = torch.zeros(2, config.sequence_length, config.vocab_size)
    for batch_index in range(2):
        for answer_index, token in enumerate(batch["answer_tokens"][batch_index]):
            answer_position = int(batch["answer_positions"][batch_index, answer_index].item())
            logits[batch_index, answer_position - 1, token] = 10.0
    score = dataset.score_batch(logits, batch)
    assert score["metric_name"] == "exact_match"
    assert score["metric_value"] == 1.0
