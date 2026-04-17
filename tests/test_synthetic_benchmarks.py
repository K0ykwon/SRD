"""Tests for synthetic long-context benchmark generation and scoring."""

from pathlib import Path

import torch

from srd.data.synthetic_benchmarks import SyntheticBenchmarkConfig, make_synthetic_dataset


def test_delayed_kv_alias_is_reproducible_with_fixed_seed() -> None:
    config = SyntheticBenchmarkConfig(family="delayed_kv", seed=5)
    dataset_a = make_synthetic_dataset(config)
    dataset_b = make_synthetic_dataset(config)
    assert dataset_a.sample(0).input_ids == dataset_b.sample(0).input_ids
    assert dataset_a.sample(0).answer_tokens == dataset_b.sample(0).answer_tokens


def test_binding_kv_uses_similar_value_shapes_and_exact_match_metric() -> None:
    config = SyntheticBenchmarkConfig(
        family="binding_kv",
        seed=7,
        segment_length=16,
        segment_count=4,
        total_length=64,
        answer_span_length=2,
        num_keys=4,
    )
    dataset = make_synthetic_dataset(config)
    sample = dataset.sample(0)
    assert sample.metric_name == "accuracy"
    assert len(sample.answer_tokens) == 2
    first_segment = sample.input_ids[: config.segment_length]
    bind_entries = []
    cursor = 0
    while cursor + 3 < len(first_segment):
        if first_segment[cursor] != 1:
            cursor += 1
            continue
        bind_entries.append(first_segment[cursor + 1 : cursor + 4])
        cursor += 4
    assert len(bind_entries) >= 2
    shared_prefixes = {tuple(entry[1:2]) for entry in bind_entries}
    assert len(shared_prefixes) == 1


def test_binding_kv_scoring_uses_exact_value_match_for_accuracy() -> None:
    config = SyntheticBenchmarkConfig(
        family="binding_kv",
        seed=11,
        segment_length=16,
        segment_count=4,
        total_length=64,
        answer_span_length=2,
        num_keys=4,
    )
    dataset = make_synthetic_dataset(config)
    batch = dataset.make_batch(0, 1, torch.device("cpu"))
    logits = torch.zeros(1, config.sequence_length, config.vocab_size)
    first_token = int(batch["answer_tokens"][0, 0].item())
    second_token = int(batch["answer_tokens"][0, 1].item())
    first_position = int(batch["answer_positions"][0, 0].item())
    second_position = int(batch["answer_positions"][0, 1].item())
    logits[0, first_position - 1, first_token] = 10.0
    wrong_token = (second_token + 1) % config.vocab_size
    logits[0, second_position - 1, wrong_token] = 10.0
    score = dataset.score_batch(logits, batch)
    assert score["metric_name"] == "accuracy"
    assert score["task_metrics"]["token_accuracy"] == 0.5
    assert score["task_metrics"]["value_span_exact_match"] == 0.0
    assert score["metric_value"] == 0.0


def test_binding_kv_scoring_distinguishes_wrong_catalog_value() -> None:
    config = SyntheticBenchmarkConfig(
        family="binding_kv",
        seed=13,
        segment_length=16,
        segment_count=4,
        total_length=64,
        answer_span_length=2,
        num_keys=4,
    )
    dataset = make_synthetic_dataset(config)
    batch = dataset.make_batch(0, 1, torch.device("cpu"))
    logits = torch.zeros(1, config.sequence_length, config.vocab_size)
    candidate_values = batch["metadata"][0]["candidate_values"]
    gold_value = batch["answer_tokens"][0].tolist()
    wrong_value = next(value for value in candidate_values if value != gold_value)
    for answer_index, token in enumerate(wrong_value):
        answer_position = int(batch["answer_positions"][0, answer_index].item())
        logits[0, answer_position - 1, token] = 10.0
    score = dataset.score_batch(logits, batch)
    assert score["task_metrics"]["wrong_value_catalog_rate"] == 1.0
    assert score["task_metrics"]["off_catalog_rate"] == 0.0
    assert score["task_metrics"]["retrieval_hit"] == 1.0
    assert score["task_metrics"]["binding_accuracy"] == 0.0


def test_binding_lite_alias_uses_easy_kv_family() -> None:
    config = SyntheticBenchmarkConfig(
        family="binding_lite_kv",
        seed=3,
        segment_length=32,
        segment_count=2,
        total_length=64,
    )
    dataset = make_synthetic_dataset(config)
    batch = dataset.make_batch(0, 1, torch.device("cpu"))
    assert batch["family"] == "easy_kv"


def test_binding_heavy_alias_uses_binding_kv_family() -> None:
    config = SyntheticBenchmarkConfig(
        family="binding_heavy_kv",
        seed=3,
        segment_length=32,
        segment_count=2,
        total_length=64,
        answer_span_length=2,
        num_keys=4,
    )
    dataset = make_synthetic_dataset(config)
    batch = dataset.make_batch(0, 1, torch.device("cpu"))
    assert batch["family"] == "binding_kv"


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
