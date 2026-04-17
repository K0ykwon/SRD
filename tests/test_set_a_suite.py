"""Tests for Experiment Set A task coverage and suite expansion."""

import json

from srd.data.synthetic_benchmarks import SyntheticBenchmarkConfig, make_synthetic_dataset
from srd.eval.benchmark_runner import _build_set_a_model_config, _task_config_for_context, expand_suite_ablations, expand_suite_runs
from srd.eval.set_a_aggregate import aggregate_results


def test_mixed_dependency_emits_joint_accuracy_metric() -> None:
    config = SyntheticBenchmarkConfig(
        family="mixed_dependency",
        seed=7,
        segment_length=8,
        segment_count=6,
        total_length=48,
    )
    dataset = make_synthetic_dataset(config)
    sample = dataset.sample(0)
    assert sample.metric_name == "joint_accuracy"
    assert len(sample.answer_tokens) == 1 + config.detail_span_length


def test_multi_hop_reasoning_contains_hop_metadata() -> None:
    config = SyntheticBenchmarkConfig(
        family="multi_hop_segment_reasoning",
        seed=3,
        segment_length=8,
        segment_count=6,
        total_length=48,
        hop_count=3,
    )
    sample = make_synthetic_dataset(config).sample(0)
    assert sample.metric_name == "accuracy"
    assert sample.metadata["hop_count"] == 3


def test_suite_pilot_expands_expected_count() -> None:
    runs = expand_suite_runs("configs/experiment/set_a/suite_pilot.json")
    assert len(runs) == 30


def test_suite_full_ablation_expands_expected_count() -> None:
    runs = expand_suite_ablations("configs/experiment/set_a/suite_full.json")
    assert len(runs) == 108


def test_required_longctx_suite_expands_expected_counts() -> None:
    main_runs = expand_suite_runs("configs/experiment/set_a/suite_reproduction_required_longctx.json")
    ablation_runs = expand_suite_ablations("configs/experiment/set_a/suite_reproduction_required_longctx.json")
    assert len(main_runs) == 216
    assert len(ablation_runs) == 180


def test_required_small_8k_suite_expands_expected_counts() -> None:
    main_runs = expand_suite_runs("configs/experiment/set_a/suite_reproduction_required_small_8k.json")
    ablation_runs = expand_suite_ablations("configs/experiment/set_a/suite_reproduction_required_small_8k.json")
    assert len(main_runs) == 36
    assert len(ablation_runs) == 45


def test_compact_set_a_backbone_supports_4k_required_runs() -> None:
    benchmark_config, _ = _task_config_for_context(
        "configs/experiment/set_a/tasks/delayed_kv.json",
        context_length=4096,
        seed=11,
    )
    srd_config = _build_set_a_model_config("srd_refresh_sufficiency_detail", "compact", benchmark_config)
    dense_config = _build_set_a_model_config("transformer_full", "compact", benchmark_config)
    assert srd_config.max_seq_len >= 4096
    assert dense_config.max_seq_len >= 4096


def test_set_a_aggregate_writes_grouped_csv(tmp_path) -> None:
    result = {
        "experiment_set": "set_a",
        "run_name": "dummy",
        "task_label": "mixed_dependency",
        "task_category": "mixed",
        "variant": "refresh_with_detail",
        "benchmark": {"family": "mixed_dependency", "mode": "pilot", "seed": 11},
        "model": {
            "segment_length": 64,
            "refresh_count": 2,
            "bank_size": 64,
            "upper_layer_only_refresh": True,
        },
        "model_size": "small",
        "context_length": 2048,
        "parameter_count": 123456,
        "trainable_parameter_count": 123456,
        "metric_name": "joint_accuracy",
        "metric_value": 0.5,
        "task_metrics": {"joint_accuracy": 0.5, "detail_part_accuracy": 0.75},
        "lm_loss": 1.2,
        "sufficiency_loss": 0.3,
        "tokens_per_second": 100.0,
        "decode_tokens_per_second": 80.0,
        "peak_memory_bytes": 2048.0,
        "throughput_per_memory": 0.01,
        "bank_read_slots": 6.0,
        "segment_count": 4.0,
    }
    (tmp_path / "run.json").write_text(json.dumps(result), encoding="utf-8")
    paths = aggregate_results(tmp_path, tmp_path / "out")
    grouped = (tmp_path / "out" / "aggregate_grouped.csv").read_text(encoding="utf-8")
    assert "joint_accuracy_mean" in grouped
    assert "detail_part_accuracy_mean" in grouped
    assert paths["grouped_csv"].endswith("aggregate_grouped.csv")
