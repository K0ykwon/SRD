"""Tests for ablation sweep expansion and artifact writing."""

import json

from srd.eval.ablation_runner import expand_ablation_grid
from srd.eval.result_artifacts import write_aggregate_csv, write_markdown_summary, write_run_json


def test_ablation_grid_expands_expected_count() -> None:
    experiment = {
        "benchmarks": ["delayed_kv", "needle"],
        "variants": ["srd_without_sufficiency", "srd_with_sufficiency"],
        "grid": {
            "segment_length": [8, 12],
            "refresh_count": [1, 2],
            "bank_size": [2],
            "upper_layer_only_refresh": [True, False],
        },
    }
    runs = expand_ablation_grid(experiment)
    assert len(runs) == 2 * 2 * 2 * 2 * 1 * 2


def test_result_artifacts_write_json_csv_and_markdown(tmp_path) -> None:
    result = {
        "variant": "transformer_full",
        "benchmark": {"family": "delayed_kv", "mode": "easy", "seed": 1},
        "model": {
            "segment_length": 8,
            "refresh_count": 2,
            "bank_size": 4,
            "upper_layer_only_refresh": True,
        },
        "parameter_count": 123456,
        "trainable_parameter_count": 123456,
        "metric_name": "accuracy",
        "metric_value": 0.5,
        "lm_loss": 1.2,
        "sufficiency_loss": 0.3,
        "tokens_per_second": 100.0,
        "decode_tokens_per_second": 80.0,
        "peak_memory_bytes": 2048.0,
        "throughput_per_memory": 0.01,
        "bank_read_slots": 6.0,
        "segment_count": 4.0,
    }
    json_path = write_run_json(tmp_path, result)
    csv_path = write_aggregate_csv(tmp_path, [result])
    md_path = write_markdown_summary(tmp_path, [result])

    assert json.loads(json_path.read_text(encoding="utf-8"))["variant"] == "transformer_full"
    assert "parameter_count" in csv_path.read_text(encoding="utf-8")
    assert "transformer_full" in md_path.read_text(encoding="utf-8")
