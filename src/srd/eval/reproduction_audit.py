"""Audit helpers for synthetic benchmark scoring and aggregate reporting."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from srd.data.synthetic_benchmarks import SyntheticBenchmarkConfig, make_synthetic_dataset
from srd.eval.result_artifacts import write_aggregate_csv, write_grouped_csv


def _perfect_logits_from_batch(batch: dict, vocab_size: int) -> torch.Tensor:
    input_ids = batch["input_ids"]
    logits = torch.zeros(input_ids.size(0), input_ids.size(1), vocab_size, dtype=torch.float)
    for batch_index in range(batch["answer_positions"].size(0)):
        for answer_index in range(batch["answer_positions"].size(1)):
            token_position = int(batch["answer_positions"][batch_index, answer_index].item()) - 1
            token_id = int(batch["answer_tokens"][batch_index, answer_index].item())
            logits[batch_index, token_position, token_id] = 10.0
    return logits


def _score_audit_rows() -> list[dict[str, Any]]:
    families = [
        ("delayed_kv", "accuracy"),
        ("needle", "accuracy"),
        ("delayed_copy", "exact_match"),
    ]
    rows: list[dict[str, Any]] = []
    for family, expected_metric in families:
        dataset = make_synthetic_dataset(SyntheticBenchmarkConfig(family=family, seed=7, segment_length=8))
        batch = dataset.make_batch(0, 2, torch.device("cpu"))
        logits = _perfect_logits_from_batch(batch, dataset.config.vocab_size)
        score = dataset.score_batch(logits, batch)
        rows.append(
            {
                "family": family,
                "metric_name": score["metric_name"],
                "metric_value": float(score["metric_value"]),
                "expected_metric_name": expected_metric,
                "score_matches_expected_metric": score["metric_name"] == expected_metric,
                "perfect_prediction_scores_one": abs(float(score["metric_value"]) - 1.0) < 1e-6,
                "task_metrics": sorted(score["task_metrics"].keys()),
            }
        )
    return rows


def _aggregate_audit_results() -> list[dict[str, Any]]:
    return [
        {
            "experiment_set": "reproduction_audit",
            "run_name": "audit_delayed_kv_seed11",
            "task_label": "delayed_kv",
            "task_category": "required_reproduction",
            "variant": "refresh_with_sufficiency",
            "parameter_count": 123,
            "trainable_parameter_count": 123,
            "benchmark": {"family": "delayed_kv", "seed": 11, "mode": "easy"},
            "model": {
                "segment_length": 8,
                "block_size": 8,
                "refresh_count": 2,
                "refresh_slots": 2,
                "refresh_enabled": True,
                "detail_enabled": False,
                "bank_size": 4,
                "upper_layer_only_refresh": True,
            },
            "metric_name": "accuracy",
            "metric_value": 0.25,
            "train_initial_loss": 1.5,
            "train_final_loss": 1.0,
            "average_train_step_time_seconds": 0.1,
            "average_eval_step_time_seconds": 0.05,
            "lm_loss": 0.8,
            "answer_loss": 0.6,
            "sufficiency_loss": 0.2,
            "tokens_per_second": 100.0,
            "decode_tokens_per_second": 80.0,
            "peak_memory_bytes": 2048.0,
            "throughput_per_memory": 100.0 / 2048.0,
            "bank_read_slots": 6.0,
            "segment_count": 8.0,
            "task_metrics": {"accuracy": 0.25, "token_accuracy": 0.25},
            "efficiency": {"tokens_per_second": 100.0},
            "debug": {"bank_read_slots": 6.0},
        },
        {
            "experiment_set": "reproduction_audit",
            "run_name": "audit_delayed_kv_seed17",
            "task_label": "delayed_kv",
            "task_category": "required_reproduction",
            "variant": "refresh_with_sufficiency",
            "parameter_count": 123,
            "trainable_parameter_count": 123,
            "benchmark": {"family": "delayed_kv", "seed": 17, "mode": "easy"},
            "model": {
                "segment_length": 8,
                "block_size": 8,
                "refresh_count": 2,
                "refresh_slots": 2,
                "refresh_enabled": True,
                "detail_enabled": False,
                "bank_size": 4,
                "upper_layer_only_refresh": True,
            },
            "metric_name": "accuracy",
            "metric_value": 0.75,
            "train_initial_loss": 1.2,
            "train_final_loss": 0.9,
            "average_train_step_time_seconds": 0.12,
            "average_eval_step_time_seconds": 0.06,
            "lm_loss": 0.7,
            "answer_loss": 0.4,
            "sufficiency_loss": 0.1,
            "tokens_per_second": 120.0,
            "decode_tokens_per_second": 90.0,
            "peak_memory_bytes": 1024.0,
            "throughput_per_memory": 120.0 / 1024.0,
            "bank_read_slots": 4.0,
            "segment_count": 8.0,
            "task_metrics": {"accuracy": 0.75, "token_accuracy": 0.75},
            "efficiency": {"tokens_per_second": 120.0},
            "debug": {"bank_read_slots": 4.0},
        },
    ]


def audit_score_and_aggregate(output_dir: str | Path) -> dict[str, str]:
    """Writes a deterministic audit of score_batch and aggregate reporting."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_rows = _score_audit_rows()
    aggregate_results = _aggregate_audit_results()
    aggregate_csv = write_aggregate_csv(output_dir, aggregate_results)
    grouped_csv = write_grouped_csv(output_dir, aggregate_results)

    grouped_rows = list(csv.DictReader(grouped_csv.open("r", encoding="utf-8")))
    if len(grouped_rows) != 1:
        raise ValueError("Expected one grouped row in aggregate audit")
    grouped_row = grouped_rows[0]

    mean_metric = float(grouped_row["metric_value_mean"])
    std_metric = float(grouped_row["metric_value_std"])
    expected_mean = 0.5
    expected_std = 0.25

    summary_path = output_dir / "score_aggregate_audit.md"
    lines = [
        "# Score And Aggregate Audit",
        "",
        "## Score Implementation",
        "",
        "| family | metric_name | expected_metric_name | perfect_prediction_metric | metric_name_ok | perfect_score_ok | task_metrics |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in score_rows:
        lines.append(
            f"| {row['family']} | {row['metric_name']} | {row['expected_metric_name']} | {row['metric_value']:.4f} | "
            f"{row['score_matches_expected_metric']} | {row['perfect_prediction_scores_one']} | {', '.join(row['task_metrics'])} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Implementation",
            "",
            "- `aggregate_results.csv` is a flat per-run export.",
            "- `aggregate_grouped.csv` groups by `task_label`, `task_category`, `variant`, `model_size`, and `context_length`.",
            "- Numeric fields are summarized as mean/std over seeds within each group.",
            "",
            "## Deterministic Grouped Sanity Check",
            "",
            f"- `metric_value_mean`: observed `{mean_metric:.4f}`, expected `{expected_mean:.4f}`",
            f"- `metric_value_std`: observed `{std_metric:.4f}`, expected `{expected_std:.4f}`",
            f"- pass: `{abs(mean_metric - expected_mean) < 1e-6 and abs(std_metric - expected_std) < 1e-6}`",
            "",
            "## Output Files",
            "",
            f"- `{aggregate_csv.name}`",
            f"- `{grouped_csv.name}`",
            f"- `{summary_path.name}`",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "aggregate_csv": str(aggregate_csv),
        "grouped_csv": str(grouped_csv),
        "summary_markdown": str(summary_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit synthetic score and aggregate implementations.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(audit_score_and_aggregate(args.output_dir))
