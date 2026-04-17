"""Helpers for serializing benchmark results into paper-friendly local artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _flatten_result(result: dict) -> dict:
    task_metrics = result.get("task_metrics", {})
    row = {
        "experiment_set": result.get("experiment_set", ""),
        "run_name": result.get("run_name", ""),
        "task_label": result.get("task_label", result["benchmark"]["family"]),
        "task_category": result.get("task_category", ""),
        "seed": result["benchmark"].get("seed", ""),
        "benchmark": result["benchmark"]["family"],
        "mode": result["benchmark"].get("mode", ""),
        "variant": result["variant"],
        "parameter_count": result.get("parameter_count", ""),
        "trainable_parameter_count": result.get("trainable_parameter_count", ""),
        "metric_name": result["metric_name"],
        "metric_value": result["metric_value"],
        "train_initial_loss": result.get("train_initial_loss", ""),
        "train_final_loss": result.get("train_final_loss", ""),
        "average_train_step_time_seconds": result.get("average_train_step_time_seconds", ""),
        "average_eval_step_time_seconds": result.get("average_eval_step_time_seconds", ""),
        "lm_loss": result["lm_loss"],
        "answer_loss": result.get("answer_loss", ""),
        "sufficiency_loss": result["sufficiency_loss"],
        "tokens_per_second": result["tokens_per_second"],
        "decode_tokens_per_second": result["decode_tokens_per_second"],
        "peak_memory_bytes": result["peak_memory_bytes"],
        "throughput_per_memory": result["throughput_per_memory"],
        "bank_read_slots": result["bank_read_slots"],
        "segment_count": result["segment_count"],
        "block_size": result["model"].get("block_size", result["model"]["segment_length"]),
        "refresh_slots": result["model"].get("refresh_slots", result["model"]["refresh_count"]),
        "refresh_enabled": result["model"].get("refresh_enabled", result["model"].get("use_refresh", "")),
        "detail_enabled": result["model"].get("detail_enabled", ""),
        "detail_slots": result["model"].get("detail_slots", ""),
        "detail_topk": result["model"].get("detail_topk", ""),
        "segment_length": result["model"]["segment_length"],
        "refresh_count": result["model"]["refresh_count"],
        "bank_size": result["model"]["bank_size"],
        "upper_layer_only_refresh": result["model"]["upper_layer_only_refresh"],
    }
    if result.get("ablation") is not None:
        row["ablation_name"] = result["ablation"].get("name", "")
        row["ablation_value"] = result["ablation"].get("value", "")
        row["ablation_scope_model_family"] = result["ablation"].get("scope_model_family", "")
    for key, value in task_metrics.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                row[f"{key}.{nested_key}"] = nested_value
        else:
            row[key] = value
    if "efficiency" in result:
        for key, value in result["efficiency"].items():
            row[f"efficiency.{key}"] = value
    if "debug" in result:
        for key, value in result["debug"].items():
            if not isinstance(value, (dict, list)):
                row[f"debug.{key}"] = value
    return row


def write_run_json(output_dir: str | Path, result: dict) -> Path:
    """Writes one structured benchmark result JSON file."""
    directory = _ensure_dir(output_dir)
    label = result.get("task_label", result["benchmark"]["family"])
    run_name = result.get("run_name", "")
    if run_name:
        filename = f"{_safe_slug(run_name)}.json"
    else:
        model_size = result.get("model_size", "")
        context_length = result.get("context_length", "")
        filename = (
            f"{label}__{result['variant']}__"
            f"size{model_size or 'na'}__ctx{context_length or 'na'}__"
            f"seg{result['model'].get('block_size', result['model']['segment_length'])}__ref{result['model'].get('refresh_slots', result['model']['refresh_count'])}__"
            f"bank{result['model']['bank_size']}__ulr{int(result['model']['upper_layer_only_refresh'])}__"
            f"seed{result['benchmark']['seed']}.json"
        )
    path = directory / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return path


def write_aggregate_csv(output_dir: str | Path, results: Iterable[dict]) -> Path:
    """Writes a flat aggregate CSV over multiple benchmark runs."""
    directory = _ensure_dir(output_dir)
    rows = [_flatten_result(result) for result in results]
    if not rows:
        raise ValueError("Cannot write aggregate CSV with no results")
    path = directory / "aggregate_results.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_grouped_csv(output_dir: str | Path, results: Iterable[dict]) -> Path:
    """Writes mean/std summaries grouped by task/model/context."""
    directory = _ensure_dir(output_dir)
    rows = [_flatten_result(result) for result in results]
    if not rows:
        raise ValueError("Cannot write grouped CSV with no results")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("task_label", ""),
            row.get("task_category", ""),
            row.get("variant", ""),
            row.get("model_size", ""),
            row.get("context_length", ""),
        )
        groups[key].append(row)

    grouped_rows = []
    metric_keys = {
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float)) and key not in {"seed"}
    }
    for key, group_rows in groups.items():
        grouped = {
            "task_label": key[0],
            "task_category": key[1],
            "variant": key[2],
            "model_size": key[3],
            "context_length": key[4],
            "seed_count": len(group_rows),
        }
        for metric_key in sorted(metric_keys):
            values = [float(row[metric_key]) for row in group_rows if row.get(metric_key, "") != ""]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            grouped[f"{metric_key}_mean"] = mean
            grouped[f"{metric_key}_std"] = variance ** 0.5
        grouped_rows.append(grouped)

    path = directory / "aggregate_grouped.csv"
    fieldnames = sorted({key for row in grouped_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grouped_rows)
    return path


def write_markdown_summary(output_dir: str | Path, results: Iterable[dict]) -> Path:
    """Writes a markdown summary table plus short notes for each run."""
    directory = _ensure_dir(output_dir)
    results = list(results)
    path = directory / "summary.md"
    lines = [
        "# Synthetic Benchmark Summary",
        "",
        "| task | category | variant | params | metric | value | train_loss | answer_loss | lm_loss | sufficiency_loss | tok/s | decode tok/s | peak_memory_bytes |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.get('task_label', result['benchmark']['family'])} | {result.get('task_category', '')} | {result['variant']} | {result.get('parameter_count', 0)} | {result['metric_name']} | "
            f"{result['metric_value']:.4f} | {result.get('train_final_loss', 0.0):.4f} | "
            f"{result.get('answer_loss', 0.0):.4f} | {result['lm_loss']:.4f} | {result['sufficiency_loss']:.4f} | "
            f"{result['tokens_per_second']:.2f} | {result['decode_tokens_per_second']:.2f} | {result['peak_memory_bytes']:.0f} |"
        )

    lines.extend(["", "## Notes", ""])
    for result in results:
        note = (
            f"- Task `{result.get('task_label', result['benchmark']['family'])}` "
            f"({result.get('task_category', 'uncategorized')}) ran with variant `{result['variant']}`, "
            f"params={result.get('parameter_count', 0)}, "
            f"block_size={result['model'].get('block_size', result['model']['segment_length'])}, "
            f"refresh_slots={result['model'].get('refresh_slots', result['model']['refresh_count'])}, "
            f"refresh_enabled={result['model'].get('refresh_enabled', result['model'].get('use_refresh', ''))}, "
            f"bank_size={result['model']['bank_size']}, upper_layer_only_refresh={result['model']['upper_layer_only_refresh']}. "
            f"Key highlight: {result['metric_name']}={result['metric_value']:.4f}, "
            f"answer_loss={result.get('answer_loss', 0.0):.4f}, "
            f"throughput_per_memory={result['throughput_per_memory']:.6f}."
        )
        lines.append(note)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path
