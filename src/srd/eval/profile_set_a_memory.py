"""Run a Set A suite and emit a compact memory/throughput summary table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from srd.eval.benchmark_runner import run_suite


def _write_memory_summary(output_dir: str | Path, results: list[dict]) -> Path:
    output_path = Path(output_dir) / "memory_summary.csv"
    fieldnames = [
        "run_name",
        "task_label",
        "task_category",
        "model_family",
        "model_size",
        "parameter_count",
        "trainable_parameter_count",
        "d_model",
        "num_layers",
        "num_heads",
        "refresh_slots",
        "bank_size",
        "detail_topk",
        "context_length",
        "metric_name",
        "metric_value",
        "tokens_per_second",
        "decode_tokens_per_second",
        "peak_memory_bytes",
        "peak_memory_mib",
        "throughput_per_memory",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "run_name": result.get("run_name", ""),
                    "task_label": result.get("task_label", result["benchmark"]["family"]),
                    "task_category": result.get("task_category", ""),
                    "model_family": result.get("model_family", ""),
                    "model_size": result.get("model_size", ""),
                    "parameter_count": result.get("parameter_count", ""),
                    "trainable_parameter_count": result.get("trainable_parameter_count", ""),
                    "d_model": result.get("model", {}).get("d_model", ""),
                    "num_layers": result.get("model", {}).get("num_layers", ""),
                    "num_heads": result.get("model", {}).get("num_heads", ""),
                    "refresh_slots": result.get("model", {}).get("refresh_slots", result.get("model", {}).get("refresh_count", "")),
                    "bank_size": result.get("model", {}).get("bank_size", ""),
                    "detail_topk": result.get("model", {}).get("detail_topk", ""),
                    "context_length": result.get("context_length", ""),
                    "metric_name": result.get("metric_name", ""),
                    "metric_value": result.get("metric_value", ""),
                    "tokens_per_second": result.get("tokens_per_second", ""),
                    "decode_tokens_per_second": result.get("decode_tokens_per_second", ""),
                    "peak_memory_bytes": result.get("peak_memory_bytes", ""),
                    "peak_memory_mib": float(result.get("peak_memory_bytes", 0.0)) / (1024.0 * 1024.0),
                    "throughput_per_memory": result.get("throughput_per_memory", ""),
                }
            )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Set A suite memory/throughput tradeoffs.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-runs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_suite(
        suite_path=args.suite,
        train_config_path=args.train_config,
        output_dir=args.output_dir,
        max_runs=args.max_runs,
        include_ablations=False,
    )
    summary_path = _write_memory_summary(args.output_dir, results)
    print({"runs": len(results), "output_dir": args.output_dir, "memory_summary_csv": str(summary_path)})


if __name__ == "__main__":
    main()
