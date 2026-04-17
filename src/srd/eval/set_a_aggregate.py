"""Aggregate Experiment Set A per-run JSON artifacts into CSV summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from srd.eval.result_artifacts import write_aggregate_csv, write_grouped_csv, write_markdown_summary


def load_run_results(input_dir: str | Path) -> list[dict]:
    """Loads all per-run JSON files in one directory."""
    directory = Path(input_dir)
    results = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("aggregate_"):
            continue
        with path.open("r", encoding="utf-8") as handle:
            results.append(json.load(handle))
    return results


def aggregate_results(input_dir: str | Path, output_dir: str | Path) -> dict[str, str]:
    """Writes flat, grouped, and markdown summaries."""
    results = load_run_results(input_dir)
    if not results:
        raise ValueError(f"No run JSON files found in {input_dir}")
    csv_path = write_aggregate_csv(output_dir, results)
    grouped_path = write_grouped_csv(output_dir, results)
    markdown_path = write_markdown_summary(output_dir, results)
    return {
        "aggregate_csv": str(csv_path),
        "grouped_csv": str(grouped_path),
        "summary_markdown": str(markdown_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Experiment Set A results.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(aggregate_results(args.input_dir, args.output_dir))
