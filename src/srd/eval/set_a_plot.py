"""Plot and table generation for Experiment Set A aggregates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_grouped_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return float(value)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_tables(rows: list[dict[str, str]], output_dir: str | Path) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    table1 = []
    table2 = []
    for row in rows:
        common = {
            "task_label": row["task_label"],
            "variant": row["variant"],
            "model_size": row.get("model_size", ""),
            "context_length": row.get("context_length", ""),
            "seed_count": row.get("seed_count", ""),
            "metric_mean": row.get("metric_value_mean", row.get("joint_accuracy_mean", row.get("accuracy_mean", ""))),
            "metric_std": row.get("metric_value_std", row.get("joint_accuracy_std", row.get("accuracy_std", ""))),
        }
        if row.get("model_size") == "small":
            table1.append(common)
        if row.get("model_size") == "base":
            expanded = {
                **common,
                "tokens_per_second_mean": row.get("tokens_per_second_mean", ""),
                "peak_memory_bytes_mean": row.get("peak_memory_bytes_mean", ""),
                "decode_tokens_per_second_mean": row.get("decode_tokens_per_second_mean", ""),
            }
            table2.append(expanded)

    table1_path = directory / "table_1.csv"
    table2_path = directory / "table_2.csv"
    _write_csv(table1_path, table1)
    _write_csv(table2_path, table2)
    return {"table_1": str(table1_path), "table_2": str(table2_path)}


def plot_context_scaling(rows: list[dict[str, str]], output_dir: str | Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["task_label"], []).append(row)

    fig, axes = plt.subplots(len(grouped), 1, figsize=(8, 4 * max(len(grouped), 1)), squeeze=False)
    for axis, (task_label, task_rows) in zip(axes.flatten(), sorted(grouped.items())):
        by_variant: dict[str, list[dict[str, str]]] = {}
        for row in task_rows:
            by_variant.setdefault(row["variant"], []).append(row)
        for variant, variant_rows in sorted(by_variant.items()):
            variant_rows = sorted(variant_rows, key=lambda item: _to_float(item, "context_length"))
            xs = [_to_float(row, "context_length") for row in variant_rows]
            ys = [_to_float(row, "metric_value_mean", _to_float(row, "accuracy_mean", _to_float(row, "joint_accuracy_mean"))) for row in variant_rows]
            axis.plot(xs, ys, marker="o", label=variant)
        axis.set_title(f"Context scaling: {task_label}")
        axis.set_xlabel("Context length")
        axis.set_ylabel("Metric mean")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    output_path = directory / "figure_1_context_scaling.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def plot_efficiency_quality(rows: list[dict[str, str]], output_dir: str | Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        x = _to_float(row, "tokens_per_second_mean")
        y = _to_float(row, "metric_value_mean", _to_float(row, "accuracy_mean", _to_float(row, "joint_accuracy_mean")))
        if x == 0.0 and y == 0.0:
            continue
        ax.scatter(x, y, label=f"{row['variant']}:{row['task_label']}", alpha=0.75)
    ax.set_xlabel("Tokens / second")
    ax.set_ylabel("Metric mean")
    ax.set_title("Efficiency vs quality")
    ax.grid(True, alpha=0.3)
    output_path = directory / "figure_2_efficiency_quality.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def generate_plot_artifacts(input_csv: str | Path, output_dir: str | Path) -> dict[str, str]:
    rows = _read_grouped_rows(input_csv)
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")
    tables = build_tables(rows, output_dir)
    figure1 = plot_context_scaling(rows, output_dir)
    figure2 = plot_efficiency_quality(rows, output_dir)
    return {**tables, "figure_1": figure1, "figure_2": figure2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Set A tables and plots.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(generate_plot_artifacts(args.input_csv, args.output_dir))
