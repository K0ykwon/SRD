"""Runs a multi-seed block-refresh paper suite and writes analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from srd.data.synthetic_benchmarks import SyntheticBenchmarkConfig
from srd.eval.benchmark_runner import build_model_config, run_benchmark_experiment
from srd.eval.result_artifacts import write_aggregate_csv, write_markdown_summary, write_run_json


def load_config(path: str | Path) -> dict:
    """Loads a paper-suite JSON config."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_run_overrides(benchmark_config: SyntheticBenchmarkConfig, model_config, overrides: dict) -> None:
    """Applies sweep overrides to benchmark and model configs."""
    if "block_size" in overrides:
        benchmark_config.segment_length = overrides["block_size"]
        model_config.block_size = overrides["block_size"]
        model_config.segment_length = overrides["block_size"]
    if "refresh_slots" in overrides:
        benchmark_config.refresh_count = overrides["refresh_slots"]
        model_config.refresh_slots = overrides["refresh_slots"]
        model_config.refresh_count = overrides["refresh_slots"]
    if "bank_size" in overrides:
        benchmark_config.bank_size = overrides["bank_size"]
        model_config.bank_size = overrides["bank_size"]
    if "upper_layer_only_refresh" in overrides:
        benchmark_config.upper_layer_only_refresh = overrides["upper_layer_only_refresh"]
        model_config.upper_layer_only_refresh = overrides["upper_layer_only_refresh"]
    if "sufficiency_loss_weight" in overrides:
        model_config.sufficiency_loss_weight = overrides["sufficiency_loss_weight"]
    if "detail_enabled" in overrides:
        model_config.detail_enabled = overrides["detail_enabled"]
    if "detail_slots" in overrides:
        model_config.detail_slots = overrides["detail_slots"]
    if "detail_topk" in overrides:
        model_config.detail_topk = overrides["detail_topk"]
    if "detail_gate_enabled" in overrides:
        model_config.detail_gate_enabled = overrides["detail_gate_enabled"]
    if "detail_anchor_first" in overrides:
        model_config.detail_anchor_first = overrides["detail_anchor_first"]
    if "detail_anchor_last" in overrides:
        model_config.detail_anchor_last = overrides["detail_anchor_last"]
    if "detail_saliency_slots" in overrides:
        model_config.detail_saliency_slots = overrides["detail_saliency_slots"]
    if "context_segments" in overrides:
        benchmark_config.context_segments = overrides["context_segments"]
    if "gap_segments" in overrides:
        benchmark_config.gap_segments = overrides["gap_segments"]
    if "num_distractors" in overrides:
        benchmark_config.num_distractors = overrides["num_distractors"]
    if "distractor_density" in overrides:
        benchmark_config.distractor_density = overrides["distractor_density"]
    if "symbol_pool_size" in overrides:
        benchmark_config.symbol_pool_size = overrides["symbol_pool_size"]
    if "mode" in overrides:
        benchmark_config.mode = overrides["mode"]


def _main_runs(experiment: dict) -> list[dict]:
    """Builds the main comparison run specifications."""
    runs = []
    variant_model_configs = experiment["variant_model_configs"]
    runner = experiment["runner"]
    seeds = experiment["seeds"]
    for task in experiment["tasks"]:
        for variant in experiment["variants"]:
            for seed in seeds:
                runs.append(
                    {
                        "task_label": task["label"],
                        "task_category": task["category"],
                        "family": task["family"],
                        "seed": seed,
                        "benchmark_overrides": task["benchmark_overrides"],
                        "variant": variant,
                        "model_config_path": variant_model_configs.get(variant),
                        "runner": runner,
                        "analysis_group": "main",
                        "sweep_name": "main",
                    }
                )
    return runs


def _sweep_runs(experiment: dict) -> list[dict]:
    """Builds run specifications for block-size / weight / slot sweeps."""
    runs = []
    variant_model_configs = experiment["variant_model_configs"]
    for sweep in experiment.get("sweeps", []):
        for task in sweep["tasks"]:
            for variant in sweep["variants"]:
                for seed in sweep["seeds"]:
                    for value in sweep["values"]:
                        overrides = deepcopy(task["benchmark_overrides"])
                        overrides[sweep["parameter"]] = value
                        runs.append(
                            {
                                "task_label": task["label"],
                                "task_category": task["category"],
                                "family": task["family"],
                                "seed": seed,
                                "benchmark_overrides": overrides,
                                "variant": variant,
                                "model_config_path": variant_model_configs.get(variant),
                                "runner": sweep["runner"],
                                "analysis_group": "sweep",
                                "sweep_name": sweep["name"],
                                "sweep_parameter": sweep["parameter"],
                                "sweep_value": value,
                            }
                        )
    return runs


def _prepare_run_result(base_result: dict, spec: dict) -> dict:
    """Attaches task and sweep metadata to one benchmark result."""
    result = dict(base_result)
    result["task_label"] = spec["task_label"]
    result["task_category"] = spec["task_category"]
    result["analysis_group"] = spec["analysis_group"]
    result["sweep_name"] = spec["sweep_name"]
    if "sweep_parameter" in spec:
        result["sweep_parameter"] = spec["sweep_parameter"]
        result["sweep_value"] = spec["sweep_value"]
    return result


def run_suite(experiment: dict, output_dir: str | Path) -> list[dict]:
    """Runs the configured large paper suite."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_specs = _main_runs(experiment) + _sweep_runs(experiment)
    results = []
    for spec in all_specs:
        benchmark_config = SyntheticBenchmarkConfig(
            family=spec["family"],
            seed=spec["seed"],
            vocab_size=experiment["benchmark_defaults"].get("vocab_size", 64),
            segment_length=spec["benchmark_overrides"].get("block_size", spec["benchmark_overrides"].get("segment_length", 8)),
            context_segments=spec["benchmark_overrides"].get("context_segments", experiment["benchmark_defaults"].get("context_segments", 6)),
            gap_segments=spec["benchmark_overrides"].get("gap_segments", experiment["benchmark_defaults"].get("gap_segments", 3)),
            num_distractors=spec["benchmark_overrides"].get("num_distractors", experiment["benchmark_defaults"].get("num_distractors", 3)),
            distractor_density=spec["benchmark_overrides"].get("distractor_density", experiment["benchmark_defaults"].get("distractor_density", 2)),
            pattern_length=spec["benchmark_overrides"].get("pattern_length", experiment["benchmark_defaults"].get("pattern_length", 3)),
            symbol_pool_size=spec["benchmark_overrides"].get("symbol_pool_size", experiment["benchmark_defaults"].get("symbol_pool_size", 8)),
            mode=spec["benchmark_overrides"].get("mode", experiment["benchmark_defaults"].get("mode", "easy")),
        )
        benchmark_config.refresh_count = spec["benchmark_overrides"].get("refresh_slots", spec["benchmark_overrides"].get("refresh_count", experiment["benchmark_defaults"].get("refresh_count", 2)))
        benchmark_config.bank_size = spec["benchmark_overrides"].get("bank_size", experiment["benchmark_defaults"].get("bank_size", 4))
        benchmark_config.upper_layer_only_refresh = spec["benchmark_overrides"].get(
            "upper_layer_only_refresh",
            experiment["benchmark_defaults"].get("upper_layer_only_refresh", True),
        )

        model_config = build_model_config(
            spec["variant"],
            benchmark_config,
            model_config_path=spec["model_config_path"],
        )
        _apply_run_overrides(benchmark_config, model_config, spec["benchmark_overrides"])

        result = run_benchmark_experiment(
            benchmark_config=benchmark_config,
            model_config=model_config,
            train_steps=spec["runner"]["train_steps"],
            eval_batches=spec["runner"]["eval_batches"],
            batch_size=spec["runner"]["batch_size"],
            learning_rate=spec["runner"]["learning_rate"],
            answer_loss_weight=spec["runner"].get("answer_loss_weight", 1.0),
            lm_loss_weight=spec["runner"].get("lm_loss_weight", 0.25),
            grad_clip_norm=spec["runner"].get("grad_clip_norm", 1.0),
            warmup_fraction=spec["runner"].get("warmup_fraction", 0.1),
        )
        result = _prepare_run_result(result, spec)
        write_run_json(output_dir, result)
        results.append(result)

    write_aggregate_csv(output_dir, results)
    write_markdown_summary(output_dir, results)
    with (output_dir / "aggregate_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def _group_mean(rows: list[dict], key_fields: tuple[str, ...], value_field: str) -> dict[tuple, float]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups[key].append(float(row[value_field]))
    return {key: mean(values) for key, values in groups.items()}


def _plot_accuracy_by_model(results: list[dict], output_dir: Path) -> None:
    main_rows = [row for row in results if row.get("analysis_group") == "main"]
    tasks = sorted({row["task_label"] for row in main_rows})
    variants = sorted({row["variant"] for row in main_rows})
    grouped = _group_mean(main_rows, ("task_label", "variant"), "metric_value")

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.12
    x_positions = list(range(len(tasks)))
    for variant_index, variant in enumerate(variants):
        values = [grouped.get((task, variant), 0.0) for task in tasks]
        xs = [x + (variant_index - (len(variants) - 1) / 2.0) * width for x in x_positions]
        ax.bar(xs, values, width=width, label=variant)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_ylabel("Metric")
    ax.set_title("Accuracy / Exact Match by Model and Task")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_model.png", dpi=160)
    plt.close(fig)


def _plot_refresh_vs_no_suff(results: list[dict], output_dir: Path) -> None:
    main_rows = [row for row in results if row.get("analysis_group") == "main" and row["variant"] in {"refresh_no_sufficiency", "refresh_with_sufficiency"}]
    tasks = sorted({row["task_label"] for row in main_rows})
    grouped = _group_mean(main_rows, ("task_label", "variant"), "metric_value")

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.35
    x_positions = list(range(len(tasks)))
    no_suf = [grouped.get((task, "refresh_no_sufficiency"), 0.0) for task in tasks]
    with_suf = [grouped.get((task, "refresh_with_sufficiency"), 0.0) for task in tasks]
    ax.bar([x - width / 2 for x in x_positions], no_suf, width=width, label="refresh_no_sufficiency")
    ax.bar([x + width / 2 for x in x_positions], with_suf, width=width, label="refresh_with_sufficiency")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_ylabel("Metric")
    ax.set_title("Refresh With vs Without Sufficiency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "refresh_with_vs_without_sufficiency.png", dpi=160)
    plt.close(fig)


def _plot_sweep(results: list[dict], output_dir: Path, sweep_name: str, title: str, filename: str) -> None:
    sweep_rows = [row for row in results if row.get("sweep_name") == sweep_name]
    if not sweep_rows:
        return
    task_labels = sorted({row["task_label"] for row in sweep_rows})
    values = sorted({row["sweep_value"] for row in sweep_rows})
    grouped = _group_mean(sweep_rows, ("task_label", "sweep_value"), "metric_value")

    fig, ax = plt.subplots(figsize=(10, 5))
    for task in task_labels:
        ys = [grouped.get((task, value), 0.0) for value in values]
        ax.plot(values, ys, marker="o", label=task)
    ax.set_xlabel(sweep_rows[0]["sweep_parameter"])
    ax.set_ylabel("Metric")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def _plot_scatter(results: list[dict], output_dir: Path, x_field: str, filename: str, title: str) -> None:
    main_rows = [row for row in results if row.get("analysis_group") == "main"]
    variants = sorted({row["variant"] for row in main_rows})
    fig, ax = plt.subplots(figsize=(8, 6))
    for variant in variants:
        rows = [row for row in main_rows if row["variant"] == variant]
        ax.scatter(
            [float(row[x_field]) for row in rows],
            [float(row["metric_value"]) for row in rows],
            label=variant,
            alpha=0.7,
        )
    ax.set_xlabel(x_field)
    ax.set_ylabel("Metric")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def _best_rows_table(results: list[dict]) -> str:
    main_rows = [row for row in results if row.get("analysis_group") == "main"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in main_rows:
        grouped[row["task_label"]].append(row)

    lines = [
        "| task | best_variant | category | metric | value | params | tok/s |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for task, rows in sorted(grouped.items()):
        variant_means: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            variant_means[row["variant"]].append(row)
        summary = []
        for variant, variant_rows in variant_means.items():
            summary.append(
                (
                    variant,
                    mean(float(row["metric_value"]) for row in variant_rows),
                    mean(float(row["tokens_per_second"]) for row in variant_rows),
                    int(variant_rows[0]["parameter_count"]),
                    variant_rows[0].get("task_category", ""),
                    variant_rows[0]["metric_name"],
                )
            )
        best_variant, best_value, best_toks, best_params, best_category, metric_name = max(summary, key=lambda item: item[1])
        lines.append(
            f"| {task} | {best_variant} | {best_category} | {metric_name} | "
            f"{best_value:.4f} | {best_params} | {best_toks:.2f} |"
        )
    return "\n".join(lines)


def _interpretation(results: list[dict]) -> list[str]:
    main_rows = [row for row in results if row.get("analysis_group") == "main"]
    task_means = _group_mean(main_rows, ("task_label", "variant"), "metric_value")
    lines = []

    refresh_friendly = [task for task in sorted({row["task_label"] for row in main_rows if row.get("task_category") == "refresh_friendly"})]
    refresh_hostile = [task for task in sorted({row["task_label"] for row in main_rows if row.get("task_category") == "refresh_hostile"})]

    if refresh_friendly:
        better_tasks = []
        for task in refresh_friendly:
            if task_means.get((task, "refresh_with_sufficiency"), -1.0) > task_means.get((task, "local_only"), -1.0):
                better_tasks.append(task)
        if better_tasks:
            lines.append(f"- Refresh with sufficiency beat local-only on refresh-friendly tasks: {', '.join(better_tasks)}.")
        else:
            lines.append("- Refresh with sufficiency did not consistently beat local-only on the designated refresh-friendly tasks.")

    if refresh_hostile:
        stronger_full = []
        for task in refresh_hostile:
            if task_means.get((task, "transformer_full"), -1.0) > task_means.get((task, "refresh_with_sufficiency"), -1.0):
                stronger_full.append(task)
        if stronger_full:
            lines.append(f"- Full Transformer remained stronger than refresh-with-sufficiency on refresh-hostile tasks: {', '.join(stronger_full)}.")

    suf_help = []
    for task in sorted({row["task_label"] for row in main_rows}):
        if task_means.get((task, "refresh_with_sufficiency"), -1.0) > task_means.get((task, "refresh_no_sufficiency"), -1.0):
            suf_help.append(task)
    if suf_help:
        lines.append(f"- The sufficiency objective improved the refresh model on: {', '.join(suf_help)}.")
    else:
        lines.append("- The sufficiency objective did not produce a consistent gain in this suite.")

    lines.append("- Conclusions should stay narrow: these synthetic results test whether scheduled refresh-only access can help on some long-context tasks, not whether it replaces full attention universally.")
    return lines


def write_analysis(results: list[dict], output_dir: str | Path) -> None:
    """Writes consolidated markdown, json, csv, and plot artifacts."""
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with (analysis_dir / "aggregate_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    write_aggregate_csv(analysis_dir, results)
    write_markdown_summary(analysis_dir, results)

    _plot_accuracy_by_model(results, analysis_dir)
    _plot_refresh_vs_no_suff(results, analysis_dir)
    _plot_sweep(results, analysis_dir, "block_size", "Effect of Block Size", "effect_of_block_size.png")
    _plot_sweep(results, analysis_dir, "sufficiency_loss_weight", "Effect of Sufficiency Loss Weight", "effect_of_sufficiency_loss_weight.png")
    _plot_sweep(results, analysis_dir, "detail_slots", "Effect of Detail Slots", "effect_of_detail_slots.png")
    _plot_sweep(results, analysis_dir, "detail_topk", "Effect of Detail Top-K", "effect_of_detail_topk.png")
    _plot_scatter(results, analysis_dir, "parameter_count", "accuracy_vs_parameter_count.png", "Accuracy vs Parameter Count")
    _plot_scatter(results, analysis_dir, "tokens_per_second", "accuracy_vs_throughput.png", "Accuracy vs Throughput")

    generated_files = [
        "aggregate_results.csv",
        "aggregate_results.json",
        "summary.md",
        "accuracy_by_model.png",
        "refresh_with_vs_without_sufficiency.png",
        "effect_of_block_size.png",
        "effect_of_sufficiency_loss_weight.png",
        "effect_of_detail_slots.png",
        "effect_of_detail_topk.png",
        "accuracy_vs_parameter_count.png",
        "accuracy_vs_throughput.png",
    ]
    existing_files = [name for name in generated_files if (analysis_dir / name).exists()]

    report_lines = [
        "# Block-Refresh Large Suite Report",
        "",
        "## Best Results Table",
        "",
        _best_rows_table(results),
        "",
        "## Interpretation",
        "",
        *_interpretation(results),
        "",
        "## Artifacts",
        "",
        *[f"- `{name}`" for name in existing_files],
    ]
    (analysis_dir / "summary_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parses the large-suite CLI."""
    parser = argparse.ArgumentParser(description="Run the large block-refresh SRD paper suite.")
    parser.add_argument("--config", required=True, help="Path to the paper-suite JSON config.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment = load_config(args.config)
    results = run_suite(experiment, args.output_dir)
    write_analysis(results, args.output_dir)
