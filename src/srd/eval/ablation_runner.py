"""Small config-grid sweep support for SRD synthetic benchmark ablations."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import List

from srd.data.synthetic_benchmarks import SyntheticBenchmarkConfig
from srd.eval.benchmark_runner import build_model_config, run_benchmark_experiment
from srd.eval.result_artifacts import write_aggregate_csv, write_markdown_summary, write_run_json


def expand_ablation_grid(experiment: dict) -> List[dict]:
    """Expands a simple product grid into concrete benchmark run specs."""
    benchmarks = experiment["benchmarks"]
    variants = experiment["variants"]
    grid = experiment["grid"]
    keys = ["segment_length", "refresh_count", "bank_size", "upper_layer_only_refresh"]
    values = [grid[key] for key in keys]

    runs = []
    for benchmark_name, variant, combination in itertools.product(benchmarks, variants, itertools.product(*values)):
        overrides = dict(zip(keys, combination))
        runs.append(
            {
                "benchmark": benchmark_name,
                "variant": variant,
                "overrides": overrides,
            }
        )
    return runs


def load_experiment(path: str | Path) -> dict:
    """Loads a JSON ablation sweep description."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_ablation_sweep(experiment: dict, output_dir: str | Path) -> List[dict]:
    """Runs the expanded sweep and writes structured result artifacts."""
    results = []
    runner = experiment["runner"]
    base = experiment["benchmark_defaults"]
    model_config_path = experiment.get("model_config_path")
    variant_model_configs = experiment.get("variant_model_configs", {})
    for run in expand_ablation_grid(experiment):
        benchmark_config = SyntheticBenchmarkConfig(
            family=run["benchmark"],
            seed=base.get("seed", 0),
            vocab_size=base.get("vocab_size", 64),
            segment_length=run["overrides"]["segment_length"],
            context_segments=base.get("context_segments", 6),
            gap_segments=base.get("gap_segments", 3),
            num_distractors=base.get("num_distractors", 3),
            distractor_density=base.get("distractor_density", 2),
            pattern_length=base.get("pattern_length", 3),
            symbol_pool_size=base.get("symbol_pool_size", 8),
            mode=base.get("mode", "easy"),
        )
        benchmark_config.refresh_count = run["overrides"]["refresh_count"]
        benchmark_config.bank_size = run["overrides"]["bank_size"]
        benchmark_config.upper_layer_only_refresh = run["overrides"]["upper_layer_only_refresh"]
        selected_model_config_path = variant_model_configs.get(run["variant"], model_config_path)
        model_config = build_model_config(
            run["variant"],
            benchmark_config,
            model_config_path=selected_model_config_path,
        )

        result = run_benchmark_experiment(
            benchmark_config=benchmark_config,
            model_config=model_config,
            train_steps=runner["train_steps"],
            eval_batches=runner["eval_batches"],
            batch_size=runner["batch_size"],
            learning_rate=runner["learning_rate"],
            answer_loss_weight=runner.get("answer_loss_weight", 1.0),
            lm_loss_weight=runner.get("lm_loss_weight", 0.25),
            grad_clip_norm=runner.get("grad_clip_norm", 1.0),
            warmup_fraction=runner.get("warmup_fraction", 0.1),
        )
        result["sweep_overrides"] = run["overrides"]
        write_run_json(output_dir, result)
        results.append(result)

    write_aggregate_csv(output_dir, results)
    write_markdown_summary(output_dir, results)
    return results


def parse_args() -> argparse.Namespace:
    """Parses the ablation sweep CLI."""
    parser = argparse.ArgumentParser(description="Run a small SRD ablation sweep.")
    parser.add_argument("--config", required=True, help="Path to an ablation sweep JSON config")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV/markdown artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation_sweep(load_experiment(args.config), args.output_dir)
