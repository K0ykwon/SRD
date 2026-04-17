"""Benchmark runner for synthetic long-context SRD validation tasks."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import torch

from srd.config import SRDConfig
from srd.data.synthetic_benchmarks import FAMILY_ALIASES, SyntheticBenchmarkConfig, make_synthetic_dataset
from srd.eval.metrics import compute_throughput_per_memory
from srd.eval.result_artifacts import (
    write_aggregate_csv,
    write_grouped_csv,
    write_markdown_summary,
    write_run_json,
)
from srd.modeling.factory import build_model
from srd.training.losses import compute_answer_loss, compute_srd_loss


MODEL_FAMILY_TO_VARIANT = {
    "transformer_local": "transformer_local",
    "transformer_full": "transformer_full",
    "transformer_full_15m": "transformer_full",
    "adaptive_slot_srd": "adaptive_slot_srd",
    "srd_refresh": "refresh_no_sufficiency",
    "srd_refresh_sufficiency": "refresh_with_sufficiency",
    "srd_refresh_sufficiency_detail": "refresh_with_detail",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_variant_name(variant: str) -> str:
    """Maps scale-specific variant labels back to the shared model family."""
    if variant in MODEL_FAMILY_TO_VARIANT:
        return MODEL_FAMILY_TO_VARIANT[variant]
    if variant.startswith("transformer_full"):
        return "transformer_full"
    if variant.startswith("transformer_local"):
        return "transformer_local"
    if variant.startswith("summary_memory"):
        return "summary_memory"
    if variant.startswith("transformer_xl_style"):
        return "transformer_xl_style"
    if variant.startswith("perceiver_latent"):
        return "perceiver_latent"
    if variant.startswith("adaptive_slot_srd"):
        return "adaptive_slot_srd"
    if variant.startswith("refresh_with_detail"):
        return "refresh_with_detail"
    if variant.startswith("refresh_with_sufficiency"):
        return "refresh_with_sufficiency"
    if variant.startswith("refresh_no_sufficiency"):
        return "refresh_no_sufficiency"
    if variant.startswith("local_only"):
        return "local_only"
    return variant


def _git_metadata() -> dict[str, str | bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": False}


def _peak_memory_bytes(device: torch.device, input_ids: torch.Tensor, config: SRDConfig) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device))
    return float(input_ids.numel() * config.d_model * 4)


def _decode_tokens_per_second(model: torch.nn.Module, batch: dict) -> float:
    answer_start = int(batch["answer_positions"][0, 0].item())
    answer_len = int(batch["answer_positions"].size(1))
    decode_input = batch["input_ids"][:, :answer_start].clone()
    start = time.perf_counter()
    with torch.inference_mode():
        if hasattr(model, "prefill") and hasattr(model, "decode_step"):
            state = model.prefill(decode_input)
            next_logits = state["next_logits"]
            for _ in range(answer_len):
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                state = model.decode_step(next_token, state)
                next_logits = state["next_logits"]
        else:
            for _ in range(answer_len):
                model_input = decode_input
                if hasattr(model, "config") and model.config.model_type in {"srd_block_refresh", "srd_block_refresh_detail"}:
                    block_size = model.config.effective_block_size()
                    remainder = model_input.size(1) % block_size
                    if remainder != 0:
                        pad = block_size - remainder
                        model_input = torch.nn.functional.pad(model_input, (0, pad), value=0)
                outputs = model(model_input)
                next_token = outputs["logits"][:, decode_input.size(1) - 1, :].argmax(dim=-1, keepdim=True)
                decode_input = torch.cat([decode_input, next_token], dim=1)
    elapsed = time.perf_counter() - start
    decoded = answer_len * batch["input_ids"].size(0)
    return decoded / max(elapsed, 1e-6)


def _normalize_model_depth(config: SRDConfig) -> SRDConfig:
    if config.model_type in {"srd_block_refresh", "srd_block_refresh_detail", "adaptive_slot_srd"}:
        pre_layers = max(1, config.num_layers // 2)
        post_layers = max(1, config.num_layers - pre_layers)
        config.num_local_layers_pre = pre_layers
        config.num_local_layers_post = post_layers
    return config


def build_model_config(
    variant: str,
    benchmark_config: SyntheticBenchmarkConfig,
    model_config_path: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> SRDConfig:
    """Builds the shared model config for one benchmark variant."""
    canonical_variant = canonical_variant_name(variant)
    if model_config_path is not None:
        config = SRDConfig.from_json_file(model_config_path)
    else:
        if canonical_variant in {"local_only", "transformer_local"}:
            config = SRDConfig.preset("local_tiny")
        elif canonical_variant == "transformer_xl_style":
            config = SRDConfig.preset("transformer_xl_style_tiny")
        elif canonical_variant == "perceiver_latent":
            config = SRDConfig.preset("perceiver_latent_tiny")
        elif canonical_variant == "adaptive_slot_srd":
            config = SRDConfig.preset("adaptive_slot_srd_tiny")
        elif canonical_variant == "refresh_no_sufficiency":
            config = SRDConfig.preset("block_refresh_tiny")
        elif canonical_variant == "refresh_with_sufficiency":
            config = SRDConfig.preset("block_refresh_suf_tiny")
        elif canonical_variant in {"refresh_with_detail", "refresh_detail_with_sufficiency"}:
            config = SRDConfig.preset("block_refresh_detail_tiny")
        elif canonical_variant == "refresh_detail_no_sufficiency":
            config = SRDConfig.preset("block_refresh_detail_no_suf_tiny")
        elif canonical_variant == "transformer_full":
            config = SRDConfig.preset("transformer_full_tiny")
        elif canonical_variant == "summary_memory":
            config = SRDConfig.preset("summary_memory_tiny")
        elif canonical_variant == "srd_without_sufficiency":
            config = SRDConfig.preset("srd_tiny")
        elif canonical_variant == "srd_with_sufficiency":
            config = SRDConfig.preset("srd_suf_tiny")
        else:
            raise ValueError(f"Unknown variant: {variant}")

    if config_overrides:
        config = SRDConfig.from_dict({**config.to_dict(), **config_overrides})

    if canonical_variant in {"local_only", "transformer_local"} and config.model_type != "srd_block_refresh":
        config.model_type = "transformer_local"
        config.use_refresh = False
        config.refresh_enabled = False
        config.sufficiency_loss_weight = 0.0
    elif canonical_variant == "local_only" and config.model_type == "srd_block_refresh":
        config.refresh_enabled = False
        config.use_refresh = False
        config.sufficiency_loss_weight = 0.0
    elif canonical_variant == "refresh_no_sufficiency":
        config.model_type = "srd_block_refresh"
        config.refresh_enabled = True
        config.use_refresh = True
        config.detail_enabled = False
        config.sufficiency_loss_weight = 0.0
    elif canonical_variant == "refresh_with_sufficiency":
        config.model_type = "srd_block_refresh"
        config.refresh_enabled = True
        config.use_refresh = True
        config.detail_enabled = False
        config.sufficiency_loss_weight = max(config.sufficiency_loss_weight, 0.1)
    elif canonical_variant in {"refresh_with_detail", "refresh_detail_with_sufficiency"}:
        config.model_type = "srd_block_refresh_detail"
        config.refresh_enabled = True
        config.use_refresh = True
        config.detail_enabled = True
        config.sufficiency_loss_weight = max(config.sufficiency_loss_weight, 0.1)
    elif canonical_variant == "refresh_detail_no_sufficiency":
        config.model_type = "srd_block_refresh_detail"
        config.refresh_enabled = True
        config.use_refresh = True
        config.detail_enabled = True
        config.sufficiency_loss_weight = 0.0
    elif canonical_variant == "transformer_full":
        config.model_type = "transformer_full"
        config.use_refresh = False
        config.refresh_enabled = False
        config.detail_enabled = False
        config.sufficiency_loss_weight = 0.0
    elif canonical_variant == "transformer_xl_style":
        config.model_type = "transformer_xl_style"
        config.use_refresh = False
        config.refresh_enabled = False
        config.detail_enabled = False
        config.sufficiency_loss_weight = 0.0
    elif canonical_variant == "perceiver_latent":
        config.model_type = "perceiver_latent"
        config.use_refresh = False
        config.refresh_enabled = False
        config.detail_enabled = False
        config.sufficiency_loss_weight = 0.0
    elif canonical_variant == "adaptive_slot_srd":
        config.model_type = "adaptive_slot_srd"
        config.use_refresh = True
        config.refresh_enabled = True
        config.detail_enabled = False
        config.sufficiency_loss_weight = max(config.sufficiency_loss_weight, 0.1)
    elif canonical_variant == "summary_memory":
        config.model_type = "summary_memory"
        config.use_refresh = False
        config.refresh_enabled = False
        config.detail_enabled = False
        config.sufficiency_loss_weight = 0.0

    config.block_size = benchmark_config.segment_length
    config.segment_length = benchmark_config.segment_length
    config.local_window = min(config.local_window, benchmark_config.segment_length)
    config.refresh_slots = max(config.refresh_slots, benchmark_config.refresh_count)
    config.refresh_slots_max = max(config.refresh_slots_max, benchmark_config.refresh_count)
    config.refresh_count = max(config.refresh_count, config.refresh_slots)
    config.bank_size = benchmark_config.bank_size
    if config.model_type == "adaptive_slot_srd" and config.memory_keep_last_n_segments > 0:
        config.bank_size = max(
            config.bank_size,
            config.memory_keep_last_n_segments * config.refresh_slots_max,
        )
    config.upper_layer_only_refresh = benchmark_config.upper_layer_only_refresh
    config.vocab_size = benchmark_config.vocab_size
    return _normalize_model_depth(config)


def run_benchmark_experiment(
    benchmark_config: SyntheticBenchmarkConfig,
    model_config: SRDConfig,
    train_steps: int = 32,
    eval_batches: int = 8,
    batch_size: int = 8,
    learning_rate: float = 3e-3,
    answer_loss_weight: float = 1.0,
    lm_loss_weight: float = 0.25,
    grad_clip_norm: float = 1.0,
    warmup_fraction: float = 0.1,
    experiment_set: str = "",
    run_name: str = "",
    task_category: str = "",
    log_prefix: str = "",
) -> Dict[str, Any]:
    """Trains one variant on one synthetic benchmark and returns metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = make_synthetic_dataset(benchmark_config)
    eval_dataset = make_synthetic_dataset(
        SyntheticBenchmarkConfig(**{**benchmark_config.to_dict(), "split": "test"})
    )

    model = build_model(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    warmup_steps = max(1, int(train_steps * warmup_fraction))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(1, train_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    best_state = None
    best_metric = -1.0
    initial_train_loss = None
    final_train_loss = None
    train_loss_curve = []
    train_sufficiency_curve = []
    train_budget_curve = []
    train_gate_entropy_curve = []
    train_step_times = []
    checkpoint_interval = max(1, train_steps // 5)
    model.train()
    for step in range(train_steps):
        step_start = time.perf_counter()
        batch = train_dataset.make_batch(step * batch_size, batch_size, device)
        outputs = model(batch["input_ids"])
        losses = compute_srd_loss(
            outputs,
            batch["labels"],
            model_config,
            token_weights=batch["loss_weights"],
            metadata=batch.get("metadata"),
        )
        answer_loss = compute_answer_loss(outputs["logits"], batch["answer_positions"], batch["answer_tokens"])
        total_loss = (
            lm_loss_weight * losses["nll_loss"]
            + answer_loss_weight * answer_loss
            + losses["regularization_loss"]
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if initial_train_loss is None:
            initial_train_loss = float(total_loss.detach().cpu())
        final_train_loss = float(total_loss.detach().cpu())
        train_loss_curve.append(float(total_loss.detach().cpu()))
        train_sufficiency_curve.append(float(losses["sufficiency_loss"].detach().cpu()))
        train_budget_curve.append(float(losses["budget_loss"].detach().cpu()))
        train_gate_entropy_curve.append(float(losses["gate_entropy_loss"].detach().cpu()))
        train_step_times.append(time.perf_counter() - step_start)

        with torch.no_grad():
            score = train_dataset.score_batch(outputs["logits"], batch)
        snapshot_due = step == 0 or (step + 1) == train_steps or (step + 1) % checkpoint_interval == 0
        if snapshot_due and score["metric_value"] >= best_metric:
            best_metric = float(score["metric_value"])
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if log_prefix and snapshot_due:
            print(
                f"{log_prefix} train step {step + 1}/{train_steps} "
                f"loss={float(total_loss.detach().cpu()):.4f} metric={float(score['metric_value']):.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    lm_losses = []
    answer_losses = []
    sufficiency_losses = []
    budget_losses = []
    gate_entropy_losses = []
    metric_values = []
    bank_read_slots = []
    segment_counts = []
    eval_step_times = []
    merged_task_metrics: dict[str, list[float]] = {}
    debug_stats: dict[str, list[float]] = {}

    model.eval()
    first_eval_batch = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    with torch.inference_mode():
        for batch_index in range(eval_batches):
            eval_step_start = time.perf_counter()
            batch = eval_dataset.make_batch(batch_index * batch_size, batch_size, device)
            outputs = model(batch["input_ids"])
            losses = compute_srd_loss(
                outputs,
                batch["labels"],
                model_config,
                token_weights=batch["loss_weights"],
                metadata=batch.get("metadata"),
            )
            answer_loss = compute_answer_loss(outputs["logits"], batch["answer_positions"], batch["answer_tokens"])
            score = eval_dataset.score_batch(outputs["logits"], batch)

            lm_losses.append(float(losses["nll_loss"].detach().cpu()))
            answer_losses.append(float(answer_loss.detach().cpu()))
            sufficiency_losses.append(float(losses["sufficiency_loss"].detach().cpu()))
            budget_losses.append(float(losses["budget_loss"].detach().cpu()))
            gate_entropy_losses.append(float(losses["gate_entropy_loss"].detach().cpu()))
            metric_values.append(float(score["metric_value"]))
            bank_read_slots.append(float(outputs["debug"].get("bank_read_slots", 0.0)))
            segment_counts.append(float(outputs["debug"].get("segment_count", 0.0)))
            eval_step_times.append(time.perf_counter() - eval_step_start)
            for key, value in score.get("task_metrics", {}).items():
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        merged_task_metrics.setdefault(f"{key}.{nested_key}", []).append(float(nested_value))
                else:
                    merged_task_metrics.setdefault(key, []).append(float(value))
            for key, value in outputs.get("debug", {}).items():
                if isinstance(value, (int, float)):
                    debug_stats.setdefault(key, []).append(float(value))

            if first_eval_batch is None:
                first_eval_batch = batch
    elapsed = time.perf_counter() - start

    if first_eval_batch is None:
        raise RuntimeError("Expected at least one evaluation batch")

    tokens = first_eval_batch["input_ids"].numel() * eval_batches
    tokens_per_second = tokens / max(elapsed, 1e-6)
    decode_tokens_per_second = _decode_tokens_per_second(model, first_eval_batch)
    peak_memory_bytes = _peak_memory_bytes(device, first_eval_batch["input_ids"], model_config)
    metric_name = eval_dataset.sample(0).metric_name

    task_metrics = {
        key: sum(values) / len(values)
        for key, values in merged_task_metrics.items()
    }
    debug_summary = {
        key: sum(values) / len(values)
        for key, values in debug_stats.items()
    }
    metadata = _git_metadata()
    result = {
        "experiment_set": experiment_set,
        "run_name": run_name,
        "variant": model_config.experiment_name(),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "benchmark": benchmark_config.to_dict(),
        "model": model_config.to_dict(),
        "metric_name": metric_name,
        "metric_value": sum(metric_values) / len(metric_values),
        "task_category": task_category,
        "task_metrics": task_metrics,
        "train_initial_loss": initial_train_loss,
        "train_final_loss": final_train_loss,
        "train_loss_curve": train_loss_curve,
        "train_sufficiency_curve": train_sufficiency_curve,
        "train_budget_curve": train_budget_curve,
        "train_gate_entropy_curve": train_gate_entropy_curve,
        "lm_loss": sum(lm_losses) / len(lm_losses),
        "answer_loss": sum(answer_losses) / len(answer_losses),
        "sufficiency_loss": sum(sufficiency_losses) / len(sufficiency_losses),
        "budget_loss": sum(budget_losses) / len(budget_losses),
        "gate_entropy_loss": sum(gate_entropy_losses) / len(gate_entropy_losses),
        "average_train_step_time_seconds": sum(train_step_times) / len(train_step_times),
        "average_eval_step_time_seconds": sum(eval_step_times) / len(eval_step_times),
        "tokens_per_second": tokens_per_second,
        "decode_tokens_per_second": decode_tokens_per_second,
        "peak_memory_bytes": peak_memory_bytes,
        "throughput_per_memory": compute_throughput_per_memory(tokens_per_second, peak_memory_bytes),
        "bank_read_slots": sum(bank_read_slots) / len(bank_read_slots),
        "segment_count": sum(segment_counts) / len(segment_counts),
        "efficiency": {
            "peak_gpu_memory_bytes": peak_memory_bytes,
            "tokens_per_second": tokens_per_second,
            "decode_tokens_per_second": decode_tokens_per_second,
            "wall_clock_step_seconds": sum(eval_step_times) / len(eval_step_times),
        },
        "debug": debug_summary,
        "average_active_soft_slots_per_segment": debug_summary.get("average_active_soft_slots_per_segment", 0.0),
        "average_gate_value": debug_summary.get("average_gate_value", 0.0),
        "average_active_hard_slots": debug_summary.get("average_active_hard_slots", 0.0),
        "memory_bank_slots_used": debug_summary.get("memory_bank_slots_used", 0.0),
        "peak_memory_bank_slots_used": debug_summary.get("peak_memory_bank_slots_used", 0.0),
        "notes": (
            f"Benchmark `{benchmark_config.family}` ran with variant `{model_config.experiment_name()}` "
            f"at block_size={model_config.effective_block_size()}, refresh_slots={model_config.effective_refresh_slots()}, "
            f"bank_size={model_config.bank_size}, upper_layer_only_refresh={model_config.upper_layer_only_refresh}."
        ),
    }
    result.update(metadata)
    if log_prefix:
        print(
            f"{log_prefix} eval done metric={result['metric_value']:.4f} "
            f"tok_s={result['tokens_per_second']:.2f} decode_tok_s={result['decode_tokens_per_second']:.2f}",
            flush=True,
        )
    return result


def load_benchmark_config(config_path: str | None, benchmark: str | None, mode: str) -> SyntheticBenchmarkConfig:
    if config_path is not None:
        return SyntheticBenchmarkConfig(**_load_json(config_path))
    if benchmark is None:
        raise ValueError("benchmark must be set when no config file is provided")
    return SyntheticBenchmarkConfig(family=FAMILY_ALIASES.get(benchmark, benchmark), mode=mode)


def _context_to_bank_size(context_length: int) -> int:
    if context_length <= 2048:
        return 64
    if context_length <= 4096:
        return 128
    return 256


def _ablation_variant_family(scope_family: str, ablation_name: str, value: Any) -> str:
    if ablation_name == "sufficiency_weight":
        return "srd_refresh_sufficiency" if float(value) > 0.0 else "srd_refresh"
    if ablation_name == "detail_topk":
        return "srd_refresh_sufficiency_detail"
    return scope_family


def _task_config_for_context(task_path: str | Path, context_length: int, seed: int) -> tuple[SyntheticBenchmarkConfig, str]:
    task_config = _load_json(task_path)
    difficulty_levels = task_config.pop("difficulty_levels", {})
    selected = None
    for name, values in difficulty_levels.items():
        if int(values.get("total_length", 0)) == context_length:
            selected = (name, values)
            break
    if selected is None:
        selected = next(iter(difficulty_levels.items()))
    difficulty_name, difficulty_values = selected
    merged = {**task_config, **difficulty_values}
    merged.pop("task_name", None)
    merged.pop("task_category", None)
    merged["family"] = task_config["task_name"]
    merged["seed"] = seed
    merged["split"] = "train"
    merged["refresh_count"] = 2
    merged["bank_size"] = _context_to_bank_size(context_length)
    merged["upper_layer_only_refresh"] = True
    merged["local_window"] = min(128, int(merged["segment_length"]))
    return SyntheticBenchmarkConfig(**merged), task_config["task_category"]


def _build_set_a_model_config(model_family: str, model_size: str, benchmark_config: SyntheticBenchmarkConfig) -> SRDConfig:
    backbone = _load_json(Path("configs/model/set_a") / f"shared_backbone_{model_size}.json")
    family = _load_json(Path("configs/model/set_a") / f"{model_family}.json")
    merged = {**backbone, **family}
    size_override_path = Path("configs/model/set_a") / f"{model_family}_{model_size}_override.json"
    if size_override_path.exists():
        merged.update(_load_json(size_override_path))
    merged["model_family"] = model_family
    merged["size_name"] = model_size
    return build_model_config(model_family, benchmark_config, config_overrides=merged)


def expand_suite_runs(suite_path: str | Path) -> list[dict[str, Any]]:
    suite = _load_json(suite_path)
    runs = []
    for model_family in suite["model_families"]:
        for model_size in suite["model_sizes"]:
            for context_length in suite["context_lengths"]:
                for task_name in suite["tasks"]:
                    task_path = Path("configs/experiment/set_a/tasks") / f"{task_name}.json"
                    for seed in suite["seeds"]:
                        benchmark_config, task_category = _task_config_for_context(task_path, context_length, seed)
                        run_name = f"set_a_{model_size}_ctx{context_length}_{task_name}_{model_family}_seed{seed}"
                        runs.append(
                            {
                                "experiment_set": suite.get("experiment_set", "set_a"),
                                "run_name": run_name,
                                "model_family": model_family,
                                "model_size": model_size,
                                "context_length": context_length,
                                "task_name": task_name,
                                "task_category": task_category,
                                "seed": seed,
                                "benchmark_config": benchmark_config,
                                "ablation": None,
                            }
                        )
    return runs


def expand_suite_ablations(suite_path: str | Path) -> list[dict[str, Any]]:
    suite = _load_json(suite_path)
    runs = []
    for ablation in suite.get("ablations", []):
        scope = ablation["scope"]
        for model_family in scope["model_families"]:
            for model_size in scope["model_sizes"]:
                for context_length in scope["context_lengths"]:
                    for task_name in scope["tasks"]:
                        task_path = Path("configs/experiment/set_a/tasks") / f"{task_name}.json"
                        for seed in scope["seeds"]:
                            benchmark_config, task_category = _task_config_for_context(task_path, context_length, seed)
                            for value in ablation["values"]:
                                effective_family = _ablation_variant_family(model_family, ablation["name"], value)
                                run_name = (
                                    f"set_a_ablation_{ablation['name']}_{value}_{model_size}_ctx{context_length}_"
                                    f"{task_name}_{effective_family}_seed{seed}"
                                )
                                runs.append(
                                    {
                                        "experiment_set": suite.get("experiment_set", "set_a"),
                                        "run_name": run_name,
                                        "model_family": effective_family,
                                        "model_size": model_size,
                                        "context_length": context_length,
                                        "task_name": task_name,
                                        "task_category": task_category,
                                        "seed": seed,
                                        "benchmark_config": benchmark_config,
                                        "ablation": {
                                            "name": ablation["name"],
                                            "value": value,
                                            "scope_model_family": model_family,
                                        },
                                    }
                                )
    return runs


def run_suite(
    suite_path: str | Path,
    train_config_path: str | Path,
    output_dir: str | Path,
    max_runs: int | None = None,
    include_ablations: bool = False,
) -> list[dict[str, Any]]:
    train_config = _load_json(train_config_path)
    runs = expand_suite_runs(suite_path)
    if include_ablations:
        runs.extend(expand_suite_ablations(suite_path))
    if max_runs is not None:
        runs = runs[:max_runs]
    results = []
    for run in runs:
        print(
            f"[suite] start run={run['run_name']} task={run['task_name']} family={run['model_family']} "
            f"size={run['model_size']} ctx={run['context_length']} seed={run['seed']}",
            flush=True,
        )
        model_config = _build_set_a_model_config(
            run["model_family"],
            run["model_size"],
            run["benchmark_config"],
        )
        if run["ablation"] is not None:
            ablation_name = run["ablation"]["name"]
            ablation_value = run["ablation"]["value"]
            if ablation_name == "sufficiency_weight":
                model_config.sufficiency_loss_weight = float(ablation_value)
            elif ablation_name == "detail_topk":
                model_config.detail_topk = int(ablation_value)
            elif ablation_name == "refresh_interval_blocks":
                model_config.refresh_interval_blocks = int(ablation_value)
        result = run_benchmark_experiment(
            benchmark_config=run["benchmark_config"],
            model_config=model_config,
            train_steps=train_config["max_steps"],
            eval_batches=max(1, train_config["eval_every"] // max(train_config["log_every"], 1)),
            batch_size=train_config["micro_batch_size"],
            learning_rate=train_config["lr"],
            answer_loss_weight=train_config.get("answer_loss_weight", 1.0),
            lm_loss_weight=train_config.get("lm_loss_weight", 0.25),
            grad_clip_norm=train_config.get("grad_clip_norm", 1.0),
            warmup_fraction=min(0.5, train_config.get("warmup_steps", 1) / max(train_config["max_steps"], 1)),
            experiment_set=run["experiment_set"],
            run_name=run["run_name"],
            task_category=run["task_category"],
            log_prefix=f"[{run['run_name']}]",
        )
        result["task_label"] = run["task_name"]
        result["model_size"] = run["model_size"]
        result["context_length"] = run["context_length"]
        result["seed"] = run["seed"]
        result["model_family"] = run["model_family"]
        result["ablation"] = run["ablation"]
        write_run_json(output_dir, result)
        results.append(result)
        print(f"[suite] finished run={run['run_name']} metric={result['metric_value']:.4f}", flush=True)
    write_aggregate_csv(output_dir, results)
    write_grouped_csv(output_dir, results)
    write_markdown_summary(output_dir, results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic long-context benchmarks.")
    parser.add_argument("--benchmark", default=None, help="Benchmark family for single-run mode")
    parser.add_argument("--variant", default="srd_with_sufficiency")
    parser.add_argument("--config", default=None, help="Optional benchmark JSON config")
    parser.add_argument("--model-config", default=None, help="Optional model JSON config")
    parser.add_argument("--mode", default="easy")
    parser.add_argument("--train-steps", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--answer-loss-weight", type=float, default=1.0)
    parser.add_argument("--lm-loss-weight", type=float, default=0.25)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--suite", default=None, help="Path to a Set A suite JSON config")
    parser.add_argument("--train-config", default=None, help="Train config for suite mode")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for suite debugging")
    parser.add_argument("--include-ablations", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.suite is not None:
        if args.train_config is None or args.output_dir is None:
            raise ValueError("--suite requires --train-config and --output-dir")
        results = run_suite(
            args.suite,
            args.train_config,
            args.output_dir,
            max_runs=args.max_runs,
            include_ablations=args.include_ablations,
        )
        print({"runs": len(results), "output_dir": args.output_dir})
    else:
        benchmark_config = load_benchmark_config(args.config, args.benchmark, args.mode)
        model_config = build_model_config(args.variant, benchmark_config, model_config_path=args.model_config)
        result = run_benchmark_experiment(
            benchmark_config=benchmark_config,
            model_config=model_config,
            train_steps=args.train_steps,
            eval_batches=args.eval_batches,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            answer_loss_weight=args.answer_loss_weight,
            lm_loss_weight=args.lm_loss_weight,
            grad_clip_norm=args.grad_clip_norm,
            warmup_fraction=args.warmup_fraction,
        )
        if args.output_dir:
            write_run_json(args.output_dir, result)
            write_aggregate_csv(args.output_dir, [result])
            write_grouped_csv(args.output_dir, [result])
            write_markdown_summary(args.output_dir, [result])
        print(result)
