"""Benchmark runner for synthetic long-context SRD validation tasks."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict

import torch

from srd.config import SRDConfig
from srd.data.synthetic_benchmarks import SyntheticBenchmarkConfig, make_synthetic_dataset
from srd.eval.metrics import compute_throughput_per_memory
from srd.eval.result_artifacts import write_aggregate_csv, write_markdown_summary, write_run_json
from srd.modeling.factory import build_model
from srd.training.losses import compute_answer_loss, compute_srd_loss


def _peak_memory_bytes(device: torch.device, input_ids: torch.Tensor, config: SRDConfig) -> float:
    """Returns GPU peak memory or a documented CPU estimate."""
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device))
    return float(input_ids.numel() * config.d_model * 4)


def _decode_tokens_per_second(model: torch.nn.Module, batch: dict) -> float:
    """Measures autoregressive decode speed over the answer span."""
    answer_start = int(batch["answer_positions"][0, 0].item())
    answer_len = int(batch["answer_positions"].size(1))
    decode_input = batch["input_ids"][:, :answer_start].clone()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(answer_len):
            outputs = model(decode_input)
            next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            decode_input = torch.cat([decode_input, next_token], dim=1)
    elapsed = time.perf_counter() - start
    decoded = answer_len * batch["input_ids"].size(0)
    return decoded / max(elapsed, 1e-6)


def build_model_config(
    variant: str,
    benchmark_config: SyntheticBenchmarkConfig,
    model_config_path: str | None = None,
) -> SRDConfig:
    """Builds the shared model config for one benchmark variant."""
    if model_config_path is not None:
        config = SRDConfig.from_json_file(model_config_path)
    else:
        if variant in {"local_only", "transformer_local"}:
            config = SRDConfig.preset("local_tiny")
        elif variant == "refresh_no_sufficiency":
            config = SRDConfig.preset("block_refresh_tiny")
        elif variant == "refresh_with_sufficiency":
            config = SRDConfig.preset("block_refresh_suf_tiny")
        elif variant in {"refresh_with_detail", "refresh_detail_with_sufficiency"}:
            config = SRDConfig.preset("block_refresh_detail_tiny")
        elif variant == "refresh_detail_no_sufficiency":
            config = SRDConfig.preset("block_refresh_detail_no_suf_tiny")
        elif variant == "transformer_full":
            config = SRDConfig.preset("transformer_full_tiny")
        elif variant == "summary_memory":
            config = SRDConfig.preset("summary_memory_tiny")
        elif variant == "srd_without_sufficiency":
            config = SRDConfig.preset("srd_tiny")
        elif variant == "srd_with_sufficiency":
            config = SRDConfig.preset("srd_suf_tiny")
        else:
            raise ValueError(f"Unknown variant: {variant}")

    if variant in {"local_only", "transformer_local"} and config.model_type != "srd_block_refresh":
        config.model_type = "transformer_local"
        config.use_refresh = False
        config.refresh_enabled = False
        config.sufficiency_loss_weight = 0.0
    elif variant == "local_only" and config.model_type == "srd_block_refresh":
        config.refresh_enabled = False
        config.use_refresh = False
        config.sufficiency_loss_weight = 0.0
    elif variant == "refresh_no_sufficiency":
        config.model_type = "srd_block_refresh"
        config.refresh_enabled = True
        config.use_refresh = True
        config.sufficiency_loss_weight = 0.0
    elif variant == "refresh_with_sufficiency":
        config.model_type = "srd_block_refresh"
        config.refresh_enabled = True
        config.use_refresh = True
        config.detail_enabled = False
        config.sufficiency_loss_weight = max(config.sufficiency_loss_weight, 0.25)
    elif variant in {"refresh_with_detail", "refresh_detail_with_sufficiency"}:
        config.model_type = "srd_block_refresh_detail"
        config.refresh_enabled = True
        config.use_refresh = True
        config.detail_enabled = True
        config.sufficiency_loss_weight = max(config.sufficiency_loss_weight, 0.25)
    elif variant == "refresh_detail_no_sufficiency":
        config.model_type = "srd_block_refresh_detail"
        config.refresh_enabled = True
        config.use_refresh = True
        config.detail_enabled = True
        config.sufficiency_loss_weight = 0.0
    elif variant == "transformer_full":
        config.model_type = "transformer_full"
        config.use_refresh = False
        config.refresh_enabled = False
        config.detail_enabled = False
        config.sufficiency_loss_weight = 0.0
    elif variant == "summary_memory":
        config.model_type = "summary_memory"
        config.use_refresh = False
        config.refresh_enabled = False
        config.detail_enabled = False
        config.sufficiency_loss_weight = 0.0
    elif variant == "srd_without_sufficiency":
        config.model_type = "srd"
        config.use_refresh = True
        config.refresh_enabled = True
        config.detail_enabled = False
        config.sufficiency_loss_weight = 0.0
    elif variant == "srd_with_sufficiency":
        config.model_type = "srd"
        config.use_refresh = True
        config.refresh_enabled = True
        config.detail_enabled = False
        config.sufficiency_loss_weight = max(config.sufficiency_loss_weight, 0.25)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    config.block_size = benchmark_config.segment_length
    config.segment_length = benchmark_config.segment_length
    config.local_window = benchmark_config.local_window
    config.refresh_slots = benchmark_config.refresh_count
    config.refresh_count = benchmark_config.refresh_count
    config.bank_size = benchmark_config.bank_size
    config.upper_layer_only_refresh = benchmark_config.upper_layer_only_refresh
    config.vocab_size = benchmark_config.vocab_size
    return config


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
) -> Dict[str, float | str | dict]:
    """Trains one variant on one synthetic benchmark and returns metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = make_synthetic_dataset(benchmark_config)
    eval_dataset = make_synthetic_dataset(SyntheticBenchmarkConfig(**{**benchmark_config.to_dict(), "seed": benchmark_config.seed + 10_000}))

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
    train_step_times = []
    model.train()
    for step in range(train_steps):
        step_start = time.perf_counter()
        batch = train_dataset.make_batch(step * batch_size, batch_size, device)
        outputs = model(batch["input_ids"])
        losses = compute_srd_loss(outputs, batch["labels"], model_config)
        answer_loss = compute_answer_loss(outputs["logits"], batch["answer_positions"], batch["answer_tokens"])
        total_loss = (
            lm_loss_weight * losses["nll_loss"]
            + answer_loss_weight * answer_loss
            + model_config.sufficiency_loss_weight * losses["sufficiency_loss"]
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
        train_step_times.append(time.perf_counter() - step_start)

        with torch.no_grad():
            score = train_dataset.score_batch(outputs["logits"], batch)
        if score["metric_value"] >= best_metric:
            best_metric = float(score["metric_value"])
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    lm_losses = []
    answer_losses = []
    sufficiency_losses = []
    metric_values = []
    bank_read_slots = []
    segment_counts = []
    eval_step_times = []

    model.eval()
    first_eval_batch = None
    first_outputs = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    with torch.no_grad():
        for batch_index in range(eval_batches):
            eval_step_start = time.perf_counter()
            batch = eval_dataset.make_batch(batch_index * batch_size, batch_size, device)
            outputs = model(batch["input_ids"])
            losses = compute_srd_loss(outputs, batch["labels"], model_config)
            answer_loss = compute_answer_loss(outputs["logits"], batch["answer_positions"], batch["answer_tokens"])
            score = eval_dataset.score_batch(outputs["logits"], batch)

            lm_losses.append(float(losses["nll_loss"].detach().cpu()))
            answer_losses.append(float(answer_loss.detach().cpu()))
            sufficiency_losses.append(float(losses["sufficiency_loss"].detach().cpu()))
            metric_values.append(float(score["metric_value"]))
            bank_read_slots.append(float(outputs["debug"]["bank_read_slots"]))
            segment_counts.append(float(outputs["debug"]["segment_count"]))
            eval_step_times.append(time.perf_counter() - eval_step_start)

            if first_eval_batch is None:
                first_eval_batch = batch
                first_outputs = outputs
    elapsed = time.perf_counter() - start

    if first_eval_batch is None or first_outputs is None:
        raise RuntimeError("Expected at least one evaluation batch")

    tokens = first_eval_batch["input_ids"].numel() * eval_batches
    tokens_per_second = tokens / max(elapsed, 1e-6)
    decode_tokens_per_second = _decode_tokens_per_second(model, first_eval_batch)
    peak_memory_bytes = _peak_memory_bytes(device, first_eval_batch["input_ids"], model_config)
    metric_name = eval_dataset.sample(0).metric_name

    return {
        "variant": model_config.experiment_name(),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "benchmark": benchmark_config.to_dict(),
        "model": model_config.to_dict(),
        "metric_name": metric_name,
        "metric_value": sum(metric_values) / len(metric_values),
        "train_initial_loss": initial_train_loss,
        "train_final_loss": final_train_loss,
        "train_loss_curve": train_loss_curve,
        "train_sufficiency_curve": train_sufficiency_curve,
        "lm_loss": sum(lm_losses) / len(lm_losses),
        "answer_loss": sum(answer_losses) / len(answer_losses),
        "sufficiency_loss": sum(sufficiency_losses) / len(sufficiency_losses),
        "average_train_step_time_seconds": sum(train_step_times) / len(train_step_times),
        "average_eval_step_time_seconds": sum(eval_step_times) / len(eval_step_times),
        "tokens_per_second": tokens_per_second,
        "decode_tokens_per_second": decode_tokens_per_second,
        "peak_memory_bytes": peak_memory_bytes,
        "throughput_per_memory": compute_throughput_per_memory(tokens_per_second, peak_memory_bytes),
        "bank_read_slots": sum(bank_read_slots) / len(bank_read_slots),
        "segment_count": sum(segment_counts) / len(segment_counts),
        "notes": (
            f"Benchmark `{benchmark_config.family}` ran with variant `{model_config.experiment_name()}` "
            f"at block_size={model_config.effective_block_size()}, refresh_slots={model_config.effective_refresh_slots()}, "
            f"bank_size={model_config.bank_size}, upper_layer_only_refresh={model_config.upper_layer_only_refresh}."
        ),
    }


def load_benchmark_config(config_path: str | None, benchmark: str | None, mode: str) -> SyntheticBenchmarkConfig:
    """Loads a synthetic benchmark config from JSON or CLI defaults."""
    if config_path is not None:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            return SyntheticBenchmarkConfig(**json.load(handle))
    if benchmark is None:
        raise ValueError("benchmark must be set when no config file is provided")
    return SyntheticBenchmarkConfig(family=benchmark, mode=mode)


def parse_args() -> argparse.Namespace:
    """Parses the synthetic benchmark CLI."""
    parser = argparse.ArgumentParser(description="Run one synthetic long-context benchmark.")
    parser.add_argument("--benchmark", default=None, help="Benchmark family: delayed_kv, needle, delayed_copy")
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
    parser.add_argument("--output-dir", default=None, help="Optional artifact directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark_config = load_benchmark_config(args.config, args.benchmark, args.mode)
    model_config = build_model_config(args.variant, benchmark_config, args.model_config)
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
    print(result)
    if args.output_dir is not None:
        write_run_json(args.output_dir, result)
        write_aggregate_csv(args.output_dir, [result])
        write_markdown_summary(args.output_dir, [result])
