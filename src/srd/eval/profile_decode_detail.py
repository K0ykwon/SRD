"""Profile prefill and incremental decode throughput on matched cells."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from srd.config import SRDConfig
from srd.data.synthetic_benchmarks import SyntheticBenchmarkConfig, make_synthetic_dataset
from srd.eval.benchmark_runner import _build_set_a_model_config, _task_config_for_context
from srd.modeling.factory import build_model


def _cpu_bytes_fallback(tensor: torch.Tensor, d_model: int) -> float:
    return float(tensor.numel() * max(d_model, 1) * 4)


def _measure_incremental_decode(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    d_model: int,
    decode_steps: int | None = None,
) -> dict[str, float]:
    answer_start = int(batch["answer_positions"][0, 0].item())
    answer_len = int(batch["answer_positions"].size(1))
    target_decode_steps = max(answer_len, int(decode_steps or 0))
    prefix = batch["input_ids"][:, :answer_start].clone()
    decode_input = prefix

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        if hasattr(model, "prefill") and hasattr(model, "decode_step"):
            state = model.prefill(prefix)
            next_logits = state["next_logits"]
        else:
            outputs = model(prefix)
            next_logits = outputs["logits"][:, -1, :]
            state = None
    prefill_seconds = time.perf_counter() - prefill_start
    if device.type == "cuda":
        prefill_peak_memory_bytes = float(torch.cuda.max_memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        prefill_peak_memory_bytes = _cpu_bytes_fallback(prefix, d_model)

    decode_start = time.perf_counter()
    with torch.inference_mode():
        if state is not None:
            for _ in range(target_decode_steps):
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                state = model.decode_step(next_token, state)
                next_logits = state["next_logits"]
        else:
            for _ in range(target_decode_steps):
                outputs = model(decode_input)
                next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
                decode_input = torch.cat([decode_input, next_token], dim=1)
    decode_seconds = time.perf_counter() - decode_start
    if device.type == "cuda":
        decode_peak_memory_bytes = float(torch.cuda.max_memory_allocated(device))
    else:
        decode_peak_memory_bytes = _cpu_bytes_fallback(batch["answer_tokens"], d_model)

    prefix_tokens = prefix.numel()
    decoded_tokens = target_decode_steps * batch["input_ids"].size(0)
    return {
        "prefix_tokens": float(prefix_tokens),
        "decoded_tokens": float(decoded_tokens),
        "decode_steps": float(target_decode_steps),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill_tokens_per_second": prefix_tokens / max(prefill_seconds, 1e-6),
        "decode_tokens_per_second": decoded_tokens / max(decode_seconds, 1e-6),
        "decode_seconds_per_step": decode_seconds / max(target_decode_steps, 1),
        "prefill_peak_memory_bytes": prefill_peak_memory_bytes,
        "decode_peak_memory_bytes": decode_peak_memory_bytes,
    }


def profile_decode_pair(
    model_config: SRDConfig,
    benchmark_config: SyntheticBenchmarkConfig,
    batch_size: int = 1,
    seed_offset: int = 0,
    decode_steps: int | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = build_model(model_config).to(device)
    model.eval()
    dataset = make_synthetic_dataset(SyntheticBenchmarkConfig(**{**benchmark_config.to_dict(), "split": "test"}))
    batch = dataset.make_batch(seed_offset * batch_size, batch_size, device)
    metrics = _measure_incremental_decode(
        model,
        batch,
        device,
        model_config.d_model,
        decode_steps=decode_steps,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "variant": model_config.experiment_name(),
        "model_family": model_config.model_family,
        "model_size": model_config.size_name,
        "parameter_count": parameter_count,
        "benchmark": benchmark_config.to_dict(),
        "context_length": benchmark_config.total_length,
        "task_label": benchmark_config.family,
        "batch_size": batch_size,
        **metrics,
    }


def _write_results(output_dir: str | Path, results: list[dict[str, Any]]) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows_path = output_path / "decode_profile_rows.json"
    with rows_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    csv_path = output_path / "decode_profile_summary.csv"
    fieldnames = [
        "variant",
        "model_family",
        "model_size",
        "task_label",
        "context_length",
        "parameter_count",
        "batch_size",
        "prefix_tokens",
        "decoded_tokens",
        "decode_steps",
        "prefill_seconds",
        "decode_seconds",
        "decode_seconds_per_step",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "prefill_peak_memory_bytes",
        "decode_peak_memory_bytes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fieldnames})
    return {"rows_json": str(rows_path), "summary_csv": str(csv_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile decode throughput on matched Set A cells.")
    parser.add_argument("--task", default="delayed_kv")
    parser.add_argument("--model-size", default="compact")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["srd_refresh_sufficiency_detail", "transformer_full"],
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_config, _ = _task_config_for_context(
        Path("configs/experiment/set_a/tasks") / f"{args.task}.json",
        args.context_length,
        args.seed,
    )
    results = []
    for family in args.families:
        model_config = _build_set_a_model_config(family, args.model_size, benchmark_config)
        results.append(
            profile_decode_pair(
                model_config=model_config,
                benchmark_config=benchmark_config,
                batch_size=args.batch_size,
                decode_steps=args.decode_steps,
            )
        )
    paths = _write_results(args.output_dir, results)
    print({"results": len(results), **paths})


if __name__ == "__main__":
    main()
