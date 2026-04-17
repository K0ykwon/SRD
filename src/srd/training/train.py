"""Minimal training entry point for tiny SRD experiments and baselines."""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from torch import Tensor

from srd.config import SRDConfig
from srd.modeling.factory import build_model
from srd.training.losses import compute_srd_loss


def make_pattern_batch(
    batch_size: int,
    seq_len: int,
    segment_length: int,
    vocab_size: int,
    device: torch.device,
) -> Tensor:
    """Creates a repeating segment pattern so tiny training runs are learnable."""
    base = torch.arange(segment_length, device=device) % max(vocab_size // 2, 2)
    repeated = base.repeat((seq_len + segment_length - 1) // segment_length)[:seq_len]
    batch = []
    for batch_index in range(batch_size):
        offset = (3 * batch_index) % max(vocab_size // 2, 2)
        batch.append((repeated + offset) % vocab_size)
    return torch.stack(batch, dim=0)


def run_tiny_train(
    config: Optional[SRDConfig] = None,
    steps: int = 8,
    lr: float = 3e-3,
    batch_size: int = 4,
    seq_len: int | None = None,
) -> dict:
    """Runs a tiny multi-step train loop for stability and baseline comparisons."""
    config = config or SRDConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = seq_len or (4 * config.segment_length)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    input_ids = make_pattern_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        segment_length=config.segment_length,
        vocab_size=config.vocab_size,
        device=device,
    )

    history = []
    for _ in range(steps):
        outputs = model(input_ids)
        losses = compute_srd_loss(outputs, input_ids, config)
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()
        history.append(float(losses["loss"].detach().cpu()))

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    return {
        "variant": config.experiment_name(),
        "model_type": config.model_type,
        "steps": steps,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "loss_delta": history[-1] - history[0],
        "nll_loss": float(losses["nll_loss"].detach().cpu()),
        "sufficiency_loss": float(losses["sufficiency_loss"].detach().cpu()),
        "budget_loss": float(losses["budget_loss"].detach().cpu()),
        "gate_entropy_loss": float(losses["gate_entropy_loss"].detach().cpu()),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "block_size": config.effective_block_size(),
        "refresh_slots": config.effective_refresh_slots(),
        "refresh_enabled": config.refresh_enabled if config.model_type in {"srd_block_refresh", "adaptive_slot_srd"} else config.use_refresh,
        "sufficiency_enabled": bool(config.sufficiency_loss_weight > 0),
        "average_active_soft_slots_per_segment": float(outputs["debug"].get("average_active_soft_slots_per_segment", 0.0)),
        "average_gate_value": float(outputs["debug"].get("average_gate_value", 0.0)),
        "memory_bank_slots_used": float(outputs["debug"].get("memory_bank_slots_used", outputs["bank_states"].size(1))),
        "peak_memory_bank_slots_used": float(outputs["debug"].get("peak_memory_bank_slots_used", outputs["bank_states"].size(1))),
        "average_active_hard_slots": float(outputs["debug"].get("average_active_hard_slots", 0.0)),
        "stable": bool(torch.isfinite(torch.tensor(history)).all().item()),
    }


def load_config(preset: str | None = None, config_path: str | None = None) -> SRDConfig:
    """Loads a config from a preset or JSON file."""
    if config_path is not None:
        return SRDConfig.from_json_file(config_path)
    if preset is not None:
        return SRDConfig.preset(preset)
    return SRDConfig.preset("srd_suf_tiny")


def parse_args() -> argparse.Namespace:
    """Parses the tiny training CLI."""
    parser = argparse.ArgumentParser(description="Run a tiny SRD or baseline training loop.")
    parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Named preset: local_tiny, srd_tiny, srd_suf_tiny, block_refresh_local_tiny, "
            "block_refresh_tiny, block_refresh_suf_tiny, adaptive_slot_srd_tiny"
        ),
    )
    parser.add_argument("--config", default=None, help="Optional JSON config file")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = run_tiny_train(
        config=load_config(args.preset, args.config),
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    print(metrics)
