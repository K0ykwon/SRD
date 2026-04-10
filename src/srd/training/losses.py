"""Loss helpers for next-token training and explicit refresh sufficiency regularization."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from srd.config import SRDConfig


def compute_answer_loss(logits: Tensor, answer_positions: Tensor, answer_tokens: Tensor) -> Tensor:
    """Computes next-token cross-entropy only on the benchmark answer span."""
    gathered_logits = []
    gathered_targets = []
    for batch_index in range(answer_positions.size(0)):
        for answer_index in range(answer_positions.size(1)):
            target_position = answer_positions[batch_index, answer_index]
            gathered_logits.append(logits[batch_index, target_position - 1, :])
            gathered_targets.append(answer_tokens[batch_index, answer_index])
    stacked_logits = torch.stack(gathered_logits, dim=0)
    stacked_targets = torch.stack(gathered_targets, dim=0)
    return F.cross_entropy(stacked_logits, stacked_targets)


def compute_srd_loss(
    outputs: Dict[str, Tensor],
    labels: Tensor,
    config: SRDConfig,
    token_weights: Tensor | None = None,
) -> Dict[str, Tensor]:
    """Combines next-token loss with optional token weighting and sufficiency loss."""
    logits = outputs["logits"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)

    if token_weights is None:
        weighted_token_losses = token_losses
        nll_loss = weighted_token_losses.mean()
        answer_loss = weighted_token_losses.mean()
    else:
        shift_weights = token_weights[:, 1:].contiguous()
        weighted_token_losses = token_losses * shift_weights
        nll_loss = weighted_token_losses.sum() / shift_weights.sum().clamp_min(1.0)

        answer_mask = shift_weights > 1.0
        if answer_mask.any():
            answer_loss = token_losses[answer_mask].mean()
        else:
            answer_loss = token_losses.mean()

    predicted_summary = outputs["predicted_summary"]
    target_summary = outputs["target_summary"]
    if predicted_summary.numel() == 0:
        refresh_loss = torch.zeros((), device=logits.device)
    else:
        # The first SRD prototype predicts the next segment embedding summary from the
        # current segment refresh output. This keeps the bottleneck explicit and easy to test.
        refresh_loss = F.mse_loss(predicted_summary, target_summary)

    total_loss = nll_loss + config.sufficiency_loss_weight * refresh_loss
    return {
        "loss": total_loss,
        "nll_loss": nll_loss,
        "answer_loss": answer_loss,
        "sufficiency_loss": refresh_loss,
    }
