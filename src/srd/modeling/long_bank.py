"""Shared bounded long-memory bank with explicit append and merge behavior."""

import torch
from torch import Tensor, nn


class LongMemoryBank(nn.Module):
    """Maintains a shared bank whose size is controlled by explicit compression."""

    def __init__(self, d_model: int, max_slots: int):
        super().__init__()
        self.d_model = d_model
        self.max_slots = max_slots

    def empty(self, batch_size: int, device: torch.device) -> Tensor:
        """Creates an empty bank tensor for a batch."""
        return torch.empty(batch_size, 0, self.d_model, device=device)

    def read(self, bank_states: Tensor) -> Tensor:
        """Returns bank states directly for refresh-only access."""
        return bank_states

    def _compress_oldest_pair(self, bank_states: Tensor) -> Tensor:
        """Merges the two oldest entries into one averaged slot."""
        merged = 0.5 * (bank_states[:, :1, :] + bank_states[:, 1:2, :])
        return torch.cat([merged, bank_states[:, 2:, :]], dim=1)

    def write(self, bank_states: Tensor, refresh_states: Tensor) -> Tensor:
        """Appends new bank entries and compresses the oldest history when full."""
        updated = torch.cat([bank_states, refresh_states.detach()], dim=1)
        while updated.size(1) > self.max_slots:
            updated = self._compress_oldest_pair(updated)
        return updated
