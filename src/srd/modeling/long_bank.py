"""Shared bounded long-memory bank with explicit append and merge behavior."""

import torch
from torch import Tensor, nn


class LongMemoryBank(nn.Module):
    """Maintains a shared bank whose size is controlled by explicit compression."""

    def __init__(self, d_model: int, max_slots: int, merge_policy: str = "oldest_pair"):
        super().__init__()
        self.d_model = d_model
        self.max_slots = max_slots
        self.merge_policy = merge_policy

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

    def _compress_lowest_importance_pair(self, bank_states: Tensor) -> Tensor:
        """Merges the lowest-norm adjacent pair as a cheap importance-aware heuristic."""
        if bank_states.size(1) <= 2:
            return self._compress_oldest_pair(bank_states)
        slot_importance = bank_states.norm(dim=-1)
        pair_scores = slot_importance[:, :-1] + slot_importance[:, 1:]
        pair_index = pair_scores.argmin(dim=1)
        updated_rows = []
        for batch_index in range(bank_states.size(0)):
            row = bank_states[batch_index]
            merge_index = int(pair_index[batch_index].item())
            merged = 0.5 * (row[merge_index] + row[merge_index + 1])
            updated_row = torch.cat(
                [
                    row[:merge_index],
                    merged.unsqueeze(0),
                    row[merge_index + 2 :],
                ],
                dim=0,
            )
            updated_rows.append(updated_row)
        return torch.stack(updated_rows, dim=0)

    def write(self, bank_states: Tensor, refresh_states: Tensor) -> Tensor:
        """Appends new bank entries and compresses the oldest history when full."""
        refresh_states = refresh_states.detach()
        if refresh_states.size(1) == 0:
            return bank_states
        if bank_states.size(1) == 0 and refresh_states.size(1) <= self.max_slots:
            return refresh_states[:, -self.max_slots :, :]
        if refresh_states.size(1) == 1 and bank_states.size(1) == self.max_slots and self.max_slots >= 2:
            updated = bank_states.clone()
            updated[:, 0, :] = 0.5 * (bank_states[:, 0, :] + bank_states[:, 1, :])
            if self.max_slots > 2:
                updated[:, 1:-1, :] = bank_states[:, 2:, :]
            updated[:, -1:, :] = refresh_states
            return updated
        updated = torch.cat([bank_states, refresh_states], dim=1)
        while updated.size(1) > self.max_slots:
            if self.merge_policy == "lowest_importance_pair":
                updated = self._compress_lowest_importance_pair(updated)
            else:
                updated = self._compress_oldest_pair(updated)
        return updated
