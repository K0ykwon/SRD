"""Tests for bounded long-memory bank behavior and compression."""

import torch

from srd.modeling.long_bank import LongMemoryBank


def test_long_bank_caps_slot_count() -> None:
    bank = LongMemoryBank(d_model=8, max_slots=4)
    state = bank.empty(batch_size=1, device=torch.device("cpu"))
    first = torch.randn(1, 3, 8)
    second = torch.randn(1, 3, 8)
    state = bank.write(state, first)
    state = bank.write(state, second)
    assert state.shape == (1, 4, 8)


def test_long_bank_merges_oldest_entries_when_full() -> None:
    bank = LongMemoryBank(d_model=2, max_slots=2)
    state = bank.empty(batch_size=1, device=torch.device("cpu"))
    state = bank.write(state, torch.tensor([[[1.0, 1.0]]]))
    state = bank.write(state, torch.tensor([[[3.0, 3.0]]]))
    state = bank.write(state, torch.tensor([[[5.0, 5.0]]]))
    expected = torch.tensor([[[2.0, 2.0], [5.0, 5.0]]])
    assert torch.allclose(state, expected)
