"""Tests for refresh-state interaction with the long-memory bank."""

import torch

from srd.modeling.refresh_block import RefreshBlock


def test_refresh_block_preserves_refresh_shape() -> None:
    block = RefreshBlock(d_model=16, num_heads=4)
    refresh = torch.randn(2, 3, 16)
    bank = torch.randn(2, 5, 16)
    updated, trace = block(refresh, bank)
    assert updated.shape == refresh.shape
    assert trace["bank_used"] is True
    assert trace["bank_read_slots"] == 3


def test_refresh_block_handles_empty_bank() -> None:
    block = RefreshBlock(d_model=16, num_heads=4)
    refresh = torch.randn(2, 2, 16)
    bank = torch.empty(2, 0, 16)
    updated, trace = block(refresh, bank)
    assert updated.shape == refresh.shape
    assert trace["bank_used"] is False
