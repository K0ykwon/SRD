"""Tests for local-only SRD token updates and causal masking."""

import torch

from srd.modeling.local_block import LocalBlock


def test_local_block_preserves_shape() -> None:
    block = LocalBlock(d_model=16, num_heads=4, window_size=2)
    x = torch.randn(2, 8, 16)
    y = block(x)
    assert y.shape == x.shape


def test_local_block_mask_blocks_future_and_distant_past() -> None:
    block = LocalBlock(d_model=8, num_heads=2, window_size=1)
    x = torch.randn(1, 5, 8)
    _, attention = block(x, return_attention=True)
    attention = attention[0, 0]

    assert attention[1, 3].item() == 0.0
    assert attention[3, 0].item() == 0.0
