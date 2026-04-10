"""Runs a minimal block-refresh SRD forward pass and prints output shapes."""

import torch

from srd.config import SRDConfig
from srd.modeling.factory import build_model


if __name__ == "__main__":
    config = SRDConfig.preset("block_refresh_suf_tiny")
    model = build_model(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 12))
    outputs = model(input_ids)
    summary = {}
    for name, value in outputs.items():
        if isinstance(value, dict):
            summary[name] = value
        else:
            summary[name] = tuple(value.shape)
    print(summary)
