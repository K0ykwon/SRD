"""Runs a tiny block-refresh SRD training smoke loop."""

from srd.config import SRDConfig
from srd.training.train import run_tiny_train


if __name__ == "__main__":
    print(run_tiny_train(config=SRDConfig.preset("block_refresh_suf_tiny")))
