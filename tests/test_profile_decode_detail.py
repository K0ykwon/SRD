from pathlib import Path

import torch

from srd.config import SRDConfig
from srd.data.synthetic_benchmarks import SyntheticBenchmarkConfig
from srd.eval.profile_decode_detail import profile_decode_pair


def test_profile_decode_pair_emits_expected_metrics() -> None:
    config = SRDConfig.preset("transformer_full_tiny")
    config.vocab_size = 64
    config.model_family = "transformer_full"
    config.size_name = "tiny"
    benchmark = SyntheticBenchmarkConfig(
        family="delayed_kv",
        seed=11,
        split="test",
        vocab_size=64,
        segment_length=8,
        segment_count=4,
        total_length=32,
        context_segments=4,
        gap_segments=2,
    )
    result = profile_decode_pair(config, benchmark, batch_size=1, decode_steps=4)
    assert result["variant"] == "transformer_full"
    assert result["prefix_tokens"] > 0
    assert result["decoded_tokens"] > 0
    assert result["decode_steps"] == 4
    assert result["prefill_tokens_per_second"] > 0
    assert result["decode_tokens_per_second"] > 0
    assert result["decode_seconds_per_step"] > 0
    if torch.cuda.is_available():
        assert result["prefill_peak_memory_bytes"] >= 0
        assert result["decode_peak_memory_bytes"] >= 0
