"""Synthetic long-context benchmarks designed to stress the SRD refresh bottleneck."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List

import torch


SPECIAL_TOKENS = {
    "pad": 0,
    "bind": 1,
    "ask": 2,
    "ans": 3,
    "needle": 4,
    "distractor": 5,
    "source": 6,
    "copy": 7,
    "sep": 8,
    "filler": 9,
}

MIN_VALUE_TOKEN = 16


@dataclass
class SyntheticBenchmarkConfig:
    """Defines a minimal, reproducible synthetic long-context benchmark setup."""

    family: str
    seed: int = 0
    vocab_size: int = 64
    segment_length: int = 8
    context_segments: int = 6
    gap_segments: int = 3
    num_distractors: int = 3
    distractor_density: int = 2
    refresh_count: int = 2
    bank_size: int = 4
    upper_layer_only_refresh: bool = True
    local_window: int = 4
    pattern_length: int = 3
    symbol_pool_size: int = 8
    mode: str = "easy"

    def to_dict(self) -> dict:
        """Returns a plain dictionary for result serialization."""
        return asdict(self)

    @property
    def sequence_length(self) -> int:
        """Returns the fixed token length used for this benchmark."""
        return self.segment_length * self.context_segments


@dataclass
class BenchmarkSample:
    """Represents one synthetic sequence and the answer span to score."""

    input_ids: List[int]
    answer_positions: List[int]
    answer_tokens: List[int]
    metric_name: str
    metadata: Dict[str, int | str]


def _make_rng(seed: int, index: int) -> random.Random:
    """Builds a reproducible RNG for a sample index."""
    return random.Random(seed + 1009 * index)


def _sample_value_tokens(rng: random.Random, vocab_size: int, count: int) -> List[int]:
    """Samples ordinary content tokens without using reserved ids."""
    upper = list(range(MIN_VALUE_TOKEN, vocab_size))
    return rng.sample(upper, count)


def _sample_benchmark_tokens(
    rng: random.Random,
    vocab_size: int,
    count: int,
    symbol_pool_size: int,
) -> List[int]:
    """Samples from a restricted symbol pool so easy modes remain learnable."""
    max_pool = max(count, symbol_pool_size)
    upper = list(range(MIN_VALUE_TOKEN, min(vocab_size, MIN_VALUE_TOKEN + max_pool)))
    return rng.sample(upper, count)


def _fill_to_length(tokens: List[int], length: int, filler_token: int = SPECIAL_TOKENS["filler"]) -> List[int]:
    """Pads or trims a token list to a fixed length."""
    if len(tokens) > length:
        return tokens[:length]
    return tokens + [filler_token] * (length - len(tokens))


class DelayedKeyValueRetrievalBenchmark:
    """Generates causal delayed key-value lookup sequences."""

    def __init__(self, config: SyntheticBenchmarkConfig):
        self.config = config

    def sample(self, index: int) -> BenchmarkSample:
        rng = _make_rng(self.config.seed, index)
        hard = self.config.mode == "hard"
        num_bindings = self.config.num_distractors + 1
        sampled = _sample_benchmark_tokens(
            rng,
            self.config.vocab_size,
            2 * num_bindings,
            self.config.symbol_pool_size if not hard else max(24, self.config.symbol_pool_size),
        )
        keys = sampled[:num_bindings]
        values = sampled[num_bindings:]
        query_index = rng.randrange(num_bindings)
        query_key = keys[query_index]
        answer_value = values[query_index]

        bindings: List[int] = []
        for binding_index in range(num_bindings):
            bindings.extend([SPECIAL_TOKENS["bind"], keys[binding_index], values[binding_index]])
            if hard:
                bindings.append(SPECIAL_TOKENS["sep"])

        query_segment_index = min(self.config.context_segments - 1, 1 + self.config.gap_segments)
        query_start = query_segment_index * self.config.segment_length
        prefix = _fill_to_length(bindings + [SPECIAL_TOKENS["filler"]] * (self.config.gap_segments * self.config.segment_length), query_start)

        query_segment = [
            SPECIAL_TOKENS["ask"],
            query_key,
            SPECIAL_TOKENS["ans"],
            answer_value,
        ]
        input_ids = _fill_to_length(prefix + query_segment, self.config.sequence_length)
        answer_position = len(prefix) + 3
        return BenchmarkSample(
            input_ids=input_ids,
            answer_positions=[answer_position],
            answer_tokens=[answer_value],
            metric_name="accuracy",
            metadata={
                "query_key": query_key,
                "answer_value": answer_value,
                "num_bindings": num_bindings,
                "mode": self.config.mode,
            },
        )


class MultiSegmentNeedleBenchmark:
    """Generates sparse multi-segment needle retrieval sequences."""

    def __init__(self, config: SyntheticBenchmarkConfig):
        self.config = config

    def sample(self, index: int) -> BenchmarkSample:
        rng = _make_rng(self.config.seed, index)
        needle_token = _sample_benchmark_tokens(
            rng,
            self.config.vocab_size,
            1,
            self.config.symbol_pool_size,
        )[0]
        needle_segment = rng.randrange(max(1, self.config.context_segments - 1))
        segments: List[List[int]] = []
        for segment_index in range(self.config.context_segments - 1):
            segment: List[int] = []
            for _ in range(self.config.distractor_density):
                distractor = _sample_benchmark_tokens(
                    rng,
                    self.config.vocab_size,
                    1,
                    self.config.symbol_pool_size,
                )[0]
                segment.extend([SPECIAL_TOKENS["distractor"], distractor])
            if segment_index == needle_segment:
                segment.extend([SPECIAL_TOKENS["needle"], needle_token])
            segments.append(_fill_to_length(segment, self.config.segment_length))

        query_segment = _fill_to_length(
            [SPECIAL_TOKENS["ask"], SPECIAL_TOKENS["needle"], SPECIAL_TOKENS["ans"], needle_token],
            self.config.segment_length,
        )
        input_ids = [token for segment in segments for token in segment] + query_segment
        answer_position = len(input_ids) - self.config.segment_length + 3
        return BenchmarkSample(
            input_ids=input_ids,
            answer_positions=[answer_position],
            answer_tokens=[needle_token],
            metric_name="accuracy",
            metadata={
                "needle_segment": needle_segment,
                "needle_token": needle_token,
                "distractor_density": self.config.distractor_density,
            },
        )


class DelayedCopyBenchmark:
    """Generates delayed copy tasks that require multi-token exact-match generation."""

    def __init__(self, config: SyntheticBenchmarkConfig):
        self.config = config

    def sample(self, index: int) -> BenchmarkSample:
        rng = _make_rng(self.config.seed, index)
        pattern = _sample_benchmark_tokens(
            rng,
            self.config.vocab_size,
            self.config.pattern_length,
            self.config.symbol_pool_size,
        )
        source_segment = _fill_to_length([SPECIAL_TOKENS["source"], *pattern], self.config.segment_length)
        filler_segment = _fill_to_length([], self.config.segment_length)
        query_prefix = [SPECIAL_TOKENS["copy"], SPECIAL_TOKENS["ans"]]
        query_segment = _fill_to_length(query_prefix + pattern, self.config.segment_length)

        query_segment_index = min(self.config.context_segments - 1, 1 + self.config.gap_segments)
        segments: List[List[int]] = []
        for segment_index in range(self.config.context_segments):
            if segment_index == 0:
                segments.append(source_segment)
            elif segment_index == query_segment_index:
                segments.append(query_segment)
            else:
                segments.append(filler_segment[:])
        input_ids = [token for segment in segments for token in segment]
        start = query_segment_index * self.config.segment_length + len(query_prefix)
        answer_positions = list(range(start, start + len(pattern)))
        return BenchmarkSample(
            input_ids=input_ids,
            answer_positions=answer_positions,
            answer_tokens=pattern,
            metric_name="exact_match",
            metadata={
                "pattern_length": len(pattern),
                "delay_segments": query_segment_index - 1,
            },
        )


BENCHMARK_FAMILIES: Dict[str, Callable[[SyntheticBenchmarkConfig], object]] = {
    "delayed_kv": DelayedKeyValueRetrievalBenchmark,
    "needle": MultiSegmentNeedleBenchmark,
    "delayed_copy": DelayedCopyBenchmark,
}


class SyntheticBenchmarkDataset:
    """Provides reproducible batch generation and scoring for one benchmark family."""

    def __init__(self, config: SyntheticBenchmarkConfig):
        if config.family not in BENCHMARK_FAMILIES:
            raise ValueError(f"Unknown benchmark family: {config.family}")
        self.config = config
        self.family = BENCHMARK_FAMILIES[config.family](config)

    def sample(self, index: int) -> BenchmarkSample:
        """Returns one deterministic sample."""
        return self.family.sample(index)

    def make_batch(self, start_index: int, batch_size: int, device: torch.device) -> dict:
        """Builds a fixed-shape batch for train or eval."""
        samples = [self.sample(start_index + offset) for offset in range(batch_size)]
        input_ids = torch.tensor([sample.input_ids for sample in samples], device=device, dtype=torch.long)
        answer_positions = torch.tensor([sample.answer_positions for sample in samples], device=device, dtype=torch.long)
        answer_tokens = torch.tensor([sample.answer_tokens for sample in samples], device=device, dtype=torch.long)
        loss_weights = torch.ones_like(input_ids, dtype=torch.float)
        for batch_index, sample in enumerate(samples):
            for position in sample.answer_positions:
                loss_weights[batch_index, position] = 2.0
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "answer_positions": answer_positions,
            "answer_tokens": answer_tokens,
            "loss_weights": loss_weights,
            "metric_name": samples[0].metric_name,
            "metadata": [sample.metadata for sample in samples],
            "family": self.config.family,
        }

    def score_batch(self, logits: torch.Tensor, batch: dict) -> dict:
        """Scores answer-token predictions using accuracy or exact match."""
        positions = batch["answer_positions"] - 1
        batch_size, answer_len = positions.shape
        predicted = torch.empty_like(batch["answer_tokens"])
        for batch_index in range(batch_size):
            for position_index in range(answer_len):
                predicted[batch_index, position_index] = logits[batch_index, positions[batch_index, position_index], :].argmax(dim=-1)

        if batch["metric_name"] == "exact_match":
            metric_value = float((predicted == batch["answer_tokens"]).all(dim=1).float().mean().item())
        else:
            metric_value = float((predicted == batch["answer_tokens"]).float().mean().item())

        return {
            "metric_name": batch["metric_name"],
            "metric_value": metric_value,
            "predicted_tokens": predicted.detach().cpu(),
        }


def make_synthetic_dataset(config: SyntheticBenchmarkConfig) -> SyntheticBenchmarkDataset:
    """Factory for the requested synthetic benchmark dataset."""
    return SyntheticBenchmarkDataset(config)
