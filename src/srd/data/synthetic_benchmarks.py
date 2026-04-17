"""Synthetic long-context benchmarks for SRD structural validation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List

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
    "rule": 10,
    "value": 11,
    "summary": 12,
    "detail": 13,
    "hop": 14,
    "query": 15,
}

MIN_VALUE_TOKEN = 32
SPLIT_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}
FAMILY_ALIASES = {
    "needle": "needle_retrieval",
    "needle_retrieval": "needle_retrieval",
    "delayed_kv": "easy_kv",
    "easy_kv": "easy_kv",
    "binding_kv": "binding_kv",
    "binding_lite_kv": "easy_kv",
    "binding_heavy_kv": "binding_kv",
    "delayed_copy": "delayed_copy",
    "mixed_dependency": "mixed_dependency",
    "multi_hop_segment_reasoning": "multi_hop_segment_reasoning",
}


@dataclass
class SyntheticBenchmarkConfig:
    """Defines a reproducible synthetic long-context benchmark setup."""

    family: str
    seed: int = 0
    split: str = "train"
    split_sizes: dict[str, int] = field(default_factory=lambda: {"train": 1000, "val": 128, "test": 128})
    vocab_size: int = 64
    total_length: int = 0
    segment_length: int = 8
    segment_count: int = 0
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
    answer_span_length: int = 1
    needle_span_length: int = 1
    hop_count: int = 3
    copy_span_length: int = 0
    num_keys: int = 0
    num_distractor_keys: int = 0
    detail_span_length: int = 4
    summary_rule_count: int = 4
    num_candidate_rules: int = 8
    exact_recovery_required: bool = False

    def __post_init__(self) -> None:
        self.family = FAMILY_ALIASES.get(self.family, self.family)
        if self.family not in FAMILY_ALIASES.values():
            raise ValueError(f"Unknown benchmark family: {self.family}")
        if self.segment_count <= 0:
            self.segment_count = self.context_segments
        if self.context_segments <= 0:
            self.context_segments = self.segment_count
        if self.total_length <= 0:
            self.total_length = self.segment_length * self.segment_count
        if self.segment_count <= 0 and self.segment_length > 0:
            self.segment_count = max(1, self.total_length // self.segment_length)
        if self.context_segments != self.segment_count:
            self.context_segments = self.segment_count
        if self.copy_span_length <= 0:
            self.copy_span_length = self.pattern_length
        if self.num_keys <= 0:
            self.num_keys = self.num_distractors + 1
        if self.num_distractor_keys <= 0:
            self.num_distractor_keys = max(0, self.num_keys - 1)
        if self.answer_span_length <= 0:
            self.answer_span_length = 1
        if self.total_length != self.segment_length * self.segment_count:
            self.total_length = self.segment_length * self.segment_count
        if self.split not in SPLIT_OFFSETS:
            raise ValueError(f"split must be one of {sorted(SPLIT_OFFSETS)}")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sequence_length(self) -> int:
        return self.total_length


@dataclass
class BenchmarkSample:
    """Represents one synthetic sequence and the answer span to score."""

    input_ids: List[int]
    answer_positions: List[int]
    answer_tokens: List[int]
    metric_name: str
    metadata: Dict[str, Any]


def _make_rng(config: SyntheticBenchmarkConfig, index: int) -> random.Random:
    offset = SPLIT_OFFSETS[config.split]
    return random.Random(config.seed + offset + 1009 * index)


def _sample_tokens(rng: random.Random, vocab_size: int, count: int, symbol_pool_size: int) -> List[int]:
    max_pool = max(count, symbol_pool_size)
    upper = list(range(MIN_VALUE_TOKEN, min(vocab_size, MIN_VALUE_TOKEN + max_pool)))
    if len(upper) < count:
        upper = list(range(MIN_VALUE_TOKEN, vocab_size))
    if not upper:
        upper = list(range(max(vocab_size - max(count, 1), 0), vocab_size))
    if len(upper) >= count:
        return rng.sample(upper, count)
    return [rng.choice(upper) for _ in range(count)]


def _fill_to_length(tokens: List[int], length: int, filler_token: int = SPECIAL_TOKENS["filler"]) -> List[int]:
    if len(tokens) > length:
        return tokens[:length]
    return tokens + [filler_token] * (length - len(tokens))


class DelayedKeyValueRetrievalBenchmark:
    """Generates causal delayed key-value lookup sequences."""

    def __init__(self, config: SyntheticBenchmarkConfig):
        self.config = config

    def sample(self, index: int) -> BenchmarkSample:
        rng = _make_rng(self.config, index)
        key_count = self.config.num_keys
        value_length = max(1, self.config.answer_span_length)
        keys = _sample_tokens(
            rng,
            self.config.vocab_size,
            key_count,
            self.config.symbol_pool_size if self.config.family == "easy_kv" else max(24, self.config.symbol_pool_size),
        )
        values = self._sample_values(rng, key_count, value_length)
        query_index = rng.randrange(key_count)
        query_key = keys[query_index]
        answer_value = values[query_index]

        segments: List[List[int]] = []
        first_segment = []
        for idx in range(key_count):
            first_segment.extend([SPECIAL_TOKENS["bind"], keys[idx], *values[idx], SPECIAL_TOKENS["sep"]])
        segments.append(_fill_to_length(first_segment, self.config.segment_length))

        query_segment_index = min(self.config.segment_count - 1, 1 + self.config.gap_segments)
        for segment_index in range(1, self.config.segment_count):
            if segment_index == query_segment_index:
                query_segment = [SPECIAL_TOKENS["ask"], query_key, SPECIAL_TOKENS["ans"], *answer_value]
                segments.append(_fill_to_length(query_segment, self.config.segment_length))
            else:
                distractor_segment = []
                for _ in range(self.config.distractor_density):
                    d_key = _sample_tokens(rng, self.config.vocab_size, 1, self.config.symbol_pool_size)[0]
                    d_val = self._sample_values(rng, 1, value_length)[0]
                    distractor_segment.extend([SPECIAL_TOKENS["bind"], d_key, *d_val, SPECIAL_TOKENS["sep"]])
                segments.append(_fill_to_length(distractor_segment, self.config.segment_length))

        input_ids = [token for segment in segments for token in segment]
        answer_start = query_segment_index * self.config.segment_length + 3
        answer_positions = list(range(answer_start, answer_start + value_length))
        return BenchmarkSample(
            input_ids=input_ids,
            answer_positions=answer_positions,
            answer_tokens=answer_value,
            metric_name="accuracy",
            metadata={
                "task_name": self.config.family,
                "split": self.config.split,
                "query_key": query_key,
                "value_length": value_length,
                "binding_task": self.config.family == "binding_kv",
                "answer_tokens": answer_value,
                "candidate_values": values,
                "support_segment_indices": [0],
                "query_segment_index": query_segment_index,
                "requires_exact_recovery": False,
            },
        )

    def _sample_values(self, rng: random.Random, key_count: int, value_length: int) -> List[List[int]]:
        if self.config.family == "binding_kv":
            if value_length == 1:
                suffixes = _sample_tokens(rng, self.config.vocab_size, key_count, max(24, self.config.symbol_pool_size))
                return [[suffix] for suffix in suffixes]
            shared_prefix = _sample_tokens(
                rng,
                self.config.vocab_size,
                value_length - 1,
                max(24, self.config.symbol_pool_size),
            )
            suffixes = _sample_tokens(rng, self.config.vocab_size, key_count, max(24, self.config.symbol_pool_size))
            return [shared_prefix + [suffix] for suffix in suffixes]

        sampled = _sample_tokens(
            rng,
            self.config.vocab_size,
            key_count * value_length,
            self.config.symbol_pool_size,
        )
        return [
            sampled[index * value_length : (index + 1) * value_length]
            for index in range(key_count)
        ]


class MultiSegmentNeedleBenchmark:
    """Generates sparse long-context needle retrieval sequences."""

    def __init__(self, config: SyntheticBenchmarkConfig):
        self.config = config

    def sample(self, index: int) -> BenchmarkSample:
        rng = _make_rng(self.config, index)
        needle = _sample_tokens(rng, self.config.vocab_size, self.config.answer_span_length, self.config.symbol_pool_size)
        needle_segment = rng.randrange(max(1, self.config.segment_count - 1))
        segments: List[List[int]] = []
        for segment_index in range(self.config.segment_count):
            if segment_index == self.config.segment_count - 1:
                query_segment = [SPECIAL_TOKENS["ask"], SPECIAL_TOKENS["needle"], SPECIAL_TOKENS["ans"], *needle]
                segments.append(_fill_to_length(query_segment, self.config.segment_length))
                continue
            segment: List[int] = []
            for _ in range(self.config.distractor_density):
                distractor = _sample_tokens(rng, self.config.vocab_size, 1, self.config.symbol_pool_size)[0]
                segment.extend([SPECIAL_TOKENS["distractor"], distractor])
            if segment_index == needle_segment:
                segment.extend([SPECIAL_TOKENS["needle"], *needle])
            segments.append(_fill_to_length(segment, self.config.segment_length))

        input_ids = [token for segment in segments for token in segment]
        answer_start = (self.config.segment_count - 1) * self.config.segment_length + 3
        return BenchmarkSample(
            input_ids=input_ids,
            answer_positions=list(range(answer_start, answer_start + len(needle))),
            answer_tokens=needle,
            metric_name="accuracy",
            metadata={
                "task_name": self.config.family,
                "split": self.config.split,
                "needle_segment": needle_segment,
                "support_segment_indices": [needle_segment],
                "query_segment_index": self.config.segment_count - 1,
                "requires_exact_recovery": False,
            },
        )


class DelayedCopyBenchmark:
    """Generates delayed exact-copy tasks."""

    def __init__(self, config: SyntheticBenchmarkConfig):
        self.config = config

    def sample(self, index: int) -> BenchmarkSample:
        rng = _make_rng(self.config, index)
        pattern = _sample_tokens(rng, self.config.vocab_size, self.config.copy_span_length, self.config.symbol_pool_size)
        query_segment_index = min(self.config.segment_count - 1, 1 + self.config.gap_segments)
        segments: List[List[int]] = []
        for segment_index in range(self.config.segment_count):
            if segment_index == 0:
                segments.append(_fill_to_length([SPECIAL_TOKENS["source"], *pattern], self.config.segment_length))
            elif segment_index == query_segment_index:
                segments.append(_fill_to_length([SPECIAL_TOKENS["copy"], SPECIAL_TOKENS["ans"], *pattern], self.config.segment_length))
            else:
                filler = []
                for _ in range(self.config.distractor_density):
                    filler.extend(_sample_tokens(rng, self.config.vocab_size, 1, self.config.symbol_pool_size))
                segments.append(_fill_to_length(filler, self.config.segment_length))

        input_ids = [token for segment in segments for token in segment]
        answer_start = query_segment_index * self.config.segment_length + 2
        return BenchmarkSample(
            input_ids=input_ids,
            answer_positions=list(range(answer_start, answer_start + len(pattern))),
            answer_tokens=pattern,
            metric_name="exact_match",
            metadata={
                "task_name": self.config.family,
                "split": self.config.split,
                "support_segment_indices": [0],
                "query_segment_index": query_segment_index,
                "requires_exact_recovery": True,
            },
        )


class MixedDependencyBenchmark:
    """Generates tasks with summary-friendly and exact-detail subparts."""

    def __init__(self, config: SyntheticBenchmarkConfig):
        self.config = config

    def sample(self, index: int) -> BenchmarkSample:
        rng = _make_rng(self.config, index)
        summary_rule = _sample_tokens(rng, self.config.vocab_size, 2, self.config.symbol_pool_size)
        detail_span = _sample_tokens(rng, self.config.vocab_size, self.config.detail_span_length, self.config.symbol_pool_size)
        summary_answer = [summary_rule[1]]
        answer_tokens = summary_answer + detail_span
        query_segment_index = self.config.segment_count - 1
        detail_segment_index = min(self.config.segment_count - 2, 1 + self.config.gap_segments)
        segments: List[List[int]] = []
        for segment_index in range(self.config.segment_count):
            if segment_index == 0:
                rule_segment = [SPECIAL_TOKENS["rule"], summary_rule[0], summary_rule[1], SPECIAL_TOKENS["summary"]]
                segments.append(_fill_to_length(rule_segment, self.config.segment_length))
            elif segment_index == detail_segment_index:
                detail_segment = [SPECIAL_TOKENS["detail"], *detail_span]
                segments.append(_fill_to_length(detail_segment, self.config.segment_length))
            elif segment_index == query_segment_index:
                query_segment = [SPECIAL_TOKENS["query"], summary_rule[0], SPECIAL_TOKENS["ans"], *answer_tokens]
                segments.append(_fill_to_length(query_segment, self.config.segment_length))
            else:
                filler = []
                for _ in range(self.config.distractor_density):
                    filler.extend(_sample_tokens(rng, self.config.vocab_size, 1, self.config.symbol_pool_size))
                segments.append(_fill_to_length(filler, self.config.segment_length))

        input_ids = [token for segment in segments for token in segment]
        answer_start = query_segment_index * self.config.segment_length + 3
        return BenchmarkSample(
            input_ids=input_ids,
            answer_positions=list(range(answer_start, answer_start + len(answer_tokens))),
            answer_tokens=answer_tokens,
            metric_name="joint_accuracy",
            metadata={
                "task_name": self.config.family,
                "split": self.config.split,
                "summary_answer_length": len(summary_answer),
                "detail_answer_length": len(detail_span),
                "support_segment_indices": [0, detail_segment_index],
                "query_segment_index": query_segment_index,
                "requires_exact_recovery": True,
            },
        )


class MultiHopSegmentReasoningBenchmark:
    """Generates multi-hop rule-composition tasks across separated segments."""

    def __init__(self, config: SyntheticBenchmarkConfig):
        self.config = config

    def sample(self, index: int) -> BenchmarkSample:
        rng = _make_rng(self.config, index)
        hop_tokens = _sample_tokens(rng, self.config.vocab_size, self.config.hop_count + 1, self.config.symbol_pool_size)
        start_token = hop_tokens[0]
        answer_token = hop_tokens[-1]
        segments: List[List[int]] = []
        support_segments = []
        for hop_index in range(self.config.hop_count):
            support_segments.append(hop_index)
            hop_segment = [SPECIAL_TOKENS["hop"], hop_tokens[hop_index], hop_tokens[hop_index + 1], SPECIAL_TOKENS["rule"]]
            segments.append(_fill_to_length(hop_segment, self.config.segment_length))

        while len(segments) < self.config.segment_count - 1:
            filler = []
            for _ in range(self.config.distractor_density):
                filler.extend(_sample_tokens(rng, self.config.vocab_size, 1, self.config.symbol_pool_size))
            segments.append(_fill_to_length(filler, self.config.segment_length))

        query_segment = [SPECIAL_TOKENS["query"], start_token, SPECIAL_TOKENS["ans"], answer_token]
        segments = segments[: self.config.segment_count - 1] + [_fill_to_length(query_segment, self.config.segment_length)]
        input_ids = [token for segment in segments for token in segment]
        answer_position = (self.config.segment_count - 1) * self.config.segment_length + 3
        return BenchmarkSample(
            input_ids=input_ids,
            answer_positions=[answer_position],
            answer_tokens=[answer_token],
            metric_name="accuracy",
            metadata={
                "task_name": self.config.family,
                "split": self.config.split,
                "hop_count": self.config.hop_count,
                "support_segment_indices": support_segments,
                "query_segment_index": self.config.segment_count - 1,
                "requires_exact_recovery": False,
            },
        )


BENCHMARK_FAMILIES: Dict[str, Callable[[SyntheticBenchmarkConfig], object]] = {
    "easy_kv": DelayedKeyValueRetrievalBenchmark,
    "binding_kv": DelayedKeyValueRetrievalBenchmark,
    "needle_retrieval": MultiSegmentNeedleBenchmark,
    "delayed_copy": DelayedCopyBenchmark,
    "mixed_dependency": MixedDependencyBenchmark,
    "multi_hop_segment_reasoning": MultiHopSegmentReasoningBenchmark,
}


class SyntheticBenchmarkDataset:
    """Provides reproducible batch generation and task-specific scoring."""

    def __init__(self, config: SyntheticBenchmarkConfig):
        if config.family not in BENCHMARK_FAMILIES:
            raise ValueError(f"Unknown benchmark family: {config.family}")
        self.config = config
        self.family = BENCHMARK_FAMILIES[config.family](config)

    def sample(self, index: int) -> BenchmarkSample:
        return self.family.sample(index)

    def make_batch(self, start_index: int, batch_size: int, device: torch.device) -> dict:
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
        positions = batch["answer_positions"] - 1
        batch_size, answer_len = positions.shape
        predicted = torch.empty_like(batch["answer_tokens"])
        for batch_index in range(batch_size):
            for position_index in range(answer_len):
                predicted[batch_index, position_index] = logits[batch_index, positions[batch_index, position_index], :].argmax(dim=-1)

        gold = batch["answer_tokens"]
        token_accuracy = float((predicted == gold).float().mean().item())
        exact_match = float((predicted == gold).all(dim=1).float().mean().item())
        normalized_edit_distance = self._normalized_edit_distance(predicted, gold)
        metric_name = batch["metric_name"]

        task_metrics: dict[str, float | dict[str, float]] = {
            "token_accuracy": token_accuracy,
            "exact_match": exact_match,
            "normalized_edit_distance": normalized_edit_distance,
        }
        if batch["family"] in {"easy_kv", "binding_kv"}:
            retrieval_hit = 0.0
            binding_accuracy = exact_match
            for batch_index in range(batch_size):
                predicted_row = predicted[batch_index].tolist()
                candidate_values = batch["metadata"][batch_index].get("candidate_values", [])
                if predicted_row in candidate_values:
                    retrieval_hit += 1.0
            retrieval_hit /= max(batch_size, 1)
            task_metrics["accuracy"] = exact_match
            task_metrics["value_span_exact_match"] = exact_match
            task_metrics["binding_accuracy"] = binding_accuracy
            task_metrics["retrieval_hit"] = retrieval_hit
            task_metrics["token_accuracy"] = token_accuracy
            if batch["family"] == "binding_kv":
                prefix_len = max(gold.size(1) - 1, 0)
                if prefix_len > 0:
                    prefix_accuracy = float((predicted[:, :prefix_len] == gold[:, :prefix_len]).all(dim=1).float().mean().item())
                else:
                    prefix_accuracy = 1.0
                suffix_accuracy = float((predicted[:, -1] == gold[:, -1]).float().mean().item())
                wrong_value_catalog_rate = 0.0
                off_catalog_rate = 0.0
                for batch_index in range(batch_size):
                    predicted_row = predicted[batch_index].tolist()
                    gold_row = gold[batch_index].tolist()
                    candidate_values = batch["metadata"][batch_index].get("candidate_values", [])
                    if predicted_row == gold_row:
                        continue
                    if predicted_row in candidate_values:
                        wrong_value_catalog_rate += 1.0
                    else:
                        off_catalog_rate += 1.0
                task_metrics["prefix_accuracy"] = prefix_accuracy
                task_metrics["suffix_accuracy"] = suffix_accuracy
                task_metrics["wrong_value_catalog_rate"] = wrong_value_catalog_rate / max(batch_size, 1)
                task_metrics["off_catalog_rate"] = off_catalog_rate / max(batch_size, 1)
        elif batch["family"] == "needle_retrieval":
            task_metrics["accuracy"] = token_accuracy
            task_metrics["retrieval_hit_rate"] = exact_match
        elif batch["family"] == "delayed_copy":
            task_metrics["exact_match"] = exact_match
            task_metrics["token_accuracy"] = token_accuracy
            task_metrics["normalized_edit_distance"] = normalized_edit_distance
        elif batch["family"] == "mixed_dependency":
            summary_len = int(batch["metadata"][0].get("summary_answer_length", 1))
            summary_pred = predicted[:, :summary_len]
            summary_gold = gold[:, :summary_len]
            detail_pred = predicted[:, summary_len:]
            detail_gold = gold[:, summary_len:]
            task_metrics["summary_part_accuracy"] = float((summary_pred == summary_gold).all(dim=1).float().mean().item())
            task_metrics["detail_part_accuracy"] = float((detail_pred == detail_gold).all(dim=1).float().mean().item())
            task_metrics["joint_accuracy"] = exact_match
        elif batch["family"] == "multi_hop_segment_reasoning":
            hop_count = float(batch["metadata"][0].get("hop_count", 0))
            task_metrics["accuracy"] = token_accuracy
            task_metrics["per_hop_failure_breakdown"] = {
                "final_answer": float(1.0 - exact_match),
                "hop_count": hop_count,
            }

        metric_value = task_metrics.get(metric_name, exact_match if metric_name == "exact_match" else token_accuracy)
        if isinstance(metric_value, dict):
            metric_value = token_accuracy
        return {
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "task_metrics": task_metrics,
            "predicted_tokens": predicted.detach().cpu(),
        }

    @staticmethod
    def _normalized_edit_distance(predicted: torch.Tensor, gold: torch.Tensor) -> float:
        distances = []
        for pred_row, gold_row in zip(predicted.tolist(), gold.tolist()):
            distances.append(_levenshtein(pred_row, gold_row) / max(len(gold_row), 1))
        return float(sum(distances) / max(len(distances), 1))


def _levenshtein(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, token_b in enumerate(b, start=1):
            current = dp[j]
            if token_a == token_b:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = current
    return dp[-1]


def make_synthetic_dataset(config: SyntheticBenchmarkConfig) -> SyntheticBenchmarkDataset:
    return SyntheticBenchmarkDataset(config)
