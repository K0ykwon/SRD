"""Small dataset utilities for synthetic SRD experiments."""

import torch
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset):
    """Yields random token sequences for prototype training and tests."""

    def __init__(self, length: int, seq_len: int, vocab_size: int):
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        del index
        return torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)


class RepeatingPatternDataset(Dataset):
    """Yields easy repeating patterns for tiny train and ablation smoke tests."""

    def __init__(self, length: int, seq_len: int, vocab_size: int, segment_length: int):
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.segment_length = segment_length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        pattern = torch.arange(self.segment_length, dtype=torch.long) % max(self.vocab_size // 2, 2)
        repeated = pattern.repeat((self.seq_len + self.segment_length - 1) // self.segment_length)[: self.seq_len]
        offset = (3 * index) % max(self.vocab_size // 2, 2)
        return (repeated + offset) % self.vocab_size
