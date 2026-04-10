"""Simple evaluation metrics for SRD efficiency reporting."""


def compute_throughput_per_memory(tokens_per_second: float, peak_memory_bytes: float) -> float:
    """Returns a simple throughput-per-memory ratio."""
    if peak_memory_bytes <= 0:
        raise ValueError("peak_memory_bytes must be positive")
    return tokens_per_second / peak_memory_bytes
