"""Evaluation utilities for ViKIS retrieval and ranking metrics."""

from __future__ import annotations

from typing import Iterable


def hit_rate_at_k(relevants: Iterable[int], predicted: Iterable[int], k: int = 1) -> float:
    """Return 1.0 if any relevant item appears in the first k predictions."""
    predictions = list(predicted)[:k]
    relevant = set(relevants)
    return float(bool(predictions and set(predictions) & relevant))


def reciprocal_rank(relevants: Iterable[int], predicted: Iterable[int]) -> float:
    """Compute MRR-style reciprocal rank for the first relevant hit."""
    relevant = set(relevants)
    for index, item in enumerate(predicted, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


if __name__ == "__main__":
    sample_predictions = [3, 1, 2]
    sample_relevants = {1, 2}
    print({
        "hit_rate_at_1": hit_rate_at_k(sample_relevants, sample_predictions, k=1),
        "reciprocal_rank": reciprocal_rank(sample_relevants, sample_predictions),
    })
