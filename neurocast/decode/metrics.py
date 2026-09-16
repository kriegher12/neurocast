"""Decoding metrics, matching the 2026 PNPL competition definitions.

The primary metric is **top-10 balanced accuracy** (BAcc@10): per-class recall@10,
averaged over classes. With a 50-word vocabulary, uniform random guessing without
replacement gives 20% in expectation.

Balanced rather than plain accuracy matters for this project specifically. A
language-model prior that re-imports corpus word frequency will systematically
demote rare words, and plain accuracy would reward that. Balanced accuracy
punishes it, which is why :mod:`neurocast.decode.scorer` uses LM *pointwise mutual
information* rather than raw LM log-probability.
"""

from __future__ import annotations

import numpy as np

__all__ = ["topk_predictions", "balanced_accuracy_at_k", "accuracy_at_k", "chance_bacc_at_k"]


def topk_predictions(scores: np.ndarray, k: int) -> np.ndarray:
    """``(n_trials, K)`` scores -> ``(n_trials, k)`` class indices, best first."""
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"expected (n_trials, n_classes), got {scores.shape}")
    k = min(int(k), scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def balanced_accuracy_at_k(scores: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """Mean over classes of recall@k. The PNPL primary metric at ``k=10``.

    Classes absent from ``y`` are skipped rather than counted as zero, matching
    the competition definition and avoiding a penalty for split composition.
    """
    y = np.asarray(y)
    top = topk_predictions(scores, k)
    hit = (top == y[:, None]).any(axis=1)
    recalls = [float(hit[y == c].mean()) for c in np.unique(y)]
    return float(np.mean(recalls)) if recalls else float("nan")


def accuracy_at_k(scores: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    top = topk_predictions(scores, k)
    return float((top == np.asarray(y)[:, None]).any(axis=1).mean())


def chance_bacc_at_k(n_classes: int, k: int = 10) -> float:
    """Expected BAcc@k under uniform random ranking: ``k / n_classes``."""
    return min(float(k) / float(n_classes), 1.0)
