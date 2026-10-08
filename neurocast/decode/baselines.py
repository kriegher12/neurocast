"""Decoders that never see the brain. Every brain score is read against these.

Two sources of word information need no neural data at all:

* **Speech timing.** When the next words start (inside the analysis window), the
  silence before this word, and its position in the sentence. A class-balanced
  multinomial logistic regression on those features, trained on the deep
  listener's *other* sessions, is :class:`TimingClassifier`.
* **Language context.** A word bigram scored as pointwise mutual information,
  ``log p(w | previous) - log p(w)`` (:class:`BigramPMI`), so word frequency
  earns nothing under balanced accuracy. It is given the *true* previous word, so
  it measures headroom, not a deployable decoder.

The lost version of this project found timing alone at 0.61-0.67 BAcc@10 on the
held-out chapter -- above the best published cross-subject brain decoder -- and
language plus timing at 0.80. ``scripts/run_libribrain_audit.py`` re-measures
both.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from .metrics import balanced_accuracy_at_k
from .words import TIMING_COLUMNS

__all__ = ["TimingClassifier", "BigramPMI", "combine_scores", "tune_weight"]


class TimingClassifier:
    """Class-balanced multinomial logistic regression on binned timing features.

    Each feature contributes a one-hot of ``n_bins`` quantile bins (edges fitted
    on training data), its standardised raw value, and an "absent" flag -- a next
    word that starts after the window ends is *absent*, not zero.
    """

    def __init__(self, n_bins: int = 12, C: float = 1.0, max_iter: int = 3000) -> None:
        self.n_bins, self.C, self.max_iter = n_bins, C, max_iter
        self.edges: dict[str, np.ndarray] = {}
        self.moments: dict[str, tuple[float, float]] = {}
        self.model = None
        self.n_classes = 0

    def _encode(self, df) -> np.ndarray:
        cols = []
        for c in TIMING_COLUMNS:
            v = df[c].to_numpy(dtype=float)
            present = np.isfinite(v)
            mu, sd = self.moments[c]
            cols.append(np.where(present, (v - mu) / sd, 0.0)[:, None])
            cols.append((~present).astype(float)[:, None])
            b = np.searchsorted(self.edges[c], np.where(present, v, np.inf), side="right")
            onehot = np.zeros((len(v), len(self.edges[c]) + 1))
            onehot[np.arange(len(v))[present], b[present]] = 1.0
            cols.append(onehot)
        return np.hstack(cols)

    def fit(self, df, y: np.ndarray, n_classes: int) -> "TimingClassifier":
        from sklearn.linear_model import LogisticRegression

        for c in TIMING_COLUMNS:
            v = df[c].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            qs = np.quantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1]) if v.size else np.array([])
            self.edges[c] = np.unique(qs)
            self.moments[c] = (float(v.mean()) if v.size else 0.0,
                               float(v.std()) + 1e-9 if v.size else 1.0)
        self.n_classes = n_classes
        self.model = LogisticRegression(C=self.C, class_weight="balanced",
                                        max_iter=self.max_iter)
        self.model.fit(self._encode(df), np.asarray(y))
        return self

    def log_proba(self, df) -> np.ndarray:
        """``(n, n_classes)``; classes never seen in training get -inf."""
        lp = self.model.predict_log_proba(self._encode(df))
        out = np.full((len(df), self.n_classes), -np.inf)
        out[:, self.model.classes_] = lp
        return out


class BigramPMI:
    """Word bigram with Dirichlet smoothing toward the unigram, scored as PMI.

    ``p(w | v) = (c(v, w) + beta * p(w)) / (c(v) + beta)`` and the score is
    ``log p(w | v) - log p(w)``: what the previous word says beyond frequency.
    """

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta
        self.uni: Counter = Counter()
        self.bi: Counter = Counter()
        self.ctx: Counter = Counter()
        self.n = 0

    def fit(self, sequences: Iterable[list[str]]) -> "BigramPMI":
        for seq in sequences:
            prev = "<s>"
            for w in seq:
                self.uni[w] += 1
                self.bi[(prev, w)] += 1
                self.ctx[prev] += 1
                prev = w
        self.n = sum(self.uni.values())
        return self

    def p_uni(self, w: str) -> float:
        return (self.uni[w] + 1.0) / (self.n + len(self.uni) + 1.0)

    def pmi(self, prev_words: Iterable[str], vocab: list[str]) -> np.ndarray:
        pu = np.array([self.p_uni(w) for w in vocab])
        rows = []
        cache: dict[str, np.ndarray] = {}
        for v in prev_words:
            if v not in cache:
                c = np.array([self.bi[(v, w)] for w in vocab], dtype=float)
                cache[v] = np.log((c + self.beta * pu) / (self.ctx[v] + self.beta)) - np.log(pu)
            rows.append(cache[v])
        return np.vstack(rows)


def combine_scores(*parts: tuple[np.ndarray, float]) -> np.ndarray:
    """``sum(weight * scores)``, treating -inf as a very low score rather than NaN."""
    total = None
    for s, w in parts:
        s = np.where(np.isfinite(s), s, -1e6)
        total = w * s if total is None else total + w * s
    return total


def tune_weight(base: np.ndarray, extra: np.ndarray, y: np.ndarray,
                grid: Iterable[float] = tuple(np.logspace(-2, 1.5, 15)), k: int = 10) -> float:
    """Weight on ``extra`` maximising BAcc@k on the data given (use the FIT split)."""
    best = max(grid, key=lambda lam: balanced_accuracy_at_k(combine_scores((base, 1.0), (extra, lam)), y, k))
    return float(best)
