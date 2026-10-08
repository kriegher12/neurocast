"""Does the brain still help once timing, length and loudness are taken away?

Two tests from the lost version of this project, both asking how much of a
decoder's score is word identity rather than the word's acoustic and temporal
envelope.

Timing-matched evaluation (:func:`matched_gain`)
------------------------------------------------
Test trials are grouped into strata that agree on everything the no-brain
baselines exploit: when the next word starts (50 ms bins), the silence since the
previous word, when the second next word starts, and duration and loudness
tertiles. Within a stratum the decoder may choose **only among the words present
in that stratum**, so timing cannot help it pick. Each scorer gets its own
permutation null -- labels shuffled across word *occurrences*, within strata --
and the reported quantity is the gain over that null, compared with the gain
over an unmatched null in the same metric.

The rule, fixed before running (pitch): if less than a third of the gain
survives matching, the decoder is mostly reading timing.

Information beyond covariates (:func:`info_beyond_covariates`)
--------------------------------------------------------------
Held-out log-likelihood gain, in nats per word, of a model given the decoder's
scores *and* the covariates over one given the covariates alone. Every word is
weighted equally; folds are cut by word occurrence; intervals come from a
bootstrap over occurrences. Scores that carry only word frequency must give zero
-- the covariate model's intercepts already know frequency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "strata",
    "matched_score",
    "MatchedResult",
    "matched_gain",
    "BeyondResult",
    "info_beyond_covariates",
]


def _bin(v: np.ndarray, width: float, cap: float) -> np.ndarray:
    out = np.where(np.isfinite(v), np.floor(np.minimum(v, cap) / width), -1)
    return out.astype(int)


def _tertile(v: np.ndarray) -> np.ndarray:
    q = np.nanquantile(v, [1 / 3, 2 / 3])
    return np.searchsorted(q, v, side="right")


def strata(df) -> np.ndarray:
    """One key per trial: next-onset bin, gap bin, second-onset bin, duration and loudness tertiles."""
    parts = [
        _bin(df["next1"].to_numpy(float), 0.05, 0.5),
        _bin(df["gap_prev"].to_numpy(float), 0.05, 0.3),
        _bin(df["next2"].to_numpy(float), 0.05, 0.5),
        _tertile(df["duration"].to_numpy(float)),
        _tertile(df["loudness_db"].to_numpy(float)),
    ]
    return np.array(["|".join(map(str, k)) for k in zip(*parts)])


def _members(groups: np.ndarray) -> list[np.ndarray]:
    """Trial indices of each group, groups in sorted-key order, indices ascending.

    The same sets, in the same order, as ``np.flatnonzero(groups == g)`` over
    ``np.unique(groups)`` -- so permutation draws are unchanged -- but computed
    once: comparing 32,000 string keys per group per permutation took ~12 min
    per scorer on the full test set.
    """
    _, inv = np.unique(groups, return_inverse=True)
    idx = np.argsort(inv, kind="stable")
    return np.split(idx, np.cumsum(np.bincount(inv))[:-1])


def matched_score(scores: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  metric: str = "rank") -> float:
    """Class-balanced score when choosing only among words present in the trial's group.

    ``metric="top1"``: is the true word the top choice among the candidates?
    ``metric="rank"``: normalised rank of the true word among candidates (1 =
    best, 0.5 = chance at any candidate-set size), which has more power.
    Groups with a single candidate word say nothing and are skipped.
    """
    return _score(scores, y, _members(groups), metric)


def _score(scores, y, members, metric):
    hit = np.full(len(y), np.nan)
    for m in members:
        cand = np.unique(y[m])
        if len(cand) < 2:
            continue
        s = scores[np.ix_(m, cand)]
        true_col = np.searchsorted(cand, y[m])
        true_s = s[np.arange(len(m)), true_col]
        if metric == "top1":
            hit[m] = (s.argmax(1) == true_col).astype(float)
        else:
            below = (s < true_s[:, None]).sum(1) + 0.5 * ((s == true_s[:, None]).sum(1) - 1)
            hit[m] = below / (len(cand) - 1)
    ok = np.isfinite(hit)
    return float(np.mean([hit[ok & (y == c)].mean() for c in np.unique(y[ok])]))


def _shuffle_within(y, occ, members, rng):
    """Permute labels across occurrences inside each group; listeners move together."""
    out = y.copy()
    for m in members:
        u, inv = np.unique(occ[m], return_inverse=True)
        first = np.zeros(len(u), dtype=int)
        first[inv[::-1]] = np.arange(len(m))[::-1]
        lab = y[m][first]
        out[m] = lab[rng.permutation(len(u))][inv]
    return out


@dataclass(frozen=True)
class MatchedResult:
    score: float
    null_mean: float
    null_p95: float
    unmatched_score: float
    unmatched_null_mean: float

    @property
    def gain(self) -> float:
        return self.score - self.null_mean

    @property
    def unmatched_gain(self) -> float:
        return self.unmatched_score - self.unmatched_null_mean

    @property
    def surviving(self) -> float:
        """Share of the unmatched gain that survives matching."""
        return self.gain / self.unmatched_gain if self.unmatched_gain > 0 else float("nan")

    @property
    def verdict(self) -> str:
        if not self.score > self.null_p95:
            return "no word information beyond timing"
        return "mostly timing" if self.surviving < 1 / 3 else "survives matching"


def matched_gain(scores, y, occ, groups, *, metric="rank", n_perm=200, seed=0) -> MatchedResult:
    rng = np.random.default_rng(seed)
    groups, one = _members(groups), [np.arange(len(y))]
    s = _score(scores, y, groups, metric)
    null = [_score(scores, _shuffle_within(y, occ, groups, rng), groups, metric)
            for _ in range(n_perm)]
    s0 = _score(scores, y, one, metric)
    null0 = [_score(scores, _shuffle_within(y, occ, one, rng), one, metric)
             for _ in range(max(20, n_perm // 5))]
    return MatchedResult(s, float(np.mean(null)), float(np.percentile(null, 95)),
                         s0, float(np.mean(null0)))


@dataclass(frozen=True)
class BeyondResult:
    nats: float
    lo: float
    hi: float
    dropped: tuple = ()          # classes left out: some trials had no score for them

    def __str__(self) -> str:
        out = f"{self.nats:+.4f} nats/word [{self.lo:+.4f}, {self.hi:+.4f}]"
        return out + (f" ({len(self.dropped)} unscored classes left out)" if self.dropped else "")


#: Decoders mark classes they never saw in training with -1e6 (``LinearPooled``).
UNSCORED = -1e5


def _log_softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max(1, keepdims=True)
    return a - np.log(np.exp(a).sum(1, keepdims=True))


def _fit_beta(logit: np.ndarray, s: np.ndarray, y: np.ndarray,
              grid=np.linspace(-1.0, 4.0, 101)) -> float:
    rows = np.arange(len(y))
    ll = [float(_log_softmax(logit + b * s)[rows, y].sum()) for b in grid]
    return float(grid[int(np.argmax(ll))])


def info_beyond_covariates(cov: np.ndarray, brain: np.ndarray, y: np.ndarray, occ: np.ndarray,
                           *, n_folds: int = 5, n_boot: int = 1000, seed: int = 0,
                           C: float = 1.0) -> BeyondResult:
    """Held-out log-likelihood gain of covariates + brain over covariates alone.

    The brain enters with **one** weight, noisy-channel style:
    ``log p(k) = log_softmax(covariate logit_k + beta * s_k)``, with ``s`` the
    decoder's scores centred per trial and scaled once globally. A first version
    fed all 50 scores to a 50-class regression -- ~3,000 parameters per fold --
    and overfitting made every decoder look *harmful* (about -1 nat/word). With
    one weight there is nothing to overfit, and a score that is constant across
    trials (word frequency) is absorbed by the covariate intercepts: gain zero.

    Classes some trials have no score for (``<= UNSCORED``: the decoder never saw
    them in training) are left out, with their trials. On MEG-MASC, scored by
    leaving one story out, the story-1 decoder had never seen "roy" and "chad"
    (story 1's characters), so only story-1 trials carried the -1e6 marker on
    those two words. The marker told the one-weight model which story a trial
    came from, which predicts the names: +0.029 nats/word from a decoder that
    adds ~0 within every story. The marker's scale also crushed every real score
    to ~0 through the global standardisation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    brain = np.asarray(brain, float)
    unscored = np.flatnonzero((~np.isfinite(brain) | (brain <= UNSCORED)).any(0))
    if unscored.size:
        keep_cls = np.setdiff1d(np.arange(brain.shape[1]), unscored)
        rows = np.isin(y, keep_cls)
        remap = np.full(brain.shape[1], -1)
        remap[keep_cls] = np.arange(len(keep_cls))
        r = info_beyond_covariates(cov[rows], brain[rows][:, keep_cls], remap[y[rows]], occ[rows],
                                   n_folds=n_folds, n_boot=n_boot, seed=seed, C=C)
        return BeyondResult(r.nats, r.lo, r.hi, tuple(int(c) for c in unscored))

    cov = (cov - cov.mean(0)) / (cov.std(0) + 1e-9)
    s = np.asarray(brain, float)
    s = np.where(np.isfinite(s), s, np.nan)
    s = s - np.nanmean(s, axis=1, keepdims=True)
    s = np.nan_to_num(s / (np.nanstd(s) + 1e-12), nan=-10.0)
    classes = np.unique(y)
    yi = np.searchsorted(classes, y)
    d = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_folds).split(cov, y, groups=occ):
        m = LogisticRegression(C=C, max_iter=3000).fit(cov[tr], y[tr])
        logit = np.full((len(y), len(classes)), -30.0)
        logit[:, np.searchsorted(classes, m.classes_)] = m.predict_log_proba(cov)
        sc = s[:, classes] if s.shape[1] > len(classes) else s
        beta = _fit_beta(logit[tr], sc[tr], yi[tr])
        base = _log_softmax(logit[te])[np.arange(len(te)), yi[te]]
        both = _log_softmax(logit[te] + beta * sc[te])[np.arange(len(te)), yi[te]]
        d[te] = both - base

    def balanced(idx):
        yy, dd = y[idx], d[idx]
        return float(np.mean([dd[yy == c].mean() for c in np.unique(yy)]))

    u, inv = np.unique(occ, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(u))]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(u), len(u))
        boots.append(balanced(np.concatenate([members[k] for k in pick])))
    return BeyondResult(balanced(np.arange(len(y))), float(np.percentile(boots, 2.5)),
                        float(np.percentile(boots, 97.5)))
