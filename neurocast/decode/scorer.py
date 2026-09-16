"""Noisy-channel decoding: invert the forward model into a decoder.

    argmax_w   log p(X | w, C)  +  lambda * [ log p_LM(w | h) - log p_LM(w) ]
               ---------------     ------------------------------------------
               neural evidence              LM pointwise mutual information

This module carries the program's highest-risk claim (H4, risk R1), so the
reasoning behind each term is set out explicitly, including one correction to
the original design.

The risk (R1)
-------------
``log p(X | w)`` sums over ~30,600 dimensions (306 channels x 100 samples) while
the information distinguishing one word from another is plausibly under a bit.
The between-candidate score difference is a tiny signal riding on a large term.

That alone would be harmless -- a term common to all candidates cannot change a
ranking. What actually hurts is **candidate-dependent estimation error**: a
learned model of ``p(X | w)`` makes per-word errors in all 30,600 dimensions, not
just the few that carry word information. Those errors do not cancel, their
variance grows with dimension, and they can bury the signal.

Correction to the design: CFG-style subtraction does NOT mitigate R1
--------------------------------------------------------------------
The design proposed ``log p(X|w) - w_cfg * log p(X|empty)`` as a way to "cancel
subject nuisance from the decision". It cannot. ``log p(X|empty)`` is the same for
every candidate in a trial, and subtracting a candidate-independent quantity --
at any weight -- leaves every ranking identical. BAcc@k is a within-trial ranking
metric, so it cannot move at all. ``scripts/validate_decode.py`` confirms this
empirically: top-1 and top-10 identical to 3 decimal places for w_cfg in 0..10.

Where the pointwise-mutual-information form *does* earn its place is **across**
trials: open-set detection ("is any word present?"), abstention thresholds, and
confidence calibration, where different trials carry different nuisance and a
raw log-likelihood is not comparable between them. It is kept for that, and
labelled as such. It is not an R1 mitigation, and that leaves R1 with fewer
defences than the plan assumed.

What actually mitigates R1 -- measured, and not what the design said
--------------------------------------------------------------------
The design proposed a discriminative subspace fitted on the labelled data. The
experiment in ``scripts/validate_decode.py`` shows that does not work, and why:

* **Full-rank LDA is mathematically identical to full-D likelihood.** Isotropic
  nearest-mean scoring depends on ``x`` only through its projection onto the span
  of the estimated class means -- at most ``K - 1`` dimensions, which is exactly the
  subspace LDA uses. Measured: identical BAcc@10 at every dimension.
* **Rank truncation barely helps.** Keeping fewer discriminant directions should
  discard noise, but those directions are estimated from the same scarce labels,
  so the choice itself is noise. At D=1600, rank 16 scored 0.273 vs full-D 0.278.

The decisive measurement: with the **true** signal subspace supplied, and class
means still estimated from only 16 labelled trials per word, BAcc@10 is 0.73-0.75
at *every* dimension. The labels are sufficient. **The bottleneck is entirely
estimating the subspace, and the labels cannot supply it.**

It must come from a different source -- abundant **unlabelled** data. PCA on
unlabelled samples recovers most of the gap, but the amount required grows with
dimension, and too little is worse than none (2,000 samples at D=1600 scored
below full-D).

This is the strongest argument in the codebase *for* the forecaster. Its job is
to learn that subspace from hours of unlabelled MEG, with a nonlinear inductive
bias that raw-sensor PCA lacks. Generative decoding in raw observation space is
not viable at these sample sizes; in a representation learned from unlabelled
data, it may be. Hence :class:`SubspaceDecoder`, which takes its basis from
outside and never fits it on the labels.

The LM term: pointwise mutual information, not log-probability
---------------------------------------------------------------
Unlike the neural PMI term, subtracting ``log p_LM(w)`` IS candidate-dependent,
so it genuinely changes the ranking. A raw LM prior re-imports corpus word
frequency and systematically demotes rare words; balanced accuracy averages
recall over classes and punishes that hard. PMI keeps the contextual information
and removes the frequency bias. This is the cheapest available gain on BAcc@10.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import balanced_accuracy_at_k

__all__ = [
    "gaussian_loglik",
    "DiscriminativeSubspace",
    "SubspaceDecoder",
    "unlabelled_basis",
    "lm_pmi",
    "neural_pmi",
    "combine",
    "LambdaCalibration",
    "calibrate_lambda",
]


def gaussian_loglik(x: np.ndarray, means: np.ndarray, var: float | np.ndarray) -> np.ndarray:
    """Isotropic/diagonal Gaussian log-likelihood of each trial under each class.

    Returns ``(n_trials, n_classes)``. Candidate-independent normalising terms are
    included, so the values are genuine log-densities rather than scores -- which
    keeps the neural PMI term meaningful for cross-trial calibration.
    """
    x = np.asarray(x, dtype=float)
    means = np.asarray(means, dtype=float)
    var = np.broadcast_to(np.asarray(var, dtype=float), (x.shape[1],))
    d2 = ((x[:, None, :] - means[None, :, :]) ** 2 / var).sum(-1)
    return -0.5 * d2 - 0.5 * float(np.log(2 * np.pi * var).sum())


@dataclass
class DiscriminativeSubspace:
    """Fisher-LDA subspace fitted on the labels. NOT an R1 mitigation.

    Kept because it is the obvious thing to try and the measurement of why it
    fails is informative. It was the design's proposed mitigation; the experiment
    in ``scripts/validate_decode.py`` rules it out:

    * at full rank (``K - 1``) it is mathematically identical to full-D
      nearest-mean scoring, which already depends on ``x`` only through the span
      of the estimated class means;
    * at reduced rank it barely helps, because the discriminant directions are
      estimated from the same scarce labels and inherit their noise.

    Use :class:`SubspaceDecoder` with a basis from unlabelled data instead.
    """

    rank: int | None = None
    shrinkage: float = 0.1
    projection: np.ndarray | None = None   # (rank, D)
    means: np.ndarray | None = None        # (K, rank)
    var: np.ndarray | None = None          # (rank,)
    classes: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DiscriminativeSubspace":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        n, d = x.shape
        mu = x.mean(axis=0)

        sw = np.zeros((d, d))
        sb = np.zeros((d, d))
        class_means = []
        for c in classes:
            xc = x[y == c]
            mc = xc.mean(axis=0)
            class_means.append(mc)
            dc = xc - mc
            sw += dc.T @ dc
            diff = (mc - mu)[:, None]
            sb += len(xc) * (diff @ diff.T)
        sw /= max(n - len(classes), 1)
        # Shrinkage toward a scaled identity. Without it, within-class covariance
        # is singular whenever D exceeds the number of training trials, which is
        # the normal case for MEG.
        sw = (1 - self.shrinkage) * sw + self.shrinkage * (np.trace(sw) / d) * np.eye(d)

        evals, evecs = np.linalg.eigh(sw)
        evals = np.maximum(evals, 1e-12)
        whiten = evecs @ np.diag(evals**-0.5) @ evecs.T
        m = whiten @ sb @ whiten
        bvals, bvecs = np.linalg.eigh(m)
        order = np.argsort(bvals)[::-1]

        rank = min(self.rank or (len(classes) - 1), len(classes) - 1, d)
        w = (whiten @ bvecs[:, order[:rank]]).T                 # (rank, D)

        z = x @ w.T
        self.projection = w
        self.classes = classes
        self.means = np.stack([z[y == c].mean(axis=0) for c in classes])
        resid = np.concatenate([z[y == c] - z[y == c].mean(axis=0) for c in classes])
        self.var = np.maximum(resid.var(axis=0), 1e-8)
        return self

    def loglik(self, x: np.ndarray) -> np.ndarray:
        if self.projection is None:
            raise RuntimeError("fit() before loglik()")
        z = np.asarray(x, dtype=float) @ self.projection.T
        return gaussian_loglik(z, self.means, self.var)


def unlabelled_basis(x_unlabelled: np.ndarray, rank: int) -> np.ndarray:
    """Top-``rank`` principal directions of unlabelled data. ``(D, rank)``.

    Uses no labels. A stand-in for what a pretrained encoder provides -- and a
    weak one, since it is linear and operates on raw sensors. Requires unlabelled
    sample counts that grow with dimension; too few picks noise directions and
    decodes *worse* than no subspace at all.
    """
    x = np.asarray(x_unlabelled, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (n_samples, D), got {x.shape}")
    if not 0 < rank <= x.shape[1]:
        raise ValueError(f"rank must lie in (0, {x.shape[1]}], got {rank}")
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    return evecs[:, np.argsort(evals)[::-1][:rank]]


@dataclass
class SubspaceDecoder:
    """Gaussian class likelihood inside a basis supplied from OUTSIDE the labels.

    The measured R1 mitigation. The basis must never be fitted on the labelled
    decoding set -- that is precisely what failed. Supply it from a pretrained
    encoder, or at minimum from :func:`unlabelled_basis` on separate data.

    Class means *are* fitted on the labels, which is fine: once the subspace is
    right, 16 trials per word was enough to reach 0.73-0.75 BAcc@10 at any
    observation dimension.
    """

    basis: np.ndarray                      # (D, rank), from outside the labels
    means: np.ndarray | None = None        # (K, rank)
    var: np.ndarray | None = None          # (rank,)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SubspaceDecoder":
        z = np.asarray(x, dtype=float) @ self.basis
        y = np.asarray(y)
        classes = np.unique(y)
        self.means = np.stack([z[y == c].mean(axis=0) for c in classes])
        resid = np.concatenate([z[y == c] - z[y == c].mean(axis=0) for c in classes])
        self.var = np.maximum(resid.var(axis=0), 1e-8)
        return self

    def loglik(self, x: np.ndarray) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("fit() before loglik()")
        return gaussian_loglik(np.asarray(x, dtype=float) @ self.basis, self.means, self.var)


def neural_pmi(
    loglik: np.ndarray, loglik_null: np.ndarray | None, w_cfg: float = 1.0
) -> np.ndarray:
    """``log p(X|w) - w_cfg * log p(X|empty)``.

    Does NOT change any within-trial ranking (see module docstring). Use it when
    scores are compared *across* trials: open-set detection, abstention,
    calibration.
    """
    if loglik_null is None:
        return np.asarray(loglik, dtype=float)
    return np.asarray(loglik, dtype=float) - w_cfg * np.asarray(loglik_null)[:, None]


def lm_pmi(lm_logp_context: np.ndarray, lm_logp_marginal: np.ndarray) -> np.ndarray:
    """``log p_LM(w | h) - log p_LM(w)``: context information without frequency bias.

    Parameters
    ----------
    lm_logp_context
        ``(n_trials, K)`` log-probability of each candidate given its left context.
    lm_logp_marginal
        ``(K,)`` unconditional log-probability of each candidate.
    """
    return np.asarray(lm_logp_context, float) - np.asarray(lm_logp_marginal, float)[None, :]


def combine(neural: np.ndarray, lm: np.ndarray | None, lam: float) -> np.ndarray:
    if lm is None or lam == 0.0:
        return np.asarray(neural, float)
    return np.asarray(neural, float) + lam * np.asarray(lm, float)


@dataclass(frozen=True)
class LambdaCalibration:
    lam: float
    bacc: float
    neural_only: float        # lambda = 0: the brain signal alone
    lm_only: float            # lambda -> inf: the language model alone
    signal_blind: float       # chosen lambda, blind neural input, same LM
    blind_ceiling: float      # what signal_blind was required to stay under
    curve: tuple[tuple[float, float, float], ...]  # (lambda, bacc, blind_bacc)
    ceilings: tuple[float, ...] = ()               # per-lambda blind ceiling

    @property
    def brain_gain(self) -> float:
        """Real brain vs. fake brain, with the LM held fixed. The honest headline.

        This isolates what the neural signal contributes. ``bacc - lm_only`` looks
        similar but is contaminated: it credits the brain for any gain that
        combining two score sources gives over one.
        """
        return self.bacc - self.signal_blind

    @property
    def gain_over_lm(self) -> float:
        return self.bacc - self.lm_only

    def summary(self) -> str:
        return (
            f"lambda={self.lam:.3g}  BAcc@10={self.bacc:.3f}  "
            f"neural-only={self.neural_only:.3f}  LM-only={self.lm_only:.3f}  "
            f"blind+LM={self.signal_blind:.3f}  brain adds {self.brain_gain:+.3f}"
        )


def calibrate_lambda(
    neural: np.ndarray,
    neural_blind: np.ndarray,
    lm: np.ndarray,
    y: np.ndarray,
    *,
    lambdas: np.ndarray | None = None,
    k: int = 10,
    chance_upper: float,
    tol: float = 0.02,
    n_null: int = 5,
    seed: int = 0,
) -> LambdaCalibration:
    """Pick lambda to maximise BAcc@k, subject to a signal-blind constraint.

    The constraint, stated carefully -- the obvious version is wrong
    ----------------------------------------------------------------
    The tempting rule is "signal-blind input must score at chance". But the LM is
    *supposed* to be informative about context. Feed blind neural scores plus a
    good LM and BAcc rises above chance from the LM alone, legitimately. Holding
    that to chance would forbid any meaningful lambda and throw the LM away.

    The right question is whether the **blind neural pathway adds anything beyond
    what signal-free scores would**. If it does, the neural channel is extracting
    "signal" from noise -- structural leakage, the 66.3%-Rank@1 failure.

    The ceiling must depend on lambda
    ---------------------------------
    A first version used one constant ceiling, ``max(chance, LM-only) + tol``, and
    it was wrong. At lambda = 0 there is no LM in the score at all, so blind input
    should sit at chance; comparing it against a strong LM-only number let a
    blind channel scoring 0.70 -- flagrant leakage -- pass under a 0.87 ceiling.

    The correct reference at each lambda is what **genuinely signal-free** neural
    scores achieve with that same LM weight. Those are obtained by permuting the
    blind scores across trials: the score distribution is preserved exactly, and
    only the relationship to the label is destroyed. So the ceiling tracks chance
    at lambda = 0 and rises toward LM-only as the LM comes to dominate.

    Report the three reference points with every result -- neural-only, LM-only,
    blind+LM -- and headline :attr:`LambdaCalibration.brain_gain`, which compares
    real and blind neural input with the LM held fixed. A decoder whose chosen
    BAcc sits near LM-only is a language model wearing a lab coat, however good
    the headline number looks.
    """
    if lambdas is None:
        lambdas = np.concatenate([[0.0], np.logspace(-2, 2, 25)])

    y = np.asarray(y)
    neural_blind = np.asarray(neural_blind, float)
    rng = np.random.default_rng(seed)
    null_blind = [neural_blind[rng.permutation(len(y))] for _ in range(n_null)]

    lm_only = balanced_accuracy_at_k(np.asarray(lm, float), y, k)

    curve = []
    ceilings = []
    best = None
    for lam in lambdas:
        acc = balanced_accuracy_at_k(combine(neural, lm, lam), y, k)
        blind = balanced_accuracy_at_k(combine(neural_blind, lm, lam), y, k)
        ref = max(
            balanced_accuracy_at_k(combine(nb, lm, lam), y, k) for nb in null_blind
        )
        ceiling = max(ref, chance_upper if lam == 0.0 else ref) + tol
        curve.append((float(lam), acc, blind))
        ceilings.append(ceiling)
        if blind <= ceiling and (best is None or acc > best[1]):
            best = (float(lam), acc, blind, ceiling)

    if best is None:  # nothing satisfies the constraint; report neural-only
        best = (0.0, curve[0][1], curve[0][2], ceilings[0])

    return LambdaCalibration(
        lam=best[0],
        bacc=best[1],
        neural_only=curve[0][1],
        lm_only=lm_only,
        signal_blind=best[2],
        blind_ceiling=best[3],
        curve=tuple(curve),
        ceilings=tuple(ceilings),
    )
