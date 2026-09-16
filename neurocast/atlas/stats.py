"""Inference for forecastability, where the usual tools break.

Neural time series are massively autocorrelated, and so are the per-sample losses
of any model forecasting them. Treat those losses as independent and every
interval comes out far too narrow: a bootstrap that resamples individual samples
reports certainty the data cannot support, and a naive t-test on a loss
difference finds "significant" gaps everywhere.

Two tools from forecasting econometrics, both essentially absent from the
neuroscience literature, fix this:

* **Moving-block bootstrap.** Resample contiguous blocks, so within-block
  dependence survives into each replicate.
* **Diebold-Mariano test with Newey-West HAC variance.** The standard test for
  "is forecaster A better than B?" on dependent losses.

Plus the Atlas's free bug detector, the **monotonicity audit**: information sets
that nest must give forecastability that does not decrease. A violation beyond
the confidence interval means the estimator or the training is broken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BootstrapCI",
    "block_bootstrap_mean",
    "iid_bootstrap_mean",
    "newey_west_lag",
    "andrews_lag",
    "newey_west_variance",
    "DMResult",
    "diebold_mariano",
    "MonotonicityViolation",
    "audit_monotonicity",
]


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    level: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def __str__(self) -> str:
        return f"{self.mean:+.4f} [{self.lo:+.4f}, {self.hi:+.4f}]"


def block_bootstrap_mean(
    x: np.ndarray,
    *,
    block_len: int,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Moving-block bootstrap CI for the mean of a dependent series.

    ``block_len`` should comfortably exceed the loss autocorrelation length. The
    design uses 10 s of data; in samples that is ``10 * fs``.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    block_len = int(max(1, min(block_len, n)))
    n_blocks = int(math.ceil(n / block_len))
    starts_max = n - block_len + 1
    rng = np.random.default_rng(seed)

    csum = np.concatenate([[0.0], np.cumsum(x)])
    block_sums = csum[block_len:] - csum[:-block_len]      # (starts_max,)
    starts = rng.integers(0, starts_max, size=(n_boot, n_blocks))
    means = block_sums[starts].sum(1) / (n_blocks * block_len)

    a = (1.0 - level) / 2.0
    return BootstrapCI(
        mean=float(x.mean()),
        lo=float(np.quantile(means, a)),
        hi=float(np.quantile(means, 1.0 - a)),
        level=level,
    )


def iid_bootstrap_mean(
    x: np.ndarray, *, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> BootstrapCI:
    """Ordinary bootstrap. Kept only to show how overconfident it is here."""
    return block_bootstrap_mean(x, block_len=1, n_boot=n_boot, level=level, seed=seed)


def newey_west_lag(n: int) -> int:
    """Fixed-rule bandwidth ``floor(4 (n/100)^(2/9))``. NOT the default -- see below.

    Kept for reference. It depends only on sample size, so it cannot adapt to
    how persistent the losses are, and neural forecasting losses are very
    persistent. Measured in ``scripts/validate_atlas.py`` on AR(1) phi=0.8 loss
    differentials (n=2000): this rule picks lag 7 and the DM test rejects a true
    null **12.5%** of the time against a nominal 5%. Use :func:`andrews_lag`.
    """
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def andrews_lag(d: np.ndarray) -> int:
    """Data-dependent Bartlett bandwidth, Andrews (1991) AR(1) plug-in.

    ``1.1447 * (alpha * n)^(1/3)`` with ``alpha = 4 rho^2 / ((1-rho)^2 (1+rho)^2)``,
    where ``rho`` is the lag-1 autocorrelation of the loss differential. The
    bandwidth grows with persistence, which is exactly what a long-run variance
    estimate needs. On the same test as above it picks lag ~39 and brings the DM
    rejection rate to 6.5% against a nominal 5%.
    """
    d = np.asarray(d, dtype=float)
    dc = d - d.mean()
    denom = float(dc[:-1] @ dc[:-1])
    rho = float(dc[1:] @ dc[:-1]) / denom if denom > 0 else 0.0
    rho = max(min(rho, 0.97), -0.97)   # guard the pole at |rho| = 1
    alpha = 4.0 * rho**2 / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    lag = int(math.ceil(1.1447 * (alpha * len(d)) ** (1.0 / 3.0)))
    return max(1, min(lag, len(d) - 1))


def newey_west_variance(d: np.ndarray, lag: int | None = None) -> float:
    """Long-run variance of ``d`` with Bartlett-weighted autocovariances."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    lag = andrews_lag(d) if lag is None else int(lag)
    dc = d - d.mean()
    lrv = float(dc @ dc) / n
    for j in range(1, min(lag, n - 1) + 1):
        gamma = float(dc[j:] @ dc[:-j]) / n
        lrv += 2.0 * (1.0 - j / (lag + 1.0)) * gamma
    return max(lrv, 1e-300)


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    mean_diff: float
    lag: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def __str__(self) -> str:
        return (
            f"DM={self.statistic:+.3f}  p={self.p_value:.4f}  "
            f"mean loss diff {self.mean_diff:+.4f}  (HAC lag {self.lag})"
        )


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, *, lag: int | None = None
) -> DMResult:
    """Is forecaster A's loss different from B's, accounting for dependence?

    ``mean_diff = mean(loss_a - loss_b)``: negative means A is better. Two-sided
    p-value under the asymptotic normal null, computed with ``math.erfc`` so no
    scipy import is needed.

    The long-run variance uses the :func:`andrews_lag` bandwidth by default. The
    common fixed rule roughly doubles the false-positive rate on persistent
    losses -- which would make the Atlas report differences that are not there.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"loss arrays must align sample-for-sample: {a.shape} vs {b.shape}"
        )
    d = a - b
    n = len(d)
    lag = andrews_lag(d) if lag is None else int(lag)
    stat = float(d.mean() / math.sqrt(newey_west_variance(d, lag) / n))
    p = float(math.erfc(abs(stat) / math.sqrt(2.0)))
    return DMResult(statistic=stat, p_value=p, mean_diff=float(d.mean()), lag=lag)


@dataclass(frozen=True)
class MonotonicityViolation:
    smaller: str
    larger: str
    info_smaller: BootstrapCI
    info_larger: BootstrapCI

    def __str__(self) -> str:
        return (
            f"'{self.smaller}' {self.info_smaller} > '{self.larger}' "
            f"{self.info_larger} although '{self.smaller}' is nested inside it"
        )


def audit_monotonicity(
    info: dict[str, BootstrapCI], nesting: list[tuple[str, str]]
) -> list[MonotonicityViolation]:
    """Flag nested information sets whose forecastability *decreases*.

    Adding information can never reduce the achievable reduction in log loss, so
    ``I(smaller) <= I(larger)`` must hold. Only CI-separated reversals are
    flagged -- sampling noise can reorder close values legitimately.

    Parameters
    ----------
    nesting
        ``(smaller, larger)`` pairs where ``smaller``'s information is a subset
        of ``larger``'s, e.g. ``("past brain", "past brain + stimulus")``.
    """
    out = []
    for small, large in nesting:
        a, b = info[small], info[large]
        if a.lo > b.hi:
            out.append(MonotonicityViolation(small, large, a, b))
    return out
