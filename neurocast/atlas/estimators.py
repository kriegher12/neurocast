"""Forecastability estimators for the Neural Predictability Atlas.

The estimand
------------
Under logarithmic loss, the mutual information between a future observation and
an information set equals the maximum achievable reduction in expected loss
(arXiv 2603.27074). So forecastability at horizon ``h`` given information set
``S`` is estimated as a difference of held-out cross-entropies:

    I_S(h)  ~=  L_marginal(h)  -  L_S(h)          [nats per sample]

Two honesty points that belong in any paper using these numbers:

* ``I_S`` is a **lower bound** on the true mutual information, because the model
  class is imperfect. It bounds from below what any decoder could extract.
* It is **held-out**, so estimation error can push it slightly negative. That is
  a feature: an in-sample estimate would be biased upward and could never
  report "no predictability".

The reference ladder
--------------------
The single most important design choice is what counts as "no structure".

* ``marginal``     knows amplitude only.
* ``spectral_null`` knows the full linear power spectrum -- 1/f slope, alpha
  peak, everything. The optimal predictor of a stationary Gaussian process is
  linear and determined by its spectrum (Wiener-Kolmogorov), so a long direct
  linear predictor approximates the PSD-matched Gaussian.
* ``heteroscedastic`` adds a variance that depends on recent signal energy:
  the simplest structure that is invisible to any spectrum.
* ``ksg_mutual_information`` is model-free, used as a cross-check.

Report everything relative to ``spectral_null``, never only relative to
``marginal``. A model that beats white noise but not the spectral null has
learned 1/f -- precisely the subject-identity carrier the Identity Trap
diagnoses. Getting this reference wrong turns the Atlas into a paper about 1/f.

All estimators are scored on **identical target samples** (fixed by
``max_lags`` and ``horizon``), so per-sample losses align for paired tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ForecastScore",
    "aligned_targets",
    "marginal",
    "linear",
    "spectral_null",
    "heteroscedastic",
    "defensive_mixture",
    "linear_with_covariate",
    "digamma",
    "ksg_mutual_information",
    "gaussian_mi",
]

_LOG2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class ForecastScore:
    """Held-out predictive loss of one model at one horizon."""

    name: str
    horizon: int
    per_sample: np.ndarray   # held-out NLL, nats, aligned across models

    @property
    def nll(self) -> float:
        return float(self.per_sample.mean())

    @property
    def n(self) -> int:
        return int(self.per_sample.shape[0])


def _as_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[None, :] if x.ndim == 1 else x


def aligned_targets(n_times: int, max_lags: int, horizon: int) -> np.ndarray:
    """Time indices ``t`` whose target ``x[t + horizon]`` every model scores.

    Fixing these by the *largest* lag count used anywhere is what makes losses
    from different models comparable sample-for-sample.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    t = np.arange(max_lags - 1, n_times - horizon)
    if t.size == 0:
        raise ValueError(
            f"series of {n_times} samples too short for max_lags={max_lags}, "
            f"horizon={horizon}"
        )
    return t


def _design(row: np.ndarray, t: np.ndarray, lags: int) -> np.ndarray:
    """``(len(t), lags)`` matrix of ``[x[t], x[t-1], ..., x[t-lags+1]]``."""
    idx = t[:, None] - np.arange(lags)[None, :]
    return row[idx]


def _gauss_nll(resid: np.ndarray, var: np.ndarray | float) -> np.ndarray:
    var = np.maximum(var, 1e-12)
    return 0.5 * (_LOG2PI + np.log(var) + resid**2 / var)


def marginal(
    train: np.ndarray, test: np.ndarray, horizon: int, max_lags: int
) -> ForecastScore:
    """Gaussian with the training mean and variance. Knows nothing temporal."""
    tr, te = _as_channels(train), _as_channels(test)
    t = aligned_targets(te.shape[1], max_lags, horizon)
    out = []
    for c in range(te.shape[0]):
        mu, var = tr[c].mean(), tr[c].var()
        out.append(_gauss_nll(te[c, t + horizon] - mu, var))
    return ForecastScore("marginal", horizon, np.concatenate(out))


def linear(
    train: np.ndarray,
    test: np.ndarray,
    horizon: int,
    lags: int,
    max_lags: int,
    *,
    ridge: float = 1e-3,
    name: str | None = None,
) -> ForecastScore:
    """Direct ``h``-step linear predictor from ``lags`` past samples, per channel.

    Fits a separate regression per horizon ("direct" strategy) rather than
    iterating a one-step model, so errors do not compound and the training
    residual variance is exactly the ``h``-step predictive variance.
    """
    if lags > max_lags:
        raise ValueError(f"lags={lags} exceeds max_lags={max_lags}")
    tr, te = _as_channels(train), _as_channels(test)
    t_tr = aligned_targets(tr.shape[1], max_lags, horizon)
    t_te = aligned_targets(te.shape[1], max_lags, horizon)
    out = []
    for c in range(te.shape[0]):
        mu = tr[c].mean()
        a_tr = np.column_stack([np.ones(len(t_tr)), _design(tr[c] - mu, t_tr, lags)])
        y_tr = tr[c, t_tr + horizon] - mu
        reg = ridge * np.eye(a_tr.shape[1])
        reg[0, 0] = 0.0
        beta = np.linalg.solve(a_tr.T @ a_tr + reg, a_tr.T @ y_tr)
        var = float(np.mean((y_tr - a_tr @ beta) ** 2))

        a_te = np.column_stack([np.ones(len(t_te)), _design(te[c] - mu, t_te, lags)])
        resid = (te[c, t_te + horizon] - mu) - a_te @ beta
        out.append(_gauss_nll(resid, var))
    return ForecastScore(name or f"linear_p{lags}", horizon, np.concatenate(out))


def spectral_null(
    train: np.ndarray, test: np.ndarray, horizon: int, max_lags: int, lags: int = 64
) -> ForecastScore:
    """The PSD-matched Gaussian reference -- the Atlas's primary null.

    Approximated by a long direct linear predictor: for a stationary Gaussian
    process the optimal predictor is linear and fixed by the power spectrum, so
    this captures all *linear/spectral* predictability, including 1/f and alpha.
    Forecastability above this line is structure beyond spectral stationarity.
    """
    return linear(train, test, horizon, min(lags, max_lags), max_lags,
                  name="spectral_null")


def defensive_mixture(
    model_nll: np.ndarray, marginal_nll: np.ndarray, eps: float = 1e-3
) -> np.ndarray:
    """Hedge a predictive density with a small share of the marginal.

    ``p = (1 - eps) * p_model + eps * p_marginal``, returned as per-sample NLL.

    Why this exists -- found, not anticipated
    -----------------------------------------
    A model whose predictive variance depends on its input can be catastrophically
    overconfident on a single sample. In validation, a heteroscedastic model beat
    the spectral null on the median sample by +0.062 nats -- yet **one** test
    sample scored 251 nats, because a volatility burst arrived right after a calm
    stretch and the model had predicted a tiny variance. A Gaussian's loss grows
    with the squared residual, without bound; the five worst samples alone flipped
    the average gain from positive to -0.0024 nats.

    In real MEG that event is an eye blink after a quiet stretch.

    The mixture gives two guarantees, both verified in ``validate_atlas.py``:

    * **Worst case bounded**: ``NLL <= NLL_marginal - log(eps)``. At eps=1e-3 that
      is 6.9 nats over the marginal; the 251-nat sample became 20.0.
    * **Negligible cost**: ``NLL <= NLL_model - log(1 - eps)``, at most 0.001 nats,
      and measured at ~1e-5 on realisations with no catastrophe.

    The result is still a proper density, so the Atlas estimate remains a valid
    lower bound on mutual information.
    """
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must lie in (0, 1), got {eps}")
    return -np.logaddexp(
        math.log1p(-eps) - np.asarray(model_nll, float),
        math.log(eps) - np.asarray(marginal_nll, float),
    )


def heteroscedastic(
    train: np.ndarray,
    test: np.ndarray,
    horizon: int,
    lags: int,
    max_lags: int,
    *,
    ridge: float = 1e-3,
    defend: float | None = 1e-3,
) -> ForecastScore:
    """Linear mean plus a variance that depends on recent squared signal.

    The simplest predictor that can exploit structure invisible to any power
    spectrum: bursts, where a large recent amplitude predicts a large upcoming
    one. A process with zero autocorrelation can still be forecastable this way,
    and the spectral null will report nothing.

    Input-dependent variance makes this model vulnerable to single catastrophic
    samples, so it is hedged with :func:`defensive_mixture` by default. Pass
    ``defend=None`` only to reproduce the failure. Homoscedastic models
    (:func:`linear`, :func:`spectral_null`) do not need it: their predictive
    variance is a constant, so their tail exposure is the marginal's own.
    """
    tr, te = _as_channels(train), _as_channels(test)
    t_tr = aligned_targets(tr.shape[1], max_lags, horizon)
    t_te = aligned_targets(te.shape[1], max_lags, horizon)
    out = []
    for c in range(te.shape[0]):
        mu = tr[c].mean()
        a_tr = np.column_stack([np.ones(len(t_tr)), _design(tr[c] - mu, t_tr, lags)])
        y_tr = tr[c, t_tr + horizon] - mu
        reg = ridge * np.eye(a_tr.shape[1])
        reg[0, 0] = 0.0
        beta = np.linalg.solve(a_tr.T @ a_tr + reg, a_tr.T @ y_tr)
        r2 = (y_tr - a_tr @ beta) ** 2

        v_tr = np.column_stack([np.ones(len(t_tr)), _design(tr[c] - mu, t_tr, lags) ** 2])
        regv = ridge * np.eye(v_tr.shape[1])
        regv[0, 0] = 0.0
        gamma = np.linalg.solve(v_tr.T @ v_tr + regv, v_tr.T @ r2)
        floor = 0.05 * float(r2.mean())

        a_te = np.column_stack([np.ones(len(t_te)), _design(te[c] - mu, t_te, lags)])
        v_te = np.column_stack([np.ones(len(t_te)), _design(te[c] - mu, t_te, lags) ** 2])
        target = te[c, t_te + horizon]
        nll = _gauss_nll(target - mu - a_te @ beta, np.maximum(v_te @ gamma, floor))
        if defend is not None:
            nll = defensive_mixture(nll, _gauss_nll(target - mu, tr[c].var()), defend)
        out.append(nll)
    name = f"heteroscedastic_p{lags}" + ("" if defend is None else "_defended")
    return ForecastScore(name, horizon, np.concatenate(out))


def linear_with_covariate(
    train: np.ndarray,
    train_cov: np.ndarray,
    test: np.ndarray,
    test_cov: np.ndarray,
    horizon: int,
    lags: int,
    cov_lags: int,
    max_lags: int,
    *,
    cov_lead: int | None = None,
    ridge: float = 1e-3,
    name: str = "linear+covariate",
) -> ForecastScore:
    """Direct linear predictor from past brain **and** an exogenous covariate.

    This is the Atlas's central decomposition. Comparing it against
    :func:`linear` on identical targets gives

        I(future ; past brain, stimulus) - I(future ; past brain)

    -- how much of the brain's near future is driven by the world rather than by
    its own dynamics.

    ``cov_lead`` sets how far past ``t`` the covariate may be read. The default,
    ``horizon``, treats the stimulus as **exogenous and known in advance**: the
    audiobook's next second is fixed before the brain hears it, so conditioning
    on it is legitimate for an encoding question. It is NOT legitimate for a
    decoding question, where the stimulus is the unknown. Label results with the
    lead used; the two answer different questions.
    """
    lead = horizon if cov_lead is None else int(cov_lead)
    tr, te = _as_channels(train), _as_channels(test)
    utr, ute = _as_channels(train_cov), _as_channels(test_cov)
    if utr.shape[1] != tr.shape[1] or ute.shape[1] != te.shape[1]:
        raise ValueError("covariate must have the same length as the signal")

    t_tr = aligned_targets(tr.shape[1], max_lags, horizon)
    t_te = aligned_targets(te.shape[1], max_lags, horizon)
    # Covariate window: samples t+lead down to t+lead-cov_lags+1.
    t_tr_u = np.minimum(t_tr + lead, tr.shape[1] - 1)
    t_te_u = np.minimum(t_te + lead, te.shape[1] - 1)

    def cov_block(u: np.ndarray, t: np.ndarray) -> np.ndarray:
        idx = np.clip(t[:, None] - np.arange(cov_lags)[None, :], 0, u.shape[1] - 1)
        return np.concatenate([u[k][idx] for k in range(u.shape[0])], axis=1)

    out = []
    for c in range(te.shape[0]):
        mu = tr[c].mean()
        a_tr = np.column_stack(
            [np.ones(len(t_tr)), _design(tr[c] - mu, t_tr, lags), cov_block(utr, t_tr_u)]
        )
        y_tr = tr[c, t_tr + horizon] - mu
        reg = ridge * np.eye(a_tr.shape[1])
        reg[0, 0] = 0.0
        beta = np.linalg.solve(a_tr.T @ a_tr + reg, a_tr.T @ y_tr)
        var = float(np.mean((y_tr - a_tr @ beta) ** 2))

        a_te = np.column_stack(
            [np.ones(len(t_te)), _design(te[c] - mu, t_te, lags), cov_block(ute, t_te_u)]
        )
        resid = (te[c, t_te + horizon] - mu) - a_te @ beta
        out.append(_gauss_nll(resid, var))
    return ForecastScore(name, horizon, np.concatenate(out))


def digamma(x: np.ndarray | float) -> np.ndarray:
    """Digamma for positive reals. NumPy only.

    Recurrence up to x >= 6, then the asymptotic series; accurate to ~1e-10 on
    the integer arguments KSG uses.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("digamma is only implemented for x > 0")
    acc = np.zeros_like(x)
    y = x.copy()
    while True:
        small = y < 6.0
        if not small.any():
            break
        acc = acc - np.where(small, 1.0 / y, 0.0)
        y = np.where(small, y + 1.0, y)
    inv = 1.0 / y
    inv2 = inv * inv
    series = (
        np.log(y)
        - 0.5 * inv
        - inv2 * (1.0 / 12 - inv2 * (1.0 / 120 - inv2 * (1.0 / 252 - inv2 / 240)))
    )
    return acc + series


def gaussian_mi(rho: float) -> float:
    """Exact mutual information of a bivariate Gaussian with correlation ``rho``."""
    return -0.5 * math.log(1.0 - rho * rho)


def ksg_mutual_information(
    x: np.ndarray, y: np.ndarray, k: int = 4, *, seed: int = 0, chunk: int = 256
) -> float:
    """Kraskov-Stoegbauer-Grassberger estimator (algorithm 1), in nats.

    Model-free, so it catches structure no parametric predictor was built to
    see. Brute-force neighbour search, which is fine to a few thousand samples
    and keeps this dependency-free.
    """
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    y = np.asarray(y, dtype=float).reshape(len(y), -1)
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of samples")
    n = len(x)
    if n <= k + 1:
        raise ValueError(f"need more than k+1={k + 1} samples, got {n}")

    rng = np.random.default_rng(seed)
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    y = (y - y.mean(0)) / (y.std(0) + 1e-12)
    x = x + 1e-10 * rng.standard_normal(x.shape)   # break ties deterministically
    y = y + 1e-10 * rng.standard_normal(y.shape)

    nx = np.empty(n)
    ny = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dx = np.abs(x[s:e, None, :] - x[None, :, :]).max(-1)
        dy = np.abs(y[s:e, None, :] - y[None, :, :]).max(-1)
        dz = np.maximum(dx, dy)
        rows = np.arange(e - s)
        dz[rows, rows + s] = np.inf
        dx[rows, rows + s] = np.inf
        dy[rows, rows + s] = np.inf
        eps = np.partition(dz, k - 1, axis=1)[:, k - 1]
        nx[s:e] = (dx < eps[:, None]).sum(1)
        ny[s:e] = (dy < eps[:, None]).sum(1)

    return float(
        digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    )
