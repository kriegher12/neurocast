"""INV-1's reference rows: every likelihood is reported against its nulls, inline.

INV-1 (:mod:`neurocast.data.canonical`) requires that a negative log-likelihood is
never reported alone. It goes out in nats per second per sensor, next to

  (a) its delta against a **PSD-matched stationary Gaussian**, and
  (b) its delta against a **per-session linear AR(16)**.

This module computes those rows, plus the white-noise marginal and a Whittle
cross-check, and packages them with the model's number so the deltas cannot be
left out.

Why these references
--------------------
(a) is the primary null. A stationary Gaussian with the session's own power
spectrum already knows the 1/f slope, the alpha peak and every band power -- the
carriers the Identity Trap names. A model that beats white noise but not this
null has learned 1/f, and nothing in its NLL would say so. (b) is the cheap
parametric baseline anyone can reproduce; a model that does not beat AR(16) is
not yet worth a GPU.

Reference (a) is computed in the time domain, not by Whittle -- measured
---------------------------------------------------------------------
The design named the *Whittle* likelihood for (a). Whittle is the classical
frequency-domain approximation to a PSD-matched Gaussian's likelihood, and
:func:`whittle_nll` implements it exactly. But it is too weak to be the null.
``scripts/validate_eval.py`` measures it on processes with known spectra: at every
window length tried (250-2000 samples), a plain AR(16) beats Whittle by 0.01-0.03
nats/sample on coloured spectra -- the price of treating correlated DFT bins as
independent and of estimating a spectrum per bin. Used as the null, it would
credit ordinary linear-Gaussian structure as "beyond the spectrum".

So (a) is the **Wiener-Kolmogorov predictor**: the optimal one-step predictor of
a stationary Gaussian is linear and fixed by its spectrum, so a long direct
linear predictor (64 lags) fitted on the training portion *is* the PSD-matched
Gaussian, evaluated by its exact conditional likelihood. It reaches the analytic
entropy rate, and it is the same null the Atlas uses
(:func:`neurocast.atlas.estimators.spectral_null`), so a "delta vs the spectral
null" means the same thing in both places. Whittle is kept as a column: a large
disagreement between the two flags non-stationarity or too little training data.

Significance is not this module's job. These are summary means; to claim a model
beats a null, run :func:`neurocast.atlas.stats.diebold_mariano` on the per-sample
losses.

All numbers are held-out, and in the space the caller passes. INV-1 requires that
to be the frozen canonical space -- and the model's NLL must be in the **same**
space. A density over PCA targets is not comparable with these rows, which are
over raw sensors; the report's ``space`` field exists to make that mismatch
visible rather than silent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..atlas.estimators import linear
from ..data.canonical import CANONICAL_FS

__all__ = [
    "periodogram_spectrum",
    "whittle_nll",
    "psd_null_nll",
    "ar_nll",
    "marginal_nll",
    "LikelihoodReport",
    "reference_report",
    "format_table",
]

_LOG2PI = math.log(2.0 * math.pi)


def _as_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"expected (n_channels, n_times) or (n_times,), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("input contains non-finite values")
    return x


def _pair(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tr, te = _as_channels(train), _as_channels(test)
    if tr.shape[0] != te.shape[0]:
        raise ValueError(f"train has {tr.shape[0]} channels, test has {te.shape[0]}")
    return tr, te


def _windows(x: np.ndarray, length: int, step: int) -> np.ndarray:
    """``(C, n_windows, length)`` windows of ``x``."""
    n_t = x.shape[1]
    starts = np.arange(0, n_t - length + 1, step)
    if starts.size == 0:
        raise ValueError(f"series of {n_t} samples is shorter than one {length}-sample window")
    idx = starts[:, None] + np.arange(length)[None, :]
    return x[:, idx]


def periodogram_spectrum(
    train: np.ndarray, length: int, *, overlap: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel spectrum ``S_k = E|Z_k|^2`` for ``length``-sample windows.

    ``Z_k`` is the unitary DFT. Estimated by averaging rectangular-window
    periodograms of exactly the length the test windows will have, so leakage and
    Fejer smoothing are the same on both sides.

    Returns ``(S, mean)``: ``S`` is ``(C, length // 2 + 1)``, ``mean`` is the
    per-channel training mean that was removed (and that test data must be
    centred on -- centring a test window on its own mean would use the test data
    to explain itself).
    """
    x = _as_channels(train)
    mean = x.mean(axis=1, keepdims=True)
    step = max(1, int(round(length * (1.0 - overlap))))
    w = _windows(x - mean, length, step)
    z = np.fft.rfft(w, axis=-1) / math.sqrt(length)
    s = (z.real**2 + z.imag**2).mean(axis=1)
    # A bin with zero power would make the density singular. Floor it far below
    # anything physical; it only matters for degenerate (e.g. constant) input.
    floor = 1e-12 * max(float(s.max()), 1e-300)
    return np.maximum(s, floor), mean[:, 0]


def whittle_nll(
    train: np.ndarray, test: np.ndarray, *, length: int = 1000, overlap: float = 0.5
) -> np.ndarray:
    """Held-out Whittle NLL, nats per sample, one value per channel.

    For a zero-mean stationary Gaussian, the unitary DFT coefficients of an
    ``L``-sample window are approximately independent with ``E|Z_k|^2 = S_k``.
    The unitary DFT has ``|det| = 1``, but the orthonormal real coordinates of an
    interior bin are ``sqrt(2) Re Z_k`` and ``sqrt(2) Im Z_k`` -- which is where
    the ``2 pi`` (not ``pi``) below comes from::

        -log p(x) = sum_{0<k<L/2} [ log(2 pi S_k) + |Z_k|^2 / S_k ]
                  + 1/2 [ log(2 pi S_0) + Z_0^2 / S_0 ]
                  + 1/2 [ log(2 pi S_{L/2}) + Z_{L/2}^2 / S_{L/2} ]   (L even)

    A cross-check on :func:`psd_null_nll`, not the primary null; see the module
    docstring for the measurement that decided that.
    """
    tr, te = _pair(train, test)
    length = int(min(length, tr.shape[1], te.shape[1]))
    if length < 16:
        raise ValueError(f"need windows of at least 16 samples, got {length}")
    s, mean = periodogram_spectrum(tr, length, overlap=overlap)

    w = _windows(te - mean[:, None], length, length)        # non-overlapping
    z = np.fft.rfft(w, axis=-1) / math.sqrt(length)
    p = z.real**2 + z.imag**2                                 # (C, n_win, L//2+1)
    sk = s[:, None, :]

    n_freq = s.shape[1]
    even = length % 2 == 0
    mid = slice(1, n_freq - 1 if even else n_freq)
    nll = (_LOG2PI + np.log(sk[..., mid]) + p[..., mid] / sk[..., mid]).sum(-1)
    nll += 0.5 * (_LOG2PI + np.log(sk[..., 0]) + p[..., 0] / sk[..., 0])
    if even:
        nll += 0.5 * (_LOG2PI + np.log(sk[..., -1]) + p[..., -1] / sk[..., -1])
    return nll.mean(axis=1) / length


def _per_channel(train: np.ndarray, test: np.ndarray, lags: int, name: str) -> np.ndarray:
    tr, te = _pair(train, test)
    score = linear(tr, te, horizon=1, lags=lags, max_lags=lags, name=name)
    return score.per_sample.reshape(te.shape[0], -1).mean(axis=1)


def psd_null_nll(train: np.ndarray, test: np.ndarray, *, lags: int = 64) -> np.ndarray:
    """Held-out PSD-matched Gaussian NLL, nats per sample, one value per channel.

    INV-1's reference (a): the Wiener-Kolmogorov one-step predictor, approximated
    by a direct linear predictor with ``lags`` past samples. Identical to the
    Atlas's ``spectral_null`` at horizon 1.
    """
    return _per_channel(train, test, lags, "psd_null")


def ar_nll(train: np.ndarray, test: np.ndarray, *, order: int = 16) -> np.ndarray:
    """Held-out one-step AR(``order``) NLL, nats per sample, one value per channel.

    INV-1's reference (b). Fitted per channel on the session's own training
    portion -- that is what "per-session" means.
    """
    return _per_channel(train, test, order, f"ar{order}")


def marginal_nll(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Held-out white-noise NLL: a Gaussian with the training mean and variance."""
    tr, te = _pair(train, test)
    mu = tr.mean(axis=1, keepdims=True)
    var = np.maximum(tr.var(axis=1, keepdims=True), 1e-300)
    return (0.5 * (_LOG2PI + np.log(var) + (te - mu) ** 2 / var)).mean(axis=1)


@dataclass(frozen=True)
class LikelihoodReport:
    """One model's NLL with its INV-1 reference rows. All nats / s / sensor.

    ``delta_*`` is model minus reference: **negative means the model is better**.
    A model that is better than ``white`` but not than ``psd`` has learned the
    spectrum and nothing else.
    """

    name: str
    space: str
    fs: float
    model: float | None
    white: float
    psd: float
    whittle: float
    ar: float
    ar_order: int = 16

    @property
    def delta_psd(self) -> float:
        return float("nan") if self.model is None else self.model - self.psd

    @property
    def delta_ar(self) -> float:
        return float("nan") if self.model is None else self.model - self.ar

    def row(self) -> str:
        m = "      n/a" if self.model is None else f"{self.model:9.2f}"
        return (
            f"{self.name:<20} {m} {self.white:9.2f} {self.psd:9.2f} {self.whittle:9.2f} "
            f"{self.ar:9.2f} {self.delta_psd:+9.2f} {self.delta_ar:+9.2f}  [{self.space}]"
        )


def reference_report(
    name: str,
    model_nll_per_sample: float | None,
    train: np.ndarray,
    test: np.ndarray,
    *,
    fs: float = CANONICAL_FS,
    space: str = "canonical sensors",
    psd_lags: int = 64,
    whittle_length: int = 1000,
    ar_order: int = 16,
) -> LikelihoodReport:
    """Build a report for a model evaluated on ``test``.

    Parameters
    ----------
    model_nll_per_sample
        The model's held-out NLL in nats per sample per sensor, on ``test``, in
        ``space``, with any normalisation Jacobian already added back
        (:meth:`neurocast.data.canonical.AffineTransform.to_canonical_logprob`).
        ``None`` reports the references alone.
    train, test
        ``(n_channels, n_times)`` from the **same session**, temporally disjoint
        and separated by a guard band -- the training portion fits the
        references, the test portion scores them.
    """
    rate = float(fs)
    return LikelihoodReport(
        name=name,
        space=space,
        fs=rate,
        model=None if model_nll_per_sample is None else float(model_nll_per_sample) * rate,
        white=float(marginal_nll(train, test).mean()) * rate,
        psd=float(psd_null_nll(train, test, lags=psd_lags).mean()) * rate,
        whittle=float(whittle_nll(train, test, length=whittle_length).mean()) * rate,
        ar=float(ar_nll(train, test, order=ar_order).mean()) * rate,
        ar_order=ar_order,
    )


def format_table(reports: list[LikelihoodReport]) -> str:
    """Render reports with the reference rows and deltas inline, never apart."""
    order = reports[0].ar_order if reports else 16
    head = (
        f"{'model':<20} {'NLL':>9} {'white':>9} {'PSD-null':>9} {'Whittle':>9} "
        f"{f'AR({order})':>9} {'d PSD':>9} {'d AR':>9}\n"
        f"  nats / s / sensor, held out; lower is better; d < 0 means the model "
        f"beats that null"
    )
    rule = "-" * 100
    return "\n".join([head, rule, *(r.row() for r in reports), rule])
