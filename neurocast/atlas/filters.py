"""Band-limiting that cannot see the future.

The Atlas asks how predictable a brain signal is from its past. Any filter
applied to that past must therefore be **causal** -- its output at time ``t``
may depend only on samples at ``t`` and earlier.

That sounds obvious and is violated constantly. The standard neuroscience tool,
zero-phase filtering (``filtfilt``: filter forward, then backward), is
non-causal by construction: the backward pass smears later samples into earlier
ones. Band-limit the *conditioning* side with it and the "past" now contains
the future. Forecastability comes out spectacular, and it is entirely an
artifact.

This module provides a causal windowed-sinc FIR, a zero-phase variant kept
**only** so the leakage can be demonstrated, and an impulse-response assertion
that fails any filter whose output precedes its input.

NumPy only, by the same rule as the rest of the audit path.

Rule, enforced by :func:`assert_causal`
    Conditioning-side filters must be causal. Zero-phase filtering may be used on
    a *target* only if the target window lies beyond the filter's impulse
    response support from the conditioning window.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "fir_bandpass",
    "causal_filter",
    "zero_phase_filter",
    "group_delay",
    "assert_causal",
    "leakage_energy",
]


def fir_bandpass(
    lo_hz: float, hi_hz: float, fs: float, n_taps: int = 129
) -> np.ndarray:
    """Windowed-sinc band-pass FIR (Hamming). Linear phase, odd length.

    ``lo_hz = 0`` gives a low-pass. Delay is ``(n_taps - 1) / 2`` samples, which
    a causal application introduces and a caller must account for when aligning
    filtered past with an unfiltered future.
    """
    if n_taps % 2 == 0:
        n_taps += 1
    nyq = fs / 2.0
    if not 0.0 <= lo_hz < hi_hz <= nyq:
        raise ValueError(f"need 0 <= lo < hi <= Nyquist ({nyq}), got {lo_hz}, {hi_hz}")

    n = np.arange(n_taps) - (n_taps - 1) / 2.0

    def lowpass(fc: float) -> np.ndarray:
        if fc <= 0:
            return np.zeros(n_taps)
        w = 2.0 * fc / fs
        return w * np.sinc(w * n)

    h = (lowpass(hi_hz) - lowpass(lo_hz)) * np.hamming(n_taps)
    # Normalise to unit gain at the band centre so filtered power is comparable
    # across bands.
    fc = 0.5 * (lo_hz + hi_hz)
    gain = abs(np.sum(h * np.exp(-2j * np.pi * fc / fs * np.arange(n_taps))))
    return h / gain if gain > 1e-12 else h


def group_delay(h: np.ndarray) -> int:
    """Delay, in samples, introduced by applying linear-phase ``h`` causally."""
    return (len(h) - 1) // 2


def causal_filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply ``h`` using only present and past samples.

    ``y[t] = sum_{k=0}^{M} h[k] * x[t - k]``. Works on the last axis of any
    shape; output has the same length as input, with the first ``M`` samples
    reflecting a zero history.
    """
    x = np.asarray(x, dtype=float)
    flat = x.reshape(-1, x.shape[-1])
    out = np.empty_like(flat)
    for i, row in enumerate(flat):
        out[i] = np.convolve(row, h, mode="full")[: row.shape[0]]
    return out.reshape(x.shape)


def zero_phase_filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Forward-backward filtering. **Non-causal.** For demonstrating leakage only.

    Never apply this to the conditioning side of a forecastability estimate.
    """
    once = causal_filter(x, h)
    return causal_filter(once[..., ::-1], h)[..., ::-1]


def leakage_energy(
    filt: Callable[[np.ndarray], np.ndarray], n: int = 1024, t0: int | None = None
) -> float:
    """Fraction of an impulse's response energy that lands *before* the impulse.

    Zero for any causal filter. Anything above zero means the filter's output at
    some time depends on input that arrives later.
    """
    t0 = n // 2 if t0 is None else t0
    impulse = np.zeros((1, n))
    impulse[0, t0] = 1.0
    y = np.asarray(filt(impulse))[0]
    total = float((y**2).sum())
    if total <= 0:
        return 0.0
    return float((y[:t0] ** 2).sum()) / total


def assert_causal(
    filt: Callable[[np.ndarray], np.ndarray], *, name: str = "filter", tol: float = 0.0
) -> None:
    """Raise if ``filt`` lets any energy precede an impulse.

    Encodes the design's ``test_no_future_leakage_through_filters``. Use it on
    every filter in the conditioning path, including ones assembled from
    library calls whose causality is not documented.
    """
    for t0 in (64, 512, 900):
        frac = leakage_energy(filt, n=1024, t0=t0)
        if frac > tol:
            raise AssertionError(
                f"{name} is non-causal: {100 * frac:.2f}% of an impulse's response "
                f"energy precedes the impulse (t0={t0}). Using it on the "
                "conditioning side would fabricate forecastability."
            )
