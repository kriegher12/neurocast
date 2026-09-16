"""Surrogate data generators for signal-blind controls.

These exist to answer one question: *does the decoder still work when the brain
signal has been destroyed but its statistics preserved?*

The motivating failure is arXiv 2605.24524, where signal-blind Gaussian noise
reached 66.3% Rank@1 on a non-invasive brain-to-language retrieval task. A
decoder that scores above chance on a surrogate is reading structure that is not
neural -- window boundaries, session drift, class priors, or amplitude.

Surrogates are ordered by strictness. Each preserves more of the real data than
the last, so passing a later one is a stronger claim:

    gaussian        - preserves per-channel mean/variance only
    ar              - preserves per-channel linear autocorrelation to order p
    phase_randomized- preserves the exact power spectrum (and, in multivariate
                      mode, the full cross-channel covariance)

``phase_randomized`` is the one that matters for this project. Because it
preserves the aperiodic 1/f slope exactly, and 1/f is the identity carrier
FMScope names (removing it drops the subject probe by 9-19pp), a model that
scores above chance on phase-randomized surrogates is using spectral identity
information rather than neural dynamics.

All functions take and return arrays shaped ``(n_trials, n_channels, n_times)``
and never modify their input.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gaussian_surrogate",
    "ar_surrogate",
    "phase_randomized_surrogate",
    "circular_time_shift",
    "shuffle_channels",
    "dc_only",
]


def _check(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_times), got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("input contains non-finite values; clean before surrogating")
    return x


def gaussian_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise matched to each channel's mean and variance.

    The weakest control: destroys everything except first- and second-order
    marginal statistics. If a decoder beats chance here, the problem is in the
    evaluation protocol, not the model.
    """
    x = _check(x)
    mu = x.mean(axis=(0, 2), keepdims=True)
    sd = x.std(axis=(0, 2), keepdims=True)
    return mu + sd * rng.standard_normal(x.shape)


def ar_surrogate(x: np.ndarray, rng: np.random.Generator, order: int = 16) -> np.ndarray:
    """Per-channel AR(p) surrogate fitted by Yule-Walker.

    Preserves linear autocorrelation up to ``order`` lags. Sits between the
    Gaussian and phase-randomized surrogates in strictness.
    """
    x = _check(x)
    n_trials, n_ch, n_t = x.shape
    if order >= n_t:
        raise ValueError(f"order={order} must be < n_times={n_t}")
    out = np.empty_like(x, dtype=float)

    for c in range(n_ch):
        # Pool trials for a stable autocovariance estimate.
        sig = x[:, c, :].astype(float)
        sig = sig - sig.mean()
        acov = np.array(
            [np.mean(sig[:, : n_t - k] * sig[:, k:]) for k in range(order + 1)]
        )
        # Yule-Walker: R a = r, with R the Toeplitz autocovariance matrix.
        R = np.empty((order, order))
        for i in range(order):
            for j in range(order):
                R[i, j] = acov[abs(i - j)]
        R.flat[:: order + 1] += 1e-10 * max(acov[0], 1e-30)  # ridge for conditioning
        try:
            coef = np.linalg.solve(R, acov[1 : order + 1])
        except np.linalg.LinAlgError:
            coef = np.zeros(order)
        resid_var = max(acov[0] - coef @ acov[1 : order + 1], 1e-30)
        scale = np.sqrt(resid_var)

        # Simulate with burn-in so the process reaches stationarity.
        burn = 10 * order
        buf = np.zeros((n_trials, n_t + burn))
        noise = scale * rng.standard_normal((n_trials, n_t + burn))
        for t in range(order, n_t + burn):
            buf[:, t] = buf[:, t - order : t][:, ::-1] @ coef + noise[:, t]
        out[:, c, :] = buf[:, burn:] + x[:, c, :].mean()

    return out


def phase_randomized_surrogate(
    x: np.ndarray, rng: np.random.Generator, *, multivariate: bool = True
) -> np.ndarray:
    """Fourier phase-randomized surrogate. Preserves the power spectrum exactly.

    This is the strict control. Because the power spectrum is preserved bin for
    bin, so is the aperiodic 1/f slope, the alpha peak, and total band power --
    every linear spectral feature a model might use as a subject fingerprint.

    Parameters
    ----------
    multivariate
        If True (default), the *same* random phase shift is applied to every
        channel, which preserves the full cross-channel covariance and
        cross-spectrum. This is the harder test: only nonlinear and
        phase-dependent structure is destroyed.
        If False, each channel is randomized independently, which additionally
        destroys spatial covariance -- a weaker control, but useful for
        isolating how much the model relies on topography.

    Notes
    -----
    Hermitian symmetry is maintained by construction (``rfft``/``irfft``), and
    the DC and Nyquist bins are left untouched because they must stay real.
    """
    x = _check(x)
    n_trials, n_ch, n_t = x.shape

    mean = x.mean(axis=2, keepdims=True)
    spec = np.fft.rfft(x - mean, axis=2)
    n_freq = spec.shape[2]

    # Bins 0 (DC) and, for even n_t, the Nyquist bin must remain real-valued.
    lo = 1
    hi = n_freq - 1 if n_t % 2 == 0 else n_freq

    shape = (n_trials, 1, hi - lo) if multivariate else (n_trials, n_ch, hi - lo)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=shape)

    rot = np.ones((n_trials, n_ch, n_freq), dtype=complex)
    rot[:, :, lo:hi] = np.exp(1j * phases)

    out = np.fft.irfft(spec * rot, n=n_t, axis=2)
    return out + mean


def circular_time_shift(x: np.ndarray, shift_samples: int) -> np.ndarray:
    """Circularly shift trials in time, breaking stimulus-response alignment.

    Preserves every within-trial statistic exactly; only the alignment between
    neural data and label is destroyed. Used for the time-shift control.
    """
    x = _check(x)
    return np.roll(x, shift_samples, axis=2)


def shuffle_channels(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute channel order with one fixed permutation across all trials.

    Destroys spatial topography while preserving every per-channel time series
    intact. Quantifies how much the decoder depends on sensor geometry.
    """
    x = _check(x)
    perm = rng.permutation(x.shape[1])
    return x[:, perm, :]


def dc_only(x: np.ndarray) -> np.ndarray:
    """Replace each trial-channel with its constant temporal mean.

    Tests for an offset shortcut: a per-trial DC level that correlates with the
    label. This deserves its own control because a constant offset survives
    *every* spectral surrogate -- DC is part of the power spectrum, so phase
    randomisation preserves it exactly. A condition-correlated offset (slow
    drift, block structure, impedance change, head movement between blocks) is
    therefore invisible to the phase-randomised control and needs testing
    directly.

    Discovered the hard way while validating this suite: a Gaussian-windowed
    sine has a large DC component, and sign-flipping it by condition creates a
    pure offset confound that phase randomisation faithfully carries through.
    """
    x = _check(x)
    return np.repeat(x.mean(axis=2, keepdims=True), x.shape[2], axis=2)
