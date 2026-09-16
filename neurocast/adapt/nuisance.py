"""Tier-3 prompt content: the subject's nuisance structure, made explicit.

The H2 mechanism, in one paragraph. A subject's idiosyncratic measurement
structure -- aperiodic 1/f slope, channel gains, head position, heart rate,
blink rate -- is *visible in the context window*. A model that can read it from
the prompt has no incentive to store it in weights. Masked-reconstruction models
with no prompt have the opposite incentive: they must encode it to reconstruct
well, which is how frozen subject-variance reaches 13-89x a random null.

So why compute these by hand rather than let the model learn them?

1. They are precisely the carriers FMScope identifies. Removing the aperiodic
   component drops the subject probe by 9-19 percentage points. Making them
   explicit creates a **named, ablatable pathway** for nuisance rather than a
   diffuse one.
2. They are nearly free -- a Welch PSD and a line fit.
3. They enable the prompt-swap intervention. Exchanging Tier-3 tokens between
   subjects while holding brain data fixed tests whether identity follows the
   *tokens* or the *data*. If decoding survives while the identity probe follows
   the swapped tokens, that is causal evidence of separation between measurement
   statistics and cognition -- which no amount of correlational probing gives.

Bands follow the usual convention; the aperiodic fit deliberately excludes the
alpha range, because a strong alpha peak otherwise biases the slope estimate and
the slope is the quantity we care about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BANDS",
    "welch_psd",
    "aperiodic_fit",
    "band_power",
    "NuisanceDescriptors",
    "compute",
]

#: Standard frequency bands (Hz).
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 70.0),
    "high_gamma": (70.0, 125.0),
}

#: Range for the aperiodic fit, and the notch excluded from it.
_FIT_RANGE = (2.0, 45.0)
_ALPHA_EXCLUDE = (7.0, 14.0)


def welch_psd(
    x: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectral density by Welch's method. NumPy only.

    Implemented here rather than imported from ``scipy.signal`` so the audit and
    prompt path depend on nothing but NumPy -- the same rule the control suite
    follows. (It also sidesteps a real deployment problem: on the development
    machine an Application Control policy blocks one of scipy's compiled
    extensions, which would otherwise make this module unimportable.)

    Matches ``scipy.signal.welch`` defaults: periodic Hann window, 50% overlap,
    per-segment mean removal, density scaling.

    Returns ``(freqs, psd)`` with ``psd`` shaped ``(n_channels, n_freq)``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n_t = x.shape[-1]
    nperseg = int(min(nperseg, n_t))
    if nperseg < 8:
        raise ValueError(f"nperseg must be at least 8, got {nperseg}")
    noverlap = nperseg // 2 if noverlap is None else int(noverlap)
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("noverlap must be smaller than nperseg")

    # Periodic Hann, as used for spectral estimation (np.hanning is symmetric).
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nperseg) / nperseg)
    norm = fs * float((win**2).sum())

    starts = range(0, n_t - nperseg + 1, step)
    acc = None
    n_seg = 0
    for s in starts:
        seg = x[:, s : s + nperseg]
        seg = seg - seg.mean(axis=-1, keepdims=True)  # detrend='constant'
        spec = np.fft.rfft(seg * win, axis=-1)
        p = (spec.real**2 + spec.imag**2) / norm
        acc = p if acc is None else acc + p
        n_seg += 1
    if acc is None:
        raise ValueError(f"no complete segment of {nperseg} samples in {n_t}")

    psd = acc / n_seg
    # One-sided: fold negative frequencies in, leaving DC and Nyquist alone.
    psd[:, 1:-1] *= 2.0
    if nperseg % 2:
        psd[:, -1] *= 2.0
    return np.fft.rfftfreq(nperseg, d=1.0 / fs), psd


def aperiodic_fit(
    psd: np.ndarray, freqs: np.ndarray, *, exclude_alpha: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``log10(P) = offset - exponent * log10(f)`` per channel.

    A deliberately simple stand-in for FOOOF: a robust line fit in log-log space
    over 2-45 Hz, with the alpha band excluded so an oscillatory peak does not
    drag the slope. Returns ``(exponent, offset)``, each ``(n_channels,)``.

    The exponent is the quantity FMScope names as a subject carrier, so it is
    reported and ablated explicitly rather than left implicit.
    """
    psd = np.atleast_2d(np.asarray(psd, dtype=float))
    freqs = np.asarray(freqs, dtype=float)
    if psd.shape[-1] != freqs.shape[0]:
        raise ValueError(
            f"psd has {psd.shape[-1]} frequency bins but freqs has {freqs.shape[0]}"
        )

    keep = (freqs >= _FIT_RANGE[0]) & (freqs <= _FIT_RANGE[1])
    if exclude_alpha:
        keep &= ~((freqs >= _ALPHA_EXCLUDE[0]) & (freqs <= _ALPHA_EXCLUDE[1]))
    if keep.sum() < 3:
        raise ValueError("too few frequency bins survive the fit range")

    lf = np.log10(freqs[keep])
    lp = np.log10(np.maximum(psd[:, keep], 1e-30))
    design = np.stack([np.ones_like(lf), lf], axis=1)          # (n_freq, 2)
    coef, *_ = np.linalg.lstsq(design, lp.T, rcond=None)        # (2, n_channels)
    offset, slope = coef[0], coef[1]
    return -slope, offset


def band_power(
    psd: np.ndarray, freqs: np.ndarray, bands: dict[str, tuple[float, float]] | None = None
) -> dict[str, np.ndarray]:
    """Integrated power per band, per channel."""
    bands = bands or BANDS
    psd = np.atleast_2d(np.asarray(psd, dtype=float))
    freqs = np.asarray(freqs, dtype=float)
    out = {}
    for name, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi)
        out[name] = (
            np.trapezoid(psd[:, m], freqs[m], axis=1)
            if m.sum() > 1
            else np.zeros(psd.shape[0])
        )
    return out


@dataclass(frozen=True)
class NuisanceDescriptors:
    """Per-channel measurement statistics for one prompt.

    Every field here is a known or suspected subject-identity carrier. That is
    the point: they are gathered in one ablatable place.
    """

    exponent: np.ndarray       # (C,) aperiodic 1/f slope
    offset: np.ndarray         # (C,) aperiodic offset
    log_variance: np.ndarray   # (C,)
    log_band_power: np.ndarray # (C, n_bands)
    bad_mask: np.ndarray       # (C,) bool
    band_names: tuple[str, ...]

    @property
    def n_channels(self) -> int:
        return int(self.exponent.shape[0])

    def to_matrix(self) -> np.ndarray:
        """``(C, 3 + n_bands)`` feature matrix, ready for projection to tokens."""
        return np.concatenate(
            [
                self.exponent[:, None],
                self.offset[:, None],
                self.log_variance[:, None],
                self.log_band_power,
                self.bad_mask.astype(float)[:, None],
            ],
            axis=1,
        )

    def summary(self) -> str:
        return (
            f"exponent {self.exponent.mean():.2f}+-{self.exponent.std():.2f}  "
            f"log-var {self.log_variance.mean():.2f}  "
            f"bands {', '.join(self.band_names)}  "
            f"bad {int(self.bad_mask.sum())}/{self.n_channels}"
        )

    def perturb(
        self, *, d_exponent: float = 0.0, gain_factor: float = 1.0
    ) -> "NuisanceDescriptors":
        """Counterfactual nuisance injection, for the robustness control.

        Shifting the 1/f exponent by +-0.3 or channel gains by +-20% should move
        label accuracy by less than ~2 percentage points if the model is reading
        cognition rather than measurement statistics.
        """
        log_gain = float(np.log10(gain_factor))
        return NuisanceDescriptors(
            exponent=self.exponent + d_exponent,
            offset=self.offset + 2 * log_gain,
            log_variance=self.log_variance + 2 * log_gain,
            log_band_power=self.log_band_power + 2 * log_gain,
            bad_mask=self.bad_mask,
            band_names=self.band_names,
        )


def compute(
    x: np.ndarray,
    fs: float = 250.0,
    *,
    bad_mask: np.ndarray | None = None,
    nperseg: int | None = None,
) -> NuisanceDescriptors:
    """Compute nuisance descriptors from a prompt's raw signal.

    Parameters
    ----------
    x
        ``(n_channels, n_times)`` in canonical units.
    fs
        Sample rate.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (n_channels, n_times), got {x.shape}")
    n_ch, n_t = x.shape
    nperseg = int(nperseg or min(n_t, int(4 * fs)))
    if nperseg < 16:
        raise ValueError(f"need at least 16 samples to estimate a PSD, got {n_t}")

    freqs, psd = welch_psd(x, fs=fs, nperseg=nperseg)
    exponent, offset = aperiodic_fit(psd, freqs)
    powers = band_power(psd, freqs)

    return NuisanceDescriptors(
        exponent=exponent,
        offset=offset,
        log_variance=np.log(np.maximum(x.var(axis=1), 1e-30)),
        log_band_power=np.log(
            np.maximum(np.stack([powers[b] for b in BANDS], axis=1), 1e-30)
        ),
        bad_mask=(
            np.zeros(n_ch, dtype=bool) if bad_mask is None else np.asarray(bad_mask, bool)
        ),
        band_names=tuple(BANDS),
    )
