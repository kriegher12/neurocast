"""INV-1: the frozen canonical observation space and log|det J| bookkeeping.

Why this module exists
----------------------
The project's central experiment compares pretraining objectives by their
likelihood. That comparison is meaningless unless every arm reports likelihood
over *the same observation space*.

Concretely: a model that normalises its input more aggressively shrinks the
support of its density, which inflates log p(y) for free. An arm that divides by
a per-session standard deviation will beat an arm that does not, having learned
nothing. The literature is full of "nats" that cannot be compared across papers
for exactly this reason.

    INV-1  All likelihoods are reported in nats per second per sensor, in a
           frozen, non-learned canonical space. Any normalisation is an explicit
           invertible affine transform whose log|det J| is added back into the
           reported number.

With INV-1 held, normalisation stops being a hidden preprocessing choice and
becomes an *experimental factor* you can measure. The gap between
``TYPE_CONST`` and ``SESSION_ROBUST`` downstream accuracy is itself a
measurement: it says how much of the decoding rides on amplitude and 1/f --
which is the roadmap's cognition-versus-measurement-statistics question, for
free, in every results table.

Canonical space
---------------
* 250 Hz, matching the LibriBrain release (1 kHz -> head-motion correction ->
  Maxwell filter -> 50/100 Hz notch -> 0.1-125 Hz bandpass -> 250 Hz).
* Fixed per-type scale constants, NOT per-session statistics. Using session
  statistics here would bake a per-session transform into the "canonical" space
  and destroy comparability across sessions -- the exact bug this module exists
  to prevent.
* Amplitude is preserved, so the aperiodic 1/f slope and absolute power survive.
  That is deliberate: those carry subject identity, and we want to *measure*
  their contribution, not silently remove it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = [
    "CANONICAL_FS",
    "ChannelType",
    "CANONICAL_SCALE",
    "AffineTransform",
    "NormalizationRegime",
    "to_canonical",
    "fit_normalizer",
    "nats_per_second_per_sensor",
]

CANONICAL_FS = 250.0


class ChannelType(str, Enum):
    """Sensor families. Scale constants below are per-family, never per-session."""

    MAG = "mag"              # MEG magnetometer, tesla
    GRAD = "grad"            # MEG planar gradiometer, tesla/metre
    EEG = "eeg"              # scalp electrode, volts
    OPM = "opm"              # optically-pumped magnetometer, tesla
    SEEG = "seeg"            # stereo-EEG depth contact, volts
    ECOG = "ecog"            # subdural grid contact, volts
    EOG = "eog"              # electro-oculogram, volts
    ECG = "ecg"              # electrocardiogram, volts
    EMG = "emg"              # electromyogram, volts
    RESP = "resp"            # respiration belt, arbitrary
    GSR = "gsr"              # galvanic skin response, siemens
    IMU = "imu"              # accelerometer, m/s^2


#: Divisors that map each sensor family onto roughly unit variance in healthy
#: recordings. These are FROZEN constants shipped with the checkpoint. Changing
#: one invalidates every previously reported likelihood, so treat edits here the
#: way you would treat editing a unit system.
CANONICAL_SCALE: dict[ChannelType, float] = {
    ChannelType.MAG: 1e-12,      # ~pT
    ChannelType.GRAD: 4e-11,     # ~40 fT/cm
    ChannelType.EEG: 5e-5,       # ~50 uV
    ChannelType.OPM: 1e-12,
    ChannelType.SEEG: 5e-4,
    ChannelType.ECOG: 5e-4,
    ChannelType.EOG: 1e-4,
    ChannelType.ECG: 5e-4,
    ChannelType.EMG: 5e-5,
    ChannelType.RESP: 1.0,
    ChannelType.GSR: 1.0,
    ChannelType.IMU: 1.0,
}


@dataclass(frozen=True)
class AffineTransform:
    """Per-channel invertible affine map ``y = scale * x + shift``.

    Carries its own Jacobian so a likelihood computed in transformed space can
    always be pushed back to canonical space. This is the whole point: the
    transform is never allowed to be implicit.
    """

    scale: np.ndarray  # (n_channels,), strictly non-zero
    shift: np.ndarray  # (n_channels,)
    name: str = "affine"

    def __post_init__(self) -> None:
        scale = np.asarray(self.scale, dtype=float)
        shift = np.asarray(self.shift, dtype=float)
        if scale.shape != shift.shape or scale.ndim != 1:
            raise ValueError(
                f"scale and shift must both be 1-D and equal length, "
                f"got {scale.shape} and {shift.shape}"
            )
        if not np.all(np.isfinite(scale)) or np.any(scale == 0):
            raise ValueError("scale must be finite and non-zero to stay invertible")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "shift", shift)

    @property
    def n_channels(self) -> int:
        return int(self.scale.shape[0])

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Map canonical -> transformed. ``x`` is (..., n_channels, n_times)."""
        self._check(x)
        return x * self.scale[:, None] + self.shift[:, None]

    def invert(self, y: np.ndarray) -> np.ndarray:
        """Map transformed -> canonical."""
        self._check(y)
        return (y - self.shift[:, None]) / self.scale[:, None]

    def log_det_jacobian(self, n_times: int) -> float:
        """log|det dy/dx| for one (n_channels, n_times) block.

        The map is diagonal and applied identically at every sample, so the
        determinant is the per-channel scale raised to the number of samples.
        """
        if n_times <= 0:
            raise ValueError(f"n_times must be positive, got {n_times}")
        return float(n_times * np.sum(np.log(np.abs(self.scale))))

    def to_canonical_logprob(self, logprob_transformed: float, n_times: int) -> float:
        """Convert a log-density computed in transformed space to canonical space.

        Change of variables: ``p_x(x) = p_y(y) * |det dy/dx|``, so

            log p_x(x) = log p_y(y) + log|det J|

        Every arm of the bake-off must funnel its reported likelihood through
        this call. That is what makes the numbers comparable.
        """
        return float(logprob_transformed) + self.log_det_jacobian(n_times)

    def compose(self, other: "AffineTransform") -> "AffineTransform":
        """Return the transform equivalent to ``other.apply(self.apply(x))``."""
        if other.n_channels != self.n_channels:
            raise ValueError(
                f"channel mismatch: {self.n_channels} vs {other.n_channels}"
            )
        return AffineTransform(
            scale=self.scale * other.scale,
            shift=other.scale * self.shift + other.shift,
            name=f"{self.name}+{other.name}",
        )

    @classmethod
    def identity(cls, n_channels: int) -> "AffineTransform":
        return cls(np.ones(n_channels), np.zeros(n_channels), name="identity")

    def _check(self, x: np.ndarray) -> None:
        if x.ndim < 2 or x.shape[-2] != self.n_channels:
            raise ValueError(
                f"expected (..., {self.n_channels}, n_times), got {x.shape}"
            )


class NormalizationRegime(str, Enum):
    """Normalisation as an experimental factor, not a hidden choice.

    TYPE_CONST
        Fixed per-family constants only. Preserves amplitude, so the 1/f slope
        and absolute power survive. This is the default for likelihood work
        because it keeps the observation space frozen across sessions.

    SESSION_ROBUST
        Per-session median/MAD standardisation. Removes gain and amplitude
        carriers of subject identity. Use it to measure how much performance
        depends on them -- the delta against TYPE_CONST is the measurement.
    """

    TYPE_CONST = "type-const"
    SESSION_ROBUST = "session-robust"


def to_canonical(
    x: np.ndarray, ch_types: list[ChannelType] | list[str]
) -> tuple[np.ndarray, AffineTransform]:
    """Map raw physical units into the frozen canonical space.

    Parameters
    ----------
    x
        ``(..., n_channels, n_times)`` in native physical units (T, T/m, V).
    ch_types
        One :class:`ChannelType` per channel.

    Returns
    -------
    (x_canonical, transform)
        ``transform`` maps raw -> canonical and carries the Jacobian needed to
        report likelihood back in raw units if ever required.
    """
    types = [ChannelType(t) for t in ch_types]
    if x.ndim < 2 or x.shape[-2] != len(types):
        raise ValueError(f"x has {x.shape[-2]} channels but {len(types)} types given")
    scale = np.array([1.0 / CANONICAL_SCALE[t] for t in types], dtype=float)
    tf = AffineTransform(scale=scale, shift=np.zeros(len(types)), name="to_canonical")
    return tf.apply(x), tf


def fit_normalizer(
    x: np.ndarray,
    regime: NormalizationRegime = NormalizationRegime.TYPE_CONST,
    *,
    clip: float | None = 20.0,
) -> AffineTransform:
    """Fit the (invertible) normalisation for a session.

    ``x`` is already in canonical space, shaped ``(..., n_channels, n_times)``.

    Note the deliberate omission: clipping is NOT part of the returned
    transform, because clipping is not invertible and would silently break the
    change-of-variables identity. If you clip, you are no longer reporting a
    density over the canonical space, and ``to_canonical_logprob`` will quietly
    lie. Clip for optimisation stability if you must, but never inside the
    likelihood path -- ``clip`` is accepted here only so callers can record the
    value they used alongside the transform.
    """
    n_ch = x.shape[-2]
    if regime is NormalizationRegime.TYPE_CONST:
        return AffineTransform.identity(n_ch)

    if regime is NormalizationRegime.SESSION_ROBUST:
        flat = x.reshape(-1, n_ch, x.shape[-1]).transpose(1, 0, 2).reshape(n_ch, -1)
        med = np.median(flat, axis=1)
        mad = np.median(np.abs(flat - med[:, None]), axis=1)
        sd = 1.4826 * mad
        # Guard against dead channels, which would otherwise make scale infinite.
        sd = np.where(sd < 1e-12, 1.0, sd)
        return AffineTransform(
            scale=1.0 / sd, shift=-med / sd, name="session-robust"
        )

    raise ValueError(f"unknown regime {regime!r}")


def nats_per_second_per_sensor(
    logprob: float, n_channels: int, n_times: int, fs: float = CANONICAL_FS
) -> float:
    """Convert a block log-density to the project's reporting unit.

    Returns negative log-likelihood in nats per second per sensor, so numbers
    are comparable across window lengths, channel counts, and sampling rates.

    Always report this alongside two references, never alone:
      1. delta against a Whittle PSD-matched stationary Gaussian, and
      2. delta against a per-session linear AR(16).

    A model that beats white noise but not the PSD-matched Gaussian has learned
    1/f, which is precisely the subject-identity carrier the Identity Trap
    diagnoses. Reporting (1) makes that impossible to hide.
    """
    if n_channels <= 0 or n_times <= 0 or fs <= 0:
        raise ValueError("n_channels, n_times and fs must all be positive")
    seconds = n_times / fs
    return float(-logprob / (n_channels * seconds))
