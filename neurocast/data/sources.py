"""Source-space synthetic corpus: the same brain, seen through any sensor array.

:class:`neurocast.data.synthetic.SyntheticCorpus` generates each channel
independently, so a model trained on one montage has nothing *spatial* to carry
to another. It cannot test H5. This corpus fixes that.

A small set of dipolar cortical sources -- each with its own oscillatory
dynamics, plus a label-dependent evoked response -- is projected to whatever
montage is supplied **at sampling time**:

* magnetic sensors (MAG, GRAD, OPM) see ``((q x d) . n) / |d|^3``, Biot-Savart-like;
* electric sensors (EEG, sEEG, ECoG) see ``(q . d) / |d|^3``, a dipole potential;
* peripheral channels see nothing brain-derived.

So a 306-channel MEGIN helmet and a 64-channel EEG cap record **the same brain**
through genuinely different physics. Geometry-conditioned transfer has something
real to transfer, and a model that memorised channel indices has nothing.

Subject identity lives where it does in real recordings: a per-subject aperiodic
exponent, per-subject source gains, and per-subject channel gains. The task label
lives in a subject-invariant spatio-temporal evoked pattern across sources.

Spectra are shaped in the frequency domain (resonant peak plus a 1/f floor), so
generation is vectorised. Evoked templates are zero-mean -- a condition-correlated
DC offset survives every spectral surrogate, a trap this project has hit once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .canonical import ChannelType

__all__ = ["SourceBatch", "SourceSpaceCorpus"]

_MAGNETIC = frozenset({ChannelType.MAG, ChannelType.GRAD, ChannelType.OPM})


@dataclass(frozen=True)
class SourceBatch:
    x: np.ndarray          # (n, C, T) sensor data, canonical units
    sources: np.ndarray    # (n, K, T) latent source activity
    subject: np.ndarray    # (n,)
    label: np.ndarray      # (n,)
    stimulus: np.ndarray   # (n, T) evoked drive envelope
    fs: float


class SourceSpaceCorpus:
    """Latent dipolar sources projected through a physical lead field.

    Parameters
    ----------
    n_sources
        Latent cortical sources.
    evoked_strength
        Amplitude of the label-dependent evoked response, relative to ongoing
        source activity (which is normalised to unit variance).
    """

    def __init__(
        self,
        *,
        n_sources: int = 6,
        n_subjects: int = 8,
        n_labels: int = 4,
        fs: float = 250.0,
        evoked_strength: float = 0.8,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.fs = float(fs)
        self.n_sources = int(n_sources)
        self.n_subjects = int(n_subjects)
        self.n_labels = int(n_labels)
        self.evoked_strength = float(evoked_strength)

        pos = rng.standard_normal((n_sources, 3))
        pos[:, 2] = np.abs(pos[:, 2]) * 0.6 + 0.2
        pos /= np.linalg.norm(pos, axis=1, keepdims=True)
        self.src_pos = pos * 0.065                       # inside a ~0.1 m head
        tang = np.cross(self.src_pos, rng.standard_normal((n_sources, 3)))
        self.src_ori = tang / np.linalg.norm(tang, axis=1, keepdims=True)

        self.peak_hz = rng.choice([6.0, 10.0, 11.0, 18.0, 22.0], size=n_sources)
        self.subject_exponent = 1.0 + 0.6 * rng.uniform(-0.5, 0.5, n_subjects)
        self.subject_source_gain = np.exp(0.3 * rng.standard_normal((n_subjects, n_sources)))
        self.label_pattern = rng.standard_normal((n_labels, n_sources))
        self._gain_seed = int(rng.integers(1 << 30))

    def leadfield(self, montage) -> np.ndarray:
        """``(C, K)`` projection from sources to the montage's sensors."""
        feats = montage.feature_matrix().astype(float)
        r_s, n_s = feats[:, 0:3], feats[:, 3:6]
        d = r_s[:, None, :] - self.src_pos[None, :, :]
        dist3 = np.maximum(np.linalg.norm(d, axis=-1) ** 3, 1e-9)
        cross = np.cross(np.broadcast_to(self.src_ori[None], d.shape), d)
        magnetic = np.einsum("ckj,cj->ck", cross, n_s) / dist3
        electric = np.einsum("kj,ckj->ck", self.src_ori, d) / dist3

        lf = np.zeros((len(montage), self.n_sources))
        for c, sensor in enumerate(montage.sensors):
            if sensor.is_peripheral:
                continue
            lf[c] = magnetic[c] if sensor.ch_type in _MAGNETIC else electric[c]
        active = np.any(lf != 0, axis=1)
        rms = float(np.sqrt(np.mean(lf[active] ** 2))) if active.any() else 1.0
        return lf / max(rms, 1e-12)

    def _channel_gain(self, montage, subject: int) -> np.ndarray:
        rng = np.random.default_rng(self._gain_seed + 7919 * int(subject) + len(montage))
        return np.exp(0.2 * rng.standard_normal(len(montage)))

    def source_activity(
        self, n: int, n_times: int, subject: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """``(n, K, T)`` ongoing activity: resonant peak plus subject-specific 1/f."""
        f = np.fft.rfftfreq(n_times, d=1.0 / self.fs)
        white = np.fft.rfft(rng.standard_normal((n, self.n_sources, n_times)), axis=-1)
        peak = 1.0 / (1.0 + ((f[None, :] - self.peak_hz[:, None]) / 2.0) ** 2)
        out = np.empty((n, self.n_sources, n_times))
        for s in np.unique(subject):
            m = subject == s
            floor = np.ones_like(f)
            floor[1:] = f[1:] ** (-self.subject_exponent[s])
            spec = (3.0 * peak + floor[None, :]) * self.subject_source_gain[s][:, None]
            out[m] = np.fft.irfft(white[m] * np.sqrt(spec)[None], n=n_times, axis=-1)
        return out / out.std()

    def evoked(self, n_times: int) -> np.ndarray:
        """``(n_times,)`` zero-mean evoked time course, peaking ~150 ms post-onset."""
        t = np.arange(n_times) / self.fs
        onset = 0.2 * n_times / self.fs
        w = np.exp(-((t - onset - 0.15) ** 2) / (2 * 0.05**2))
        tpl = w * np.sin(2 * np.pi * 8.0 * (t - onset))
        return tpl - tpl.mean()

    def batch(
        self,
        montage,
        n: int,
        n_times: int,
        rng: np.random.Generator,
        *,
        subjects=None,
        noise: float = 0.3,
    ) -> SourceBatch:
        """Draw ``n`` trials projected to ``montage``."""
        pool = np.arange(self.n_subjects) if subjects is None else np.asarray(subjects)
        subject = rng.choice(pool, size=n)
        label = rng.integers(0, self.n_labels, size=n)

        src = self.source_activity(n, n_times, subject, rng)
        tpl = self.evoked(n_times)
        src = src + self.evoked_strength * self.label_pattern[label][:, :, None] * tpl

        x = np.einsum("ck,nkt->nct", self.leadfield(montage), src)
        for s in np.unique(subject):
            m = subject == s
            x[m] *= self._channel_gain(montage, s)[None, :, None]
        x = x + noise * rng.standard_normal(x.shape)

        stim = np.broadcast_to(np.abs(tpl), (n, n_times)).copy()
        return SourceBatch(x=x, sources=src, subject=subject, label=label,
                           stimulus=stim, fs=self.fs)
