"""Synthetic corpus with known ground truth, for rehearsing the bake-off.

Real MEG is expensive to download and slow to iterate on. This corpus is built
so the Phase-1 apparatus can be exercised end to end in seconds, and -- more
usefully -- so the *right answer is known in advance*.

The generative structure deliberately mirrors the problem H2 is about:

**Subject identity is strong and easy.** Each subject gets its own aperiodic 1/f
exponent and a per-channel gain vector. These are the exact carriers FMScope
names: removing the aperiodic component drops the subject probe by 9-19pp in
real models. Here they are large, stable, and present in every sample.

**The task label is weak and hard.** A small zero-mean evoked template, shared
across all subjects, added at low amplitude. It is the only thing a decoder
should be reading.

So a representation that scores well on subject identity and poorly on label has
taken exactly the shortcut the Identity Trap describes. Any objective that
resists that on synthetic data has at least a chance of resisting it on real
data; one that fails here will certainly fail there.

Templates are zero-mean by construction. A windowed sinusoid is not, and a
condition-correlated DC offset survives every spectral surrogate -- a trap this
project hit once already while validating the control suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..tokenizer.descriptor import Montage

__all__ = ["SyntheticCorpus", "Batch"]


@dataclass(frozen=True)
class Batch:
    x: np.ndarray          # (n, C, T) canonical units
    subject: np.ndarray    # (n,) subject index
    label: np.ndarray      # (n,) task label
    fs: float


class SyntheticCorpus:
    """Subject-fingerprinted noise plus a weak shared evoked response.

    Parameters
    ----------
    identity_strength
        Spread of per-subject 1/f exponents and channel gains. Larger makes the
        identity shortcut more tempting.
    label_strength
        Amplitude of the evoked template, relative to noise. The default is
        deliberately low: an easy label would let every arm succeed and the
        comparison would say nothing.
    """

    def __init__(
        self,
        montage: Montage,
        *,
        n_subjects: int = 6,
        n_labels: int = 4,
        fs: float = 250.0,
        identity_strength: float = 1.0,
        label_strength: float = 0.30,
        seed: int = 0,
    ) -> None:
        self.montage = montage
        self.n_channels = len(montage)
        self.n_subjects = int(n_subjects)
        self.n_labels = int(n_labels)
        self.fs = float(fs)
        self.label_strength = float(label_strength)

        rng = np.random.default_rng(seed)
        # Per-subject fingerprint: aperiodic exponent and channel gains.
        self.exponents = 1.0 + identity_strength * rng.uniform(-0.5, 0.5, n_subjects)
        self.gains = np.exp(
            identity_strength * 0.35 * rng.standard_normal((n_subjects, self.n_channels))
        )
        self._template_seed = int(rng.integers(1 << 30))

    def _templates(self, n_times: int) -> np.ndarray:
        """``(n_labels, n_times)`` zero-mean evoked templates."""
        rng = np.random.default_rng(self._template_seed)
        t = np.arange(n_times) / self.fs
        out = []
        for _ in range(self.n_labels):
            centre = rng.uniform(0.15, 0.45) * (n_times / self.fs)
            freq = rng.uniform(5.0, 12.0)
            phase = rng.uniform(0, 2 * np.pi)
            w = np.exp(-((t - centre) ** 2) / (2 * 0.035**2))
            tpl = w * np.sin(2 * np.pi * freq * t + phase)
            out.append(tpl - tpl.mean())  # zero-mean: no DC shortcut
        return np.stack(out)

    def _colored_noise(
        self, rng: np.random.Generator, n: int, n_times: int, exponent: float
    ) -> np.ndarray:
        spec = np.fft.rfft(rng.standard_normal((n, self.n_channels, n_times)), axis=-1)
        f = np.fft.rfftfreq(n_times, d=1.0 / self.fs)
        scale = np.ones_like(f)
        scale[1:] = f[1:] ** (-exponent / 2.0)
        return np.fft.irfft(spec * scale, n=n_times, axis=-1)

    def batch(self, n: int, n_times: int, rng: np.random.Generator) -> Batch:
        """Draw ``n`` trials of ``n_times`` samples."""
        subject = rng.integers(0, self.n_subjects, size=n)
        label = rng.integers(0, self.n_labels, size=n)
        templates = self._templates(n_times)

        x = np.empty((n, self.n_channels, n_times))
        for s in range(self.n_subjects):
            m = subject == s
            if not m.any():
                continue
            noise = self._colored_noise(rng, int(m.sum()), n_times, self.exponents[s])
            x[m] = noise * self.gains[s][None, :, None]

        # Shared evoked response: identical across subjects, so it is the only
        # subject-invariant information in the signal.
        x += self.label_strength * templates[label][:, None, :]
        return Batch(x=x, subject=subject, label=label, fs=self.fs)

    def describe(self) -> str:
        return (
            f"SyntheticCorpus {self.n_subjects} subjects x {self.n_labels} labels, "
            f"{self.n_channels} ch\n"
            f"  1/f exponents {self.exponents.min():.2f}-{self.exponents.max():.2f} "
            f"(subject fingerprint, strong)\n"
            f"  label amplitude {self.label_strength:.2f} (shared evoked, weak)"
        )
