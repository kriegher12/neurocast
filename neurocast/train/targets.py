"""Observation-space prediction targets that can be turned back into sensors.

The bake-off rehearsal predicted a frozen random projection of each quadrant's
channel *average* -- enough to compare objectives, but lossy in a way that makes
per-sensor forecasting impossible. The design's actual target is richer:

    target for (patch t, quadrant g) = whitened PCA of that quadrant's
                                       sensor-by-sample patch block

:class:`QuadrantBasis` fits one PCA per quadrant on training data and freezes it.
Because PCA is linear and the retained components carry most of the variance, a
predicted target maps **back to per-sensor signal**. That is what lets the demo
render the model's own forecast as a topographic movie, sensor by sensor.

Two INV-1 details:

* Components are whitened to unit variance so the density head predicts on a
  sane scale. The whitening is an invertible linear map inside the retained
  subspace, and :meth:`QuadrantBasis.log_det_jacobian` reports its Jacobian so
  likelihoods can be pushed back to canonical units.
* Truncation discards variance. :meth:`QuadrantBasis.explained_variance` reports
  how much, per quadrant -- a likelihood over a subspace is not a likelihood over
  the full observation, and results must say which one they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["QuadrantBasis"]


@dataclass
class QuadrantBasis:
    """Frozen per-quadrant PCA over sensor patch blocks."""

    rank: int = 32
    patch: int = 16
    n_groups: int = 5
    means: list = field(default_factory=list)       # per group (n_g*P,)
    comps: list = field(default_factory=list)       # per group (r_g, n_g*P)
    scales: list = field(default_factory=list)      # per group (r_g,)
    channels: list = field(default_factory=list)    # per group channel indices
    evr: list = field(default_factory=list)         # per group explained variance ratio

    def fit(self, x: np.ndarray, quadrant: np.ndarray) -> "QuadrantBasis":
        """Fit on ``(n, C, T)`` training data."""
        x = np.asarray(x, dtype=float)
        n, c, t = x.shape
        n_patch = t // self.patch
        blocks = x[:, :, : n_patch * self.patch].reshape(n, c, n_patch, self.patch)
        self.means, self.comps, self.scales, self.channels, self.evr = [], [], [], [], []

        for g in range(self.n_groups):
            idx = np.flatnonzero(np.asarray(quadrant) == g)
            self.channels.append(idx)
            if idx.size == 0:
                self.means.append(np.zeros(0))
                self.comps.append(np.zeros((0, 0)))
                self.scales.append(np.zeros(0))
                self.evr.append(float("nan"))
                continue
            # (n * n_patch, n_g * P): each row is one quadrant patch block.
            v = blocks[:, idx].transpose(0, 2, 1, 3).reshape(n * n_patch, -1)
            mu = v.mean(axis=0)
            vc = v - mu
            _, sv, vt = np.linalg.svd(vc, full_matrices=False)
            r = min(self.rank, vt.shape[0])
            var = sv**2 / max(vc.shape[0] - 1, 1)
            self.means.append(mu)
            self.comps.append(vt[:r])
            self.scales.append(np.sqrt(np.maximum(var[:r], 1e-12)))
            self.evr.append(float(var[:r].sum() / max(var.sum(), 1e-12)))
        return self

    def explained_variance(self) -> list[float]:
        """Fraction of each quadrant's variance the retained components keep."""
        return list(self.evr)

    def log_det_jacobian(self) -> float:
        """log|det| of the whitening map, summed over quadrants, per patch.

        Whitening divides each retained component by its scale, so the Jacobian
        of observation-in-subspace -> target is ``-sum(log scale)``.
        """
        return float(-sum(np.log(s).sum() for s in self.scales if s.size))

    def encode(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` -> ``(B, T'*G, rank)`` whitened targets in AR order."""
        xt = torch.as_tensor(x, dtype=torch.float32)
        b, c, t = xt.shape
        n_patch = t // self.patch
        blocks = xt[:, :, : n_patch * self.patch].reshape(b, c, n_patch, self.patch)
        out = torch.zeros(b, n_patch, self.n_groups, self.rank)
        for g in range(self.n_groups):
            idx = self.channels[g]
            if idx.size == 0:
                continue
            v = blocks[:, torch.as_tensor(idx)].permute(0, 2, 1, 3).reshape(b, n_patch, -1)
            mu = torch.as_tensor(self.means[g], dtype=torch.float32)
            comp = torch.as_tensor(self.comps[g], dtype=torch.float32)
            scale = torch.as_tensor(self.scales[g], dtype=torch.float32)
            z = ((v - mu) @ comp.T) / scale
            out[:, :, g, : z.shape[-1]] = z
        return out.reshape(b, n_patch * self.n_groups, self.rank)

    def decode_group(self, g: int, z: np.ndarray | torch.Tensor) -> np.ndarray:
        """``(rank,)`` target for quadrant ``g`` -> ``(n_g, P)`` sensor patch."""
        idx = self.channels[g]
        if idx.size == 0:
            return np.zeros((0, self.patch))
        comp, scale = self.comps[g], self.scales[g]
        zz = np.asarray(z, dtype=float)[: comp.shape[0]]
        v = self.means[g] + (zz * scale) @ comp
        return v.reshape(idx.size, self.patch)

    def reconstruction_r2(self, x: np.ndarray) -> float:
        """Fraction of held-out variance recovered by encode -> decode."""
        x = np.asarray(x, dtype=float)
        z = self.encode(x).numpy()
        b, c, t = x.shape
        n_patch = t // self.patch
        rec = np.zeros((b, c, n_patch * self.patch))
        zz = z.reshape(b, n_patch, self.n_groups, self.rank)
        for g in range(self.n_groups):
            idx = self.channels[g]
            if idx.size == 0:
                continue
            for i in range(b):
                for p in range(n_patch):
                    rec[i, idx, p * self.patch:(p + 1) * self.patch] = self.decode_group(g, zz[i, p, g])
        target = x[:, :, : n_patch * self.patch]
        active = np.zeros(c, dtype=bool)
        for idx in self.channels:
            active[idx] = True
        num = ((target[:, active] - rec[:, active]) ** 2).sum()
        den = ((target[:, active] - target[:, active].mean()) ** 2).sum()
        return float(1.0 - num / max(den, 1e-12))
