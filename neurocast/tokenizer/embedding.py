"""Descriptor -> vector. Frozen Fourier features plus a small MLP.

Two properties matter here, and both are structural rather than learned:

**Determinism across runs and arrays.** The random projection matrices are
sampled once from a fixed seed and registered as buffers, so they travel with
the checkpoint. A given sensor position always maps to the same features, in
training and at deployment, on a montage the model has never seen.

**No new parameters for a new montage.** Everything is a function of physical
descriptor fields. Adding a 64-channel EEG cap, an OPM helmet, or a depth
electrode requires fitting nothing. Compare a learned per-channel table, which
must be extended and identified for every new array.

Bandwidth choice
----------------
Positions are in metres and a head spans roughly 0.2 m. A Fourier feature with
frequency sigma resolves structure at scale ~1/sigma, so the position scales
{8, 16, 32, 64} m^-1 span 12.5 cm (whole-head gradients) down to 1.5 cm --
which is about MEG's spatial resolution limit. Going finer would encode
positional noise: co-registration error alone is several millimetres.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data.canonical import ChannelType
from .descriptor import Montage, Quadrant, Reference

__all__ = [
    "GaussianFourierFeatures",
    "SensorEmbedding",
    "POSITION_SCALES",
    "ORIENTATION_SCALES",
]

#: Spatial frequencies for position encoding, in m^-1. See module docstring.
POSITION_SCALES: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0)

#: Frequencies for unit-vector orientation encoding (dimensionless).
ORIENTATION_SCALES: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)


class GaussianFourierFeatures(nn.Module):
    """Random Fourier features with a frozen, seeded projection.

    Maps ``x`` of shape ``(..., in_dim)`` to
    ``[sin(2 pi Bx), cos(2 pi Bx)]`` of shape ``(..., 2 * n_pairs * n_scales)``,
    with ``B`` drawn once from ``N(0, sigma^2 I)`` per scale.

    The matrix is a buffer, not a parameter: it is saved with the checkpoint and
    never receives gradient. Making it learnable would defeat the point -- the
    embedding of an unseen sensor position must not depend on which positions
    happened to appear in training.
    """

    def __init__(
        self,
        in_dim: int,
        scales: tuple[float, ...],
        n_pairs_per_scale: int = 32,
        seed: int = 0xC0FFEE,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or n_pairs_per_scale <= 0 or not scales:
            raise ValueError("in_dim, n_pairs_per_scale and scales must be non-empty")
        self.in_dim = in_dim
        self.scales = tuple(float(s) for s in scales)
        self.n_pairs_per_scale = int(n_pairs_per_scale)

        gen = torch.Generator().manual_seed(seed)
        blocks = [
            torch.randn(in_dim, n_pairs_per_scale, generator=gen) * s
            for s in self.scales
        ]
        self.register_buffer("B", torch.cat(blocks, dim=1), persistent=True)

    @property
    def out_dim(self) -> int:
        return 2 * self.n_pairs_per_scale * len(self.scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"expected last dim {self.in_dim}, got {x.shape[-1]}")
        proj = 2.0 * torch.pi * (x.to(self.B.dtype) @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class SensorEmbedding(nn.Module):
    """Embed a montage's descriptors into ``d_sensor``-dimensional vectors.

    Output is added to every temporal patch token for that channel, so the model
    always knows *which sensor, pointing which way* produced a given sample.

    Parameters
    ----------
    d_sensor
        Output width. 256 by default, matching the patch embedding.
    """

    def __init__(
        self,
        d_sensor: int = 256,
        *,
        n_pairs_per_scale: int = 32,
        hidden: int = 512,
        seed: int = 0xC0FFEE,
    ) -> None:
        super().__init__()
        self.d_sensor = int(d_sensor)

        self.gamma_pos = GaussianFourierFeatures(
            3, POSITION_SCALES, n_pairs_per_scale, seed
        )
        self.gamma_ori = GaussianFourierFeatures(
            3, ORIENTATION_SCALES, n_pairs_per_scale // 2, seed + 1
        )
        # Scalars (baseline length, head radius, log scale, contact size) share
        # one encoder: they are all smooth 1-D quantities.
        self.gamma_scalar = GaussianFourierFeatures(
            1, (1.0, 4.0, 16.0), n_pairs_per_scale // 4, seed + 2
        )

        self.type_emb = nn.Embedding(len(ChannelType), 32)
        self.ref_emb = nn.Embedding(len(Reference), 8)
        self.bad_emb = nn.Embedding(2, 8)

        in_dim = (
            self.gamma_pos.out_dim          # position
            + 2 * self.gamma_ori.out_dim    # normal + baseline direction
            + 4 * self.gamma_scalar.out_dim # baseline_m, radius, log_scale, contact
            + 32 + 8 + 8                    # type, reference, bad
            + 1                             # is_intracranial
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_sensor),
            nn.LayerNorm(d_sensor),
        )

    @staticmethod
    def montage_tensors(
        montage: Montage, device: torch.device | str = "cpu"
    ) -> dict[str, torch.Tensor]:
        """Pack a :class:`Montage` into the tensors :meth:`forward` expects.

        Done once per montage and cached -- it is pure geometry and does not
        change between batches.
        """
        feats = torch.from_numpy(montage.feature_matrix()).to(device)
        return {
            "pos": feats[:, 0:3],
            "ori_normal": feats[:, 3:6],
            "ori_base": feats[:, 6:9],
            "baseline_m": feats[:, 9:10],
            "radius": feats[:, 10:11],
            "log_scale": feats[:, 11:12],
            "contact_mm": feats[:, 12:13],
            "is_intra": feats[:, 13:14],
            "is_bad": feats[:, 14:15],
            "type_id": feats[:, 15].long(),
            "ref_id": feats[:, 16].long(),
            "quadrant": torch.from_numpy(montage.quadrants).to(device),
        }

    def forward(self, t: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return ``(n_channels, d_sensor)`` sensor embeddings."""
        parts = [
            self.gamma_pos(t["pos"]),
            self.gamma_ori(t["ori_normal"]),
            self.gamma_ori(t["ori_base"]),
            self.gamma_scalar(t["baseline_m"]),
            self.gamma_scalar(t["radius"]),
            self.gamma_scalar(t["log_scale"]),
            self.gamma_scalar(t["contact_mm"]),
            self.type_emb(t["type_id"]),
            self.ref_emb(t["ref_id"]),
            self.bad_emb(t["is_bad"].squeeze(-1).long()),
            t["is_intra"],
        ]
        return self.mlp(torch.cat(parts, dim=-1))

    def n_parameters(self) -> int:
        """Trainable parameter count. Independent of channel count, by design."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
