"""Temporal patching: raw samples -> per-channel tokens.

Parameters follow the design pass:

===================  =========================================================
sample rate 250 Hz   native LibriBrain release; preserves 70-125 Hz high-gamma
patch P = 16         64 ms, about one 15.6 Hz frame -- long enough that a linear
                     projection captures within-patch spectrum, short enough for
                     sub-word temporal resolution
stride = P           NON-overlapping, and this is not negotiable: overlapping
                     patches break the autoregressive chain rule, because a
                     "future" token would contain samples already emitted
groups G = 5         4 anatomical quadrants + peripherals (see descriptor.py)
token rate 62.5/s    600 h of MEG -> ~1.35e8 sequence positions
===================  =========================================================

The projection is **per sensor family, not per sensor**. Twelve matrices, one
per :class:`ChannelType`, so a new montage of a known family needs no new
weights -- the same property the sensor embedding provides for geometry.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.canonical import ChannelType

__all__ = ["PATCH_SAMPLES", "TOKENS_PER_SECOND", "TypedPatchEmbed"]

PATCH_SAMPLES = 16
TOKENS_PER_SECOND = 250.0 / PATCH_SAMPLES  # 62.5


class TypedPatchEmbed(nn.Module):
    """Linear patch embedding with one projection per channel family.

    Input ``(B, C, T)`` in canonical units, output ``(B, C, T', d)`` where
    ``T' = T // patch``.
    """

    def __init__(self, d_model: int = 256, patch: int = PATCH_SAMPLES) -> None:
        super().__init__()
        self.patch = int(patch)
        self.d_model = int(d_model)
        n_types = len(ChannelType)
        # (n_types, patch, d) -- indexed by type id, applied by einsum. Scaled
        # like a standard Linear init so no family starts out dominant.
        self.weight = nn.Parameter(torch.randn(n_types, patch, d_model) / patch**0.5)
        self.bias = nn.Parameter(torch.zeros(n_types, d_model))

    def forward(self, x: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            ``(B, C, T)`` canonical-space signal. ``T`` must be divisible by
            ``patch``; a ragged tail would silently shift every later token, so
            it raises instead of truncating.
        type_id
            ``(C,)`` long tensor of :class:`ChannelType` indices.
        """
        if x.ndim != 3:
            raise ValueError(f"expected (B, C, T), got {tuple(x.shape)}")
        b, c, t = x.shape
        if type_id.shape != (c,):
            raise ValueError(f"type_id must be ({c},), got {tuple(type_id.shape)}")
        if t % self.patch:
            raise ValueError(
                f"T={t} is not divisible by patch={self.patch}; truncating here "
                "would shift every subsequent token and corrupt the AR alignment"
            )

        n_patch = t // self.patch
        xp = x.reshape(b, c, n_patch, self.patch)
        w = self.weight[type_id]          # (C, patch, d)
        bias = self.bias[type_id]         # (C, d)
        return torch.einsum("bctp,cpd->bctd", xp, w) + bias[None, :, None, :]
