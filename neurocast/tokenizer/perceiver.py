"""SensorPerceiver: variable channel sets -> one token per (timestep, quadrant).

This module carries the single most load-bearing architectural decision in the
project, so it is worth stating precisely what it buys.

The problem
-----------
Two things are in tension. We want (a) a model that ingests any montage --
306-channel MEGIN, 275-channel CTF, a 64-channel EEG cap, a depth electrode --
and (b) an **exact** autoregressive likelihood, because the likelihood is
simultaneously the pretraining objective, the Atlas measurement instrument, and
the forward model of the noisy-channel decoder. A bound will not do for any of
those three.

The usual Perceiver answer to (a) -- pool all channels into a fixed latent set
per timestep -- gives one token per timestep, which forces
``p(x_t | x_<t)`` to be modelled as a single joint over all 306 channels. That
is either a strong independence assumption or an intractable joint.

The resolution
--------------
Cross-attend to latents **per quadrant**, so each timestep produces ``G`` tokens,
one per spatial group, and each token summarises exactly that group's channels.
Ordering the flattened sequence as

    [g0(t0), g1(t0), ... g4(t0), g0(t1), g1(t1), ...]

with a standard causal mask gives an exact chain-rule factorisation over space
*and* time::

    p(x_1:T) = prod_t prod_g p(x_t^(g) | x_t^(<g), x_<t)

The groups are a genuine partition of the channels (enforced by
``descriptor.assert_partition``), so this is an identity, not an approximation.
``log p(x)`` is exact up to the frozen whitening Jacobian, which
:mod:`neurocast.data.canonical` accounts for.

Peripherals occupy the last group, so ``p(brain_t | ...)`` never conditions on
same-timestep EOG/ECG. Without that ordering, "the model uses eye movement"
would be built into the factorisation and the artifact-only control would be
meaningless.

Cost is ``O(T' * M * C)`` -- for 306 channels, 80 latents and 256 patches that is
~6.3M attention pairs per layer per second of data. Negligible next to the
backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .descriptor import Quadrant

__all__ = ["SensorPerceiver", "ar_flatten", "ar_unflatten", "causal_mask"]


class SensorPerceiver(nn.Module):
    """Pool a variable channel set into ``G`` tokens per timestep.

    Parameters
    ----------
    d_model
        Width of the input channel tokens and the output group tokens.
    latents_per_group
        Learned queries per quadrant. 16 by default.
    n_groups
        Spatial groups. 5 = four anatomical quadrants plus peripherals.
    """

    def __init__(
        self,
        d_model: int = 256,
        *,
        latents_per_group: int = 8,
        n_groups: int = Quadrant.n_all(),
        n_heads: int = 4,
        n_cross_layers: int = 1,
    ) -> None:
        """
        Defaults chosen from a measured ablation, not from taste.

        The front-end turned out to dominate the training step -- 81.2% of all
        FLOPs, stable across window lengths from 1 s to 16 s. Four of every five
        FLOPs were being spent pooling channels rather than modelling dynamics,
        which is a poor allocation for a project whose scientific claim is about
        temporal prediction.

        Measured cost of one perceiver step, 309 channels, d=384, 4.1 s window::

            16 latents, 2 layers   141.8 GFLOP   1.00x   5,940,096 params
             8 latents, 2 layers    94.2 GFLOP   0.66x   4,745,088
            16 latents, 1 layer     73.2 GFLOP   0.52x   4,165,632
             8 latents, 1 layer     48.2 GFLOP   0.34x   2,970,624   <- default
             4 latents, 1 layer     35.8 GFLOP   0.25x   2,373,120

        Dropping the second cross layer is the single biggest win and costs
        little: one layer already lets every latent see every channel in its
        quadrant, and the backbone provides the depth.

        This is a **provisional** default. Capacity here could plausibly affect
        the H1 outcome, so ``latents_per_group`` and ``n_cross_layers`` belong
        in the Phase-1 ablation alongside ``n_groups``, and the chosen point
        must be reported with the bake-off rather than assumed.
        """
        super().__init__()
        self.d_model = int(d_model)
        self.n_groups = int(n_groups)
        self.latents_per_group = int(latents_per_group)
        self.n_latents = self.n_groups * self.latents_per_group

        self.latents = nn.Parameter(torch.randn(self.n_latents, d_model) * 0.02)

        self.cross = nn.ModuleList(
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(n_cross_layers)
        )
        self.cross_norm = nn.ModuleList(
            nn.LayerNorm(d_model) for _ in range(n_cross_layers)
        )
        self.ffn = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
            for _ in range(n_cross_layers)
        )
        # Pool each group's latents into a single token.
        self.pool = nn.Sequential(
            nn.Linear(self.latents_per_group * d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def group_mask(self, quadrant: torch.Tensor) -> torch.Tensor:
        """``(n_latents, C)`` bool mask, True where attention is **blocked**.

        Latents belonging to group ``g`` may attend only to channels in group
        ``g``. This confinement is what makes each output token a function of
        exactly one partition cell -- and therefore what makes the
        factorisation exact rather than approximate.
        """
        groups = torch.arange(self.n_groups, device=quadrant.device)
        latent_group = groups.repeat_interleave(self.latents_per_group)  # (n_latents,)
        return latent_group[:, None] != quadrant[None, :]

    def forward(self, tokens: torch.Tensor, quadrant: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens
            ``(B, C, T', d)`` per-channel patch tokens.
        quadrant
            ``(C,)`` long tensor of quadrant ids in ``[0, n_groups)``.

        Returns
        -------
        ``(B, T', G, d)`` -- one token per timestep per spatial group.
        """
        if tokens.ndim != 4:
            raise ValueError(f"expected (B, C, T', d), got {tuple(tokens.shape)}")
        b, c, t, d = tokens.shape
        if d != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d}")
        if quadrant.shape != (c,):
            raise ValueError(f"quadrant must be ({c},), got {tuple(quadrant.shape)}")
        if int(quadrant.max()) >= self.n_groups or int(quadrant.min()) < 0:
            raise ValueError(f"quadrant ids must lie in [0, {self.n_groups})")

        # Fold time into batch: every timestep is pooled independently.
        kv = tokens.permute(0, 2, 1, 3).reshape(b * t, c, d)
        z = self.latents.unsqueeze(0).expand(b * t, -1, -1)
        mask = self.group_mask(quadrant)

        # A group with no channels in this montage would have every key masked,
        # producing NaN from a fully -inf softmax row. Detect and neutralise.
        empty = mask.all(dim=1)
        if bool(empty.any()):
            mask = mask.clone()
            mask[empty] = False

        for attn, norm, ff in zip(self.cross, self.cross_norm, self.ffn):
            q = norm(z)
            out, _ = attn(q, kv, kv, attn_mask=mask, need_weights=False)
            z = z + out
            z = z + ff(z)

        if bool(empty.any()):
            # Zero out groups that had no channels, so they contribute nothing
            # rather than contributing attention over unrelated sensors.
            z = z.masked_fill(empty[None, :, None], 0.0)

        z = z.reshape(b * t, self.n_groups, self.latents_per_group * d)
        g = self.pool(z)
        return g.reshape(b, t, self.n_groups, d)


def ar_flatten(group_tokens: torch.Tensor) -> torch.Tensor:
    """``(B, T', G, d)`` -> ``(B, T'*G, d)`` in autoregressive order.

    Ordering is ``[g0(t0) ... g4(t0), g0(t1) ... ]`` -- groups fastest, time
    slowest. Combined with a standard causal mask this yields exactly
    ``p(x_t^(g) | x_t^(<g), x_<t)``.
    """
    if group_tokens.ndim != 4:
        raise ValueError(f"expected (B, T', G, d), got {tuple(group_tokens.shape)}")
    b, t, g, d = group_tokens.shape
    return group_tokens.reshape(b, t * g, d)


def ar_unflatten(seq: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Inverse of :func:`ar_flatten`."""
    b, n, d = seq.shape
    if n % n_groups:
        raise ValueError(f"sequence length {n} is not divisible by {n_groups} groups")
    return seq.reshape(b, n // n_groups, n_groups, d)


def causal_mask(n: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """``(n, n)`` bool mask, True above the diagonal (i.e. blocked).

    Strictly upper triangular: position ``i`` sees ``j <= i``. The AR head then
    predicts position ``i+1`` from the output at ``i``, which is what makes the
    shift-by-one give the intended conditional.
    """
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
