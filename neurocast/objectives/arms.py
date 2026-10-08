"""The five bake-off arms.

Phase 1 is a loop over this registry with the trunk, tokenizer, data, context
length and compute budget all held fixed. Only the objective varies. That is the
comparison the Sept-2026 MEG roadmap says is missing: *"the decisive comparison
at matched data, compute and architecture is still missing."*

    M        masked signal reconstruction     -- the control arm, MEG-XL's objective
    J        masked latent prediction (JEPA)  -- "completely unexplored" for MEG
    A-lat    causal latent forecasting        -- JEPA, but causal
    A-lik    causal forecasting + density     -- flagship; the only calibrated likelihood
    P        peripheral auxiliary             -- zero prior work as a pretraining objective

Each arm reports, at minimum, downstream transfer **and** FMScope
subject-identity leakage. A results table without the leakage column is not
reportable under this design -- the whole H2 claim lives in that column.

Context keys an arm may read
----------------------------
``h``                  ``(B, N, d)`` trunk output
``target``             ``(B, N, k)`` observation or EMA-teacher target
``mask``               ``(B, N)`` positions hidden from the trunk's input (M, J)
``valid``              ``(B, N)`` positions with a real target; a quadrant with
                       no channels yields a constant target that a density head
                       would fit to a delta and "win" on
``peripheral_target``  ``(B, N_p, k_p)`` for arm P, aligned with ``h`` by the loop
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from ..models.heads import (
    GaussianMixtureHead,
    LatentPredictionHead,
    MaskedReconstructionHead,
    PeripheralHead,
)

__all__ = [
    "ar_shift",
    "block_mask",
    "EMATeacher",
    "CollapseMonitor",
    "Objective",
    "OBJECTIVES",
    "build",
]


def ar_shift(h: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align backbone outputs with next-position targets.

    The single highest-risk off-by-one in the project. The backbone output at
    position ``i`` has seen positions ``<= i``; it must predict position
    ``i + 1``. Getting this backwards means the model predicts what it has
    already observed, which produces spectacular forecasting numbers that are
    entirely an artifact.

    Returns ``(h[:, :-1], target[:, 1:])``.
    """
    if h.shape[1] != target.shape[1]:
        raise ValueError(
            f"sequence length mismatch: h has {h.shape[1]}, target has {target.shape[1]}"
        )
    if h.shape[1] < 2:
        raise ValueError("need at least 2 positions to form an AR pair")
    return h[:, :-1], target[:, 1:]


def block_mask(
    shape: tuple[int, int],
    *,
    block_len: int,
    target_frac: float,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Contiguous-block mask, matching MEG-XL's recipe (3 s blocks to 40%).

    Contiguous rather than scattered: with i.i.d. per-token masking the model
    interpolates from immediate neighbours, which is a trivial task on a signal
    this autocorrelated and teaches nothing.
    """
    b, n = shape
    if not 0.0 < target_frac < 1.0:
        raise ValueError(f"target_frac must be in (0, 1), got {target_frac}")
    block_len = max(1, min(int(block_len), n))
    mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    need = int(round(target_frac * n))
    for i in range(b):
        guard = 0
        while int(mask[i].sum()) < need and guard < 10 * n:
            start = int(torch.randint(0, n, (1,), generator=generator).item())
            mask[i, start : start + block_len] = True
            guard += block_len
    return mask


class EMATeacher(nn.Module):
    """Exponential-moving-average copy of an encoder, for arms J and A-lat.

    The momentum schedule (0.996 -> 0.9999, cosine) is the I-JEPA/V-JEPA recipe.
    A slow teacher is what stops the student from tracking it exactly and
    collapsing to a constant.
    """

    def __init__(self, module: nn.Module, base: float = 0.996, final: float = 0.9999) -> None:
        super().__init__()
        self.teacher = copy.deepcopy(module)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.base, self.final = float(base), float(final)

    def momentum(self, step: int, total_steps: int) -> float:
        import math

        frac = min(max(step / max(total_steps, 1), 0.0), 1.0)
        return self.final - (self.final - self.base) * (math.cos(math.pi * frac) + 1) / 2

    @torch.no_grad()
    def update(self, student: nn.Module, step: int, total_steps: int) -> float:
        m = self.momentum(step, total_steps)
        for tp, sp in zip(self.teacher.parameters(), student.parameters()):
            tp.mul_(m).add_(sp.detach(), alpha=1.0 - m)
        for tb, sb in zip(self.teacher.buffers(), student.buffers()):
            tb.copy_(sb)
        return m

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.teacher(*args, **kwargs)


@dataclass
class CollapseMonitor:
    """Hard abort conditions for latent-prediction arms.

    A JEPA arm that quietly collapses is worse than one that crashes: it becomes
    a strawman, and then H1 is unfalsifiable in one direction. These monitors
    make collapse loud.

    The constant-predictor control is the one that is usually missing and the
    one that actually matters -- it measures directly whether the objective is
    doing anything at all beyond predicting the batch mean.
    """

    min_stable_rank_frac: float = 0.30
    min_std: float = 0.05
    max_offdiag_corr: float = 0.30
    patience: int = 500
    _strikes: int = field(default=0, repr=False)

    @torch.no_grad()
    def check(self, z: torch.Tensor) -> dict[str, float | bool]:
        z = z.reshape(-1, z.shape[-1]).float()
        d = z.shape[-1]
        zc = z - z.mean(0, keepdim=True)
        cov = (zc.T @ zc) / max(zc.shape[0] - 1, 1)

        sv = torch.linalg.svdvals(cov)
        stable_rank = float(sv.sum() / sv[0].clamp_min(1e-12))
        std = zc.std(0)
        corr = cov / (std[:, None] * std[None, :] + 1e-12)
        offdiag = float(
            (corr - torch.diag(torch.diag(corr))).abs().sum() / max(d * (d - 1), 1)
        )
        min_std = float(std.min())

        bad = (
            stable_rank < self.min_stable_rank_frac * d
            or min_std < self.min_std
            or offdiag > self.max_offdiag_corr
        )
        self._strikes = self._strikes + 1 if bad else 0
        return {
            "stable_rank": stable_rank,
            "stable_rank_frac": stable_rank / d,
            "min_std": min_std,
            "offdiag_corr": offdiag,
            "collapsing": bad,
            "abort": self._strikes >= self.patience,
        }


class Objective(nn.Module):
    """Base class. Subclasses own a head and a loss; the trunk is shared."""

    name: str = "base"
    causal: bool = True

    def loss(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.loss(ctx)


class MaskedReconstruction(Objective):
    """Reconstruct the hidden positions -- and, optionally, the visible ones too.

    ``visible_weight`` adds the reconstruction error at *visible* positions. The
    lost version of this project found the masked arm collapsed on real MEG --
    its output ignored its input -- because hidden positions are hard enough to
    predict that the mean is a strong local optimum, and nothing ties the trunk's
    output to what it is shown. Scoring visible positions (weight 1.0) forces the
    output to carry the input, and fixed it; lower learning rates and other
    targets did not. Default 0.0 keeps the plain MEG-XL-style objective.
    """

    name, causal = "masked", False

    def __init__(self, d_model: int, target_dim: int, spectral_alpha: float = 0.0,
                 visible_weight: float = 0.0):
        super().__init__()
        self.head = MaskedReconstructionHead(d_model, target_dim, spectral_alpha)
        self.visible_weight = float(visible_weight)

    def loss(self, ctx):
        mask, valid = ctx.get("mask"), ctx.get("valid")
        hidden = mask
        if mask is not None and valid is not None:
            hidden = mask & valid
        elif valid is not None:
            hidden = valid
        loss = self.head(ctx["h"], ctx["target"], hidden)
        out = {"recon": loss.detach()}
        if self.visible_weight > 0 and mask is not None:
            shown = ~mask if valid is None else (~mask & valid)
            vis = self.head(ctx["h"], ctx["target"], shown)
            out["visible"] = vis.detach()
            loss = loss + self.visible_weight * vis
        out["loss"] = loss
        return out


class JEPA(Objective):
    """Masked latent prediction. Scored on masked positions only.

    The student sees a masked input (the loop replaces those positions with the
    mask token); the EMA teacher sees the full input. Predicting the teacher at
    an *unmasked* position would be self-distillation of an identical view,
    which teaches nothing and invites collapse.
    """

    name, causal = "jepa", False

    def __init__(self, d_model: int, d_target: int | None = None, sigreg: float = 0.05):
        super().__init__()
        self.head = LatentPredictionHead(d_model, d_target)
        self.sigreg = float(sigreg)

    def loss(self, ctx):
        h, target, mask = ctx["h"], ctx["target"], ctx.get("mask")
        if mask is not None:
            if not bool(mask.any()):
                pred = h.sum() * 0.0
            else:
                pred = self.head(h[mask], target[mask])
        else:
            pred = self.head(h, target)
        out = {"latent": pred.detach()}
        total = pred
        if self.sigreg > 0:
            reg = _sigreg(ctx["h"])
            total = total + self.sigreg * reg
            out["sigreg"] = reg.detach()
        out["loss"] = total
        return out


class ARLatent(Objective):
    """Causal latent forecasting -- JEPA, but the mask is causal."""

    name, causal = "ar_latent", True

    def __init__(self, d_model: int, d_target: int | None = None, sigreg: float = 0.05):
        super().__init__()
        self.head = LatentPredictionHead(d_model, d_target)
        self.sigreg = float(sigreg)

    def loss(self, ctx):
        h, target = ar_shift(ctx["h"], ctx["target"])
        pred = self.head(h, target)
        out = {"latent": pred.detach()}
        total = pred
        if self.sigreg > 0:
            reg = _sigreg(h)
            total = total + self.sigreg * reg
            out["sigreg"] = reg.detach()
        out["loss"] = total
        return out


class ARLikelihood(Objective):
    """Causal forecasting with an explicit density. The flagship arm."""

    name, causal = "ar_lik", True

    def __init__(self, d_model: int, target_dim: int, n_components: int = 5):
        super().__init__()
        self.head = GaussianMixtureHead(d_model, target_dim, n_components)

    def loss(self, ctx):
        h, target = ar_shift(ctx["h"], ctx["target"])
        valid = ctx.get("valid")
        if valid is not None:
            v = valid[:, 1:]           # validity of the *target* position
            h, target = h[v], target[v]
        logp = self.head.log_prob(h, target)
        nll = -logp.mean()
        return {"loss": nll, "nll": nll.detach(), "logp_total": logp.sum().detach()}


class Peripheral(Objective):
    """Predict the peripheral channels of patch ``t`` from the brain.

    The loop hands this arm ``h`` taken at the *last brain group* of each patch
    -- position ``(t, G-2)`` -- which has seen every brain quadrant at ``t`` and
    everything before ``t``, but not the peripherals at ``t``. That is exactly the
    AR factor ``p(x_t^periph | x_t^brain, x_<t)``. Reading ``h`` at the peripheral
    position itself would let the trunk copy its own input.
    """

    name, causal = "peripheral", True

    def __init__(self, d_model: int, n_peripheral: int, gradient_reversal: float = 0.0):
        super().__init__()
        self.head = PeripheralHead(d_model, n_peripheral, gradient_reversal)

    def loss(self, ctx):
        loss = self.head(ctx["h"], ctx["peripheral_target"])
        return {"loss": loss, "periph": loss.detach()}


def _sigreg(z: torch.Tensor, n_slices: int = 128) -> torch.Tensor:
    """Sketched isotropic-Gaussian regularisation (LeJEPA-style).

    Sliced-Wasserstein distance between the batch of embeddings and an isotropic
    Gaussian reference, over random 1-D projections resampled every call.
    Provides collapse resistance without relying solely on the EMA teacher,
    which is fragile on signals this autocorrelated.
    """
    z = z.reshape(-1, z.shape[-1])
    n, d = z.shape
    dirs = torch.randn(d, n_slices, device=z.device, dtype=z.dtype)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-12)

    zc = (z - z.mean(0, keepdim=True)) / z.std(0, keepdim=True).clamp_min(1e-6)
    proj = (zc @ dirs).T                                  # (n_slices, n)
    ref = torch.randn(n_slices, n, device=z.device, dtype=z.dtype)
    return (proj.sort(dim=1).values - ref.sort(dim=1).values).pow(2).mean()


#: Arm registry. Phase 1 iterates this with everything else frozen.
OBJECTIVES: dict[str, Callable[..., Objective]] = {
    "masked": MaskedReconstruction,
    "jepa": JEPA,
    "ar_latent": ARLatent,
    "ar_lik": ARLikelihood,
    "peripheral": Peripheral,
}


def build(name: str, **kwargs) -> Objective:
    if name not in OBJECTIVES:
        raise ValueError(f"unknown objective {name!r}; choose from {sorted(OBJECTIVES)}")
    return OBJECTIVES[name](**kwargs)
