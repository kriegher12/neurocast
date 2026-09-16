"""Prediction heads. One per bake-off arm, sharing a single trunk.

The heads are where the five objectives actually differ. Everything upstream --
tokenizer, sensor embedding, perceiver, backbone -- is held identical, which is
what makes the comparison a controlled experiment rather than five papers.

The flagship is :class:`GaussianMixtureHead`. It is the only head that yields a
**calibrated log-density in the canonical observation space**, and that single
property is what lets one artifact serve three roles:

* a pretraining objective (arm A-lik),
* the measurement instrument for the Neural Predictability Atlas, and
* the forward model ``p(brain | stimulus)`` of the noisy-channel decoder.

A VQ/cross-entropy head cannot do the latter two. Its "likelihood" lives in code
space, is bounded above by the codec's reconstruction error, and changes when
you retrain the codec -- so an Atlas built on it would measure the predictability
of a codebook, not of a brain.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GaussianMixtureHead",
    "LatentPredictionHead",
    "MaskedReconstructionHead",
    "PeripheralHead",
]


class GaussianMixtureHead(nn.Module):
    """Factorised mixture-of-Gaussians density over the next target block.

    Predicts each of ``target_dim`` coordinates with an independent ``K``-component
    mixture. Factorised across coordinates, which is an approximation *within* a
    patch -- but note the spatial-temporal factorisation across quadrants and
    timesteps remains exact, because the perceiver confines each token to one
    partition cell. If within-patch correlation turns out to matter, the upgrade
    is the within-patch autoregressive head (design head v2), not a change here.

    Returns log-density in **nats**, which
    :meth:`neurocast.data.canonical.AffineTransform.to_canonical_logprob` then
    pushes back to canonical space.
    """

    def __init__(self, d_model: int, target_dim: int, n_components: int = 5) -> None:
        super().__init__()
        self.target_dim = int(target_dim)
        self.k = int(n_components)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, target_dim * self.k * 3)
        # Start near a standard normal (zero mean, unit scale, uniform weights)
        # so the first steps are well-conditioned.
        #
        # The weight is small but NOT zero. A fully zero-initialised projection
        # makes d(log p)/d(h) identically zero, so the trunk receives no
        # gradient at all on the first step -- the head trains, the encoder does
        # not. Caught by validate_backbone.py, which checks that every arm's
        # gradient actually reaches a shared trunk.
        nn.init.normal_(self.proj.weight, std=1e-3)
        nn.init.zeros_(self.proj.bias)

    def distribution(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_weights, mu, log_sigma)``, each ``(..., target_dim, K)``."""
        raw = self.proj(self.norm(h))
        raw = raw.reshape(*h.shape[:-1], self.target_dim, self.k, 3)
        logit, mu, log_sigma = raw.unbind(-1)
        # Clamp keeps the density from collapsing onto a delta, which would send
        # log p to +inf and silently win the bake-off.
        log_sigma = log_sigma.clamp(-7.0, 7.0)
        return F.log_softmax(logit, dim=-1), mu, log_sigma

    def log_prob(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Log-density of ``target`` given context ``h``.

        Parameters
        ----------
        h
            ``(..., d_model)`` backbone output at the position *preceding* the
            target. The shift is the caller's responsibility -- see
            :func:`neurocast.objectives.ar_shift`.
        target
            ``(..., target_dim)``.

        Returns
        -------
        ``(...)`` total log-density summed over ``target_dim``.
        """
        if target.shape[-1] != self.target_dim:
            raise ValueError(
                f"target last dim {target.shape[-1]} != target_dim {self.target_dim}"
            )
        log_w, mu, log_sigma = self.distribution(h)
        t = target.unsqueeze(-1)
        z = (t - mu) * torch.exp(-log_sigma)
        comp = -0.5 * z**2 - log_sigma - 0.5 * math.log(2 * math.pi)
        return torch.logsumexp(log_w + comp, dim=-1).sum(-1)

    def forward(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood, mean over positions."""
        return -self.log_prob(h, target).mean()


class LatentPredictionHead(nn.Module):
    """Predict an EMA target encoder's embedding. Used by arms J and A-lat.

    Smooth-L1 rather than MSE: neural data has heavy-tailed artifacts, and MSE
    lets a handful of eyeblinks dominate the gradient.
    """

    def __init__(self, d_model: int, d_target: int | None = None, hidden: int = 2) -> None:
        super().__init__()
        d_target = d_target or d_model
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden * d_model),
            nn.GELU(),
            nn.Linear(hidden * d_model, d_target),
        )

    def forward(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # stop-gradient on the EMA target is essential; without it the pair
        # collapses to a constant in a few hundred steps.
        return F.smooth_l1_loss(self.net(h), target.detach())


class MaskedReconstructionHead(nn.Module):
    """Reconstruct masked patches. Arm M -- the control arm, matching MEG-XL.

    Optionally adds a log-magnitude-spectrum term. The MEG roadmap flags
    spectral reconstruction as entirely unexplored, so carrying it as
    ``alpha in {0, 0.1}`` closes a named gap for the cost of one extra run.
    """

    def __init__(self, d_model: int, target_dim: int, spectral_alpha: float = 0.0) -> None:
        super().__init__()
        self.alpha = float(spectral_alpha)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, target_dim),
        )

    def forward(
        self, h: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        pred = self.net(h)
        if mask is not None:
            if not bool(mask.any()):
                return pred.sum() * 0.0  # keep the graph alive, contribute nothing
            pred, target = pred[mask], target[mask]
        loss = F.mse_loss(pred, target)
        if self.alpha > 0:
            eps = 1e-8
            ps = torch.log(torch.abs(torch.fft.rfft(pred, dim=-1)) + eps)
            ts = torch.log(torch.abs(torch.fft.rfft(target, dim=-1)) + eps)
            loss = loss + self.alpha * F.l1_loss(ps, ts)
        return loss


class PeripheralHead(nn.Module):
    """Predict peripheral channels from brain activity. Arm P.

    Zero papers use peripheral signals as a self-supervised auxiliary
    pretraining objective -- they appear only as artifacts to remove or as
    late-fusion features. Two hypotheses this tests, which can both be true:

    * **H-aux** -- predicting body state is free supervision for exactly the
      nuisance the trunk would otherwise memorise per subject.
    * **H-adv** -- with :attr:`gradient_reversal` on, the trunk is pressured
      *not* to encode ocular/cardiac state, which should lower FMScope
      subject-variance.

    Sweeping ``eta`` traces the trade-off, and that curve is publishable
    whichever way it points.
    """

    def __init__(self, d_model: int, n_peripheral: int, gradient_reversal: float = 0.0) -> None:
        super().__init__()
        self.eta = float(gradient_reversal)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_peripheral),
        )

    def forward(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.eta > 0:
            h = _GradientReversal.apply(h, self.eta)
        return F.smooth_l1_loss(self.net(h), target)


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eta: float) -> torch.Tensor:
        ctx.eta = eta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.eta * grad, None
