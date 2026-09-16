"""Autoregressive rollout: the model dreaming a brain forward.

Generates future sensor activity by feeding the model's own predictions back as
input, one spatial group at a time, in exactly the order the likelihood
factorises::

    for each future patch t:
        for each quadrant g = 0..4:
            read the backbone output at position (t, g) - 1
            take the predictive mean for quadrant g's target
            invert the quadrant PCA -> per-sensor patch
            write it into the signal, so later groups and patches can see it

Filling quadrant ``g`` of the new patch before predicting ``g + 1`` is only valid
because of **group confinement**: the perceiver makes each group's token a
function of that group's channels alone, so the still-empty channels of later
groups cannot contaminate earlier predictions. ``scripts/validate_tokenizer.py``
verifies that property to machine precision, and this module depends on it.

The rollout is a *mean* forecast. Neural activity is stochastic, and a mean
forecast smooths toward the predictable part -- which is the honest thing to
render next to the truth, and the reason the Atlas exists to say how far ahead
that predictable part extends.
"""

from __future__ import annotations

import numpy as np
import torch

from ..train.targets import QuadrantBasis

__all__ = ["mixture_mean", "rollout"]


def mixture_mean(head, h: torch.Tensor) -> torch.Tensor:
    """Predictive mean of a :class:`GaussianMixtureHead` at ``h``: ``sum_k pi_k mu_k``."""
    log_w, mu, _ = head.distribution(h)
    return (log_w.exp() * mu).sum(-1)


@torch.no_grad()
def rollout(
    model,
    head,
    basis: QuadrantBasis,
    context: np.ndarray,
    n_patches: int,
    *,
    prompt: torch.Tensor | None = None,
) -> np.ndarray:
    """Forecast ``n_patches`` patches past the end of ``context``.

    Parameters
    ----------
    context
        ``(C, T_ctx)`` observed signal; ``T_ctx`` must be a multiple of the patch.

    Returns
    -------
    ``(C, n_patches * patch)`` forecast, sensor by sensor.
    """
    model.eval()
    patch, groups = basis.patch, basis.n_groups
    c, t_ctx = context.shape
    if t_ctx % patch:
        raise ValueError(f"context length {t_ctx} must be a multiple of patch {patch}")
    ctx_patches = t_ctx // patch
    total = ctx_patches + n_patches
    signal = np.zeros((c, total * patch))
    signal[:, :t_ctx] = context

    for k in range(n_patches):
        t = ctx_patches + k
        for g in range(groups):
            idx = basis.channels[g]
            if idx.size == 0:
                continue
            # Only patches <= t are needed; causality makes later ones irrelevant.
            x = torch.as_tensor(signal[None, :, : (t + 1) * patch], dtype=torch.float32)
            h = model(x, prompt=prompt)
            pos = t * groups + g - 1           # output here predicts (t, g)
            z = mixture_mean(head, h[:, pos]).squeeze(0).numpy()
            signal[idx, t * patch:(t + 1) * patch] = basis.decode_group(g, z)

    return signal[:, t_ctx:]
