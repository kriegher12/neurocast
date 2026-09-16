"""Train the flagship generative arm (A-lik) on quadrant-PCA targets.

The bake-off loop in :mod:`neurocast.train.loop` compares objectives on a lossy
target. This trainer is the one used when the model has to *do* something with
its likelihood -- roll the brain forward for the demo, supply representations
for cross-array transfer, or feed the Atlas. It predicts the design's actual
target (whitened per-quadrant PCA of sensor patches), which is invertible back to
per-sensor signal.

Defensive mixing is fitted from target statistics before training, so a single
artifact-like sample cannot dominate the loss (see
:func:`neurocast.atlas.estimators.defensive_mixture` for how that was found).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..models.heads import GaussianMixtureHead
from ..models.neurocast import NeuroCast
from ..objectives.arms import ar_shift
from .targets import QuadrantBasis

__all__ = ["GenerativeModel", "train_generative"]


@dataclass
class GenerativeModel:
    model: NeuroCast
    head: GaussianMixtureHead
    basis: QuadrantBasis
    losses: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def initial_loss(self) -> float:
        return float(np.mean(self.losses[:10])) if self.losses else float("nan")

    @property
    def final_loss(self) -> float:
        return float(np.mean(self.losses[-10:])) if self.losses else float("nan")

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "head": self.head.state_dict()}


def train_generative(
    corpus,
    montage,
    *,
    rung: str = "tiny",
    n_patch: int = 32,
    batch: int = 4,
    steps: int = 300,
    rank: int = 16,
    lr: float = 2e-3,
    subjects=None,
    seed: int = 0,
    verbose: bool = True,
) -> GenerativeModel:
    """Train NeuroCast + mixture head by next-group likelihood.

    ``corpus`` must provide ``batch(montage, n, n_times, rng, subjects=...)`` --
    :class:`neurocast.data.sources.SourceSpaceCorpus` does.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NeuroCast(montage, rung=rung, window=256)
    patch = model.patch_samples
    n_times = n_patch * patch

    calib = corpus.batch(montage, 32, n_times, rng, subjects=subjects).x
    basis = QuadrantBasis(rank=rank, patch=patch, n_groups=model.n_groups)
    basis.fit(calib, montage.quadrants)

    head = GaussianMixtureHead(model.d_model, rank, n_components=4)
    targets = basis.encode(calib)
    head.set_background(targets.reshape(-1, rank).mean(0),
                        targets.reshape(-1, rank).std(0).clamp_min(1e-3))

    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))

    out = GenerativeModel(model=model, head=head, basis=basis)
    t0 = time.time()
    model.train()
    for step in range(steps):
        x = corpus.batch(montage, batch, n_times, rng, subjects=subjects).x
        xt = torch.as_tensor(x, dtype=torch.float32)
        h, tgt = ar_shift(model(xt), basis.encode(x))
        loss = -head.log_prob(h, tgt).mean() / rank
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        out.losses.append(float(loss.detach()))
        if verbose and (step % max(steps // 6, 1) == 0 or step == steps - 1):
            print(f"    step {step:>4}/{steps}  NLL {out.losses[-1]:+.4f} nats/component")
    out.seconds = time.time() - t0
    model.eval()
    return out
