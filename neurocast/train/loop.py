"""Training loop. The assembly that makes Phase 1 a loop over `objectives/`.

Holds the trunk, data, context length and compute budget fixed, and varies only
the objective -- which is the controlled comparison the MEG roadmap says is
missing.

Two things it enforces that a normal loop would not:

* **The compute ledger is the stopping condition**, not a step count. Arms
  differ in cost per step (the EMA teacher alone is +33%), so a fixed step count
  hands some arms more compute. Steps are whatever the measured budget affords.
* **Collapse monitors abort latent arms.** A JEPA arm that quietly collapses
  becomes a strawman, and then H1 is unfalsifiable in one direction. Collapse is
  made loud and recorded as a result, not hidden.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.synthetic import SyntheticCorpus
from ..models.neurocast import NeuroCast
from ..objectives.arms import CollapseMonitor, EMATeacher, build
from .flops import ComputeLedger, measure_flops

__all__ = ["TrainConfig", "TrainResult", "group_targets", "train"]

#: Arms whose target comes from an EMA teacher rather than the observation.
_LATENT_ARMS = frozenset({"jepa", "ar_latent"})
#: Arms whose target lives in observation space.
_OBSERVATION_ARMS = frozenset({"masked", "ar_lik"})


@dataclass
class TrainConfig:
    arm: str
    rung: str = "6m"
    tier: str = "t0"
    budget: float | None = None      # overrides the tier, for quick runs
    lr: float = 1e-3
    batch_size: int = 4
    n_patch: int = 24
    target_dim: int = 32
    seed: int = 0
    max_steps: int = 10_000
    log_every: int = 25
    device: str = "cpu"


@dataclass
class TrainResult:
    arm: str
    steps: int
    losses: list[float] = field(default_factory=list)
    ledger: ComputeLedger | None = None
    collapsed: bool = False
    flops_per_step: int = 0

    @property
    def final_loss(self) -> float:
        tail = self.losses[-10:]
        return float(np.mean(tail)) if tail else float("nan")

    @property
    def initial_loss(self) -> float:
        head = self.losses[:10]
        return float(np.mean(head)) if head else float("nan")

    @property
    def improvement(self) -> float:
        return self.initial_loss - self.final_loss


def group_targets(
    x: torch.Tensor,
    quadrant: torch.Tensor,
    n_groups: int,
    patch: int,
    proj: torch.Tensor,
) -> torch.Tensor:
    """Per-(patch, group) observation-space targets.

    Averages each quadrant's channels within a patch, then applies a **frozen
    random projection** to reach ``target_dim``. The projection is fixed and
    shared across arms, so every arm predicts an identical target -- otherwise
    the losses are not comparable and the bake-off is not a comparison.

    Returns ``(B, T'*G, target_dim)``, aligned with ``ar_flatten`` ordering.
    """
    b, c, t = x.shape
    n_patch = t // patch
    xp = x[:, :, : n_patch * patch].reshape(b, c, n_patch, patch)

    per_group = []
    for g in range(n_groups):
        m = quadrant == g
        if bool(m.any()):
            per_group.append(xp[:, m].mean(dim=1))       # (B, T', P)
        else:
            per_group.append(torch.zeros(b, n_patch, patch, device=x.device))
    stacked = torch.stack(per_group, dim=2)              # (B, T', G, P)
    flat = stacked.reshape(b, n_patch * n_groups, patch)
    return flat @ proj


def train(
    cfg: TrainConfig,
    corpus: SyntheticCorpus,
    montage,
    *,
    verbose: bool = True,
) -> tuple[NeuroCast, TrainResult]:
    """Train one arm under a measured compute budget."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)

    model = NeuroCast(montage, rung=cfg.rung, window=256, device=device).to(device)
    n_periph = int((model.quadrant == 4).sum())

    head_kwargs = {
        "masked": dict(d_model=model.d_model, target_dim=cfg.target_dim),
        "ar_lik": dict(d_model=model.d_model, target_dim=cfg.target_dim),
        "jepa": dict(d_model=model.d_model),
        "ar_latent": dict(d_model=model.d_model),
        "peripheral": dict(d_model=model.d_model, n_peripheral=max(n_periph, 1)),
    }[cfg.arm]
    objective = build(cfg.arm, **head_kwargs).to(device)

    teacher = EMATeacher(model) if cfg.arm in _LATENT_ARMS else None
    monitor = CollapseMonitor() if cfg.arm in _LATENT_ARMS else None

    params = list(model.parameters()) + list(objective.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01)

    proj = torch.randn(model.patch_samples, cfg.target_dim, device=device)
    proj /= model.patch_samples**0.5

    n_times = cfg.n_patch * model.patch_samples

    def make_batch():
        b = corpus.batch(cfg.batch_size, n_times, rng)
        return torch.from_numpy(b.x).float().to(device), b

    def one_step() -> torch.Tensor:
        x, _ = make_batch()
        h = model(x)
        if cfg.arm in _OBSERVATION_ARMS:
            target = group_targets(x, model.quadrant, model.n_groups,
                                   model.patch_samples, proj)
        elif cfg.arm in _LATENT_ARMS:
            with torch.no_grad():
                target = teacher(x)
        else:  # peripheral
            full = group_targets(x, model.quadrant, model.n_groups,
                                 model.patch_samples, proj)
            target = full[..., : max(n_periph, 1)]

        ctx = {"h": h, "target": target, "peripheral_target": target}
        if cfg.arm == "masked":
            from ..objectives.arms import block_mask

            ctx["mask"] = block_mask(
                (h.shape[0], h.shape[1]), block_len=12, target_frac=0.4,
                device=device,
            )
        return objective.loss(ctx)["loss"]

    # Measure the true cost of one step, including the EMA teacher.
    opt.zero_grad(set_to_none=True)
    meas = measure_flops(lambda: one_step().backward(), label=cfg.arm)
    opt.zero_grad(set_to_none=True)

    from .flops import TIERS

    budget = cfg.budget if cfg.budget is not None else TIERS[cfg.tier]
    ledger = ComputeLedger(cfg.arm, cfg.tier, budget)
    affordable = min(ledger.steps_affordable(meas.total), cfg.max_steps)

    if verbose:
        print(f"  {cfg.arm:<11} {meas.total / 1e9:8.2f} GFLOP/step  "
              f"-> {affordable:,} steps at budget {budget:.1e}")

    result = TrainResult(arm=cfg.arm, steps=0, ledger=ledger,
                         flops_per_step=meas.total)

    for step in range(affordable):
        opt.zero_grad(set_to_none=True)
        loss = one_step()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        ledger.spend_train(meas.total, 1)
        result.losses.append(float(loss.detach()))
        result.steps += 1

        if teacher is not None:
            teacher.update(model, step, affordable)
        if monitor is not None and step % 20 == 0:
            with torch.no_grad():
                x, _ = make_batch()
                status = monitor.check(model(x))
            if status["abort"]:
                result.collapsed = True
                if verbose:
                    print(f"  {cfg.arm}: ABORTED -- representation collapsed")
                break

    return model, result


@torch.no_grad()
def extract_representations(
    model: NeuroCast,
    corpus: SyntheticCorpus,
    *,
    n: int = 240,
    n_patch: int = 24,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frozen pooled representations, for FMScope and label probes.

    Mirrors the Identity Trap protocol: the encoder is frozen and pooled, not
    fine-tuned. Fine-tuned features would measure something else entirely --
    and, per that paper, would leak considerably more.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    n_times = n_patch * model.patch_samples
    zs, subs, labs = [], [], []
    for _ in range(0, n, 16):
        b = corpus.batch(16, n_times, rng)
        x = torch.from_numpy(b.x).float()
        zs.append(model.pooled(x).cpu().numpy())
        subs.append(b.subject)
        labs.append(b.label)
    return np.concatenate(zs), np.concatenate(subs), np.concatenate(labs)
