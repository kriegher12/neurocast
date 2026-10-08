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

And three things that keep each arm's target out of its own input -- every one
of them a leak that an earlier version of this loop had:

* **M and J really mask.** Their trunk is bidirectional and the masked positions
  are replaced by a mask token before the backbone. Without that, "masked
  reconstruction" reconstructs a token the trunk has just been shown, and JEPA
  distils a teacher that saw the identical input.
* **P reads the brain, predicts the body.** ``h`` is taken at the last brain group
  of each patch and the target is that patch's peripheral channels, so the
  prediction is the AR factor ``p(periph_t | brain_t, x_<t)`` -- never a copy.
* **Empty quadrants are excluded.** A montage with no channels in some quadrant
  (MEG-only LibriBrain has no peripherals) yields a constant target there; a
  density head fits that to a delta and its NLL runs away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.synthetic import SyntheticCorpus
from ..models.neurocast import NeuroCast
from ..objectives.arms import OBJECTIVES, CollapseMonitor, EMATeacher, block_mask, build
from .flops import TIERS, ComputeLedger, measure_flops

__all__ = [
    "TrainConfig",
    "TrainResult",
    "group_targets",
    "valid_positions",
    "peripheral_targets",
    "arm_context",
    "build_arm",
    "train",
    "extract_representations",
]

#: Arms whose target comes from an EMA teacher rather than the observation.
_LATENT_ARMS = frozenset({"jepa", "ar_latent"})
#: Arms whose target lives in observation space.
_OBSERVATION_ARMS = frozenset({"masked", "ar_lik"})
#: Arms whose input is masked; their trunk is bidirectional.
_MASKED_ARMS = frozenset(n for n, cls in OBJECTIVES.items() if not cls.causal)


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


def valid_positions(quadrant: torch.Tensor, n_groups: int, n_patch: int) -> torch.Tensor:
    """``(T'*G,)`` bool: True where the position's quadrant has any channels."""
    present = torch.bincount(quadrant.long(), minlength=n_groups)[:n_groups] > 0
    return present.repeat(n_patch)


def peripheral_targets(
    x: torch.Tensor, quadrant: torch.Tensor, n_groups: int, patch: int
) -> torch.Tensor:
    """``(B, T', n_periph * patch)`` raw peripheral samples of each patch."""
    idx = torch.nonzero(quadrant == n_groups - 1).flatten()
    if idx.numel() == 0:
        raise ValueError(
            "the peripheral arm needs peripheral channels (EOG/ECG/...), and this "
            "montage has none -- see 'The peripheral-channel problem' in the README"
        )
    b, _, t = x.shape
    n_patch = t // patch
    xp = x[:, idx, : n_patch * patch].reshape(b, idx.numel(), n_patch, patch)
    return xp.permute(0, 2, 1, 3).reshape(b, n_patch, idx.numel() * patch)


def build_arm(arm: str, model: NeuroCast, target_dim: int):
    """Instantiate an arm's objective sized for ``model``."""
    n_periph = int((model.quadrant == model.n_groups - 1).sum())
    kwargs = {
        "masked": dict(d_model=model.d_model, target_dim=target_dim),
        "ar_lik": dict(d_model=model.d_model, target_dim=target_dim),
        "jepa": dict(d_model=model.d_model),
        "ar_latent": dict(d_model=model.d_model),
        # One output per peripheral sample in the patch.
        "peripheral": dict(d_model=model.d_model,
                           n_peripheral=max(n_periph, 1) * model.patch_samples),
    }[arm]
    return build(arm, **kwargs)


def arm_context(
    arm: str,
    model: NeuroCast,
    x: torch.Tensor,
    *,
    proj: torch.Tensor,
    teacher: EMATeacher | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Run the trunk and assemble everything ``arm``'s loss reads.

    Kept separate from the step so ``scripts/validate_backbone.py`` can check,
    by perturbation, that no arm's target reaches its own input.
    """
    b, _, t = x.shape
    patch, g = model.patch_samples, model.n_groups
    n_patch = t // patch
    n_pos = n_patch * g
    valid = valid_positions(model.quadrant, g, n_patch).to(x.device)
    valid = valid[None].expand(b, n_pos)

    mask = None
    if arm in _MASKED_ARMS:
        mask = block_mask((b, n_pos), block_len=12, target_frac=0.4,
                          generator=generator, device=x.device)
        h = model(x, mask=mask)
    else:
        h = model(x)

    ctx: dict[str, torch.Tensor] = {"h": h, "valid": valid}
    if mask is not None:
        ctx["mask"] = mask

    if arm in _OBSERVATION_ARMS:
        ctx["target"] = group_targets(x, model.quadrant, g, patch, proj)
    elif arm in _LATENT_ARMS:
        if teacher is None:
            raise ValueError(f"arm {arm!r} needs an EMA teacher")
        with torch.no_grad():
            ctx["target"] = teacher(x)          # the teacher always sees everything
    elif arm == "peripheral":
        # Last brain group of each patch: has seen brain_t, not periph_t.
        ctx["h"] = h[:, g - 2 :: g]
        ctx["peripheral_target"] = peripheral_targets(x, model.quadrant, g, patch)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return ctx


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

    model = NeuroCast(montage, rung=cfg.rung, window=256,
                      causal=OBJECTIVES[cfg.arm].causal, device=device).to(device)
    objective = build_arm(cfg.arm, model, cfg.target_dim).to(device)

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
        ctx = arm_context(cfg.arm, model, x, proj=proj, teacher=teacher)
        return objective.loss(ctx)["loss"]

    # Measure the true cost of one step, including the EMA teacher.
    opt.zero_grad(set_to_none=True)
    meas = measure_flops(lambda: one_step().backward(), label=cfg.arm)
    opt.zero_grad(set_to_none=True)

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
