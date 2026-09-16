"""Measure real compute, and quantify how wrong the 6ND estimate is.

H1 is a claim about behaviour *at matched compute*. If "matched" is computed
with a formula that misses a large, arm-dependent fraction of the real cost,
then the comparison is not controlled and the result is an artifact of the
accounting.

This script measures an actual training step -- front-end, backbone, head, loss,
backward -- and reports:

  1. measured FLOPs vs. the conventional 6ND estimate
  2. how much of the real cost lives in the SensorPerceiver front-end,
     whose cost scales with *channel count* and does not appear in 6ND at all
  3. the EMA teacher overhead that arms J and A-lat pay and arms M and A-lik
     do not
  4. that the ledger charges hyperparameter search to the arm that spent it
  5. that mismatched arms are detected before any comparison table is written

Run:  .venv/Scripts/python.exe scripts/validate_flops.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocast.models.backbone import Backbone, BackboneConfig  # noqa: E402
from neurocast.objectives.arms import build  # noqa: E402
from neurocast.tokenizer.embedding import SensorEmbedding  # noqa: E402
from neurocast.tokenizer.patch import PATCH_SAMPLES, TypedPatchEmbed  # noqa: E402
from neurocast.tokenizer.perceiver import SensorPerceiver, ar_flatten  # noqa: E402
from neurocast.train.flops import (  # noqa: E402
    TIERS,
    ComputeLedger,
    chinchilla_optimal_params,
    compare_arms,
    estimate_6nd,
    measure_flops,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_tokenizer import megin_montage  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


class FullModel(nn.Module):
    """Front-end + backbone, as one step would actually run it."""

    def __init__(self, montage, rung: str = "6m"):
        super().__init__()
        cfg = BackboneConfig(rung, window=64, global_every=5, prompt_every=6)
        d = cfg.d_model
        self.sensor = SensorEmbedding(d)
        self.patch = TypedPatchEmbed(d)
        self.perceiver = SensorPerceiver(d)
        self.backbone = Backbone(cfg)
        self.d_model = d
        self.register_buffer("type_id", torch.from_numpy(montage.type_ids))
        self.register_buffer("quadrant", torch.from_numpy(montage.quadrants))
        self._mt = SensorEmbedding.montage_tensors(montage)

    def front_end(self, x: torch.Tensor) -> torch.Tensor:
        s = self.sensor(self._mt)
        tok = self.patch(x, self.type_id) + s[None, :, None, :]
        return ar_flatten(self.perceiver(tok, self.quadrant))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.front_end(x))


def main() -> int:
    torch.manual_seed(0)
    ok = True

    montage = megin_montage()
    model = FullModel(montage)
    n_params = sum(p.numel() for p in model.parameters())

    b, n_patch = 1, 16
    t_len = n_patch * PATCH_SAMPLES
    n_groups = 5
    n_positions = n_patch * n_groups
    x = torch.randn(b, len(montage), t_len)

    head = build("ar_lik", d_model=model.d_model, target_dim=64)

    print("=" * 78)
    print("Setup")
    print("=" * 78)
    print(f"  montage          {len(montage)} channels")
    print(f"  window           {t_len} samples = {t_len / 250:.2f} s")
    print(f"  sequence         {n_patch} patches x {n_groups} groups = {n_positions} positions")
    print(f"  model            {n_params:,} parameters")

    def step() -> None:
        h = model(x)
        target = torch.randn(b, n_positions, 64)
        loss = head.loss({"h": h, "target": target})["loss"]
        loss.backward()

    print()
    print("=" * 78)
    print("1. Measured FLOPs vs. the 6ND estimate")
    print("=" * 78)

    m_full = measure_flops(step, label="full step")
    tokens = b * n_positions
    est = estimate_6nd(n_params, tokens)
    ratio = m_full.total / est

    print(f"  measured (fwd+bwd)   {m_full.total / 1e9:12.3f} GFLOP")
    print(f"  6ND estimate         {est / 1e9:12.3f} GFLOP")
    print(f"  measured / 6ND       {ratio:12.2f}x")
    print()
    print("  where the measured cost actually goes:")
    print(m_full.breakdown(top=6))

    ok &= check(
        "measured and estimated compute differ materially",
        ratio > 1.2 or ratio < 0.8,
        f"6ND is off by {abs(ratio - 1) * 100:.0f}% here",
    )

    print()
    print("=" * 78)
    print("2. The front-end is invisible to 6ND")
    print("=" * 78)

    model.zero_grad(set_to_none=True)

    def front_only() -> None:
        model.front_end(x).sum().backward()

    m_front = measure_flops(front_only, label="front-end")
    frac = m_front.fraction_of(m_full)
    print(f"  front-end            {m_front.total / 1e9:12.3f} GFLOP "
          f"({100 * frac:.1f}% of the full step)")
    print("  The SensorPerceiver cost scales with CHANNEL COUNT, which does not")
    print("  appear in 6ND at all. A 306-channel array and a 64-channel cap have")
    print("  identical 6ND and very different real cost.")
    ok &= check("front-end accounts for a non-trivial share", frac > 0.05,
                f"{100 * frac:.1f}%")

    print()
    print("=" * 78)
    print("3. EMA teacher overhead (arms J and A-lat pay it; M and A-lik do not)")
    print("=" * 78)

    model.zero_grad(set_to_none=True)
    import copy

    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    jepa = build("jepa", d_model=model.d_model)

    def step_with_teacher() -> None:
        h = model(x)
        with torch.no_grad():
            tgt = teacher(x)
        jepa.loss({"h": h, "target": tgt})["loss"].backward()

    m_ema = measure_flops(step_with_teacher, label="jepa step")
    model.zero_grad(set_to_none=True)
    jepa_no_teacher = build("jepa", d_model=model.d_model)

    def step_no_teacher() -> None:
        h = model(x)
        jepa_no_teacher.loss({"h": h, "target": torch.randn_like(h)})["loss"].backward()

    m_noema = measure_flops(step_no_teacher, label="jepa step, no teacher")
    overhead = m_ema.total / m_noema.total - 1.0
    print(f"  with EMA teacher     {m_ema.total / 1e9:12.3f} GFLOP")
    print(f"  without              {m_noema.total / 1e9:12.3f} GFLOP")
    print(f"  overhead             {100 * overhead:12.1f}%")
    print("  Excluding this -- which is standard -- silently hands the latent arms")
    print("  extra compute at a nominally matched budget.")
    ok &= check("EMA teacher cost is material and measured", overhead > 0.10,
                f"{100 * overhead:.1f}%")

    print()
    print("=" * 78)
    print("4. Ledger: HPO is charged to the arm that spent it")
    print("=" * 78)

    per_step = m_full.total
    tier, budget = "t0", TIERS["t0"]
    tuned = ComputeLedger("ar_lik", tier, budget)
    untuned = ComputeLedger("masked", tier, budget)

    hpo_steps = int(0.03 * budget / per_step)
    tuned.spend_hpo(per_step, hpo_steps, n_trials=16)
    untuned.spend_hpo(per_step, hpo_steps, n_trials=2)

    for lg in (tuned, untuned):
        lg.spend_train(per_step, lg.steps_affordable(per_step))

    print("  " + tuned.summary())
    print("  " + untuned.summary())
    print()
    print(f"  the 16-trial arm affords {untuned.steps - tuned.steps:,} fewer training")
    print("  steps than the 2-trial arm. That is the cost of tuning, made visible.")

    ok &= check("heavier search buys fewer training steps",
                tuned.steps < untuned.steps)
    ok &= check("both arms land within their budget",
                tuned.total <= budget and untuned.total <= budget)
    matched, report = compare_arms([tuned, untuned])
    ok &= check("arms are compute-matched after charging", matched)

    print()
    print("=" * 78)
    print("5. Mismatched arms are caught before any table is written")
    print("=" * 78)

    cheat = ComputeLedger("cheating", tier, budget)
    cheat.spend_train(per_step, int(1.5 * budget / per_step))
    bad, report = compare_arms([tuned, untuned, cheat])
    print("  " + report.replace("\n", "\n  "))
    ok &= check("a 50%-over arm is rejected", not bad)

    print()
    print("=" * 78)
    print("6. Corpus size caps the model, not the other way round")
    print("=" * 78)

    # 250 Hz / 16-sample patches = 15.625 patches/s; each patch emits one token
    # per spatial group. Do NOT multiply by the design note's "62.5 tok/s" --
    # that figure already folds in G=4.
    patches_per_s = 250.0 / PATCH_SAMPLES
    positions_per_s = patches_per_s * n_groups
    corpus_positions = int(600 * 3600 * positions_per_s)
    opt = chinchilla_optimal_params(corpus_positions, epochs=4)
    print(f"  token rate           {patches_per_s:.3f} patches/s x {n_groups} groups "
          f"= {positions_per_s:.2f} positions/s")
    print(f"  ~600 h corpus        {corpus_positions:,} sequence positions")
    print(f"  compute-optimal N    {opt / 1e6:.1f}M parameters at 4 epochs")
    print("  MEG-XL is 20M and is SOTA. That is not a coincidence, and it is why")
    print("  the rungs top out at 320M rather than 1B.")
    ok &= check("optimal size lands in the tens of millions",
                5e6 < opt < 3e8, f"{opt / 1e6:.0f}M")

    print()
    print("=" * 78)
    if ok:
        print("COMPUTE ACCOUNTING VALIDATED: measured, itemised, HPO charged.")
        return 0
    print("COMPUTE ACCOUNTING FAILED validation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
