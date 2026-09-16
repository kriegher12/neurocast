"""Measured compute accounting. Without this, the bake-off is not an experiment.

The problem with "matched compute"
----------------------------------
Almost every objective comparison in this literature reports compute as ``6ND``
-- six FLOPs per parameter per token -- and almost none audits whether that
estimate holds for the architecture actually being trained. For NeuroCast it
does not, and the gap is not a rounding error:

* The **SensorPerceiver front-end** cross-attends 80 latents over up to ~320
  channels at every timestep. Its cost scales with *channel count*, which does
  not appear in ``6ND`` at all.
* The **EMA teacher** in arms J and A-lat runs a full extra forward pass every
  step. Routinely excluded; worth roughly a third of the step.
* **Predictor and density heads** differ in size across arms by design -- the
  mixture head emits ``target_dim * K * 3`` values per position.

So an arm can look compute-matched on paper while receiving substantially more
real compute. Since H1 is precisely a claim about behaviour *at matched
compute*, that would not be a bug in a metric -- it would be the whole result.

What this module does
---------------------
Measures actual FLOPs with :class:`torch.utils.flop_counter.FlopCounterMode`
wrapped around the complete step (forward, loss, backward), and tracks a
per-arm ledger that also **charges hyperparameter search** to the arm that
consumed it. Charging tuning cost is what separates "objective X is better"
from "we tuned X harder", and essentially nobody does it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.utils.flop_counter import FlopCounterMode

__all__ = [
    "FlopMeasurement",
    "measure_flops",
    "estimate_6nd",
    "ComputeLedger",
    "TIERS",
    "chinchilla_optimal_params",
]

#: Measured-FLOP budget tiers for the bake-off, 4x apart.
TIERS: dict[str, float] = {
    "t0": 1e18,
    "t1": 4e18,
    "t2": 1.6e19,
    "t3": 6.4e19,
}


@dataclass(frozen=True)
class FlopMeasurement:
    """Result of one measured step."""

    total: int
    by_module: dict[str, int] = field(default_factory=dict)
    label: str = ""

    @property
    def tflops(self) -> float:
        return self.total / 1e12

    def fraction_of(self, other: "FlopMeasurement") -> float:
        return self.total / other.total if other.total else float("nan")

    def breakdown(self, top: int = 8) -> str:
        if not self.by_module:
            return "(no per-module breakdown)"
        items = sorted(self.by_module.items(), key=lambda kv: -kv[1])[:top]
        lines = [
            f"    {name:<44} {v / 1e9:10.3f} GFLOP  {100 * v / max(self.total, 1):5.1f}%"
            for name, v in items
        ]
        return "\n".join(lines)


def measure_flops(
    fn: Callable[[], object], *, label: str = "", depth: int = 2
) -> FlopMeasurement:
    """Measure FLOPs for everything ``fn`` does.

    Pass a closure covering the **entire** step -- front-end, backbone, head,
    loss, and ``backward()``. Measuring only the backbone forward is how the
    front-end and the EMA teacher go missing from published compute budgets.
    """
    with FlopCounterMode(display=False, depth=depth) as counter:
        fn()
    counts = counter.get_flop_counts()
    by_module = {
        mod: int(sum(ops.values()))
        for mod, ops in counts.items()
        if mod != "Global"
    }
    return FlopMeasurement(
        total=int(counter.get_total_flops()), by_module=by_module, label=label
    )


def estimate_6nd(n_params: int, n_tokens: int) -> int:
    """The conventional ``6ND`` estimate: 6 FLOPs per parameter per token.

    Provided so the gap against the measured value can be *reported*, not to be
    used as a budget. It assumes a dense transformer whose cost is dominated by
    parameter-weighted matmuls, which is false for any architecture with a
    channel-dimension front-end.
    """
    return int(6 * n_params * n_tokens)


def chinchilla_optimal_params(n_positions: int, epochs: int = 4) -> int:
    """Rough compute-optimal parameter count for a corpus of this size.

    Uses the ~20 tokens-per-parameter heuristic against *effective* tokens
    (positions x epochs). For the realistic MEG corpus -- ~600 h at 62.5
    tokens/s with 5 groups, about 1.35e8 positions -- four epochs gives ~27M
    parameters.

    That is the arithmetic behind the project's refusal to scale: MEG-XL is 20M
    and is state of the art, and a 1B-parameter brain foundation model on
    current public data is a category error. Rungs exist to measure the scaling
    curve, not because the large end is expected to win.
    """
    if n_positions <= 0 or epochs <= 0:
        raise ValueError("n_positions and epochs must be positive")
    return int(n_positions * epochs / 20)


@dataclass
class ComputeLedger:
    """Per-arm compute budget, including charged hyperparameter search.

    One ledger per (arm, tier). ``spend_hpo`` and ``spend_train`` both count
    against the same budget, so an arm that searches harder has less left to
    train with -- which is the point.
    """

    arm: str
    tier: str
    budget: float
    train_flops: int = 0
    hpo_flops: int = 0
    steps: int = 0
    hpo_trials: int = 0

    @property
    def total(self) -> int:
        return self.train_flops + self.hpo_flops

    @property
    def remaining(self) -> float:
        return self.budget - self.total

    @property
    def hpo_fraction(self) -> float:
        return self.hpo_flops / self.total if self.total else 0.0

    @property
    def exhausted(self) -> bool:
        return self.total >= self.budget

    def spend_train(self, per_step: int, n_steps: int = 1) -> None:
        if per_step < 0 or n_steps < 0:
            raise ValueError("cannot spend negative FLOPs")
        self.train_flops += per_step * n_steps
        self.steps += n_steps

    def spend_hpo(self, per_step: int, n_steps: int, n_trials: int = 1) -> None:
        """Charge a hyperparameter search to this arm.

        The design budgets 16 ASHA trials at 3% of the tier each. Whatever the
        search costs, it comes out of this arm's allowance.
        """
        self.hpo_flops += per_step * n_steps * n_trials
        self.hpo_trials += n_trials

    def steps_affordable(self, per_step: int) -> int:
        """Training steps left after HPO, at the given per-step cost."""
        if per_step <= 0:
            raise ValueError("per_step must be positive")
        return max(int(self.remaining // per_step), 0)

    def summary(self) -> str:
        return (
            f"{self.arm:<12} tier={self.tier:<3} "
            f"budget={self.budget:.2e}  "
            f"train={self.train_flops:.3e}  "
            f"hpo={self.hpo_flops:.3e} ({100 * self.hpo_fraction:4.1f}%)  "
            f"steps={self.steps:,}  "
            f"{'EXHAUSTED' if self.exhausted else f'left={self.remaining:.2e}'}"
        )


def compare_arms(ledgers: list[ComputeLedger], tol: float = 0.02) -> tuple[bool, str]:
    """Check that arms really did receive matched compute.

    Returns ``(ok, report)``. Call before writing any comparison table: if the
    spread exceeds ``tol``, the arms were not matched and H1 is not testable
    from these runs.
    """
    if not ledgers:
        return True, "no ledgers"
    totals = [lg.total for lg in ledgers]
    lo, hi = min(totals), max(totals)
    spread = (hi - lo) / hi if hi else 0.0
    ok = spread <= tol
    lines = [lg.summary() for lg in ledgers]
    lines.append(
        f"spread across arms: {100 * spread:.2f}% "
        f"({'within' if ok else 'EXCEEDS'} {100 * tol:.0f}% tolerance)"
    )
    if not ok:
        lines.append(
            "arms did not receive matched compute -- H1 is not testable from these runs"
        )
    return ok, "\n".join(lines)
