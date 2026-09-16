"""The null-control suite. Nothing in this project is a result until it passes.

Policy (from the plan): *a number without its controls is not a number.* Every
reported metric carries its control rows inline, not in an appendix.

The suite substitutes the neural input with surrogates of increasing strictness
and requires the score to fall to chance. Chance is not assumed -- it is
measured by an exact label-permutation null, because balanced accuracy on small
or unbalanced eval sets does not sit where the arithmetic says it should.

Usage
-----
    report = run_control_suite(scorer, X, y, rng=np.random.default_rng(0))
    print(report.to_table())
    assert report.passed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from .surrogates import (
    ar_surrogate,
    dc_only,
    circular_time_shift,
    gaussian_surrogate,
    phase_randomized_surrogate,
    shuffle_channels,
)

__all__ = [
    "Scorer",
    "ControlResult",
    "ControlReport",
    "permutation_null",
    "run_control_suite",
]


class Scorer(Protocol):
    """Evaluates a decoder. Must be deterministic given (x, y)."""

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float: ...


@dataclass(frozen=True)
class ControlResult:
    name: str
    score: float
    passed: bool
    expectation: str
    threshold: float
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        thr = "     n/a" if np.isnan(self.threshold) else f"{self.threshold:8.4f}"
        return f"[{mark}] {self.name:<24} {self.score:8.4f}  thr {thr}  {self.detail}"


@dataclass
class ControlReport:
    baseline_score: float
    null_mean: float
    null_p95: float
    n_permutations: int
    results: list[ControlResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[ControlResult]:
        return [r for r in self.results if not r.passed]

    def to_table(self) -> str:
        rule = "-" * 78
        head = (
            f"baseline score    {self.baseline_score:8.4f}\n"
            f"permutation null  mean {self.null_mean:.4f}   p95 {self.null_p95:.4f}"
            f"   (n={self.n_permutations})\n{rule}"
        )
        body = "\n".join(str(r) for r in self.results)
        if self.passed:
            verdict = "ALL CONTROLS PASSED"
        else:
            names = ", ".join(r.name for r in self.failures)
            verdict = f"{len(self.failures)} CONTROL(S) FAILED: {names}"
        return f"{head}\n{body}\n{rule}\n{verdict}"


def permutation_null(
    scorer: Scorer,
    x: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int = 1000,
) -> np.ndarray:
    """Exact null distribution of the metric under label permutation.

    Permuting labels (rather than substituting the input) isolates the
    relationship between data and label while leaving every property of the data
    itself -- autocorrelation, drift, class counts -- untouched. That makes it
    the correct reference for "what would chance look like here?".
    """
    y = np.asarray(y)
    return np.array(
        [float(scorer(x, y[rng.permutation(len(y))])) for _ in range(n_permutations)]
    )


def run_control_suite(
    scorer: Scorer,
    x: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int = 1000,
    artifact_channels: Sequence[int] | None = None,
    time_shift_samples: int | None = None,
    extra: dict[str, Callable[[], float]] | None = None,
) -> ControlReport:
    """Run the standard battery and return a pass/fail report.

    Parameters
    ----------
    scorer
        Callable ``(x, y) -> float``, higher is better. Must run the *entire*
        pipeline -- preprocessing, normalisation, conditioning, rescoring -- not
        just the model forward pass. A control that skips preprocessing cannot
        catch leakage introduced by preprocessing.
    artifact_channels
        Indices of peripheral channels (EOG/ECG/EMG). If given, adds the
        artifact-only row: the same decoder using *only* these channels. This
        row is missing from essentially every paper in the field.
    time_shift_samples
        Shift for the time-shift control. Defaults to half the window, which
        guarantees stimulus-response misalignment.
    extra
        Additional named at-chance controls as zero-argument callables. Used for
        controls needing state the suite does not own (subject-prompt shuffling,
        stimulus-disjoint splits).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_times), got {x.shape}")
    if len(y) != x.shape[0]:
        raise ValueError(f"x has {x.shape[0]} trials but y has {len(y)} labels")

    baseline = float(scorer(x, y))
    null = permutation_null(scorer, x, y, rng=rng, n_permutations=n_permutations)
    null_mean = float(null.mean())
    null_p95 = float(np.percentile(null, 95))

    results: list[ControlResult] = []

    def at_chance(name: str, surrogate: np.ndarray, detail: str = "") -> None:
        score = float(scorer(surrogate, y))
        results.append(
            ControlResult(
                name=name,
                score=score,
                passed=score <= null_p95,
                expectation="at_chance",
                threshold=null_p95,
                detail=detail,
            )
        )

    at_chance("dc_only", dc_only(x), "per-trial offset shortcut")
    at_chance("signal_blind_gaussian", gaussian_surrogate(x, rng), "matched mean/var")
    at_chance("signal_blind_ar16", ar_surrogate(x, rng, order=16), "matched autocorr")
    at_chance(
        "phase_randomized_mv",
        phase_randomized_surrogate(x, rng, multivariate=True),
        "exact PSD + spatial cov",
    )
    at_chance(
        "phase_randomized_uv",
        phase_randomized_surrogate(x, rng, multivariate=False),
        "exact PSD per channel",
    )

    # Within-window circular shift. NOTE: this is a *diagnostic*, not an
    # at-chance control. It shifts every trial identically, so a scorer that
    # refits internally can simply relearn the shifted template and score
    # unchanged. It therefore measures reliance on absolute within-window
    # timing, nothing more.
    #
    # The real time-shift control -- desynchronising stimulus from response --
    # must be applied in continuous time *before* epoching, which this suite
    # cannot do because it receives epochs. Pass it via `extra` from the data
    # layer, where the continuous recording is still available.
    shift = time_shift_samples if time_shift_samples is not None else x.shape[2] // 2
    shifted = float(scorer(circular_time_shift(x, shift), y))
    results.append(
        ControlResult(
            name="within_window_shift",
            score=shifted,
            passed=True,
            expectation="diagnostic",
            threshold=float("nan"),
            detail=f"shift={shift}; delta {shifted - baseline:+.4f} (informational)",
        )
    )

    # Channel shuffle is diagnostic, not pass/fail: a topography-free decoder
    # legitimately scores the same. Record the delta and always pass.
    shuffled = float(scorer(shuffle_channels(x, rng), y))
    results.append(
        ControlResult(
            name="channel_shuffle",
            score=shuffled,
            passed=True,
            expectation="diagnostic",
            threshold=float("nan"),
            detail=f"delta {shuffled - baseline:+.4f} vs baseline (informational)",
        )
    )

    if artifact_channels is not None:
        idx = np.asarray(list(artifact_channels), dtype=int)
        masked = np.zeros_like(x)
        masked[:, idx, :] = x[:, idx, :]
        score = float(scorer(masked, y))
        gain = max(baseline - null_mean, 1e-9)
        pct = 100.0 * max(score - null_mean, 0.0) / gain
        results.append(
            ControlResult(
                name="artifact_only",
                score=score,
                passed=score < baseline,
                expectation="below_baseline",
                threshold=baseline,
                detail=f"{len(idx)} peripheral ch; explains {pct:.1f}% of gain",
            )
        )

    for name, fn in (extra or {}).items():
        score = float(fn())
        results.append(
            ControlResult(
                name=name,
                score=score,
                passed=score <= null_p95,
                expectation="at_chance",
                threshold=null_p95,
            )
        )

    return ControlReport(
        baseline_score=baseline,
        null_mean=null_mean,
        null_p95=null_p95,
        n_permutations=n_permutations,
        results=results,
    )
