"""Prove the control suite actually catches leakage.

A harness that never fails is worthless. This script builds two synthetic
datasets with *known* ground truth and checks the suite reaches the right
verdict on each.

Case A -- spectral shortcut (must be CAUGHT)
    Classes differ only in alpha-band power. A decoder reading log-variance
    succeeds. Because Fourier phase randomisation preserves each trial's power
    spectrum exactly, the decoder still succeeds on the surrogate -- so the
    phase-randomised control must FAIL, correctly reporting "this is power, not
    dynamics."

    This is the case that matters for the project. FMScope names the aperiodic
    1/f slope as a subject-identity carrier; any model whose accuracy survives
    phase randomisation is reading spectral identity, not neural dynamics.

Case B -- genuine temporal structure (must PASS)
    Classes share an identical expected power spectrum and differ only in the
    SIGN of an evoked deflection. All discriminative information lives in
    Fourier phase, so every surrogate destroys it and every control passes.

Run:  .venv/Scripts/python.exe scripts/validate_controls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocast.audit.controls import run_control_suite  # noqa: E402

N_TRIALS = 240
N_CH = 8
N_TIMES = 100
FS = 250.0


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        m = y_true == c
        if m.sum() == 0:
            continue
        recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def make_nearest_centroid_scorer(feature_fn):
    """Refit-internally scorer: fit centroids on the first half, score the second.

    Refitting inside the scorer is deliberate -- it means a surrogate cannot be
    defeated by a stale decoder, and it mirrors how a real evaluation retrains
    on the surrogate. The split is a fixed contiguous halving, which is the
    temporally-blocked split the project mandates elsewhere.
    """

    def scorer(x: np.ndarray, y: np.ndarray) -> float:
        f = feature_fn(x)
        f = (f - f.mean(0)) / (f.std(0) + 1e-12)
        n = len(y)
        tr = np.arange(n // 2)
        te = np.arange(n // 2, n)
        classes = np.unique(y)
        cents = []
        for c in classes:
            m = y[tr] == c
            cents.append(f[tr][m].mean(0) if m.any() else np.full(f.shape[1], np.inf))
        cents = np.stack(cents)
        d = ((f[te][:, None, :] - cents[None]) ** 2).sum(-1)
        pred = classes[np.argmin(d, axis=1)]
        return balanced_accuracy(y[te], pred)

    return scorer


def log_variance(x: np.ndarray) -> np.ndarray:
    return np.log(x.var(axis=2) + 1e-12)


def raw_waveform(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def pink_noise(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """1/f noise, so the synthetic data has a realistic aperiodic slope."""
    n_t = shape[-1]
    spec = np.fft.rfft(rng.standard_normal(shape), axis=-1)
    freqs = np.fft.rfftfreq(n_t, d=1.0 / FS)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    return np.fft.irfft(spec * scale, n=n_t, axis=-1)


def case_a_spectral(rng):
    """Classes differ ONLY in alpha power. Phase randomisation preserves it."""
    y = np.tile([0, 1], N_TRIALS // 2)
    x = pink_noise(rng, (N_TRIALS, N_CH, N_TIMES))
    t = np.arange(N_TIMES) / FS
    alpha = np.sin(2 * np.pi * 10.0 * t)
    # Class 1 gets extra alpha with random phase per trial, so only POWER
    # differs -- no consistent phase for a template matcher to latch onto.
    for i in np.where(y == 1)[0]:
        ph = rng.uniform(0, 2 * np.pi)
        x[i] += 1.6 * np.sin(2 * np.pi * 10.0 * t + ph)
    return x, y, alpha


def case_b_evoked(rng):
    """Classes share an expected PSD and differ only in evoked SIGN (phase).

    The template MUST be zero-mean. A Gaussian-windowed sine is not: windowing
    an 8 Hz sine at t=0.2 s leaves a substantial DC component, and sign-flipping
    it puts class information into the DC bin. Phase randomisation legitimately
    preserves DC (it is part of the power spectrum), so the surrogate would
    carry the class signal and the control would fail -- correctly, but for a
    reason that has nothing to do with evoked dynamics.

    That is not a hypothetical: it is exactly what happened on the first run of
    this script, and it is a real trap for anyone building surrogate controls.
    A constant per-trial offset that correlates with condition survives every
    spectral surrogate.
    """
    y = np.tile([0, 1], N_TRIALS // 2)
    x = pink_noise(rng, (N_TRIALS, N_CH, N_TIMES))
    t = np.arange(N_TIMES) / FS
    template = np.exp(-((t - 0.20) ** 2) / (2 * 0.03**2)) * np.sin(2 * np.pi * 8.0 * t)
    template = template - template.mean()  # kill the DC shortcut
    sign = np.where(y == 0, 1.0, -1.0)
    x += 1.1 * sign[:, None, None] * template[None, None, :]
    return x, y, template


def case_c_offset(rng):
    """Classes differ ONLY by a constant DC offset -- a pure block confound.

    This is the shortcut that survives every spectral surrogate. It stands in
    for slow drift, impedance change, or head movement that happens to
    correlate with condition -- the block-design confound that the EEG-to-image
    literature was built on before it was audited.
    """
    y = np.tile([0, 1], N_TRIALS // 2)
    x = pink_noise(rng, (N_TRIALS, N_CH, N_TIMES))
    x += np.where(y == 0, 0.0, 0.45)[:, None, None]
    return x, y, None


def report_case(name, x, y, feature_fn, expect_pass, rng):
    scorer = make_nearest_centroid_scorer(feature_fn)
    rep = run_control_suite(scorer, x, y, rng=rng, n_permutations=500)
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(rep.to_table())

    ok = rep.passed == expect_pass
    want = "PASS" if expect_pass else "FAIL (leak must be caught)"
    got = "PASS" if rep.passed else "FAIL"
    print(f"\nexpected suite verdict: {want}\nactual suite verdict:   {got}")
    print("--> harness behaved correctly" if ok else "--> HARNESS IS BROKEN")
    return ok, rep


def main() -> int:
    rng = np.random.default_rng(0)

    xa, ya, _ = case_a_spectral(rng)
    ok_a, rep_a = report_case(
        "CASE A - alpha-power shortcut, log-variance decoder\n"
        "         truth: decodable, but ONLY from the power spectrum\n"
        "         the phase-randomised control must FAIL",
        xa, ya, log_variance, expect_pass=False, rng=rng,
    )

    xb, yb, _ = case_b_evoked(rng)
    ok_b, rep_b = report_case(
        "CASE B - sign-flipped evoked response, waveform decoder\n"
        "         truth: decodable from genuine temporal structure only\n"
        "         every control must PASS",
        xb, yb, raw_waveform, expect_pass=True, rng=rng,
    )

    xc, yc, _ = case_c_offset(rng)
    ok_c, rep_c = report_case(
        "CASE C - pure DC offset confound, mean-only decoder\n"
        "         truth: not neural at all, just a per-trial offset\n"
        "         the dc_only control must FAIL",
        xc, yc, lambda z: z.mean(axis=2), expect_pass=False, rng=rng,
    )

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"Case A baseline {rep_a.baseline_score:.3f} (null p95 {rep_a.null_p95:.3f})")
    pr_a = next(r for r in rep_a.results if r.name == "phase_randomized_mv")
    print(f"  phase-randomised score {pr_a.score:.3f} -- survives, so this is spectral")
    print(f"Case B baseline {rep_b.baseline_score:.3f} (null p95 {rep_b.null_p95:.3f})")
    pr_b = next(r for r in rep_b.results if r.name == "phase_randomized_mv")
    print(f"  phase-randomised score {pr_b.score:.3f} -- collapses, so this is dynamics")

    dc_c = next(r for r in rep_c.results if r.name == "dc_only")
    print(f"Case C baseline {rep_c.baseline_score:.3f} (null p95 {rep_c.null_p95:.3f})")
    print(f"  dc_only score {dc_c.score:.3f} -- offset confound caught")

    if ok_a and ok_b and ok_c:
        print("\nControl harness VALIDATED: it discriminates spectral shortcuts")
        print("from genuine temporal structure.")
        return 0
    print("\nControl harness FAILED validation. Do not trust any downstream number.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
