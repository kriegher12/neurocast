"""Enforce INV-1: likelihood must be invariant to normalisation.

This is the test that keeps the objective bake-off honest. It does three things:

1. Round-trip and composition sanity for :class:`AffineTransform`.
2. THE INVARIANT -- fit the same density family in canonical space and in an
   arbitrarily normalised space, and check the canonical-corrected log-density
   agrees to numerical precision.
3. THE FAILURE MODE -- show what happens without the correction: an arm that
   normalises harder reports a better likelihood having learned nothing. This is
   the bug INV-1 exists to prevent, and it is worth seeing the size of it.

Run:  .venv/Scripts/python.exe scripts/validate_canonical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocast.data.canonical import (  # noqa: E402
    AffineTransform,
    CANONICAL_FS,
    ChannelType,
    NormalizationRegime,
    fit_normalizer,
    nats_per_second_per_sensor,
    to_canonical,
)

N_CH = 12
N_TIMES = 500
TOL = 1e-8


def gaussian_mle_logprob(x: np.ndarray) -> float:
    """Total log-density of a diagonal Gaussian fitted to x by maximum likelihood.

    Deliberately a *fitted* model, not a fixed one: that is the situation in the
    bake-off, where each arm fits its own density. A fitted model is exactly
    where the normalisation cheat bites, because the fit absorbs the scale.
    """
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    var = np.maximum(var, 1e-30)
    return float(
        np.sum(-0.5 * np.log(2 * np.pi * var) - (x - mu) ** 2 / (2 * var))
    )


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


def main() -> int:
    rng = np.random.default_rng(7)
    ok = True

    # Realistic-ish raw MEG: magnetometers in tesla, gradiometers in T/m.
    types = [ChannelType.MAG] * 4 + [ChannelType.GRAD] * 4 + [ChannelType.EEG] * 4
    raw = np.concatenate(
        [
            rng.standard_normal((4, N_TIMES)) * 2e-12,
            rng.standard_normal((4, N_TIMES)) * 6e-11,
            rng.standard_normal((4, N_TIMES)) * 3e-5,
        ]
    )

    print("=" * 78)
    print("1. AffineTransform mechanics")
    print("=" * 78)

    canon, tf_canon = to_canonical(raw, types)
    ok &= check(
        "to_canonical brings families to comparable scale",
        0.05 < canon.std() < 20.0,
        f"std={canon.std():.3f}",
    )
    ok &= check(
        "invert round-trips",
        np.allclose(tf_canon.invert(canon), raw, rtol=1e-10),
    )

    a = AffineTransform(rng.uniform(0.5, 2.0, N_CH), rng.normal(0, 1, N_CH), "a")
    b = AffineTransform(rng.uniform(0.5, 2.0, N_CH), rng.normal(0, 1, N_CH), "b")
    ok &= check(
        "compose matches sequential application",
        np.allclose(a.compose(b).apply(canon), b.apply(a.apply(canon)), rtol=1e-10),
    )
    ok &= check(
        "log|det J| is additive under composition",
        abs(
            a.compose(b).log_det_jacobian(N_TIMES)
            - (a.log_det_jacobian(N_TIMES) + b.log_det_jacobian(N_TIMES))
        )
        < 1e-6,
    )

    print()
    print("=" * 78)
    print("2. INV-1: likelihood invariance under normalisation")
    print("=" * 78)

    lp_canon = gaussian_mle_logprob(canon)
    print(f"log p in canonical space                     {lp_canon:14.4f}")

    rows = []
    for label, norm in [
        ("identity", AffineTransform.identity(N_CH)),
        ("session-robust", fit_normalizer(canon, NormalizationRegime.SESSION_ROBUST)),
        ("random affine", AffineTransform(rng.uniform(0.1, 10, N_CH), rng.normal(0, 5, N_CH))),
        ("pathological x1e6", AffineTransform(np.full(N_CH, 1e6), np.zeros(N_CH))),
    ]:
        y = norm.apply(canon)
        lp_y = gaussian_mle_logprob(y)
        lp_corrected = norm.to_canonical_logprob(lp_y, N_TIMES)
        rows.append((label, lp_y, lp_corrected))
        agrees = abs(lp_corrected - lp_canon) < max(TOL * abs(lp_canon), 1e-6)
        ok &= check(
            f"corrected log p matches canonical  [{label}]",
            agrees,
            f"raw={lp_y:12.2f}  corrected={lp_corrected:12.2f}",
        )

    print()
    print("=" * 78)
    print("3. The failure mode INV-1 prevents")
    print("=" * 78)
    print(f"{'normalisation':<20} {'UNCORRECTED log p':>20} {'CORRECTED log p':>20}")
    print("-" * 78)
    for label, lp_y, lp_corrected in rows:
        print(f"{label:<20} {lp_y:20.2f} {lp_corrected:20.2f}")
    print("-" * 78)
    spread_raw = max(r[1] for r in rows) - min(r[1] for r in rows)
    spread_fix = max(r[2] for r in rows) - min(r[2] for r in rows)
    print(f"spread across arms   uncorrected {spread_raw:12.2f} nats")
    print(f"                     corrected   {spread_fix:12.2e} nats")
    print()
    print("Without the correction, the arm that normalises hardest 'wins' by")
    print(f"{spread_raw:,.0f} nats while modelling exactly the same data. That is")
    print("the entire objective bake-off, decided by a preprocessing choice.")

    ok &= check(
        "uncorrected likelihoods genuinely diverge (the bug is real)",
        spread_raw > 1e4,
        f"{spread_raw:.3e} nats",
    )
    ok &= check(
        "corrected likelihoods agree (the fix works)",
        spread_fix < 1e-6,
        f"{spread_fix:.3e} nats",
    )

    print()
    print("=" * 78)
    print("4. Reporting unit")
    print("=" * 78)
    npsps = nats_per_second_per_sensor(lp_canon, N_CH, N_TIMES, CANONICAL_FS)
    print(f"NLL = {npsps:.4f} nats/s/sensor  ({N_CH} ch, {N_TIMES/CANONICAL_FS:.1f} s)")

    # Reference: differential entropy of N(0, sigma^2) is 0.5*log(2*pi*e) + log(sigma).
    # Canonical space deliberately does NOT force unit variance -- the fixed
    # per-family constants preserve amplitude, because amplitude and 1/f are
    # exactly what we want to measure rather than silently remove. So the
    # reference must use the channels' actual scales, not assume sigma = 1.
    sigma = canon.std(axis=1)
    expected = (0.5 * np.log(2 * np.pi * np.e) + np.mean(np.log(sigma))) * CANONICAL_FS
    print(f"channel sigma in canonical space: {np.round(sigma, 2)}")
    ok &= check(
        "matches Gaussian differential entropy at the observed scales",
        abs(npsps - expected) / expected < 0.02,
        f"got {npsps:.1f}, expected ~{expected:.1f}",
    )

    print()
    print("=" * 78)
    if ok:
        print("INV-1 VALIDATED. Likelihoods are comparable across normalisations.")
        return 0
    print("INV-1 VIOLATED. Do not trust any likelihood number.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
