"""Validate INV-1's reference rows against processes with known answers.

Every NLL this project reports carries two nulls inline: a PSD-matched stationary
Gaussian and a per-session AR(16). A null that is too weak is worse than none --
it turns ordinary linear structure into an apparent discovery -- so the rows are
tested the way the Atlas is: against analytic values.

1. Whittle is exact where it should be: white noise hits the Gaussian entropy.
2. The PSD-matched null reaches the analytic entropy rate of AR(1) processes.
3. THE MEASUREMENT THAT CHOSE THE NULL: on coloured spectra a plain AR(16) beats
   the Whittle approximation at every window length -- so Whittle cannot be the
   primary null -- while it cannot beat the time-domain PSD null.
4. INV-1: a per-channel gain shifts every raw NLL by exactly log(gain) and leaves
   every delta untouched. Deltas are the reportable quantity.
5. "Learned 1/f" is exposed: a model that is only a PSD-matched Gaussian beats
   white noise by a mile and the PSD null by nothing.
6. Real beyond-spectral structure (volatility clustering) does beat the PSD null.
7. The report renders with the reference rows inline.

Run:  .venv/Scripts/python.exe scripts/validate_eval.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocast.atlas import estimators as est  # noqa: E402
from neurocast.data.canonical import AffineTransform  # noqa: E402
from neurocast.eval.references import (  # noqa: E402
    ar_nll,
    format_table,
    marginal_nll,
    psd_null_nll,
    reference_report,
    whittle_nll,
)

FS = 250.0
N = 40_000
ENTROPY = 0.5 * math.log(2 * math.pi * math.e)    # unit-variance innovations


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


def ar1(phi: float, n: int, rng: np.random.Generator, burn: int = 2000) -> np.ndarray:
    e = rng.standard_normal(n + burn)
    x = np.empty(n + burn)
    x[0] = e[0]
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def pink(n: int, rng: np.random.Generator, exponent: float) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, d=1.0 / FS)
    scale = np.ones_like(f)
    scale[1:] = f[1:] ** (-exponent / 2.0)
    return np.fft.irfft(spec * scale, n=n)


def garch(n: int, rng: np.random.Generator, burn: int = 2000) -> np.ndarray:
    omega, alpha, beta = 0.05, 0.15, 0.80
    e = rng.standard_normal(n + burn)
    x = np.zeros(n + burn)
    s2 = omega / (1 - alpha - beta)
    for t in range(1, n + burn):
        s2 = omega + alpha * x[t - 1] ** 2 + beta * s2
        x[t] = math.sqrt(s2) * e[t]
    return x[burn:]


def main() -> int:
    rng = np.random.default_rng(0)
    ok = True

    print("=" * 84)
    print("1. Whittle on white noise -- exact where the approximation is exact")
    print("=" * 84)
    tr, te = rng.standard_normal(N), rng.standard_normal(N)
    for length in (500, 1000):
        w = float(whittle_nll(tr, te, length=length)[0])
        ok &= check(f"white noise, L={length}: Whittle hits the Gaussian entropy",
                    abs(w - ENTROPY) < 0.01, f"{w:.4f} vs {ENTROPY:.4f}")

    print()
    print("=" * 84)
    print("2. The PSD-matched null reaches the analytic entropy rate")
    print("=" * 84)
    for phi in (0.9, 0.98):
        tr, te = ar1(phi, N, rng), ar1(phi, N, rng)
        p = float(psd_null_nll(tr, te)[0])
        m = float(marginal_nll(tr, te)[0])
        ok &= check(f"AR(1) phi={phi}: PSD null at the entropy rate",
                    abs(p - ENTROPY) < 0.01,
                    f"{p:.4f} vs {ENTROPY:.4f}  (white {m:.4f})")

    print()
    print("=" * 84)
    print("3. Why the primary null is NOT Whittle -- measured")
    print("=" * 84)
    print(f"  {'process':<12} {'L=250':>8} {'L=500':>8} {'L=1000':>8} {'L=2000':>8} "
          f"{'AR(16)':>8} {'PSD null':>9}   (NLL, nats/sample)")
    worst_whittle_gap, worst_psd_gap = np.inf, -np.inf
    for name, make in (("AR(1) .98", lambda: ar1(0.98, N, rng)),
                       ("1/f^1.5", lambda: pink(N, rng, 1.5))):
        tr, te = make(), make()
        ws = [float(whittle_nll(tr, te, length=L)[0]) for L in (250, 500, 1000, 2000)]
        a, p = float(ar_nll(tr, te)[0]), float(psd_null_nll(tr, te)[0])
        print(f"  {name:<12} " + " ".join(f"{w:8.4f}" for w in ws) + f" {a:8.4f} {p:9.4f}")
        worst_whittle_gap = min(worst_whittle_gap, min(ws) - a)
        worst_psd_gap = max(worst_psd_gap, p - a)
    ok &= check("AR(16) beats Whittle at EVERY window length on coloured spectra",
                worst_whittle_gap > 0.005,
                f"by at least {worst_whittle_gap:.4f} nats/sample -- Whittle as the null "
                "would call linear structure 'beyond the spectrum'")
    ok &= check("AR(16) cannot beat the time-domain PSD null beyond estimation noise",
                worst_psd_gap < 0.002, f"worst PSD - AR(16) = {worst_psd_gap:+.4f}")

    print()
    print("=" * 84)
    print("4. INV-1: gains shift raw NLLs by log(gain); deltas do not move")
    print("=" * 84)
    c = 4
    x_tr = np.stack([pink(N // 2, rng, 1.0) for _ in range(c)])
    x_te = np.stack([pink(N // 2, rng, 1.0) for _ in range(c)])
    model = est.linear(x_tr, x_te, 1, 16, 64, name="model").nll   # stand-in model
    base = reference_report("model", model, x_tr, x_te, fs=FS)
    gains = np.array([1e-3, 0.5, 7.0, 1e4])
    tf = AffineTransform(gains, np.zeros(c), "gain")
    s_tr, s_te = tf.apply(x_tr), tf.apply(x_te)
    model_s = est.linear(s_tr, s_te, 1, 16, 64, name="model").nll
    scaled = reference_report("model", model_s, s_tr, s_te, fs=FS)
    shift = float(np.mean(np.log(gains))) * FS     # nats/s/sensor
    raw_gap = max(abs((scaled.psd - base.psd) - shift), abs((scaled.white - base.white) - shift),
                  abs((scaled.model - base.model) - shift))
    d_gap = max(abs(scaled.delta_psd - base.delta_psd), abs(scaled.delta_ar - base.delta_ar))
    print(f"  per-channel gains {gains.tolist()}  ->  every raw NLL moves by "
          f"{shift:+.2f} nats/s/sensor")
    ok &= check("raw NLLs shift by exactly the mean log gain", raw_gap < 1e-6 * abs(shift),
                f"max error {raw_gap:.1e}")
    ok &= check("deltas are invariant to the gain", d_gap < 1e-6, f"max change {d_gap:.1e}")

    print()
    print("=" * 84)
    print("5. 'Learned 1/f' cannot hide")
    print("=" * 84)
    tr, te = pink(N, rng, 1.0)[None], pink(N, rng, 1.0)[None]
    spectral_only = est.spectral_null(tr, te, 1, 64).nll    # a model that knows only the PSD
    rep = reference_report("PSD-only model", spectral_only, tr, te, fs=FS)
    print(f"  vs white noise {rep.model - rep.white:+.1f}   vs PSD null {rep.delta_psd:+.2f}  "
          f"(nats/s/sensor)")
    ok &= check("it beats white noise by a mile", rep.model - rep.white < -50)
    ok &= check("and the PSD null by nothing", abs(rep.delta_psd) < 0.05,
                "beating the marginal alone is evidence of 1/f, nothing more")

    print()
    print("=" * 84)
    print("6. Genuine beyond-spectral structure is credited")
    print("=" * 84)
    tr, te = garch(N, rng)[None], garch(N, rng)[None]
    het = est.heteroscedastic(tr, te, 1, 16, 64).nll
    rep_g = reference_report("heteroscedastic", het, tr, te, fs=FS)
    print(f"  volatility clustering: d PSD {rep_g.delta_psd:+.2f}  d AR {rep_g.delta_ar:+.2f}")
    ok &= check("a variance model beats the PSD null on GARCH", rep_g.delta_psd < -2.0,
                f"{rep_g.delta_psd:+.2f} nats/s/sensor")

    print()
    print("=" * 84)
    print("7. The report")
    print("=" * 84)
    refs = reference_report("references only", None, x_tr, x_te, fs=FS)
    table = format_table([refs, base, rep, rep_g])
    print(table)
    ok &= check("references-only row renders with n/a deltas",
                math.isnan(refs.delta_psd) and "n/a" in refs.row())
    ok &= check("nats/s/sensor = nats/sample x fs",
                abs(refs.white - float(marginal_nll(x_tr, x_te).mean()) * FS) < 1e-9)

    print()
    print("=" * 84)
    if ok:
        print("REFERENCES VALIDATED: the PSD null reaches the entropy rate, Whittle is")
        print("demoted to a cross-check on measurement, and deltas are gain-invariant.")
        return 0
    print("REFERENCES FAILED validation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
