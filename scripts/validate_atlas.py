"""Validate the Neural Predictability Atlas against processes with known answers.

The Atlas is the program's insurance policy -- publishable whether or not any
model hypothesis holds -- so its estimators get the strictest test in the
codebase: agreement with **analytic** forecastability, not just plausibility.

 1. AR(1): I(h) = -1/2 log(1 - phi^(2h)) exactly. The estimator must match.
 2. White noise: no predictability at any horizon. The estimator must not
    hallucinate any.
 3. Volatility clustering (GARCH): zero autocorrelation, so the spectral null
    sees nothing -- yet the future IS predictable. The Atlas must find it.
 4. KSG mutual information vs. the exact bivariate-Gaussian value.
 5. The filter trap: zero-phase filtering on the conditioning side fabricates
    forecastability of white noise. Causal filtering must not.
 6. Block vs. i.i.d. bootstrap: empirical coverage on autocorrelated losses.
 7. Diebold-Mariano: correct size under the null, power under the alternative.
 8. Monotonicity audit catches an impossible ordering.
 9. Stimulus decomposition: separates world-driven from self-driven dynamics.
10. A linear 1/f process: beats the marginal, adds nothing over the spectral
    null. This is the signature of "learned 1/f" the Atlas must expose.

Run:  .venv/Scripts/python.exe scripts/validate_atlas.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocast.atlas import estimators as est  # noqa: E402
from neurocast.atlas.filters import (  # noqa: E402
    assert_causal,
    causal_filter,
    fir_bandpass,
    leakage_energy,
    zero_phase_filter,
)
from neurocast.atlas.stats import (  # noqa: E402
    BootstrapCI,
    audit_monotonicity,
    block_bootstrap_mean,
    diebold_mariano,
    iid_bootstrap_mean,
)

FS = 250.0
MAX_LAGS = 64


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


def garch(n: int, rng: np.random.Generator, burn: int = 2000) -> np.ndarray:
    """Volatility clustering with zero autocorrelation in the signal itself."""
    omega, alpha, beta = 0.05, 0.15, 0.80
    e = rng.standard_normal(n + burn)
    x = np.zeros(n + burn)
    s2 = omega / (1 - alpha - beta)
    for t in range(1, n + burn):
        s2 = omega + alpha * x[t - 1] ** 2 + beta * s2
        x[t] = math.sqrt(s2) * e[t]
    return x[burn:]


def pink(n: int, rng: np.random.Generator, exponent: float = 1.0) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, d=1.0 / FS)
    scale = np.ones_like(f)
    scale[1:] = f[1:] ** (-exponent / 2.0)
    return np.fft.irfft(spec * scale, n=n)


def info_ci(ref: est.ForecastScore, model: est.ForecastScore, seed: int = 0) -> BootstrapCI:
    """Forecastability = reference loss - model loss, with a block-bootstrap CI."""
    return block_bootstrap_mean(ref.per_sample - model.per_sample,
                                block_len=int(2 * FS), n_boot=1000, seed=seed)


def main() -> int:
    rng = np.random.default_rng(0)
    ok = True
    n = 20_000

    print("=" * 82)
    print("1. AR(1) -- estimated forecastability vs. the exact analytic value")
    print("=" * 82)
    # One realisation of a slow process cannot pin the answer down tightly: at
    # phi=0.98 the spread across 80-second recordings is ~0.035 nats. So compare
    # the MEAN over replicates to the analytic value, in units of standard error.
    reps = 6
    print(f"  {reps} independent replicates per phi, 20,000 train + 20,000 test samples")
    print(f"  {'phi':>5} {'h':>4} {'analytic':>9} {'mean est':>9} {'error':>8} "
          f"{'SE':>7} {'err/SE':>7}")
    worst_over, worst_under, spreads = -np.inf, -np.inf, {}
    for phi in (0.5, 0.9, 0.98):
        per_h = {h: [] for h in (1, 2, 5, 10, 20)}
        for _ in range(reps):
            tr, te = ar1(phi, n, rng), ar1(phi, n, rng)
            for h in per_h:
                per_h[h].append(est.marginal(tr, te, h, MAX_LAGS).nll
                                - est.spectral_null(tr, te, h, MAX_LAGS).nll)
        for h, vals in per_h.items():
            exact = -0.5 * math.log(1 - phi ** (2 * h))
            mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
            se = max(sd / math.sqrt(reps), 1e-4)
            z = (mean - exact) / se
            worst_over = max(worst_over, z)
            worst_under = max(worst_under, (exact - mean - 2.0 * MAX_LAGS / n) / se)
            spreads[phi] = max(spreads.get(phi, 0.0), sd)
            print(f"  {phi:>5.2f} {h:>4} {exact:>9.4f} {mean:>9.4f} {mean - exact:>+8.4f} "
                  f"{se:>7.4f} {z:>+7.2f}")
    print(f"  single-recording spread: " + "  ".join(
        f"phi={k}: +-{v:.4f}" for k, v in spreads.items()))
    # The estimate is a LOWER bound on I, and held-out fitting of MAX_LAGS=64
    # coefficients predictably costs about p/n nats (~0.003 here). So the two
    # directions get different treatment. Above the truth is the dangerous
    # direction -- it would mean hallucinated predictability -- and must stay
    # within sampling error. Below the truth is expected, and must stay within
    # sampling error plus the predicted overfitting penalty.
    penalty = 2.0 * MAX_LAGS / n
    ok &= check("never significantly ABOVE the truth (no hallucinated predictability)",
                worst_over < 3.0, f"worst {worst_over:+.2f} SE")
    ok &= check("below the truth only by the predicted p/n overfitting penalty",
                worst_under < 3.0,
                f"worst {worst_under:+.2f} SE after allowing {penalty:.4f} nats")
    ok &= check("slow processes need more data (spread grows with memory)",
                spreads[0.98] > 3 * spreads[0.5],
                f"phi=0.98 spread is {spreads[0.98] / spreads[0.5]:.0f}x phi=0.5")

    print()
    print("=" * 82)
    print("2. White noise -- no hallucinated predictability")
    print("=" * 82)
    tr, te = rng.standard_normal(n), rng.standard_normal(n)
    clean = True
    for h in (1, 5, 25):
        m = est.marginal(tr, te, h, MAX_LAGS)
        for model in (est.spectral_null(tr, te, h, MAX_LAGS),
                      est.heteroscedastic(tr, te, h, 16, MAX_LAGS)):
            ci = info_ci(m, model)
            spurious = ci.lo > 0.005
            clean &= not spurious
            print(f"  h={h:>3}  {model.name:<22} I = {ci}")
    ok &= check("no model finds significant predictability in white noise", clean)

    print()
    print("=" * 82)
    print("3. Volatility clustering -- predictable, but invisible to any spectrum")
    print("=" * 82)
    tr, te = garch(n, rng), garch(n, rng)
    lag1 = float(np.corrcoef(te[:-1], te[1:])[0, 1])
    print(f"  lag-1 autocorrelation of the signal: {lag1:+.4f}  (spectrum is flat)")
    m = est.marginal(tr, te, 1, MAX_LAGS)
    spec = est.spectral_null(tr, te, 1, MAX_LAGS)
    het = est.heteroscedastic(tr, te, 1, 16, MAX_LAGS)
    het_raw = est.heteroscedastic(tr, te, 1, 16, MAX_LAGS, defend=None)
    raw_gain = float(np.mean(spec.per_sample - het_raw.per_sample))
    med_gain = float(np.median(spec.per_sample - het_raw.per_sample))
    worst_raw, worst_def = float(het_raw.per_sample.max()), float(het.per_sample.max())
    bound = float(m.per_sample[np.argmax(het_raw.per_sample)] - math.log(1e-3))
    print(f"  UNDEFENDED  mean gain {raw_gain:+.4f}  median {med_gain:+.4f}  "
          f"worst sample {worst_raw:.1f} nats")
    print(f"  DEFENDED    worst sample {worst_def:.1f} nats  "
          f"(theoretical cap marginal - log eps = {bound:.1f})")
    ci_spec, ci_het = info_ci(m, spec), info_ci(m, het)
    ci_beyond = info_ci(spec, het)
    print(f"  spectral null      I = {ci_spec}")
    print(f"  heteroscedastic    I = {ci_het}")
    print(f"  beyond spectral      = {ci_beyond}")
    dm = diebold_mariano(het.per_sample, spec.per_sample)
    print(f"  DM heteroscedastic vs spectral: {dm}")
    ok &= check("spectral null finds ~nothing", abs(ci_spec.mean) < 0.01,
                f"{ci_spec.mean:+.4f}")
    ok &= check("defensive mixture caps the worst sample at its theoretical bound",
                worst_def <= bound + 1e-6, f"{worst_raw:.1f} -> {worst_def:.1f} nats")
    ok &= check("heteroscedastic model finds real predictability",
                ci_beyond.lo > 0, f"{ci_beyond.mean:+.4f} nats beyond spectral")
    ok &= check("the gain is statistically significant (DM, HAC)",
                dm.significant and dm.mean_diff < 0)

    # KSG has a resolution floor. Measure it here rather than assume it, then
    # require the real effect to clear it. The single-lag dependence in GARCH is
    # below that floor at this sample size, so condition on a 16-sample volatility
    # proxy, which carries most of the predictive information.
    n_ksg = 3000
    floor = [est.ksg_mutual_information(rng.standard_normal(n_ksg),
                                        rng.standard_normal(n_ksg), seed=k)
             for k in range(5)]
    floor_sd = float(np.std(floor))
    idx = np.arange(16, 16 + n_ksg)
    vol = np.array([np.mean(te[i - 15:i + 1] ** 2) for i in idx])
    ksg_real = est.ksg_mutual_information(np.abs(te[idx + 1]), vol)
    ksg_shuf = est.ksg_mutual_information(np.abs(te[idx + 1]), vol[rng.permutation(n_ksg)])
    print(f"  KSG resolution floor (independent data, N={n_ksg}): sd {floor_sd:.4f}")
    print(f"  KSG I(|x_t+1|; recent volatility) = {ksg_real:+.4f}   "
          f"shuffled = {ksg_shuf:+.4f}")
    ok &= check("model-free KSG independently confirms it, clear of its noise floor",
                ksg_real > ksg_shuf + 2 * floor_sd,
                f"{(ksg_real - ksg_shuf) / floor_sd:.1f} floor-SDs above shuffled")

    print()
    print("=" * 82)
    print("4. KSG vs. exact bivariate-Gaussian mutual information")
    print("=" * 82)
    worst = 0.0
    for rho in (0.0, 0.5, 0.9):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        z = rng.multivariate_normal([0, 0], cov, size=2000)
        got = est.ksg_mutual_information(z[:, 0], z[:, 1])
        exact = est.gaussian_mi(rho)
        worst = max(worst, abs(got - exact))
        print(f"  rho={rho:.1f}  exact {exact:.4f}  KSG {got:.4f}  error {got - exact:+.4f}")
    ok &= check("KSG matches the exact value", worst < 0.05, f"max |error| {worst:.4f}")
    dg = float(est.digamma(np.array([1.0, 10.0]))[0])
    ok &= check("digamma(1) = -Euler-Mascheroni", abs(dg + 0.5772156649) < 1e-8,
                f"{dg:.10f}")

    print()
    print("=" * 82)
    print("5. THE FILTER TRAP -- zero-phase filtering fabricates forecastability")
    print("=" * 82)
    h_fir = fir_bandpass(8.0, 13.0, FS, n_taps=65)
    causal = lambda z: causal_filter(z, h_fir)      # noqa: E731
    zphase = lambda z: zero_phase_filter(z, h_fir)  # noqa: E731
    print(f"  impulse energy preceding the impulse: causal "
          f"{100 * leakage_energy(causal):.2f}%   zero-phase "
          f"{100 * leakage_energy(zphase):.2f}%")
    try:
        assert_causal(causal, name="causal FIR")
        ok &= check("causal filter passes the impulse test", True)
    except AssertionError as exc:
        ok &= check("causal filter passes the impulse test", False, str(exc))
    try:
        assert_causal(zphase, name="zero-phase FIR")
        ok &= check("zero-phase filter is rejected", False)
    except AssertionError:
        ok &= check("zero-phase filter is rejected", True)

    wtr, wte = rng.standard_normal(n), rng.standard_normal(n)
    h = 5
    m = est.marginal(wtr, wte, h, MAX_LAGS)
    for label, filt in (("causal    ", causal), ("zero-phase", zphase)):
        f_tr, f_te = filt(wtr[None])[0], filt(wte[None])[0]
        s = est.linear_with_covariate(wtr, f_tr, wte, f_te, h, lags=1, cov_lags=32,
                                      max_lags=MAX_LAGS, cov_lead=0)
        ci = info_ci(m, s)
        print(f"  {label} band-limited PAST -> raw future of WHITE NOISE: I = {ci}")
        if label.startswith("zero"):
            ok &= check("zero-phase conditioning fabricates predictability", ci.lo > 0.02,
                        "the 'past' contains the future")
        else:
            ok &= check("causal conditioning correctly finds none", ci.lo < 0.005)

    print()
    print("=" * 82)
    print("6. Block vs. i.i.d. bootstrap -- coverage on autocorrelated losses")
    print("=" * 82)
    reps, cover_iid, cover_blk, w_iid, w_blk = 150, 0, 0, [], []
    for r in range(reps):
        d = ar1(0.95, 2000, rng)
        ci_i = iid_bootstrap_mean(d, n_boot=400, seed=r)
        ci_b = block_bootstrap_mean(d, block_len=100, n_boot=400, seed=r)
        cover_iid += ci_i.lo <= 0.0 <= ci_i.hi
        cover_blk += ci_b.lo <= 0.0 <= ci_b.hi
        w_iid.append(ci_i.width)
        w_blk.append(ci_b.width)
    c_i, c_b = cover_iid / reps, cover_blk / reps
    print(f"  nominal 95%   i.i.d. coverage {c_i:.3f} (width {np.mean(w_iid):.3f})   "
          f"block coverage {c_b:.3f} (width {np.mean(w_blk):.3f})")
    ok &= check("i.i.d. bootstrap badly under-covers", c_i < 0.60, f"{c_i:.3f}")
    ok &= check("block bootstrap is close to nominal", c_b > 0.80, f"{c_b:.3f}")

    print()
    print("=" * 82)
    print("7. Diebold-Mariano: size under the null, power under the alternative")
    print("=" * 82)
    from neurocast.atlas.stats import newey_west_lag
    rej_fixed = rej_andrews = 0
    reps_dm = 300
    for r in range(reps_dm):
        base = ar1(0.8, 2000, rng)
        a = base + 0.5 * ar1(0.8, 2000, rng)
        b = base + 0.5 * ar1(0.8, 2000, rng)
        rej_fixed += diebold_mariano(a, b, lag=newey_west_lag(len(a))).significant
        rej_andrews += diebold_mariano(a, b).significant
    s_fixed, s_andrews = rej_fixed / reps_dm, rej_andrews / reps_dm
    print(f"  equal forecasters, nominal 5% -- fixed NW lag: {s_fixed:.3f}   "
          f"Andrews bandwidth (default): {s_andrews:.3f}")
    ok &= check("the textbook fixed lag over-rejects on persistent losses",
                s_fixed > 0.08, f"{s_fixed:.3f}")
    ok &= check("the default (Andrews) bandwidth controls size", s_andrews < 0.10,
                f"{s_andrews:.3f}")
    base = ar1(0.8, 4000, rng)
    better = diebold_mariano(base + 0.4 * ar1(0.8, 4000, rng) - 0.3,
                             base + 0.4 * ar1(0.8, 4000, rng))
    print(f"  genuinely better forecaster: {better}")
    ok &= check("DM detects a real difference", better.significant and better.mean_diff < 0)

    print()
    print("=" * 82)
    print("8. Monotonicity audit")
    print("=" * 82)
    good = {"past": BootstrapCI(0.10, 0.08, 0.12, 0.95),
            "past+stim": BootstrapCI(0.18, 0.15, 0.21, 0.95)}
    bad = {"past": BootstrapCI(0.30, 0.27, 0.33, 0.95),
           "past+stim": BootstrapCI(0.12, 0.10, 0.14, 0.95)}
    nest = [("past", "past+stim")]
    ok &= check("consistent ordering passes", not audit_monotonicity(good, nest))
    v = audit_monotonicity(bad, nest)
    ok &= check("adding information that REDUCES forecastability is flagged", len(v) == 1)
    if v:
        print(f"         caught: {v[0]}")

    print()
    print("=" * 82)
    print("9. Stimulus decomposition -- world-driven vs. self-driven dynamics")
    print("=" * 82)

    def driven(nn: int):
        u = rng.standard_normal(nn)
        e = rng.standard_normal(nn)
        y = np.zeros(nn)
        for t in range(2, nn):
            y[t] = 0.8 * y[t - 1] + 0.6 * u[t - 2] + e[t]
        return y, u

    ytr, utr = driven(n)
    yte, ute = driven(n)
    rows = {}
    for h in (1, 3):
        m = est.marginal(ytr, yte, h, MAX_LAGS)
        own = est.linear(ytr, yte, h, 8, MAX_LAGS)
        both = est.linear_with_covariate(ytr, utr, yte, ute, h, lags=8, cov_lags=8,
                                         max_lags=MAX_LAGS)
        i_own, i_both = info_ci(m, own), info_ci(m, both)
        rows[h] = (i_own, i_both)
        print(f"  h={h}  own past {i_own}   + stimulus {i_both}   "
              f"stimulus share {100 * (i_both.mean - i_own.mean) / i_both.mean:.0f}%")
    audit = audit_monotonicity({"own": rows[1][0], "both": rows[1][1]}, [("own", "both")])
    ok &= check("the stimulus adds information beyond the brain's own past",
                rows[1][1].lo > rows[1][0].hi)
    ok &= check("estimates satisfy the monotonicity audit", not audit)

    print()
    print("=" * 82)
    print("10. Linear 1/f process -- the signature of 'learned 1/f'")
    print("=" * 82)
    tr, te = pink(n, rng), pink(n, rng)
    h = 5
    m = est.marginal(tr, te, h, MAX_LAGS)
    spec = est.spectral_null(tr, te, h, MAX_LAGS)
    het = est.heteroscedastic(tr, te, h, 16, MAX_LAGS)
    over_marg = info_ci(m, spec)
    beyond = info_ci(spec, het)
    print(f"  spectral null vs marginal   I = {over_marg}")
    print(f"  nonlinear vs spectral null  I = {beyond}")
    ok &= check("1/f is highly predictable relative to the marginal", over_marg.lo > 0.1)
    ok &= check("but nothing survives beyond the spectral null", beyond.hi < 0.01,
                "a model that only beats the marginal has learned 1/f, nothing more")

    print()
    print("=" * 82)
    if ok:
        print("ATLAS VALIDATED: matches analytic forecastability, rejects the filter")
        print("trap, and separates spectral from beyond-spectral structure.")
        return 0
    print("ATLAS FAILED validation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
