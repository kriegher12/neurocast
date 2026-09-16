"""Verify the generative stack end to end: corpus, targets, training, rollout.

1. **Same brain, two physics.** Sources recovered by least squares from a MEG
   array and, separately, from an EEG cap both track the true sources.
2. **Targets invert.** Quadrant PCA reconstructs held-out sensors; its variance
   loss is reported, and the whitening Jacobian is finite.
3. **Training reduces next-group NLL.**
4. **Rollout rests on group confinement -- checked on the trained model.** Filling
   later groups of the new patch with garbage must not move the prediction for an
   earlier group. The rollout is only valid if this holds.
5. **Forecast quality, reported honestly** against two baselines (channel-mean
   and last-patch persistence), per horizon. On a briefly-trained rehearsal model
   the bar is the first patch; how far the advantage extends is exactly the
   question the Atlas exists to answer.

Run:  .venv/Scripts/python.exe scripts/validate_generative.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurocast.data.sources import SourceSpaceCorpus  # noqa: E402
from neurocast.models.forecast import mixture_mean, rollout  # noqa: E402
from neurocast.train.generative import train_generative  # noqa: E402
from neurocast.train.targets import QuadrantBasis  # noqa: E402
from validate_tokenizer import eeg_montage, megin_montage  # noqa: E402


def ar_forecast(context: np.ndarray, n_ahead: int, order: int = 32) -> np.ndarray:
    """Per-channel least-squares AR(order) fitted on the context, iterated forward.

    The achievable-predictability reference: these sources are narrow-band
    oscillations, which a linear predictor forecasts well. A generative model that
    falls far short of this has not learned the dynamics, whatever it beats.
    """
    c, t = context.shape
    out = np.empty((c, n_ahead))
    for ch in range(c):
        x = context[ch] - context[ch].mean()
        a = np.stack([x[order - 1 - j: t - 1 - j] for j in range(order)], axis=1)
        coef, *_ = np.linalg.lstsq(a, x[order:], rcond=None)
        hist = list(x[-order:])
        for k in range(n_ahead):
            nxt = float(np.dot(coef, hist[::-1][:order]))
            out[ch, k] = nxt
            hist.append(nxt)
        out[ch] += context[ch].mean()
    return out


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


def main() -> int:
    ok = True
    rng = np.random.default_rng(0)
    corpus = SourceSpaceCorpus(n_sources=6, n_subjects=8, seed=0)
    meg, eeg = megin_montage(n_sites=24), eeg_montage(48)
    patch = 16

    print("=" * 80)
    print("1. Same brain, two physics")
    print("=" * 80)
    n_times = 1024
    subj = 0
    b = corpus.batch(meg, 1, n_times, np.random.default_rng(5), subjects=[subj], noise=0.05)
    src = b.sources[0]
    x_meg = np.einsum("ck,kt->ct", corpus.leadfield(meg), src)
    x_eeg = np.einsum("ck,kt->ct", corpus.leadfield(eeg), src)
    for name, lf, xs in (("MEG", corpus.leadfield(meg), x_meg), ("EEG", corpus.leadfield(eeg), x_eeg)):
        est, *_ = np.linalg.lstsq(lf, xs, rcond=None)
        r = [float(np.corrcoef(est[k], src[k])[0, 1]) for k in range(corpus.n_sources)]
        print(f"  {name}: sources recovered, min r = {min(r):.3f}")
        ok &= check(f"{name} array recovers the shared sources", min(r) > 0.95)
    lf_m, lf_e = corpus.leadfield(meg), corpus.leadfield(eeg)
    ok &= check("the two arrays really see different projections",
                lf_m.shape != lf_e.shape and bool(np.all(lf_m[-3:] == 0)),
                "peripheral channels receive nothing brain-derived")

    print()
    print("=" * 80)
    print("2. Quadrant PCA targets invert back to sensors")
    print("=" * 80)
    calib = corpus.batch(meg, 32, 32 * patch, rng).x
    held = corpus.batch(meg, 8, 32 * patch, rng).x
    basis = QuadrantBasis(rank=16, patch=patch).fit(calib, meg.quadrants)
    r2 = basis.reconstruction_r2(held)
    evr = [e for e in basis.explained_variance() if not np.isnan(e)]
    print(f"  explained variance per quadrant: " + "  ".join(f"{e:.2f}" for e in evr))
    print(f"  held-out reconstruction R^2 = {r2:.3f}   whitening log|det| = "
          f"{basis.log_det_jacobian():.1f}")
    ok &= check("held-out sensors reconstruct from rank-16 targets", r2 > 0.6, f"R^2 {r2:.3f}")
    ok &= check("whitening Jacobian is finite", np.isfinite(basis.log_det_jacobian()))

    print()
    print("=" * 80)
    print("3. Train the generative arm (rehearsal rung)")
    print("=" * 80)
    gm = train_generative(corpus, meg, n_patch=32, batch=4, steps=240, rank=16, seed=0)
    print(f"  {gm.model.describe()}")
    print(f"  NLL {gm.initial_loss:+.4f} -> {gm.final_loss:+.4f} nats/component "
          f"in {gm.seconds:.0f}s")
    ok &= check("next-group NLL decreases", gm.final_loss < gm.initial_loss - 0.05,
                f"{gm.initial_loss - gm.final_loss:+.4f}")

    print()
    print("=" * 80)
    print("4. Group confinement holds in the trained model (rollout depends on it)")
    print("=" * 80)
    ctx = corpus.batch(meg, 1, 48 * patch, np.random.default_rng(11)).x[0]
    t_new = 32
    sig = ctx[:, : (t_new + 1) * patch].copy()
    sig[:, t_new * patch:] = 0.0
    garbage = sig.copy()
    later = np.concatenate([gm.basis.channels[g] for g in range(1, gm.basis.n_groups)])
    garbage[later, t_new * patch:] = 50.0 * np.random.default_rng(3).standard_normal(
        (later.size, patch))
    with torch.no_grad():
        pos = t_new * gm.basis.n_groups + 0 - 1          # predicts (t_new, group 0)
        a = mixture_mean(gm.head, gm.model(torch.as_tensor(sig[None], dtype=torch.float32))[:, pos])
        c = mixture_mean(gm.head, gm.model(torch.as_tensor(garbage[None], dtype=torch.float32))[:, pos])
    delta = float((a - c).abs().max())
    ok &= check("garbage in later groups cannot move an earlier group's prediction",
                delta == 0.0, f"max |delta| {delta:.3e}")

    print()
    print("=" * 80)
    print("5. Forecast quality vs. baselines, per horizon")
    print("=" * 80)
    n_future = 8
    reps = 6
    errs = {"rollout": np.zeros(n_future), "channel mean": np.zeros(n_future),
            "persistence": np.zeros(n_future), "linear AR ref": np.zeros(n_future)}
    per_rep_first = []
    for r in range(reps):
        full = corpus.batch(meg, 1, (32 + n_future) * patch, np.random.default_rng(100 + r)).x[0]
        context, future = full[:, : 32 * patch], full[:, 32 * patch:]
        fc = rollout(gm.model, gm.head, gm.basis, context, n_future)
        mean_fc = np.repeat(context.mean(axis=1, keepdims=True), n_future * patch, axis=1)
        persist = np.tile(context[:, -patch:], (1, n_future))
        active = np.concatenate([gm.basis.channels[g] for g in range(gm.basis.n_groups - 1)])
        ar_fc = ar_forecast(context, n_future * patch)
        rep_err = {}
        for name, pred in (("rollout", fc), ("channel mean", mean_fc),
                           ("persistence", persist), ("linear AR ref", ar_fc)):
            e = ((pred[active] - future[active]) ** 2).reshape(active.size, n_future, patch)
            rep_err[name] = e.mean(axis=(0, 2))
            errs[name] += rep_err[name] / reps
        per_rep_first.append(rep_err["rollout"][0] / rep_err["channel mean"][0])

    ms = 1000 * patch / corpus.fs
    print(f"  {'horizon':>9}  " + "  ".join(f"{k:>13}" for k in errs))
    for p in range(n_future):
        print(f"  {int((p + 1) * ms):>6} ms  " + "  ".join(f"{errs[k][p]:>13.4f}" for k in errs))
    first = errs["rollout"][0] / errs["channel mean"][0]
    late = errs["rollout"][-1] / errs["channel mean"][-1]
    gain_model = errs["channel mean"][0] - errs["rollout"][0]
    gain_ref = errs["channel mean"][0] - errs["linear AR ref"][0]
    exploitation = gain_model / gain_ref if gain_ref > 0 else float("nan")
    ar_ratio = errs["linear AR ref"][0] / errs["channel mean"][0]
    print(f"  first patch MSE ratio vs channel mean: rollout {first:.3f}   "
          f"linear AR reference {ar_ratio:.3f}")
    print(f"  per-replicate rollout ratios: " + " ".join(f"{v:.3f}" for v in per_rep_first))
    print(f"  EXPLOITATION RATIO (model gain / achievable gain) = {exploitation:.2f}")
    print()
    print(f"  The rehearsal model (1.3M params, ~20 s CPU) captures {100 * exploitation:.0f}% of the")
    print("  first-patch gain a per-channel linear predictor finds -- but that achievable")
    print(f"  gain is itself small ({100 * (1 - ar_ratio):.1f}%). Both facts matter; see the ceiling below.")
    ok &= check("rollout is no worse than the channel-mean forecast", first <= 1.0,
                f"MSE ratio {first:.3f}")

    # Where does this corpus's forecastability actually run out? A first version of
    # this test assumed the sources were "narrow-band oscillations a linear
    # predictor forecasts well" and required the AR reference to beat the mean by
    # 20% at the first patch. It managed 5.7%. The assumption was wrong: the
    # resonances have a 2 Hz half-width, so their coherence time is ~80 ms -- about
    # one patch. Measure the ceiling with the Atlas instead of assuming it.
    from neurocast.atlas import estimators as est
    long_tr = corpus.batch(meg, 1, 40_000, np.random.default_rng(900)).x[0]
    long_te = corpus.batch(meg, 1, 40_000, np.random.default_rng(901)).x[0]
    ch = int(gm.basis.channels[0][0])
    print()
    print("  Atlas forecastability of this corpus (one sensor, spectral null, nats):")
    info = {}
    for h in (1, 4, 16, 64):
        m = est.marginal(long_tr[ch], long_te[ch], h, 64)
        sn = est.spectral_null(long_tr[ch], long_te[ch], h, 64)
        info[h] = m.nll - sn.nll
        print(f"    h = {h:>3} samples ({1000 * h / corpus.fs:5.0f} ms)   I = {info[h]:.4f}")
    # I = -1/2 log(1 - R^2) for a Gaussian predictor, so "more than half the next
    # sample's variance is predictable" means I > -1/2 log(0.5) = 0.347 nats.
    r2_short = 1.0 - float(np.exp(-2.0 * info[1]))
    ok &= check("over half the next sample's variance is predictable (R^2 > 0.5)",
                r2_short > 0.5, f"I(4 ms) = {info[1]:.3f} nats -> R^2 = {r2_short:.3f}")
    ok &= check("and that forecastability is largely gone by one patch (64 ms)",
                info[16] < 0.25 * info[1],
                f"I(64 ms) is {100 * info[16] / info[1]:.0f}% of I(4 ms)")
    print("  So a 64 ms-patch forecast sits near this corpus's predictability horizon.")
    print("  The rollout's small edge is what the ceiling allows, not only what the")
    print("  rehearsal model failed to learn. This is exactly the misreading the Atlas")
    print("  exists to prevent.")

    print()
    print("=" * 80)
    if ok:
        print("GENERATIVE STACK VALIDATED: shared-source corpus, invertible targets, and")
        print("confinement holds in a trained model. Forecast skill is measured, and is low")
        print("for the rehearsal model -- see the exploitation ratio above.")
        return 0
    print("GENERATIVE STACK FAILED validation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
