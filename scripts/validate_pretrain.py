"""Validate the pre-training trainer: schedule, empty groups, and bit-exact resume.

The lost version's trainer claimed "bit-exact resume (verified on CUDA)". This
checks the rebuilt one on CPU, on synthetic recordings shaped like LibriBrain's
(306-channel MEGIN geometry, no peripheral channels):

1. The learning rate warms up linearly for 100 steps, then decays by cosine to
   10% -- the pitch's schedule.
2. The empty peripheral group contributes no target coordinates.
3. Training 6 steps straight equals training 3, checkpointing, loading into a
   *fresh* trainer and training 3 more -- every parameter bit-identical, for both
   arms. A resume that silently re-draws batches or re-seeds masks fails this.
4. The visible-position term is reported and changes the masked loss.
5. The masked-collapse diagnostic (``audit/collapse.py``) on targets with a known
   answer: per-group AR(1) patches, phi = 0.8, so one patch hidden between
   visible neighbours is linearly predictable with R^2 = 2 phi^2 / (1 + phi^2)
   = 0.78. A constant predictor is flagged collapsed, a predictor that reads its
   input is not, and under the pitch's 1-s block masks the same-mask reference
   falls far below the isolated one -- which is why both are reported.
6. The path for tens of hours: sessions memory-mapped time-major from disk give
   the same batches as in RAM; prefetching on background threads changes no
   parameter; whole held-out sessions never reach a training batch; the
   ``.best.pt`` checkpoint holds the weights of the held-out minimum.
7. The latent arms (``jepa``, ``ar_latent``) and the calibration guard: resume is
   bit-exact for both, teacher included; the teacher moves by EMA and stays apart
   from the student; their ``.best.pt`` is the latest evaluation, with the collapse
   monitor logged; the guard skips a window holding a planted 1,000x artifact
   (unguarded, that window shrinks every other window's whitened targets ~20x),
   and with nothing to reject it leaves the run bit-identical to an unguarded one.
8. The forecaster with a spectra-conditioned head: its first loss equals the
   plain forecaster's (the map starts at zero); resume is bit-exact, map
   included; after training, the head reads the condition (swapping two
   recordings' profiles changes the loss); and a recording's profile comes from
   its training span only (a planted artifact in its held-out tail leaves it
   unchanged).

Run:  .venv/Scripts/python.exe scripts/validate_pretrain.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurocast.audit.collapse import diagnose, isolated_r2, neighbour_r2  # noqa: E402
from neurocast.train.pretrain import (  # noqa: E402
    PretrainConfig,
    Pretrainer,
    Recording,
    _lr_lambda,
    open_cached,
)
from validate_tokenizer import megin_montage  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


def main() -> int:
    ok = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    montage = megin_montage(n_sites=12, with_peripherals=False)
    rng = np.random.default_rng(0)
    recs = [Recording(f"r{i}", rng.standard_normal((len(montage), 6000)).astype(np.float16), 5000)
            for i in range(2)]

    print("=" * 78)
    print("1. Learning-rate schedule")
    print("=" * 78)
    cfg = PretrainConfig(steps=4000)
    f = _lr_lambda(cfg)
    ok &= check("warm-up is linear over 100 steps", abs(f(0) - 0.01) < 1e-9 and abs(f(99) - 1.0) < 1e-9)
    ok &= check("cosine decay ends at 10%", abs(f(3999) - 0.10) < 1e-3, f"{f(3999):.4f}")
    ok &= check("halfway through the decay it is 55%", abs(f(100 + 1950) - 0.55) < 1e-3)

    tmp = Path(tempfile.mkdtemp(prefix="neurocast-pretrain-"))
    try:
        for arm, extra in (("ar_lik", {}), ("masked", {"visible_weight": 1.0})):
            print()
            print("=" * 78)
            print(f"2-4. Arm {arm}")
            print("=" * 78)
            base = dict(arm=arm, rung="tiny", n_patch=8, batch=2, steps=6, rank=8, warmup=2,
                        eval_every=1000, device=None, **extra)
            base.pop("device")
            mk = lambda: Pretrainer(PretrainConfig(**base), montage, recs, device="cpu")  # noqa: E731

            straight = mk()
            if arm == "ar_lik":
                cm = straight.basis.component_mask()
                ok &= check("no peripheral channels -> group 4 has no target coordinates",
                            not cm[4].any() and bool(cm[:4].any()))
                # Whitened targets must be O(1). A first version calibrated the PCA on
                # 64 copies of ONE window (a re-seeded generator per draw), whitened a
                # zero-variance direction, and produced targets around 1e5.
                t = straight.basis.encode(straight.batch(0).cpu())
                real = straight.basis.position_mask(8).expand_as(t)
                sd = float(t[real].std())
                ok &= check("whitened targets are O(1)", 0.3 < sd < 3.0, f"sd {sd:.3f}")
            straight.fit(verbose=False)

            ck = tmp / f"{arm}.pt"
            first = mk()
            first.fit(ck, stop_at=3, verbose=False)
            first.save(ck)
            second = mk()
            second.fit(ck, verbose=False)
            same = all(torch.equal(a, b) for a, b in zip(straight.model.state_dict().values(),
                                                         second.model.state_dict().values()))
            same &= all(torch.equal(a, b) for a, b in zip(straight.head.state_dict().values(),
                                                          second.head.state_dict().values()))
            ok &= check(f"{arm}: 6 straight steps == 3 + checkpoint + fresh trainer + 3",
                        same and second.step == 6,
                        "bit-identical parameters" if same else "parameters differ")
            if arm == "masked":
                parts = straight.log[-1]
                ok &= check("visible-position term is reported alongside the hidden error",
                            "visible_mse" in parts and "hidden_mse" in parts,
                            f"hidden {parts.get('hidden_mse', float('nan')):.3f}  "
                            f"visible {parts.get('visible_mse', float('nan')):.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 78)
    print("5. Masked-collapse diagnostic on targets with a known answer")
    print("=" * 78)
    ok &= collapse_checks()

    print()
    print("=" * 78)
    print("6. Memory-mapped sessions, prefetch, held-out sessions, best checkpoint")
    print("=" * 78)
    ok &= scale_checks(montage)

    print()
    print("=" * 78)
    print("7. Latent arms and the calibration guard")
    print("=" * 78)
    ok &= latent_checks(montage)

    print()
    print("=" * 78)
    print("8. Forecasting with a spectra-conditioned head")
    print("=" * 78)
    ok &= condition_checks(montage)

    print()
    print("=" * 78)
    if ok:
        print("PRETRAINER VALIDATED: the pitch's schedule, empty groups excluded,")
        print("resume is bit-exact for both arms, the collapse diagnostic separates a")
        print("collapsed predictor from one that reads its input, and the memory-mapped,")
        print("prefetched path trains exactly as in RAM.")
        return 0
    print("PRETRAINER FAILED validation.")
    return 1


def scale_checks(montage) -> bool:
    ok = True
    rng = np.random.default_rng(3)
    names = [f"MEG{i:04d}" for i in range(len(montage))]
    arrays = {f"s{i}": rng.standard_normal((len(montage), 3000 + 500 * i)).astype(np.float16)
              for i in range(4)}
    tmp = Path(tempfile.mkdtemp(prefix="neurocast-scale-"))
    try:
        for k, x in arrays.items():                 # what cache_recordings writes
            np.save(tmp / f"{k}.npy", np.ascontiguousarray(x.T))
            (tmp / f"{k}.json").write_text(json.dumps({"story": [0, x.shape[1]], "sfreq": 250.0,
                                                       "channel_names": names}))
        keys = sorted(arrays)
        held = ["s3"]
        mapped = open_cached(tmp, keys, heldout=held)
        in_ram = [Recording(k, arrays[k], 0 if k in held else arrays[k].shape[1]) for k in keys]
        base = dict(arm="ar_lik", rung="tiny", n_patch=8, batch=3, steps=6, rank=8, warmup=2,
                    eval_every=2, eval_batches=2)
        a = Pretrainer(PretrainConfig(**base), montage, mapped, device="cpu")
        b = Pretrainer(PretrainConfig(**base), montage, in_ram, device="cpu")
        same = all(np.array_equal(a._batch_np(s, t), b._batch_np(s, t))
                   for s in range(20) for t in (True, False))
        ok &= check("memory-mapped time-major batches == in-RAM batches", same)

        # Held-out sessions: only s3 can supply a held-out window, and it cannot
        # supply a training one.
        train_keys = {r.key for r in a._pool[True]}
        held_keys = {r.key for r in a._pool[False]}
        ok &= check("held-out sessions never reach the training pool",
                    train_keys == {"s0", "s1", "s2"} and held_keys == {"s3"},
                    f"train {sorted(train_keys)}  held-out {sorted(held_keys)}")

        plain = Pretrainer(PretrainConfig(**base), montage, mapped, device="cpu")
        plain.fit(verbose=False)
        fetched = Pretrainer(PretrainConfig(**base, prefetch=3), montage, mapped, device="cpu")
        ck = tmp / "run.pt"
        fetched.fit(ck, verbose=False)
        same = all(torch.equal(x, y) for x, y in zip(plain.model.state_dict().values(),
                                                     fetched.model.state_dict().values()))
        ok &= check("prefetch on 3 threads == no prefetch, bit for bit", same)

        held = [e["heldout"] for e in fetched.log if "heldout" in e]
        best = Pretrainer.best_path(ck)
        s = torch.load(best, map_location="cpu", weights_only=False)
        ok &= check(".best.pt is the held-out minimum, weights only",
                    best.exists() and "opt" not in s
                    and s["log"][-1]["heldout"] == min(held),
                    f"held-out {[round(h, 4) for h in held]} -> best at step {s['step']}")
        fresh = Pretrainer(PretrainConfig(**base), montage, mapped, device="cpu")
        fresh.load(best)
        ok &= check("a .best.pt checkpoint loads for evaluation", fresh.step == s["step"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


def latent_checks(montage) -> bool:
    ok = True
    rng = np.random.default_rng(5)
    recs = [Recording(f"r{i}", rng.standard_normal((len(montage), 6000)).astype(np.float16), 5000)
            for i in range(2)]
    tmp = Path(tempfile.mkdtemp(prefix="neurocast-latent-"))
    try:
        for arm in ("jepa", "ar_latent"):
            base = dict(arm=arm, rung="tiny", n_patch=8, batch=2, steps=6, rank=8, warmup=2,
                        eval_every=3, eval_batches=1)
            mk = lambda: Pretrainer(PretrainConfig(**base), montage, recs, device="cpu")  # noqa: E731
            straight = mk()
            ok &= check(f"{arm}: trunk is {'bidirectional' if arm == 'jepa' else 'causal'}",
                        straight.model.causal == (arm == "ar_latent"))
            t0 = [p.detach().clone() for p in straight.teacher.teacher.parameters()]
            straight.fit(tmp / f"{arm}_straight.pt", verbose=False)
            moved = any(not torch.equal(a, b) for a, b in zip(t0, straight.teacher.teacher.parameters()))
            apart = any(not torch.equal(a, b) for a, b in zip(straight.model.parameters(),
                                                              straight.teacher.teacher.parameters()))
            ok &= check(f"{arm}: the teacher moves by EMA and stays apart from the student",
                        moved and apart)
            ck = tmp / f"{arm}.pt"
            first = mk()
            first.fit(ck, stop_at=4, verbose=False)
            second = mk()
            second.fit(ck, verbose=False)
            pairs = [(straight.model, second.model), (straight.head, second.head),
                     (straight.teacher.teacher, second.teacher.teacher)]
            same = all(torch.equal(a, b) for m1, m2 in pairs
                       for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
            ok &= check(f"{arm}: 6 straight steps == 4 + checkpoint + fresh trainer + 2, teacher included",
                        same and second.step == 6)
            # A finished run slimmed to save disk (no optimiser, scheduler or teacher) still
            # loads for evaluation; a resumable checkpoint missing its teacher does not.
            full = torch.load(ck, map_location="cpu", weights_only=False)
            torch.save({k: v for k, v in full.items() if k not in ("opt", "sched", "teacher")},
                       tmp / f"{arm}_slim.pt")
            ev = mk()
            ev.load(tmp / f"{arm}_slim.pt")
            torch.save({k: v for k, v in full.items() if k != "teacher"}, tmp / f"{arm}_broken.pt")
            try:
                mk().load(tmp / f"{arm}_broken.pt")
                refused = False
            except KeyError:
                refused = True
            ok &= check(f"{arm}: a slimmed checkpoint loads for evaluation; a resumable one without "
                        "its teacher is refused",
                        refused and ev.step == 6 and all(torch.equal(a, b) for a, b in zip(
                            ev.model.state_dict().values(), second.model.state_dict().values())))
            s = torch.load(Pretrainer.best_path(tmp / f"{arm}_straight.pt"), map_location="cpu",
                           weights_only=False)
            mon = straight.log[-1].get("monitor", {})
            ok &= check(f"{arm}: .best.pt is the latest evaluation; collapse monitor logged",
                        s["step"] == 6 and "stable_rank_frac" in mon,
                        f"step {s['step']}, stable rank {mon.get('stable_rank_frac', float('nan')):.2f}")

        # The guard. Plant a 1,000x artifact in the window the unguarded run's first
        # calibration draw reads, and nowhere else.
        base = dict(arm="ar_lik", rung="tiny", n_patch=8, batch=2, steps=2, rank=8, warmup=2,
                    eval_every=1000)
        clean = Pretrainer(PretrainConfig(**base), montage, recs, device="cpu")
        r0, st0, _ = clean.calibration_draws()[0]
        bad = [Recording(r.key, r.data.copy(), r.train_stop) for r in recs]
        hit = next(r for r in bad if r.key == r0.key)
        hit.data[:, st0:st0 + clean.n_times] = (
            1000.0 * rng.standard_normal((len(montage), clean.n_times))).astype(np.float16)
        x0 = clean.batch(0).cpu()                      # clean windows, encoded by each basis
        width = lambda tr: float(tr.basis.encode(x0)[  # noqa: E731
            tr.basis.position_mask(8).expand(2, -1, -1)].std())
        unguarded = Pretrainer(PretrainConfig(**base), montage, bad, device="cpu")
        guarded = Pretrainer(PretrainConfig(**base, calib_guard=100.0), montage, bad, device="cpu")
        ok &= check("unguarded, the artifact window corrupts the whitened targets",
                    width(unguarded) < 0.3 * width(clean),
                    f"target sd {width(unguarded):.3f} vs clean {width(clean):.3f}")
        ok &= check("the guard rejects that window and the targets are O(1) again",
                    guarded.calib_rejected >= 1 and 0.3 < width(guarded) < 3.0,
                    f"{guarded.calib_rejected} rejected, target sd {width(guarded):.3f}")
        noop = Pretrainer(PretrainConfig(**base, calib_guard=100.0), montage, recs, device="cpu")
        clean.fit(verbose=False)
        noop.fit(verbose=False)
        same = noop.calib_rejected == 0 and all(
            torch.equal(a, b) for a, b in zip(clean.model.state_dict().values(),
                                              noop.model.state_dict().values()))
        ok &= check("with nothing to reject, a guarded run is bit-identical to an unguarded one", same)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


def condition_checks(montage) -> bool:
    ok = True
    rng = np.random.default_rng(9)
    recs = [Recording(f"r{i}", (rng.standard_normal((len(montage), 6000)) * (1 + i)).astype(np.float16),
                      5000) for i in range(3)]
    base = dict(arm="ar_lik", rung="tiny", n_patch=8, batch=2, steps=6, rank=8, warmup=2, eval_every=1000)
    tmp = Path(tempfile.mkdtemp(prefix="neurocast-cond-"))
    try:
        plain = Pretrainer(PretrainConfig(**base), montage, recs, device="cpu")
        mk = lambda: Pretrainer(PretrainConfig(**base, condition="spectra"), montage, recs,  # noqa: E731
                                device="cpu")
        cond = mk()
        x = plain.batch(0)
        cond._keys = cond._batch_keys(0, True)
        l0, l1 = float(plain.loss(x, 0)[0]), float(cond.loss(x, 0)[0])
        ok &= check("the conditioned forecaster starts as the plain one", abs(l0 - l1) < 1e-6,
                    f"{l0:.6f} vs {l1:.6f}")
        straight = mk()
        straight.fit(verbose=False)
        ck = tmp / "cond.pt"
        first = mk()
        first.fit(ck, stop_at=3, verbose=False)
        second = mk()
        second.fit(ck, verbose=False)
        same = all(torch.equal(a, b) for m1, m2 in ((straight.model, second.model),
                                                     (straight.head, second.head),
                                                     (straight.cond_proj, second.cond_proj))
                   for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
        ok &= check("conditioned: 6 straight steps == 3 + checkpoint + fresh trainer + 3, map included",
                    same and second.step == 6)
        with torch.no_grad():                           # make the map large enough to matter
            straight.cond_proj.weight.normal_(0, 1.0)
        straight._keys = straight._batch_keys(0, True)
        la = float(straight.loss(x, 0)[0])
        straight.cond["r0"], straight.cond["r1"] = straight.cond["r1"], straight.cond["r0"]
        lb = float(straight.loss(x, 0)[0])
        ok &= check("the head reads the condition: swapping two recordings' profiles changes the loss",
                    abs(la - lb) > 1e-4, f"{la:.4f} -> {lb:.4f}")
        tail = [Recording(r.key, r.data.copy(), r.train_stop) for r in recs]
        tail[0].data[:, 5500:5600] = 1000.0
        a = mk().cond["r0"]
        b = Pretrainer(PretrainConfig(**base, condition="spectra"), montage, tail, device="cpu").cond["r0"]
        ok &= check("a recording's profile comes from its training span only", torch.equal(a, b))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


def collapse_checks() -> bool:
    ok = True
    rng = np.random.default_rng(1)
    b, t, g, k, phi = 64, 64, 5, 8, 0.8
    x = np.zeros((b, t, g, k))
    x[:, 0] = rng.standard_normal((b, g, k))
    for i in range(1, t):                  # stationary AR(1), unit variance
        x[:, i] = phi * x[:, i - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal((b, g, k))
    y = x.reshape(b, t * g, k)
    hid = np.zeros((b, t), bool)            # two 1-s blocks (16 patches), every group
    for i in range(b):
        for s0 in rng.choice(np.arange(0, t - 16, 16), 2, replace=False):
            hid[i, s0:s0 + 16] = True
    hid = np.repeat(hid, g, axis=1)

    want = 2 * phi ** 2 / (1 + phi ** 2)
    iso = isolated_r2(y, g)
    ok &= check("isolated R^2 matches the AR(1) answer", abs(iso - want) < 0.03,
                f"{iso:.3f} vs {want:.3f}")
    same = neighbour_r2(y, hid, g)
    ok &= check("1-s block masks leave the same-mask reference far below it", same < iso - 0.3,
                f"same mask {same:+.3f}")

    flat = np.zeros_like(y)
    r = diagnose(flat, flat, y, hid, g)
    ok &= check("a constant predictor is flagged collapsed", r.collapsed, r.summary())
    noisy = y + 0.5 * rng.standard_normal(y.shape)
    swapped = np.roll(y, 1, axis=0) + 0.5 * rng.standard_normal(y.shape)
    r = diagnose(noisy, swapped, y, hid, g)
    ok &= check("a predictor that reads its input is not", not r.collapsed and r.model_r2 > 0.5,
                r.summary())
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
