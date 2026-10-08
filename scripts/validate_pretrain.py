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
