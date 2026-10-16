"""Removing person information while fine-tuning, judged on people never seen (README 2.8).

Forecasting features name a listener as reliably as the raw signal does (section
2.6). Other fields remove the equivalent nuisance -- the speaker in speech, the
subject in EEG -- in a few ways; three cheap ones are tested here, on speech vs
silence, fine-tuned end to end (section 2.7's recipe) across the 32 broad
listeners:

* people: 20 to train on (chapter 11), 4 to choose the learning rate and step
  (chapter 12), 8 to test (chapter 12) -- a fixed random split by person;
* arms, each from section 2.2's forecasting checkpoints (seeds 0-2):
  - ``plain``: speech loss only;
  - ``adversarial``: plus a listener classifier on the pooled features behind a
    gradient-reversal layer, its weight ramped 0 -> 1 (Ganin et al.'s schedule);
    the EEG version is Ozdenizci et al. 2020;
  - ``instnorm``: every window z-scored per channel before the model, which
    strips per-channel level and gain (the voice-conversion trick of Chou et al.
    2019; the 2025 LibriBrain teams' "per-window normalisation");
  - and ``masked_plain``: ``plain`` from section 2.2's masked + visible
    checkpoints, the arm with the least person information so far;
* after training, LEACE (``eval/yardstick.leace``) fitted on the 20 training
  people's chapter-11 features, applied to every arm.

Measured on chapter 12: speech macro-F1 and AUC for the 8 unseen people
(primary) and the 20 seen ones; the identity audit among the unseen people and
among the seen ones; and cross-chapter identification -- the seen people's
chapter-11 windows as references, their chapter-12 windows to name.

    .venv/Scripts/python.exe     scripts/run_identity_removal.py --stage prepare   # cache (CPU)
    .venv-gpu/Scripts/python.exe scripts/run_identity_removal.py --stage train     # resumable
    .venv/Scripts/python.exe     scripts/run_identity_removal.py --stage report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from neurocast.data.paths import dataset_dir  # noqa: E402

OUT = ROOT / "runs" / "identity_removal"
BROAD = [str(i) for i in range(1, 33)]
ARMS = ("plain", "adversarial", "instnorm", "masked_plain")
#: README 2.14: plain fine-tuning from starts pre-trained on many people (seeds 0-1).
MULTI_ARMS = ("plain_multi", "masked_plain_multi")
START = {"plain": "ar_lik", "adversarial": "ar_lik", "instnorm": "ar_lik",
         "masked_plain": "masked_visible", "plain_multi": "ar_lik_multi",
         "masked_plain_multi": "masked_visible_multi"}
SEEDS = (0, 1, 2)
LRS = (1e-4, 1e-3)
STEPS, WARMUP, EVAL_EVERY, BATCH = 2000, 100, 250, 8
N_PATCH, KEEP, PATCH, N_TIMES = 64, (16, 48), 16, 1024
PER_PERSON = 64                       # identity windows per person and chapter
LAMBDA_MAX = 1.0


def cache_dir() -> Path:
    return dataset_dir("derived") / "pretrain_cache" / "broad-session-scalar"


def split() -> dict:
    perm = [BROAD[i] for i in np.random.default_rng(0).permutation(len(BROAD))]
    return {"train": sorted(perm[:20], key=int), "val": sorted(perm[20:24], key=int),
            "test": sorted(perm[24:], key=int)}


def runs(ses: str):
    from neurocast.data.libribrain import find_runs

    return {r.subject: r for r in find_runs(task="Sherlock1", sessions={ses}, subjects=set(BROAD),
                                            require_meg=True)}


def stage_prepare(args) -> int:
    from neurocast.train.pretrain import cache_recordings

    for ses in ("11", "12"):
        rs = list(runs(ses).values())
        print(f"chapter {ses}: {len(rs)} listeners", flush=True)
        cache_recordings(rs, cache_dir())
    print(json.dumps(split()))
    return 0


class Person:
    """One listener's chapter: memory-mapped, session-normalised, with speech labels."""

    def __init__(self, run) -> None:
        from run_bakeoff_real import session_labels

        from neurocast.train.pretrain import open_cached

        key = "_".join(run.key)
        self.rec = open_cached(cache_dir(), [key])[0]
        self.n_patches = self.rec.n_samples // PATCH
        self.labels = session_labels(run.events, self.n_patches, self.rec.story[0] / 250.0)["speech"]

    def window(self, s: int) -> np.ndarray:
        return self.rec.window(PATCH * s, N_TIMES)


def load(people, ses):
    rs = runs(ses)
    return {p: Person(rs[p]) for p in people if p in rs}


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def instnorm(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)


class Model(nn.Module):
    def __init__(self, backbone, n_people: int, arm: str) -> None:
        from run_finetune import Head

        super().__init__()
        self.backbone, self.arm = backbone, arm
        self.head = Head(backbone.d_model)
        self.adv = nn.Sequential(nn.Linear(backbone.d_model, 256), nn.GELU(),
                                 nn.Linear(256, n_people)) if arm == "adversarial" else None

    def features(self, x):
        if self.arm == "instnorm":
            x = instnorm(x)
        from neurocast.tokenizer.perceiver import ar_unflatten

        h = self.backbone(x).float()                               # (B, N, d)
        g = ar_unflatten(h, self.backbone.n_groups).mean(2)        # (B, 64, d)
        return g, h.mean(1)                                        # patches, pooled (as section 2.2)

    def forward(self, x):
        g, pooled = self.features(x)
        return self.head(g), pooled


def build(arm, seed, n_people, montage, device):
    from run_bakeoff_real import load_model, specs

    backbone, _ = load_model(specs()[f"{START[arm]}_s{seed}"], montage, device)
    torch.manual_seed(seed + 1000)
    return Model(backbone, n_people, arm).to(device)


def batch(train: dict, seed: int, step: int):
    rng = np.random.default_rng([seed, step, 11])
    people = list(train)
    xs, ys, ps = [], [], []
    for _ in range(BATCH):
        i = int(rng.integers(len(people)))
        p = train[people[i]]
        s = int(rng.integers(0, p.n_patches - N_PATCH + 1))
        xs.append(p.window(s))
        ys.append(p.labels[s + KEEP[0]:s + KEEP[1]])
        ps.append(i)
    return (np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), np.array(ps))


@torch.no_grad()
def predict(model, people: dict, device, amp):
    from run_finetune import scores

    model.eval()
    lo, la = [], []
    for p in people.values():
        starts = list(range(0, p.n_patches - N_PATCH + 1, KEEP[1] - KEEP[0]))
        for i in range(0, len(starts), 16):
            st = starts[i:i + 16]
            xb = torch.as_tensor(np.stack([p.window(s) for s in st]).astype(np.float32), device=device)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                lo.append(model(xb)[0].float().cpu().numpy().ravel())
            la.append(np.concatenate([p.labels[s + KEEP[0]:s + KEEP[1]] for s in st]))
    model.train()
    return scores(np.concatenate(lo), np.concatenate(la).astype(int))


@torch.no_grad()
def pooled(model, people: dict, device, amp, seed: int = 0):
    """``PER_PERSON`` random windows per person (a fixed draw): pooled features and owners."""
    model.eval()
    f, who = [], []
    for name, p in people.items():
        rng = np.random.default_rng([seed, int(name)])
        st = rng.integers(0, p.n_patches - N_PATCH + 1, PER_PERSON)
        for i in range(0, len(st), 16):
            xb = torch.as_tensor(np.stack([p.window(int(s)) for s in st[i:i + 16]]).astype(np.float32),
                                 device=device)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                f.append(model.features(xb)[1].float().cpu().numpy())
        who += [name] * PER_PERSON
    model.train()
    return np.concatenate(f), np.array(who)


def cross_chapter(f11, w11, f12, w12) -> float:
    """Name each chapter-12 window by the nearest chapter-11 person mean (standardised on 11)."""
    mu, sd = f11.mean(0), f11.std(0) + 1e-6
    a, b = (f11 - mu) / sd, (f12 - mu) / sd
    names = np.unique(w11)
    cents = np.stack([a[w11 == n].mean(0) for n in names])
    pred = names[np.argmin(((b[:, None] - cents[None]) ** 2).sum(-1), axis=1)]
    return float((pred == w12).mean())


def evaluate(model, data, device, amp) -> dict:
    from run_pretrain_pilot import identity

    from neurocast.eval.yardstick import leace

    out = {"speech_test": predict(model, data["test12"], device, amp),
           "speech_seen": predict(model, data["train12"], device, amp)}
    f11, w11 = pooled(model, data["train11"], device, amp)
    f12s, w12s = pooled(model, data["train12"], device, amp)
    f12u, w12u = pooled(model, data["test12"], device, amp)
    erase = leace(f11, w11)
    out["identity_unseen"] = identity(f12u, w12u)
    out["identity_seen"] = identity(f12s, w12s)
    out["cross_chapter"] = cross_chapter(f11, w11, f12s, w12s)
    out["leace"] = {"identity_unseen": identity(erase(f12u), w12u),
                    "identity_seen": identity(erase(f12s), w12s),
                    "cross_chapter": cross_chapter(erase(f11), w11, erase(f12s), w12s)}
    return out


def fit_one(arm, seed, lr, data, montage, device, amp, n_people):
    model = build(arm, seed, n_people, montage, device)
    labs = np.concatenate([p.labels for p in data["train11"].values()])
    pos = float(labs.mean())
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=device))
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / WARMUP if s < WARMUP else
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / (STEPS - WARMUP))))
    best, curve, t0 = None, [], time.time()
    model.train()
    for step in range(1, STEPS + 1):
        xb, yb, pb = batch(data["train11"], seed, step)
        xb, yb = torch.as_tensor(xb, device=device), torch.as_tensor(yb, device=device)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logit, pool = model(xb)
        loss = bce(logit.float(), yb)
        parts = {"speech": float(loss.detach())}
        if model.adv is not None:
            lam = LAMBDA_MAX * (2 / (1 + math.exp(-10 * step / STEPS)) - 1)
            adv = ce(model.adv(GradReverse.apply(pool, lam)), torch.as_tensor(pb, device=device))
            loss = loss + adv
            parts["adversary"] = float(adv.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0:
            v = predict(model, data["val12"], device, amp)
            curve.append((step, v["f1"], parts))
            if best is None or v["f1"] > best[1]:
                best = (step, v["f1"], {k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
            print(f"    lr {lr:.0e} step {step:4d}  {parts}  val F1 {v['f1']:.3f}  "
                  f"({(time.time() - t0) / step:.2f} s/step)", flush=True)
    return best, curve


def stage_train(args) -> int:
    from run_pretrain_pilot import montage_for
    from run_pretrain_scale import deep_runs

    if args.gpu_mem_frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
    device = torch.device(args.device)
    amp = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) \
        if device.type == "cuda" else None
    OUT.mkdir(parents=True, exist_ok=True)
    sp = split()
    data = {"train11": load(sp["train"], "11"), "train12": load(sp["train"], "12"),
            "val12": load(sp["val"], "12"), "test12": load(sp["test"], "12")}
    if args.people:                                    # smoke tests only
        data = {k: dict(list(v.items())[:args.people]) for k, v in data.items()}
        for v in data.values():
            for q in v.values():
                q.n_patches = min(q.n_patches, args.max_patches)
                q.labels = q.labels[:q.n_patches]
    n_people = len(data["train11"])
    print({k: len(v) for k, v in data.items()}, flush=True)
    montage = montage_for(deep_runs()[0])

    arms = args.arms or ARMS
    multi = any(a in MULTI_ARMS for a in arms)
    for seed in (args.seeds or SEEDS):
        f = OUT / f"pretrained{'_multi' if multi else ''}_s{seed}.json"
        if not f.exists():                             # the starting points, before any fine-tuning
            res = {}
            pairs = ((("ar_lik_multi", "plain_multi"), ("masked_visible_multi", "masked_plain_multi"))
                     if multi else (("ar_lik", "plain"), ("masked_visible", "masked_plain")))
            for start, arm in pairs:
                m = build(arm, seed, n_people, montage, device)
                f11, w11 = pooled(m, data["train11"], device, amp)
                f12s, w12s = pooled(m, data["train12"], device, amp)
                f12u, w12u = pooled(m, data["test12"], device, amp)
                from run_pretrain_pilot import identity
                res[start] = {"identity_unseen": identity(f12u, w12u), "identity_seen": identity(f12s, w12s),
                              "cross_chapter": cross_chapter(f11, w11, f12s, w12s)}
                del m
            f.write_text(json.dumps(res, indent=2, default=float))
        for arm in arms:
            f = OUT / f"{arm}_s{seed}.json"
            if f.exists():
                continue
            print(f"  {arm}_s{seed}", flush=True)
            res, chosen = {"arm": arm, "seed": seed, "split": sp, "lrs": {}}, None
            for lr in LRS:
                best, curve = fit_one(arm, seed, lr, data, montage, device, amp, n_people)
                res["lrs"][str(lr)] = {"curve": curve, "best_step": best[0], "val_f1": best[1]}
                if chosen is None or best[1] > chosen[2]:
                    chosen = (lr, best[0], best[1], best[2])
                torch.cuda.empty_cache()
            model = build(arm, seed, n_people, montage, device)
            model.load_state_dict(chosen[3])
            res.update({"lr": chosen[0], "best_step": chosen[1], "val_f1": chosen[2],
                        **evaluate(model, data, device, amp)})
            del model
            torch.cuda.empty_cache()
            f.write_text(json.dumps(res, indent=2, default=float))
            print(f"  {arm}_s{seed}: unseen F1 {res['speech_test']['f1']:.3f}, identity unseen "
                  f"{res['identity_unseen']['ratio']:.1f}x (probe {res['identity_unseen']['probe']:.2f}), "
                  f"cross-chapter {res['cross_chapter']:.2f}; after LEACE "
                  f"{res['leace']['identity_unseen']['ratio']:.1f}x", flush=True)
    return 0


def verdicts(rows: dict) -> dict:
    """The rules fixed in README 2.8 before any run."""
    out: dict = {}

    def get(arm, s, path):
        r = rows.get(f"{arm}_s{s}")
        for k in path:
            if r is None:
                return None
            r = r.get(k)
        return r

    for arm in ("adversarial", "instnorm", "masked_plain"):
        sd = [s for s in SEEDS if get(arm, s, ["speech_test"]) and get("plain", s, ["speech_test"])]
        if len(sd) < 2:
            out[arm] = "not enough seeds"
            continue
        d = [get(arm, s, ["speech_test", "f1"]) - get("plain", s, ["speech_test", "f1"]) for s in sd]
        m = float(np.mean(d))
        dec = ("equal" if abs(m) < 0.01 else "helps" if all(x > 0 for x in d)
               else "hurts" if all(x < 0 for x in d) else "no consistent difference")
        ir = [get(arm, s, ["identity_unseen", "ratio"]) / get("plain", s, ["identity_unseen", "ratio"])
              for s in sd]
        im = [get(arm, s, ["identity_unseen", "ratio"]) / get("masked_plain", s, ["identity_unseen", "ratio"])
              for s in sd if get("masked_plain", s, ["identity_unseen"])]
        out[arm] = {"decoding_unseen": dec, "f1_differences": d, "identity_over_plain": ir,
                    "reduces_identity": all(x <= 0.75 for x in ir),
                    "identity_over_masked": im,
                    "below_masked": bool(im) and all(x < 1 for x in im) if arm != "masked_plain" else None}
        if arm != "masked_plain":
            out[arm]["answer"] = bool(out[arm]["below_masked"]) and dec in ("equal", "helps")
    le = [get("plain", s, ["leace", "identity_unseen", "ratio"]) / get("plain", s, ["identity_unseen", "ratio"])
          for s in SEEDS if get("plain", s, ["leace"])]
    out["leace_generalises"] = {"ratios": le, "verdict": bool(le) and all(x <= 0.75 for x in le)}

    # README 2.14: pre-trained on many people against pre-trained on one, seeds 0-1.
    pre = {s: json.loads((OUT / f"pretrained_multi_s{s}.json").read_text()) for s in (0, 1)
           if (OUT / f"pretrained_multi_s{s}.json").exists()}
    one = {s: json.loads((OUT / f"pretrained_s{s}.json").read_text()) for s in (0, 1)
           if (OUT / f"pretrained_s{s}.json").exists()}
    for arm, base, start, start_base in (("plain_multi", "plain", "ar_lik_multi", "ar_lik"),
                                         ("masked_plain_multi", "masked_plain", "masked_visible_multi",
                                          "masked_visible")):
        sd = [s for s in (0, 1) if get(arm, s, ["speech_test"]) and get(base, s, ["speech_test"])]
        if len(sd) < 2:
            out[arm] = "not enough seeds"
            continue
        d = [get(arm, s, ["speech_test", "f1"]) - get(base, s, ["speech_test", "f1"]) for s in sd]
        m = float(np.mean(d))
        ratio = [get(arm, s, ["identity_unseen", "ratio"]) / get(base, s, ["identity_unseen", "ratio"])
                 for s in sd]
        before = [pre[s][start]["identity_unseen"]["ratio"] / one[s][start_base]["identity_unseen"]["ratio"]
                  for s in sd if s in pre and s in one]
        out[arm] = {"f1_differences": d,
                    "decoding_unseen": ("equal" if abs(m) < 0.01 else "helps" if all(x > 0 for x in d)
                                        else "hurts" if all(x < 0 for x in d) else "no consistent difference"),
                    "identity_over_single_person": ratio,
                    "identity": ("reduces" if all(x <= 0.75 for x in ratio) else
                                 "raises" if all(x >= 1.25 for x in ratio) else "no consistent change"),
                    "before_fine_tuning_identity_over_single_person": before,
                    "before_fine_tuning": ("reduces" if before and all(x <= 0.75 for x in before) else
                                           "raises" if before and all(x >= 1.25 for x in before)
                                           else "no consistent change"),
                    "cross_chapter": [get(arm, s, ["cross_chapter"]) for s in sd],
                    "cross_chapter_single_person": [get(base, s, ["cross_chapter"]) for s in sd]}
    return out


def stage_report(args) -> int:
    rows = {f.stem: json.loads(f.read_text()) for f in sorted(OUT.glob("*.json")) if f.stem != "verdicts"}
    print("| Run | Unseen F1 / AUC | Seen F1 | Identity unseen (probe) | Seen | Cross-chapter | LEACE: unseen, cross-chapter |")
    print("|---|---|---|---|---|---|---|")
    for k, r in rows.items():
        if k.startswith("pretrained"):
            for start, v in r.items():
                print(f"| {start} before fine-tuning (s{k[-1]}) | - | - | {v['identity_unseen']['ratio']:.1f}x "
                      f"({v['identity_unseen']['probe']:.2f}) | {v['identity_seen']['ratio']:.1f}x | "
                      f"{v['cross_chapter']:.2f} | - |")
            continue
        print(f"| {k} | {r['speech_test']['f1']:.3f} / {r['speech_test']['auc']:.3f} | "
              f"{r['speech_seen']['f1']:.3f} | {r['identity_unseen']['ratio']:.1f}x "
              f"({r['identity_unseen']['probe']:.2f}) | {r['identity_seen']['ratio']:.1f}x | "
              f"{r['cross_chapter']:.2f} | {r['leace']['identity_unseen']['ratio']:.1f}x, "
              f"{r['leace']['cross_chapter']:.2f} |")
    v = verdicts(rows)
    (OUT / "verdicts.json").write_text(json.dumps(v, indent=2, default=float))
    print("\nverdicts (README 2.8 rules):")
    for k, x in v.items():
        print(f"  {k}: {x}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("prepare", "train", "report"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.9)
    ap.add_argument("--arms", nargs="*", choices=ARMS + MULTI_ARMS)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--steps", type=int, help="smoke tests only")
    ap.add_argument("--eval-every", type=int, help="smoke tests only")
    ap.add_argument("--people", type=int, help="smoke tests only: people per split")
    ap.add_argument("--max-patches", type=int, default=400, help="smoke tests only")
    args = ap.parse_args()
    global OUT, STEPS, EVAL_EVERY
    if args.out:
        OUT = args.out.resolve()
    STEPS = args.steps or STEPS
    EVAL_EVERY = args.eval_every or EVAL_EVERY
    return {"prepare": stage_prepare, "train": stage_train, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
