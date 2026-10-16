"""Does self-supervised pre-training pay off as label efficiency? (README section 2.7)

Section 2.6 scored frozen features with a linear probe, and no objective beat the
raw signal. Pre-training usually pays off differently: a pre-trained network
fine-tuned on few labels beats the same network trained from scratch on them.
This tests that on speech vs silence, the one task where MEG beats every no-brain
row (section 2.6):

* starting points: the best checkpoints of ``ar_lik`` and ``masked_visible``
  (section 2.2) and of ``jepa`` and ``ar_latent`` (section 2.6), seeds 0-2; and the
  same architecture from scratch, causal and bidirectional, seeds 0-2;
* labels: the deep listener's Sherlock1 sessions 1-10 in order, the first 0.5 h or
  all 3.2 h (their MEG is in the pre-training data, their labels never were);
* model: the whole network trained end to end on 4.1-s windows, plus a linear
  head that reads, for each 64-ms patch, the features of the 4 patches after it
  (64-320 ms later, as the section 2.6 probe does); every labelled patch has at
  least 1 s of context on either side within its window;
* recipe: AdamW, batch 8, 2,000 steps, 100 warm-up steps, cosine decay to 10%,
  class-balanced loss, bf16; learning rate 1e-4 or 1e-3, chosen together with the
  step by chapter-11 macro-F1 every 250 steps; scored once on chapter 12. Single
  models, no ensembling, no smoothing.

    .venv-gpu/Scripts/python.exe scripts/run_finetune.py --stage train     # resumable
    .venv/Scripts/python.exe     scripts/run_finetune.py --stage report

One JSON per run in ``runs/finetune/``; finished runs are skipped.
"""

from __future__ import annotations

import argparse
import copy
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

OUT = ROOT / "runs" / "finetune"
INITS = ("scratch_causal", "scratch_bidir", "ar_lik", "masked_visible", "jepa", "ar_latent")
#: The scratch baseline each pre-trained objective is judged against (same causality).
BASELINE = {"ar_lik": "scratch_causal", "ar_latent": "scratch_causal",
            "masked_visible": "scratch_bidir", "jepa": "scratch_bidir"}
TIERS = (0.5, None)                    # hours of labels; None = all of sessions 1-10
SEEDS = (0, 1, 2)
LRS = (1e-4, 1e-3)
STEPS, WARMUP, EVAL_EVERY, BATCH = 2000, 100, 250, 8
N_PATCH, KEEP = 64, (16, 48)           # 4.1-s windows; labelled patches 16-47
LAGS = (1, 2, 3, 4)
PATCH = 16


def run_name(init: str, tier, seed: int) -> str:
    return f"{init}_{'all' if tier is None else f'{tier}h'}_s{seed}"


class Head(nn.Module):
    """For each labelled patch t, a logit from the features of patches t+1..t+4."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(len(LAGS) * d)
        self.lin = nn.Linear(len(LAGS) * d, 1)
        self.register_buffer("idx", torch.arange(*KEEP))

    def forward(self, g: torch.Tensor) -> torch.Tensor:          # g (B, 64, d)
        f = torch.cat([g[:, self.idx + lag] for lag in LAGS], dim=-1)
        return self.lin(self.norm(f)).squeeze(-1)                 # (B, 32)


class Net(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.head = Head(model.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.model.groups(x).float().mean(2))


def load_sessions(sessions, hours=None):
    """``[(x (T, C) float16, speech labels per patch)]`` for sessions, cut to ``hours``."""
    from run_bakeoff_real import read_session, session_labels

    out, need = [], None if hours is None else int(round(hours * 3600 / (PATCH / 250.0)))
    for ses in sessions:
        x, t0, events = read_session(ses)
        p = x.shape[0] // PATCH
        lab = session_labels(events, p, t0)["speech"]
        if need is not None:
            if need < N_PATCH:
                break
            p = min(p, need)
            need -= p
        out.append((x[:p * PATCH], lab[:p]))
        if need is not None and need <= 0:
            break
    return out


def batch(data, seed: int, step: int):
    rng = np.random.default_rng([seed, step, 7])
    w = np.array([len(lab) - N_PATCH + 1 for _, lab in data], float)
    xs, ys = [], []
    for _ in range(BATCH):
        i = int(rng.choice(len(data), p=w / w.sum()))
        x, lab = data[i]
        s = int(rng.integers(0, len(lab) - N_PATCH + 1))
        xs.append(x[PATCH * s:PATCH * (s + N_PATCH)].T)
        ys.append(lab[s + KEEP[0]:s + KEEP[1]])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


@torch.no_grad()
def predict(net, data, device, amp):
    """Logits and labels for every patch 16-47 of windows at a 32-patch stride."""
    net.eval()
    logits, labels = [], []
    for x, lab in data:
        starts = list(range(0, len(lab) - N_PATCH + 1, KEEP[1] - KEEP[0]))
        for i in range(0, len(starts), 16):
            st = starts[i:i + 16]
            xb = torch.as_tensor(np.stack([x[PATCH * s:PATCH * (s + N_PATCH)].T for s in st])
                                 .astype(np.float32), device=device)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                logits.append(net(xb).float().cpu().numpy().ravel())
            labels.append(np.concatenate([lab[s + KEEP[0]:s + KEEP[1]] for s in st]))
    net.train()
    return np.concatenate(logits), np.concatenate(labels).astype(int)


def scores(logit, y) -> dict:
    from neurocast.eval.yardstick import auc, balanced_accuracy, macro_f1

    pred = (logit > 0).astype(int)
    return {"f1": macro_f1(pred, y, 2), "bacc": balanced_accuracy(pred, y, 2), "auc": auc(logit, y)}


def build(init: str, seed: int, montage, device):
    from run_bakeoff_real import load_model, specs

    from neurocast.models.neurocast import NeuroCast

    torch.manual_seed(seed)
    if init.startswith("scratch"):
        model = NeuroCast(montage, rung="6m", window=256, causal=init == "scratch_causal",
                          device=device).to(device)
    else:
        model, _ = load_model(specs()[f"{init}_s{seed}"], montage, device)
    torch.manual_seed(seed + 1000)
    return Net(model).to(device)


def fit_one(init, tier, seed, lr, train, val, montage, device, amp, log):
    net = build(init, seed, montage, device)
    pos = float(np.mean(np.concatenate([lab for _, lab in train])))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=device))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / WARMUP if s < WARMUP else
                                              0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - WARMUP)
                                                                              / (STEPS - WARMUP))))
    best, curve, t0 = None, [], time.time()
    net.train()
    for step in range(1, STEPS + 1):
        xb, yb = batch(train, seed, step)
        xb, yb = torch.as_tensor(xb, device=device), torch.as_tensor(yb, device=device)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            out = net(xb)
        loss = lossf(out.float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0:
            v = scores(*predict(net, val, device, amp))
            curve.append((step, v["f1"], float(loss.detach())))
            if best is None or v["f1"] > best[1]:
                best = (step, v["f1"], {k: t.detach().cpu().clone() for k, t in net.state_dict().items()})
            log(f"    lr {lr:.0e} step {step:4d}  loss {float(loss.detach()):.3f}  val F1 {v['f1']:.3f}  "
                f"({(time.time() - t0) / step:.2f} s/step)")
    return net, best, curve


def stage_train(args) -> int:
    from run_bakeoff_real import FIT_SESSIONS, TEST, VAL
    from run_pretrain_pilot import montage_for
    from run_pretrain_scale import deep_runs

    if args.gpu_mem_frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
    device = torch.device(args.device)
    amp = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) \
        if device.type == "cuda" else None
    OUT.mkdir(parents=True, exist_ok=True)
    montage = montage_for(deep_runs()[0])
    val, test = load_sessions([VAL]), load_sessions([TEST])
    tiers = TIERS if not args.tiers else tuple(None if t == "all" else float(t) for t in args.tiers)
    for tier in tiers:
        train = load_sessions(FIT_SESSIONS, tier)
        hours = sum(len(lab) for _, lab in train) * PATCH / 250.0 / 3600
        print(f"tier {tier or 'all'}: {hours:.2f} h of labels in {len(train)} sessions", flush=True)
        for seed in (args.seeds or SEEDS):
            for init in (args.inits or INITS):
                name = run_name(init, tier, seed)
                f = OUT / f"{name}.json"
                if f.exists():
                    continue
                print(f"  {name}", flush=True)
                res = {"init": init, "tier": tier, "seed": seed, "hours": hours, "lrs": {}}
                chosen = None
                for lr in LRS:
                    net, best, curve = fit_one(init, tier, seed, lr, train, val, montage, device, amp,
                                               lambda m: print(m, flush=True))
                    res["lrs"][str(lr)] = {"curve": curve, "best_step": best[0], "val_f1": best[1]}
                    if chosen is None or best[1] > chosen[2]:
                        chosen = (lr, best[0], best[1], best[2])
                    del net
                    torch.cuda.empty_cache()
                net = build(init, seed, montage, device)
                net.load_state_dict(chosen[3])
                res.update({"lr": chosen[0], "best_step": chosen[1], "val_f1": chosen[2],
                            "test": scores(*predict(net, test, device, amp))})
                del net
                torch.cuda.empty_cache()
                f.write_text(json.dumps(res, indent=2, default=float))
                print(f"  {name}: lr {chosen[0]:.0e}, step {chosen[1]}, val F1 {chosen[2]:.3f}, "
                      f"test F1 {res['test']['f1']:.3f}, AUC {res['test']['auc']:.3f}", flush=True)
    return 0


def verdicts(rows: dict) -> dict:
    """The rules fixed in README 2.7 before any run."""
    out: dict = {}
    for tier in ("0.5h", "all"):
        per = {}
        for init, base in BASELINE.items():
            d = [rows[f"{init}_{tier}_s{s}"]["test"]["f1"] - rows[f"{base}_{tier}_s{s}"]["test"]["f1"]
                 for s in SEEDS if f"{init}_{tier}_s{s}" in rows and f"{base}_{tier}_s{s}" in rows]
            if len(d) < 2:
                per[init] = {"verdict": "not enough seeds", "differences": d}
                continue
            m = float(np.mean(d))
            per[init] = {"differences": d, "mean": m, "verdict":
                         "equal" if abs(m) < 0.01 else "helps" if all(x > 0 for x in d)
                         else "hurts" if all(x < 0 for x in d) else "no consistent difference"}
        out[tier] = per
    helps = [i for i, v in out["0.5h"].items() if v["verdict"] == "helps"]
    out["self_supervised_direction"] = (
        "worth continuing: " + ", ".join(helps) if helps else
        "dead end on this data and model" if all(v["verdict"] != "not enough seeds"
                                                  for v in out["0.5h"].values()) else "not decided yet")
    out["label_efficiency"] = [i for i in helps if out["all"].get(i, {}).get("verdict") == "equal"]
    return out


def stage_report(args) -> int:
    rows = {f.stem: json.loads(f.read_text()) for f in sorted(OUT.glob("*.json")) if f.stem != "verdicts"}
    print("| Start | Labels | Test F1 (seeds 0 / 1 / 2) | Mean | Test AUC, mean | lr, best step |")
    print("|---|---|---|---|---|---|")
    for tier in ("0.5h", "all"):
        for init in INITS:
            rs = [rows.get(f"{init}_{tier}_s{s}") for s in SEEDS]
            got = [r for r in rs if r]
            if not got:
                continue
            f1 = " / ".join(f"{r['test']['f1']:.3f}" if r else "—" for r in rs)
            print(f"| {init} | {tier} | {f1} | {np.mean([r['test']['f1'] for r in got]):.3f} | "
                  f"{np.mean([r['test']['auc'] for r in got]):.3f} | "
                  + ", ".join(f"{r['lr']:.0e}@{r['best_step']}" for r in got) + " |")
    v = verdicts(rows)
    (OUT / "verdicts.json").write_text(json.dumps(v, indent=2, default=float))
    print("\nverdicts (README 2.7 rules):")
    for k, x in v.items():
        print(f"  {k}: {x}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "report"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.9)
    ap.add_argument("--inits", nargs="*", choices=INITS)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--tiers", nargs="*", help="0.5 and/or all")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--steps", type=int, help="smoke tests only")
    ap.add_argument("--eval-every", type=int, help="smoke tests only")
    args = ap.parse_args()
    global OUT, STEPS, EVAL_EVERY
    if args.out:
        OUT = args.out.resolve()
    STEPS = args.steps or STEPS
    EVAL_EVERY = args.eval_every or EVAL_EVERY
    if args.stage == "train":
        return stage_train(args)
    return stage_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
