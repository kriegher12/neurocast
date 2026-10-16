"""Fine-tuning again: lower learning rates, linear probe then fine-tune, unseen sessions (README 2.12).

Section 2.7 fine-tuned each pre-trained start and the same network from scratch
on 0.5 h of speech labels, with the learning rate chosen from {1e-4, 1e-3}. Every
run chose 1e-4, the bottom of the grid, and the pre-trained starts drove their
training loss to ~0.003 (against ~0.07 from scratch) while scoring lower on new
chapters. Two explanations are tested here:

* the recipe: learning rates below the old grid, {1e-5, 3e-5, 1e-4}, for every
  start; and, for the pre-trained starts, a linear probe then fine-tuning (LP-FT,
  Kumar et al. 2022): 500 steps training the head alone at 1e-3 on the frozen
  trunk, then 1,500 steps of the whole network at the chosen rate;
* memorisation: the 0.5 h of labels came from sessions the pre-trained models
  had seen during pre-training. The same comparison is repeated with 0.5 h of
  labels from sessions held out of pre-training (section 2.2), 15 min from each
  of the first two, each start using the recipe and rate chosen on the first part
  (best mean chapter-11 F1).

Starts: ``ar_lik`` and ``masked_visible`` (section 2.2, seeds 0-2) and the same
network from scratch, causal and bidirectional. Everything else is section 2.7's:
the head, 4.1-s windows, batch 8, 2,000 steps, the class-balanced loss, the step
chosen every 250 steps on chapter 11, and the score on chapter 12.

    .venv-gpu/Scripts/python.exe scripts/run_finetune_followup.py --stage train    # resumable
    .venv/Scripts/python.exe     scripts/run_finetune_followup.py --stage report
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

import run_finetune as rf  # noqa: E402

OUT = ROOT / "runs" / "finetune_followup"
STARTS = ("scratch_causal", "scratch_bidir", "ar_lik", "masked_visible")
BASELINE = {"ar_lik": "scratch_causal", "masked_visible": "scratch_bidir"}
LRS = (1e-5, 3e-5, 1e-4)
SEEDS = (0, 1, 2)
LP_STEPS, LP_LR = 500, 1e-3
HOURS = 0.5


def recipes(start: str) -> tuple[str, ...]:
    return ("full",) if start.startswith("scratch") else ("full", "lpft")


def load_unseen(hours: float, n_sessions: int = 2):
    """``hours`` of labels from sessions held out of pre-training, split evenly over the first
    ``n_sessions`` in key order (section 2.7's 0.5 h also spans two sessions)."""
    from run_bakeoff_real import session_labels
    from run_pretrain_scale import cache_dir, deep_runs, heldout_keys

    from neurocast.train.pretrain import open_cached

    runs = {"_".join(r.key): r for r in deep_runs()}
    each = int(round(hours * 3600 / (rf.PATCH / 250.0))) // n_sessions
    out = []
    for key in sorted(heldout_keys(list(runs.values())))[:n_sessions]:
        rec = open_cached(cache_dir(), [key])[0]
        x = np.array(rec.data)
        p = min(x.shape[0] // rf.PATCH, each)
        lab = session_labels(runs[key].events, x.shape[0] // rf.PATCH, rec.story[0] / 250.0)["speech"]
        out.append((x[:p * rf.PATCH], lab[:p]))
    return out


def schedule(opt, lr_steps: int):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / rf.WARMUP if s < rf.WARMUP else
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - rf.WARMUP) / max(lr_steps - rf.WARMUP, 1))))


def fit(start, seed, lr, recipe, train, val, test, montage, device, amp) -> dict:
    net = rf.build(start, seed, montage, device)
    pos = float(np.mean(np.concatenate([lab for _, lab in train])))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=device))
    if recipe == "lpft":
        for q in net.model.parameters():
            q.requires_grad_(False)
        opt = torch.optim.AdamW(net.head.parameters(), lr=LP_LR, weight_decay=0.01)
        sched = schedule(opt, LP_STEPS)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
        sched = schedule(opt, rf.STEPS)
    best, curve, t0 = None, [], time.time()
    net.train()
    for step in range(1, rf.STEPS + 1):
        if recipe == "lpft" and step == LP_STEPS + 1:          # unfreeze: fine-tune everything
            for q in net.model.parameters():
                q.requires_grad_(True)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
            sched = schedule(opt, rf.STEPS - LP_STEPS)
        xb, yb = rf.batch(train, seed, step)
        xb, yb = torch.as_tensor(xb, device=device), torch.as_tensor(yb, device=device)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            out = net(xb)
        loss = lossf(out.float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for q in net.parameters() if q.requires_grad], 1.0)
        opt.step()
        sched.step()
        if step % rf.EVAL_EVERY == 0:
            v = rf.scores(*rf.predict(net, val, device, amp))
            curve.append((step, v["f1"], float(loss.detach())))
            if best is None or v["f1"] > best[1]:
                best = (step, v["f1"], float(loss.detach()),
                        {k: t.detach().cpu().clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[3])
    res = {"best_step": best[0], "val_f1": best[1], "train_loss_at_best": best[2], "curve": curve,
           "test": rf.scores(*rf.predict(net, test, device, amp)),
           "s_per_step": (time.time() - t0) / rf.STEPS}
    del net
    torch.cuda.empty_cache()
    return res


def chosen(rows: dict, start: str) -> tuple[str, float]:
    """A start's recipe and rate with the best chapter-11 F1, averaged over seeds."""
    best = None
    for recipe in recipes(start):
        for lr in LRS:
            v = [rows[f"{start}_{recipe}_seen_s{s}"]["lrs"][str(lr)]["val_f1"] for s in SEEDS
                 if f"{start}_{recipe}_seen_s{s}" in rows]
            if len(v) == len(SEEDS) and (best is None or np.mean(v) > best[0]):
                best = (float(np.mean(v)), recipe, lr)
    return (best[1], best[2]) if best else (None, None)


def load_rows() -> dict:
    return {f.stem: json.loads(f.read_text()) for f in sorted(OUT.glob("*.json")) if f.stem != "verdicts"}


def stage_train(args) -> int:
    from run_bakeoff_real import FIT_SESSIONS, TEST, VAL
    from run_pretrain_pilot import montage_for
    from run_pretrain_scale import deep_runs

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
    device = torch.device(args.device)
    amp = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) \
        if device.type == "cuda" else None
    OUT.mkdir(parents=True, exist_ok=True)
    montage = montage_for(deep_runs()[0])
    val, test = rf.load_sessions([VAL]), rf.load_sessions([TEST])
    if args.steps:                                     # smoke tests: a minute of each chapter
        val, test = ([(x[:rf.PATCH * 1000], lab[:1000]) for x, lab in d] for d in (val, test))
    seen = rf.load_sessions(FIT_SESSIONS, HOURS)
    seeds = args.seeds or SEEDS
    for seed in seeds:
        for start in args.starts or STARTS:
            for recipe in recipes(start):
                f = OUT / f"{start}_{recipe}_seen_s{seed}.json"
                if f.exists():
                    continue
                res = {"start": start, "recipe": recipe, "labels": "seen", "seed": seed, "lrs": {}}
                for lr in LRS:
                    res["lrs"][str(lr)] = r = fit(start, seed, lr, recipe, seen, val, test, montage, device, amp)
                    print(f"  {f.stem} lr {lr:.0e}: step {r['best_step']}, val F1 {r['val_f1']:.3f}, "
                          f"test F1 {r['test']['f1']:.3f}, train loss {r['train_loss_at_best']:.3f} "
                          f"({r['s_per_step']:.2f} s/step)", flush=True)
                f.write_text(json.dumps(res, indent=2, default=float))
    del seen
    unseen = load_unseen(HOURS)
    print(f"unseen sessions: {sum(len(lab) for _, lab in unseen) * rf.PATCH / 250 / 3600:.2f} h of labels "
          f"in {len(unseen)} sessions", flush=True)
    rows = load_rows()
    for seed in seeds:
        for start in args.starts or STARTS:
            f = OUT / f"{start}_unseen_s{seed}.json"
            recipe, lr = chosen(rows, start)
            if f.exists() or recipe is None:
                continue
            r = fit(start, seed, lr, recipe, unseen, val, test, montage, device, amp)
            f.write_text(json.dumps({"start": start, "recipe": recipe, "lr": lr, "labels": "unseen",
                                     "seed": seed, **r}, indent=2, default=float))
            print(f"  {f.stem} ({recipe}, lr {lr:.0e}): test F1 {r['test']['f1']:.3f}, train loss "
                  f"{r['train_loss_at_best']:.3f}", flush=True)
    return 0


def verdicts(rows: dict) -> dict:
    """The rules fixed in README 2.12 before any run."""
    out: dict = {}

    def seen_best(start, s):
        """This seed's run with the best chapter-11 F1 over the start's recipes and rates."""
        opts = [(v["val_f1"], v["test"]["f1"], recipe, lr)
                for recipe in recipes(start) if f"{start}_{recipe}_seen_s{s}" in rows
                for lr, v in rows[f"{start}_{recipe}_seen_s{s}"]["lrs"].items()]
        return max(opts) if opts else None

    def rule(d):
        if len(d) < len(SEEDS):
            return "not enough seeds"
        m = float(np.mean(d))
        return ("equal" if abs(m) < 0.01 else "helps" if all(x > 0 for x in d)
                else "hurts" if all(x < 0 for x in d) else "no consistent difference")

    old = {f.stem: json.loads(f.read_text()) for f in (ROOT / "runs" / "finetune").glob("*_0.5h_s*.json")}
    for start, base in BASELINE.items():
        a = [seen_best(start, s) for s in SEEDS]
        b = [seen_best(base, s) for s in SEEDS]
        d = [x[1] - y[1] for x, y in zip(a, b) if x and y]
        d27 = [old[f"{start}_0.5h_s{s}"]["test"]["f1"] - old[f"{base}_0.5h_s{s}"]["test"]["f1"]
               for s in SEEDS if f"{start}_0.5h_s{s}" in old and f"{base}_0.5h_s{s}" in old]
        row = {"chosen_per_seed": [x[2:] if x else None for x in a], "differences": d, "verdict": rule(d),
               "section_2_7_differences": d27, "section_2_7_verdict": rule(d27)}
        row["recipe_was_the_shortfall"] = row["section_2_7_verdict"] == "hurts" and row["verdict"] in ("equal",
                                                                                                         "helps")
        # Memorisation: the same start-level recipe and rate, seen against unseen sessions' labels.
        cfg_a, cfg_b = chosen(rows, start), chosen(rows, base)
        if None not in cfg_a + cfg_b:
            seen_gap = [rows[f"{start}_{cfg_a[0]}_seen_s{s}"]["lrs"][str(cfg_a[1])]["test"]["f1"]
                        - rows[f"{base}_{cfg_b[0]}_seen_s{s}"]["lrs"][str(cfg_b[1])]["test"]["f1"] for s in SEEDS]
            un = [rows[f"{start}_unseen_s{s}"]["test"]["f1"] - rows[f"{base}_unseen_s{s}"]["test"]["f1"]
                  for s in SEEDS if f"{start}_unseen_s{s}" in rows and f"{base}_unseen_s{s}" in rows]
            row.update({"config": cfg_a, "baseline_config": cfg_b, "seen_gap": seen_gap,
                        "unseen_gap": un, "unseen_verdict": rule(un)})
            if len(un) == len(SEEDS):
                row["memorisation_contributes"] = float(np.mean(un) - np.mean(seen_gap)) >= 0.01
        out[start] = row
    return out


def stage_report(args) -> int:
    rows = load_rows()
    print("| Start | Recipe | lr | Test F1, seeds 0 / 1 / 2 (seen-session labels) | Val F1 mean | Train loss at best |")
    print("|---|---|---|---|---|---|")
    for start in STARTS:
        for recipe in recipes(start):
            for lr in LRS:
                rs = [rows.get(f"{start}_{recipe}_seen_s{s}", {}).get("lrs", {}).get(str(lr)) for s in SEEDS]
                got = [r for r in rs if r]
                if not got:
                    continue
                print(f"| {start} | {recipe} | {lr:.0e} | " + " / ".join(
                    f"{r['test']['f1']:.3f}" if r else "-" for r in rs) +
                    f" | {np.mean([r['val_f1'] for r in got]):.3f} | "
                    f"{np.mean([r['train_loss_at_best'] for r in got]):.3f} |")
    print("\nunseen-session labels:")
    for start in STARTS:
        rs = [rows.get(f"{start}_unseen_s{s}") for s in SEEDS]
        if any(rs):
            r0 = next(r for r in rs if r)
            print(f"  {start} ({r0['recipe']}, lr {r0['lr']:.0e}): test F1 " + " / ".join(
                f"{r['test']['f1']:.3f}" if r else "-" for r in rs))
    v = verdicts(rows)
    (OUT / "verdicts.json").write_text(json.dumps(v, indent=2, default=float))
    print("\nverdicts (README 2.12 rules):")
    for k, x in v.items():
        print(f"  {k}: {x}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "report"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--starts", nargs="*", choices=STARTS)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--steps", type=int, help="smoke tests only")
    ap.add_argument("--eval-every", type=int, help="smoke tests only")
    ap.add_argument("--lp-steps", type=int, help="smoke tests only")
    args = ap.parse_args()
    global OUT, LP_STEPS
    if args.out:
        OUT = args.out.resolve()
    rf.STEPS = args.steps or rf.STEPS
    rf.EVAL_EVERY = args.eval_every or rf.EVAL_EVERY
    LP_STEPS = args.lp_steps or LP_STEPS
    return stage_train(args) if args.stage == "train" else stage_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
