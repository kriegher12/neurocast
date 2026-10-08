"""Forecasting vs fixed masked pre-training on the deep listener's hours: H1/H2.

The pitch's plan after the pilot: "forecasting vs fixed masked pre-training, 25M
parameters, 3 seeds, 5-20 h of data, held-out loss tracking, identity audit of
frozen features". The pilot (``run_pretrain_pilot.py``) trained on 38 min and its
forecasting arm overfit by step 2,000; here the same recipe runs on every
Sherlock session of the deep listener (sub-0) the word probe does not use:

* data: sub-0, all Sherlock books, minus Sherlock1 sessions 11-12 (the chapters
  the frozen-feature word probe is fitted and tested on -- stimulus-disjoint);
  the last session of each of Sherlock2-9 is held out whole for loss tracking;
* arms: likelihood forecasting (``ar_lik``) and masked reconstruction with the
  visible-position term (the pitch's fix), 3 seeds each; plain masked once, to
  see whether the collapse survives ~90x more data (~60 h against 38 min);
* recipe: the pilot's (6m rung, 24.9M parameters; 4-s windows; batch 8; AdamW
  1e-3; warm-up and cosine), for ``--steps`` steps; the best held-out
  checkpoint is what gets evaluated;
* evaluation: masked-collapse diagnostic, identity audit at a stated N (64
  windows per listener, 32 broad listeners, chapter 12), and the frozen-feature
  word probe (fit chapter 11, test chapter 12) -- the pilot's, on every model.

    .venv/Scripts/python.exe scripts/run_pretrain_scale.py --prepare        # cache sessions (CPU)
    .venv-gpu/Scripts/python.exe scripts/run_pretrain_scale.py --queue     # all runs, resumable
    .venv-gpu/Scripts/python.exe scripts/run_pretrain_scale.py --evaluate  # after the queue
    .venv/Scripts/python.exe scripts/run_pretrain_scale.py --plot          # held-out curves, any time

Each run resumes from its checkpoint; ``--queue`` skips finished runs, so it can
be stopped and restarted at any point.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from neurocast.data.libribrain import find_runs  # noqa: E402
from neurocast.data.paths import dataset_dir  # noqa: E402
from neurocast.train.pretrain import PretrainConfig, Pretrainer, cache_recordings, open_cached  # noqa: E402

OUT = ROOT / "runs" / "pretrain_scale"
PROBE_CHAPTERS = {("Sherlock1", "11"), ("Sherlock1", "12")}
ARMS = {"ar_lik": dict(arm="ar_lik"),
        "masked_visible": dict(arm="masked", visible_weight=1.0),
        "masked": dict(arm="masked", visible_weight=0.0)}
#: Run order: one seed of each main arm before the next seed, plain masked last.
QUEUE = [("ar_lik", 0), ("masked_visible", 0), ("ar_lik", 1), ("masked_visible", 1),
         ("ar_lik", 2), ("masked_visible", 2), ("masked", 0)]


def cache_dir() -> Path:
    return dataset_dir("derived") / "pretrain_cache" / "session-scalar"


def deep_runs():
    """sub-0's Sherlock runs with MEG, minus the word probe's chapters, in a fixed order."""
    base = dataset_dir("LibriBrain100")
    tasks = sorted(p.name for p in base.iterdir()
                   if p.name.startswith("Sherlock") and (p / "derivatives").is_dir())
    runs = [r for t in tasks for r in find_runs(task=t, subjects={"0"}, require_meg=True)
            if (r.task, r.session) not in PROBE_CHAPTERS]
    return sorted(runs, key=lambda r: (r.task, int(r.session), r.run))


def heldout_keys(runs) -> list[str]:
    """The last session of every book but Sherlock1 (whose last ones are the probe's)."""
    last: dict[str, object] = {}
    for r in runs:
        if r.task != "Sherlock1":
            last[r.task] = r                         # runs are sorted, so this ends at the last
    return ["_".join(r.key) for r in last.values()]


def run_name(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def build(arm: str, seed: int, steps: int, device: str, prefetch: int = 8) -> Pretrainer:
    from run_pretrain_pilot import montage_for

    runs = deep_runs()
    keys = ["_".join(r.key) for r in runs]
    missing = [k for k in keys if not (cache_dir() / f"{k}.npy").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} sessions not cached (e.g. {missing[0]}); run --prepare")
    recs = open_cached(cache_dir(), keys, heldout=heldout_keys(runs))
    cfg = PretrainConfig(steps=steps, seed=seed, eval_every=500, eval_batches=32,
                         prefetch=prefetch, **ARMS[arm])
    return Pretrainer(cfg, montage_for(runs[0]), recs, device=device)


def describe_data() -> dict:
    runs = deep_runs()
    held = set(heldout_keys(runs))
    out = {"sessions": 0, "train_h": 0.0, "heldout_h": 0.0, "heldout": sorted(held)}
    for r in runs:
        k = "_".join(r.key)
        f = cache_dir() / f"{k}.npy"
        if not f.exists():
            continue
        n = np.load(f, mmap_mode="r").shape[0] / 250.0 / 3600.0
        out["sessions"] += 1
        out["heldout_h" if k in held else "train_h"] += n
    return out


def cmd_prepare(args) -> int:
    runs = deep_runs()
    print(f"sub-0 Sherlock runs with MEG, minus the probe chapters: {len(runs)}")
    cache_recordings(runs, cache_dir())
    d = describe_data()
    print(f"cached: {d['sessions']} sessions, train {d['train_h']:.1f} h, held out "
          f"{d['heldout_h']:.1f} h ({', '.join(d['heldout'])})")
    return 0


def cmd_train(arm: str, seed: int, args) -> None:
    name = run_name(arm, seed)
    ck = OUT / f"{name}.pt"
    tr = build(arm, seed, args.steps, args.device)
    if ck.exists():
        tr.load(ck)
    if tr.step >= args.steps:
        print(f"{name}: finished ({tr.step} steps), skipped", flush=True)
        return
    d = describe_data()
    print(f"{name}: {tr.model.describe()}; {d['train_h']:.1f} h train, {d['heldout_h']:.1f} h "
          f"held out; {args.steps} steps; {tr.device} {tr.amp_dtype}", flush=True)
    t0 = time.time()
    tr.fit(ck)
    print(f"{name}: done in {(time.time() - t0) / 60:.1f} min, best held-out {tr.best:.4f}", flush=True)
    del tr
    gc.collect()
    torch.cuda.empty_cache()


def verdicts(res: dict) -> dict:
    """The decision rules fixed in README section 2.2 before any run finished.

    * H2 -- forecasting leaks "more" person information if its identity ratio is
      above fixed masked's for every seed, "less" if below for every seed,
      otherwise "no consistent difference";
    * decoding -- "equal" if mean BAcc@10 differs by under 0.01, otherwise the
      higher arm if every seed agrees, otherwise "no consistent difference";
    * collapse -- the diagnostic's own rule, on the plain masked run;
    * long enough -- a run whose best held-out loss is its last evaluation is
      "under-trained".
    """
    runs = {k: v for k, v in res.items() if not k.startswith("_") and "identity" in v}
    seeds = sorted({v["seed"] for v in runs.values() if v["arm"] == "ar_lik"}
                   & {v["seed"] for v in runs.values() if v["arm"] == "masked_visible"})
    out: dict = {"paired_seeds": seeds}
    if seeds:
        f = {sd: runs[run_name("ar_lik", sd)] for sd in seeds}
        m = {sd: runs[run_name("masked_visible", sd)] for sd in seeds}
        d_id = [f[sd]["identity"]["features"]["ratio"] - m[sd]["identity"]["features"]["ratio"]
                for sd in seeds]
        out["identity"] = ("more" if all(x > 0 for x in d_id) else
                           "less" if all(x < 0 for x in d_id) else "no consistent difference")
        out["identity_ratio_forecasting_over_masked"] = [
            f[sd]["identity"]["features"]["ratio"] / m[sd]["identity"]["features"]["ratio"]
            for sd in seeds]
        d_acc = [f[sd]["word_probe"]["bacc10"] - m[sd]["word_probe"]["bacc10"] for sd in seeds]
        if abs(float(np.mean(d_acc))) < 0.01:
            out["decoding"] = "equal"
        elif all(x > 0 for x in d_acc):
            out["decoding"] = "forecasting higher"
        elif all(x < 0 for x in d_acc):
            out["decoding"] = "masked higher"
        else:
            out["decoding"] = "no consistent difference"
        out["decoding_mean_difference"] = float(np.mean(d_acc))
    plain = runs.get(run_name("masked", 0))
    if plain and "collapse" in plain:
        out["plain_masked_collapsed"] = bool(plain["collapse"]["collapsed"])
    out["under_trained"] = sorted(
        k for k, v in runs.items()
        if v.get("heldout_curve") and v["best_step"] == v["heldout_curve"][-1][0])
    return out


def cmd_evaluate(args) -> int:
    from run_pretrain_pilot import collapse, identity, listener_features, word_probe

    res, models, n_times = {}, {}, None
    saved = OUT / "scale_eval.json"
    for arm, seed in QUEUE:
        name = run_name(arm, seed)
        best = Pretrainer.best_path(OUT / f"{name}.pt")
        if not best.exists():
            print(f"  {name}: no best checkpoint, skipped")
            continue
        tr = build(arm, seed, args.steps, args.device, prefetch=0)
        tr.load(best)
        full = torch.load(OUT / f"{name}.pt", map_location="cpu", weights_only=False)
        held = [(e["step"], e["heldout"]) for e in full["log"] if e.get("heldout") is not None]
        r = {"arm": arm, "seed": seed, "best_step": tr.step, "best_heldout": tr.best,
             "steps_trained": full["step"], "heldout_curve": held}
        del full
        if tr.cfg.arm == "masked":
            rep = collapse(tr)
            r["collapse"] = {"spread": rep.spread, "sensitivity": rep.sensitivity,
                             "model_r2": rep.model_r2, "reference_r2": rep.reference_r2,
                             "isolated_r2": rep.isolated_r2, "collapsed": rep.collapsed}
            print(f"  {name:<18} collapse: {rep.summary()}", flush=True)
        models[name], n_times, res[name] = tr.model.eval(), tr.n_times, r
        del tr
        gc.collect()
    print(f"  reading the 32 broad listeners once, through {len(models)} models ...", flush=True)
    feats = listener_features(models, n_times, args.device, per_listener=args.per_listener)
    subj = feats["subject"]
    raw = identity(feats["raw"], subj)
    vocab = json.loads((ROOT / "runs" / "libribrain_audit" / "nobrain.json").read_text())["vocab"]
    for name, f in feats["models"].items():
        r = res[name]
        r["identity"] = {"features": identity(f["identity"], subj), "raw": raw}
        r["word_probe"] = word_probe(f["words"], len(vocab))
        print(f"  {name:<18} identity {r['identity']['features']['ratio']:5.1f}x (raw {raw['ratio']:.1f}x, "
              f"N {raw['n']}, ceiling {raw['ceiling']:.0f}x)   word probe BAcc@1 "
              f"{r['word_probe']['bacc1']:.3f} BAcc@10 {r['word_probe']['bacc10']:.3f}", flush=True)
        f.clear()                                    # free this model's features
        saved.write_text(json.dumps(res, indent=2, default=float))

    print("\n  per arm, mean +- sd over seeds (best held-out checkpoints):")
    print(f"  {'arm':<16} {'seeds':>5} {'held-out':>16} {'identity x':>14} {'BAcc@10':>16}")
    summary = {}
    for arm in ARMS:
        rows = [v for v in res.values() if v["arm"] == arm and "identity" in v]
        if not rows:
            continue
        def ms(xs):
            xs = np.asarray(xs, float)
            return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else float("nan")
        h, idn, acc = (ms([v["best_heldout"] for v in rows]),
                       ms([v["identity"]["features"]["ratio"] for v in rows]),
                       ms([v["word_probe"]["bacc10"] for v in rows]))
        summary[arm] = {"n": len(rows), "heldout": h, "identity": idn, "bacc10": acc}
        print(f"  {arm:<16} {len(rows):>5} {h[0]:>9.4f} +- {h[1]:.4f} {idn[0]:>7.1f} +- {idn[1]:.1f} "
              f"{acc[0]:>9.3f} +- {acc[1]:.3f}")
    res["_summary"] = summary
    res["_data"] = describe_data()
    res["_verdicts"] = verdicts(res)
    print("\n  verdicts under the rules fixed before the run (README 2.2):")
    for k, v in res["_verdicts"].items():
        print(f"    {k}: {v}")
    saved.write_text(json.dumps(res, indent=2, default=float))
    print(f"  saved {saved}")
    print("  pitch (pilot): decode about equally well; forecasting ~3x the person information (29x vs 10x)")
    return 0


def cmd_plot(args) -> int:
    """Held-out loss against step for every run so far, one panel per objective."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"ar_lik": "forecasting (NLL, nats per target dim)",
              "masked_visible": "masked + visible term (MSE)", "masked": "masked, plain (MSE)"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, arm in zip(axes, ARMS):
        for seed in range(3):
            ck = OUT / f"{run_name(arm, seed)}.pt"
            if not ck.exists():
                continue
            log = torch.load(ck, map_location="cpu", weights_only=False)["log"]
            h = [(e["step"], e["heldout"]) for e in log if e.get("heldout") is not None]
            st, hv = zip(*h)
            line, = ax.plot(st, hv, label=f"seed {seed}")
            b = int(np.argmin(hv))
            ax.plot(st[b], hv[b], "o", color=line.get_color())
        ax.set_title(titles[arm], fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8)
    axes[0].set_ylabel("held-out loss (8 whole sessions)")
    fig.suptitle("Pre-training on the deep listener's 61 h: held-out loss (dot = best checkpoint)",
                 fontsize=11)
    fig.tight_layout()
    out = OUT / "heldout.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    return 0


def cmd_report(args) -> int:
    """Markdown tables from ``scale_eval.json``, for the README and the ledger."""
    res = json.loads((OUT / "scale_eval.json").read_text())
    runs = {k: v for k, v in res.items() if not k.startswith("_")}
    print("| Run | Best held-out (step) | Identity, N = 2,048 | Subject probe | BAcc@1 / BAcc@10 | Collapse |")
    print("|---|---|---|---|---|---|")
    for k, v in runs.items():
        idn = v.get("identity", {}).get("features", {})
        wp = v.get("word_probe", {})
        col = v.get("collapse")
        col_s = ("collapsed" if col["collapsed"] else f"reads input (sens. {col['sensitivity']:.2f})") if col else "—"
        last = v["heldout_curve"][-1][0] if v.get("heldout_curve") else None
        flag = " (last eval)" if last == v["best_step"] else ""
        print(f"| {k} | {v['best_heldout']:.4f} ({v['best_step']}{flag}) | "
              f"{idn.get('ratio', float('nan')):.1f}x | {idn.get('probe', float('nan')):.3f} | "
              f"{wp.get('bacc1', float('nan')):.3f} / {wp.get('bacc10', float('nan')):.3f} | {col_s} |")
    raw = next(iter(runs.values()))["identity"]["raw"] if runs else None
    if raw:
        print()
        print(f"raw signal (log amplitude, same windows): {raw['ratio']:.1f}x, N {raw['n']}, "
              f"ceiling {raw['ceiling']:.0f}x")
    for arm, sm in res.get("_summary", {}).items():
        print(f"{arm}: identity {sm['identity'][0]:.1f} +- {sm['identity'][1]:.1f}x, "
              f"BAcc@10 {sm['bacc10'][0]:.3f} +- {sm['bacc10'][1]:.3f} ({sm['n']} seeds)")
    print("verdicts:", json.dumps(res.get("_verdicts", {}), default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="cache the sessions on the data drive")
    ap.add_argument("--queue", action="store_true", help="train every run in QUEUE, resumably")
    ap.add_argument("--run", choices=tuple(ARMS), help="train one arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--plot", action="store_true", help="held-out curves of every run so far")
    ap.add_argument("--report", action="store_true", help="markdown tables from scale_eval.json")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--per-listener", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        return cmd_prepare(args)
    if args.run:
        cmd_train(args.run, args.seed, args)
        return 0
    if args.queue:
        for arm, seed in QUEUE:
            cmd_train(arm, seed, args)
        return 0
    if args.evaluate:
        return cmd_evaluate(args)
    if args.plot:
        return cmd_plot(args)
    if args.report:
        return cmd_report(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
