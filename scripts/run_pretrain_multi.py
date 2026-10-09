"""Forecasting vs masked pre-training across many listeners: H2 in the setting it was written for.

``run_pretrain_scale.py`` pre-trained on one listener's 61 h and found the
forecasting arm's features carry 6-8x the person information of the masked
arm's. H2 was written for a model trained across many people, where person
differences have to be modelled rather than memorised. This runs the same
comparison on MEG-MASC (Gwilliams et al.): a different MEG system (208 KIT axial
gradiometers, montage from the digitised head positions,
:func:`neurocast.data.megmasc.kit_montage`), many listeners, the same recipe.

* pre-training: listeners 01-20, stories 0-2 (~12.4 h); held-out loss on
  listeners 21-27, stories 0-2, so held-out means unseen people;
* evaluation, story 3 only (never seen in pre-training):
  - identity audit on the 7 unseen listeners (64 windows each, N = 448,
    ceiling 74.5x) and on the 20 seen ones (N = 1,280, ceiling 67x);
  - frozen-feature word probe on the unseen listeners: fit on stories 0-2,
    test on story 3 (the same linear decoder as the LibriBrain audit);
  - the masked-collapse diagnostic;
* arms: likelihood forecasting and masked with the visible term, 3 seeds each,
  10,000 steps, best held-out checkpoints.

    .venv/Scripts/python.exe scripts/run_pretrain_multi.py --prepare
    .venv-gpu/Scripts/python.exe scripts/run_pretrain_multi.py --queue
    .venv-gpu/Scripts/python.exe scripts/run_pretrain_multi.py --evaluate
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from neurocast.data import megmasc  # noqa: E402
from neurocast.data.paths import dataset_dir  # noqa: E402
from neurocast.train.pretrain import PretrainConfig, Pretrainer, open_cached  # noqa: E402

OUT = ROOT / "runs" / "pretrain_multi"
TRAIN = tuple(f"{i:02d}" for i in range(1, 21))
HELD = tuple(f"{i:02d}" for i in range(21, 28))
PRE_TASKS, EVAL_TASK = ("0", "1", "2"), "3"
ARMS = {"ar_lik": dict(arm="ar_lik"), "masked_visible": dict(arm="masked", visible_weight=1.0)}
QUEUE = [(a, s) for s in range(3) for a in ARMS]
STEPS = 10_000


def cache_dir() -> Path:
    return dataset_dir("derived") / "pretrain_cache" / "megmasc"


def montage():
    """The KIT montage, computed once from the training listeners' head positions."""
    f = OUT / "montage.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    m = megmasc.kit_montage(list(TRAIN))
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(m))
    return m


def run_name(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def build(arm: str, seed: int, device: str, prefetch: int = 8) -> Pretrainer:
    keys = [f"sub-{s}_task-{t}" for s in TRAIN + HELD for t in PRE_TASKS]
    missing = [k for k in keys if not (cache_dir() / f"{k}.npy").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} recordings not cached (e.g. {missing[0]}); run --prepare")
    held = [f"sub-{s}_task-{t}" for s in HELD for t in PRE_TASKS]
    recs = open_cached(cache_dir(), keys, heldout=held)
    cfg = PretrainConfig(steps=STEPS, seed=seed, eval_every=500, eval_batches=32, prefetch=prefetch,
                         **ARMS[arm])
    return Pretrainer(cfg, montage(), recs, device=device)


def cmd_prepare(args) -> int:
    keys = megmasc.cache_for_pretraining(TRAIN + HELD, PRE_TASKS, cache_dir())
    hours = {"train": 0.0, "held": 0.0}
    for k in keys:
        n = np.load(cache_dir() / f"{k}.npy", mmap_mode="r").shape[0] / 250 / 3600
        hours["held" if k.split("_")[0][4:] in HELD else "train"] += n
    m = montage()
    print(f"{len(keys)} recordings: {hours['train']:.1f} h train ({len(TRAIN)} listeners), "
          f"{hours['held']:.1f} h held out ({len(HELD)} unseen listeners)")
    print(m.summary())
    return 0


def cmd_train(arm: str, seed: int, args) -> None:
    name = run_name(arm, seed)
    ck = OUT / f"{name}.pt"
    tr = build(arm, seed, args.device)
    if ck.exists():
        tr.load(ck)
    if tr.step >= STEPS:
        print(f"{name}: finished, skipped", flush=True)
        return
    print(f"{name}: {tr.model.describe()}; {STEPS} steps; {tr.device} {tr.amp_dtype}", flush=True)
    t0 = time.time()
    tr.fit(ck)
    print(f"{name}: done in {(time.time() - t0) / 60:.1f} min, best held-out {tr.best:.4f}", flush=True)
    del tr
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def features(models: dict, n_times: int, device, per_listener: int = 64, seed: int = 0) -> dict:
    """Story-3 identity windows of every listener, and the unseen listeners' word windows."""
    from neurocast.data.libribrain import onset_windows
    from neurocast.decode.words import label_trials, vocabulary

    ref = megmasc.word_table("01", "0", PRE_TASKS + (EVAL_TASK,), window_s=0.512)
    vocab = vocabulary(ref, 50)
    rng = np.random.default_rng(seed)
    ident = {k: {"seen": [], "unseen": []} for k in models}
    raw, subj = {"seen": [], "unseen": []}, {"seen": [], "unseen": []}
    words = {k: {"fit": [], "test": []} for k in models}
    labels = {"fit": ([], []), "test": ([], [])}
    src = dataset_dir("derived") / "megmasc"
    t0 = time.time()
    for sub in TRAIN + HELD:
        group = "unseen" if sub in HELD else "seen"
        tasks = PRE_TASKS + (EVAL_TASK,) if sub in HELD else (EVAL_TASK,)
        w_all = megmasc.word_table(sub, "0", tasks, window_s=0.512)
        for task in tasks:
            x, sfreq = megmasc.load_preprocessed(src / f"sub-{sub}_ses-0_task-{task}.npy")
            if task == EVAL_TASK:
                starts = rng.integers(0, x.shape[1] - n_times, per_listener)
                w = np.stack([x[:, s:s + n_times] for s in starts])
                for i in range(0, len(w), 8):
                    wt = torch.as_tensor(w[i:i + 8], device=device)
                    for k, m in models.items():
                        ident[k][group].append(m.pooled(wt).float().cpu().numpy())
                raw[group].append(np.log(w.std(axis=2) + 1e-6))
                subj[group] += [sub] * per_listener
            if sub in HELD:
                split = "test" if task == EVAL_TASK else "fit"
                t = label_trials(w_all[w_all["task"] == task], vocab)
                win, kept = onset_windows(x, sfreq, t["onset"].to_numpy(), n_samples=128, decim=1)
                for i in range(0, len(win), 64):
                    xb = torch.as_tensor(win[i:i + 64], device=device)
                    for k, m in models.items():
                        g = m.groups(xb).float().mean(2)
                        words[k][split].append(g.reshape(len(xb), -1).cpu().numpy().astype(np.float16))
                labels[split][0].append(t["label"].to_numpy()[kept])
                labels[split][1].append(t["occ"].to_numpy()[kept])
        print(f"    sub-{sub} ({group}) {time.time() - t0:.0f}s", flush=True)
    out = {"vocab": vocab, "subject": {g: np.array(v) for g, v in subj.items()},
           "raw": {g: np.concatenate(v) for g, v in raw.items()},
           "labels": {s: (np.concatenate(y), np.concatenate(o)) for s, (y, o) in labels.items()},
           "models": {}}
    for k in models:
        out["models"][k] = {"identity": {g: np.concatenate(v) for g, v in ident[k].items()},
                            "words": {s: np.concatenate(v) for s, v in words[k].items()}}
    return out


def probe(words: dict, labels: dict, n_classes: int) -> dict:
    from neurocast.decode.decoders import LinearPooled
    from neurocast.decode.metrics import balanced_accuracy_at_k

    (yf, of), (yt, _) = labels["fit"], labels["test"]
    m = LinearPooled(seed=0).fit(words["fit"].astype(np.float32), yf, of)
    s = m.scores(words["test"].astype(np.float32), n_classes)
    return {"bacc1": balanced_accuracy_at_k(s, yt, 1), "bacc10": balanced_accuracy_at_k(s, yt, 10),
            "n_fit": len(yf), "n_test": len(yt)}


def verdicts(res: dict) -> dict:
    """The rules fixed in README section 2.3 before any run finished."""
    runs = {k: v for k, v in res.items() if not k.startswith("_")}
    seeds = sorted({v["seed"] for v in runs.values() if v["arm"] == "ar_lik"}
                   & {v["seed"] for v in runs.values() if v["arm"] == "masked_visible"})
    out: dict = {"paired_seeds": seeds}
    if not seeds:
        return out
    f = {sd: runs[run_name("ar_lik", sd)] for sd in seeds}
    m = {sd: runs[run_name("masked_visible", sd)] for sd in seeds}
    for group in ("unseen", "seen"):
        d = [f[sd]["identity"][group]["ratio"] - m[sd]["identity"][group]["ratio"] for sd in seeds]
        out[f"identity_{group}"] = ("more" if all(x > 0 for x in d) else
                                    "less" if all(x < 0 for x in d) else "no consistent difference")
        out[f"identity_{group}_forecasting_over_masked"] = [
            f[sd]["identity"][group]["ratio"] / m[sd]["identity"][group]["ratio"] for sd in seeds]
    d = [f[sd]["word_probe"]["bacc10"] - m[sd]["word_probe"]["bacc10"] for sd in seeds]
    out["decoding"] = ("equal" if abs(float(np.mean(d))) < 0.01 else
                       "forecasting higher" if all(x > 0 for x in d) else
                       "masked higher" if all(x < 0 for x in d) else "no consistent difference")
    out["decoding_mean_difference"] = float(np.mean(d))
    out["under_trained"] = sorted(k for k, v in runs.items()
                                  if v["best_step"] == v["heldout_curve"][-1][0])
    return out


def cmd_evaluate(args) -> int:
    from run_pretrain_pilot import collapse, identity

    res, models, n_times = {}, {}, None
    for arm, seed in QUEUE:
        name = run_name(arm, seed)
        best = Pretrainer.best_path(OUT / f"{name}.pt")
        if not best.exists():
            print(f"  {name}: no best checkpoint, skipped")
            continue
        tr = build(arm, seed, args.device, prefetch=0)
        tr.load(best)
        full = torch.load(OUT / f"{name}.pt", map_location="cpu", weights_only=False)
        r = {"arm": arm, "seed": seed, "best_step": tr.step, "best_heldout": tr.best,
             "heldout_curve": [(e["step"], e["heldout"]) for e in full["log"]
                               if e.get("heldout") is not None]}
        del full
        if tr.cfg.arm == "masked":
            rep = collapse(tr)
            r["collapse"] = {"spread": rep.spread, "sensitivity": rep.sensitivity,
                             "model_r2": rep.model_r2, "collapsed": rep.collapsed}
            print(f"  {name:<18} collapse: {rep.summary()}", flush=True)
        models[name], n_times, res[name] = tr.model.eval(), tr.n_times, r
        del tr
        gc.collect()
    print(f"  story 3 of all 27 listeners, stories 0-3 of the 7 unseen, through {len(models)} models ...",
          flush=True)
    f = features(models, n_times, args.device)
    raw = {g: identity(f["raw"][g], f["subject"][g]) for g in ("unseen", "seen")}
    for name, fm in f["models"].items():
        r = res[name]
        r["identity"] = {g: identity(fm["identity"][g], f["subject"][g]) for g in ("unseen", "seen")}
        r["identity_raw"] = raw
        r["word_probe"] = probe(fm["words"], f["labels"], len(f["vocab"]))
        print(f"  {name:<18} identity unseen {r['identity']['unseen']['ratio']:5.1f}x "
              f"(raw {raw['unseen']['ratio']:.1f}x, ceiling {raw['unseen']['ceiling']:.0f}x)  "
              f"seen {r['identity']['seen']['ratio']:5.1f}x (raw {raw['seen']['ratio']:.1f}x)  "
              f"probe BAcc@1 {r['word_probe']['bacc1']:.3f} BAcc@10 {r['word_probe']['bacc10']:.3f}",
              flush=True)
    res["_verdicts"] = verdicts(res)
    print("\n  verdicts under the rules fixed before the run (README 2.3):")
    for k, v in res["_verdicts"].items():
        print(f"    {k}: {v}")
    (OUT / "multi_eval.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"  saved {OUT / 'multi_eval.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--queue", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        return cmd_prepare(args)
    if args.queue:
        for arm, seed in QUEUE:
            cmd_train(arm, seed, args)
        return 0
    if args.evaluate:
        return cmd_evaluate(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
