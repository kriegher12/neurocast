"""Re-run the lost version's pre-training pilot on the deep listener's real MEG.

Pitch, verbatim in substance: "the masked model ... had collapsed on real data:
its output ignored its input. We diagnosed why, fixed it [a visible-position
term], and ran a fair comparison. The two approaches decode about equally well,
but the forecasting model packs about three times more information about who the
person is into its features" (29x vs 10x; raw signal 3x).

Recipe (pitch): 6m rung (24.9M), 4-s windows (64 patches), batch 8, 4,000 steps,
sub-0's three Sherlock2 sessions (47.5 min -- re-measured: exactly 47.5), real
306-sensor MEGIN geometry from the release's fine-calibration file.

    .venv-gpu/Scripts/python.exe scripts/run_pretrain_pilot.py --arm ar_lik
    .venv-gpu/Scripts/python.exe scripts/run_pretrain_pilot.py --arm masked
    .venv-gpu/Scripts/python.exe scripts/run_pretrain_pilot.py --arm masked --visible-weight 1.0
    .venv-gpu/Scripts/python.exe scripts/run_pretrain_pilot.py --evaluate     # after all three
    .venv-gpu/Scripts/python.exe scripts/run_pretrain_pilot.py --name masked_lowlr
    .venv-gpu/Scripts/python.exe scripts/run_pretrain_pilot.py --evaluate --collapse-only

Training resumes from its checkpoint if interrupted.
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

from neurocast.data.libribrain import find_runs, meg_info  # noqa: E402
from neurocast.data.megin import vectorview_montage  # noqa: E402
from neurocast.data.paths import dataset_dir  # noqa: E402
from neurocast.train.pretrain import PretrainConfig, Pretrainer, load_recordings  # noqa: E402

OUT = ROOT / "runs" / "pretrain"
BROAD = {str(i) for i in range(1, 33)}
ARMS = {"ar_lik": dict(arm="ar_lik"), "masked": dict(arm="masked", visible_weight=0.0),
        "masked_visible": dict(arm="masked", visible_weight=1.0),
        # The pitch: "lower learning rates ... still collapse".
        "masked_lowlr": dict(arm="masked", visible_weight=0.0, lr=3e-4)}


def montage_for(run):
    info = meg_info(run.h5)
    cal = dataset_dir("LibriBrain100") / "metadata" / "neo" / "sss_cal.dat"
    return vectorview_montage(cal, info["channel_names"])


def build(name: str, steps: int, device: str):
    runs = find_runs(task="Sherlock2", subjects={"0"}, require_meg=True)
    if len(runs) != 3:
        raise SystemExit(f"expected sub-0 Sherlock2 sessions 1-3, found {len(runs)}; fetch them with "
                         "data_setup.py fetch --run-key 0,1,Sherlock2,1 ...")
    cfg = PretrainConfig(steps=steps, **ARMS[name])
    recs = load_recordings(runs, heldout_frac=cfg.heldout_frac)
    return Pretrainer(cfg, montage_for(runs[0]), recs, device=device), recs


@torch.no_grad()
def collapse(tr: Pretrainer, n: int = 6):
    """The masked-collapse diagnostic on held-out windows."""
    from neurocast.audit.collapse import diagnose
    from neurocast.objectives.arms import block_mask

    tr.model.eval()
    preds, swaps, tgts, hids = [], [], [], []
    for i in range(n):
        x = tr.batch(20_000_000 + i, train=False)
        y = tr.basis.encode(x.cpu()).to(tr.device)
        g = torch.Generator().manual_seed(i)
        blk = max(1, int(round(tr.cfg.mask_block_s * 250.0 / tr.patch)) * tr.G)
        hid = block_mask((x.shape[0], y.shape[1]), block_len=blk, target_frac=tr.cfg.mask_frac,
                         generator=g, device=tr.device)
        p = tr.head(tr.model(x, mask=hid)).float()
        ps = tr.head(tr.model(x.roll(1, dims=0), mask=hid)).float()   # another sample's context
        real = tr.dim_mask[None].expand_as(y)
        preds.append((p * real).cpu().numpy())
        swaps.append((ps * real).cpu().numpy())
        tgts.append((y * real).cpu().numpy())
        hids.append(hid.cpu().numpy())
    return diagnose(np.concatenate(preds), np.concatenate(swaps), np.concatenate(tgts),
                    np.concatenate(hids), tr.G)


def _read_with_retry(run, vocab, attempts: int = 3):
    """One recording and its word trials, retrying transient read errors.

    A USB drive can drop a read (Windows logs disk warning 153, "the IO
    operation ... was retried") and pandas surfaces it as ``OSError: [Errno 22]
    Invalid argument`` -- 25 minutes into a pass that then has to start over.
    """
    from neurocast.data.libribrain import normalized_meg
    from neurocast.decode.words import label_trials, word_table

    for i in range(attempts):
        try:
            x, info = normalized_meg(run)
            return x, info, label_trials(word_table([run]), vocab)
        except OSError as e:
            if i == attempts - 1:
                raise
            print(f"    {run.h5.name}: {e}; retrying in 10 s", flush=True)
            time.sleep(10)


@torch.no_grad()
def listener_features(models: dict, n_times: int, device, per_listener: int = 12,
                      seed: int = 0, batch: int = 64) -> dict:
    """One read of the 32 broad listeners' chapters 11 and 12, through every model.

    Reading a chapter from the USB data drive dominates the cost; a first
    version re-read all 64 recordings for each model and each analysis, ~45 min
    per model per analysis. Returns, per model, frozen features for

    - the identity audit: ``per_listener`` random 4.096 s windows of chapter 12
      (the same windows for every model), mean-pooled; plus the raw signal's
      per-channel log amplitude on the same windows;
    - the word probe: each 0.512 s word window (8 patches) of chapters 11 and
      12, the group-averaged representation of each patch, concatenated (8 x d).
    """
    from neurocast.data.libribrain import onset_windows

    vocab = json.loads((ROOT / "runs" / "libribrain_audit" / "nobrain.json").read_text())["vocab"]
    rng = np.random.default_rng(seed)
    ident = {k: [] for k in models}
    raw, subj = [], []
    words = {ses: {"x": {k: [] for k in models}, "y": [], "occ": []} for ses in ("11", "12")}
    t0 = time.time()
    for ses in ("12", "11"):                     # chapter 12 first: the identity windows' draw order
        for r in find_runs(task="Sherlock1", sessions={ses}, subjects=BROAD, require_meg=True):
            x, info, t = _read_with_retry(r, vocab)
            if ses == "12":
                starts = rng.integers(0, x.shape[1] - n_times, per_listener)
                w = np.stack([x[:, s:s + n_times] for s in starts])
                for i in range(0, len(w), 8):              # 8 x 64 patches x 306 sensors fits 4 GB
                    wt = torch.as_tensor(w[i:i + 8], device=device)
                    for k, m in models.items():
                        ident[k].append(m.pooled(wt).float().cpu().numpy())
                raw.append(np.log(w.std(axis=2) + 1e-6))           # per-channel log amplitude
                subj += [r.subject] * per_listener
            w, kept = onset_windows(x, info["sfreq"], t["onset"].to_numpy(), decim=1)
            del x
            for i in range(0, len(w), batch):
                xb = torch.as_tensor(w[i:i + batch], device=device)
                for k, m in models.items():
                    g = m.groups(xb).float().mean(2)               # (B, 8, d)
                    words[ses]["x"][k].append(g.reshape(len(xb), -1).cpu().numpy().astype(np.float16))
            words[ses]["y"].append(t["label"].to_numpy()[kept])
            words[ses]["occ"].append(t["occ"].to_numpy()[kept])
            print(f"    sub-{r.subject} ses-{ses}: {kept.sum()} words ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for k in models:
        out[k] = {"identity": np.concatenate(ident[k]),
                  "words": {ses: (np.concatenate(d["x"][k]), np.concatenate(d["y"]), np.concatenate(d["occ"]))
                            for ses, d in words.items()}}
    return {"models": out, "raw": np.concatenate(raw), "subject": np.array(subj)}


def identity(f: np.ndarray, subj: np.ndarray, seed: int = 0) -> dict:
    """Subject eta^2 over its permutation null, on standardised features.

    The ratio is not scale-free: the null's mean is ~(S - 1) / (N - 1) for S
    subjects and N windows, so the ratio can never exceed ``ceiling`` = (N - 1) /
    (S - 1) -- 12.4x at 12 windows per listener, where a first version of this
    script ran, below the pitch's 29x. Compare arms at one N; report N.
    """
    from neurocast.audit.fmscope import subject_variance

    f = (f - f.mean(0)) / (f.std(0) + 1e-6)
    rep = subject_variance(f, subj, n_permutations=100, seed=seed)
    n, n_sub = len(subj), len(np.unique(subj))
    return {"ratio": rep.ratio_to_null, "eta2": rep.eta2_subject, "null": rep.eta2_null_mean,
            "probe": rep.probe_accuracy, "chance": rep.chance_accuracy,
            "n": n, "ceiling": (n - 1) / (n_sub - 1), "summary": rep.summary()}


def word_probe(words: dict, n_classes: int, seed: int = 0) -> dict:
    """Frozen features on the 50-word task: probe fitted on chapter 11, tested on 12.

    Same linear decoder, vocabulary and split as the raw-signal audit.
    """
    from neurocast.decode.decoders import LinearPooled
    from neurocast.decode.metrics import balanced_accuracy_at_k

    (xf, yf, of), (xt, yt, _) = words["11"], words["12"]
    m = LinearPooled(seed=seed).fit(xf.astype(np.float32), yf, of)
    s = m.scores(xt.astype(np.float32), n_classes)
    return {"bacc1": balanced_accuracy_at_k(s, yt, 1), "bacc10": balanced_accuracy_at_k(s, yt, 10),
            "n_fit": len(yf), "n_test": len(yt)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("ar_lik", "masked"), default="ar_lik")
    ap.add_argument("--visible-weight", type=float, default=0.0)
    ap.add_argument("--name", choices=tuple(ARMS), help="train this configuration (overrides --arm)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--evaluate", action="store_true", help="diagnostics on the three checkpoints")
    ap.add_argument("--collapse-only", action="store_true",
                    help="with --evaluate: re-run only the collapse diagnostic, into the saved results")
    ap.add_argument("--per-listener", type=int, default=64,
                    help="identity-audit windows per listener (the ratio's ceiling grows with it)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.evaluate:
        res, models, n_times = {}, {}, None
        saved = OUT / "pilot_eval.json"
        if args.collapse_only:
            res = json.loads(saved.read_text()) if saved.exists() else {}
        for name in ARMS:
            ck = OUT / f"{name}.pt"
            if not ck.exists():
                print(f"  {name}: no checkpoint, skipped")
                continue
            if args.collapse_only and ARMS[name]["arm"] != "masked":
                continue
            tr, recs = build(name, args.steps, args.device)
            tr.load(ck)
            held = [e["heldout"] for e in tr.log if e.get("heldout") is not None]
            r = res.setdefault(name, {})
            r.update({"step": tr.step, "final_heldout": held[-1] if held else None,
                      "best_heldout": min(held) if held else None})
            if tr.cfg.arm == "masked":
                rep = collapse(tr)
                r["collapse"] = {"spread": rep.spread, "sensitivity": rep.sensitivity,
                                 "model_r2": rep.model_r2, "reference_r2": rep.reference_r2,
                                 "isolated_r2": rep.isolated_r2, "collapsed": rep.collapsed}
                print(f"  {name:<15} collapse: {rep.summary()}", flush=True)
            models[name], n_times = tr.model.eval(), tr.n_times
            del tr, recs
            gc.collect()                     # the pilot recordings: ~0.4 GB per arm
        if args.collapse_only:
            saved.write_text(json.dumps(res, indent=2, default=float))
            return 0
        print("  reading the 32 broad listeners once, through every model ...", flush=True)
        feats = listener_features(models, n_times, args.device, per_listener=args.per_listener)
        subj = feats["subject"]
        # The same audit on the first 12 windows of each listener: the ratio's
        # dependence on N, without another read of the data.
        first12 = np.concatenate([np.flatnonzero(subj == s)[:12] for s in dict.fromkeys(subj)])
        raw = {"all": identity(feats["raw"], subj), "12": identity(feats["raw"][first12], subj[first12])}
        n_classes = len(json.loads((ROOT / "runs" / "libribrain_audit" / "nobrain.json").read_text())["vocab"])
        for name, f in feats["models"].items():
            r = res[name]
            r["identity"] = {
                "all": {"features": identity(f["identity"], subj), "raw": raw["all"]},
                "12": {"features": identity(f["identity"][first12], subj[first12]), "raw": raw["12"]},
            }
            for idn in r["identity"].values():
                print(f"  {name:<15} identity: features {idn['features']['ratio']:5.1f}x   "
                      f"raw {idn['raw']['ratio']:5.1f}x   (N {idn['raw']['n']}, ceiling "
                      f"{idn['raw']['ceiling']:.0f}x)", flush=True)
            r["word_probe"] = word_probe(f["words"], n_classes)
            print(f"  {name:<15} frozen-feature word probe: BAcc@1 {r['word_probe']['bacc1']:.3f}  "
                  f"BAcc@10 {r['word_probe']['bacc10']:.3f}", flush=True)
            saved.write_text(json.dumps(res, indent=2, default=float))
        print("  pitch: masked collapsed without the visible term; forecasting 29x vs masked 10x "
              "(raw signal 3x)")
        return 0

    name = args.name or ("ar_lik" if args.arm == "ar_lik" else
                         ("masked_visible" if args.visible_weight > 0 else "masked"))
    t0 = time.time()
    tr, recs = build(name, args.steps, args.device)
    print(f"{name}: {tr.model.describe()}")
    print(f"  device {tr.device}, precision {tr.amp_dtype}, {sum(r.data.shape[1] for r in recs) / 250 / 60:.1f} "
          f"min of story (train {sum(r.train_stop for r in recs) / 250 / 60:.1f} min)")
    tr.fit(OUT / f"{name}.pt")
    print(f"  done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
