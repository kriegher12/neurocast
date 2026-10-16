"""Phase 1 on real MEG: the bake-off, a yardstick with brain signal, data scaling (README 2.6).

Roadmap steps 1, 2, 5 and 6 (README section 18), on the deep listener's hours
(section 2.2's data: sub-0, Sherlock1-9 minus the probe chapters, 61 h train,
8 whole sessions held out):

* ``flops``    measured FLOPs of one training step of each arm (forward, loss,
               backward and, for the latent arms, the EMA teacher's forward). The
               new arms train for the steps that spend what ``ar_lik``'s 20,000
               steps spent;
* ``train``    the two arms section 2.2 lacked -- JEPA (masked latent
               prediction) and causal latent forecasting (``ar_latent``) -- 3
               seeds, everything else as section 2.2;
* ``scale``    data scaling: ``ar_lik`` and ``masked_visible``, seed 0, on 1, 4
               and 16 h of the training sessions, 20,000 steps each (61 h is
               section 2.2's seed 0);
* ``evaluate`` every model, resumably (one JSON per model):
               - the identity audit at N 2,048 (section 2.2's windows, which
                 reproduces its numbers) and the collapse monitor on tokens;
               - step 2: the audit again after regressing per-channel and
                 per-region spectral descriptors of each window out of the
                 features, beside the same descriptors shuffled across windows;
               - the frozen word probe (section 2.2's) for models without one;
               - step 1's yardstick on the deep listener's held-out chapter 12:
                 speech vs silence per 64-ms patch with 0.5, 1 and 3.2 h of labels,
                 39 phonemes and voicing, single trials -- beside the raw signal,
                 shuffled-MEG rows (each session's features rolled by half its
                 length) and no-brain rows (class prior; phoneme timing);
* ``cond``     README 2.10: forecasting with a head conditioned on each recording's
               spectra (``PretrainConfig.condition``), 3 seeds, 20,000 steps; the
               next ``evaluate`` scores it like every other model;
* ``multi``    README 2.14: forecasting and masked + visible pre-trained on the deep
               listener and 20 broad listeners' chapter 11 (the training people of
               ``run_identity_removal.split()``), seeds 0-1; judged by
               ``run_identity_removal.py`` on the 8 people none of it saw;
* ``report``   tables and the verdicts under the rules fixed in README 2.6.

    .venv/Scripts/python.exe     scripts/run_bakeoff_real.py --stage flops
    .venv-gpu/Scripts/python.exe scripts/run_bakeoff_real.py --stage train
    .venv-gpu/Scripts/python.exe scripts/run_bakeoff_real.py --stage evaluate
    .venv-gpu/Scripts/python.exe scripts/run_bakeoff_real.py --stage scale
    .venv-gpu/Scripts/python.exe scripts/run_bakeoff_real.py --stage evaluate
    .venv/Scripts/python.exe     scripts/run_bakeoff_real.py --stage report

Every stage resumes: training from checkpoints, evaluation from per-model JSON.
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

from neurocast.data.paths import dataset_dir  # noqa: E402
from neurocast.train.pretrain import (  # noqa: E402
    LATENT_ARMS,
    PretrainConfig,
    Pretrainer,
    Recording,
    open_cached,
)

OUT = ROOT / "runs" / "bakeoff_real"
SCALE_RUNS = ROOT / "runs" / "pretrain_scale"
BUDGET_STEPS = 20_000
ARMS = {"ar_lik": dict(arm="ar_lik"), "masked_visible": dict(arm="masked", visible_weight=1.0),
        "masked": dict(arm="masked", visible_weight=0.0), "jepa": dict(arm="jepa"),
        "ar_latent": dict(arm="ar_latent"),
        "ar_lik_cond": dict(arm="ar_lik", condition="spectra"),
        "ar_lik_multi": dict(arm="ar_lik"),
        "masked_visible_multi": dict(arm="masked", visible_weight=1.0)}
#: README 2.14: arms pre-trained on many people, and their seeds.
MULTI = [(a, s) for s in (0, 1) for a in ("ar_lik_multi", "masked_visible_multi")]
OLD = [("ar_lik", 0), ("masked_visible", 0), ("ar_lik", 1), ("masked_visible", 1),
       ("ar_lik", 2), ("masked_visible", 2), ("masked", 0)]
NEW = [(a, s) for s in range(3) for a in ("jepa", "ar_latent")]
SCALE_HOURS = (1, 4, 16)
SCALE = [(a, h) for h in SCALE_HOURS for a in ("ar_lik", "masked_visible")]
#: Seeds of the scaling runs: 0 judges (README 2.6); 1 runs after everything else, if
#: time allows, and is reported beside it.
SCALE_SEEDS = (0, 1)
#: Yardstick: label tiers from sub-0 Sherlock1 sessions 1-10 (None: all 3.2 h), the
#: weight decay chosen on chapter 11, the score on chapter 12 (held out of pre-training).
FIT_SESSIONS = tuple(str(i) for i in range(1, 11))
VAL, TEST = "11", "12"
TIERS_H = (0.5, 1.0, None)
STRIDE_FIT = 2                     # every other patch for fitting: neighbours are near-copies
SFREQ = 250.0
N_TIMES = 1024


def feat_dir() -> Path:
    return dataset_dir("derived") / "bakeoff_features"


def eval_dir() -> Path:
    return OUT / "eval"


# -- models ------------------------------------------------------------------------------
def specs() -> dict[str, dict]:
    """Every model the bake-off judges, by name."""
    out = {}
    for arm, seed in OLD:
        out[f"{arm}_s{seed}"] = dict(arm=arm, seed=seed, hours=61.25, source="2.2",
                                     ckpt=SCALE_RUNS / f"{arm}_s{seed}.pt")
    for arm, seed in NEW:
        out[f"{arm}_s{seed}"] = dict(arm=arm, seed=seed, hours=61.25, source="2.6",
                                     ckpt=OUT / f"{arm}_s{seed}.pt")
    for seed in range(3):
        out[f"ar_lik_cond_s{seed}"] = dict(arm="ar_lik_cond", seed=seed, hours=61.25, source="2.10",
                                           ckpt=OUT / "cond" / f"ar_lik_cond_s{seed}.pt")
    for arm, seed in MULTI:
        out[f"{arm}_s{seed}"] = dict(arm=arm, seed=seed, hours=None, source="2.14",
                                     ckpt=OUT / "multi" / f"{arm}_s{seed}.pt")
    for seed in SCALE_SEEDS:
        for arm, h in SCALE:
            out[f"{arm}_s{seed}_h{h}"] = dict(arm=arm, seed=seed, hours=h, source="2.6 scaling",
                                              ckpt=OUT / "scale" / f"{arm}_s{seed}_h{h}.pt")
    return out


def subset_hours(recs: list[Recording], hours: float, seed: int = 0) -> list[Recording]:
    """The first ``hours`` of the training sessions in a fixed shuffled order; held-out kept.

    The last session taken is cut short through ``train_stop``.
    """
    train = [r for r in recs if r.train_stop > 0]
    need = int(round(hours * 3600 * SFREQ))
    out = []
    for i in np.random.default_rng(seed).permutation(len(train)):
        r = train[int(i)]
        take = min(r.n_samples, need)
        if take < N_TIMES:
            break
        out.append(Recording(r.key, r.data, take, r.story, time_major=r.time_major))
        need -= take
        if need <= 0:
            break
    return out + [r for r in recs if r.train_stop == 0]


def broad_training_people() -> list[Recording]:
    """README 2.14: chapter 11 of the 20 broad listeners ``run_identity_removal`` trains on."""
    from run_identity_removal import cache_dir as broad_cache, runs as broad_runs, split

    r11 = broad_runs("11")
    return open_cached(broad_cache(), ["_".join(r11[p].key) for p in split()["train"]])


def make_trainer(arm: str, seed: int, steps: int, device: str, *, hours: float | None = None,
                 prefetch: int = 8) -> Pretrainer:
    from run_pretrain_pilot import montage_for
    from run_pretrain_scale import cache_dir, deep_runs, heldout_keys

    runs = deep_runs()
    keys = ["_".join(r.key) for r in runs]
    recs = open_cached(cache_dir(), keys, heldout=heldout_keys(runs))
    if hours is not None:
        recs = subset_hours(recs, hours)
    if arm.endswith("_multi"):
        recs = recs + broad_training_people()
    cfg = PretrainConfig(steps=steps, seed=seed, eval_every=500, eval_batches=32,
                         prefetch=prefetch, **ARMS[arm])
    return Pretrainer(cfg, montage_for(runs[0]), recs, device=device)


def matched_steps(arm: str) -> int:
    f = OUT / "flops.json"
    if not f.exists():
        raise SystemExit("run --stage flops first")
    return int(json.loads(f.read_text())["steps"][arm])


def train_one(name: str, arm: str, seed: int, steps: int, ck: Path, device: str,
              hours: float | None = None) -> None:
    tr = make_trainer(arm, seed, steps, device, hours=hours)
    if ck.exists():
        tr.load(ck)
    if tr.step >= steps:
        print(f"{name}: finished ({tr.step} steps), skipped", flush=True)
        return
    h = sum(r.train_stop for r in tr.recs) / SFREQ / 3600
    print(f"{name}: {tr.model.describe()}; {h:.2f} h train; {steps} steps; {tr.device} "
          f"{tr.amp_dtype}; calibration windows rejected {tr.calib_rejected}", flush=True)
    t0 = time.time()
    tr.fit(ck)
    print(f"{name}: done in {(time.time() - t0) / 60:.1f} min, best held-out {tr.best:.4f}", flush=True)
    del tr
    gc.collect()
    torch.cuda.empty_cache()


def gpu_cap(frac: float) -> None:
    if frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)


# -- stages: flops, train, scale ----------------------------------------------------------
def stage_flops(args) -> dict:
    from run_pretrain_pilot import montage_for
    from run_pretrain_scale import deep_runs

    from neurocast.train.flops import measure_flops

    montage = montage_for(deep_runs()[0])
    rng = np.random.default_rng(0)
    recs = [Recording(f"r{i}", rng.standard_normal((len(montage), 6000)).astype(np.float16), 4000)
            for i in range(2)]
    per, params = {}, {}
    for arm in ARMS:
        tr = Pretrainer(PretrainConfig(**ARMS[arm]), montage, recs, device="cpu")
        x = tr.batch(0)

        def step():
            tr.opt.zero_grad(set_to_none=True)
            loss, _ = tr.loss(x, 0)
            loss.backward()

        m = measure_flops(step, label=arm)
        n = sum(p.numel() for p in tr.params)
        # EMA update: a multiply and an add per teacher parameter (not a counted op).
        per[arm] = m.total + (4 * sum(p.numel() for p in tr.model.parameters())
                              if arm in LATENT_ARMS else 0)
        params[arm] = n
        print(f"  {arm:<15} {per[arm] / 1e12:7.3f} TFLOP/step   {n:,} trained parameters", flush=True)
        del tr
    budget = per["ar_lik"] * BUDGET_STEPS
    steps = {a: int(round(budget / f)) for a, f in per.items()}
    spread = {a: per[a] * (BUDGET_STEPS if a in ("ar_lik", "masked_visible", "masked") else steps[a])
              / budget - 1 for a in per}
    res = {"per_step": per, "params": params, "budget": budget, "steps": steps,
           "spent_vs_budget": spread}
    for a in per:
        print(f"  {a:<15} matched steps {steps[a]:6d}; as trained, {100 * spread[a]:+.1f}% of the budget")
    (OUT / "flops.json").write_text(json.dumps(res, indent=2))
    return res


def stage_train(args) -> dict:
    gpu_cap(args.gpu_mem_frac)
    for arm, seed in NEW:
        name = f"{arm}_s{seed}"
        train_one(name, arm, seed, matched_steps(arm), OUT / f"{name}.pt", args.device)
    return {}


def stage_multi(args) -> dict:
    gpu_cap(args.gpu_mem_frac)
    (OUT / "multi").mkdir(exist_ok=True)
    for arm, seed in MULTI:
        name = f"{arm}_s{seed}"
        train_one(name, arm, seed, BUDGET_STEPS, OUT / "multi" / f"{name}.pt", args.device)
    return {}


def stage_cond(args) -> dict:
    gpu_cap(args.gpu_mem_frac)
    (OUT / "cond").mkdir(exist_ok=True)
    for seed in range(3):
        name = f"ar_lik_cond_s{seed}"
        train_one(name, "ar_lik_cond", seed, BUDGET_STEPS, OUT / "cond" / f"{name}.pt", args.device)
    return {}


def stage_scale(args) -> dict:
    gpu_cap(args.gpu_mem_frac)
    (OUT / "scale").mkdir(exist_ok=True)
    for arm, h in SCALE:
        name = f"{arm}_s{args.seed}_h{h}"
        train_one(name, arm, args.seed, BUDGET_STEPS, OUT / "scale" / f"{name}.pt", args.device, hours=h)
    return {}


# -- evaluation ---------------------------------------------------------------------------
def load_model(sp: dict, montage, device):
    from neurocast.models.neurocast import NeuroCast

    s = torch.load(Pretrainer.best_path(sp["ckpt"]), map_location="cpu", weights_only=False)
    cfg = PretrainConfig(**s["cfg"])
    torch.manual_seed(0)
    m = NeuroCast(montage, rung=cfg.rung, window=256, causal=cfg.arm not in ("masked", "jepa"),
                  device=device).to(device)
    m.load_state_dict(s["model"])
    return m.eval(), s["step"]


def train_info(sp: dict) -> dict:
    full = torch.load(sp["ckpt"], map_location="cpu", weights_only=False)
    log = full["log"]
    held = [(e["step"], e["heldout"]) for e in log if e.get("heldout") is not None]
    mon = [e["monitor"] for e in log if "monitor" in e]
    best = min(held, key=lambda t: t[1]) if held else (None, None)
    return {"steps_trained": full["step"], "heldout_curve": held, "best_heldout": best[1],
            "best_heldout_step": best[0], "final_heldout": held[-1][1] if held else None,
            "train_monitor": mon[-1] if mon else None}


def valid_positions(montage, n_patch: int = 64) -> np.ndarray:
    from neurocast.tokenizer.descriptor import Quadrant

    g = Quadrant.n_all()
    filled = np.isin(np.arange(g), np.unique(montage.quadrants))
    return np.tile(filled, n_patch)


def read_session(ses: str):
    """sub-0 Sherlock1 ``ses``: (x (T, C) float16 story span, t0 recording-time s, events)."""
    from run_pretrain_scale import cache_dir, deep_runs

    from neurocast.data.libribrain import find_runs, normalized_meg
    from neurocast.train.pretrain import _story_span

    if ses in FIT_SESSIONS:
        r = next(r for r in deep_runs() if r.task == "Sherlock1" and r.session == ses)
        k = "_".join(r.key)
        rec = open_cached(cache_dir(), [k])[0]
        x = np.array(rec.data)                                   # (T, C), one read
        return x, rec.story[0] / SFREQ, r.events
    r = find_runs(task="Sherlock1", sessions={ses}, subjects={"0"}, require_meg=True)[0]
    x, info = normalized_meg(r)
    x = x.astype(np.float16)
    a, b = _story_span(r, x, info["sfreq"])
    return np.ascontiguousarray(x[:, a:b].T), a / SFREQ, r.events


def session_labels(events: Path, n_patches: int, t0: float) -> dict:
    import pandas as pd

    from neurocast.eval.yardstick import patch_index, phoneme_class, speech_labels

    ev = pd.read_csv(events, sep="\t")
    sil = ev[ev["kind"] == "silence"]
    speech = speech_labels(sil["timemeg"].astype(float), sil["duration"].astype(float), n_patches, t0)
    words = ev[ev["kind"] == "word"]
    wkey = {(float(s), float(w)): (float(o), float(d)) for s, w, o, d in
            zip(words["sentenceidx"], words["wordidx"], words["timemeg"], words["duration"])}
    ph = ev[ev["kind"] == "phoneme"]
    cls = np.array([phoneme_class(s) for s in ph["segment"]]).reshape(-1, 2)
    on = ph["timemeg"].astype(float).to_numpy()
    dur = ph["duration"].astype(float).to_numpy()
    wd = np.array([wkey.get((float(s), float(w)), (np.nan, np.nan))
                   for s, w in zip(ph["sentenceidx"], ph["wordidx"])]).reshape(-1, 2)
    keep = (cls[:, 0] >= 0) & (cls[:, 1] >= 0)
    pos = np.eye(4)[np.clip(cls[:, 1], 0, 3)]
    since = np.nan_to_num(on - wd[:, 0])
    timing = np.column_stack([dur, np.log(dur + 1e-3), pos, np.nan_to_num(wd[:, 1]), since,
                              pd.to_numeric(ph["phonemeidx"], errors="coerce").fillna(0).to_numpy()])
    return {"speech": speech, "ph_idx": patch_index(on, t0)[keep], "ph_cls": cls[keep, 0],
            "ph_timing": timing[keep].astype(np.float32)}


def patch_features(models: dict, x: np.ndarray, device) -> tuple[dict, np.ndarray]:
    """Per-patch, group-averaged features of every model over a session; and a valid mask.

    4.1-s windows every 2.05 s; each keeps its middle 2.05 s (patches 16-47), so every
    feature has seen at least 1 s before it, causal and bidirectional models alike.
    """
    p = x.shape[0] // 16
    feats = {k: np.zeros((p, m.d_model), np.float16) for k, m in models.items()}
    valid = np.zeros(p, bool)
    starts = list(range(0, p - 64 + 1, 32))
    with torch.no_grad():
        for i in range(0, len(starts), 8):
            st = starts[i:i + 8]
            xb = torch.as_tensor(np.stack([x[16 * s:16 * (s + 64)].T for s in st]).astype(np.float32),
                                 device=device)
            for k, m in models.items():
                g = m.groups(xb).float().mean(2)[:, 16:48].cpu().numpy().astype(np.float16)
                for j, s in enumerate(st):
                    feats[k][s + 16:s + 48] = g[j]
            for s in st:
                valid[s + 16:s + 48] = True
    return feats, valid


def stage_evaluate(args) -> dict:
    from run_pretrain_pilot import BROAD, _read_with_retry, identity, montage_for
    from run_pretrain_scale import deep_runs

    from neurocast.data.libribrain import find_runs, onset_windows
    from neurocast.eval.yardstick import (cross_listener, region_means, spread, top_pcs,
                                          window_descriptors)
    from neurocast.objectives.arms import CollapseMonitor

    gpu_cap(args.gpu_mem_frac)
    eval_dir().mkdir(parents=True, exist_ok=True)
    # README 2.14's models trained on 20 of the 32 broad listeners: their identity
    # audit over all 32 would mix seen and unseen people; run_identity_removal judges them.
    sp = {k: v for k, v in specs().items() if Pretrainer.best_path(v["ckpt"]).exists()
          and v["source"] != "2.14"}
    if args.only:
        sp = {k: v for k, v in sp.items() if k in args.only}
    # A run still training has a .best.pt but no final step yet: evaluate only finished runs.
    done = {}
    for k, v in sp.items():
        f = eval_dir() / f"{k}.json"
        res = json.loads(f.read_text()) if f.exists() else {}
        if "train" not in res:
            res["train"] = train_info(v)
        want = (0 if v["source"] == "2.2" else          # finished in section 2.2
                BUDGET_STEPS if v["arm"] in ("ar_lik", "masked_visible", "ar_lik_cond")
                else matched_steps(v["arm"]))
        if res["train"]["steps_trained"] < want:
            print(f"  {k}: {res['train']['steps_trained']} of {want} steps, not evaluated yet")
            continue
        res.update({kk: v[kk] for kk in ("arm", "seed", "hours", "source")})
        done[k] = res
    old = json.loads((SCALE_RUNS / "scale_eval.json").read_text())
    for k, res in done.items():
        if sp[k]["source"] == "2.2" and "word_probe" not in res:
            res["word_probe"] = {**old[k]["word_probe"], "from": "runs/pretrain_scale/scale_eval.json"}
    need_ident = [k for k, r in done.items() if "nuisance" not in r]
    need_words = [k for k, r in done.items() if "word_probe" not in r]
    need_yard = [k for k, r in done.items() if "yardstick" not in r]
    print(f"  models: {len(done)}; identity {len(need_ident)}, word probe {len(need_words)}, "
          f"yardstick {len(need_yard)}", flush=True)
    load = sorted(set(need_ident) | set(need_words) | set(need_yard))
    if not load:
        return verdicts(done)

    montage = montage_for(deep_runs()[0])
    models, steps = {}, {}
    for k in load:
        models[k], steps[k] = load_model(sp[k], montage, args.device)
    vpos = torch.as_tensor(valid_positions(montage), device=args.device)
    vocab = json.loads((ROOT / "runs" / "libribrain_audit" / "nobrain.json").read_text())["vocab"]

    # Pass A: the 32 broad listeners. Chapter 12: identity windows (section 2.2's
    # draw) and their spectral descriptors; chapters 11 and 12: word windows.
    if need_ident or need_words:
        rng = np.random.default_rng(0)
        ident = {k: [] for k in need_ident}
        toks = {k: [] for k in need_ident}
        raw, desc, subj = [], [], []
        wdir = feat_dir() / "words"
        for k in need_words:
            (wdir / k).mkdir(parents=True, exist_ok=True)
        (wdir / "_labels").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for ses in ("12", "11"):
            if ses == "11" and not need_words:
                break
            for r in find_runs(task="Sherlock1", sessions={ses}, subjects=BROAD, require_meg=True):
                x, info, t = _read_with_retry(r, vocab)
                if ses == "12":
                    starts = rng.integers(0, x.shape[1] - N_TIMES, args.per_listener)
                    if need_ident:
                        w = np.stack([x[:, s:s + N_TIMES] for s in starts])
                        with torch.no_grad():
                            for i in range(0, len(w), 8):
                                wt = torch.as_tensor(w[i:i + 8], device=args.device)
                                for k in need_ident:
                                    h = models[k](wt)
                                    ident[k].append(h.mean(1).float().cpu().numpy())
                                    if len(toks[k]) < 8:
                                        toks[k].append(h[:, vpos].reshape(-1, h.shape[-1]).float().cpu())
                        raw.append(np.log(w.std(axis=2) + 1e-6))
                        desc.append(window_descriptors(w).astype(np.float32))
                        subj += [r.subject] * args.per_listener
                        del w
                if need_words:
                    ww, kept = onset_windows(x, info["sfreq"], t["onset"].to_numpy(), decim=1)
                    del x
                    with torch.no_grad():
                        for k in need_words:
                            parts = []
                            for i in range(0, len(ww), 64):
                                g = models[k].groups(torch.as_tensor(ww[i:i + 64], device=args.device))
                                g = g.float().mean(2)
                                parts.append(g.reshape(len(g), -1).cpu().numpy().astype(np.float16))
                            np.save(wdir / k / f"{ses}_{r.subject}.npy", np.concatenate(parts))
                    np.savez(wdir / "_labels" / f"{ses}_{r.subject}.npz",
                             y=t["label"].to_numpy()[kept], occ=np.asarray(t["occ"].astype(str).to_numpy()[kept], dtype=str))
                    del ww
                print(f"    sub-{r.subject} ses-{ses} ({time.time() - t0:.0f} s)", flush=True)
        if need_ident:
            subj = np.array(subj)
            raw = np.concatenate(raw)
            desc = np.concatenate(desc)
            n = len(subj)
            d_region = region_means(desc, montage.quadrants)
            d_chan = top_pcs(desc.reshape(n, -1), args.n_pcs)
            perm = np.random.default_rng(1).permutation(n)
            base = {"raw": identity(raw, subj), "descriptors_region": identity(d_region, subj),
                    "descriptors_channel_pcs": identity(d_chan, subj)}
            for k in need_ident:
                f = np.concatenate(ident[k])
                res = done[k]
                res["identity"] = {"features": identity(f, subj), "raw": base["raw"]}
                if sp[k]["source"] == "2.2":
                    res["identity"]["section_2_2_ratio"] = old[k]["identity"]["features"]["ratio"]
                nu = {"descriptors": {kk: v["ratio"] for kk, v in base.items()}}
                for nm, dd in (("channel_pcs", d_chan), ("region", d_region)):
                    cl, cc = cross_listener(f, dd, subj), cross_listener(f, dd[perm], subj)
                    nu[nm] = {"r2_between": cl["r2_between"], "alpha": cl["alpha"],
                              "control_r2_between": cc["r2_between"],
                              "residual": identity(cl["residual"], subj)["ratio"],
                              "shuffled_control": identity(cc["residual"], subj)["ratio"]}
                res["nuisance"] = nu
                z = torch.cat(toks[k])
                res["monitor"] = {**{kk: float(v) for kk, v in CollapseMonitor().check(z).items()
                                     if kk != "abort"}, **spread(z.numpy())}
                (eval_dir() / f"{k}.json").write_text(json.dumps(res, indent=2, default=float))
                print(f"  {k:<24} identity {res['identity']['features']['ratio']:5.1f}x"
                      + (f" (2.2: {res['identity']['section_2_2_ratio']:.1f}x)" if sp[k]['source'] == '2.2' else "")
                      + f"; spectra predict listener means R2 {nu['channel_pcs']['r2_between']:.2f} "
                        f"(control {nu['channel_pcs']['control_r2_between']:.2f}); net of them "
                        f"{nu['channel_pcs']['residual']:.1f}x (control {nu['channel_pcs']['shuffled_control']:.1f}x); "
                        f"varying {res['monitor']['varying']:.3f}, "
                        f"participation {res['monitor']['participation']:.1f}", flush=True)
        if need_words:
            from run_pretrain_pilot import word_probe

            runs = {s: find_runs(task="Sherlock1", sessions={s}, subjects=BROAD, require_meg=True)
                    for s in ("11", "12")}
            for k in need_words:
                words = {}
                for s in ("11", "12"):
                    lab = [np.load(wdir / "_labels" / f"{s}_{r.subject}.npz") for r in runs[s]]
                    words[s] = (np.concatenate([np.load(wdir / k / f"{s}_{r.subject}.npy") for r in runs[s]]),
                                np.concatenate([z["y"] for z in lab]), np.concatenate([z["occ"] for z in lab]))
                done[k]["word_probe"] = word_probe(words, len(vocab))
                del words
                gc.collect()
                (eval_dir() / f"{k}.json").write_text(json.dumps(done[k], indent=2, default=float))
                print(f"  {k:<24} word probe BAcc@10 {done[k]['word_probe']['bacc10']:.3f}", flush=True)

    # Pass B: the deep listener's Sherlock1 sessions 1-12, per-patch features.
    if need_yard:
        ydir = feat_dir() / "yard"
        for k in need_yard + ["_raw"]:
            (ydir / k).mkdir(parents=True, exist_ok=True)
        meta = {}
        for ses in FIT_SESSIONS + (VAL, TEST):
            x, t0, events = read_session(ses)
            feats, valid = patch_features({k: models[k] for k in need_yard}, x, args.device)
            for k, f in feats.items():
                np.save(ydir / k / f"{ses}.npy", f)
            p = len(valid)
            np.save(ydir / "_raw" / f"{ses}.npy",
                    x[:p * 16].reshape(p, 16, -1).astype(np.float32).mean(1).astype(np.float16))
            meta[ses] = {"t0": t0, "events": str(events), "n_patches": p, "valid": valid}
            print(f"    yardstick features: Sherlock1 ses-{ses}, {p * 0.064 / 60:.1f} min", flush=True)
            del x, feats
        del models
        gc.collect()
        torch.cuda.empty_cache()
        labels = {s: session_labels(Path(m["events"]), m["n_patches"], m["t0"]) for s, m in meta.items()}
        valid = {s: m["valid"] for s, m in meta.items()}
        nobrain_f = OUT / "yardstick_nobrain.json"
        if not nobrain_f.exists():
            nb = yardstick_nobrain(labels, valid, args.device)
            nb["raw"] = yardstick(lambda s: np.load(ydir / "_raw" / f"{s}.npy"), labels, valid,
                                  args.device, shuffled=True)
            nobrain_f.write_text(json.dumps(nb, indent=2, default=float))
            print_yard("raw signal", nb["raw"])
        for k in need_yard:
            res = done[k]
            res["yardstick"] = yardstick(lambda s, k=k: np.load(ydir / k / f"{s}.npy"), labels, valid,
                                         args.device, shuffled=k.endswith("_s0") and k.count("_h") == 0)
            (eval_dir() / f"{k}.json").write_text(json.dumps(res, indent=2, default=float))
            print_yard(k, res["yardstick"])
    return verdicts(done)


def _fit_rows(feats, labels, valid, sessions, kind, *, max_hours=None, stride=1):
    """Lagged features and labels over ``sessions``; ``kind`` "speech" or "phoneme"."""
    from neurocast.eval.yardstick import PATCH_S, lagged, usable

    xs, ys, ts, got = [], [], [], 0.0
    for s in sessions:
        f, v, lab = feats(s), valid[s], labels[s]
        if kind == "speech":
            idx = np.arange(0, len(v), stride)
            idx = idx[usable(idx, v)]
            y = lab["speech"][idx]
            if max_hours is not None:
                room = int((max_hours - got) * 3600 / (PATCH_S * stride))
                idx, y = idx[:room], y[:room]
                got += len(idx) * PATCH_S * stride / 3600
            xs.append(lagged(f, idx))
            ys.append(y)
        else:
            ok = usable(lab["ph_idx"], v)
            xs.append(lagged(f, lab["ph_idx"][ok]))
            ys.append(lab["ph_cls"][ok])
            ts.append(lab["ph_timing"][ok])
        if max_hours is not None and got >= max_hours - 1e-9:
            break
    out = (np.concatenate(xs), np.concatenate(ys))
    return out + ((np.concatenate(ts),) if kind != "speech" else ())


def _score(probe, x, y, n_classes) -> dict:
    from neurocast.eval.yardstick import auc, balanced_accuracy, macro_f1

    s = probe.scores(x)
    pred = s.argmax(1)
    out = {"f1": macro_f1(pred, y, n_classes), "bacc": balanced_accuracy(pred, y, n_classes),
           "acc": float((pred == y).mean()), "n_test": int(len(y))}
    if n_classes == 2:
        out["auc"] = auc(s[:, 1] - s[:, 0], y)
    return out


def yardstick(feats, labels, valid, device, *, shuffled: bool = False) -> dict:
    """Speech at each label tier, phonemes and voicing on chapter 12; optionally shuffled rows."""
    from neurocast.eval.yardstick import PHONEMES, VOICED, fit_select

    voiced = np.array([p in VOICED for p in PHONEMES])
    out = {}
    variants = [("", feats, valid)]
    if shuffled:
        def rolled(s, feats=feats):
            f = feats(s)
            return np.roll(f, len(f) // 2, axis=0)
        variants.append(("shuffled_", rolled, {s: np.roll(v, len(v) // 2) for s, v in valid.items()}))
    for pre, ff, vv in variants:
        xv, yv = _fit_rows(ff, labels, vv, [VAL], "speech", stride=STRIDE_FIT)
        xt, yt = _fit_rows(ff, labels, vv, [TEST], "speech")
        tiers = TIERS_H if not pre else (1.0,)
        for h in tiers:
            xf, yf = _fit_rows(ff, labels, vv, FIT_SESSIONS, "speech", max_hours=h, stride=STRIDE_FIT)
            probe, sel = fit_select(xf, yf, xv, yv, 2, device=device)
            row = {**_score(probe, xt, yt, 2), **sel, "n_fit": int(len(yf)),
                   "hours": len(yf) * STRIDE_FIT * 0.064 / 3600}
            out[f"{pre}speech_{'all' if h is None else h}h"] = row
            del xf, yf, probe
        del xv, yv, xt, yt
        xf, yf, _ = _fit_rows(ff, labels, vv, FIT_SESSIONS, "phoneme")
        xv, yv, _ = _fit_rows(ff, labels, vv, [VAL], "phoneme")
        xt, yt, _ = _fit_rows(ff, labels, vv, [TEST], "phoneme")
        probe, sel = fit_select(xf, yf, xv, yv, len(PHONEMES), device=device)
        out[f"{pre}phoneme"] = {**_score(probe, xt, yt, len(PHONEMES)), **sel, "n_fit": int(len(yf))}
        probe, sel = fit_select(xf, voiced[yf].astype(int), xv, voiced[yv].astype(int), 2, device=device)
        out[f"{pre}voicing"] = {**_score(probe, xt, voiced[yt].astype(int), 2), **sel, "n_fit": int(len(yf))}
        del xf, xv, xt, probe
        gc.collect()
    return out


def yardstick_nobrain(labels, valid, device) -> dict:
    """Rows that need no brain: the class prior, and a probe on phoneme timing alone."""
    from neurocast.eval.yardstick import PHONEMES, VOICED, fit_select

    voiced = np.array([p in VOICED for p in PHONEMES])
    ones = lambda s: np.ones((len(valid[s]), 1), np.float16)  # noqa: E731
    _, yt = _fit_rows(ones, labels, valid, [TEST], "speech")
    maj = np.ones_like(yt)
    from neurocast.eval.yardstick import balanced_accuracy, macro_f1

    out = {"speech_prior": {"f1": macro_f1(maj, yt, 2), "bacc": balanced_accuracy(maj, yt, 2),
                            "auc": 0.5, "speech_fraction": float(yt.mean())}}
    _, yf, tf = _fit_rows(ones, labels, valid, FIT_SESSIONS, "phoneme")
    _, yv, tv = _fit_rows(ones, labels, valid, [VAL], "phoneme")
    _, yt, tt = _fit_rows(ones, labels, valid, [TEST], "phoneme")
    top = np.bincount(yf, minlength=len(PHONEMES)).argmax()
    out["phoneme_prior"] = {"f1": macro_f1(np.full_like(yt, top), yt, len(PHONEMES)),
                            "bacc": 1 / len(np.unique(yt)), "chance_f1_uniform": 1 / len(PHONEMES)}
    probe, sel = fit_select(tf, yf, tv, yv, len(PHONEMES), device=device)
    out["phoneme_timing"] = {**_score(probe, tt, yt, len(PHONEMES)), **sel}
    probe, sel = fit_select(tf, voiced[yf].astype(int), tv, voiced[yv].astype(int), 2, device=device)
    out["voicing_timing"] = {**_score(probe, tt, voiced[yt].astype(int), 2), **sel}
    out["voicing_prior"] = {"bacc": 0.5, "voiced_fraction": float(voiced[yt].mean())}
    return out


def print_yard(name: str, y: dict) -> None:
    print(f"  {name:<24} speech F1 " + " / ".join(
        f"{y[k]['f1']:.3f}" for k in ("speech_0.5h", "speech_1.0h", "speech_allh")) +
        f" (0.5/1/3.2 h)   phonemes F1 {y['phoneme']['f1']:.4f}   voicing BAcc {y['voicing']['bacc']:.3f}"
        + (f"   shuffled: speech {y['shuffled_speech_1.0h']['f1']:.3f}, phonemes "
           f"{y['shuffled_phoneme']['f1']:.4f}, voicing {y['shuffled_voicing']['bacc']:.3f}"
           if "shuffled_phoneme" in y else ""), flush=True)


# -- verdicts -----------------------------------------------------------------------------
TASKS = {"speech_0.5h": ("yardstick", "speech_0.5h", "f1"), "speech_3.2h": ("yardstick", "speech_allh", "f1"),
         "phonemes": ("yardstick", "phoneme", "f1"), "voicing": ("yardstick", "voicing", "bacc"),
         "words": ("word_probe", None, "bacc10")}


def _get(r: dict, task: str):
    top, key, metric = TASKS[task]
    try:
        v = r[top] if key is None else r[top][key]
        return float(v[metric])
    except (KeyError, TypeError):
        return None


def verdicts(done: dict) -> dict:
    """The rules fixed in README 2.6 before any of these runs finished."""
    out: dict = {}
    main = {k: r for k, r in done.items() if r.get("source") in ("2.2", "2.6")}
    ref = {}
    for m in ("varying", "participation"):
        h = [r["monitor"][m] for r in main.values()
             if r["arm"] in ("ar_lik", "masked_visible") and "monitor" in r]
        ref[m] = float(np.median(h)) if h else None
    out["collapse_reference"] = ref
    collapsed = {k: bool(ref["varying"] and "monitor" in r and r["arm"] in LATENT_ARMS
                         and any(r["monitor"][m] < 0.25 * ref[m] for m in ref))
                 for k, r in main.items()}
    out["collapsed"] = sorted(k for k, c in collapsed.items() if c)

    def seeds(arm):
        return sorted(r["seed"] for k, r in main.items() if r["arm"] == arm and not collapsed[k])

    def val(arm, seed, task):
        return _get(main.get(f"{arm}_s{seed}", {}), task)

    comp = {}
    for arm in ("masked_visible", "jepa", "ar_latent"):
        sd = sorted(set(seeds("ar_lik")) & set(seeds(arm)))
        row = {"seeds": sd}
        for task in TASKS:
            d = [val("ar_lik", s, task) - val(arm, s, task) for s in sd
                 if val("ar_lik", s, task) is not None and val(arm, s, task) is not None]
            if len(d) < 2:
                row[task] = "not enough seeds"
                continue
            m = float(np.mean(d))
            row[task] = ("equal" if abs(m) < 0.01 else "ar_lik higher" if all(x > 0 for x in d)
                         else f"{arm} higher" if all(x < 0 for x in d) else "no consistent difference")
            row[f"{task}_mean_difference"] = m
        idr = [main[f"ar_lik_s{s}"]["identity"]["features"]["ratio"] /
               main[f"{arm}_s{s}"]["identity"]["features"]["ratio"] for s in sd
               if "identity" in main.get(f"ar_lik_s{s}", {}) and "identity" in main.get(f"{arm}_s{s}", {})]
        row["identity_ratio_ar_lik_over_arm"] = idr
        row["identity"] = ("ar_lik leaks more" if idr and all(x > 1.25 for x in idr) else
                           "ar_lik leaks less" if idr and all(x < 0.8 for x in idr) else
                           "no consistent difference" if idr else "not measured")
        comp[arm] = row
    out["ar_lik_vs"] = comp
    brain = ("speech_0.5h", "phonemes", "voicing")
    beats = {a: sum(comp[a].get(t) == f"{a} higher" for t in brain) for a in comp}
    lost = {a: sum(comp[a].get(t) == "ar_lik higher" for t in brain) for a in comp}
    ok_all = all(comp[a].get(t) in ("equal", "ar_lik higher") for a in comp for t in brain)
    out["H1"] = ("killed on this yardstick" if any(v >= 2 for v in beats.values()) else
                 "supported" if ok_all and any(v >= 1 for v in lost.values()) else "undecided")

    # Step 2: what the forecaster's person information is.
    ex = [main[f"ar_lik_s{s}"]["nuisance"]["channel_pcs"]["r2_between"] for s in range(3)
          if "nuisance" in main.get(f"ar_lik_s{s}", {})]
    out["step2_r2_between_ar_lik"] = ex
    ctl = [r["nuisance"]["channel_pcs"]["control_r2_between"] for r in main.values() if "nuisance" in r]
    out["step2_control_r2_max"] = max(ctl) if ctl else None
    out["step2"] = ("invalid: shuffled control predicts" if ctl and max(ctl) >= 0.1 else
                    "spectral calibration explains most" if ex and all(e >= 0.5 for e in ex) else
                    "explains some" if ex and all(e >= 0.25 for e in ex) else
                    "not spectral calibration" if ex and all(e < 0.25 for e in ex) else
                    "mixed" if ex else "not measured")
    gap = [main[f"ar_lik_s{s}"]["nuisance"]["channel_pcs"]["residual"] /
           main[f"masked_visible_s{s}"]["nuisance"]["channel_pcs"]["residual"] for s in range(3)
           if "nuisance" in main.get(f"ar_lik_s{s}", {}) and "nuisance" in main.get(f"masked_visible_s{s}", {})]
    out["step2_gap_after"] = gap
    out["step2_gap_closes"] = bool(gap) and all(g < 1.5 for g in gap)

    # Yardstick sanity: a task carries brain signal if the raw signal beats its shuffled row.
    nb = OUT / "yardstick_nobrain.json"
    if nb.exists():
        raw = json.loads(nb.read_text())["raw"]
        gap = lambda k, m, thr: (raw[k][m] - raw[f"shuffled_{k}"][m] >= thr  # noqa: E731
                                 if k in raw and f"shuffled_{k}" in raw else None)
        out["brain_signal"] = {"speech": gap("speech_1.0h", "f1", 0.02),
                               "phonemes": gap("phoneme", "f1", 0.005),
                               "voicing": gap("voicing", "bacc", 0.02)}

    # README 2.10: the forecaster with a spectra-conditioned head, against the plain one.
    cond = [s for s in range(3) if "identity" in done.get(f"ar_lik_cond_s{s}", {})
            and "identity" in done.get(f"ar_lik_s{s}", {})]
    if len(cond) >= 2:
        c, f = ({s: done[f"{a}_s{s}"] for s in cond} for a in ("ar_lik_cond", "ar_lik"))
        ratio = [c[s]["identity"]["features"]["ratio"] / f[s]["identity"]["features"]["ratio"] for s in cond]
        row = {"seeds": cond, "identity_cond_over_plain": ratio,
               "identity": ("reduces" if all(x <= 0.75 for x in ratio) else
                            "raises" if all(x >= 1.25 for x in ratio) else "no consistent change"),
               "heldout_cond_minus_plain": [c[s]["train"]["best_heldout"] - f[s]["train"]["best_heldout"]
                                            for s in cond],
               "spectra_r2_cond": [c[s].get("nuisance", {}).get("channel_pcs", {}).get("r2_between")
                                   for s in cond]}
        for task in TASKS:
            d = [_get(c[s], task) - _get(f[s], task) for s in cond
                 if _get(c[s], task) is not None and _get(f[s], task) is not None]
            if len(d) >= 2:
                m = float(np.mean(d))
                row[task] = ("equal" if abs(m) < 0.01 else "conditioned higher" if all(x > 0 for x in d)
                             else "plain higher" if all(x < 0 for x in d) else "no consistent difference")
        mv = [done[f"masked_visible_s{s}"]["identity"]["features"]["ratio"] for s in cond
              if "identity" in done.get(f"masked_visible_s{s}", {})]
        row["below_masked"] = bool(mv) and all(c[s]["identity"]["features"]["ratio"] < m_
                                               for s, m_ in zip(cond, mv))
        out["conditioned_forecaster"] = row

    # Data scaling: seed 0 judges (exploratory); seed 1, if it ran, is reported beside it.
    for seed in SCALE_SEEDS:
        sfx = "" if seed == 0 else f"_seed{seed}"
        sc = {}
        for arm in ("ar_lik", "masked_visible"):
            pts = ([(h, done.get(f"{arm}_s{seed}_h{h}")) for h in SCALE_HOURS]
                   + [(61.25, done.get(f"{arm}_s{seed}"))])
            pts = [(h, r) for h, r in pts if r]
            sc[arm] = {"hours": [h for h, _ in pts],
                       "best_heldout": [r["train"]["best_heldout"] for _, r in pts],
                       "identity": [r.get("identity", {}).get("features", {}).get("ratio") for _, r in pts],
                       **{t: [_get(r, t) for _, r in pts] for t in TASKS}}
        out[f"scaling{sfx}"] = sc
        a, m = sc.get("ar_lik", {}), sc.get("masked_visible", {})
        if a.get("hours") == m.get("hours") and a.get("hours") and len(a["hours"]) == 4 \
                and None not in a["identity"] + m["identity"]:
            g = np.array(a["identity"]) / np.array(m["identity"])
            out[f"scaling_identity_gap{sfx}"] = g.tolist()
            out[f"scaling_gap_trend{sfx}"] = ("grows with data" if np.all(np.diff(g) > 0) else
                                              "shrinks with data" if np.all(np.diff(g) < 0)
                                              else "no monotone trend")
        for arm, v_ in sc.items():
            for t in ("speech_0.5h", "phonemes"):
                v = [x for x in v_.get(t, []) if x is not None]
                if len(v) == 4:
                    out.setdefault(f"scaling_helps{sfx}", {})[f"{arm}_{t}"] = bool(np.all(np.diff(v) > 0))
    (OUT / "verdicts.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n  verdicts (README 2.6 rules):")
    for k, v in out.items():
        print(f"    {k}: {v}")
    return out


def stage_report(args) -> dict:
    rows = {}
    for f in sorted(eval_dir().glob("*.json")):
        rows[f.stem] = json.loads(f.read_text())
    nb = json.loads((OUT / "yardstick_nobrain.json").read_text()) if (OUT / "yardstick_nobrain.json").exists() else {}
    print("| Model | h | Steps | Held-out | Identity | Net of spectra (control) | BAcc@10 | "
          "Speech F1 0.5 / 1 / 3.2 h | Phonemes F1 | Voicing BAcc |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, r in rows.items():
        y, i, n = r.get("yardstick", {}), r.get("identity", {}), r.get("nuisance", {}).get("channel_pcs", {})
        g = lambda d, kk, m: f"{d[kk][m]:.3f}" if kk in d else "—"  # noqa: E731
        print(f"| {k} | {r.get('hours', 0):.0f} | {r['train']['steps_trained']} | "
              f"{r['train']['best_heldout']:.4f} | "
              f"{i.get('features', {}).get('ratio', float('nan')):.1f}x | "
              f"{n.get('residual', float('nan')):.1f}x ({n.get('shuffled_control', float('nan')):.1f}x) | "
              f"{r.get('word_probe', {}).get('bacc10', float('nan')):.3f} | "
              f"{g(y, 'speech_0.5h', 'f1')} / {g(y, 'speech_1.0h', 'f1')} / {g(y, 'speech_allh', 'f1')} | "
              f"{g(y, 'phoneme', 'f1')} | {g(y, 'voicing', 'bacc')} |")
    if nb:
        print("\nno-brain and raw rows:", json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()
                                                         if isinstance(vv, float)}
                                                     for k, v in nb.items() if k != "raw"}))
        print_yard("raw signal", nb["raw"])
    if (OUT / "verdicts.json").exists():
        print("\nverdicts:", (OUT / "verdicts.json").read_text())
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("flops", "train", "scale", "cond", "multi", "evaluate", "report"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.9)
    ap.add_argument("--per-listener", type=int, default=64)
    ap.add_argument("--n-pcs", type=int, default=64, help="per-channel descriptor PCs regressed out")
    ap.add_argument("--only", nargs="*", help="evaluate only these models")
    ap.add_argument("--seed", type=int, default=0, help="--stage scale: the training seed")
    ap.add_argument("--out", type=Path, help="run directory (default runs/bakeoff_real)")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    {"flops": stage_flops, "train": stage_train, "scale": stage_scale, "cond": stage_cond,
     "multi": stage_multi,
     "evaluate": stage_evaluate, "report": stage_report}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
