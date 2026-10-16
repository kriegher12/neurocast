"""Supervised decoders at scale: speech and phonemes from 1 to 60 h of labels (README 2.13).

Sections 2.6-2.9 judged frozen and briefly fine-tuned models on at most 3.2 h of
labels. The 2025 entries trained on ~50 h. This trains one network per run on
both tasks at once, from the deep listener's labelled sessions, and reports every
score beside the rows that need no brain:

* labels: the sessions section 2.2 pre-trained on (sub-0, Sherlock1-9 minus the
  probe chapters and the 8 held-out sessions), minus Sherlock4 chapter 8 (bad
  annotations, README 9); 1, 4 and 16 h in a fixed random session order
  (``run_bakeoff_real.subset_hours``), or all ~60 h;
* starts: scratch, bidirectional and causal; section 2.2's masked + visible and
  forecasting checkpoints (pre-trained, unlabelled, on the same sessions);
* model: the whole network on 4.1-s windows. A speech head reads, for each of the
  window's middle 32 patches, the features of the 4 patches after it (section
  2.7's head); a phoneme head reads the same 4 patches after each phoneme onset
  in the middle 32 patches and names one of 39 phonemes. Voicing is read off the
  phoneme posterior;
* recipe: AdamW at 1e-4 (section 2.7's choice), 500 warm-up steps, cosine to 10%,
  batch 8, 20,000 steps, class-balanced losses summed, bf16. Chapter 11 is
  checked every 1,000 steps; each task keeps its own best step; chapter 12 scores
  it once;
* scores: speech macro-F1 and AUC; phoneme macro-F1 and balanced accuracy;
  voicing balanced accuracy; and each as nats per trial beyond its no-brain
  model (speech: the class prior; phonemes and voicing: a probe on phoneme timing
  fitted on the same labels), with a bootstrap over 10-s blocks or words.

Every run checkpoints every 2,500 steps to the data drive and resumes exactly.

    .venv/Scripts/python.exe     scripts/run_supervised.py --stage selftest
    .venv-gpu/Scripts/python.exe scripts/run_supervised.py --stage train --plan main
    .venv-gpu/Scripts/python.exe scripts/run_supervised.py --stage train --plan later
    .venv/Scripts/python.exe     scripts/run_supervised.py --stage report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from neurocast.data.paths import dataset_dir, replace_file  # noqa: E402

OUT = ROOT / "runs" / "supervised"
STARTS = ("scratch_bidir", "scratch_causal", "masked_visible", "ar_lik")
HOURS = (1, 4, 16, None)                     # None: every labelled session (~60 h)
#: Run order: every start at all labels first (the headline), then 1, 16 and 4 h, then a
#: second seed of the two ends; "later" adds the second seed of the middle points.
PLANS = {
    "main": [(s, h, 0) for h in (None, 1, 16, 4) for s in STARTS]
    + [(s, h, 1) for h in (None, 1) for s in STARTS],
    "later": [(s, h, 1) for h in (16, 4) for s in STARTS],
}
EXCLUDE = {"0_8_Sherlock4_1"}                # bad annotations (README 9)
STEPS, WARMUP, LR, BATCH = 20_000, 500, 1e-4, 8
EVAL_EVERY, CKPT_EVERY = 1000, 2500
N_PATCH, KEEP, PATCH, LAGS = 64, (16, 48), 16, (1, 2, 3, 4)
N_PH = 39


def ckpt_dir() -> Path:
    return dataset_dir("derived") / "supervised_ckpt"


def run_name(start: str, hours, seed: int) -> str:
    return f"{start}_{'all' if hours is None else f'{hours}h'}_s{seed}"


# -- data ----------------------------------------------------------------------------------
def labels_for(events, n_patches: int, t0: float) -> dict:
    """Speech per patch; phonemes with patch, class, timing features and a word key."""
    import pandas as pd
    from run_bakeoff_real import session_labels

    lab = session_labels(events, n_patches, t0)
    from neurocast.eval.yardstick import phoneme_class

    ev = pd.read_csv(events, sep="\t")
    ph = ev[ev["kind"] == "phoneme"]
    cls = np.array([phoneme_class(x) for x in ph["segment"]]).reshape(-1, 2)
    keep = (cls[:, 0] >= 0) & (cls[:, 1] >= 0)
    word = (pd.to_numeric(ph["sentenceidx"], errors="coerce").fillna(-1).to_numpy() * 10_000
            + pd.to_numeric(ph["wordidx"], errors="coerce").fillna(-1).to_numpy())[keep]
    lab["ph_word"] = word
    return lab


class Session:
    """One labelled session: a window reader and its labels, cut to ``n_patches``."""

    def __init__(self, window, n_patches: int, lab: dict, name: str) -> None:
        self.window, self.n, self.name = window, n_patches, name
        self.speech = lab["speech"][:n_patches]
        ok = (lab["ph_idx"] >= 0) & (lab["ph_idx"] < n_patches)
        self.ph_idx, self.ph_cls = lab["ph_idx"][ok], lab["ph_cls"][ok]
        self.ph_timing, self.ph_word = lab["ph_timing"][ok], lab["ph_word"][ok]
        order = np.argsort(self.ph_idx, kind="stable")
        self.ph_idx, self.ph_cls = self.ph_idx[order], self.ph_cls[order]
        self.ph_timing, self.ph_word = self.ph_timing[order], self.ph_word[order]


def train_sessions(hours) -> list[Session]:
    from run_bakeoff_real import subset_hours
    from run_pretrain_scale import cache_dir, deep_runs, heldout_keys

    from neurocast.train.pretrain import open_cached

    runs = {"_".join(r.key): r for r in deep_runs()}
    keys = [k for k in runs if k not in EXCLUDE]
    recs = open_cached(cache_dir(), keys, heldout=heldout_keys(list(runs.values())))
    recs = [r for r in (subset_hours(recs, hours) if hours is not None else recs) if r.train_stop > 0]
    out = []
    for r in recs:
        n = r.train_stop // PATCH
        lab = labels_for(runs[r.key].events, r.n_samples // PATCH, r.story[0] / 250.0)
        out.append(Session(lambda s, r=r: r.window(PATCH * s, N_PATCH * PATCH), n, lab, r.key))
    return out


def chapter(ses: str) -> Session:
    """Sherlock1 chapter 11 or 12 of the deep listener, in RAM."""
    from run_bakeoff_real import read_session

    x, t0, events = read_session(ses)
    n = x.shape[0] // PATCH
    return Session(lambda s: np.asarray(x[PATCH * s:PATCH * (s + N_PATCH)]).T, n, labels_for(events, n, t0),
                   f"Sherlock1_{ses}")


def batch(sessions: list[Session], seed: int, step: int):
    """Windows, speech labels and phoneme targets of batch ``step``: a function of (seed, step)."""
    rng = np.random.default_rng([seed, step, 13])
    w = np.array([s.n - N_PATCH + 1 for s in sessions], float)
    xs, ys, ph = [], [], []
    for b in range(BATCH):
        s = sessions[int(rng.choice(len(sessions), p=w / w.sum()))]
        st = int(rng.integers(0, s.n - N_PATCH + 1))
        xs.append(s.window(st))
        ys.append(s.speech[st + KEEP[0]:st + KEEP[1]])
        lo, hi = np.searchsorted(s.ph_idx, [st + KEEP[0], st + KEEP[1]])
        ph += [(b, int(p - st), int(c)) for p, c in zip(s.ph_idx[lo:hi], s.ph_cls[lo:hi])]
    p = np.array(ph, dtype=np.int64).reshape(-1, 3)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), p


# -- model ---------------------------------------------------------------------------------
def lag_features(g: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """For each (window b, patch p): the features of patches p+1..p+4, concatenated."""
    return torch.cat([g[b, p + lag] for lag in LAGS], dim=-1)


class Net(nn.Module):
    def __init__(self, trunk) -> None:
        from run_finetune import Head

        super().__init__()
        d = trunk.d_model
        self.trunk = trunk
        self.speech = Head(d)
        self.phoneme = nn.Sequential(nn.LayerNorm(len(LAGS) * d), nn.Linear(len(LAGS) * d, N_PH))

    def forward(self, x, ph):
        g = self.trunk.groups(x).float().mean(2)                  # (B, 64, d)
        sp = self.speech(g)
        pl = self.phoneme(lag_features(g, ph[:, 0], ph[:, 1])) if len(ph) else g.new_zeros((0, N_PH))
        return sp, pl


def build(start: str, seed: int, montage, device) -> Net:
    from run_bakeoff_real import load_model, specs

    from neurocast.models.neurocast import NeuroCast

    torch.manual_seed(seed)
    if start.startswith("scratch"):
        trunk = NeuroCast(montage, rung="6m", window=256, causal=start == "scratch_causal",
                          device=device).to(device)
    else:
        trunk, _ = load_model(specs()[f"{start}_s{seed}"], montage, device)
    torch.manual_seed(seed + 1000)
    return Net(trunk).to(device)


@torch.no_grad()
def predict(net, s: Session, device, amp) -> dict:
    """Speech logits for every middle patch and phoneme logits for every onset, stride 32."""
    net.eval()
    sp, sp_y, sp_i, pl, pl_k = [], [], [], [], []
    starts = list(range(0, s.n - N_PATCH + 1, KEEP[1] - KEEP[0]))
    for i in range(0, len(starts), 16):
        st = starts[i:i + 16]
        ph = []
        for b, w0 in enumerate(st):
            lo, hi = np.searchsorted(s.ph_idx, [w0 + KEEP[0], w0 + KEEP[1]])
            ph += [(b, int(s.ph_idx[k] - w0), k) for k in range(lo, hi)]
        p = np.array(ph, dtype=np.int64).reshape(-1, 3)
        xb = torch.as_tensor(np.stack([s.window(w0) for w0 in st]).astype(np.float32), device=device)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            a, c = net(xb, torch.as_tensor(p[:, :2], device=device))
        sp.append(a.float().cpu().numpy().ravel())
        sp_y.append(np.concatenate([s.speech[w0 + KEEP[0]:w0 + KEEP[1]] for w0 in st]))
        sp_i.append(np.concatenate([np.arange(w0 + KEEP[0], w0 + KEEP[1]) for w0 in st]))
        pl.append(c.float().cpu().numpy())
        pl_k.append(p[:, 2])
    net.train()
    k = np.concatenate(pl_k)
    return {"speech": np.concatenate(sp), "speech_y": np.concatenate(sp_y).astype(int),
            "speech_idx": np.concatenate(sp_i), "phoneme": np.concatenate(pl), "ph_k": k,
            "ph_y": s.ph_cls[k], "ph_timing": s.ph_timing[k], "ph_word": s.ph_word[k]}


def voiced_logit(ph_logits: np.ndarray) -> np.ndarray:
    from neurocast.eval.yardstick import PHONEMES, VOICED

    v = np.array([q in VOICED for q in PHONEMES])
    a = ph_logits - ph_logits.max(1, keepdims=True)
    e = np.exp(a)
    pv = e[:, v].sum(1) / e.sum(1)
    pv = np.clip(pv, 1e-6, 1 - 1e-6)
    return np.log(pv / (1 - pv))


def val_scores(pr: dict) -> tuple[float, float]:
    from neurocast.eval.yardstick import macro_f1

    return (macro_f1((pr["speech"] > 0).astype(int), pr["speech_y"], 2),
            macro_f1(pr["phoneme"].argmax(1), pr["ph_y"], N_PH))


# -- training ------------------------------------------------------------------------------
def train_run(start, hours, seed, montage, device, amp, val: Session, test: Session) -> dict:
    name = run_name(start, hours, seed)
    sessions = train_sessions(hours)
    label_h = sum(s.n for s in sessions) * PATCH / 250 / 3600
    net = build(start, seed, montage, device)
    sp_all = np.concatenate([s.speech for s in sessions])
    pos = float(sp_all.mean())
    counts = np.bincount(np.concatenate([s.ph_cls for s in sessions]), minlength=N_PH).astype(float)
    cw = np.where(counts > 0, counts.sum() / (np.count_nonzero(counts) * np.maximum(counts, 1)), 0.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=device))
    ce = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / WARMUP if s < WARMUP else
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / (STEPS - WARMUP))))
    state = {"step": 0, "curve": [], "best": {"speech": None, "phoneme": None}}
    ck = ckpt_dir() / f"{name}.pt"
    if ck.exists():
        s = torch.load(ck, map_location=device, weights_only=False)
        net.load_state_dict(s["net"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        state = s["state"]
        print(f"  {name}: resumed at step {state['step']}", flush=True)
    print(f"  {name}: {label_h:.2f} h of labels in {len(sessions)} sessions; {STEPS} steps", flush=True)
    pool = ThreadPoolExecutor(4)
    ahead: dict = {}
    t0, first = time.time(), state["step"]
    net.train()
    try:
        while state["step"] < STEPS:
            step = state["step"] + 1
            for k in range(step, min(step + 6, STEPS + 1)):
                if k not in ahead:
                    ahead[k] = pool.submit(batch, sessions, seed, k)
            xb, yb, ph = ahead.pop(step).result()
            xb, yb = torch.as_tensor(xb, device=device), torch.as_tensor(yb, device=device)
            pt = torch.as_tensor(ph, device=device)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                sp, pl = net(xb, pt[:, :2])
            loss = bce(sp.float(), yb)
            if len(ph):
                loss = loss + ce(pl.float(), pt[:, 2])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            state["step"] = step
            if step % EVAL_EVERY == 0 or step == STEPS:
                f_sp, f_ph = val_scores(predict(net, val, device, amp))
                state["curve"].append((step, f_sp, f_ph, float(loss.detach())))
                for task, f in (("speech", f_sp), ("phoneme", f_ph)):
                    b = state["best"][task]
                    if b is None or f > b[1]:
                        state["best"][task] = (step, f, {k: t.detach().cpu().clone()
                                                         for k, t in net.state_dict().items()})
                print(f"    {name} step {step:5d}  loss {float(loss.detach()):.3f}  val speech F1 {f_sp:.3f}  "
                      f"phoneme F1 {f_ph:.4f}  ({(time.time() - t0) / max(step - first, 1):.2f} s/step)",
                      flush=True)
            if step % CKPT_EVERY == 0 and step < STEPS:
                ck.parent.mkdir(parents=True, exist_ok=True)
                tmp = ck.with_suffix(".tmp")
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "state": state}, tmp)
                replace_file(tmp, ck)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    res = {"start": start, "hours": hours, "seed": seed, "label_hours": label_h,
           "curve": state["curve"]}
    for task in ("speech", "phoneme"):
        step_, f, sd = state["best"][task]
        net.load_state_dict(sd)
        res[f"{task}_best_step"], res[f"{task}_val_f1"] = step_, f
        res[f"_{task}_pred"] = predict(net, test, device, amp)
    res.update(score(res, sessions, val))
    for k in [k for k in res if k.startswith("_")]:
        res.pop(k)
    del net
    torch.cuda.empty_cache()
    return res


def score(res: dict, sessions: list[Session], val: Session) -> dict:
    """Test scores, and each as nats beyond its no-brain model."""
    from neurocast.audit.timing_matched import UNSCORED, info_beyond_model
    from neurocast.eval.yardstick import (PHONEMES, VOICED, auc, balanced_accuracy, fit_select,
                                          macro_f1)

    out = {}
    sp = res["_speech_pred"]
    pred = (sp["speech"] > 0).astype(int)
    out["speech"] = {"f1": macro_f1(pred, sp["speech_y"], 2), "bacc": balanced_accuracy(pred, sp["speech_y"], 2),
                     "auc": auc(sp["speech"], sp["speech_y"]), "n": int(len(pred))}
    p = float(np.concatenate([s.speech for s in sessions]).mean())
    base = np.tile(np.log([1 - p, p]), (len(pred), 1))
    brain = np.stack([np.zeros_like(sp["speech"]), sp["speech"]], 1)
    blocks = sp["speech_idx"] // 156
    r = info_beyond_model(base, brain, sp["speech_y"], blocks)
    out["speech_beyond_prior"] = {"nats": r.nats, "lo": r.lo, "hi": r.hi}

    ph = res["_phoneme_pred"]
    y, logits = ph["ph_y"], ph["phoneme"].astype(np.float64)
    out["phoneme"] = {"f1": macro_f1(logits.argmax(1), y, N_PH),
                      "bacc": balanced_accuracy(logits.argmax(1), y, N_PH),
                      "acc": float((logits.argmax(1) == y).mean()), "n": int(len(y))}
    voiced = np.array([q in VOICED for q in PHONEMES])
    vy = voiced[y].astype(int)
    vl = voiced_logit(logits)
    out["voicing"] = {"bacc": balanced_accuracy((vl > 0).astype(int), vy, 2),
                      "f1": macro_f1((vl > 0).astype(int), vy, 2), "auc": auc(vl, vy)}
    # No-brain models: a probe on phoneme timing, fitted on the same labelled sessions.
    tf = np.concatenate([s.ph_timing for s in sessions])
    yf = np.concatenate([s.ph_cls for s in sessions])
    present = np.bincount(yf, minlength=N_PH) > 0
    for name, ytr, yva, yte, k, brain_s in (
            ("phoneme", yf, val.ph_cls, y, N_PH, np.where(present[None], logits, UNSCORED)),
            ("voicing", voiced[yf].astype(int), voiced[val.ph_cls].astype(int), vy, 2,
             np.stack([np.zeros_like(vl), vl], 1))):
        probe, _ = fit_select(tf, ytr, val.ph_timing, yva, k)
        b = probe.scores(ph["ph_timing"]).astype(np.float64)
        b = b - b.max(1, keepdims=True)
        b = b - np.log(np.exp(b).sum(1, keepdims=True))
        r = info_beyond_model(b, brain_s, yte, ph["ph_word"])
        out[f"{name}_beyond_timing"] = {"nats": r.nats, "lo": r.lo, "hi": r.hi}
        out[f"{name}_timing_only"] = {"f1": macro_f1(probe.predict(ph["ph_timing"]), yte, k),
                                      "bacc": balanced_accuracy(probe.predict(ph["ph_timing"]), yte, k)}
    return out


def stage_train(args) -> int:
    from run_pretrain_pilot import montage_for
    from run_pretrain_scale import deep_runs

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
    device = torch.device(args.device)
    amp = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) \
        if device.type == "cuda" else None
    OUT.mkdir(parents=True, exist_ok=True)
    montage = montage_for(deep_runs()[0])
    val, test = chapter("11"), chapter("12")
    if args.smoke:                                     # a minute of each chapter
        for c in (val, test):
            c.n = min(c.n, 1000)
            keep = c.ph_idx < c.n
            c.ph_idx, c.ph_cls, c.ph_timing, c.ph_word = (a[keep] for a in (c.ph_idx, c.ph_cls,
                                                                              c.ph_timing, c.ph_word))
    for start, hours, seed in PLANS[args.plan]:
        if args.starts and start not in args.starts:
            continue
        hours = args.hours if args.smoke else hours
        f = OUT / f"{run_name(start, hours, seed)}.json"
        if f.exists():
            continue
        res = train_run(start, hours, seed, montage, device, amp, val, test)
        f.write_text(json.dumps(res, indent=2, default=float))
        print(f"  {f.stem}: speech F1 {res['speech']['f1']:.3f} (beyond prior "
              f"{res['speech_beyond_prior']['nats']:+.3f}); phonemes F1 {res['phoneme']['f1']:.4f}, beyond "
              f"timing {res['phoneme_beyond_timing']['nats']:+.4f} [{res['phoneme_beyond_timing']['lo']:+.4f}, "
              f"{res['phoneme_beyond_timing']['hi']:+.4f}]; voicing BAcc {res['voicing']['bacc']:.3f}", flush=True)
        if args.smoke:
            break
    return 0


def stage_selftest(args) -> int:
    """The phoneme head reads patches p+1..p+4 of the right window."""
    g = torch.arange(2 * 64, dtype=torch.float32).reshape(2, 64, 1).expand(2, 64, 3)
    b, p = torch.tensor([0, 1, 1]), torch.tensor([16, 20, 47])
    f = lag_features(g, b, p)[:, ::3]
    want = torch.tensor([[17, 18, 19, 20], [64 + 21, 64 + 22, 64 + 23, 64 + 24], [64 + 48, 64 + 49, 64 + 50, 64 + 51]],
                        dtype=torch.float32)
    ok = torch.equal(f, want)
    print(f"[{'PASS' if ok else 'FAIL'}] the phoneme head reads the 4 patches after each onset, in its own window")
    s = Session(lambda st: np.zeros((306, 1024)), 200,
                {"speech": np.arange(200) % 2, "ph_idx": np.array([30, 10, 100]), "ph_cls": np.array([1, 2, 3]),
                 "ph_timing": np.zeros((3, 9)), "ph_word": np.array([0, 0, 1])}, "t")
    xb, yb, ph = batch([s], 0, 1)
    rng = np.random.default_rng([0, 1, 13])
    rng.choice(1, p=[1.0])
    st = int(rng.integers(0, 200 - 64 + 1))
    exp = [(0, int(q - st), int(c)) for q, c in zip(s.ph_idx, s.ph_cls) if st + 16 <= q < st + 48]
    first = sorted(tuple(t) for t in ph.tolist() if t[0] == 0)    # the batch's first window
    ok2 = first == sorted(exp) and np.array_equal(yb[0], s.speech[st + 16:st + 48])
    print(f"[{'PASS' if ok2 else 'FAIL'}] a batch carries the onsets in its window's middle patches, sorted, "
          f"and their speech labels")
    return 0 if ok and ok2 else 1


def stage_report(args) -> int:
    rows = {f.stem: json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))}
    print("| Start | Labels | Seed | Speech F1 / AUC | Speech beyond prior | Phonemes F1 | Phonemes beyond timing | "
          "Voicing BAcc | Voicing beyond timing |")
    print("|---|---|---|---|---|---|---|---|---|")

    def ci(d):
        return f"{d['nats']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"

    for k, r in rows.items():
        print(f"| {r['start']} | {r['label_hours']:.1f} h | {r['seed']} | {r['speech']['f1']:.3f} / "
              f"{r['speech']['auc']:.3f} | {ci(r['speech_beyond_prior'])} | {r['phoneme']['f1']:.4f} | "
              f"{ci(r['phoneme_beyond_timing'])} | {r['voicing']['bacc']:.3f} | {ci(r['voicing_beyond_timing'])} |")
    if rows:
        r0 = next(iter(rows.values()))
        print(f"\ntiming-only rows (from one run's labels): phonemes F1 {r0['phoneme_timing_only']['f1']:.4f}, "
              f"voicing BAcc {r0['voicing_timing_only']['bacc']:.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "train", "report"))
    ap.add_argument("--plan", choices=tuple(PLANS), default="main")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--starts", nargs="*", choices=STARTS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--smoke", action="store_true", help="tiny run on CPU")
    ap.add_argument("--hours", type=float, default=0.05, help="with --smoke")
    ap.add_argument("--steps", type=int, help="with --smoke")
    args = ap.parse_args()
    global OUT, STEPS, EVAL_EVERY, CKPT_EVERY, WARMUP
    if args.out:
        OUT = args.out.resolve()
    if args.smoke:
        STEPS, EVAL_EVERY, CKPT_EVERY, WARMUP = args.steps or 6, 3, 4, 2
    return {"selftest": stage_selftest, "train": stage_train, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
