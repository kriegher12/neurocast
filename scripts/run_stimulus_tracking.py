"""What MEG adds beyond the stimulus: nats beyond no-brain models, envelope tracking, retrieval.

README section 2.9. Roadmap steps 1 and 4 left three measurements open, and the
pivot (section 18) asks that every decoding score be reported as information
beyond a model that needs no brain. All three run on the per-patch features
section 2.6 saved for every model and the raw signal (the deep listener's
Sherlock1 sessions 1-12; fit on 1-10, choose on 11, score on 12):

* ``beyond``: decoding as information beyond no-brain models, in nats, with
  ``audit/timing_matched.info_beyond_model`` -- speech vs silence per patch
  beyond the class prior; the 39 phonemes and voicing beyond a probe on phoneme
  timing alone (section 2.6's no-brain row). Probes are fitted on all 3.2 h of
  labels; folds and bootstrap are cut by word (phonemes) or by 10-s block
  (speech), so neighbouring trials never sit on both sides;
* ``envelope``: envelope tracking -- ridge regression from each patch's lagged
  features (64-320 ms later, as the probes read them) to the audio's level in
  that patch, broadband and in 8 bands (``decode/covariates.py``). Pearson r on
  chapter 12; beside the raw signal, shuffled MEG, and a no-brain row that knows
  only whether each patch and the next three are speech. The brain's share is the
  gain of a model given both the speech pattern and the MEG over the pattern
  alone, with a bootstrap over 10-s blocks: a smoke run found the pattern alone
  tracks the envelope at r 0.82, above raw MEG's 0.52;
* ``retrieval``: which 3-s stretch of chapter 12 was heard -- the decoded
  9-band envelope of each stretch against the actual one of every stretch, top-1
  and top-10 accuracy. Reported as retrieval, never as detection, beside the
  same retrieval from the speech/silence pattern alone, and from both.

    .venv/Scripts/python.exe scripts/run_stimulus_tracking.py --stage all
    .venv/Scripts/python.exe scripts/run_stimulus_tracking.py --stage report

One JSON per model in ``runs/stimulus_tracking/``; finished models are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "runs" / "stimulus_tracking"
SEG = 47                     # retrieval stretches: 47 patches = 3.0 s
BLOCK = 156                  # speech folds: 10-s blocks of patches
ALPHAS = tuple(np.logspace(0, 6, 7))


def ydir() -> Path:
    from run_bakeoff_real import feat_dir

    return feat_dir() / "yard"


def sessions():
    from run_bakeoff_real import FIT_SESSIONS, TEST, VAL

    return FIT_SESSIONS, VAL, TEST


def meta(ses: str) -> dict:
    """Story start, events and the valid-patch mask of a session, without reading its MEG."""
    from run_bakeoff_real import FIT_SESSIONS
    from run_pretrain_scale import cache_dir, deep_runs

    from neurocast.data.libribrain import find_runs, read_words

    p = len(np.load(ydir() / "_raw" / f"{ses}.npy", mmap_mode="r"))
    valid = np.zeros(p, bool)
    for s in range(0, p - 64 + 1, 32):                         # run_bakeoff_real.patch_features
        valid[s + 16:s + 48] = True
    if ses in FIT_SESSIONS:
        r = next(r for r in deep_runs() if r.task == "Sherlock1" and r.session == ses)
        t0 = json.loads((cache_dir() / f"{'_'.join(r.key)}.json").read_text())["story"][0] / 250.0
    else:
        r = find_runs(task="Sherlock1", sessions={ses}, subjects={"0"}, require_meg=True)[0]
        t0 = int(read_words(r.events)["onset"].iloc[0] * 250.0) / 250.0
    return {"n": p, "valid": valid, "t0": t0, "events": r.events}


def labels(m: dict) -> dict:
    """Speech per patch; phonemes with class, timing and word key (section 2.6's rows)."""
    import pandas as pd
    from run_bakeoff_real import session_labels

    lab = session_labels(m["events"], m["n"], m["t0"])
    ev = pd.read_csv(m["events"], sep="\t")
    ph = ev[ev["kind"] == "phoneme"]
    from neurocast.eval.yardstick import phoneme_class

    cls = np.array([phoneme_class(s) for s in ph["segment"]]).reshape(-1, 2)
    keep = (cls[:, 0] >= 0) & (cls[:, 1] >= 0)
    word = (pd.to_numeric(ph["sentenceidx"], errors="coerce").fillna(-1).to_numpy() * 10_000
            + pd.to_numeric(ph["wordidx"], errors="coerce").fillna(-1).to_numpy())[keep]
    assert len(word) == len(lab["ph_idx"])
    lab["ph_word"] = word
    return lab


def envelope(m: dict) -> np.ndarray:
    """Audio level per patch, dB, broadband and 8 bands ``(n, 9)``; inserted silence floored."""
    from neurocast.data.paths import dataset_dir
    from neurocast.decode.covariates import (band_power, chapter_audio_path, event_times, read_wav,
                                             window_levels)

    ses = Path(m["events"]).name.split("_ses-")[1].split("_")[0]
    audio, rate = read_wav(chapter_audio_path(dataset_dir("LibriBrain100"), "Sherlock1", ses))
    power, centres = band_power(audio, rate)
    ev = event_times(m["events"])
    on = m["t0"] + np.arange(m["n"]) * 0.064
    lv = window_levels(power, centres, on, 0.064, ev["meg"].to_numpy(), ev["chapter"].to_numpy(),
                       bin_s=0.064)[:, 0, :].astype(np.float64)
    lv = lv[:, [8, *range(8)]]                                 # broadband first, then the bands
    for j in range(lv.shape[1]):                               # floor at the 1st percentile of sound
        real = lv[:, j] > -100
        lo = np.percentile(lv[real, j], 1) if real.any() else -100
        lv[:, j] = np.maximum(lv[:, j], lo)
    return lv


# -- ridge ---------------------------------------------------------------------------------
class Ridge:
    """Ridge regression on standardised features, penalty chosen on a validation set."""

    def fit(self, x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray) -> "Ridge":
        n, d = x.shape
        s1, s2 = np.zeros(d), np.zeros(d)
        for a in range(0, n, 8192):
            c = np.asarray(x[a:a + 8192], np.float64)
            s1 += c.sum(0)
            s2 += (c ** 2).sum(0)
        self.mu = s1 / n
        self.sd = np.sqrt(np.maximum(s2 / n - self.mu ** 2, 0)) + 1e-6
        self.ym = y.mean(0)
        xtx, xty = np.zeros((d, d)), np.zeros((d, y.shape[1]))
        for a in range(0, n, 8192):
            z = (np.asarray(x[a:a + 8192], np.float64) - self.mu) / self.sd
            xtx += z.T @ z
            xty += z.T @ (y[a:a + 8192] - self.ym)
        lam, v = np.linalg.eigh(xtx)
        proj = v.T @ xty
        best = None
        for al in ALPHAS:
            w = v @ (proj / (lam + al)[:, None])
            r = np.mean(pearson(self._predict(xv, w), yv))
            if best is None or r > best[0]:
                best = (r, al, w)
        self.alpha, self.w = best[1], best[2]
        return self

    def _predict(self, x, w):
        out = [((np.asarray(x[a:a + 8192], np.float64) - self.mu) / self.sd) @ w
               for a in range(0, len(x), 8192)]
        return np.concatenate(out) + self.ym

    def predict(self, x):
        return self._predict(x, self.w)


def pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0) + 1e-12)


# -- the three measurements -------------------------------------------------------------------
def speech_rows(feats, lab, valid, ses_list, stride=1):
    from neurocast.eval.yardstick import lagged, usable

    xs, ys, gs = [], [], []
    for s in ses_list:
        idx = np.arange(0, len(valid[s]), stride)
        idx = idx[usable(idx, valid[s])]
        xs.append(lagged(feats(s), idx))
        ys.append(lab[s]["speech"][idx])
        gs.append(np.array([f"{s}:{i // BLOCK}" for i in idx]))
    return np.concatenate(xs), np.concatenate(ys).astype(int), np.concatenate(gs), idx


def phoneme_rows(feats, lab, valid, ses_list):
    from neurocast.eval.yardstick import lagged, usable

    xs, ys, ts, ws = [], [], [], []
    for s in ses_list:
        ok = usable(lab[s]["ph_idx"], valid[s])
        xs.append(lagged(feats(s), lab[s]["ph_idx"][ok]))
        ys.append(lab[s]["ph_cls"][ok])
        ts.append(lab[s]["ph_timing"][ok])
        ws.append(np.array([f"{s}:{w}" for w in lab[s]["ph_word"][ok]]))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ts), np.concatenate(ws)


def logits(probe, x) -> np.ndarray:
    from neurocast.audit.timing_matched import UNSCORED

    s = probe.scores(x).astype(np.float64)
    s[:, ~probe.present] = UNSCORED
    return s


def log_softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max(1, keepdims=True)
    return a - np.log(np.exp(a).sum(1, keepdims=True))


def beyond(feats, lab, valid, nobrain: dict, n_boot: int) -> dict:
    """Nats per trial beyond the no-brain models, held out, with a bootstrap over words/blocks."""
    from neurocast.audit.timing_matched import info_beyond_model
    from neurocast.eval.yardstick import PHONEMES, VOICED, fit_select

    fit, val, test = sessions()
    out = {}
    xf, yf, _, _ = speech_rows(feats, lab, valid, fit, stride=2)
    xv, yv, _, _ = speech_rows(feats, lab, valid, [val], stride=2)
    xt, yt, gt, _ = speech_rows(feats, lab, valid, [test])
    probe, _ = fit_select(xf, yf, xv, yv, 2)
    p = float(yf.mean())
    base = np.tile(np.log([1 - p, p]), (len(yt), 1))
    r = info_beyond_model(base, logits(probe, xt), yt, gt, n_boot=n_boot)
    out["speech_beyond_prior"] = {"nats": r.nats, "lo": r.lo, "hi": r.hi, "n": int(len(yt))}
    del xf, xv, xt

    voiced = np.array([q in VOICED for q in PHONEMES])
    xf, yf, tf, _ = phoneme_rows(feats, lab, valid, fit)
    xv, yv, tv, _ = phoneme_rows(feats, lab, valid, [val])
    xt, yt, tt, wt = phoneme_rows(feats, lab, valid, [test])
    for name, f, k in (("phonemes", lambda y: y, len(PHONEMES)),
                       ("voicing", lambda y: voiced[y].astype(int), 2)):
        probe, _ = fit_select(xf, f(yf), xv, f(yv), k)
        base = log_softmax(logits(nobrain[name], tt))
        r = info_beyond_model(base, logits(probe, xt), f(yt), wt, n_boot=n_boot)
        out[f"{name}_beyond_timing"] = {"nats": r.nats, "lo": r.lo, "hi": r.hi, "n": int(len(yt)),
                                        "unscored": list(r.unscored) if hasattr(r, "unscored") else []}
    return out


def tracking(feats, lab, valid, env, *, roll: bool = False, extra=None,
             rolled_rows: bool = False) -> dict:
    """Envelope tracking on chapter 12, and retrieval of 3-s stretches from it.

    ``roll`` pairs each patch with features from half a session away (the
    shuffled-MEG row). ``extra`` adds columns that are never rolled (the
    speech/silence pattern). ``rolled_rows`` keeps only the patches a rolled run
    can use, without rolling anything, so a baseline sees the same rows.
    Predictions ride along under ``_pred`` etc.
    """
    from neurocast.eval.yardstick import lagged, usable

    fit, val, test = sessions()

    def rows(ses_list, stride):
        xs, ys, sp = [], [], []
        for s in ses_list:
            f = feats(s)
            v = valid[s]
            if roll or rolled_rows:
                v = v & np.roll(v, len(v) // 2)
            if roll:
                f = np.roll(f, len(f) // 2, axis=0)
            idx = np.arange(0, len(v), stride)
            idx = idx[usable(idx, v)]
            x = lagged(f, idx)
            if extra is not None:
                x = np.concatenate([x, lagged(extra(s), idx).astype(x.dtype)], axis=1)
            xs.append(x)
            e = env[s]
            ys.append((e[idx] - e.mean(0)) / (e.std(0) + 1e-9))   # per-session z-scored dB
            sp.append(lab[s]["speech"][idx])
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(sp), idx

    xf, yf, _, _ = rows(fit, 2)
    xv, yv, _, _ = rows([val], 2)
    xt, yt, st, it = rows([test], 1)
    model = Ridge().fit(xf, yf, xv, yv)
    pred = model.predict(xt)
    r = pearson(pred, yt)
    rs = pearson(pred[st == 1], yt[st == 1])
    out = {"alpha": float(model.alpha), "r_broadband": float(r[0]), "r_bands": r[1:].tolist(),
           "r_mean": float(r.mean()), "r_speech_only_broadband": float(rs[0]),
           "r_speech_only_mean": float(rs.mean())}
    out["retrieval"] = retrieval(pred, yt, it)
    out.update(_pred=pred, _yt=yt, _it=it)
    return out


def gain(with_meg: dict, without: dict, n_boot: int, seed: int = 0) -> dict:
    """Envelope r and retrieval of (pattern + MEG) minus the pattern alone, on the same test
    patches, with a bootstrap over 10-s blocks for the r difference."""
    assert np.array_equal(with_meg["_it"], without["_it"])
    yt, pa, pb = with_meg["_yt"], with_meg["_pred"], without["_pred"]
    blocks = with_meg["_it"] // BLOCK
    ub = np.unique(blocks)
    members = [np.flatnonzero(blocks == b) for b in ub]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = np.concatenate([members[j] for j in rng.integers(0, len(ub), len(ub))])
        d.append(float(np.mean(pearson(pa[i], yt[i]) - pearson(pb[i], yt[i]))))
    return {"r_mean_gain": float(np.mean(pearson(pa, yt) - pearson(pb, yt))),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "r_broadband_gain": float(pearson(pa, yt)[0] - pearson(pb, yt)[0]),
            "top1_gain": with_meg["retrieval"]["top1"] - without["retrieval"]["top1"],
            "top10_gain": with_meg["retrieval"]["top10"] - without["retrieval"]["top10"]}


def public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def retrieval(pred: np.ndarray, actual: np.ndarray, idx: np.ndarray) -> dict:
    """Name each 3-s stretch by matching its decoded 9-band envelope against every stretch's."""
    starts = [i for i in range(0, len(idx) - SEG + 1, SEG) if idx[i + SEG - 1] - idx[i] == SEG - 1]

    def flat(a):
        z = np.stack([a[i:i + SEG].ravel() for i in starts])
        z = z - z.mean(1, keepdims=True)
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

    sim = flat(pred) @ flat(actual).T
    rank = (sim > np.diag(sim)[:, None]).sum(1)
    n = len(starts)
    return {"n": n, "top1": float((rank == 0).mean()), "top10": float((rank < 10).mean()),
            "median_rank": float(np.median(rank)), "chance_top1": 1 / n, "chance_top10": min(10 / n, 1)}


def stage_all(args) -> int:
    from neurocast.eval.yardstick import PHONEMES, VOICED, fit_select

    OUT.mkdir(parents=True, exist_ok=True)
    fit, val, test = sessions()
    sess = list(fit) + [val, test]
    ms = {s: meta(s) for s in sess}
    lab = {s: labels(m) for s, m in ms.items()}
    valid = {s: m["valid"] for s, m in ms.items()}
    env = {s: envelope(m) for s, m in ms.items()}
    print("labels and envelopes ready", flush=True)

    # No-brain rows: phoneme timing probes (section 2.6's), and the speech/silence pattern.
    ones = lambda s: np.ones((len(valid[s]), 1), np.float16)  # noqa: E731
    _, yf, tf, _ = phoneme_rows(ones, lab, valid, fit)
    _, yv, tv, _ = phoneme_rows(ones, lab, valid, [val])
    voiced = np.array([q in VOICED for q in PHONEMES])
    nobrain = {"phonemes": fit_select(tf, yf, tv, yv, len(PHONEMES))[0],
               "voicing": fit_select(tf, voiced[yf].astype(int), tv, voiced[yv].astype(int), 2)[0]}
    # Read through lags 1-4, this gives each patch its own label and the next three's.
    pattern = lambda s: np.roll(lab[s]["speech"], 1)[:, None].astype(np.float16)  # noqa: E731
    pattern_only = tracking(pattern, lab, valid, env)
    nb = OUT / "_nobrain.json"
    if not nb.exists():
        res = {"envelope_from_speech_pattern": public(pattern_only)}
        nb.write_text(json.dumps(res, indent=2, default=float))
        t = res["envelope_from_speech_pattern"]
        print(f"  no brain (speech/silence only): envelope r {t['r_broadband']:.3f}, within speech "
              f"{t['r_speech_only_broadband']:.3f}; retrieval top-1 {t['retrieval']['top1']:.3f} "
              f"(chance {t['retrieval']['chance_top1']:.3f})", flush=True)

    names = ["_raw"] + sorted(p.name for p in ydir().iterdir() if p.is_dir() and p.name != "_raw")
    if args.only:
        names = [n for n in names if n in args.only]
    for name in names:
        f = OUT / f"{name}.json"
        if f.exists():
            continue
        feats = lambda s, name=name: np.load(ydir() / name / f"{s}.npy")  # noqa: E731
        both = tracking(feats, lab, valid, env, extra=pattern)
        res = {"beyond": beyond(feats, lab, valid, nobrain, args.n_boot),
               "envelope": public(tracking(feats, lab, valid, env)),
               "envelope_with_pattern": public(both),
               "envelope_gain_over_pattern": gain(both, pattern_only, args.n_boot)}
        if name == "_raw" or (name.endswith("_s0") and "_h" not in name):
            res["envelope_shuffled"] = public(tracking(feats, lab, valid, env, roll=True))
            res["envelope_gain_shuffled"] = gain(
                tracking(feats, lab, valid, env, roll=True, extra=pattern),
                tracking(pattern, lab, valid, env, rolled_rows=True), args.n_boot)
        f.write_text(json.dumps(res, indent=2, default=float))
        b, e, g = res["beyond"], res["envelope"], res["envelope_gain_over_pattern"]
        print(f"  {name:<22} beyond: speech {b['speech_beyond_prior']['nats']:+.4f}, phonemes "
              f"{b['phonemes_beyond_timing']['nats']:+.4f} [{b['phonemes_beyond_timing']['lo']:+.4f}, "
              f"{b['phonemes_beyond_timing']['hi']:+.4f}], voicing {b['voicing_beyond_timing']['nats']:+.4f}"
              f"  envelope r {e['r_broadband']:.3f}; MEG over the speech pattern: r "
              f"{g['r_mean_gain']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}], retrieval top-1 "
              f"{g['top1_gain']:+.3f}", flush=True)
    return 0


def stage_report(args) -> int:
    rows = {f.stem: json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))}
    nb = rows.pop("_nobrain", None)
    print("| Features | Speech beyond prior | Phonemes beyond timing | Voicing beyond timing | "
          "Envelope r | Retrieval top-1 / top-10 | MEG over the speech pattern: r, top-1 |")
    print("|---|---|---|---|---|---|---|")

    def ci(d):
        return f"{d['nats']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"

    for k, r in rows.items():
        b, e, g = r["beyond"], r["envelope"], r["envelope_gain_over_pattern"]
        print(f"| {k} | {ci(b['speech_beyond_prior'])} | {ci(b['phonemes_beyond_timing'])} | "
              f"{ci(b['voicing_beyond_timing'])} | {e['r_broadband']:.3f} | "
              f"{e['retrieval']['top1']:.3f} / {e['retrieval']['top10']:.3f} | "
              f"{g['r_mean_gain']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}], {g['top1_gain']:+.3f} |")
        if "envelope_shuffled" in r:
            s, g = r["envelope_shuffled"], r["envelope_gain_shuffled"]
            print(f"| {k}, shuffled | | | | {s['r_broadband']:.3f} | "
                  f"{s['retrieval']['top1']:.3f} / {s['retrieval']['top10']:.3f} | "
                  f"{g['r_mean_gain']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}], {g['top1_gain']:+.3f} |")
    if nb:
        t = nb["envelope_from_speech_pattern"]
        print(f"| no brain: speech/silence pattern | | | | {t['r_broadband']:.3f} | "
              f"{t['retrieval']['top1']:.3f} / {t['retrieval']['top10']:.3f} (chance "
              f"{t['retrieval']['chance_top1']:.3f} / {t['retrieval']['chance_top10']:.3f}) | |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("all", "report"))
    ap.add_argument("--only", nargs="*", help="these feature sets only")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = args.out.resolve()
    return {"all": stage_all, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
