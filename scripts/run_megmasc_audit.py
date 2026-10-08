"""The LibriBrain audit, replicated on MEG-MASC: timing vs brain, matched and unmatched.

``run_megmasc_timing.py`` showed that speech timing alone predicts the heard word
on MEG-MASC too. This is the brain half: the same 50-word task, decoded from
MEG, through the same two tests -- how much of a decoder's gain survives
matching trials on timing, length and loudness, and how many nats per word it
adds beyond them.

* Data: MEG-MASC session 0 of every listener with data (27 at most), 208 KIT
  axial gradiometers, notch 60 Hz, band-pass 0.1-100 Hz, 250 Hz
  (:func:`neurocast.data.megmasc.preprocess`), session-scalar normalisation;
  0.512 s windows from each word's onset, 64 ms patch means.
* Task: the 50 most frequent words of the four stories; trials are listener x
  word occurrence; every listener heard the same audio.
* Stimulus-disjoint folds: leave one story out (4 folds), pooled. The linear
  decoder (the audit's ``LinearPooled``, ridge CV grouped by occurrence) and the
  timing classifier are fitted on the other three stories in each fold.
* Loudness: each word's RMS level in its own stimulus file.

    .venv/Scripts/python.exe scripts/run_megmasc_audit.py --stage prepare   # filter + cache (CPU)
    .venv/Scripts/python.exe scripts/run_megmasc_audit.py --stage brain
    .venv/Scripts/python.exe scripts/run_megmasc_audit.py --stage matched
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocast.data import megmasc  # noqa: E402
from neurocast.data.libribrain import onset_windows  # noqa: E402
from neurocast.data.paths import dataset_dir  # noqa: E402
from neurocast.decode.metrics import balanced_accuracy_at_k  # noqa: E402
from neurocast.decode.words import label_trials, vocabulary  # noqa: E402

TASKS = ("0", "1", "2", "3")
SUBJECTS = tuple(f"{i:02d}" for i in range(1, 28))
OUT = ROOT / "runs" / "megmasc"
COVARIATES = ("next1", "next2", "next3", "next4", "gap_prev", "pos_in_sentence",
              "duration", "loudness_db")


def cache_dir() -> Path:
    return dataset_dir("derived") / "megmasc"


def stage_prepare(args) -> dict:
    t0, done, missing = time.time(), 0, []
    for sub in SUBJECTS:
        for task in TASKS:
            if not megmasc.raw_path(sub, args.session, task).exists():
                missing.append(f"{sub}/{task}")
                continue
            megmasc.preprocess(sub, args.session, task, cache_dir())
            done += 1
            print(f"  sub-{sub} task-{task} ({time.time() - t0:.0f}s)", flush=True)
    print(f"{done} recordings cached; {len(missing)} not downloaded")
    return {"cached": done, "missing": missing}


def complete_listeners(session: str) -> list[str]:
    """Listeners with all four stories preprocessed -- partial ones would unbalance the folds."""
    return [s for s in SUBJECTS
            if all((cache_dir() / f"sub-{s}_ses-{session}_task-{t}.npy").exists() for t in TASKS)]


def trials_and_windows(session: str, vocab, window_s: float = 0.512):
    """Complete listeners' vocabulary words of session ``session``, and their windows."""
    import pandas as pd

    tabs, wins = [], []
    for sub in complete_listeners(session):
        w = megmasc.word_table(sub, session, TASKS, window_s=window_s)
        if w is None:
            continue
        for task in TASKS:
            f = cache_dir() / f"sub-{sub}_ses-{session}_task-{task}.npy"
            t = label_trials(w[w["task"] == task], vocab)
            if not f.exists() or t.empty:
                continue
            x, sfreq = megmasc.load_preprocessed(f)
            win, kept = onset_windows(x, sfreq, t["onset"].to_numpy(),
                                      n_samples=int(round(window_s * sfreq)), decim=2)
            tabs.append(t[kept])
            wins.append(win.astype(np.float16))
        print(f"  sub-{sub}: {sum(len(t) for t in tabs):,} trials so far", flush=True)
    return pd.concat(tabs, ignore_index=True), np.concatenate(wins)


def stage_brain(args) -> dict:
    from neurocast.decode.baselines import TimingClassifier
    from neurocast.decode.decoders import LinearPooled, patch_features

    t0 = time.time()
    ref = megmasc.word_table("01", args.session, TASKS, window_s=0.512)
    vocab = vocabulary(ref, 50)
    te, w = trials_and_windows(args.session, vocab)
    y, K = te["label"].to_numpy(), len(vocab)
    print(f"{len(te):,} trials, {te['subject'].nunique()} listeners, {w.shape[1]} sensors")
    x = patch_features(w)
    del w
    scores = {"timing": np.full((len(te), K), -np.inf), "linear": np.full((len(te), K), -1e6)}
    for task in TASKS:
        test = (te["task"] == task).to_numpy()
        fit = ~test
        once = te[fit].drop_duplicates("occ")            # timing is the same for every listener
        tc = TimingClassifier().fit(once, once["label"].to_numpy(), n_classes=K)
        scores["timing"][test] = tc.log_proba(te[test])
        lin = LinearPooled(seed=0).fit(x[fit], y[fit], te["occ"].to_numpy()[fit])
        scores["linear"][test] = lin.scores(x[test], K)
        print(f"  story {task} held out: {test.sum():,} trials ({time.time() - t0:.0f}s)", flush=True)
    res = {"n_trials": int(len(te)), "listeners": int(te["subject"].nunique()), "vocab": vocab}
    for k, s in scores.items():
        res[k] = (balanced_accuracy_at_k(s, y, 1), balanced_accuracy_at_k(s, y, 10))
        print(f"  {k:<8} BAcc@1 {res[k][0]:.3f}  BAcc@10 {res[k][1]:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    te.to_csv(OUT / "test_trials.csv", index=False)
    np.savez_compressed(OUT / "test_scores.npz", y=y, **scores)
    return res


def loudness(te) -> np.ndarray:
    from neurocast.decode.covariates import read_wav, word_loudness_db

    root = dataset_dir("MEG-MASC")
    out = np.full(len(te), np.nan)
    for sound, idx in te.groupby("sound").indices.items():
        audio, rate = read_wav(root / sound)
        out[idx] = word_loudness_db(audio, rate, te["t_sound"].to_numpy()[idx],
                                    te["duration"].to_numpy()[idx])
    return out


def stage_matched(args) -> dict:
    import pandas as pd

    from neurocast.audit.timing_matched import info_beyond_covariates, matched_gain, strata

    te = pd.read_csv(OUT / "test_trials.csv")
    z = np.load(OUT / "test_scores.npz")
    y, occ = z["y"], te["occ"].to_numpy()
    te["loudness_db"] = loudness(te)
    groups = strata(te)
    cov = np.column_stack([np.nan_to_num(te[c].to_numpy(float), nan=-1.0) for c in COVARIATES]
                          + [np.isnan(te[c].to_numpy(float)) for c in ("next1", "next2", "next3")])
    res = {"n_strata": int(pd.Series(groups).nunique())}
    print(f"{len(te):,} trials in {res['n_strata']} strata")
    for k in ("timing", "linear"):
        r = matched_gain(z[k], y, occ, groups, metric="rank", n_perm=args.n_perm, seed=0)
        res[f"matched_{k}"] = {"gain": r.gain, "unmatched_gain": r.unmatched_gain,
                               "surviving": r.surviving, "p95": r.null_p95, "verdict": r.verdict}
        print(f"  {k:<8} unmatched {r.unmatched_gain:+.4f}  matched {r.gain:+.4f}  "
              f"surviving {100 * r.surviving:.0f}%  {r.verdict}", flush=True)
    b = info_beyond_covariates(cov, z["linear"], y, occ, n_boot=args.n_boot, seed=0)
    res["beyond_linear"] = {"nats": b.nats, "lo": b.lo, "hi": b.hi}
    print(f"  linear beyond timing, length, loudness: {b}")
    freq = np.log(np.bincount(y, minlength=z["linear"].shape[1]) + 1.0)
    b = info_beyond_covariates(cov, np.tile(freq, (len(y), 1)), y, occ, n_boot=200)
    res["beyond_frequency_only"] = {"nats": b.nats, "lo": b.lo, "hi": b.hi}
    print(f"  frequency only (must be 0): {b}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("prepare", "brain", "matched"), required=True)
    ap.add_argument("--session", default="0")
    ap.add_argument("--subjects", default="", help="comma-separated subset, e.g. 01,02")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    global SUBJECTS
    if args.subjects:
        SUBJECTS = tuple(args.subjects.split(","))
    res = {"prepare": stage_prepare, "brain": stage_brain, "matched": stage_matched}[args.stage](args)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"audit_{args.stage}.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"  saved {(OUT / f'audit_{args.stage}.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
