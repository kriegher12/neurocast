"""Does speech timing alone predict the heard word in a second dataset? MEG-MASC.

The pitch planned "MEG-MASC as a second dataset for the timing test". Its
LibriBrain finding: a classifier that never sees brain data, only when the
surrounding words start, puts the heard word in the top ten of 50 about 0.61-0.67
of the time (re-measured: 0.666). This asks the same of MEG-MASC (Gwilliams et
al.): synthesised speech, four stories interleaved with random word lists, the
same 50-most-frequent-words task.

* Timing features are the audit's (:func:`neurocast.decode.words.timing_features`):
  onsets of the next four words inside the window, the gap since the previous
  word, the position in the sequence. Every listener heard the same audio and
  the relative timing agrees across listeners to 1e-13 s, so one listener's
  events (sub-01, session 0) stand for all.
* Stimulus-disjoint: leave one story out -- fit on three stories' words
  (sentences and word lists), score the fourth -- and pool the four folds.
* Reported separately for words in running sentences and words in random word
  lists: in a list there is no syntax, so whatever timing still predicts there is
  the word's own length.
* Null: the same classifier on labels permuted across the fit stories' words.

    .venv/Scripts/python.exe scripts/run_megmasc_timing.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocast.data import megmasc  # noqa: E402
from neurocast.decode.baselines import TimingClassifier  # noqa: E402
from neurocast.decode.metrics import balanced_accuracy_at_k  # noqa: E402
from neurocast.decode.words import label_trials, vocabulary  # noqa: E402

TASKS = ("0", "1", "2", "3")
OUT = ROOT / "runs" / "megmasc"


def leave_one_story_out(t, n_classes: int, labels: np.ndarray | None = None, seed: int = 0):
    """Pooled out-of-story log-probabilities; ``labels`` replaces the fit labels (for nulls)."""
    y = t["label"].to_numpy() if labels is None else labels
    out = np.full((len(t), n_classes), -np.inf)
    for task in TASKS:
        te = (t["task"] == task).to_numpy()
        if not te.any():
            continue
        clf = TimingClassifier().fit(t[~te], y[~te], n_classes=n_classes)
        out[te] = clf.log_proba(t[te])
    return out


def bacc(s, y):
    return balanced_accuracy_at_k(s, y, 1), balanced_accuracy_at_k(s, y, 10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="01")
    ap.add_argument("--session", default="0")
    ap.add_argument("--n-null", type=int, default=20)
    args = ap.parse_args()
    res = {}
    for window in (0.512, 1.0):
        w = megmasc.word_table(args.subject, args.session, TASKS, window_s=window)
        vocab = vocabulary(w, 50)
        t = label_trials(w, vocab)
        y = t["label"].to_numpy()
        s = leave_one_story_out(t, len(vocab))
        r = {"n_words": int(len(w)), "n_trials": int(len(t)), "vocab": vocab, "all": bacc(s, y)}
        for cond in ("sentence", "word_list"):
            m = (t["condition"] == cond).to_numpy()
            r[cond] = {"n": int(m.sum()), "bacc": bacc(s[m], y[m])}
        rng = np.random.default_rng(0)
        null = np.array([bacc(leave_one_story_out(t, len(vocab), labels=rng.permutation(y)), y)
                         for _ in range(args.n_null)])
        r["null_mean"], r["null_p95"] = null.mean(0).tolist(), np.percentile(null, 95, axis=0).tolist()
        res[f"{window:.3f}"] = r
        print(f"\nwindow {window:.3f} s: {len(w):,} words, {len(t):,} in the 50-word vocabulary "
              f"(e.g. {vocab[:10]})")
        print(f"  {'':<28} {'n':>6} {'BAcc@1':>7} {'BAcc@10':>8}")
        print(f"  {'all, leave one story out':<28} {len(t):>6} {r['all'][0]:7.3f} {r['all'][1]:8.3f}")
        for cond in ("sentence", "word_list"):
            print(f"  {'  ' + cond:<28} {r[cond]['n']:>6} {r[cond]['bacc'][0]:7.3f} {r[cond]['bacc'][1]:8.3f}")
        print(f"  {'shuffled labels (mean, p95)':<28} {'':>6} {r['null_mean'][0]:7.3f} {r['null_mean'][1]:8.3f}"
              f"   p95 {r['null_p95'][1]:.3f}")
    print("\n  LibriBrain (0.512 s, chapter 12): 0.166 / 0.666; pitch 0.15-0.18 / 0.61-0.67")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "timing.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"  saved {(OUT / 'timing.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
