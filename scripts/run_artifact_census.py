"""Large transients in the deep listener's pre-training hours, and where calibration fell.

README section 2.5 found a ~1,000x artifact in Sherlock1 session 3 whose energy the
release's zero-phase filters spread ~14 s backwards; one target-calibration window
landing in that pre-echo corrupted a whole run. This counts such events over every
cached session of the section 2.2 pre-training data (sub-0, Sherlock1-9, story
spans, session-normalised units -- ordinary 4-s windows peak at ~10):

* per second, the largest ``|x|`` over channels;
* events, as runs of consecutive seconds above 20, 50, 100 and 1,000, per hour;
* the 4-s window maximum's distribution (from the per-second maxima);
* each section 2.2 seed's 64 calibration windows (every arm of a seed draws the
  same ones): their peaks, and how many a guard at ``GUARD`` would reject.

    .venv/Scripts/python.exe scripts/run_artifact_census.py

Writes ``runs/artifact_census/census.json`` and the per-second maxima
(``persecond.npz``); a session already in the npz is not read again.
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
sys.path.insert(0, str(ROOT / "scripts"))

from neurocast.train.pretrain import _read_retrying  # noqa: E402

OUT = ROOT / "runs" / "artifact_census"
THRESHOLDS = (20.0, 50.0, 100.0, 1000.0)
GUARD = 100.0
SFREQ = 250


def per_second_max(path: Path, chunk_s: int = 120) -> np.ndarray:
    """Largest ``|x|`` over channels in each whole second of a cached session."""
    a = np.load(path, mmap_mode="r")
    n, c = a.shape
    item = a.dtype.itemsize
    out = []
    for s0 in range(0, n // SFREQ, chunk_s):
        n_s = min(chunk_s, n // SFREQ - s0)
        buf = _read_retrying(str(path), a.offset + s0 * SFREQ * c * item, n_s * SFREQ * c * item)
        x = np.frombuffer(buf, dtype=a.dtype).reshape(n_s, SFREQ, c)
        out.append(np.abs(x.astype(np.float32)).max(axis=(1, 2)))
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def events(m: np.ndarray, thr: float) -> list[tuple[int, int, float]]:
    """Runs of consecutive seconds above ``thr``: ``(start s, duration s, peak)``."""
    above = np.concatenate([[False], m > thr, [False]])
    d = np.diff(above.astype(int))
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(a), int(b - a), float(m[a:b].max())) for a, b in zip(starts, stops)]


def main() -> int:
    from run_pretrain_scale import cache_dir, deep_runs, heldout_keys

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = deep_runs()
    keys = ["_".join(r.key) for r in runs]
    held = set(heldout_keys(runs))
    store = OUT / "persecond.npz"
    per = dict(np.load(store)) if store.exists() else {}
    t0 = time.time()
    for i, k in enumerate(keys):
        if k in per:
            continue
        per[k] = per_second_max(cache_dir() / f"{k}.npy")
        np.savez(store, **per)
        print(f"  {k}: {len(per[k]) / 60:.1f} min, peak {per[k].max():.0f} "
              f"({i + 1}/{len(keys)}, {time.time() - t0:.0f} s)", flush=True)

    hours = sum(len(v) for v in per.values()) / 3600
    res = {"sessions": len(per), "hours": hours, "guard": GUARD, "thresholds": {}}
    for thr in THRESHOLDS:
        ev = [(k, *e) for k, m in per.items() for e in events(m, thr)]
        res["thresholds"][str(thr)] = {
            "events": len(ev), "per_hour": len(ev) / hours,
            "seconds": int(sum(e[2] for e in ev)),
            "sessions": sorted({e[0] for e in ev}),
            "heldout_sessions": sorted({e[0] for e in ev} & held)}
    allev = sorted(((k, *e) for k, m in per.items() for e in events(m, THRESHOLDS[0])),
                   key=lambda e: -e[3])
    res["largest"] = [{"session": k, "start_s": a, "duration_s": d, "peak": p}
                      for k, a, d, p in allev[:25]]
    # A 4-s window's peak, from the per-second maxima (windows at whole seconds).
    win = np.concatenate([np.lib.stride_tricks.sliding_window_view(m, 5).max(1)
                          for m in per.values() if len(m) >= 5])
    res["window_peak_percentiles"] = {str(q): float(np.percentile(win, q))
                                      for q in (50, 90, 99, 99.9, 99.99)}
    res["window_fraction_above_guard"] = float((win > GUARD).mean())

    # Calibration windows of the section 2.2 runs: a tiny-rung trainer draws exactly
    # the windows the 6m runs drew (the draws depend on seed, data and window only).
    from run_pretrain_pilot import montage_for

    from neurocast.train.pretrain import PretrainConfig, Pretrainer, open_cached

    recs = open_cached(cache_dir(), keys, heldout=sorted(held))
    res["calibration"] = {}
    for seed in args.seeds:
        tr = Pretrainer(PretrainConfig(rung="tiny", seed=seed), montage_for(runs[0]), recs,
                        device="cpu")
        peaks = [float(np.abs(r.window(st, tr.n_times).astype(np.float32)).max())
                 for r, st, _ in tr.calibration_draws()]
        worst = int(np.argmax(peaks))
        r, st, _ = tr.calibration_draws()[worst]
        res["calibration"][str(seed)] = {
            "max_peak": max(peaks), "median_peak": float(np.median(peaks)),
            "worst": {"session": r.key, "start_s": st / SFREQ},
            "rejected_at_guard": int(sum(p > GUARD for p in peaks))}
        print(f"  seed {seed}: calibration peaks median {np.median(peaks):.1f}, max {max(peaks):.1f} "
              f"({r.key} at {st / SFREQ:.0f} s); a guard at {GUARD:.0f} rejects "
              f"{res['calibration'][str(seed)]['rejected_at_guard']}", flush=True)

    print(f"\n  {res['sessions']} sessions, {hours:.1f} h")
    for thr, v in res["thresholds"].items():
        print(f"  above {float(thr):6.0f}: {v['events']:4d} events ({v['per_hour']:.2f}/h), "
              f"{v['seconds']} s, in {len(v['sessions'])} sessions "
              f"({len(v['heldout_sessions'])} held out)")
    print("  4-s window peak percentiles: " + ", ".join(
        f"{q}% {v:.1f}" for q, v in res["window_peak_percentiles"].items()))
    print("  largest events:")
    for e in res["largest"][:10]:
        print(f"    {e['session']:<22} {e['start_s']:6d} s  {e['duration_s']:4d} s  peak {e['peak']:.0f}")
    (OUT / "census.json").write_text(json.dumps(res, indent=2))
    print(f"saved {(OUT / 'census.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
