"""How predictable is the listening brain beyond its power spectrum? Re-measured.

The lost version's pitch: "Almost all of it comes from the signal's frequency
spectrum, and about half of what first looked like extra structure was the
recording changing between sessions." Its figure (``docs/recovered/
pitch-predictability.png``): 12 listeners x 8 sensors, five bands, horizons
4-256 ms, beyond-spectrum predictability *across* sessions (fit chapter 11, test
chapter 12) against *within* one session (two halves of chapter 12), with
40-60% of the across-session excess in broadband, theta and alpha attributed to
the session changing.

Beyond the spectrum = held-out NLL of the PSD-matched null (64-lag linear
predictor) minus that of the defended heteroscedastic model (same mean, variance
from recent energy) -- both from :mod:`neurocast.atlas.estimators`, which are
validated against analytic answers. Bands are causal FIRs, so no future leaks
into the past.

    .venv/Scripts/python.exe scripts/run_libribrain_predictability.py
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

from neurocast.atlas import estimators as est  # noqa: E402
from neurocast.atlas.filters import causal_filter, fir_bandpass  # noqa: E402
from neurocast.data.canonical import to_canonical  # noqa: E402
from neurocast.data.libribrain import find_runs, read_meg  # noqa: E402

BANDS = {"broadband": None, "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 70.0)}
HORIZONS = (1, 4, 16, 64)         # samples at 250 Hz: 4, 16, 64, 256 ms
MAX_LAGS = 64
OUT = ROOT / "runs" / "libribrain_audit"


def sensors(info: dict, n: int) -> np.ndarray:
    """``n`` magnetometers spread evenly through the channel list (deterministic)."""
    mags = np.flatnonzero(np.array(info["channel_types"]) == "mag")
    return mags[np.linspace(0, len(mags) - 1, n).round().astype(int)]


def session_share(across: np.ndarray, within: np.ndarray) -> float:
    """Share of the across-session excess that is the session changing.

    Only horizons where there IS structure across sessions count, and negative
    within-session values count as zero: a model doing worse than the null is not
    "negative structure". A first version summed signed values and printed
    shares like 209%.
    """
    pos = across > 0
    if not pos.any():
        return float("nan")
    return float(1.0 - np.maximum(within[pos], 0).sum() / across[pos].sum())


def beyond(train: np.ndarray, test: np.ndarray, h: int) -> float:
    # The variance model's MEAN must have the null's order (64 lags). With a
    # 16-lag mean, band-limited Gaussian noise -- nothing beyond its spectrum --
    # reads -0.16 to -0.45 nats/sample: the difference measures mean-model order,
    # not structure. validate_atlas.py section 11 pins this.
    spec = est.spectral_null(train, test, h, MAX_LAGS)
    het = est.heteroscedastic(train, test, h, MAX_LAGS, MAX_LAGS)
    return float(spec.nll - het.nll)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listeners", type=int, default=12)
    ap.add_argument("--sensors", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=300.0, help="per segment")
    ap.add_argument("--guard", type=float, default=60.0, help="between within-session halves")
    args = ap.parse_args()
    t0 = time.time()

    subs = [str(i) for i in range(1, args.listeners + 1)]       # sub-1..12: whole sessions
    res = {b: {"across": np.zeros((len(subs), len(HORIZONS))),
               "within": np.zeros((len(subs), len(HORIZONS)))} for b in BANDS}
    for si, sub in enumerate(subs):
        r11 = find_runs(sessions={"11"}, subjects={sub}, require_meg=True)
        r12 = find_runs(sessions={"12"}, subjects={sub}, require_meg=True)
        if not (r11 and r12):
            print(f"  sub-{sub}: missing MEG, skipped")
            continue
        x11, i11 = read_meg(r11[0].h5)
        x12, i12 = read_meg(r12[0].h5)
        ch = sensors(i12, args.sensors)
        fs = i12["sfreq"]
        n = int(args.seconds * fs)
        g = int(args.guard * fs)
        a = to_canonical(x11[ch], [i11["channel_types"][c] for c in ch])[0][:, :n]
        b12 = to_canonical(x12[ch], [i12["channel_types"][c] for c in ch])[0]
        within_tr, within_te = b12[:, :n], b12[:, n + g: 2 * n + g]
        across_te = b12[:, n + g: 2 * n + g]           # the same test data in both conditions
        for band, edges in BANDS.items():
            if edges is None:
                f = lambda z: z  # noqa: E731
            else:
                h_fir = fir_bandpass(edges[0], edges[1], fs, n_taps=129)
                f = lambda z, h_fir=h_fir: causal_filter(z, h_fir)[:, 128:]  # noqa: E731
            for hi, h in enumerate(HORIZONS):
                res[band]["across"][si, hi] = beyond(f(a), f(across_te), h)
                res[band]["within"][si, hi] = beyond(f(within_tr), f(within_te), h)
        print(f"  sub-{sub} done ({time.time() - t0:.0f}s)", flush=True)

    print(f"\nbeyond the power spectrum, nats/sample, mean over {len(subs)} listeners x "
          f"{args.sensors} sensors (horizons {[4 * h for h in HORIZONS]} ms)")
    print(f"  {'band':<10} {'across sessions':>30} {'within one session':>30} {'session share':>14}")
    summary = {}
    for band in BANDS:
        acr, wit = res[band]["across"].mean(0), res[band]["within"].mean(0)
        share = session_share(acr, wit)
        summary[band] = {"across": acr.tolist(), "within": wit.tolist(), "session_share": share}
        print(f"  {band:<10} {' '.join(f'{v:+7.4f}' for v in acr):>30} "
              f"{' '.join(f'{v:+7.4f}' for v in wit):>30} {100 * share:>13.0f}%")
    print("  pitch: across > within for broadband, theta, alpha (40-60% from the session);")
    print("         beta and gamma similar.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predictability.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  {time.time() - t0:.0f}s; saved {(OUT / 'predictability.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
