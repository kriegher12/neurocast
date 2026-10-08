"""The demo, rehearsed: a brain dreamed forward, next to the brain itself.

Left, a held-out subject's real sensor activity as a topographic movie. Right,
NeuroCast's forecast of the same interval, rendered on the same colour scale.
Below, the spatial correlation between the two, next to a control: the same model
forecasting from *another trial's* context. The forecast is only worth looking at
where its trace separates from the control's.

    .venv/Scripts/python.exe scripts/run_demo.py                 # ~2 min, CPU
    .venv/Scripts/python.exe scripts/run_demo.py --steps 600 --sites 64

Writes ``runs/demo/forecast.gif`` and ``runs/demo/forecast_strip.png``.

What to expect, stated before it is seen
----------------------------------------
This is the rehearsal rung on the synthetic source-space corpus, and
``scripts/validate_generative.py`` has already measured that corpus with the
Atlas: forecastability is largely gone within one 64 ms patch. So the forecast
should track the truth for the first tens of milliseconds and then relax toward
the mean -- fading on the shared scale rather than inventing structure -- with
its correlation trace falling to the control's. A forecast that stayed vivid and
correlated for a full second on this corpus would be a bug, not a result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurocast.data.sources import SourceSpaceCorpus  # noqa: E402
from neurocast.models.forecast import rollout  # noqa: E402
from neurocast.train.generative import train_generative  # noqa: E402
from neurocast.viz.topomap import (  # noqa: E402
    render_movie,
    render_strip,
    select_channels,
    spatial_correlation,
)
from validate_tokenizer import megin_montage  # noqa: E402

PATCH = 16
TRAIN_SUBJECTS = [0, 1, 2, 3, 4, 5]
HELD_OUT = [6, 7]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=48, help="MEGIN sites (3 channels each)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--context", type=int, default=32, help="context patches (64 ms each)")
    ap.add_argument("--horizon", type=int, default=16, help="forecast patches (16 = 1.02 s)")
    ap.add_argument("--trials", type=int, default=6, help="held-out trials to average")
    ap.add_argument("--out", default=str(ROOT / "runs" / "demo"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    corpus = SourceSpaceCorpus(n_sources=6, n_subjects=8, seed=args.seed)
    meg = megin_montage(n_sites=args.sites)
    n_ctx, n_fut = args.context * PATCH, args.horizon * PATCH

    print("=" * 80)
    print("DEMO REHEARSAL -- real vs forecast, held-out subjects")
    print("=" * 80)
    print(f"  montage {len(meg)} ch; train subjects {TRAIN_SUBJECTS}, shown {HELD_OUT}")
    print(f"  context {n_ctx / corpus.fs:.2f} s, forecast {n_fut / corpus.fs:.2f} s")
    print()
    gm = train_generative(corpus, meg, n_patch=args.context, steps=args.steps, rank=16,
                          subjects=TRAIN_SUBJECTS, seed=args.seed, verbose=True)
    print(f"  {gm.model.describe()}")
    print(f"  NLL {gm.initial_loss:+.3f} -> {gm.final_loss:+.3f} nats/component "
          f"in {gm.seconds:.0f}s")

    rng = np.random.default_rng(args.seed + 1000)
    trials = corpus.batch(meg, args.trials + 1, n_ctx + n_fut, rng, subjects=HELD_OUT).x
    idx = select_channels(meg)
    brain = np.concatenate([gm.basis.channels[g] for g in range(gm.basis.n_groups - 1)])

    r_model, r_ctrl, mse_model, mse_mean = [], [], [], []
    shown = None
    for i in range(args.trials):
        ctx, future = trials[i, :, :n_ctx], trials[i, :, n_ctx:]
        other = trials[i + 1, :, :n_ctx]                     # another trial's context
        fc = rollout(gm.model, gm.head, gm.basis, ctx, args.horizon)
        ctrl = rollout(gm.model, gm.head, gm.basis, other, args.horizon)
        r_model.append(spatial_correlation(future[idx], fc[idx]))
        r_ctrl.append(spatial_correlation(future[idx], ctrl[idx]))
        mean_fc = np.repeat(ctx.mean(axis=1, keepdims=True), n_fut, axis=1)
        mse_model.append(((fc[brain] - future[brain]) ** 2).reshape(brain.size, -1, PATCH).mean((0, 2)))
        mse_mean.append(((mean_fc[brain] - future[brain]) ** 2).reshape(brain.size, -1, PATCH).mean((0, 2)))
        if shown is None:
            shown = (future, fc, ctrl)

    r_model, r_ctrl = np.mean(r_model, 0), np.mean(r_ctrl, 0)
    ratio = np.mean(mse_model, 0) / np.mean(mse_mean, 0)

    print()
    print(f"  averaged over {args.trials} held-out trials")
    print(f"  {'window':>14} {'spatial r':>10} {'control r':>10} {'MSE / mean-forecast':>20}")
    for p in range(args.horizon):
        sl = slice(p * PATCH, (p + 1) * PATCH)
        lo, hi = 1000 * p * PATCH / corpus.fs, 1000 * (p + 1) * PATCH / corpus.fs
        print(f"  {lo:5.0f}-{hi:4.0f} ms {r_model[sl].mean():>10.3f} {r_ctrl[sl].mean():>10.3f} "
              f"{ratio[p]:>20.3f}")
        if p == 3 and args.horizon > 6:
            print(f"  {'...':>14}")
            break
    tail = slice(4 * PATCH, None)
    print(f"  {'256 ms - end':>14} {r_model[tail].mean():>10.3f} {r_ctrl[tail].mean():>10.3f} "
          f"{np.mean(ratio[4:]):>20.3f}")

    out = Path(args.out)
    future, fc, ctrl = shown
    gif = render_movie(future, fc, meg, out / "forecast.gif", fs=corpus.fs, control=ctrl,
                       title="held-out subject")
    png = render_strip(future, fc, meg, out / "forecast_strip.png", fs=corpus.fs)
    print()
    print(f"  wrote {gif.relative_to(ROOT)} and {png.relative_to(ROOT)}")
    print()
    print("  Reading it: the forecast should track the truth for the first patch or two")
    print("  and then relax toward the mean -- fading on the shared colour scale -- as its")
    print("  correlation falls to the control's. That is this corpus's measured")
    print("  predictability horizon (see validate_generative.py), not a rendering choice.")
    print(f"\n  wall clock {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
