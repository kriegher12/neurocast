"""Forecast loss on people the forecaster never saw, against per-person linear nulls (README 2.11).

Roadmap step 1's last item. The forecaster was pre-trained on one listener
(section 2.2). Does it forecast the 32 broad listeners' MEG better than a
predictor fitted to each of them?

The forecaster's loss is in its own target space: whitened per-region PCA of
each 64-ms patch (``train/targets.py``), in nats per target dimension. A
per-sensor PSD-matched null (``eval/references.py``) lives in another space, so
the nulls here are fitted in the same target space, per listener and per model:

* ``linear``: a Gaussian linear predictor of each region's target from the
  previous 8 patches of every region and the earlier regions of the same patch --
  the order the forecaster factorises in -- ridge-fitted on the listener's own
  chapter 11 (penalty chosen on its last fifth), residual variance per
  dimension. For a stationary Gaussian this is the PSD-matched predictor;
* ``marginal``: a Gaussian per dimension, from chapter 11.

Everything is scored on chapter 12 (never in pre-training), on the targets of
patches 8-63 of non-overlapping 4.1-s windows, so the nulls and the forecaster
see the same context. The deep listener's chapters 11-12 (also never in
pre-training) are the seen-person reference. Models: section 2.2's forecasters
and, when trained, section 2.10's conditioned ones (conditioned on the
listener's chapter-11 spectra).

    .venv-gpu/Scripts/python.exe scripts/run_forecast_unseen.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "runs" / "forecast_unseen"
K = 8                        # patches of context for the linear null
MAX_EVAL = 200               # chapter-12 windows per listener
ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3)


class Chapter:
    """Non-overlapping 4.1-s windows of one listener's chapter, in session-normalised units."""

    def __init__(self, listener: str, ses: str) -> None:
        if listener == "0":
            from run_bakeoff_real import read_session

            x, _, _ = read_session(ses)                        # (T, C) float16
            self.window = lambda s: np.asarray(x[s:s + 1024]).T
            self.n = x.shape[0]
        else:
            from run_identity_removal import cache_dir, runs

            from neurocast.train.pretrain import open_cached

            rec = open_cached(cache_dir(), ["_".join(runs(ses)[listener].key)])[0]
            self.window = lambda s: rec.window(s, 1024)
            self.n = rec.n_samples
        self.starts = list(range(0, self.n - 1024 + 1, 1024))

    def batches(self, limit=None, size=16):
        st = self.starts if limit is None else self.starts[:limit]
        for i in range(0, len(st), size):
            yield np.stack([self.window(s) for s in st[i:i + size]]).astype(np.float32)


def targets(tr, x: np.ndarray) -> np.ndarray:
    """``(B, P, G, r)`` whitened targets, zero where a dimension is not real."""
    t = tr.basis.encode(torch.as_tensor(x)).numpy()
    m = tr.dim_mask.cpu().numpy()
    return (t * m).reshape(x.shape[0], -1, tr.G, t.shape[-1])


def design(z: np.ndarray, g: int, dims: list[np.ndarray]):
    """Rows for predicting region ``g`` at patches K.. from the past K patches and earlier regions."""
    b, p = z.shape[:2]
    cols = []
    for k in range(1, K + 1):
        cols += [z[:, K - k:p - k, h][..., dims[h]] for h in range(len(dims)) if len(dims[h])]
    cols += [z[:, K:, h][..., dims[h]] for h in range(g) if len(dims[h])]
    x = np.concatenate(cols, axis=-1).reshape(-1, sum(c.shape[-1] for c in cols))
    y = z[:, K:, g][..., dims[g]].reshape(-1, len(dims[g]))
    return x, y


def fit_linear(x, y):
    """Ridge on standardised inputs, penalty by the last fifth; returns a predictor and variances."""
    n = len(x)
    cut = int(n * 0.8)
    mu, sd = x[:cut].mean(0), x[:cut].std(0) + 1e-6

    def solve(a, rows):
        z = (x[rows] - mu) / sd
        ym = y[rows].mean(0)
        w = np.linalg.solve(z.T @ z + a * len(z) * np.eye(z.shape[1]) / 1000, z.T @ (y[rows] - ym))
        return w, ym

    best = None
    for a in ALPHAS:
        w, ym = solve(a, slice(0, cut))
        err = np.mean((((x[cut:] - mu) / sd) @ w + ym - y[cut:]) ** 2)
        if best is None or err < best[0]:
            best = (err, a)
    w, ym = solve(best[1], slice(0, n))
    var = np.mean((((x - mu) / sd) @ w + ym - y) ** 2, axis=0) + 1e-6
    return (lambda q: ((q - mu) / sd) @ w + ym), var, best[1]


def gauss_nll(err: np.ndarray, var: np.ndarray) -> float:
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * err ** 2 / var))


@torch.no_grad()
def model_nll(tr, x: np.ndarray, cond=None) -> np.ndarray:
    """Per-target-token NLL per dimension, ``(B, P, G)`` (NaN where nothing is predicted)."""
    from neurocast.objectives.arms import ar_shift

    xt = torch.as_tensor(x, device=tr.device)
    tgt = tr.basis.encode(xt.cpu()).to(tr.device)
    with torch.autocast(tr.device.type, dtype=tr.amp_dtype, enabled=tr.amp_dtype is not None):
        h, t = ar_shift(tr.model(xt), tgt)
        if cond is not None:
            h = h + tr.cond_proj(cond)[None, None, :].to(h.dtype)
    m = tr.dim_mask[1:].expand_as(t)
    lp = tr.head.log_prob(h.float(), t.float(), m)
    dims = m.float().sum(-1)
    nll = (-lp / dims.clamp_min(1)).float().cpu().numpy()
    nll[(dims == 0).cpu().numpy()] = np.nan
    full = np.full((x.shape[0], nll.shape[1] + 1), np.nan)
    full[:, 1:] = nll                                          # token j's NLL sits at j
    return full.reshape(x.shape[0], -1, tr.G)


def listener_cond(tr, ch: Chapter):
    """A conditioned model's profile for a listener, from that listener's chapter 11."""
    from neurocast.adapt.nuisance import compute

    starts = np.linspace(0, ch.n - 1024, 16).astype(int)
    d = np.mean([compute(ch.window(int(s)).astype(np.float64), 250.0).to_matrix()[:, :-1]
                 for s in starts], axis=0).ravel()
    z = (d - tr.cond_stats["mean"]) / tr.cond_stats["sd"]
    return torch.as_tensor(z, dtype=torch.float32, device=tr.device)


def main() -> int:
    from run_bakeoff_real import make_trainer, specs

    from neurocast.train.pretrain import Pretrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--listeners", nargs="*", help="smoke tests only")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--max-eval", type=int, default=MAX_EVAL)
    ap.add_argument("--out", type=Path, help="smoke tests only")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = args.out.resolve()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
    OUT.mkdir(parents=True, exist_ok=True)
    sp = specs()
    names = args.models or [k for k in ("ar_lik_s0", "ar_lik_s1", "ar_lik_s2", "ar_lik_cond_s0",
                                        "ar_lik_cond_s1", "ar_lik_cond_s2")
                            if Pretrainer.best_path(sp[k]["ckpt"]).exists()]
    listeners = args.listeners or ["0"] + [str(i) for i in range(1, 33)]
    for name in names:
        f = OUT / f"{name}.json"
        res = json.loads(f.read_text()) if f.exists() else {}
        todo = [p for p in listeners if p not in res]
        if not todo:
            continue
        arm = sp[name]["arm"]
        tr = make_trainer(arm, sp[name]["seed"], 20_000, args.device, prefetch=0)
        tr.load(Pretrainer.best_path(sp[name]["ckpt"]))
        tr.model.eval()
        dims = [np.flatnonzero(tr.dim_mask.cpu().numpy()[g]) for g in range(tr.G)]
        for who in todo:
            fit, ev = Chapter(who, "11"), Chapter(who, "12")
            z_fit = np.concatenate([targets(tr, x) for x in fit.batches()])
            cond = listener_cond(tr, fit) if tr.cond is not None else None
            z_ev, nl = [], []
            for x in ev.batches(limit=args.max_eval):
                z_ev.append(targets(tr, x))
                nl.append(model_nll(tr, x, cond))
            z_ev, nl = np.concatenate(z_ev), np.concatenate(nl)
            row = {"model": [], "linear": [], "marginal": [], "alpha": []}
            for g in range(tr.G):
                if not len(dims[g]):
                    continue
                xf, yf = design(z_fit, g, dims)
                xe, ye = design(z_ev, g, dims)
                pred, var, a = fit_linear(xf, yf)
                row["linear"].append(gauss_nll(pred(xe) - ye, var))
                mu, mv = yf.mean(0), yf.var(0) + 1e-6
                row["marginal"].append(gauss_nll(ye - mu, mv))
                row["model"].append(float(np.nanmean(nl[:, K:, g])))
                row["alpha"].append(a)
            out = {k: float(np.mean(v)) for k, v in row.items() if k != "alpha"}
            out.update(delta_linear=out["model"] - out["linear"], delta_marginal=out["model"] - out["marginal"],
                       n_windows=len(z_ev), alphas=row["alpha"])
            res[who] = out
            f.write_text(json.dumps(res, indent=2))
            print(f"  {name} listener {who:>2}: model {out['model']:.4f}  linear null {out['linear']:.4f}  "
                  f"marginal {out['marginal']:.4f}  ->  model - linear {out['delta_linear']:+.4f}", flush=True)
        del tr
        torch.cuda.empty_cache()

    summary = {}
    for f in sorted(OUT.glob("*.json")):
        if f.stem == "summary":
            continue
        r = json.loads(f.read_text())
        unseen = [v["delta_linear"] for k, v in r.items() if k != "0"]
        if "0" in r and unseen:
            d0 = r["0"]["delta_linear"]
            summary[f.stem] = {"deep_listener": d0, "unseen_mean": float(np.mean(unseen)),
                               "unseen_sd": float(np.std(unseen)), "unseen_n": len(unseen),
                               "unseen_beaten_fraction": float(np.mean(np.array(unseen) < 0)),
                               "transfer_ratio": float(np.mean(unseen) / d0) if d0 < 0 else None}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"  {k}: deep listener {v['deep_listener']:+.4f}; unseen {v['unseen_mean']:+.4f} +- "
              f"{v['unseen_sd']:.4f} ({100 * v['unseen_beaten_fraction']:.0f}% beat the linear null)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
