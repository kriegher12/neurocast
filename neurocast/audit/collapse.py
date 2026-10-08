"""Is a masked model reading its input? The masked-collapse diagnostic.

The lost version of this project found that its masked-reconstruction model had
collapsed on real MEG: "its output ignored its input." That is not a wiring
error -- gradients flow, and an untrained model passes its input through -- but
a training failure: hidden patches are hard enough to predict that the mean is a
strong optimum. It diagnosed the failure with three measurements, rebuilt here,
which any masked checkpoint can be put through:

1. **Spread.** Variance of the predictions at hidden positions across samples,
   relative to the targets' variance. A collapsed model predicts (nearly) the
   same thing for every input: spread -> 0.
2. **Input sensitivity.** How much hidden-position predictions move when the
   visible context is swapped for another sample's, relative to how much they
   vary at all. Collapsed: ~0.
3. **Achievable versus achieved.** Held-out R^2 of the model at hidden positions,
   next to a ridge regression predicting each hidden target from its visible
   neighbours' targets -- under the *same* mask. A model far below a linear
   baseline is not using its context.

The pitch's "a linear predictor recovers 34-48% of a hidden patch from its
neighbours" cannot be the same-mask number: its masks hide whole 1-s blocks (16
patches, every region at once), so a hidden patch's near neighbours are almost
always hidden too. It is the *isolated* case -- one patch hidden, its neighbours
visible -- which says whether the information is there at all. Both are
reported: ``reference_r2`` (same mask) and ``isolated_r2``.

All functions take arrays the caller extracted from its own model, so the
diagnostic does not depend on one architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CollapseReport", "neighbour_r2", "isolated_r2", "diagnose"]


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def neighbour_r2(targets: np.ndarray, hidden: np.ndarray, n_groups: int,
                 reach: int = 2, ridge: float = 1.0, seed: int = 0,
                 visible: np.ndarray | None = None) -> float:
    """Linear reference: predict each hidden target from visible neighbours' targets.

    ``targets`` is ``(B, N, k)`` in AR order (``N = T' * G``), ``hidden`` ``(B, N)``.
    For a hidden position ``(t, g)`` the features are the targets of the same
    group at patches ``t-reach..t+reach`` (zeroed where not ``visible`` -- by
    default ``~hidden`` -- or out of range). Fitted on half the samples, scored
    on the other half.
    """
    b, n, k = targets.shape
    t_count = n // n_groups
    tg = targets.reshape(b, t_count, n_groups, k)
    hid = hidden.reshape(b, t_count, n_groups)
    vis = (~hidden if visible is None else visible).reshape(b, t_count, n_groups)
    feats, ys, owner = [], [], []
    for i in range(b):
        for t in range(t_count):
            for g in range(n_groups):
                if not hid[i, t, g]:
                    continue
                parts = []
                for dt in range(-reach, reach + 1):
                    if dt == 0:
                        continue
                    u = t + dt
                    ok = 0 <= u < t_count and vis[i, u, g]
                    parts.append(tg[i, u, g] if ok else np.zeros(k))
                feats.append(np.concatenate(parts))
                ys.append(tg[i, t, g])
                owner.append(i)
    x, y, owner = np.array(feats), np.array(ys), np.array(owner)
    half = np.random.default_rng(seed).permutation(b)[: b // 2]
    tr = np.isin(owner, half)
    a = np.column_stack([np.ones(tr.sum()), x[tr]])
    w = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ y[tr])
    return _r2(np.column_stack([np.ones((~tr).sum()), x[~tr]]) @ w, y[~tr])


def isolated_r2(targets: np.ndarray, n_groups: int, valid: np.ndarray | None = None,
                **kw) -> float:
    """:func:`neighbour_r2` with one patch hidden at a time: is the information there?

    Every position is a target and its neighbours are always visible. ``valid``
    (``(B, N)`` bool) drops positions that carry no target, e.g. empty regions.
    """
    every = np.ones(targets.shape[:2], bool) if valid is None else valid
    return neighbour_r2(targets, every, n_groups, visible=every, **kw)


@dataclass(frozen=True)
class CollapseReport:
    spread: float          # var(pred) / var(target) at hidden positions
    sensitivity: float     # change under context swap / spread of predictions
    model_r2: float        # model's R^2 at hidden positions
    reference_r2: float    # linear neighbour baseline, same mask
    isolated_r2: float = float("nan")   # linear, one patch hidden at a time

    @property
    def collapsed(self) -> bool:
        return self.spread < 0.05 or self.sensitivity < 0.05

    def summary(self) -> str:
        state = "COLLAPSED -- output ignores input" if self.collapsed else "reads its input"
        return (f"spread {self.spread:.3f}  sensitivity {self.sensitivity:.3f}  "
                f"R^2 model {self.model_r2:+.3f} vs linear neighbours {self.reference_r2:+.3f} "
                f"(one patch hidden: {self.isolated_r2:+.3f})  [{state}]")


def diagnose(pred: np.ndarray, pred_swapped: np.ndarray, targets: np.ndarray,
             hidden: np.ndarray, n_groups: int, valid: np.ndarray | None = None) -> CollapseReport:
    """Run the three measurements.

    ``pred`` -- ``(B, N, k)`` model predictions with the true context;
    ``pred_swapped`` -- the same hidden positions, with each sample's *visible*
    input replaced by another sample's (same mask); ``targets`` and ``hidden`` as
    in :func:`neighbour_r2`; ``valid`` as in :func:`isolated_r2`.
    """
    p, ps, y = pred[hidden], pred_swapped[hidden], targets[hidden]
    spread = float(p.var(0).sum() / max(y.var(0).sum(), 1e-12))
    sens = float(((p - ps) ** 2).mean() / max(p.var(0).mean() * 2, 1e-12))
    return CollapseReport(spread=spread, sensitivity=sens, model_r2=_r2(p, y),
                          reference_r2=neighbour_r2(targets, hidden, n_groups),
                          isolated_r2=isolated_r2(targets, n_groups, valid))
