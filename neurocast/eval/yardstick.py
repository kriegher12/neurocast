"""A yardstick with brain signal in it, and person information net of spectra (README 2.6).

The 50-word task cannot separate pre-training objectives: a no-brain model that
hears only the window's broadband envelope beats every brain decoder (section
2.4). Roadmap step 1 asks for tasks where MEG is known to carry the answer, each
beside rows that need no brain; step 2 asks how much of a model's person
information is the spectral calibration a likelihood head needs. This module
holds the pieces both share, on frozen features of a pre-trained model:

* **labels from the events files**: speech or silence for every 64-ms patch, and
  the 39 ARPAbet phonemes (``oov`` dropped) with voicing, word position and timing;
* :func:`lagged`: a patch's label read from the features 64-320 ms later, where
  the auditory response is, for causal and bidirectional models alike;
* :class:`LinearProbe`: multinomial logistic regression on standardised features,
  class-balanced, weight decay chosen on a validation chapter, deterministic;
* metrics: macro-F1 (the 2025 competition's), balanced accuracy, AUC;
* step 2: per-region and per-channel spectral descriptors of a window
  (``adapt/nuisance.py``) and :func:`cross_listener`: how much of the listener
  differences in the features a map from descriptors predicts for listeners it
  never saw, and the identity audit on what it leaves -- beside a control with
  the same descriptors shuffled across windows;
* :func:`spread`: collapse measures for token features that need no calibration;
* :func:`leace`: closed-form linear erasure of a concept (Belrose et al., NeurIPS
  2023): after it, every class has the same mean, so no linear classifier beats
  the majority guess on the data it was fitted on, at the least change to the
  features (README 2.8 uses it on person identity).

NumPy and torch only.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "PHONEMES", "VOICED", "POSITIONS", "PATCH_S", "LAGS",
    "phoneme_class", "speech_labels", "patch_index", "lagged", "usable",
    "macro_f1", "balanced_accuracy", "auc",
    "LinearProbe", "fit_select",
    "window_descriptors", "region_means", "residualise", "top_pcs", "spread", "cross_listener",
    "leace",
]

#: The 39 ARPAbet phonemes of the LibriBrain events (stress removed); "oov" is dropped.
PHONEMES = ("aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh", "eh", "er", "ey", "f",
            "g", "hh", "ih", "iy", "jh", "k", "l", "m", "n", "ng", "ow", "oy", "p", "r", "s",
            "sh", "t", "th", "uh", "uw", "v", "w", "y", "z", "zh")
#: Voiced phonemes: every vowel and the voiced consonants.
VOICED = frozenset("aa ae ah ao aw ay eh er ey ih iy ow oy uh uw "
                   "b d dh g jh l m n ng r v w y z zh".split())
#: Position of a phoneme in its word, from the events' suffix: begin, inside, end, single.
POSITIONS = ("B", "I", "E", "S")
PATCH_S = 16 / 250.0
#: Patches after the labelled one whose features are read: 64-320 ms.
LAGS = (1, 2, 3, 4)


def phoneme_class(segment: str) -> tuple[int, int]:
    """``"uh_I"`` -> (index of ``uh``, index of ``I``); unknown or ``oov`` -> (-1, -1)."""
    base, _, pos = str(segment).strip().lower().partition("_")
    base = base.rstrip("012")
    if base not in PHONEMES:
        return -1, -1
    pos = pos.upper()
    return PHONEMES.index(base), POSITIONS.index(pos) if pos in POSITIONS else -1


def speech_labels(onsets_s: np.ndarray, durations_s: np.ndarray, n_patches: int,
                  t0_s: float = 0.0, patch_s: float = PATCH_S) -> np.ndarray:
    """1 where a patch's centre is outside every silence interval, else 0; int8."""
    on = np.asarray(onsets_s, float)
    order = np.argsort(on)
    on, end = on[order], on[order] + np.asarray(durations_s, float)[order]
    c = t0_s + (np.arange(n_patches) + 0.5) * patch_s
    i = np.searchsorted(on, c, side="right") - 1
    inside = (i >= 0) & (c < end[np.clip(i, 0, None)]) if len(on) else np.zeros(n_patches, bool)
    return (~inside).astype(np.int8)


def patch_index(t_s: np.ndarray, t0_s: float = 0.0, patch_s: float = PATCH_S) -> np.ndarray:
    """The patch holding each time (s); ``t0_s`` is the time of patch 0's start."""
    return np.floor((np.asarray(t_s, float) - t0_s) / patch_s + 1e-9).astype(int)


def usable(idx: np.ndarray, valid: np.ndarray, lags=LAGS) -> np.ndarray:
    """Patches whose every lagged feature exists."""
    idx = np.asarray(idx)
    ok = (idx >= 0) & (idx + max(lags) < len(valid))
    for lag in lags:
        ok[ok] &= valid[idx[ok] + lag]
    return ok


def lagged(feats: np.ndarray, idx: np.ndarray, lags=LAGS) -> np.ndarray:
    """``(len(idx), len(lags) * d)``: each patch's features ``lags`` patches later."""
    return np.concatenate([feats[np.asarray(idx) + lag] for lag in lags], axis=1)


# -- metrics ---------------------------------------------------------------------------
def _confusion(pred: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(np.asarray(y) * n + np.asarray(pred), minlength=n * n).reshape(n, n)


def macro_f1(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """Mean F1 over classes present in the truth or the predictions (sklearn's default)."""
    cm = _confusion(pred, y, n_classes)
    tp, p, t = np.diag(cm), cm.sum(0), cm.sum(1)
    present = (p + t) > 0
    f1 = np.where(present, 2 * tp / np.maximum(p + t, 1), 0.0)
    return float(f1[present].mean())


def balanced_accuracy(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """Mean recall over the classes present in the truth."""
    cm = _confusion(pred, y, n_classes)
    t = cm.sum(1)
    return float((np.diag(cm)[t > 0] / t[t > 0]).mean())


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve for binary ``y``, by ranks (ties averaged)."""
    score, y = np.asarray(score, float), np.asarray(y).astype(bool)
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):                                   # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# -- probe -----------------------------------------------------------------------------
class LinearProbe:
    """Multinomial logistic regression on standardised features, class-balanced.

    Minibatch Adam over a feature matrix that stays float16 in RAM (an hour of
    labels at 1,536 dimensions is ~0.2 GB); every shuffle comes from a seeded
    generator, so a fit is a function of its inputs. Classes absent from the fit
    set get no weight and are never predicted.
    """

    def __init__(self, n_classes: int, weight_decay: float = 1e-3, *, epochs: int = 12,
                 batch: int = 2048, lr: float = 3e-3, seed: int = 0, device: str = "cpu") -> None:
        self.n_classes, self.wd, self.epochs = n_classes, float(weight_decay), epochs
        self.batch, self.lr, self.seed, self.device = batch, lr, seed, torch.device(device)

    def _z(self, x: np.ndarray, i: np.ndarray) -> torch.Tensor:
        z = torch.as_tensor(np.asarray(x[i], np.float32), device=self.device)
        return (z - self.mu) / self.sd

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearProbe":
        y = np.asarray(y, int)
        n, d = x.shape
        s1, s2 = np.zeros(d), np.zeros(d)
        for a in range(0, n, 8192):
            c = np.asarray(x[a:a + 8192], np.float64)
            s1 += c.sum(0)
            s2 += (c ** 2).sum(0)
        mu = s1 / n
        self.mu = torch.as_tensor(mu, dtype=torch.float32, device=self.device)
        self.sd = torch.as_tensor(np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6,
                                  dtype=torch.float32, device=self.device)
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        self.present = counts > 0
        w = np.where(self.present, n / (self.present.sum() * np.maximum(counts, 1)), 0.0)
        cw = torch.as_tensor(w, dtype=torch.float32, device=self.device)
        g = torch.Generator().manual_seed(self.seed)
        lin = torch.nn.Linear(d, self.n_classes).to(self.device)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(self.n_classes, d, generator=g) * 0.01)
            lin.bias.zero_()
        opt = torch.optim.Adam(lin.parameters(), lr=self.lr)
        yt = torch.as_tensor(y, device=self.device)
        absent = torch.as_tensor(~self.present, device=self.device)
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g).numpy()
            for a in range(0, n, self.batch):
                i = np.sort(perm[a:a + self.batch])
                logits = lin(self._z(x, i)).masked_fill(absent, -1e4)
                loss = torch.nn.functional.cross_entropy(logits, yt[i], weight=cw)
                loss = loss + self.wd * lin.weight.pow(2).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.lin = lin.eval()
        return self

    @torch.no_grad()
    def scores(self, x: np.ndarray) -> np.ndarray:
        """``(n, n_classes)`` logits; absent classes at -1e4."""
        absent = torch.as_tensor(~self.present, device=self.device)
        out = [self.lin(self._z(x, np.arange(a, min(a + 8192, len(x))))).masked_fill(absent, -1e4)
               .cpu().numpy() for a in range(0, len(x), 8192)]
        return np.concatenate(out) if out else np.zeros((0, self.n_classes), np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.scores(x).argmax(1)


def fit_select(xf, yf, xv, yv, n_classes: int, *, wds=(1e-5, 1e-3, 1e-1), **kw):
    """Fit one probe per weight decay; keep the best macro-F1 on the validation set."""
    best, scores = None, {}
    for wd in wds:
        p = LinearProbe(n_classes, wd, **kw).fit(xf, yf)
        scores[wd] = macro_f1(p.predict(xv), yv, n_classes)
        if best is None or scores[wd] > scores[best[0]]:
            best = (wd, p)
    return best[1], {"weight_decay": best[0], "validation_f1": scores}


# -- step 2: spectral descriptors -------------------------------------------------------
def window_descriptors(w: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """``(n, C, 9)``: per channel, 1/f exponent and offset, log variance, 6 log band powers."""
    from ..adapt.nuisance import compute

    return np.stack([compute(np.asarray(x, float), fs).to_matrix()[:, :-1] for x in w])


def region_means(desc: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """``(n, C, k)`` -> ``(n, k * G)``: descriptors averaged over each non-empty region."""
    groups = np.asarray(groups)
    return np.concatenate([desc[:, groups == g].mean(1) for g in np.unique(groups)], axis=1)


def top_pcs(d: np.ndarray, k: int) -> np.ndarray:
    """The ``k`` leading principal-component scores of standardised ``d`` (n, m)."""
    z = (d - d.mean(0)) / (d.std(0) + 1e-9)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    return u[:, :k] * s[:k]


def residualise(f: np.ndarray, d: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """``f`` minus the part a linear map of ``d`` explains; both standardised first.

    With ``groups`` (the listener of each window) the map is fitted on
    within-group variation only -- each group's mean removed from ``f`` and ``d``
    -- and then applied to the full ``d``. That is the test step 2 needs. Fitted on
    all windows at once, descriptors that merely *identify* the listener (64
    components from 32 listeners span the listener means) regress out every
    listener-mean difference in ``f``, whatever carries it. Fitted within
    listeners, the map is identified only by how features follow window-to-window
    spectral changes, so it removes across-listener differences only insofar as
    they follow the same map.
    """
    f = (f - f.mean(0)) / (f.std(0) + 1e-9)
    d = (d - d.mean(0)) / (d.std(0) + 1e-9)
    if groups is None:
        a = np.concatenate([np.ones((len(d), 1)), d], axis=1)
        beta, *_ = np.linalg.lstsq(a, f, rcond=None)
        return f - a @ beta
    groups = np.asarray(groups)
    fw, dw = f.copy(), d.copy()
    for g in np.unique(groups):
        m = groups == g
        fw[m] -= fw[m].mean(0)
        dw[m] -= dw[m].mean(0)
    beta, *_ = np.linalg.lstsq(dw, fw, rcond=None)
    return f - d @ beta


def cross_listener(f: np.ndarray, d: np.ndarray, groups: np.ndarray,
                   alphas=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)) -> dict:
    """How much of the listener differences in ``f`` a map from ``d`` predicts for unseen listeners.

    Leave-one-listener-out ridge regression from descriptors ``d`` to features
    ``f`` (both standardised): each listener's windows are predicted by a map
    learned on the other listeners only. The score is the cross-validated R^2 of
    the listener means -- 1 if every listener's mean features follow from their
    spectra through one shared map, ~0 if not. The ridge penalty is the one that
    scores best (the same search runs on the shuffled-descriptor control).

    Why not a regression on all windows: descriptors that merely identify the
    listener (64 components from 32 listeners span the listener means) would
    remove every listener difference, whatever carries it. Why not a map fitted
    within listeners and subtracted: where features follow spectra within a
    listener but listener means do not follow the same map, subtracting it
    *injects* the descriptors' own listener structure (a smoke run took a masked
    model from 5x to 32x). A prediction for an unseen listener can do neither.

    Returns ``r2_between``, ``alpha`` and ``residual`` (``f`` minus the prediction).
    """
    f = (f - f.mean(0)) / (f.std(0) + 1e-9)
    d = (d - d.mean(0)) / (d.std(0) + 1e-9)
    groups = np.asarray(groups)
    ug = np.unique(groups)
    preds = {a: np.zeros_like(f) for a in alphas}
    for g in ug:
        tr = groups != g
        dm, fm = d[tr].mean(0), f[tr].mean(0)
        u, sv, vt = np.linalg.svd(d[tr] - dm, full_matrices=False)
        uf = u.T @ (f[tr] - fm)
        for a in alphas:
            b = vt.T @ ((sv / (sv ** 2 + a))[:, None] * uf)
            preds[a][~tr] = (d[~tr] - dm) @ b + fm
    n = np.array([(groups == g).sum() for g in ug], float)[:, None]
    mu = np.stack([f[groups == g].mean(0) for g in ug])
    tot = float((n * (mu - f.mean(0)) ** 2).sum())
    best = None
    for a in alphas:
        mh = np.stack([preds[a][groups == g].mean(0) for g in ug])
        r2 = 1.0 - float((n * (mh - mu) ** 2).sum()) / max(tot, 1e-12)
        if best is None or r2 > best[1]:
            best = (a, r2)
    return {"alpha": best[0], "r2_between": best[1], "residual": f - preds[best[0]]}


def spread(z: np.ndarray) -> dict:
    """Collapse measures of token features ``(n, d)`` that need no calibration.

    ``varying``: the share of the features' energy that varies across tokens
    (0 for a constant output). ``participation``: the participation ratio of the
    covariance's eigenvalues, an effective number of dimensions (1 for a
    representation that varies along one direction).
    """
    z = np.asarray(z, np.float64)
    zc = z - z.mean(0)
    lam = np.clip(np.linalg.eigvalsh(zc.T @ zc / max(len(z) - 1, 1)), 0, None)
    return {"varying": float(lam.sum() / max((z ** 2).sum(1).mean(), 1e-12)),
            "participation": float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-30))}


def leace(x: np.ndarray, groups: np.ndarray, eps: float = 1e-6):
    """Fit LEACE on ``x`` (n, d) to erase ``groups``; returns ``erase(z) -> z'``.

    ``r(z) = z - W+ P W (z - mu)``, with ``W`` the whitening transform of ``x``'s
    covariance, ``W+`` its pseudo-inverse and ``P`` the orthogonal projection onto
    the column space of ``W`` times the cross-covariance of ``x`` with the one-hot
    groups. Guaranteed on the fitted data; on new groups (people never seen) it
    removes only the directions the fitted groups differed along.
    """
    x = np.asarray(x, np.float64)
    groups = np.asarray(groups)
    ug = np.unique(groups)
    zc = (groups[:, None] == ug[None]).astype(np.float64)
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / len(x)
    lam, v = np.linalg.eigh(cov)
    keep = lam > eps * lam.max()
    w = (v[:, keep] / np.sqrt(lam[keep])) @ v[:, keep].T            # Sigma^(-1/2)
    w_pinv = (v[:, keep] * np.sqrt(lam[keep])) @ v[:, keep].T       # Sigma^(1/2)
    sxz = xc.T @ (zc - zc.mean(0)) / len(x)                          # (d, k)
    u, sv, _ = np.linalg.svd(w @ sxz, full_matrices=False)
    u = u[:, sv > eps * max(sv.max(), 1e-30)]
    proj = w_pinv @ (u @ u.T) @ w

    def erase(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, np.float64)
        return z - (z - mu) @ proj.T

    return erase
