"""Brain decoders for the 50-word task, as specified in the lost version's pitch.

=================  ==========================================================
Linear (pooled)    64 ms channel means (306 x 8 = 2,448) -> PCA 128 fitted on
                   training data only -> ridge one-vs-rest, penalty by 3-fold
                   CV over 1..1e4
Linear, per person the same, one model per listener
MLP                PCA 256 -> 256 -> 256 -> 50, GELU; dropout 0.2 in, 0.5
                   hidden; AdamW 1e-3 / wd 1e-2, batch 256, label smoothing 0.1,
                   class-balanced loss, early stopping (patience 8)
CNN                raw 125 Hz windows (306 x 64) -> 1x1 spatial conv to 32 ->
                   two temporal convs (kernel 9, GELU, BatchNorm, dropout 0.3)
                   -> adaptive pool to 8 -> linear to 50; AdamW 2e-3 / wd 1e-3,
                   batch 128, label smoothing 0.1, class-balanced, patience 6
=================  ==========================================================

Stronger decoders, the pitch's next step ("CNN tuned on fit data only,
EEGNet"), built here:

=================  ==========================================================
CNN, configurable  the CNN with spatial width, kernel and dropout as arguments;
                   ``run_decoder_sweep.py`` chooses them on chapter 11 only
Subject CNN        the CNN behind a per-listener 1x1 spatial layer (one 306 ->
                   width map per person, everything after it shared)
EEGNet             Lawhern et al. 2018 adapted to 306 sensors: temporal conv
                   (F1 = 16, 0.26 s) -> depthwise spatial conv (D = 4) ->
                   separable conv (F2 = 64) -> two average pools -> linear
=================  ==========================================================

One deliberate tightening: every internal split -- the ridge CV folds and the
early-stopping hold-out -- is made by **word occurrence**, not by trial. All
listeners heard the same chapter, so a trial-level split puts the same spoken
word on both sides and lets tuning reward memorising the audio.

Every decoder returns ``(n, 50)`` scores, higher is better, for
:func:`neurocast.decode.metrics.balanced_accuracy_at_k`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "patch_features",
    "LinearPooled",
    "LinearPerPerson",
    "MLPDecoder",
    "CNNDecoder",
    "SubjectCNNDecoder",
    "EEGNetDecoder",
    "occurrence_split",
]


def patch_features(windows: np.ndarray, patch: int = 8) -> np.ndarray:
    """``(n, C, T)`` decimated windows -> ``(n, C * T/patch)`` patch means, float32.

    At 125 Hz, 8 samples are the 16-sample (64 ms) patches of the 250 Hz signal.
    """
    n, c, t = windows.shape
    return windows.astype(np.float32).reshape(n, c, t // patch, patch).mean(-1).reshape(n, -1)


def occurrence_split(occ: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Boolean mask selecting about ``frac`` of the *occurrences* (all their trials)."""
    u = np.unique(occ)
    pick = np.random.default_rng(seed).permutation(u)[: max(1, int(round(frac * len(u))))]
    return np.isin(occ, pick)


def _class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)
    return w


class LinearPooled:
    def __init__(self, n_components: int = 128, alphas=tuple(np.logspace(0, 4, 9)),
                 seed: int = 0) -> None:
        self.n_components, self.alphas, self.seed = n_components, alphas, seed

    def fit(self, x: np.ndarray, y: np.ndarray, occ: np.ndarray) -> "LinearPooled":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.model_selection import GroupKFold

        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        z = (x - self.mu) / self.sd
        self.pca = PCA(min(self.n_components, z.shape[0] - 1, z.shape[1]),
                       random_state=self.seed).fit(z)
        f = self.pca.transform(z)
        folds = list(GroupKFold(n_splits=3).split(f, y, groups=occ))
        self.clf = RidgeClassifierCV(alphas=self.alphas, cv=folds,
                                     class_weight="balanced").fit(f, y)
        self.classes_ = self.clf.classes_
        return self

    def scores(self, x: np.ndarray, n_classes: int) -> np.ndarray:
        s = self.clf.decision_function(self.pca.transform((x - self.mu) / self.sd))
        out = np.full((len(x), n_classes), -1e6)
        out[:, self.classes_] = s
        return out


class LinearPerPerson:
    def __init__(self, **kw) -> None:
        self.kw = kw
        self.models: dict = {}

    def fit(self, x, y, occ, subject) -> "LinearPerPerson":
        for s in np.unique(subject):
            m = subject == s
            self.models[s] = LinearPooled(**self.kw).fit(x[m], y[m], occ[m])
        return self

    def scores(self, x, n_classes, subject) -> np.ndarray:
        out = np.full((len(x), n_classes), -1e6)
        for s, model in self.models.items():
            m = subject == s
            if m.any():
                out[m] = model.scores(x[m], n_classes)
        return out


def _inputs(item, device) -> tuple:
    """A batch from a view -- an array, or a tuple (windows, subject ids) -- as tensors."""
    items = item if isinstance(item, tuple) else (item,)
    return tuple(torch.as_tensor(a, dtype=torch.long if np.issubdtype(np.asarray(a).dtype, np.integer)
                                 else torch.float32, device=device) for a in items)


def _train_torch(net, xtr, ytr, xva, yva, n_classes, *, lr, wd, batch, patience,
                 max_epochs=100, seed=0, label_smoothing=0.1, verbose=False, device="cpu"):
    """Class-balanced, label-smoothed AdamW with early stopping on held-out BAcc@10."""
    from .metrics import balanced_accuracy_at_k

    torch.manual_seed(seed)
    net.to(device)
    w = torch.as_tensor(_class_weights(ytr, n_classes), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    best, best_state, bad = -1.0, None, 0
    for epoch in range(max_epochs):
        net.train()
        for idx in torch.randperm(len(ytr), generator=g).split(batch):
            loss = loss_fn(net(*_inputs(xtr[idx.numpy()], device)), ytr_t[idx.to(device)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        score = balanced_accuracy_at_k(_predict(net, xva, device=device), yva, 10)
        if verbose:
            print(f"    epoch {epoch:3d}  val BAcc@10 {score:.3f}", flush=True)
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best, epoch + 1


@torch.no_grad()
def _predict(net, x, batch=None, device="cpu") -> np.ndarray:
    # 1,024 at a time on CPU, as every reported CPU result was computed. On a 4 GB
    # GPU that is EEGNet's first layer at 2.4 GB for 1.024 s windows: out of memory
    # (or, under the Windows driver, a slow spill into system RAM).
    batch = batch or (1024 if str(device) == "cpu" else 128)
    net.eval()
    return torch.cat([net(*_inputs(x[i:i + batch], device)).float().cpu()
                      for i in range(0, len(x), batch)]).numpy()


class MLPDecoder:
    def __init__(self, n_components: int = 256, hidden: int = 256, seed: int = 0) -> None:
        self.n_components, self.hidden, self.seed = n_components, hidden, seed

    def fit(self, x, y, occ, n_classes, verbose=False) -> "MLPDecoder":
        from sklearn.decomposition import PCA

        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        self.pca = PCA(self.n_components, random_state=self.seed).fit((x - self.mu) / self.sd)
        f = self._f(x)
        self.fs = f.std(0) + 1e-6
        f = f / self.fs
        va = occurrence_split(occ, 0.1, self.seed)
        self.net = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(f.shape[1], self.hidden), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(self.hidden, n_classes),
        )
        self.net, self.val_bacc, self.epochs = _train_torch(
            self.net, f[~va], y[~va], f[va], y[va], n_classes, lr=1e-3, wd=1e-2,
            batch=256, patience=8, seed=self.seed, verbose=verbose)
        return self

    def _f(self, x):
        return self.pca.transform((x - self.mu) / self.sd).astype(np.float32)

    def scores(self, x, n_classes=None) -> np.ndarray:
        return _predict(self.net, self._f(x) / self.fs)


class _CNN(nn.Module):
    def __init__(self, n_ch: int, n_classes: int, spatial: int = 32, kernel: int = 9,
                 dropout: float = 0.3) -> None:
        super().__init__()

        def block(c):
            return nn.Sequential(nn.Conv1d(c, c, kernel, padding=kernel // 2),
                                 nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(dropout))

        self.spatial = nn.Conv1d(n_ch, spatial, 1)
        self.temporal = nn.Sequential(block(spatial), block(spatial))
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.out = nn.Linear(spatial * 8, n_classes)

    def forward(self, x):
        h = self.pool(self.temporal(self.spatial(x)))
        return self.out(h.flatten(1))


class CNNDecoder:
    """Works on the raw decimated windows, ``(n, C, 64)`` at 125 Hz.

    The defaults are the pitch's CNN; ``spatial``, ``kernel``, ``dropout`` and
    ``lr`` are what ``run_decoder_sweep.py`` tunes on the fit chapter.
    """

    def __init__(self, seed: int = 0, *, spatial: int = 32, kernel: int = 9, dropout: float = 0.3,
                 lr: float = 2e-3, device: str = "cpu") -> None:
        self.seed, self.device = seed, device
        self.arch = dict(spatial=spatial, kernel=kernel, dropout=dropout)
        self.lr = lr

    def _net(self, n_ch, n_classes):
        return _CNN(n_ch, n_classes, **self.arch)

    def _view(self, w, subjects=None):
        return _ScaledView(w, self.scale)

    def fit(self, w, y, occ, n_classes, verbose=False, subjects=None) -> "CNNDecoder":
        self.scale = float(np.std(w[:2000].astype(np.float32))) + 1e-6
        va = occurrence_split(occ, 0.1, self.seed)
        x = self._view(w, subjects)
        torch.manual_seed(self.seed)
        self.net = self._net(w.shape[1], n_classes)
        self.net, self.val_bacc, self.epochs = _train_torch(
            self.net, x.subset(~va), y[~va], x.subset(va), y[va], n_classes,
            lr=self.lr, wd=1e-3, batch=128, patience=6, seed=self.seed, verbose=verbose,
            device=self.device)
        return self

    def scores(self, w, n_classes=None, subjects=None) -> np.ndarray:
        return _predict(self.net, self._view(w, subjects), device=self.device)


class _SubjectCNN(_CNN):
    """A 306 -> width spatial map per listener in place of the shared one."""

    def __init__(self, n_ch, n_classes, n_subjects, **kw) -> None:
        super().__init__(n_ch, n_classes, **kw)
        width = self.spatial.out_channels
        self.subject_w = nn.Parameter(torch.randn(n_subjects, width, n_ch) / np.sqrt(n_ch))
        self.subject_b = nn.Parameter(torch.zeros(n_subjects, width, 1))
        del self.spatial

    def forward(self, x, sid):
        h = torch.einsum("boc,bct->bot", self.subject_w[sid], x) + self.subject_b[sid]
        return self.out(self.pool(self.temporal(h)).flatten(1))


class SubjectCNNDecoder(CNNDecoder):
    """The CNN behind a per-listener spatial layer; needs ``subjects`` in fit and scores."""

    def fit(self, w, y, occ, n_classes, verbose=False, subjects=None) -> "SubjectCNNDecoder":
        if subjects is None:
            raise ValueError("SubjectCNNDecoder needs the listener of every trial")
        self.ids = {s: i for i, s in enumerate(sorted(set(subjects)))}
        return super().fit(w, y, occ, n_classes, verbose, subjects)

    def _net(self, n_ch, n_classes):
        return _SubjectCNN(n_ch, n_classes, len(self.ids), **self.arch)

    def _view(self, w, subjects=None):
        sid = np.array([self.ids[s] for s in subjects], dtype=np.int64)
        return _ScaledView(w, self.scale, sid)


class _EEGNet(nn.Module):
    def __init__(self, n_ch, n_times, n_classes, f1=16, d=4, f2=64, k1=33, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, f1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_ch, 1), groups=f1, bias=False), nn.BatchNorm2d(f1 * d),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(f1 * d, f1 * d, (1, 17), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, 1, bias=False), nn.BatchNorm2d(f2),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.out = nn.Linear(f2 * (n_times // 32), n_classes)

    def forward(self, x):
        return self.out(self.net(x[:, None]).flatten(1))


class EEGNetDecoder(CNNDecoder):
    """EEGNet (Lawhern et al. 2018) on the raw 125 Hz windows."""

    def __init__(self, seed: int = 0, *, lr: float = 1e-3, dropout: float = 0.5,
                 device: str = "cpu") -> None:
        super().__init__(seed, lr=lr, device=device)
        self.arch = dict(dropout=dropout)

    def fit(self, w, y, occ, n_classes, verbose=False, subjects=None) -> "EEGNetDecoder":
        self.n_times = w.shape[2]
        return super().fit(w, y, occ, n_classes, verbose, subjects)

    def _net(self, n_ch, n_classes):
        return _EEGNet(n_ch, self.n_times, n_classes, **self.arch)


class _ScaledView:
    """float16 windows, scaled to unit-ish variance lazily per batch (saves RAM).

    With ``sid`` (one integer per trial) a batch is ``(windows, sid)``.
    """

    def __init__(self, w: np.ndarray, scale: float, sid: np.ndarray | None = None) -> None:
        self.w, self.scale, self.sid = w, scale, sid

    def subset(self, mask: np.ndarray) -> "_ScaledView":
        return _ScaledView(self.w[mask], self.scale, None if self.sid is None else self.sid[mask])

    def __len__(self) -> int:
        return len(self.w)

    def __getitem__(self, idx):
        x = self.w[idx].astype(np.float32) / self.scale
        return x if self.sid is None else (x, self.sid[idx])
