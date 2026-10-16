"""Pre-training on real MEG: the forecasting-versus-masked pilot, rebuilt.

The lost version's pilot (pitch, "ML configuration reference"): 6m rung (24.9M
parameters), 4-s windows (64 patches), batch 8, 4,000 steps, the deep listener's
data, comparing likelihood forecasting (``ar_lik``) with masked reconstruction --
and finding the masked arm had *collapsed* on real data until a visible-position
term was added.

Configuration, as quoted:

* targets: whitened per-region PCA of each patch (:class:`QuadrantBasis`), rank 32;
* masked: hidden 1-s blocks covering 40% of the window, bidirectional trunk,
  optional visible-position term (``visible_weight``; the fix used 1.0);
* AdamW lr 1e-3, betas (0.9, 0.95), weight decay 0.01, 100 warm-up steps, cosine
  decay to 10%, gradient clipping at 1.0, bf16/fp16 mixed precision on a GPU;
* a held-out tail of every session for loss tracking -- or, for runs on many
  hours, whole held-out sessions (``train_stop = 0``);
* the best held-out checkpoint is kept beside the latest (``<name>.best.pt``):
  the pilot's forecasting arm was best at step 2,000 of 4,000 and the trainer
  then kept only the overfit final weights;
* **bit-exact resume**: every random draw is a function of ``(seed, step)``, and a
  checkpoint carries model, head, optimiser and scheduler state, so a resumed run
  continues the identical sequence of batches and updates.

The two latent-prediction arms of the bake-off (README section 2.6) train here too:

* ``jepa``: masked latent prediction. A bidirectional student sees the window with
  the same 1-s blocks hidden as the masked arm and predicts, at the hidden
  positions, an EMA teacher's layer-normed representation of the full window;
* ``ar_latent``: the same, causal: the student's output at position ``i`` predicts
  the causal teacher's representation at ``i + 1``.

Both add the sketched isotropic-Gaussian regulariser (``objectives.arms._sigreg``)
and move the teacher by EMA after every step (``objectives.arms.EMATeacher``). Their
loss is measured against a moving teacher, so a lower held-out value later in
training does not mean a better model: their ``.best.pt`` is the latest evaluation,
and the held-out log also carries the collapse monitor's statistics.

``condition="spectra"`` (forecasting only; README section 2.10) gives the likelihood
head each recording's spectral profile: per channel, the 1/f exponent and offset,
log variance and six log band powers (``adapt/nuisance.py``), averaged over 16
windows of the recording's training span (a held-out session uses its own span),
standardised over the training recordings, through a linear map that starts at
zero. Only the head sees it, so the trunk need not carry each person's spectral
calibration -- the conditioning ContentVec gives its predictor so the
representation can drop the speaker. Off by default; it changes nothing else.

``calib_guard`` rejects target-calibration windows whose largest absolute value
exceeds it (session-normalised units; ordinary 4-s windows peak at ~10). A ~1,000x
artifact in Sherlock1 session 3 leaks ~14 s backwards through the release's
zero-phase filters, and one calibration window landing in it corrupted a whole
run's targets (README section 2.5). Off by default; when it rejects nothing the
draws, and so the run, are exactly those of an unguarded run.

Empty quadrants (LibriBrain's H5 has no EOG/ECG, so the peripheral group is
empty) are excluded from every likelihood and reconstruction -- see
:meth:`QuadrantBasis.component_mask`.

Tens of hours do not fit in RAM (the deep listener's ~64 h of 306 channels at
250 Hz is 35 GB in float16). :func:`cache_recordings` writes each session once, time-major, to the
data drive and :func:`open_cached` memory-maps it, so a 4-s window is one
contiguous read; ``prefetch`` draws upcoming batches on background threads.
Batches stay a pure function of ``(seed, step)``, so resume is still exact.
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.paths import replace_file
from ..models.heads import GaussianMixtureHead
from ..models.neurocast import NeuroCast
from ..objectives.arms import ar_shift, block_mask
from .targets import QuadrantBasis

__all__ = ["PretrainConfig", "Recording", "load_recordings", "cache_recordings", "open_cached",
           "Pretrainer", "LATENT_ARMS"]

#: Arms trained against an EMA teacher's representation rather than the signal.
LATENT_ARMS = ("jepa", "ar_latent")

#: Serialises reads of memory-mapped sessions across prefetch threads (see ``Recording.window``).
_DISK_LOCK = threading.Lock()
#: Open session files, by path (see ``_read_retrying``).
_HANDLES: dict = {}


def _read_retrying(path, offset: int, size: int, attempts: int = 5, delay: float = 5.0) -> bytes:
    """``size`` bytes at ``offset``, retrying transient errors of a USB disk.

    The data drive has dropped reads (``OSError: [Errno 22] Invalid argument``,
    Windows disk warning 153) and each one stopped an hours-long queue. Retrying
    returns the same bytes, so batches stay a pure function of ``(seed, step)``.
    """
    for i in range(attempts):
        try:
            # One handle per file, kept open: on Windows every open() passes the
            # antivirus on-access hook, and a window per open cost ~15 ms. Callers
            # hold _DISK_LOCK, so seek-then-read on a shared handle is safe.
            f = _HANDLES.get(path)
            if f is None:
                f = _HANDLES[path] = open(path, "rb")
            f.seek(offset)
            buf = f.read(size)
            if len(buf) == size:
                return buf
            raise OSError(f"short read: {len(buf)} of {size} bytes from {path}")
        except OSError:
            stale = _HANDLES.pop(path, None)
            if stale is not None:
                try:
                    stale.close()
                except OSError:
                    pass
            if i == attempts - 1:
                raise
            time.sleep(delay)
    raise AssertionError("unreachable")


@dataclass
class PretrainConfig:
    arm: str = "ar_lik"                 # "ar_lik", "masked", "jepa" or "ar_latent"
    rung: str = "6m"
    n_patch: int = 64                   # 4.1 s windows
    batch: int = 8
    steps: int = 4000
    lr: float = 1e-3
    warmup: int = 100
    final_lr_frac: float = 0.10
    weight_decay: float = 0.01
    clip: float = 1.0
    rank: int = 32
    mask_frac: float = 0.40
    mask_block_s: float = 1.0
    visible_weight: float = 0.0
    heldout_frac: float = 0.16          # last 16% of each session
    eval_every: int = 200
    eval_batches: int = 8
    seed: int = 0
    precision: str = "auto"             # "auto" | "fp32" | "bf16" | "fp16"
    prefetch: int = 0                   # batches drawn ahead on background threads
    ema_base: float = 0.996             # latent arms: teacher momentum, cosine to ema_final
    ema_final: float = 0.9999
    sigreg: float = 0.05                # latent arms: isotropic-Gaussian regulariser weight
    calib_guard: float = 0.0            # reject calibration windows above this |x| (0: off)
    condition: str = ""                 # "spectra": the head sees each recording's spectra


@dataclass
class Recording:
    """One session in canonical, session-normalised units, float16."""

    key: str
    data: np.ndarray                    # (C, T) in RAM, or (T, C) memory-mapped if time_major
    train_stop: int                     # samples [0, train_stop) train; rest held out
    story: tuple[int, int] = (0, 0)     # first/last word sample, the "story" span
    time_major: bool = False

    @property
    def n_samples(self) -> int:
        return self.data.shape[0] if self.time_major else self.data.shape[1]

    def window(self, start: int, n: int) -> np.ndarray:
        """``(C, n)`` samples from ``start``, whatever the layout.

        A memory-mapped session is read with one explicit read per window. Through
        the map, a 4-s window became ~10 separate 64 KB page faults; on a USB
        spinning disk with 61 h that do not fit in the page cache, that starved
        the GPU (0.50 s/step against 0.21). Same bytes either way.
        """
        if self.time_major and isinstance(self.data, np.memmap):
            c = self.data.shape[1]
            item = self.data.dtype.itemsize
            start = max(0, min(start, self.data.shape[0]))
            n = min(n, self.data.shape[0] - start)
            # One reader at a time: Windows splits a 627 KB read into 64 KB requests,
            # and eight prefetch threads interleaving them made every request a seek
            # (14 MB/s, 0.4 s/step). Serialised, each window is one sequential pass.
            with _DISK_LOCK:
                buf = _read_retrying(self.data.filename, self.data.offset + start * c * item,
                                     n * c * item)
            return np.frombuffer(buf, dtype=self.data.dtype).reshape(n, c).T
        if self.time_major:
            return np.asarray(self.data[start:start + n]).T
        return self.data[:, start:start + n]


def load_recordings(runs, *, heldout_frac: float, regime: str = "session-scalar") -> list[Recording]:
    """Load runs, keep the story span, and carve a held-out tail off each."""
    from ..data.canonical import NormalizationRegime, fit_normalizer, to_canonical
    from ..data.libribrain import read_meg, read_words

    out = []
    for r in runs:
        x, info = read_meg(r.h5)
        x, _ = to_canonical(x, info["channel_types"])
        tf = fit_normalizer(x[:, ::4], NormalizationRegime(regime), ch_types=info["channel_types"])
        x = tf.apply(x).astype(np.float16)
        w = read_words(r.events)
        a = int(w["onset"].iloc[0] * info["sfreq"])
        b = min(int((w["onset"].iloc[-1] + w["duration"].iloc[-1]) * info["sfreq"]), x.shape[1])
        x = x[:, a:b]
        out.append(Recording("_".join(r.key), x, int(x.shape[1] * (1 - heldout_frac)), (a, b)))
    return out


def _story_span(run, x: np.ndarray, sfreq: float) -> tuple[int, int]:
    from ..data.libribrain import read_words

    w = read_words(run.events)
    a = int(w["onset"].iloc[0] * sfreq)
    b = min(int((w["onset"].iloc[-1] + w["duration"].iloc[-1]) * sfreq), x.shape[1])
    return a, b


def cache_recordings(runs, cache_dir: Path, *, regime: str = "session-scalar",
                     verbose: bool = True) -> list[str]:
    """Write each run's story span once, time-major float16, for :func:`open_cached`.

    Same pipeline as :func:`load_recordings` (canonical units, ``regime``
    normalisation fitted on the whole recording, story span). Files already
    present are kept; a file appears only when complete. Returns the keys.
    """
    from ..data.canonical import NormalizationRegime, fit_normalizer, to_canonical
    from ..data.libribrain import read_meg

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = []
    for i, r in enumerate(runs):
        key = "_".join(r.key)
        keys.append(key)
        f, meta = cache_dir / f"{key}.npy", cache_dir / f"{key}.json"
        if f.exists() and meta.exists():
            continue
        x, info = read_meg(r.h5)
        x, _ = to_canonical(x, info["channel_types"])
        tf = fit_normalizer(x[:, ::4], NormalizationRegime(regime), ch_types=info["channel_types"])
        x = tf.apply(x).astype(np.float16)
        a, b = _story_span(r, x, info["sfreq"])
        tmp = cache_dir / f"{key}.tmp.npy"
        np.save(tmp, np.ascontiguousarray(x[:, a:b].T))
        replace_file(tmp, f)
        meta.write_text(json.dumps({"story": [a, b], "sfreq": info["sfreq"], "regime": regime,
                                    "channel_names": list(info["channel_names"])}))
        if verbose:
            print(f"  cached {key}: {(b - a) / info['sfreq'] / 60:.1f} min ({i + 1}/{len(runs)})",
                  flush=True)
    return keys


def open_cached(cache_dir: Path, keys, *, heldout=()) -> list[Recording]:
    """Memory-map cached sessions; sessions in ``heldout`` are held out whole.

    Every session must carry the same channels in the same order -- one montage.
    """
    cache_dir = Path(cache_dir)
    held = set(heldout)
    out, names = [], None
    for key in keys:
        meta = json.loads((cache_dir / f"{key}.json").read_text())
        if names is None:
            names = meta["channel_names"]
        elif meta["channel_names"] != names:
            raise ValueError(f"{key}: channels differ from {keys[0]}; one montage per run")
        data = np.load(cache_dir / f"{key}.npy", mmap_mode="r")
        out.append(Recording(key, data, 0 if key in held else data.shape[0],
                             tuple(meta["story"]), time_major=True))
    return out


def _lr_lambda(cfg: PretrainConfig):
    def f(step: int) -> float:
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        p = min((step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1), 1.0)
        return cfg.final_lr_frac + (1 - cfg.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))
    return f


class Pretrainer:
    """Trains one arm; ``fit`` resumes from ``ckpt`` if it exists."""

    def __init__(self, cfg: PretrainConfig, montage, recordings: list[Recording],
                 device: str | torch.device | None = None) -> None:
        self.cfg = cfg
        self.recs = recordings
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(cfg.seed)
        causal = cfg.arm not in ("masked", "jepa")
        self.model = NeuroCast(montage, rung=cfg.rung, window=256, causal=causal,
                               device=self.device).to(self.device)
        self.patch = self.model.patch_samples
        self.n_times = cfg.n_patch * self.patch
        self.G = self.model.n_groups
        # Recordings that can supply a window to each split. With a held-out tail on
        # every session (the pilot) both pools are all recordings, and the draws are
        # exactly those of the version that picked from ``recordings`` directly.
        self._pool = {True: [r for r in recordings if r.train_stop >= self.n_times],
                      False: [r for r in recordings if r.n_samples - r.train_stop >= self.n_times]}
        if not self._pool[True]:
            raise ValueError("no recording has a training span of one window")

        # Targets: per-region PCA fitted on training windows only.
        draws = self.calibration_draws()
        self.calib_rejected = sum(1 for d in draws if not d[2])
        calib = np.stack([r.window(st, self.n_times) for r, st, keep in draws if keep])
        self.basis = QuadrantBasis(rank=cfg.rank, patch=self.patch, n_groups=self.G)
        self.basis.fit(calib.astype(np.float32), montage.quadrants)
        self.dim_mask = self.basis.position_mask(cfg.n_patch).to(self.device)   # (N, rank)
        if cfg.arm == "ar_lik":
            self.head = GaussianMixtureHead(self.model.d_model, cfg.rank, n_components=4).to(self.device)
            t = self.basis.encode(calib).reshape(-1, cfg.rank)
            m = self.basis.position_mask(cfg.n_patch).repeat(len(calib), 1).float()
            cnt = m.sum(0).clamp_min(1.0)
            mu = (t * m).sum(0) / cnt
            sd = (((t - mu) ** 2 * m).sum(0) / cnt).sqrt()
            self.head.set_background(mu.to(self.device), sd.clamp_min(1e-3).to(self.device))
        elif cfg.arm == "masked":
            d = self.model.d_model
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(d), torch.nn.Linear(d, 2 * d), torch.nn.GELU(),
                torch.nn.Linear(2 * d, cfg.rank)).to(self.device)
        elif cfg.arm in LATENT_ARMS:
            from ..models.heads import LatentPredictionHead
            from ..objectives.arms import EMATeacher

            d = self.model.d_model
            self.head = LatentPredictionHead(d, d).to(self.device)
            self.teacher = EMATeacher(self.model, cfg.ema_base, cfg.ema_final).to(self.device)
        else:
            raise ValueError(f"arm must be 'ar_lik', 'masked', 'jepa' or 'ar_latent', got {cfg.arm!r}")
        # Positions with a real target: the empty peripheral group has none.
        self.valid = self.dim_mask.any(-1)                                      # (N,)

        params = list(self.model.parameters()) + list(self.head.parameters())
        self.cond, self._keys = None, None
        if cfg.condition:
            if cfg.condition != "spectra" or cfg.arm != "ar_lik":
                raise ValueError("condition='spectra' is for the forecasting arm (ar_lik) only")
            self.cond, self.cond_stats = self._recording_spectra()
            dim = next(iter(self.cond.values())).numel()
            self.cond_proj = torch.nn.Linear(dim, self.model.d_model).to(self.device)
            with torch.no_grad():                      # starts as the plain forecaster
                self.cond_proj.weight.zero_()
                self.cond_proj.bias.zero_()
            params += list(self.cond_proj.parameters())
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95),
                                     weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, _lr_lambda(cfg))
        self.step = 0
        self.log: list[dict] = []
        self.best = float("inf")
        self.amp_dtype = self._precision()

    # -- data -------------------------------------------------------------------------
    def _draw(self, rng: np.random.Generator, train: bool) -> tuple[Recording, int]:
        pool = self._pool[train]
        r = pool[int(rng.integers(len(pool)))]
        lo, hi = (0, r.train_stop) if train else (r.train_stop, r.n_samples)
        return r, int(rng.integers(lo, max(hi - self.n_times, lo + 1)))

    def _window(self, rng: np.random.Generator, train: bool) -> np.ndarray:
        r, start = self._draw(rng, train)
        return r.window(start, self.n_times)

    def calibration_draws(self, n: int = 64) -> list[tuple[Recording, int, bool]]:
        """The target-calibration windows as ``(recording, start, kept)``, in draw order.

        One generator draws until ``n`` windows are kept: 64 distinct windows (a
        first version re-seeded per draw and calibrated on 64 copies of one).
        With ``calib_guard`` a window whose largest ``|x|`` exceeds it is drawn,
        reported and skipped; with the guard off every draw is kept.
        """
        rng = np.random.default_rng(self.cfg.seed + 7)
        out, kept = [], 0
        while kept < n:
            r, start = self._draw(rng, train=True)
            ok = True
            if self.cfg.calib_guard > 0:
                peak = float(np.abs(r.window(start, self.n_times).astype(np.float32)).max())
                ok = peak <= self.cfg.calib_guard
            out.append((r, start, ok))
            kept += ok
            if len(out) > 20 * n:
                raise ValueError(f"calib_guard {self.cfg.calib_guard}: {len(out) - kept} of "
                                 f"{len(out)} calibration windows rejected; is the data normalised?")
        return out

    def _recording_spectra(self, n_windows: int = 16):
        """Each recording's spectral profile, standardised over the training recordings."""
        from ..adapt.nuisance import compute

        raw = {}
        for r in self.recs:
            hi = r.train_stop if r.train_stop >= self.n_times else r.n_samples
            starts = np.linspace(0, max(hi - self.n_times, 0), n_windows).astype(int)
            d = [compute(r.window(int(st), self.n_times).astype(np.float64), 250.0).to_matrix()[:, :-1]
                 for st in starts]
            raw[r.key] = np.mean(d, axis=0).ravel()
        train = np.stack([raw[r.key] for r in self._pool[True]])
        mu, sd = train.mean(0), train.std(0) + 1e-6
        cond = {k: torch.as_tensor((v - mu) / sd, dtype=torch.float32, device=self.device)
                for k, v in raw.items()}
        return cond, {"mean": mu, "sd": sd}

    def _batch_keys(self, step: int, train: bool) -> list[str]:
        """The recordings behind batch ``step``: the same draws as ``_batch_np``, no reads."""
        rng = np.random.default_rng([self.cfg.seed, step, int(train)])
        return [self._draw(rng, train)[0].key for _ in range(self.cfg.batch)]

    def _batch_np(self, step: int, train: bool) -> np.ndarray:
        rng = np.random.default_rng([self.cfg.seed, step, int(train)])
        return np.stack([self._window(rng, train) for _ in range(self.cfg.batch)]).astype(np.float32)

    def batch(self, step: int, train: bool = True) -> torch.Tensor:
        """Batch ``step`` is a pure function of (seed, step, split): resume is exact."""
        return torch.as_tensor(self._batch_np(step, train), device=self.device)

    # -- objective --------------------------------------------------------------------
    def _precision(self):
        p = self.cfg.precision
        if self.device.type != "cuda" or p == "fp32":
            return None
        if p == "auto":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return {"bf16": torch.bfloat16, "fp16": torch.float16}[p]

    def _hidden(self, batch: int, n_pos: int, step: int) -> torch.Tensor:
        """The masked arms' 1-s blocks for ``step``: a pure function of (seed, step)."""
        g = torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + step)
        blk = max(1, int(round(self.cfg.mask_block_s * 250.0 / self.patch)) * self.G)
        return block_mask((batch, n_pos), block_len=blk, target_frac=self.cfg.mask_frac,
                          generator=g, device=self.device)

    def latent_loss(self, x: torch.Tensor, step: int) -> tuple[torch.Tensor, dict]:
        """``jepa`` / ``ar_latent``: predict the EMA teacher's layer-normed representation."""
        from ..objectives.arms import _sigreg

        cfg = self.cfg
        # The regulariser draws random projections from the global generator; seeding
        # it here keeps every draw a function of (seed, step), so resume stays exact.
        torch.manual_seed(cfg.seed * 1_000_003 + step + 17)
        with torch.no_grad():
            t = self.teacher(x).float()
            t = F.layer_norm(t, t.shape[-1:])
        if cfg.arm == "jepa":
            hidden = self._hidden(x.shape[0], t.shape[1], step)
            h = self.model(x, mask=hidden).float()
            sel = hidden & self.valid[None]
            pred = self.head(h[sel], t[sel])
            used = h[:, self.valid]
        else:
            h, tt = ar_shift(self.model(x).float(), t)
            v = self.valid[1:]
            pred = self.head(h[:, v], tt[:, v])
            used = h[:, v]
        reg = _sigreg(used)
        return pred + cfg.sigreg * reg, {"latent": float(pred.detach()), "sigreg": float(reg.detach())}

    def loss(self, x: torch.Tensor, step: int) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        if cfg.arm in LATENT_ARMS:
            return self.latent_loss(x, step)
        tgt = self.basis.encode(x.cpu()).to(self.device)                        # (B, N, r)
        if cfg.arm == "ar_lik":
            h, t = ar_shift(self.model(x), tgt)
            if self.cond is not None:
                if self._keys is None or len(self._keys) != x.shape[0]:
                    raise RuntimeError("a conditioned loss needs the batch's recordings in _keys")
                c = self.cond_proj(torch.stack([self.cond[k] for k in self._keys]))
                h = h + c[:, None, :].to(h.dtype)
            m = self.dim_mask[1:].expand_as(t)
            lp = self.head.log_prob(h.float(), t, m)
            nll = -lp.sum() / m.sum()
            return nll, {"nll": float(nll.detach())}
        hidden = self._hidden(x.shape[0], tgt.shape[1], step)
        pred = self.head(self.model(x, mask=hidden)).float()
        real = self.dim_mask.expand_as(tgt)
        err = ((pred - tgt) ** 2) * real
        h_err = err[hidden].sum() / real[hidden].sum().clamp_min(1)
        out = {"hidden_mse": float(h_err.detach())}
        total = h_err
        if cfg.visible_weight > 0:
            v_err = err[~hidden].sum() / real[~hidden].sum().clamp_min(1)
            out["visible_mse"] = float(v_err.detach())
            total = total + cfg.visible_weight * v_err
        return total, out

    # -- training ---------------------------------------------------------------------
    def state(self) -> dict:
        return {"cfg": asdict(self.cfg), "step": self.step, "model": self.model.state_dict(),
                "head": self.head.state_dict(), "opt": self.opt.state_dict(),
                **({"cond_proj": self.cond_proj.state_dict(), "cond_stats": self.cond_stats}
                   if self.cond is not None else {}),
                "sched": self.sched.state_dict(), "log": self.log,
                "basis": {k: getattr(self.basis, k) for k in
                          ("rank", "patch", "n_groups", "means", "comps", "scales", "channels", "evr")},
                **({"teacher": self.teacher.teacher.state_dict()} if self.cfg.arm in LATENT_ARMS else {})}

    def save(self, path: Path, *, weights_only: bool = False) -> None:
        """Atomic checkpoint; ``weights_only`` drops optimiser and scheduler state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state()
        if weights_only:
            state.pop("opt")
            state.pop("sched")
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        replace_file(tmp, path)                         # never leave a half-written checkpoint

    @staticmethod
    def best_path(path: Path) -> Path:
        path = Path(path)
        return path.with_name(path.stem + ".best.pt")

    def load(self, path: Path) -> None:
        """Resume from a full checkpoint, or load weights from a ``.best.pt`` one.

        A weights-only checkpoint of a latent arm may lack the EMA teacher, which
        only resuming needs (finished runs are slimmed to save disk); a resumable
        one (with optimiser state) must carry it.
        """
        s = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(s["model"])
        self.head.load_state_dict(s["head"])
        if self.cond is not None:
            self.cond_proj.load_state_dict(s["cond_proj"])
        if self.cfg.arm in LATENT_ARMS and ("teacher" in s or "opt" in s):
            self.teacher.teacher.load_state_dict(s["teacher"])
        if "opt" in s:
            self.opt.load_state_dict(s["opt"])
            self.sched.load_state_dict(s["sched"])
        self.step, self.log = s["step"], s["log"]
        held = [e["heldout"] for e in self.log if e.get("heldout") is not None]
        self.best = min(held) if held else float("inf")
        for k, v in s["basis"].items():
            setattr(self.basis, k, v)

    @property
    def selects_best(self) -> bool:
        """Whether ``.best.pt`` is the held-out minimum (False: the latest evaluation)."""
        return self.cfg.arm not in LATENT_ARMS

    @torch.no_grad()
    def monitor(self) -> dict:
        """Collapse statistics of the token representation on held-out windows.

        Tokens, not pooled windows: 32 pooled vectors cap the stable rank at 32 of
        384 dimensions whatever the model does.
        """
        from ..objectives.arms import CollapseMonitor

        self.model.eval()
        z = []
        for i in range(4):
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                z.append(self.model(self.batch(30_000_000 + i, train=False))[:, self.valid].float()
                         .reshape(-1, self.model.d_model))
        self.model.train()
        rep = CollapseMonitor().check(torch.cat(z))
        return {k: float(v) for k, v in rep.items() if k != "abort"}

    @torch.no_grad()
    def heldout(self) -> float:
        if not self._pool[False]:
            return float("nan")
        self.model.eval()
        self.head.eval()
        vals = []
        for i in range(self.cfg.eval_batches):
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                self._keys = self._batch_keys(10_000_000 + i, False) if self.cond is not None else None
                loss, _ = self.loss(self.batch(10_000_000 + i, train=False), 10_000_000 + i)
            vals.append(float(loss))
        self.model.train()
        self.head.train()
        return float(np.mean(vals))

    def fit(self, ckpt: Path | None = None, *, stop_at: int | None = None,
            verbose: bool = True) -> list[dict]:
        if ckpt is not None and Path(ckpt).exists():
            self.load(ckpt)
            if verbose:
                print(f"  resumed {self.cfg.arm} at step {self.step}")
        scaler = torch.amp.GradScaler(enabled=self.amp_dtype == torch.float16)
        end = min(self.cfg.steps, stop_at if stop_at is not None else self.cfg.steps)
        t0, start = time.time(), self.step
        pool = ThreadPoolExecutor(self.cfg.prefetch) if self.cfg.prefetch > 0 else None
        ahead: dict = {}

        def next_batch(step: int) -> torch.Tensor:
            if pool is None:
                return self.batch(step)
            for s in range(step, min(step + self.cfg.prefetch + 1, end)):
                if s not in ahead:
                    ahead[s] = pool.submit(self._batch_np, s, True)
            return torch.as_tensor(ahead.pop(step).result(), device=self.device)

        self.model.train()
        try:
            self._fit_loop(end, next_batch, scaler, ckpt, verbose, t0, start)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        return self.log

    def _fit_loop(self, end, next_batch, scaler, ckpt, verbose, t0, start) -> None:
        while self.step < end:
            x = next_batch(self.step)
            if self.cond is not None:
                self._keys = self._batch_keys(self.step, True)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                loss, parts = self.loss(x, self.step)
            self.opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.params, self.cfg.clip)
            scaler.step(self.opt)
            scaler.update()
            self.sched.step()
            if self.cfg.arm in LATENT_ARMS:
                self.teacher.update(self.model, self.step, self.cfg.steps)
            self.step += 1
            rec = {"step": self.step, "loss": float(loss.detach()), **parts,
                   "lr": self.sched.get_last_lr()[0]}
            if self.step % self.cfg.eval_every == 0 or self.step == end:
                rec["heldout"] = self.heldout()
                if self.cfg.arm in LATENT_ARMS and self._pool[False]:
                    rec["monitor"] = self.monitor()
                if verbose:
                    print(f"  {self.cfg.arm:<7} step {self.step:5d}  loss {rec['loss']:+.4f}  "
                          f"held-out {rec['heldout']:+.4f}  lr {rec['lr']:.2e}  "
                          f"{(time.time() - t0) / max(self.step - start, 1):.2f} s/step", flush=True)
                if ckpt is not None:
                    self.log.append(rec)
                    if rec["heldout"] < self.best or not self.selects_best:
                        self.best = min(self.best, rec["heldout"])
                        self.save(self.best_path(ckpt), weights_only=True)
                    self.save(ckpt)
                    continue
            self.log.append(rec)
