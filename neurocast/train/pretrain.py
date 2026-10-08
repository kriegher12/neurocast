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
           "Pretrainer"]

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
    arm: str = "ar_lik"                 # "ar_lik" or "masked"
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
        causal = cfg.arm != "masked"
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
        crng = np.random.default_rng(cfg.seed + 7)        # one generator: 64 distinct windows
        calib = np.stack([self._window(crng, train=True) for _ in range(64)])
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
        else:
            raise ValueError(f"arm must be 'ar_lik' or 'masked', got {cfg.arm!r}")

        params = list(self.model.parameters()) + list(self.head.parameters())
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95),
                                     weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, _lr_lambda(cfg))
        self.step = 0
        self.log: list[dict] = []
        self.best = float("inf")
        self.amp_dtype = self._precision()

    # -- data -------------------------------------------------------------------------
    def _window(self, rng: np.random.Generator, train: bool) -> np.ndarray:
        pool = self._pool[train]
        r = pool[int(rng.integers(len(pool)))]
        lo, hi = (0, r.train_stop) if train else (r.train_stop, r.n_samples)
        start = int(rng.integers(lo, max(hi - self.n_times, lo + 1)))
        return r.window(start, self.n_times)

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

    def loss(self, x: torch.Tensor, step: int) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        tgt = self.basis.encode(x.cpu()).to(self.device)                        # (B, N, r)
        if cfg.arm == "ar_lik":
            h, t = ar_shift(self.model(x), tgt)
            m = self.dim_mask[1:].expand_as(t)
            lp = self.head.log_prob(h.float(), t, m)
            nll = -lp.sum() / m.sum()
            return nll, {"nll": float(nll.detach())}
        g = torch.Generator().manual_seed(cfg.seed * 1_000_003 + step)
        blk = max(1, int(round(cfg.mask_block_s * 250.0 / self.patch)) * self.G)
        hidden = block_mask((x.shape[0], tgt.shape[1]), block_len=blk, target_frac=cfg.mask_frac,
                            generator=g, device=self.device)
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
                "sched": self.sched.state_dict(), "log": self.log,
                "basis": {k: getattr(self.basis, k) for k in
                          ("rank", "patch", "n_groups", "means", "comps", "scales", "channels", "evr")}}

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
        """Resume from a full checkpoint, or load weights from a ``.best.pt`` one."""
        s = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(s["model"])
        self.head.load_state_dict(s["head"])
        if "opt" in s:
            self.opt.load_state_dict(s["opt"])
            self.sched.load_state_dict(s["sched"])
        self.step, self.log = s["step"], s["log"]
        held = [e["heldout"] for e in self.log if e.get("heldout") is not None]
        self.best = min(held) if held else float("inf")
        for k, v in s["basis"].items():
            setattr(self.basis, k, v)

    @torch.no_grad()
    def heldout(self) -> float:
        if not self._pool[False]:
            return float("nan")
        self.model.eval()
        self.head.eval()
        vals = []
        for i in range(self.cfg.eval_batches):
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
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
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                loss, parts = self.loss(x, self.step)
            self.opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.params, self.cfg.clip)
            scaler.step(self.opt)
            scaler.update()
            self.sched.step()
            self.step += 1
            rec = {"step": self.step, "loss": float(loss.detach()), **parts,
                   "lr": self.sched.get_last_lr()[0]}
            if self.step % self.cfg.eval_every == 0 or self.step == end:
                rec["heldout"] = self.heldout()
                if verbose:
                    print(f"  {self.cfg.arm:<7} step {self.step:5d}  loss {rec['loss']:+.4f}  "
                          f"held-out {rec['heldout']:+.4f}  lr {rec['lr']:.2e}  "
                          f"{(time.time() - t0) / max(self.step - start, 1):.2f} s/step", flush=True)
                if ckpt is not None:
                    self.log.append(rec)
                    if rec["heldout"] < self.best:
                        self.best = rec["heldout"]
                        self.save(self.best_path(ckpt), weights_only=True)
                    self.save(ckpt)
                    continue
            self.log.append(rec)
