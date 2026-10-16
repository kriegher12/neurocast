"""Causal re-preprocessing check (README section 2.5).

The released LibriBrain ``.h5`` files were band-passed with zero-phase filters, so
each sample already holds a little of its own future, and every forecasting run
here trained on them. This script re-preprocesses the deep listener's raw 1 kHz
recordings (Sherlock 1 sessions 1-10) three ways and asks whether that matters:

* ``ZR`` -- pnpl's own temporal steps (zero-phase FIR notch and band-pass, FFT
  resampling): only to check that this pipeline reproduces the release;
* ``Z``  -- zero-phase notch, band-pass and a 110 Hz anti-alias low-pass, then
  every fourth sample;
* ``C``  -- the same filters converted to minimum phase (causal), then every
  fourth sample. Z and C share a magnitude response; only causality differs.

The spatial steps match the release (bad channels from ``find_bad_channels_maxwell``,
the release's cached head positions, SSS with movement correction to (0, 0, 0.04)
m and its calibration files). SSS runs in 5-minute pieces: a whole session at
1 kHz in float64 does not fit in RAM, and SSS has no temporal filter, so the
pieces join exactly. The temporal filters then run on whole sessions, a block of
channels at a time, from a channel-major float32 memmap on the data drive.

    .venv/Scripts/python.exe scripts/run_causal_check.py --stage selftest
    .venv/Scripts/python.exe scripts/run_causal_check.py --stage preprocess
    .venv/Scripts/python.exe scripts/run_causal_check.py --stage fidelity
    .venv/Scripts/python.exe scripts/run_causal_check.py --stage evoked
    .venv-gpu/Scripts/python.exe scripts/run_causal_check.py --stage train
    .venv-gpu/Scripts/python.exe scripts/run_causal_check.py --stage evaluate
    .venv-gpu/Scripts/python.exe scripts/run_causal_check.py --stage guarded

``guarded`` repeats seed 1 with the calibration guard (``PretrainConfig.calib_guard``):
seed 1's zero-phase run calibrated its targets on the pre-echo of an artifact and
failed. A version whose guard rejects nothing reuses its unguarded run, which is
then identical by construction (``validate_pretrain.py`` section 7).

Every stage skips work already on disk, so the chain can be stopped and restarted.
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

from neurocast.data.libribrain import Run, find_runs, read_words  # noqa: E402
from neurocast.data.paths import dataset_dir, replace_file  # noqa: E402

OUT = ROOT / "runs" / "causal_check"
TASK = "Sherlock1"
SESSIONS = tuple(str(i) for i in range(1, 11))
TRAIN = tuple(str(i) for i in range(1, 9))
HELD = ("9", "10")
VERSIONS = ("ZR", "Z", "C")
SFREQ_RAW, DOWN = 1000.0, 4
CHUNK_S = 300.0
LP_FREQ, LP_TRANS = 110.0, 15.0
DESTINATION = (0.0, 0.0, 0.04)


def libribrain() -> Path:
    return dataset_dir("LibriBrain100")


def derived(version: str) -> Path:
    return dataset_dir("derived") / "causal_check" / version


def stem(session: str) -> str:
    return f"sub-0_ses-{session}_task-{TASK}_run-1"


def raw_path(session: str) -> Path:
    return libribrain() / TASK / "sub-0" / f"ses-{session}" / "meg" / f"{stem(session)}_meg.fif"


def out_h5(version: str, session: str) -> Path:
    return derived(version) / f"{stem(session)}_proc-causalcheck_meg.h5"


def release_run(session: str) -> Run:
    return find_runs(task=TASK, sessions={session}, subjects={"0"}, require_meg=True)[0]


# ----------------------------------------------------------------------------- filters

def temporal(x: np.ndarray, version: str) -> np.ndarray:
    """(channels, samples) at 1 kHz -> float32 at 250 Hz, filtered as ``version``."""
    import mne

    kw = {"verbose": False}
    x = np.asarray(x, dtype=np.float64)
    if version == "ZR":                       # pnpl's notch, bp and ds steps, MNE defaults
        x = mne.filter.notch_filter(x, SFREQ_RAW, [50.0, 100.0], **kw)
        x = mne.filter.filter_data(x, SFREQ_RAW, 0.1, 125.0, **kw)
        return mne.filter.resample(x, down=float(DOWN), npad="auto", **kw).astype(np.float32)
    phase = {"Z": "zero", "C": "minimum"}[version]
    x = mne.filter.notch_filter(x, SFREQ_RAW, [50.0, 100.0], phase=phase, **kw)
    x = mne.filter.filter_data(x, SFREQ_RAW, 0.1, 125.0, phase=phase, **kw)
    x = mne.filter.filter_data(x, SFREQ_RAW, None, LP_FREQ, h_trans_bandwidth=LP_TRANS,
                               phase=phase, **kw)
    return np.ascontiguousarray(x[:, ::DOWN]).astype(np.float32)


def impulse_check() -> dict:
    """Z and C impulse responses: C has no energy before the impulse, Z does; same magnitude."""
    n = 120_000                                       # 120 s: longer than the 33 s high-pass
    x = np.zeros((1, n))
    x[0, n // 2] = 1.0
    out = {}
    for v in ("Z", "C"):
        h = temporal(x, v)[0].astype(np.float64)
        c = n // 2 // DOWN
        out[v] = {"pre_energy": float((h[:c] ** 2).sum() / (h ** 2).sum()),
                  "spectrum": np.abs(np.fft.rfft(h))}
    f = np.fft.rfftfreq(n // DOWN, DOWN / SFREQ_RAW)
    # Passband only: the notch nulls and the low-pass transition are ratios of ~zeros.
    band = (f > 1.0) & (f < 95.0) & (np.abs(f - 50.0) > 3.0)
    db = 20 * np.log10(out["Z"]["spectrum"][band] / out["C"]["spectrum"][band])
    return {"pre_energy_Z": out["Z"]["pre_energy"], "pre_energy_C": out["C"]["pre_energy"],
            "max_magnitude_difference_db": float(np.abs(db).max())}


def onset_leak(x: np.ndarray, onsets: np.ndarray, sfreq: float, *, n_boot: int = 0,
               seed: int = 0) -> dict:
    """Field power before sentence onsets, as % of the post-onset peak (README section 2.5).

    ``x`` (channels, samples), z-scored per channel by the caller. Epochs run
    -1.0 to +0.5 s; each epoch's channel means over the -600 to -400 ms baseline
    are removed. GFP is the RMS over channels of the average epoch. Leak = mean GFP
    in [-100, 0) ms minus mean GFP in the baseline, over max GFP in [50, 200] ms.
    """
    pre, post = int(round(1.0 * sfreq)), int(round(0.5 * sfreq))
    idx = np.round(np.asarray(onsets) * sfreq).astype(int)
    idx = idx[(idx - pre >= 0) & (idx + post <= x.shape[1])]
    ep = np.stack([x[:, i - pre:i + post] for i in idx]).astype(np.float64)   # (E, C, T)
    t = (np.arange(-pre, post)) / sfreq
    base = (t >= -0.6) & (t < -0.4)
    ep -= ep[:, :, base].mean(2, keepdims=True)

    def leak(sel):
        gfp = np.sqrt((ep[sel].mean(0) ** 2).mean(0))
        peak = gfp[(t >= 0.05) & (t <= 0.2)].max()
        return float((gfp[(t >= -0.1) & (t < 0)].mean() - gfp[base].mean()) / peak), gfp

    value, gfp = leak(np.arange(len(ep)))
    out = {"leak": value, "n_onsets": int(len(ep)), "gfp": gfp.tolist(), "t": t.tolist()}
    if n_boot:
        rng = np.random.default_rng(seed)
        out["boot"] = [leak(rng.integers(0, len(ep), len(ep)))[0] for _ in range(n_boot)]
    return out


# ----------------------------------------------------------------------------- stages

def stage_selftest(args) -> dict:
    r = impulse_check()
    # A broadband zero-phase filter keeps most energy in its centre tap, so Z's share
    # before the impulse is small but real; C's must be numerically nothing.
    ok = r["pre_energy_C"] < 1e-9 and r["pre_energy_Z"] > 1e-3 and r["max_magnitude_difference_db"] < 0.1
    print(f"impulse energy before the impulse: Z {100 * r['pre_energy_Z']:.1f}%, "
          f"C {100 * r['pre_energy_C']:.2e}%; |Z| vs |C| over the 1-95 Hz passband: max "
          f"{r['max_magnitude_difference_db']:.3f} dB  ->  {'PASS' if ok else 'FAIL'}")
    r["pass"] = bool(ok)
    return r


def _chunk_head_pos(head_pos: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Positions inside [t0, t1], led by the last earlier one moved to t0."""
    inside = head_pos[(head_pos[:, 0] >= t0) & (head_pos[:, 0] <= t1)]
    before = head_pos[head_pos[:, 0] < t0]
    if len(before):
        lead = before[-1:].copy()
        lead[0, 0] = t0
        inside = np.vstack([lead, inside])
    return inside


def spatial(session: str, mm_path: Path, names: list[str]) -> dict:
    """Bad channels, head positions and SSS in 5-minute pieces, into a channel-major memmap."""
    import mne

    raw = mne.io.read_raw_fif(raw_path(session), allow_maxshield=True, preload=False, verbose=False)
    sf = raw.info["sfreq"]
    if sf != SFREQ_RAW:
        raise SystemExit(f"session {session}: expected 1 kHz raw, got {sf}")
    neo = libribrain() / "metadata" / "neo"
    cal, ct = neo / "sss_cal.dat", neo / "ct_sparse.fif"
    head_pos = mne.chpi.read_head_pos(
        libribrain() / TASK / "derivatives" / "preproc" / "headpos" / f"{stem(session)}.csv")
    n = int(raw.n_times)
    first = raw.first_samp / sf
    print(f"  raw: {n / sf / 60:.1f} min, first sample at {first:.2f} s; head positions "
          f"{head_pos[0, 0]:.2f}-{head_pos[-1, 0]:.2f} s ({len(head_pos)})", flush=True)

    bads: set[str] = set()
    span = int(CHUNK_S * sf)
    for a in sorted({0, max(0, n // 2 - span // 2), max(0, n - span)}):
        seg = raw.copy().crop(a / sf, (min(a + span, n) - 1) / sf).load_data(verbose=False)
        seg.info["bads"] = []
        noisy, flat = mne.preprocessing.find_bad_channels_maxwell(
            seg, calibration=cal, cross_talk=ct, h_freq=40.0, verbose=False)
        bads |= set(noisy) | set(flat)
        del seg
    print(f"  bad channels (union of three 5-min stretches): {sorted(bads)}", flush=True)

    mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float32, shape=(len(names), n))
    for a in range(0, n, span):
        b = min(a + span, n)
        seg = raw.copy().crop(a / sf, (b - 1) / sf).load_data(verbose=False)
        seg.info["bads"] = sorted(bads)
        t0, t1 = seg.first_samp / sf, (seg.first_samp + seg.n_times - 1) / sf
        sss = mne.preprocessing.maxwell_filter(
            seg, origin="auto", coord_frame="head", destination=DESTINATION, calibration=cal,
            cross_talk=ct, head_pos=_chunk_head_pos(head_pos, t0, t1), verbose=False)
        mm[:, a:b] = sss.get_data(picks=names).astype(np.float32)
        del seg, sss
        print(f"    SSS {b / sf / 60:5.1f} / {n / sf / 60:.1f} min", flush=True)
    mm.flush()
    del mm
    return {"bads": sorted(bads), "n_samples_1k": int(n), "first_time_s": first}


def write_h5(path: Path, x: np.ndarray, attrs: dict) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("data", data=x)
        for k, v in attrs.items():
            f.attrs[k] = v
    replace_file(tmp, path)


def stage_preprocess(args) -> dict:
    import h5py

    res = {}
    for s in args.sessions:
        if all(out_h5(v, s).exists() for v in VERSIONS):
            print(f"session {s}: done earlier")
            continue
        t0 = time.time()
        rel = release_run(s)
        with h5py.File(rel.h5, "r") as f:
            attrs = {k: f.attrs[k] for k in ("channel_names", "channel_types") if k in f.attrs}
        names = [c.strip() for c in str(attrs["channel_names"]).split(",") if c.strip()]
        print(f"session {s}: {raw_path(s).name}", flush=True)
        mm_path = dataset_dir("derived") / "causal_check" / f"sss_{stem(s)}.npy"
        mm_path.parent.mkdir(parents=True, exist_ok=True)
        info = spatial(s, mm_path, names)
        mm = np.load(mm_path, mmap_mode="r")
        outs = {v: np.zeros((len(names), -(-mm.shape[1] // DOWN)), np.float32) for v in VERSIONS}
        for i in range(0, len(names), args.block):
            block = np.asarray(mm[i:i + args.block], dtype=np.float64)
            for v in VERSIONS:
                y = temporal(block, v)
                outs[v][i:i + args.block, :y.shape[1]] = y
            print(f"    filtered channels {min(i + args.block, len(names))} / {len(names)}", flush=True)
        del mm
        n250 = min(o.shape[1] for o in outs.values())
        for v in VERSIONS:
            write_h5(out_h5(v, s), outs[v][:, :n250],
                     {**attrs, "sample_frequency": 250.0, "version": v, "bads": ",".join(info["bads"])})
        mm_path.unlink()
        res[s] = {**info, "minutes": (time.time() - t0) / 60}
        print(f"  session {s} written ({res[s]['minutes']:.1f} min)", flush=True)
    return res


def story_span(session: str) -> tuple[int, int]:
    """First to last word, in 250 Hz samples of the recording (the release's clock)."""
    w = read_words(release_run(session).events)
    return int(w["onset"].min() * 250), int((w["onset"] + w["duration"]).max() * 250) + 1


def load_version(version: str, session: str) -> np.ndarray:
    import h5py

    path = release_run(session).h5 if version == "release" else out_h5(version, session)
    with h5py.File(path, "r") as f:
        return np.asarray(f["data"][...], dtype=np.float32)


def stage_fidelity(args) -> dict:
    """Per-channel correlation, within the story span: ZR vs release, Z vs ZR, C vs Z."""
    res = {}
    print(f"  {'session':<8} {'ZR vs release':>22} {'Z vs ZR':>16} {'C vs Z':>16}   (median r, 5th pct)")
    for s in args.sessions:
        a, b = story_span(s)
        x = {v: load_version(v, s)[:, a:b].astype(np.float64) for v in ("release", "ZR", "Z", "C")}
        n = min(v.shape[1] for v in x.values())

        def r(p, q):
            p, q = x[p][:, :n], x[q][:, :n]
            p, q = p - p.mean(1, keepdims=True), q - q.mean(1, keepdims=True)
            c = (p * q).sum(1) / np.sqrt((p ** 2).sum(1) * (q ** 2).sum(1))
            return float(np.median(c)), float(np.percentile(c, 5))

        res[s] = {"zr_release": r("ZR", "release"), "z_zr": r("Z", "ZR"), "c_z": r("C", "Z")}
        z = res[s]
        print(f"  ses-{s:<4} {z['zr_release'][0]:12.4f} {z['zr_release'][1]:8.4f} "
              f"{z['z_zr'][0]:9.4f} {z['z_zr'][1]:6.4f} {z['c_z'][0]:9.4f} {z['c_z'][1]:6.4f}", flush=True)
        del x
    med = [v["zr_release"][0] for v in res.values()]
    res["_verdict"] = "reproduces" if min(med) >= 0.95 else "does not reproduce"
    print(f"  rule (median r >= 0.95 in every session): {res['_verdict']}")
    return res


def sentence_onsets(session: str) -> np.ndarray:
    """First word of a sentence after at least 0.5 s of silence, recording time (s)."""
    from neurocast.decode.words import timing_features

    w = timing_features(read_words(release_run(session).events))
    return w.loc[(w["widx"] == 0) & (w["gap_prev"] >= 0.5), "onset"].to_numpy()


def stage_evoked(args) -> dict:
    """Field power around sentence onsets: does Z put response before the onset, and C not?"""
    res, boots = {}, {}
    for v in ("release", "ZR", "Z", "C"):
        eps = []
        for s in args.sessions:
            a, b = story_span(s)
            x = load_version(v, s)
            mu = x[:, a:b].mean(1, keepdims=True)
            sd = x[:, a:b].std(1, keepdims=True) + 1e-30
            x = ((x - mu) / sd).astype(np.float32)
            on = sentence_onsets(s)
            eps.append((x, on))
        # Pool onsets across sessions as one epoch set, in a fixed order for every version.
        r = onset_leak_pooled(eps, n_boot=args.n_boot, seed=args.seed)
        res[v] = {k: r[k] for k in ("leak", "n_onsets", "gfp", "t")}
        boots[v] = np.asarray(r["boot"])
        print(f"  {v:<8} leak {100 * r['leak']:+6.2f}% of the post-onset peak "
              f"({r['n_onsets']} sentence onsets)", flush=True)
    d = boots["Z"] - boots["C"]
    res["z_minus_c"] = {"leak": res["Z"]["leak"] - res["C"]["leak"],
                        "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
    zc = res["z_minus_c"]
    res["_verdict"] = "leaks" if zc["leak"] >= 0.05 and zc["lo"] > 0 else "does not leak"
    print(f"  Z - C: {100 * zc['leak']:+.2f}% [{100 * zc['lo']:+.2f}, {100 * zc['hi']:+.2f}] "
          f"-> {res['_verdict']} (rule: >= 5% with the interval above 0)")
    return res


def onset_leak_pooled(parts, *, n_boot: int, seed: int) -> dict:
    """:func:`onset_leak` over several sessions' epochs at once (resampled together)."""
    pre, post = 250, 125
    ep = []
    for x, on in parts:
        idx = np.round(on * 250).astype(int)
        idx = idx[(idx - pre >= 0) & (idx + post <= x.shape[1])]
        ep.extend(x[:, i - pre:i + post] for i in idx)
    ep = np.stack(ep)                                        # (E, C, T) float32
    t = np.arange(-pre, post) / 250.0
    base = (t >= -0.6) & (t < -0.4)
    ep -= ep[:, :, base].mean(2, keepdims=True)
    flat = ep.reshape(len(ep), -1)
    pre_m, post_m = (t >= -0.1) & (t < 0), (t >= 0.05) & (t <= 0.2)

    def leak_of(mean_flat):
        gfp = np.sqrt((mean_flat.reshape(ep.shape[1], -1) ** 2).mean(0))
        return float((gfp[pre_m].mean() - gfp[base].mean()) / gfp[post_m].max()), gfp

    value, gfp = leak_of(flat.mean(0))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(ep), len(ep)), minlength=len(ep)).astype(np.float32)
        boot.append(leak_of(w @ flat / len(ep))[0])
    return {"leak": value, "n_onsets": int(len(ep)), "gfp": gfp.tolist(), "t": t.tolist(),
            "boot": boot}


# ----------------------------------------------------------------------------- training

ARMS = {"ar_lik": dict(arm="ar_lik"), "masked_visible": dict(arm="masked", visible_weight=1.0)}
QUEUE = [("ar_lik", 0), ("ar_lik", 1), ("masked_visible", 0)]
STEPS = 5_000


def version_runs(version: str) -> list[Run]:
    out = []
    for s in SESSIONS:
        r = release_run(s)
        out.append(Run(r.subject, r.session, r.task, r.run, events=r.events,
                       h5=out_h5(version, s), desc=r.desc))
    return out


def cache_dir(version: str) -> Path:
    return dataset_dir("derived") / "causal_check" / f"cache_{version}"


#: Calibration guard for ``--stage guarded`` (session-normalised units; README 2.5).
GUARD = 100.0


def build(version: str, arm: str, seed: int, device: str, prefetch: int = 8, guard: float = 0.0):
    from run_pretrain_pilot import montage_for

    from neurocast.train.pretrain import PretrainConfig, Pretrainer, cache_recordings, open_cached

    runs = version_runs(version)
    keys = cache_recordings(runs, cache_dir(version), verbose=False)
    held = ["_".join(r.key) for r in runs if r.session in HELD]
    recs = open_cached(cache_dir(version), keys, heldout=held)
    cfg = PretrainConfig(steps=STEPS, seed=seed, eval_every=250, eval_batches=32,
                         prefetch=prefetch, calib_guard=guard, **ARMS[arm])
    return Pretrainer(cfg, montage_for(release_run(SESSIONS[0])), recs, device=device)


def ckpt(version: str, arm: str, seed: int, guard: bool = False) -> Path:
    return OUT / f"{version}_{arm}_s{seed}{'_guard' if guard else ''}.pt"


def stage_train(args) -> dict:
    import gc

    import torch

    if args.gpu_mem_frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
    res = {}
    for arm, seed in QUEUE:
        for v in ("Z", "C"):
            ck = ckpt(v, arm, seed)
            tr = build(v, arm, seed, args.device)
            if ck.exists():
                tr.load(ck)
            if tr.step >= STEPS:
                print(f"{ck.stem}: finished, skipped", flush=True)
                continue
            print(f"{ck.stem}: {tr.model.describe()}; {STEPS} steps; {tr.device}", flush=True)
            t0 = time.time()
            tr.fit(ck)
            res[ck.stem] = {"best": tr.best, "minutes": (time.time() - t0) / 60}
            print(f"{ck.stem}: done in {res[ck.stem]['minutes']:.1f} min, best held-out {tr.best:.4f}",
                  flush=True)
            del tr
            gc.collect()
            torch.cuda.empty_cache()
    return res


POST_S = 0.2


def token_nll(tr, x):
    """Per-token NLL per target dimension, ``(B, N - 1)``; token j predicts token j + 1."""
    import torch

    from neurocast.objectives.arms import ar_shift

    with torch.no_grad(), torch.autocast(tr.device.type, dtype=tr.amp_dtype,
                                         enabled=tr.amp_dtype is not None):
        tgt = tr.basis.encode(x.cpu()).to(tr.device)
        h, t = ar_shift(tr.model(x), tgt)
        m = tr.dim_mask[1:].expand_as(t)
        lp = tr.head.log_prob(h.float(), t.float(), m)
        dims = m.float().sum(-1)
    nll = (-lp / dims.clamp_min(1)).float().cpu().numpy()
    nll[(dims == 0).cpu().numpy()] = np.nan
    return nll


def onset_windows(tr, version: str):
    """Held-out windows with a sentence onset at the centre; per-onset post and other NLL."""
    import torch

    from neurocast.train.pretrain import open_cached

    post, other = [], []
    for s in HELD:
        key = f"0_{s}_{TASK}_1"
        rec = open_cached(cache_dir(version), [key], heldout=[key])[0]
        a = rec.story[0]
        half = tr.n_times // 2
        starts = np.round(sentence_onsets(s) * 250).astype(int) - a - half
        starts = starts[(starts >= 0) & (starts + tr.n_times <= rec.n_samples)]
        for i in range(0, len(starts), 16):
            x = torch.as_tensor(np.stack([rec.window(int(st), tr.n_times) for st in starts[i:i + 16]])
                                .astype(np.float32), device=tr.device)
            nll = token_nll(tr, x)                                   # (b, N - 1)
            patch = (np.arange(1, nll.shape[1] + 1)) // tr.G          # target token's patch
            t_rel = (patch * tr.patch - half) / 250.0                 # patch start vs onset (s)
            is_post = (t_rel >= 0) & (t_rel < POST_S)
            post.extend(np.nanmean(nll[:, is_post], 1))
            other.extend(np.nanmean(nll[:, ~is_post], 1))
    return np.asarray(post), np.asarray(other)


def stage_evaluate(args) -> dict:
    """Best held-out loss, onset-locked anticipation (forecasting) and the collapse verdict."""
    import torch
    from run_pretrain_pilot import collapse

    from neurocast.train.pretrain import Pretrainer

    if args.gpu_mem_frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
    res, onset = {}, {}
    for arm, seed in QUEUE:
        for v in ("Z", "C"):
            ck = ckpt(v, arm, seed)
            tr = build(v, arm, seed, args.device, prefetch=0)
            tr.load(Pretrainer.best_path(ck))
            full = torch.load(ck, map_location="cpu", weights_only=False)
            r = {"best_heldout": tr.best, "best_step": tr.step, "steps": full["step"],
                 "curve": [(e["step"], e["heldout"]) for e in full["log"] if e.get("heldout") is not None]}
            del full
            tr.model.eval()
            if arm == "ar_lik":
                onset[(v, seed)] = onset_windows(tr, v)
                p, o = onset[(v, seed)]
                r.update(post_nll=float(np.mean(p)), other_nll=float(np.mean(o)), n_onsets=len(p))
            else:
                rep = collapse(tr)
                r["collapse"] = {"spread": rep.spread, "sensitivity": rep.sensitivity,
                                 "collapsed": rep.collapsed}
            res[ck.stem] = r
            print(f"  {ck.stem:<24} best held-out {tr.best:.4f} (step {tr.step})"
                  + (f"   onset windows: post {r['post_nll']:.4f}, other {r['other_nll']:.4f} "
                     f"({r['n_onsets']})" if arm == "ar_lik" else
                     f"   collapse: {rep.summary()}"), flush=True)
            del tr
            torch.cuda.empty_cache()

    verdict = {}
    d = [res[f"Z_ar_lik_s{s}"]["best_heldout"] - res[f"C_ar_lik_s{s}"]["best_heldout"] for s in (0, 1)]
    verdict["heldout_z_minus_c"] = d
    verdict["forecasting"] = ("inflates" if all(x <= -0.01 for x in d) else
                              "no effect" if all(abs(x) < 0.01 for x in d) else "mixed")
    ant = []
    rng = np.random.default_rng(args.seed)
    for s in (0, 1):
        (zp, zo), (cp, co) = onset[("Z", s)], onset[("C", s)]
        di = (cp - zp) - (co - zo)
        boots = [di[rng.integers(0, len(di), len(di))].mean() for _ in range(args.n_boot)]
        ant.append({"mean": float(di.mean()), "lo": float(np.percentile(boots, 2.5)),
                    "hi": float(np.percentile(boots, 97.5)), "n": int(len(di))})
    verdict["anticipation"] = ant
    verdict["anticipates"] = all(a["lo"] > 0 for a in ant)
    cz, cc = (res[f"{v}_masked_visible_s0"]["collapse"]["collapsed"] for v in ("Z", "C"))
    verdict["collapse_same"] = cz == cc
    res["_verdicts"] = verdict
    print(f"\n  held-out loss, Z - C: seed 0 {d[0]:+.4f}, seed 1 {d[1]:+.4f} nats/dim -> "
          f"{verdict['forecasting']} (rule: 0.01)")
    for s, a in zip((0, 1), ant):
        print(f"  anticipation, seed {s}: (C - Z) after onsets minus elsewhere {a['mean']:+.4f} "
              f"[{a['lo']:+.4f}, {a['hi']:+.4f}] over {a['n']} onsets")
    print(f"  -> {'anticipates' if verdict['anticipates'] else 'no anticipation'}; "
          f"collapse verdict Z {'collapsed' if cz else 'reads input'}, C "
          f"{'collapsed' if cc else 'reads input'} -> {'same' if verdict['collapse_same'] else 'DIFFERENT'}")
    return res


def stage_guarded(args) -> dict:
    """Seed 1 of the forecasting arm again with the calibration guard, Z and C.

    Rules (README 2.5, written before this ran): the guard counts as having fixed
    the failure if the guarded Z run's best held-out loss is within 0.05 nats/dim
    of C's; the forecasting verdict is then re-read from seed 0 (whose runs the
    guard leaves identical if it rejects nothing there) and guarded seed 1, under
    the original 0.01 rule, and anticipation from guarded seed 1.
    """
    import gc

    import torch

    from neurocast.train.pretrain import Pretrainer

    if args.gpu_mem_frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
    res: dict = {"guard": GUARD, "rejected": {}}
    paths = {}
    for seed in (0, 1):
        for v in ("Z", "C"):
            tr = build(v, "ar_lik", seed, args.device, prefetch=8, guard=GUARD)
            res["rejected"][f"{v}_s{seed}"] = tr.calib_rejected
            if tr.calib_rejected == 0:
                paths[(v, seed)] = ckpt(v, "ar_lik", seed)          # identical run
                print(f"{v} seed {seed}: guard rejects nothing; the unguarded run stands", flush=True)
                del tr
                continue
            ck = ckpt(v, "ar_lik", seed, guard=True)
            paths[(v, seed)] = ck
            if ck.exists():
                tr.load(ck)
            if tr.step < STEPS:
                print(f"{ck.stem}: guard rejected {tr.calib_rejected} calibration windows; "
                      f"training {STEPS} steps", flush=True)
                t0 = time.time()
                tr.fit(ck)
                print(f"{ck.stem}: done in {(time.time() - t0) / 60:.1f} min, best held-out "
                      f"{tr.best:.4f}", flush=True)
            del tr
            gc.collect()
            torch.cuda.empty_cache()

    onset = {}
    for (v, seed), ck in paths.items():
        tr = build(v, "ar_lik", seed, args.device, prefetch=0,
                   guard=GUARD if ck.stem.endswith("_guard") else 0.0)
        tr.load(Pretrainer.best_path(ck))
        tr.model.eval()
        onset[(v, seed)] = onset_windows(tr, v)
        p, o = onset[(v, seed)]
        res[ck.stem] = {"best_heldout": tr.best, "best_step": tr.step, "post_nll": float(np.mean(p)),
                        "other_nll": float(np.mean(o)), "n_onsets": len(p)}
        print(f"  {ck.stem:<24} best held-out {tr.best:.4f} (step {tr.step})   onset windows: "
              f"post {np.mean(p):.4f}, other {np.mean(o):.4f} ({len(p)})", flush=True)
        del tr
        torch.cuda.empty_cache()

    best = {k: res[c.stem]["best_heldout"] for k, c in paths.items()}
    d = [best[("Z", s)] - best[("C", s)] for s in (0, 1)]
    verdict = {"heldout_z_minus_c": d, "fixed": abs(d[1]) < 0.05}
    verdict["forecasting"] = ("inflates" if all(x <= -0.01 for x in d) else
                              "no effect" if all(abs(x) < 0.01 for x in d) else "mixed")
    rng = np.random.default_rng(args.seed)
    (zp, zo), (cp, co) = onset[("Z", 1)], onset[("C", 1)]
    di = (cp - zp) - (co - zo)
    boots = [di[rng.integers(0, len(di), len(di))].mean() for _ in range(args.n_boot)]
    verdict["anticipation_s1"] = {"mean": float(di.mean()), "lo": float(np.percentile(boots, 2.5)),
                                  "hi": float(np.percentile(boots, 97.5)), "n": int(len(di))}
    res["_verdicts"] = verdict
    print(f"\n  guard rejected: {res['rejected']}")
    print(f"  held-out loss, Z - C: seed 0 {d[0]:+.4f}, seed 1 (guarded) {d[1]:+.4f} -> "
          f"{'fixed' if verdict['fixed'] else 'NOT fixed'}; forecasting verdict {verdict['forecasting']}")
    a = verdict["anticipation_s1"]
    print(f"  anticipation, guarded seed 1: {a['mean']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}]")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("selftest", "preprocess", "fidelity", "evoked", "train", "evaluate",
                             "guarded"))
    ap.add_argument("--sessions", nargs="*", default=list(SESSIONS))
    ap.add_argument("--block", type=int, default=34, help="channels filtered at once")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"selftest": stage_selftest, "preprocess": stage_preprocess, "fidelity": stage_fidelity,
           "evoked": stage_evoked, "train": stage_train,
           "evaluate": stage_evaluate, "guarded": stage_guarded}[args.stage](args)
    f = OUT / f"{args.stage}.json"
    old = json.loads(f.read_text()) if f.exists() and args.stage == "preprocess" else {}
    f.write_text(json.dumps({**old, **res}, indent=2, default=float))
    print(f"saved {f.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
