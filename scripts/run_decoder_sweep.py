"""Stronger decoders through the timing-matched test: does more decoder find more word?

The pitch's plan after its audit: "stronger decoders through the timing-matched
test (CNN tuned on fit data only, EEGNet)". Its finding to extend: "the stronger
the decoder, the more genuine word information it finds" (linear -> MLP -> CNN).
This script adds three decoders and asks the same two questions of each --
how much of its gain survives matching trials on timing, length and loudness,
and how many nats per word it adds beyond those covariates:

* **CNN, tuned**: spatial width x kernel x dropout x learning rate, chosen by
  early-stopping validation BAcc@10 on an occurrence-grouped 10% of chapter 11.
  Chapter 12 is never looked at until the choice is made;
* **CNN, tuned, 5 seeds**: the chosen CNN averaged over 5 seeds (log-softmax);
* **subject CNN**: the chosen CNN behind a per-listener spatial layer;
* **EEGNet** (Lawhern et al. 2018), adapted to 306 sensors.

Same protocol as ``run_libribrain_audit.py``: 32 listeners, 50 words, 0.512 s
windows, fit chapter 11, test chapter 12. Reads the windows the audit cached.
``--window 1.024 --vocab competition`` runs it under the competition's protocol,
after the audit has run that way (its folder holds the vocabulary and scores).

    .venv-gpu/Scripts/python.exe scripts/run_decoder_sweep.py
    .venv-gpu/Scripts/python.exe scripts/run_decoder_sweep.py --window 1.024 --vocab competition
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from neurocast.decode.decoders import CNNDecoder, EEGNetDecoder, SubjectCNNDecoder  # noqa: E402
from neurocast.decode.metrics import balanced_accuracy_at_k  # noqa: E402
from neurocast.decode.words import WINDOW_S  # noqa: E402
from run_libribrain_audit import BROAD, load_split, matched_setup  # noqa: E402

GRID = {"spatial": (32, 64, 128), "kernel": (5, 9, 15), "dropout": (0.3, 0.5), "lr": (1e-3, 2e-3)}


def log_softmax(s: np.ndarray) -> np.ndarray:
    s = s - s.max(1, keepdims=True)
    return s - np.log(np.exp(s).sum(1, keepdims=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--window", type=float, default=WINDOW_S)
    ap.add_argument("--vocab", choices=("top50", "competition"), default="top50")
    ap.add_argument("--hp", default="", help="skip tuning with a setting chosen earlier, as JSON, "
                    'e.g. {"spatial": 128, "kernel": 9, "dropout": 0.3, "lr": 0.001}')
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85,
                    help="cap on this process's share of GPU memory")
    args = ap.parse_args()
    if args.device.startswith("cuda"):
        # On Windows the driver can spill CUDA allocations into shared system RAM
        # instead of failing; the 1.024 s, width-128 CNN did that on a 4 GB card and
        # ran ~20x slower. A cap makes PyTorch fail or pick smaller workspaces.
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
    t0 = time.time()
    OUT = ROOT / "runs" / "libribrain_audit"
    if abs(args.window - WINDOW_S) > 1e-9 or args.vocab != "top50":
        OUT = ROOT / "runs" / f"libribrain_audit_w{int(round(args.window * 1000))}_{args.vocab}"

    vocab = json.loads((OUT / "nobrain.json").read_text())["vocab"]
    K = len(vocab)
    w_fit, fit, _ = load_split("11", vocab, BROAD, "session-scalar", args.window)
    w_te, te, _ = load_split("12", vocab, BROAD, "session-scalar", args.window)
    yf, yt, occ = fit["label"].to_numpy(), te["label"].to_numpy(), fit["occ"].to_numpy()
    sf, st = fit["subject"].astype(str).to_numpy(), te["subject"].astype(str).to_numpy()

    def bacc(s):
        return balanced_accuracy_at_k(s, yt, 1), balanced_accuracy_at_k(s, yt, 10)

    # 1. Tune the CNN on chapter 11 only (or take the setting an earlier run chose).
    tuning = []
    grid = [] if args.hp else list(itertools.product(*GRID.values()))
    if grid:
        print(f"tuning the CNN on chapter 11 ({len(grid)} settings, {args.device}) ...", flush=True)
    for values in grid:
        hp = dict(zip(GRID, values))
        m = CNNDecoder(seed=0, device=args.device, **hp).fit(w_fit, yf, occ, K)
        tuning.append({**hp, "val_bacc10": m.val_bacc, "epochs": m.epochs})
        print(f"  {hp}  val BAcc@10 {m.val_bacc:.3f}  ({m.epochs} epochs, {time.time() - t0:.0f}s)",
              flush=True)
    if args.hp:
        hp = json.loads(args.hp)
        print(f"setting chosen by an earlier run on chapter 11: {hp}", flush=True)
    else:
        best = max(tuning, key=lambda r: r["val_bacc10"])
        hp = {k: best[k] for k in GRID}
        print(f"chosen on chapter 11: {hp} (val BAcc@10 {best['val_bacc10']:.3f})", flush=True)

    # 2. Score the new decoders on chapter 12.
    scores, res = {}, {"tuning": tuning, "chosen": hp}
    ens = []
    for seed in range(args.seeds):
        m = CNNDecoder(seed=seed, device=args.device, **hp).fit(w_fit, yf, occ, K)
        s = m.scores(w_te)
        ens.append(log_softmax(s))
        if seed == 0:
            scores["cnn_tuned"] = s
    scores["cnn_tuned_ens"] = np.mean(ens, axis=0)
    arch = {k: hp[k] for k in ("spatial", "kernel", "dropout")}
    m = SubjectCNNDecoder(seed=0, device=args.device, lr=hp["lr"], **arch).fit(
        w_fit, yf, occ, K, subjects=sf)
    scores["subject_cnn"] = m.scores(w_te, subjects=st)
    m = EEGNetDecoder(seed=0, device=args.device).fit(w_fit, yf, occ, K)
    scores["eegnet"] = m.scores(w_te)
    del w_fit, w_te

    base = np.load(OUT / "test_scores.npz")
    assert np.array_equal(base["y"], yt), "test trials differ from the audit's"
    for k in ("linear", "cnn"):
        scores[k] = base[k]
    print(f"\n  {'decoder':<16} {'BAcc@1':>7} {'BAcc@10':>8}")
    for k, s in scores.items():
        res[k] = {"bacc": bacc(s)}
        print(f"  {k:<16} {res[k]['bacc'][0]:7.3f} {res[k]['bacc'][1]:8.3f}")

    # 3. The timing-matched test and information beyond covariates.
    from neurocast.audit.timing_matched import info_beyond_covariates, matched_gain

    groups, cov = matched_setup(te)
    print(f"\n  {'decoder':<16} {'unmatched':>10} {'matched':>9} {'surviving':>10} "
          f"{'nats/word beyond covariates':>34}")
    occ_t = te["occ"].to_numpy()
    for k, s in scores.items():
        r = matched_gain(s, yt, occ_t, groups, metric="rank", n_perm=args.n_perm, seed=0)
        b = info_beyond_covariates(cov, s, yt, occ_t, n_boot=args.n_boot, seed=0)
        res[k].update({"matched_gain": r.gain, "unmatched_gain": r.unmatched_gain,
                       "surviving": r.surviving, "verdict": r.verdict,
                       "beyond": {"nats": b.nats, "lo": b.lo, "hi": b.hi}})
        print(f"  {k:<16} {r.unmatched_gain:>+10.4f} {r.gain:>+9.4f} {100 * r.surviving:>9.0f}%  "
              f"{b}", flush=True)
    np.savez_compressed(OUT / "test_scores_sweep.npz", y=yt, **scores)
    (OUT / "decoder_sweep.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n  {time.time() - t0:.0f}s; saved {OUT / 'decoder_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
