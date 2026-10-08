"""Why does the phase-scrambled control sit above chance? A decomposition.

The audit's phase scramble (``audit.surrogates.phase_randomized_surrogate``,
multivariate) randomises the Fourier phases of each 0.512 s word window, one
draw shared by all channels. It keeps, per trial, the window mean of every
channel (the DC bin), the power spectrum and the cross-spectrum. On LibriBrain
the linear decoder reads it at BAcc@10 0.236 -- above the shuffled-label p95
(0.209) and above the pitch's chart (~0.217). Both kept quantities are
onset-locked: the evoked field shifts each window's mean, and evoked power
raises its low-frequency spectrum. This script refits the same linear decoder on
inputs that keep one at a time:

    as run         multivariate scramble, window means kept (the audit's control)
    no DC          the same, with each window's channel means replaced by the
                   recording's (so only the spectrum survives)
    DC only        each window replaced by its channel means, constant in time
                   (so only the window mean survives)

Reads the windows ``run_libribrain_audit.py --stage brain`` cached on the data
drive; writes ``runs/libribrain_audit/phase_control.json``.

    .venv/Scripts/python.exe scripts/run_phase_control.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocast.audit.surrogates import phase_randomized_surrogate  # noqa: E402
from neurocast.decode.decoders import LinearPooled, patch_features  # noqa: E402
from neurocast.decode.metrics import balanced_accuracy_at_k  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_libribrain_audit import BROAD, OUT, load_split  # noqa: E402


def scrambled(w: np.ndarray, rng, keep_dc: bool, chunk: int = 2000) -> np.ndarray:
    out = []
    for i in range(0, len(w), chunk):
        x = w[i:i + chunk].astype(np.float32)
        s = phase_randomized_surrogate(x, rng)
        if not keep_dc:
            s = s - x.mean(axis=2, keepdims=True)       # recording mean is ~0 after normalisation
        out.append(s.astype(np.float16))
    return np.concatenate(out)


def dc_only(w: np.ndarray, chunk: int = 2000) -> np.ndarray:
    out = []
    for i in range(0, len(w), chunk):
        m = w[i:i + chunk].astype(np.float32).mean(axis=2, keepdims=True)
        out.append(np.broadcast_to(m, w[i:i + chunk].shape).astype(np.float16))
    return np.concatenate(out)


def main() -> int:
    t0 = time.time()
    vocab = json.loads((OUT / "nobrain.json").read_text())["vocab"]
    K = len(vocab)
    w_fit, fit, _ = load_split("11", vocab, BROAD, "session-scalar")
    w_te, te, _ = load_split("12", vocab, BROAD, "session-scalar")
    yf, yt, occ = fit["label"].to_numpy(), te["label"].to_numpy(), fit["occ"].to_numpy()

    def linear(wf, wt):
        m = LinearPooled(seed=0).fit(patch_features(wf), yf, occ)
        s = m.scores(patch_features(wt), K)
        return balanced_accuracy_at_k(s, yt, 1), balanced_accuracy_at_k(s, yt, 10)

    res = {}
    for name, make in (("as_run", lambda w, r: scrambled(w, r, True)),
                       ("no_dc", lambda w, r: scrambled(w, r, False)),
                       ("dc_only", lambda w, r: dc_only(w))):
        rng = np.random.default_rng(0)
        res[name] = linear(make(w_fit, rng), make(w_te, rng))
        print(f"  {name:<8} BAcc@1 {res[name][0]:.3f}  BAcc@10 {res[name][1]:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    brain = json.loads((OUT / "brain.json").read_text())
    gain = brain["linear"][1] - 0.20
    print(f"\n  linear on the real signal: BAcc@10 {brain['linear'][1]:.3f}; "
          f"shuffled-label p95 {brain['shuffled_label_null']['p95'][1]:.3f}")
    for name, v in res.items():
        print(f"  {name:<8} {100 * (v[1] - 0.20) / gain:5.0f}% of the linear gain")
    (OUT / "phase_control.json").write_text(json.dumps(res, indent=2))
    print(f"\n  {time.time() - t0:.0f}s; saved {(OUT / 'phase_control.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
