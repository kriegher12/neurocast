"""H5 rehearsal: pretrain on MEG, decode on an EEG cap the model has never seen.

H5 claims a geometry-conditioned model learns something about the *brain* rather
than about one sensor array, so it should transfer to a different array -- with
different physics -- without retraining. The Sept-2026 MEG roadmap frames the
open question as whether representations are "truly about cognition or merely
measurement statistics of a particular modality, sensor array and preprocessing
pipeline."

Setup (source-space corpus, so MEG and EEG genuinely record the same sources):

  A  transferred    pretrain on MEG, retarget to EEG, ZERO gradient steps
  B  random init    same architecture, untrained, retargeted  (random features)
  C  from scratch   trained on EEG with a small matched budget
  D  raw sensors    no model: per-channel evoked-window means

All four are read out by the same frozen nearest-centroid probe, trained on a few
labelled EEG trials and tested on held-out EEG trials from **held-out subjects**
-- so subject identity cannot be the shortcut.

Gate (from the plan): A must beat C. A beating B shows pretraining matters beyond
the architecture's inductive bias.

This is a rehearsal on synthetic data with a CPU-sized model. It exercises the
experiment end to end; it is not evidence for H5.

    .venv/Scripts/python.exe scripts/run_transfer.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurocast.data.sources import SourceSpaceCorpus  # noqa: E402
from neurocast.models.neurocast import NeuroCast  # noqa: E402
from neurocast.train.generative import train_generative  # noqa: E402
from validate_tokenizer import eeg_montage, megin_montage  # noqa: E402

PATCH = 16
N_PATCH = 32
TRAIN_SUBJECTS = [0, 1, 2, 3, 4, 5]
TEST_SUBJECTS = [6, 7]


def evoked_window(fs: float, n_times: int) -> slice:
    """Patches covering 50-300 ms after onset, where the label lives."""
    onset = 0.2 * n_times
    lo = int((onset + 0.05 * fs) // PATCH)
    hi = int((onset + 0.30 * fs) // PATCH) + 1
    return slice(lo, hi)


@torch.no_grad()
def features(model: NeuroCast, x: np.ndarray, win: slice) -> np.ndarray:
    """Frozen group tokens across the evoked window, per patch -> (n, P*G*d).

    Per patch, NOT averaged over the window. The evoked response is an 8 Hz
    oscillation; averaging it over a 250 ms window cancels the sinusoid and erases
    the label. A first version did exactly that, and every method -- including raw
    sensors -- landed at chance, making the comparison powerless.
    """
    model.eval()
    out = []
    for i in range(0, len(x), 8):
        g = model.groups(torch.as_tensor(x[i:i + 8], dtype=torch.float32))
        out.append(g[:, win].reshape(g.shape[0], -1).numpy())
    return np.concatenate(out)


def per_patch(arr: np.ndarray, win: slice) -> np.ndarray:
    """Per-patch means of (n, K, T) signal across the window -> (n, K * P)."""
    seg = arr[:, :, win.start * PATCH: win.stop * PATCH]
    n, k, t = seg.shape
    return seg.reshape(n, k, t // PATCH, PATCH).mean(axis=3).reshape(n, -1)


def probe(f_tr, y_tr, f_te, y_te) -> float:
    """Standardised nearest-centroid, balanced accuracy on the held-out set."""
    mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-8
    a, b = (f_tr - mu) / sd, (f_te - mu) / sd
    classes = np.unique(y_tr)
    cents = np.stack([a[y_tr == c].mean(0) for c in classes])
    pred = classes[np.argmin(((b[:, None] - cents[None]) ** 2).sum(-1), axis=1)]
    return float(np.mean([(pred[y_te == c] == c).mean() for c in np.unique(y_te)]))


def main() -> int:
    t_start = time.time()
    corpus = SourceSpaceCorpus(n_sources=6, n_subjects=8, n_labels=4, seed=0)
    meg, eeg = megin_montage(n_sites=24), eeg_montage(48)
    n_times = N_PATCH * PATCH
    win = evoked_window(corpus.fs, n_times)
    chance = 1.0 / corpus.n_labels

    print("=" * 84)
    print("H5 REHEARSAL -- pretrain on MEG, decode on an unseen EEG cap")
    print("=" * 84)
    print(f"  MEG {len(meg)} ch (magnetic physics)  ->  EEG {len(eeg)} ch (electric physics)")
    print(f"  train subjects {TRAIN_SUBJECTS}, evaluation subjects {TEST_SUBJECTS} (held out)")

    rng = np.random.default_rng(1)
    eeg_tr = corpus.batch(eeg, 48, n_times, rng, subjects=TEST_SUBJECTS)
    eeg_te = corpus.batch(eeg, 160, n_times, rng, subjects=TEST_SUBJECTS)
    print(f"  probe: {len(eeg_tr.label)} labelled EEG trials to fit, "
          f"{len(eeg_te.label)} to test; chance {chance:.2f}")

    print()
    print("  A. pretraining on MEG ...")
    gm = train_generative(corpus, meg, n_patch=N_PATCH, steps=320, rank=16,
                          subjects=TRAIN_SUBJECTS, seed=0, verbose=False)
    print(f"     NLL {gm.initial_loss:+.3f} -> {gm.final_loss:+.3f} in {gm.seconds:.0f}s")
    n_params = gm.model.n_parameters()
    gm.model.retarget(eeg)
    added = gm.model.n_parameters() - n_params
    acc_a = probe(features(gm.model, eeg_tr.x, win), eeg_tr.label,
                  features(gm.model, eeg_te.x, win), eeg_te.label)

    print("  B. random-init model ...")
    torch.manual_seed(123)
    rand = NeuroCast(eeg, rung="tiny", window=256)
    acc_b = probe(features(rand, eeg_tr.x, win), eeg_tr.label,
                  features(rand, eeg_te.x, win), eeg_te.label)

    scratch_steps = 80
    print(f"  C. from scratch on EEG, {scratch_steps} steps ...")
    sc = train_generative(corpus, eeg, n_patch=N_PATCH, steps=scratch_steps, rank=16,
                          subjects=TRAIN_SUBJECTS, seed=1, verbose=False)
    acc_c = probe(features(sc.model, eeg_tr.x, win), eeg_tr.label,
                  features(sc.model, eeg_te.x, win), eeg_te.label)

    acc_d = probe(per_patch(eeg_tr.x, win), eeg_tr.label,
                  per_patch(eeg_te.x, win), eeg_te.label)
    # Oracle: the true latent sources. Establishes whether this experiment can
    # separate methods at all. If the oracle is near chance, the task is too hard
    # and no ordering below it means anything.
    acc_o = probe(per_patch(eeg_tr.sources, win), eeg_tr.label,
                  per_patch(eeg_te.sources, win), eeg_te.label)

    print()
    print("=" * 84)
    print("RESULTS  (balanced accuracy on held-out EEG subjects)")
    print("=" * 84)
    rows = [
        ("oracle: true latent sources (task ceiling)", acc_o),
        ("A  transferred from MEG, 0 EEG gradient steps", acc_a),
        ("B  random-init architecture", acc_b),
        (f"C  from scratch on EEG, {scratch_steps} steps", acc_c),
        ("D  raw sensor means", acc_d),
    ]
    for name, acc in rows:
        bar = "#" * int(round(40 * acc))
        print(f"  {name:<48} {acc:.3f}  {bar}")
    print(f"  {'chance':<48} {chance:.3f}")
    print()
    print(f"  parameters added when retargeting MEG -> EEG: {added}")

    se = float(np.sqrt(0.25 * 0.75 / len(eeg_te.label)))
    powered = acc_o > chance + 4 * se
    gate = acc_a > acc_c
    print()
    print(f"  approximate SE of each accuracy: +-{se:.3f}")
    print(f"  experiment has power (oracle > chance + 4 SE): {'YES' if powered else 'NO'}")
    if not powered:
        print("  -> the task is too hard to separate methods; the gate below is uninformative.")
    print(f"  H5 gate (A > C): {'MET' if gate else 'NOT MET'}   A-C = {acc_a - acc_c:+.3f} "
          f"({(acc_a - acc_c) / (se * 2 ** 0.5):+.1f} SE of a difference)")
    print(f"  pretraining vs architecture alone (A - B): {acc_a - acc_b:+.3f}")
    print()
    print("  This is a rehearsal: a 1.3M-parameter model, a few hundred CPU steps, and")
    print("  synthetic sources. It demonstrates that the experiment runs end to end with")
    print("  zero added parameters on a new array -- not that H5 holds on real MEG/EEG.")
    print(f"\n  wall clock {time.time() - t_start:.0f}s")
    return 0 if added == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
