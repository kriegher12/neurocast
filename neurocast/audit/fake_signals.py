"""Fake signals that carry exactly one kind of information. The timing control.

The phase-randomised and Gaussian surrogates in :mod:`neurocast.audit.surrogates`
destroy everything a decoder could use. On LibriBrain that is not the dangerous
case. The dangerous case is a decoder that reads **when words happen** rather
than **which word**: a window starting at a word contains the evoked responses
to the next words too, and their timing depends on this word's length.

:func:`onset_only` builds a signal that carries word timing and nothing else:

    fake_i  =  sum_k  ERF(t - (onset_k - onset_i))   +   remainder_j

* ``ERF`` is the recording's **average** word-onset response -- one waveform per
  channel, identical for every word, so it says *when*, never *which*;
* the sum runs over every word whose response overlaps trial ``i``'s window,
  placed at the true onsets;
* ``remainder_j = window_j - (its own timing template)`` from a random other
  trial ``j`` of the same recording supplies realistic noise, unrelated to ``i``.

Refit a decoder on these and whatever it scores above chance is timing. The lost
version of this project found it reproduced 21-53% of a linear decoder's gain.
"""

from __future__ import annotations

import numpy as np

__all__ = ["average_response", "timing_template", "onset_only"]


def average_response(windows: np.ndarray) -> np.ndarray:
    """``(n, C, T)`` -> ``(C, T)`` mean word-onset response."""
    return windows.astype(np.float32).mean(axis=0)


def timing_template(erf: np.ndarray, onset: float, all_onsets: np.ndarray,
                    sfreq: float) -> np.ndarray:
    """The ERF summed at every word onset overlapping the window starting at ``onset``."""
    c, t = erf.shape
    win_s = t / sfreq
    out = np.zeros((c, t), dtype=np.float32)
    d = np.asarray(all_onsets, float) - onset
    for dk in d[(d > -win_s) & (d < win_s)]:
        shift = int(round(dk * sfreq))
        if shift >= 0:
            out[:, shift:] += erf[:, : t - shift]
        else:
            out[:, : t + shift] += erf[:, -shift:]
    return out


def onset_only(
    windows: np.ndarray,
    onsets: np.ndarray,
    run_ids: np.ndarray,
    run_onsets: dict,
    sfreq: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Timing-only versions of ``windows`` (n, C, T), one recording at a time.

    ``run_onsets[run_id]`` holds the onsets of **all** words in that recording,
    not only the 50 task words, since every word's response lands in the window.
    """
    out = np.empty(windows.shape, dtype=np.float16)
    for r in np.unique(run_ids):
        m = np.flatnonzero(run_ids == r)
        w = windows[m].astype(np.float32)
        erf = average_response(w)
        tmpl = np.stack([timing_template(erf, onsets[i], run_onsets[r], sfreq) for i in m])
        remainder = w - tmpl
        donor = rng.permutation(len(m))
        same = donor == np.arange(len(m))
        donor[same] = (donor[same] + 1) % len(m)          # never a trial's own noise
        out[m] = (tmpl + remainder[donor]).astype(np.float16)
    return out
