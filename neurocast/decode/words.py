"""The 50-word task on LibriBrain, and the speech-timing features that leak into it.

Task, as in the lost version's configuration reference: classify which of the
50 most frequent words was heard, from a window starting at the word's onset.
Fit on Sherlock1 session 11 (chapter 11), test on session 12 (chapter 12), so no
audio is shared between the two.

Why timing features live next to the task
-----------------------------------------
A window that starts at a word also contains the brain's response to the *next*
words, and when they arrive is not independent of which word this one is: after
"a" or "the" the next word comes sooner than after "holmes". So the timing of the
surrounding words is information about the label that needs no brain at all --
and the competition's test data provides word onsets. :func:`timing_features`
computes exactly what a window starting at the onset could expose: the onsets of
the next four words *inside the window*, the silent gap since the previous word,
and the position in the sentence.

Trials are per listener, but a word *occurrence* -- (task, session, sentence,
word index) -- is shared by every listener of that session. ``occ`` identifies it;
permutations and cross-fitting must group by it.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..data.libribrain import Run, read_words

__all__ = [
    "WINDOW_S",
    "N_NEXT",
    "TIMING_COLUMNS",
    "word_table",
    "vocabulary",
    "timing_features",
    "label_trials",
]

#: Analysis window: 128 samples at 250 Hz.
WINDOW_S = 0.512
#: Upcoming words whose onsets can fall inside the window.
N_NEXT = 4
TIMING_COLUMNS = [f"next{k}" for k in range(1, N_NEXT + 1)] + ["gap_prev", "pos_in_sentence"]


def timing_features(words, window_s: float = WINDOW_S):
    """Add timing columns to one run's word table (all words, in order).

    ``nextK`` is the onset of the K-th following word relative to this onset,
    **NaN when it falls outside the window** -- a window cannot reveal a word that
    starts after it ends. ``gap_prev`` is the silence since the previous word
    ended (NaN at the start of a run). ``pos_in_sentence`` is the word index.
    """
    w = words.copy()
    on = w["onset"].to_numpy()
    for k in range(1, N_NEXT + 1):
        nxt = np.full(len(on), np.nan)
        nxt[:-k] = on[k:] - on[:-k]
        nxt[nxt >= window_s] = np.nan
        w[f"next{k}"] = nxt
    prev_end = np.r_[np.nan, (on + w["duration"].to_numpy())[:-1]]
    w["gap_prev"] = np.maximum(on - prev_end, 0.0)
    w["pos_in_sentence"] = w["widx"].astype(float)
    w["prev_word"] = w["word"].shift(1).fillna("<s>")
    return w


def word_table(runs: Iterable[Run], *, window_s: float = WINDOW_S):
    """All words of the given runs, with timing features and identifiers."""
    import pandas as pd

    parts = []
    for r in runs:
        w = timing_features(read_words(r.events), window_s)
        w.insert(0, "subject", r.subject)
        w.insert(1, "session", r.session)
        w.insert(2, "task", r.task)
        w["occ"] = (r.task + "/" + r.session + "/" + w["sentence"].astype(str) + "/"
                    + w["widx"].astype(str))
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def vocabulary(words, n: int = 50) -> list[str]:
    """The ``n`` most frequent words, counted once per occurrence (not per listener).

    Counting per listener would weight the start of the chapter -- which the
    partial-session listeners also heard -- above its end.
    """
    once = words.drop_duplicates("occ")
    counts = once["word"].value_counts()
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:n]]


def label_trials(words, vocab: list[str]):
    """Rows whose word is in ``vocab``, with an integer ``label`` column."""
    index = {w: i for i, w in enumerate(vocab)}
    t = words[words["word"].isin(index)].copy()
    t["label"] = t["word"].map(index).astype(int)
    return t.reset_index(drop=True)
