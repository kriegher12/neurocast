"""LibriBrain100 on disk: runs, word events, and MEG windows.

Reads exactly what ``scripts/data_setup.py`` fetches into
``<data root>/LibriBrain100``, which mirrors the Hugging Face layout::

    <task>/derivatives/events/sub-S_ses-E_task-T_run-R_events.tsv
    <task>/derivatives/serialised/sub-S_ses-E_task-T_run-R_proc-P_meg.h5
    <task>/derivatives/serialised_competition/..._proc-P_desc-firsthalf_meg.h5

The last line is why this module exists alongside pnpl. Twenty of the 32 broad
listeners (sub-13..22 first halves, sub-23..32 first quarters) are published
only under ``serialised_competition/`` with a ``desc-`` suffix that pnpl's
filename builder never generates, so its loader cannot find them -- as the lost
version of this project found. :func:`find_runs` locates every run, whatever
folder or suffix it was published under.

Events files carry words, phonemes and silences with onsets in recording time
(``timemeg``). All listeners of a session heard the same audio, so a word
*occurrence* -- (task, session, sentence, word index) -- is shared across
listeners. That key is what permutations and cross-fitting must respect: shuffle
or split by occurrence, never by trial, or the same spoken word sits on both
sides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import dataset_dir

__all__ = [
    "Run",
    "find_runs",
    "normalize_word",
    "read_words",
    "meg_info",
    "read_meg",
    "normalized_meg",
    "onset_windows",
    "extract_windows",
]

_STEM = re.compile(
    r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_]+)"
)


@dataclass(frozen=True)
class Run:
    """One recording of one listener: events file plus (maybe) its MEG."""

    subject: str
    session: str
    task: str
    run: str
    events: Path
    h5: Path | None
    desc: str          # "full", "firsthalf", "firstquarter"

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.subject, self.session, self.task, self.run)


def find_runs(
    *,
    task: str = "Sherlock1",
    sessions: set[str] | None = None,
    subjects: set[str] | None = None,
    root: Path | None = None,
    require_meg: bool = False,
) -> list[Run]:
    """Every run with an events file, matched to its H5 wherever it was published."""
    base = Path(root) if root is not None else dataset_dir("LibriBrain100")
    h5_by_key: dict[tuple, tuple[Path, str]] = {}
    for folder in ("serialised", "serialised_competition"):
        for h5 in (base / task / "derivatives" / folder).glob("*_meg.h5"):
            m = _STEM.search(h5.name)
            if m:
                d = re.search(r"_desc-([^_]+)_", h5.name)
                h5_by_key[(m["sub"], m["ses"], m["task"], m["run"])] = (h5, d.group(1) if d else "full")
    out = []
    for ev in sorted((base / task / "derivatives" / "events").glob("*_events.tsv")):
        m = _STEM.search(ev.name)
        if not m:
            continue
        key = (m["sub"], m["ses"], m["task"], m["run"])
        if sessions and key[1] not in sessions or subjects and key[0] not in subjects:
            continue
        h5, desc = h5_by_key.get(key, (None, "full"))
        if require_meg and h5 is None:
            continue
        out.append(Run(*key, events=ev, h5=h5, desc=desc))
    return sorted(out, key=lambda r: (r.task, int(r.session), int(r.subject), r.run))


def normalize_word(s: str) -> str:
    """Lowercase, letters and apostrophes only: ``"Holmes,"`` -> ``"holmes"``.

    Curly apostrophes become straight ones first: the Sherlock events spell
    "it's", the 2026 competition vocabulary "it\u2019s".
    """
    s = str(s).replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"[^a-z']", "", s.lower()).strip("'")


def read_words(events: Path):
    """Word rows of one events file, in order, as a pandas DataFrame.

    Columns: ``onset`` and ``duration`` (s, recording time), ``word``
    (normalised), ``sentence`` and ``widx`` (ints; together with task and
    session they identify the occurrence), ``t_sentence`` (s from sentence
    start).
    """
    import pandas as pd

    ev = pd.read_csv(events, sep="\t")
    w = ev[ev["kind"] == "word"].copy()

    def optional(col):  # not every corpus' events carry every column
        if col not in w:
            return np.full(len(w), np.nan)
        return pd.to_numeric(w[col], errors="coerce").to_numpy()

    out = pd.DataFrame({
        "onset": w["timemeg"].astype(float).to_numpy(),
        "duration": w["duration"].astype(float).to_numpy(),
        "word": [normalize_word(s) for s in w["segment"]],
        "sentence": w["sentenceidx"].astype(float).astype(int).to_numpy(),
        "widx": w["wordidx"].astype(float).astype(int).to_numpy(),
        "t_sentence": optional("timesentence"),
        "t_chapter": optional("timechapter"),
    })
    out = out[out["word"] != ""].sort_values("onset", kind="stable").reset_index(drop=True)
    return out


def _split_attr(v) -> list[str]:
    s = v.decode() if isinstance(v, bytes) else str(v)
    return [t.strip() for t in s.split(",") if t.strip()]


def meg_info(h5: Path) -> dict:
    """Shape, rate and channel metadata without reading the signal."""
    import h5py

    with h5py.File(h5, "r") as f:
        n_ch, n_t = f["data"].shape
        sfreq = float(f.attrs["sample_frequency"])
        return {
            "n_channels": int(n_ch), "n_samples": int(n_t), "sfreq": sfreq,
            "duration_s": n_t / sfreq,
            "channel_names": _split_attr(f.attrs.get("channel_names", "")),
            "channel_types": _split_attr(f.attrs.get("channel_types", "")),
        }


def read_meg(h5: Path) -> tuple[np.ndarray, dict]:
    """``(data (C, T) float32 in SI units, info)`` from a pnpl-serialised H5.

    pnpl writes ``data`` (channels x samples, tesla / tesla-per-metre), ``times``,
    and file attributes ``sample_frequency``, ``channel_names``,
    ``channel_types`` (comma-joined). Sample 0 is recording time 0, which is the
    clock ``timemeg`` in the events files uses -- also for the partial
    (first-half / first-quarter) releases, which are cropped at the end.
    """
    import h5py

    info = meg_info(h5)
    with h5py.File(h5, "r") as f:
        x = np.asarray(f["data"][...], dtype=np.float32)
    return x, info


def extract_windows(
    run: Run,
    onsets_s: np.ndarray,
    *,
    n_samples: int = 128,
    decim: int = 2,
    regime: str = "session-scalar",
) -> tuple[np.ndarray, np.ndarray]:
    """Word-onset windows of one recording, normalised, decimated, float16.

    Returns ``(windows, kept)``: ``windows`` is ``(n_kept, C, n_samples // decim)``
    and ``kept`` the boolean mask over ``onsets_s`` of windows that lie fully
    inside the recording. The partial releases end mid-chapter, while their
    events files cover the whole chapter -- this is where those words drop out.

    Pipeline: SI units -> frozen canonical units (INV-1) -> the session
    normalisation ``regime`` (an explicit invertible affine, fitted on this
    whole recording) -> windows from ``floor(onset * sfreq)`` -> mean over
    ``decim``-sample blocks. With ``decim=2`` the 16-sample (64 ms) patch means
    are exactly means over 8 decimated samples.
    """
    x, info = normalized_meg(run, regime)
    win, kept = onset_windows(x, info["sfreq"], onsets_s, n_samples=n_samples, decim=decim)
    return win.astype(np.float16), kept


def normalized_meg(run: Run, regime: str = "session-scalar") -> tuple[np.ndarray, dict]:
    """A whole recording in canonical units under ``regime``: ``(x (C, T) float32, info)``.

    The first half of :func:`extract_windows`, for callers that cut several
    kinds of window from one read -- each read of a chapter from the USB data
    drive costs seconds.
    """
    from .canonical import NormalizationRegime, fit_normalizer, to_canonical

    if run.h5 is None:
        raise FileNotFoundError(f"no MEG for {run.key}")
    x, info = read_meg(run.h5)
    types = info["channel_types"]
    if len(types) != x.shape[0]:
        raise ValueError(f"{run.h5.name}: {len(types)} channel types for {x.shape[0]} channels")
    x, _ = to_canonical(x, types)
    tf = fit_normalizer(x[:, ::4], NormalizationRegime(regime), ch_types=types)
    return tf.apply(x).astype(np.float32), info


def onset_windows(x: np.ndarray, sfreq: float, onsets_s: np.ndarray, *, n_samples: int = 128,
                  decim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Windows of ``x`` from ``floor(onset * sfreq)``, block-mean decimated, float32.

    Returns ``(windows (n_kept, C, n_samples // decim), kept)``; ``kept`` marks
    the onsets whose window lies fully inside the recording.
    """
    start = np.floor(np.asarray(onsets_s, float) * sfreq).astype(int)
    kept = (start >= 0) & (start + n_samples <= x.shape[1])
    idx = start[kept][:, None] + np.arange(n_samples)[None, :]
    win = x[:, idx].transpose(1, 0, 2)                       # (n, C, n_samples)
    if decim > 1:
        win = win.reshape(win.shape[0], win.shape[1], n_samples // decim, decim).mean(-1)
    return win, kept
