"""Acoustic covariates of each spoken word: duration, loudness and envelope.

A decoder can tell words apart by *how long* and *how loud* they are without
knowing which word it is: "a" is short and quiet, "holmes" long and stressed.
The timing-matched test and the information-beyond-covariates test both condition
on these, measured from the actual audiobook audio, so a brain score that
survives them is about word identity rather than its acoustic envelope.

Loudness is the RMS level, in dB, of the chapter audio over the word's interval.
The events' ``timechapter`` column places each word in the chapter file; all
listeners of a session heard the same file, so loudness is a property of the
word *occurrence*.

Envelope features (README section 2.4) describe the sound over the same window a
MEG decoder sees, ``[onset, onset + W)``, in 32 ms bins: broadband level and the
level in 8 log-spaced bands, 100 Hz to 8 kHz. Nothing finer -- a spectrogram
would name the word, which says nothing about the brain. MEG time maps to audio
time through each events file's ``timemeg`` / ``timechapter`` pairs; the audio
played in the scanner is the chapter file with short silences inserted, and a
bin that falls in one reads as silence. NumPy only.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

__all__ = [
    "read_wav",
    "word_loudness_db",
    "chapter_audio_path",
    "BIN_S",
    "BAND_EDGES",
    "band_power",
    "event_times",
    "meg_to_chapter",
    "window_levels",
    "alignment_check",
]

#: Envelope bins: 32 ms, so a 0.512 s window is 16 bins and 1.024 s is 32.
BIN_S = 0.032
#: 8 log-spaced bands, 100 Hz to 8 kHz; ``band_power`` adds a broadband column.
BAND_EDGES = tuple(float(f) for f in np.geomspace(100.0, 8000.0, 9))


def chapter_audio_path(root: Path, task: str, session: str) -> Path:
    """``<root>/<task>/stimuli/audio/<book>_<NN>_doyle_64kb.wav``, chapter NN = session."""
    d = Path(root) / task / "stimuli" / "audio"
    hits = sorted(d.glob(f"*_{int(session):02d}_*.wav"))
    if not hits:
        raise FileNotFoundError(f"no audio for {task} session {session} in {d}")
    return hits[0]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Mono float32 samples in [-1, 1] and the sample rate. 16-bit PCM only."""
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        rate, n_ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(1)
    return x, rate


def word_loudness_db(audio: np.ndarray, rate: int, t_start: np.ndarray,
                     duration: np.ndarray, floor_db: float = -90.0) -> np.ndarray:
    """RMS level in dB over ``[t_start, t_start + duration)`` for each word."""
    csum = np.concatenate([[0.0], np.cumsum(audio.astype(np.float64) ** 2)])
    a = np.clip(np.round(np.asarray(t_start) * rate).astype(int), 0, len(audio))
    b = np.clip(np.round((np.asarray(t_start) + np.asarray(duration)) * rate).astype(int), 0, len(audio))
    n = np.maximum(b - a, 1)
    ms = (csum[b] - csum[a]) / n
    return np.maximum(10.0 * np.log10(np.maximum(ms, 1e-12)), floor_db)


def band_power(audio: np.ndarray, rate: int, *, edges=BAND_EDGES, frame: int = 512,
               hop_s: float = 0.008, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Power of Hann-windowed frames per band, ``(n_frames, n_bands + 1)``, and frame centres (s).

    Columns are the bands between consecutive ``edges``, then broadband (every
    FFT bin). Units are arbitrary but common to all columns and all frames.
    """
    edges = np.asarray(edges, float)
    hop = max(1, int(round(hop_s * rate)))
    n = 1 + max(0, (len(audio) - frame) // hop)
    win = np.hanning(frame).astype(np.float32)
    band = np.searchsorted(edges, np.fft.rfftfreq(frame, 1.0 / rate), side="right") - 1
    nb = len(edges) - 1
    out = np.zeros((n, nb + 1))
    offs = np.arange(frame)
    for s in range(0, n, chunk):
        idx = np.arange(s, min(n, s + chunk))
        p = np.abs(np.fft.rfft(audio[idx[:, None] * hop + offs] * win, axis=1)) ** 2
        for b in range(nb):
            out[idx, b] = p[:, band == b].sum(1)
        out[idx, nb] = p.sum(1)
    return out, (np.arange(n) * hop + frame / 2) / rate


def event_times(events: Path):
    """Every event row with both clocks: ``kind``, ``meg``, ``chapter``, ``duration``, by MEG time."""
    import pandas as pd

    ev = pd.read_csv(events, sep="\t")
    out = pd.DataFrame({
        "kind": ev["kind"].astype(str),
        "meg": pd.to_numeric(ev["timemeg"], errors="coerce"),
        "chapter": pd.to_numeric(ev["timechapter"], errors="coerce"),
        "duration": pd.to_numeric(ev["duration"], errors="coerce"),
    })
    out = out[np.isfinite(out["meg"]) & np.isfinite(out["chapter"])]
    return out.sort_values("meg", kind="stable").reset_index(drop=True)


def meg_to_chapter(t: np.ndarray, meg: np.ndarray, chapter: np.ndarray) -> np.ndarray:
    """Chapter-audio time of each MEG time; NaN inside a silence inserted for presentation.

    Between events the two clocks differ by a constant. Where silence was inserted
    before event ``i + 1``, mapping with event ``i``'s offset runs past event
    ``i + 1``'s chapter time; those instants were silence in the scanner.
    """
    t = np.asarray(t, float)
    i = np.clip(np.searchsorted(meg, t, side="right") - 1, 0, len(meg) - 1)
    c = t - (meg[i] - chapter[i])
    nxt = np.minimum(i + 1, len(meg) - 1)
    c[(i + 1 < len(meg)) & (c > chapter[nxt] + 1e-9)] = np.nan
    return c


def window_levels(power: np.ndarray, centres: np.ndarray, onsets: np.ndarray, window_s: float,
                  meg: np.ndarray, chapter: np.ndarray, *, bin_s: float = BIN_S,
                  floor: float = 1e-12) -> np.ndarray:
    """Mean power in dB over 32 ms bins of ``[onset, onset + window_s)``, ``(n, bins, columns)``.

    Onsets are MEG times. Bins in inserted silence, before the audio starts or past
    its end read as ``floor`` power.
    """
    n_bins = int(round(window_s / bin_s))
    t = np.asarray(onsets, float)[:, None] + (np.arange(n_bins) + 0.5) * bin_s
    c = meg_to_chapter(t.ravel(), meg, chapter)
    ok = np.isfinite(c)
    lo = np.where(ok, c - bin_s / 2, -1.0)
    i0 = np.searchsorted(centres, lo)
    i1 = np.searchsorted(centres, np.where(ok, c + bin_s / 2, -1.0))
    cs = np.vstack([np.zeros((1, power.shape[1])), np.cumsum(power, 0)])
    count = (i1 - i0)[:, None]
    mean = np.where(count > 0, (cs[i1] - cs[i0]) / np.maximum(count, 1), floor)
    mean[~ok] = floor
    return (10.0 * np.log10(np.maximum(mean, floor))).reshape(len(t), n_bins, -1).astype(np.float32)


def alignment_check(audio: np.ndarray, rate: int, ev) -> dict:
    """Is this audio the chapter the events describe? Word minus silence level, and overrun.

    ``ev`` is :func:`event_times`. Silences shorter than 0.2 s are skipped (their
    edges blur into speech). A wrong chapter gives a contrast near 0 dB.
    """
    words = ev[ev["kind"] == "word"]
    sil = ev[(ev["kind"] == "silence") & (ev["duration"] >= 0.2)]
    lw = word_loudness_db(audio, rate, words["chapter"].to_numpy(), words["duration"].to_numpy())
    ls = word_loudness_db(audio, rate, sil["chapter"].to_numpy(), sil["duration"].to_numpy())
    end = float((words["chapter"] + words["duration"]).max()) if len(words) else 0.0
    return {"contrast_db": float(np.mean(lw) - np.mean(ls)) if len(ls) and len(lw) else float("nan"),
            "word_level_db": float(np.median(lw)) if len(lw) else float("nan"),
            "overrun_s": end - len(audio) / rate, "n_words": int(len(words)),
            "n_silences": int(len(sil))}
