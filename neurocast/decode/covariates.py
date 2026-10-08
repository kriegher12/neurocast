"""Acoustic covariates of each spoken word: duration and loudness.

A decoder can tell words apart by *how long* and *how loud* they are without
knowing which word it is: "a" is short and quiet, "holmes" long and stressed.
The timing-matched test and the information-beyond-covariates test both condition
on these, measured from the actual audiobook audio, so a brain score that
survives them is about word identity rather than its acoustic envelope.

Loudness is the RMS level, in dB, of the chapter audio over the word's interval.
The events' ``timechapter`` column places each word in the chapter file; all
listeners of a session heard the same file, so loudness is a property of the
word *occurrence*.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

__all__ = ["read_wav", "word_loudness_db", "chapter_audio_path"]


def chapter_audio_path(root: Path, task: str, session: str) -> Path:
    """``<root>/<task>/stimuli/audio/studyinscarlet_<session>_doyle_64kb.wav`` for Sherlock1."""
    d = Path(root) / task / "stimuli" / "audio"
    hits = sorted(d.glob(f"*_{int(session)}_*.wav"))
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
