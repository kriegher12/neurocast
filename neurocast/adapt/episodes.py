"""Episodic sampling: the subject as a prompt, not a fine-tune.

H3 claims that per-subject adaptation belongs in the context window rather than
in the weights. Training for that means constructing episodes the way inference
will see them: K minutes of *this subject's own data* as context, a query window
drawn from elsewhere in their recording, and no gradient step in between.

Four choices here are deliberate, and each closes a specific failure.

**Guard bands (>= 60 s).** The single most likely way to manufacture a
spectacular H3 result. Sample a prompt from 10 seconds before the query and the
two share slow drift, head position, and alpha state; the model "adapts to the
subject" by reading an autocorrelated neighbour. Enforced in the sampler and
asserted independently in ``scripts/validate_adapt.py`` -- the sampler is not
trusted to police itself.

**K = 0 with p = 0.1.** Prompt-free episodes give an unconditional model for
free, and make the value of context directly measurable as
``delta log p = log p(x | C_K) - log p(x | empty)``. That quantity is
``I(x; C_K)`` -- the mutual information between a query and K minutes of the
subject's own data -- which is an Atlas axis obtained as a by-product.

**Cross-session prompts 30% of the time.** Otherwise the model learns "match
this session's noise floor", which collapses the moment a subject returns on a
different day. Session drift is the thing a deployed BCI actually faces.

**Mismatched-prompt negatives.** Without them the model will happily ignore a
decorative prompt and the whole comparison is untested. Pairing a subject's
prompt with another subject's query, under a margin loss, *forces* the prompt to
be used -- and the resulting margin doubles as a subject-identification score,
which is itself a leakage measurement.

K is sampled log-uniformly over 30 s to 40 min so the model sees the whole
adaptation curve, which is exactly the axis H3 is evaluated on
(K in {0, 10, 20, 40} minutes against fine-tuning at matched minutes).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..data.splits import DEFAULT_GUARD_SECONDS, Segment

__all__ = ["Episode", "EpisodeSampler"]


@dataclass(frozen=True)
class Episode:
    """One training example: a prompt, a query, and how they relate."""

    query: Segment
    prompt: tuple[Segment, ...]
    prompt_subject: str
    prompt_minutes: float
    cross_session: bool
    mismatched: bool

    @property
    def query_subject(self) -> str:
        return self.query.subject

    @property
    def is_prompt_free(self) -> bool:
        return len(self.prompt) == 0

    def describe(self) -> str:
        kind = (
            "prompt-free"
            if self.is_prompt_free
            else ("MISMATCHED" if self.mismatched else "matched")
        )
        return (
            f"{kind:<11} query={self.query_subject}/{self.query.session} "
            f"prompt={self.prompt_subject} "
            f"K={self.prompt_minutes:5.1f}min "
            f"{'cross' if self.cross_session else 'same '}-session "
            f"({len(self.prompt)} chunks)"
        )


class EpisodeSampler:
    """Draws prompt/query episodes with enforced guard bands.

    Parameters
    ----------
    segments
        The pool to sample from. Should be a single split -- never mix train and
        eval material here.
    fs
        Sample rate, for converting seconds to sample indices.
    query_samples
        Query window length. 4096 samples = 16.4 s at 250 Hz.
    guard_seconds
        Minimum separation between any prompt chunk and the query, within the
        same recording. Cross-session prompts are exempt by construction.
    chunk_seconds
        Prompt granularity. 8 s matches the summariser's compression unit.
    """

    def __init__(
        self,
        segments: Sequence[Segment],
        *,
        fs: float = 250.0,
        query_samples: int = 4096,
        guard_seconds: float = DEFAULT_GUARD_SECONDS,
        chunk_seconds: float = 8.0,
        min_minutes: float = 0.5,
        max_minutes: float = 40.0,
        p_prompt_free: float = 0.10,
        p_cross_session: float = 0.30,
        p_mismatch: float = 0.15,
    ) -> None:
        if not segments:
            raise ValueError("no segments to sample from")
        if not 0.0 <= p_prompt_free <= 1.0 or not 0.0 <= p_mismatch <= 1.0:
            raise ValueError("probabilities must lie in [0, 1]")
        if min_minutes <= 0 or max_minutes < min_minutes:
            raise ValueError("require 0 < min_minutes <= max_minutes")

        self.fs = float(fs)
        self.query_samples = int(query_samples)
        self.guard = int(round(guard_seconds * fs))
        self.chunk = int(round(chunk_seconds * fs))
        self.min_minutes = float(min_minutes)
        self.max_minutes = float(max_minutes)
        self.p_prompt_free = float(p_prompt_free)
        self.p_cross_session = float(p_cross_session)
        self.p_mismatch = float(p_mismatch)

        self.by_subject: dict[str, list[Segment]] = defaultdict(list)
        for s in segments:
            if s.n_samples >= self.query_samples:
                self.by_subject[s.subject].append(s)
        if not self.by_subject:
            raise ValueError(
                f"no segment is long enough for a {self.query_samples}-sample query"
            )
        self.subjects = sorted(self.by_subject)

    def available_minutes(self, subject: str) -> float:
        total = sum(s.n_samples for s in self.by_subject[subject])
        return total / self.fs / 60.0

    def _chunks(self, seg: Segment) -> list[Segment]:
        """Split a segment into fixed-length prompt chunks."""
        out = []
        for start in range(seg.start, seg.stop - self.chunk + 1, self.chunk):
            out.append(
                Segment(seg.subject, seg.session, seg.run, start,
                        start + self.chunk, seg.stimulus_id)
            )
        return out

    def _sample_query(self, subject: str, rng: np.random.Generator) -> Segment:
        segs = self.by_subject[subject]
        weights = np.array([s.n_samples - self.query_samples + 1 for s in segs], float)
        seg = segs[int(rng.choice(len(segs), p=weights / weights.sum()))]
        start = int(rng.integers(seg.start, seg.stop - self.query_samples + 1))
        return Segment(
            seg.subject, seg.session, seg.run, start,
            start + self.query_samples, seg.stimulus_id,
        )

    def _eligible_chunks(
        self, subject: str, query: Segment, cross_session: bool
    ) -> list[Segment]:
        """Chunks that may legally serve as prompt material for this query.

        A chunk is eligible if it is in a different recording from the query, or
        it is in the same recording and separated by at least the guard band.
        """
        out: list[Segment] = []
        for seg in self.by_subject[subject]:
            same_recording = seg.key == query.key
            if cross_session and seg.session == query.session:
                continue
            for c in self._chunks(seg):
                if same_recording and c.overlaps(query, guard=self.guard):
                    continue
                out.append(c)
        return out

    def sample(self, rng: np.random.Generator) -> Episode:
        """Draw one episode."""
        prompt_subject = str(rng.choice(self.subjects))

        # K, log-uniform, capped so a prompt never consumes more than half the
        # subject's material (the rest has to remain usable as queries).
        k_min = float(
            math.exp(
                rng.uniform(math.log(self.min_minutes), math.log(self.max_minutes))
            )
        )
        k_min = min(k_min, self.available_minutes(prompt_subject) * 0.5)
        if rng.random() < self.p_prompt_free:
            k_min = 0.0

        cross_session = bool(rng.random() < self.p_cross_session)

        # Mismatched negatives: the query comes from a different subject, so the
        # prompt is actively wrong rather than merely uninformative.
        mismatched = bool(
            rng.random() < self.p_mismatch and len(self.subjects) > 1
        )
        if mismatched:
            others = [s for s in self.subjects if s != prompt_subject]
            query_subject = str(rng.choice(others))
        else:
            query_subject = prompt_subject

        query = self._sample_query(query_subject, rng)

        if k_min <= 0.0:
            return Episode(query, (), prompt_subject, 0.0, cross_session, mismatched)

        eligible = self._eligible_chunks(prompt_subject, query, cross_session)
        if not eligible and cross_session:
            # Single-session subject: fall back to same-session with the guard
            # band still enforced, rather than silently dropping the guard.
            cross_session = False
            eligible = self._eligible_chunks(prompt_subject, query, False)
        if not eligible:
            return Episode(query, (), prompt_subject, 0.0, cross_session, mismatched)

        n_want = max(1, int(round(k_min * 60.0 * self.fs / self.chunk)))
        n_take = min(n_want, len(eligible))
        idx = rng.choice(len(eligible), size=n_take, replace=False)
        chunks = tuple(sorted(eligible[int(i)] for i in idx))
        actual_minutes = n_take * self.chunk / self.fs / 60.0

        return Episode(
            query, chunks, prompt_subject, actual_minutes, cross_session, mismatched
        )

    def sample_many(self, n: int, rng: np.random.Generator) -> list[Episode]:
        return [self.sample(rng) for _ in range(n)]

    def swap_prompts(self, a: Episode, b: Episode) -> tuple[Episode, Episode]:
        """Exchange two episodes' prompts, keeping their queries.

        Backs the prompt-swap experiment, a headline result candidate: if
        decoding survives the swap while a subject-identity probe follows the
        *swapped prompt* rather than the query data, that is causal evidence of
        separation between measurement statistics and cognition -- an
        intervention, not a correlation.
        """
        return (
            Episode(a.query, b.prompt, b.prompt_subject, b.prompt_minutes,
                    b.cross_session, a.query.subject != b.prompt_subject),
            Episode(b.query, a.prompt, a.prompt_subject, a.prompt_minutes,
                    a.cross_session, b.query.subject != a.prompt_subject),
        )
