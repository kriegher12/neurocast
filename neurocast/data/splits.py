"""Split construction with enforced disjointness.

Three leakage vectors this module exists to close, all documented in the
literature and all still common:

1. **Temporal autocorrelation.** Random or adjacent-window train/test splits let
   a classifier exploit signal continuity rather than task structure
   (arXiv 2405.17024). Fix: contiguous blocks plus a guard band wide enough that
   train and test share no slow drift.

2. **Subject identity.** Cross-subject claims require subject-disjoint splits.
   Without them, "generalisation" can be subject memorisation -- the Identity
   Trap result (frozen subject-variance 13-89x a random null).

3. **Stimulus memorisation.** The under-appreciated one. Every LibriBrain
   subject hears the same Sherlock Holmes text. A model that learns *position in
   the audiobook* can predict the next word with zero brain signal. Fix:
   hold out whole books, and always report the stimulus-only baseline.

The module is deliberately paranoid. Every split carries a fingerprint over the
exact sample ranges it covers, and :func:`assert_disjoint` fails loudly on any
overlap. A guard band is required, not optional -- passing ``guard=0`` is
allowed but must be explicit, so it shows up in review.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = [
    "Segment",
    "Split",
    "blocked_time_split",
    "subject_split",
    "stimulus_split",
    "assert_disjoint",
    "assert_fixed_duration",
    "DEFAULT_GUARD_SECONDS",
]

#: Guard band between train and eval segments. 60 s is the project default,
#: chosen so that prompt and query cannot share slow drift. It is also the
#: guard the episodic sampler enforces between a subject prompt and its query.
DEFAULT_GUARD_SECONDS = 60.0


@dataclass(frozen=True, order=True)
class Segment:
    """A contiguous span of one continuous recording, in sample indices."""

    subject: str
    session: str
    run: str
    start: int
    stop: int  # exclusive
    stimulus_id: str | None = None

    def __post_init__(self) -> None:
        if self.stop <= self.start:
            raise ValueError(f"empty or inverted segment: [{self.start}, {self.stop})")
        if self.start < 0:
            raise ValueError(f"negative start index: {self.start}")

    @property
    def key(self) -> tuple[str, str, str]:
        """Identifies the continuous recording this segment lives in."""
        return (self.subject, self.session, self.run)

    @property
    def n_samples(self) -> int:
        return self.stop - self.start

    def overlaps(self, other: "Segment", guard: int = 0) -> bool:
        """True if the two spans touch, after expanding each by ``guard``."""
        if self.key != other.key:
            return False
        return self.start - guard < other.stop and other.start - guard < self.stop


@dataclass(frozen=True)
class Split:
    """A named collection of segments, with a content fingerprint."""

    name: str
    segments: tuple[Segment, ...]

    @property
    def subjects(self) -> frozenset[str]:
        return frozenset(s.subject for s in self.segments)

    @property
    def stimuli(self) -> frozenset[str]:
        return frozenset(s.stimulus_id for s in self.segments if s.stimulus_id)

    @property
    def n_samples(self) -> int:
        return sum(s.n_samples for s in self.segments)

    def fingerprint(self) -> str:
        """Stable SHA-256 over the exact sample ranges covered.

        Log this next to every result. Two runs claiming the same split must
        produce the same fingerprint, which turns "did we evaluate on the same
        data?" from a question of trust into a string comparison.
        """
        payload = json.dumps(
            sorted(
                [s.subject, s.session, s.run, s.start, s.stop, s.stimulus_id or ""]
                for s in self.segments
            ),
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def summary(self) -> str:
        return (
            f"{self.name:<12} {len(self.segments):5d} segments  "
            f"{len(self.subjects):3d} subjects  "
            f"{self.n_samples:10d} samples  fp={self.fingerprint()}"
        )


def _group(segments: Iterable[Segment]) -> dict[tuple[str, str, str], list[Segment]]:
    groups: dict[tuple[str, str, str], list[Segment]] = defaultdict(list)
    for s in segments:
        groups[s.key].append(s)
    for v in groups.values():
        v.sort(key=lambda s: s.start)
    return groups


def blocked_time_split(
    segments: Sequence[Segment],
    fractions: dict[str, float],
    *,
    guard_samples: int,
) -> dict[str, Split]:
    """Split each recording into contiguous blocks with guard bands between them.

    Blocks are allocated in order, so the eval block is always temporally after
    the train block within a recording -- never interleaved. The guard band is
    carved out of the *end* of each block, so no sample within ``guard_samples``
    of a boundary appears in any split.

    Parameters
    ----------
    fractions
        Ordered mapping of split name to fraction, e.g.
        ``{"train": 0.8, "val": 0.1, "test": 0.1}``. Must sum to 1.
    guard_samples
        Samples discarded at each internal boundary. Required, not optional --
        pass 0 explicitly if you genuinely want none, so it is visible in review.
    """
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {total}")
    if guard_samples < 0:
        raise ValueError("guard_samples must be non-negative")

    out: dict[str, list[Segment]] = {name: [] for name in fractions}

    for _key, segs in _group(segments).items():
        span_start = segs[0].start
        span_stop = segs[-1].stop
        length = span_stop - span_start
        names = list(fractions)
        cursor = span_start

        for i, name in enumerate(names):
            is_last = i == len(names) - 1
            block_len = length - (cursor - span_start) if is_last else int(length * fractions[name])
            block_stop = cursor + block_len
            usable_stop = block_stop if is_last else block_stop - guard_samples

            if usable_stop > cursor:
                # Intersect the block with the actual (possibly gappy) segments,
                # so we never invent samples that were not recorded.
                for s in segs:
                    lo, hi = max(s.start, cursor), min(s.stop, usable_stop)
                    if hi > lo:
                        out[name].append(
                            Segment(s.subject, s.session, s.run, lo, hi, s.stimulus_id)
                        )
            cursor = block_stop

    return {name: Split(name, tuple(sorted(v))) for name, v in out.items()}


def subject_split(
    segments: Sequence[Segment], assignment: dict[str, str]
) -> dict[str, Split]:
    """Partition by subject. ``assignment`` maps subject id -> split name.

    Any subject missing from ``assignment`` raises, rather than being silently
    dropped -- a silently dropped subject is how eval sets quietly shrink.
    """
    missing = {s.subject for s in segments} - set(assignment)
    if missing:
        raise ValueError(f"no split assigned for subjects: {sorted(missing)}")
    out: dict[str, list[Segment]] = defaultdict(list)
    for s in segments:
        out[assignment[s.subject]].append(s)
    return {name: Split(name, tuple(sorted(v))) for name, v in out.items()}


def stimulus_split(
    segments: Sequence[Segment], assignment: dict[str, str]
) -> dict[str, Split]:
    """Partition by stimulus id (e.g. hold out whole books).

    This is the split most often omitted. When every subject hears the same
    material, a train/test split that shares stimuli lets the model memorise the
    text rather than decode the brain.
    """
    unlabelled = [s for s in segments if s.stimulus_id is None]
    if unlabelled:
        raise ValueError(
            f"{len(unlabelled)} segments have no stimulus_id; cannot split by stimulus"
        )
    missing = {s.stimulus_id for s in segments} - set(assignment)
    if missing:
        raise ValueError(f"no split assigned for stimuli: {sorted(missing)}")
    out: dict[str, list[Segment]] = defaultdict(list)
    for s in segments:
        out[assignment[s.stimulus_id]].append(s)
    return {name: Split(name, tuple(sorted(v))) for name, v in out.items()}


def assert_disjoint(
    splits: dict[str, Split],
    *,
    guard_samples: int = 0,
    subject_disjoint: bool = False,
    stimulus_disjoint: bool = False,
) -> None:
    """Fail loudly if any two splits share samples, subjects, or stimuli.

    Raises
    ------
    AssertionError
        With the specific offending pair named, because a split bug you cannot
        localise is a split bug you will not fix.
    """
    names = list(splits)
    for i, a_name in enumerate(names):
        for b_name in names[i + 1 :]:
            a, b = splits[a_name], splits[b_name]

            by_key = _group(b.segments)
            for sa in a.segments:
                for sb in by_key.get(sa.key, ()):
                    if sa.overlaps(sb, guard=guard_samples):
                        raise AssertionError(
                            f"sample overlap between '{a_name}' and '{b_name}' "
                            f"in {sa.key}: [{sa.start},{sa.stop}) vs "
                            f"[{sb.start},{sb.stop}) with guard={guard_samples}"
                        )

            if subject_disjoint and (shared := a.subjects & b.subjects):
                raise AssertionError(
                    f"subject overlap between '{a_name}' and '{b_name}': "
                    f"{sorted(shared)}"
                )
            if stimulus_disjoint and (shared := a.stimuli & b.stimuli):
                raise AssertionError(
                    f"stimulus overlap between '{a_name}' and '{b_name}': "
                    f"{sorted(shared)}"
                )


def assert_fixed_duration(windows: Sequence[Segment], expected: int) -> None:
    """Every evaluation window must be exactly ``expected`` samples long.

    Variable-length windows are how signal-blind Gaussian noise reached 66.3%
    Rank@1 in arXiv 2605.24524: window length leaks the label. The PNPL word
    task uses a fixed 0.2-0.6 s window (100 samples at 250 Hz), and this
    assertion holds callers to it.
    """
    bad = [w for w in windows if w.n_samples != expected]
    if bad:
        raise AssertionError(
            f"{len(bad)}/{len(windows)} windows are not {expected} samples "
            f"(e.g. {bad[0].n_samples} in {bad[0].key}); variable-length windows "
            f"leak the label"
        )
