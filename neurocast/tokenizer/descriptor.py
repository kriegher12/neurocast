"""Sensor descriptors: what the model knows about a channel before it sees data.

The central design choice of NeuroCast's front-end. A channel is identified by
its **physical properties** -- where it sits, which way it points, what it
measures -- and never by an index into a fixed montage.

Why this matters: a learned per-channel embedding (POYO's per-unit embeddings,
most EEG models' channel tables) must be *fitted* for every new array, and
onboarding a new session means identifying which learned unit each channel
corresponds to. Descriptors remove that step entirely. A never-before-seen OPM
helmet, a 64-channel EEG cap, or a depth electrode gets a deterministic
embedding from its coordinates with **zero new parameters and no identification
procedure**. That is the property H5 rests on.

One fix relative to MEG-XL, which encodes sensor position and orientation but
only the coil *normal*: a planar gradiometer's response depends on the normal
**and** the in-plane baseline direction, and the two co-located planar
gradiometers at a MEGIN site differ *only* in that baseline. Encoding just the
normal makes them indistinguishable except by type id. We carry ``ori_base``.

Quadrants
---------
Each descriptor is assigned to one of four anatomical quadrants (L/R x
anterior/posterior), or to a fifth peripheral group. These drive the spatial
autoregressive factorisation in :mod:`neurocast.tokenizer.perceiver`:

    p(x_t) = prod_g p(x_t^(g) | x_t^(<g), x_<t)

That is only an exact chain-rule factorisation if the quadrants form a genuine
**partition** of the channels -- every channel in exactly one group, no channel
in two. :func:`assert_partition` enforces it, and the likelihood is a real
likelihood rather than a bound only because it holds.

Peripherals are deliberately placed **last** in the AR order, so that
``p(brain_t | ...)`` never conditions on same-timestep peripheral activity. That
ordering is what keeps the artifact-only control meaningful: if the brain
prediction could peek at concurrent EOG, "the model uses eye movement" would be
baked into the factorisation itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

import numpy as np

from ..data.canonical import CANONICAL_SCALE, ChannelType

__all__ = [
    "Quadrant",
    "Reference",
    "SensorDescriptor",
    "Montage",
    "assign_quadrant",
    "assert_partition",
]


class Quadrant(IntEnum):
    """Spatial groups, in autoregressive order. Peripherals go last, by design."""

    LEFT_ANTERIOR = 0
    RIGHT_ANTERIOR = 1
    LEFT_POSTERIOR = 2
    RIGHT_POSTERIOR = 3
    PERIPHERAL = 4

    @classmethod
    def n_brain(cls) -> int:
        return 4

    @classmethod
    def n_all(cls) -> int:
        return 5


class Reference(IntEnum):
    """Referencing scheme. Affects what a channel's value physically means."""

    NONE = 0
    AVERAGE = 1
    MASTOID = 2
    CZ = 3
    BIPOLAR = 4
    COMMON_MEDIAN = 5


#: Channel families that are not brain sensors. These are first-class inputs --
#: tokenised like any other channel -- but they live in the peripheral quadrant.
PERIPHERAL_TYPES: frozenset[ChannelType] = frozenset(
    {
        ChannelType.EOG,
        ChannelType.ECG,
        ChannelType.EMG,
        ChannelType.RESP,
        ChannelType.GSR,
        ChannelType.IMU,
    }
)

#: Families recorded from inside the skull.
INTRACRANIAL_TYPES: frozenset[ChannelType] = frozenset(
    {ChannelType.SEEG, ChannelType.ECOG}
)


def assign_quadrant(pos: np.ndarray, ch_type: ChannelType) -> Quadrant:
    """Assign a channel to its spatial group from head-centred RAS coordinates.

    RAS convention: +x is right, +y is anterior, +z is superior, in metres.

    Peripheral families are routed to :attr:`Quadrant.PERIPHERAL` regardless of
    position, because their coordinates are nominal placeholders (an ECG
    electrode on the hip has a position, but it is not brain geometry).

    The midline is assigned to the right/anterior side by the ``>= 0``
    convention. Arbitrary, but it must be *deterministic* -- a channel that
    changes quadrant between runs would silently change the factorisation.
    """
    if ch_type in PERIPHERAL_TYPES:
        return Quadrant.PERIPHERAL
    pos = np.asarray(pos, dtype=float)
    if pos.shape != (3,):
        raise ValueError(f"pos must be shape (3,), got {pos.shape}")
    right = bool(pos[0] >= 0.0)
    anterior = bool(pos[1] >= 0.0)
    if anterior:
        return Quadrant.RIGHT_ANTERIOR if right else Quadrant.LEFT_ANTERIOR
    return Quadrant.RIGHT_POSTERIOR if right else Quadrant.LEFT_POSTERIOR


def _unit(v: np.ndarray, *, allow_zero: bool = False) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"expected a 3-vector, got shape {v.shape}")
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        if allow_zero:
            return np.zeros(3)
        raise ValueError("orientation vector has zero length")
    return v / n


@dataclass(frozen=True)
class SensorDescriptor:
    """Everything the model knows about one channel, before any data arrives.

    Attributes
    ----------
    pos
        Head-centred RAS position in **metres**. The head spans roughly 0.2 m,
        which is what sets the Fourier feature bandwidths in
        :mod:`neurocast.tokenizer.embedding`.
    ori_normal
        Unit vector: coil normal for MEG, outward surface normal for electrodes.
    ori_base
        Unit vector: gradiometer baseline direction. Zeros for non-gradiometers.
        This is what distinguishes co-located planar gradiometer pairs.
    baseline_m
        Gradiometer baseline length in metres; 0.0 otherwise.
    ch_type
        Sensor family. Also determines the canonical scale constant.
    reference
        Referencing scheme.
    contact_mm
        Electrode or contact diameter in mm; 0.0 for MEG.
    is_bad
        Marked bad during acquisition or QC. Bad channels are *kept* and flagged
        rather than dropped, so the channel count stays stable and the model can
        learn to discount them.
    """

    name: str
    pos: np.ndarray
    ori_normal: np.ndarray
    ch_type: ChannelType
    ori_base: np.ndarray | None = None
    baseline_m: float = 0.0
    reference: Reference = Reference.NONE
    contact_mm: float = 0.0
    is_bad: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", np.asarray(self.pos, dtype=float).reshape(3))
        object.__setattr__(self, "ori_normal", _unit(self.ori_normal))
        base = np.zeros(3) if self.ori_base is None else _unit(self.ori_base, allow_zero=True)
        object.__setattr__(self, "ori_base", base)
        object.__setattr__(self, "ch_type", ChannelType(self.ch_type))
        if self.baseline_m < 0:
            raise ValueError("baseline_m must be non-negative")
        if self.ch_type is ChannelType.GRAD and float(np.linalg.norm(base)) < 1e-12:
            raise ValueError(
                f"gradiometer {self.name!r} needs ori_base: co-located planar "
                "gradiometers differ only in baseline direction, so without it "
                "they are indistinguishable to the model"
            )

    @property
    def quadrant(self) -> Quadrant:
        return assign_quadrant(self.pos, self.ch_type)

    @property
    def type_id(self) -> int:
        return list(ChannelType).index(self.ch_type)

    @property
    def log_scale(self) -> float:
        """log10 of the canonicalisation constant applied to this family."""
        return float(np.log10(CANONICAL_SCALE[self.ch_type]))

    @property
    def is_intracranial(self) -> bool:
        return self.ch_type in INTRACRANIAL_TYPES

    @property
    def is_peripheral(self) -> bool:
        return self.ch_type in PERIPHERAL_TYPES

    def feature_vector(self) -> np.ndarray:
        """Raw descriptor fields, pre-Fourier. Order is part of the checkpoint.

        Layout (17 values):
            pos(3) ori_normal(3) ori_base(3) baseline_m(1) radius(1)
            log_scale(1) contact_mm(1) is_intra(1) is_bad(1)
            type_id(1) reference(1)
        """
        return np.concatenate(
            [
                self.pos,
                self.ori_normal,
                self.ori_base,
                [self.baseline_m],
                [float(np.linalg.norm(self.pos))],
                [self.log_scale],
                [self.contact_mm],
                [float(self.is_intracranial)],
                [float(self.is_bad)],
                [float(self.type_id)],
                [float(self.reference)],
            ]
        ).astype(np.float32)


@dataclass(frozen=True)
class Montage:
    """An ordered collection of descriptors: one recording array."""

    name: str
    sensors: tuple[SensorDescriptor, ...]

    def __post_init__(self) -> None:
        names = [s.name for s in self.sensors]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate channel names in montage: {dupes}")
        assert_partition(self.sensors)

    def __len__(self) -> int:
        return len(self.sensors)

    @property
    def quadrants(self) -> np.ndarray:
        """(n_channels,) int array of quadrant ids."""
        return np.array([int(s.quadrant) for s in self.sensors], dtype=np.int64)

    @property
    def type_ids(self) -> np.ndarray:
        return np.array([s.type_id for s in self.sensors], dtype=np.int64)

    @property
    def ch_types(self) -> list[ChannelType]:
        return [s.ch_type for s in self.sensors]

    @property
    def bad_mask(self) -> np.ndarray:
        return np.array([s.is_bad for s in self.sensors], dtype=bool)

    def indices_in(self, quadrant: Quadrant) -> np.ndarray:
        return np.flatnonzero(self.quadrants == int(quadrant))

    def feature_matrix(self) -> np.ndarray:
        """(n_channels, 17) descriptor features."""
        return np.stack([s.feature_vector() for s in self.sensors])

    def summary(self) -> str:
        counts = {q: int((self.quadrants == int(q)).sum()) for q in Quadrant}
        by_type: dict[str, int] = {}
        for s in self.sensors:
            by_type[s.ch_type.value] = by_type.get(s.ch_type.value, 0) + 1
        types = " ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
        quads = " ".join(f"{q.name}={counts[q]}" for q in Quadrant)
        return f"{self.name}: {len(self)} ch  [{types}]\n  quadrants: {quads}"


def assert_partition(sensors: Sequence[SensorDescriptor] | Iterable[SensorDescriptor]) -> None:
    """Verify the quadrants form an exact partition of the channels.

    The autoregressive factorisation over space is only a valid chain-rule
    decomposition if every channel belongs to exactly one group. If this fails,
    ``log p(x)`` stops being a likelihood and every number in the Atlas and the
    noisy-channel decoder becomes meaningless -- silently, with no crash.

    Raises
    ------
    AssertionError
        If any channel is unassigned, or the groups do not cover every channel
        exactly once.
    """
    sensors = list(sensors)
    if not sensors:
        raise AssertionError("empty montage: nothing to partition")

    assigned: dict[int, list[str]] = {int(q): [] for q in Quadrant}
    for s in sensors:
        q = s.quadrant
        if int(q) not in assigned:
            raise AssertionError(f"channel {s.name!r} has invalid quadrant {q!r}")
        assigned[int(q)].append(s.name)

    covered = sum(len(v) for v in assigned.values())
    if covered != len(sensors):
        raise AssertionError(
            f"partition covers {covered} assignments for {len(sensors)} channels"
        )

    all_names = [n for v in assigned.values() for n in v]
    if len(set(all_names)) != len(all_names):
        dupes = sorted({n for n in all_names if all_names.count(n) > 1})
        raise AssertionError(f"channels assigned to more than one quadrant: {dupes}")

    brain = [s for s in sensors if not s.is_peripheral]
    if brain and all(s.quadrant is Quadrant.PERIPHERAL for s in brain):
        raise AssertionError("no brain channel landed in a brain quadrant")
