"""Verify the tokenizer's structural guarantees.

Four claims, each of which silently invalidates something important if false:

1. **Quadrants partition the channels.** If not, the spatial-temporal chain rule
   is not an identity and ``log p(x)`` stops being a likelihood -- with no crash.
2. **Group confinement.** Each output token must be a function of *only* its own
   quadrant's channels. This is what makes the factorisation exact rather than
   an approximation, and it is testable by perturbation.
3. **Zero new parameters for a new montage.** The whole cross-array transfer
   claim (H5) rests on this. A 306-channel MEGIN array and a 64-channel EEG cap
   must be handled by an identical parameter set.
4. **Deterministic geometry encoding.** A sensor position must embed identically
   across process restarts, or a checkpoint cannot be deployed on new hardware.

Run:  .venv/Scripts/python.exe scripts/validate_tokenizer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocast.data.canonical import ChannelType  # noqa: E402
from neurocast.tokenizer.descriptor import (  # noqa: E402
    Montage,
    Quadrant,
    Reference,
    SensorDescriptor,
    assert_partition,
)
from neurocast.tokenizer.embedding import SensorEmbedding  # noqa: E402
from neurocast.tokenizer.patch import PATCH_SAMPLES, TypedPatchEmbed  # noqa: E402
from neurocast.tokenizer.perceiver import (  # noqa: E402
    SensorPerceiver,
    ar_flatten,
    ar_unflatten,
    causal_mask,
)

D_MODEL = 256


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


def _sphere_points(n: int, radius: float, seed: int) -> np.ndarray:
    """Roughly uniform points on the upper hemisphere -- a helmet-like layout."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[:, 2] = np.abs(v[:, 2]) * 0.6 + 0.2
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * radius


def megin_montage(n_sites: int = 102, with_peripherals: bool = True) -> Montage:
    """MEGIN-like array: one magnetometer + two planar gradiometers per site."""
    pos = _sphere_points(n_sites, 0.105, seed=11)
    sensors: list[SensorDescriptor] = []
    for i, p in enumerate(pos):
        normal = p / np.linalg.norm(p)
        # Two orthogonal in-plane baseline directions at the same site.
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(float(normal @ tmp)) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0])
        b1 = np.cross(normal, tmp)
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(normal, b1)
        sensors.append(
            SensorDescriptor(f"MEG{i:04d}1", p, normal, ChannelType.MAG)
        )
        sensors.append(
            SensorDescriptor(
                f"MEG{i:04d}2", p, normal, ChannelType.GRAD,
                ori_base=b1, baseline_m=0.0168,
            )
        )
        sensors.append(
            SensorDescriptor(
                f"MEG{i:04d}3", p, normal, ChannelType.GRAD,
                ori_base=b2, baseline_m=0.0168,
            )
        )
    if with_peripherals:
        sensors += [
            SensorDescriptor("EOG061", [0.0, 0.09, -0.02], [0, 1, 0],
                             ChannelType.EOG, reference=Reference.BIPOLAR),
            SensorDescriptor("EOG062", [-0.03, 0.09, -0.01], [0, 1, 0],
                             ChannelType.EOG, reference=Reference.BIPOLAR),
            SensorDescriptor("ECG063", [0.0, -0.05, -0.35], [0, 0, 1],
                             ChannelType.ECG, reference=Reference.BIPOLAR),
        ]
    return Montage("megin-306", tuple(sensors))


def eeg_montage(n: int = 64) -> Montage:
    """A completely different array: scalp electrodes, average reference."""
    pos = _sphere_points(n, 0.092, seed=23)
    return Montage(
        "eeg-64",
        tuple(
            SensorDescriptor(
                f"EEG{i:03d}", p, p / np.linalg.norm(p), ChannelType.EEG,
                reference=Reference.AVERAGE, contact_mm=8.0,
            )
            for i, p in enumerate(pos)
        ),
    )


def main() -> int:
    torch.manual_seed(0)
    ok = True

    meg = megin_montage()
    eeg = eeg_montage()

    print("=" * 78)
    print("1. Montages and the partition guarantee")
    print("=" * 78)
    print("  " + meg.summary().replace("\n", "\n  "))
    print("  " + eeg.summary().replace("\n", "\n  "))

    ok &= check("MEGIN montage has 306 brain + 3 peripheral channels",
                len(meg) == 309, f"got {len(meg)}")
    ok &= check("quadrants partition the MEGIN montage",
                _no_raise(lambda: assert_partition(meg.sensors)))
    ok &= check("quadrants partition the EEG montage",
                _no_raise(lambda: assert_partition(eeg.sensors)))

    counts = np.bincount(meg.quadrants, minlength=Quadrant.n_all())
    ok &= check("every quadrant is populated and sums to n_channels",
                int(counts.sum()) == len(meg) and bool((counts > 0).all()),
                f"counts={counts.tolist()}")
    ok &= check("peripherals land in the last (highest) AR group",
                bool((meg.quadrants[-3:] == int(Quadrant.PERIPHERAL)).all()))

    try:
        SensorDescriptor("bad", [0.1, 0, 0], [1, 0, 0], ChannelType.GRAD)
        ok &= check("gradiometer without ori_base is rejected", False)
    except ValueError:
        ok &= check("gradiometer without ori_base is rejected", True,
                    "co-located pairs would be indistinguishable")

    print()
    print("=" * 78)
    print("2. Zero new parameters for a new montage (H5 rests on this)")
    print("=" * 78)

    emb = SensorEmbedding(D_MODEL)
    patch = TypedPatchEmbed(D_MODEL)
    perc = SensorPerceiver(D_MODEL)
    n_before = emb.n_parameters()

    t_meg = SensorEmbedding.montage_tensors(meg)
    t_eeg = SensorEmbedding.montage_tensors(eeg)
    e_meg = emb(t_meg)
    e_eeg = emb(t_eeg)
    n_after = emb.n_parameters()

    ok &= check("sensor embedding handles both montages",
                e_meg.shape == (len(meg), D_MODEL) and e_eeg.shape == (len(eeg), D_MODEL),
                f"{tuple(e_meg.shape)} and {tuple(e_eeg.shape)}")
    ok &= check("parameter count unchanged by a new montage",
                n_before == n_after, f"{n_before:,} params, channel-count independent")

    total = sum(p.numel() for m in (emb, patch, perc) for p in m.parameters())
    print(f"       front-end total: {total:,} trainable parameters")

    print()
    print("=" * 78)
    print("3. Deterministic geometry encoding")
    print("=" * 78)

    torch.manual_seed(999)  # different seed; Fourier buffers must not care
    emb2 = SensorEmbedding(D_MODEL)
    f1 = emb.gamma_pos(t_meg["pos"])
    f2 = emb2.gamma_pos(t_meg["pos"])
    ok &= check("frozen Fourier features reproduce across instantiations",
                torch.allclose(f1, f2), "same seed -> same projection")
    ok &= check("Fourier projection is a buffer, not a parameter",
                not any(p is emb.gamma_pos.B for p in emb.parameters()))

    print()
    print("=" * 78)
    print("4. GROUP CONFINEMENT -- the test that makes the factorisation exact")
    print("=" * 78)

    b, n_patch = 2, 8
    t_len = n_patch * PATCH_SAMPLES
    x = torch.randn(b, len(meg), t_len)
    type_id = torch.from_numpy(meg.type_ids)
    quad = torch.from_numpy(meg.quadrants)

    with torch.no_grad():
        tok = patch(x, type_id) + e_meg[None, :, None, :]
        out1 = perc(tok, quad)

        # Perturb ONLY channels in quadrant 1.
        target = int(Quadrant.RIGHT_ANTERIOR)
        idx = torch.from_numpy(meg.indices_in(Quadrant.RIGHT_ANTERIOR))
        x2 = x.clone()
        x2[:, idx, :] += 5.0 * torch.randn(b, len(idx), t_len)
        tok2 = patch(x2, type_id) + e_meg[None, :, None, :]
        out2 = perc(tok2, quad)

    ok &= check("output shape is (B, T', G, d)",
                out1.shape == (b, n_patch, Quadrant.n_all(), D_MODEL),
                f"{tuple(out1.shape)}")

    deltas = (out1 - out2).abs().amax(dim=(0, 1, 3))
    print(f"       per-group max |delta| after perturbing quadrant {target}:")
    for g in Quadrant:
        print(f"         {g.name:<18} {float(deltas[int(g)]):.3e}")

    others = [float(deltas[int(g)]) for g in Quadrant if int(g) != target]
    ok &= check("perturbed group DID change",
                float(deltas[target]) > 1e-3, f"{float(deltas[target]):.3e}")
    ok &= check("all other groups are bit-identical",
                max(others) < 1e-6, f"max leak {max(others):.3e}")
    print("       ^ each token is a function of exactly one partition cell,")
    print("         so prod_g p(x^(g) | x^(<g), x_<t) is an identity.")

    print()
    print("=" * 78)
    print("5. AR sequence layout and causal mask")
    print("=" * 78)

    seq = ar_flatten(out1)
    ok &= check("ar_flatten gives (B, T'*G, d)",
                seq.shape == (b, n_patch * Quadrant.n_all(), D_MODEL),
                f"{tuple(seq.shape)}")
    ok &= check("ar_unflatten round-trips",
                torch.allclose(ar_unflatten(seq, Quadrant.n_all()), out1))

    # Groups fastest, time slowest: position i encodes (t = i // G, g = i % G).
    g_idx, t_idx = 3, 5
    flat = t_idx * Quadrant.n_all() + g_idx
    ok &= check("flattened index maps to the intended (t, g)",
                torch.allclose(seq[:, flat], out1[:, t_idx, g_idx]))

    m = causal_mask(6)
    ok &= check("causal mask is strictly upper triangular",
                bool((m == torch.triu(torch.ones(6, 6, dtype=torch.bool), 1)).all()))
    ok &= check("position i attends to itself and the past",
                not bool(m[3, 3]) and not bool(m[3, 0]) and bool(m[3, 4]))

    print()
    print("=" * 78)
    print("6. Patch alignment guard")
    print("=" * 78)
    try:
        patch(torch.randn(1, len(meg), t_len + 3), type_id)
        ok &= check("non-divisible T is rejected", False)
    except ValueError:
        ok &= check("non-divisible T is rejected", True,
                    "silent truncation would shift every later token")

    print()
    print("=" * 78)
    if ok:
        print("TOKENIZER VALIDATED: partition exact, groups confined,")
        print("montage-agnostic with zero new parameters.")
        return 0
    print("TOKENIZER FAILED validation.")
    return 1


def _no_raise(fn) -> bool:
    try:
        fn()
        return True
    except Exception as exc:
        print(f"       unexpected: {exc}")
        return False


if __name__ == "__main__":
    raise SystemExit(main())
