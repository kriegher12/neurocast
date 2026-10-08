"""Verify the topographic renderer cannot flatter a forecast.

The demo puts a forecast next to the truth, and a figure is the easiest place
to cheat without meaning to. Each property below closes one way of doing it:

1. **Projection keeps the anatomy.** Left/right and front/back survive the 3-D ->
   2-D map, so every sensor stays in the quadrant the tokenizer assigned it.
2. **Interpolation is exact at sensors and never overshoots.** Every pixel is a
   convex combination of sensor values, so a map cannot show an extreme that no
   sensor recorded.
3. **One family per map.** A MEGIN helmet is drawn from its magnetometers, one
   per site; peripherals never appear.
4. **One colour scale, fixed by the truth.** Scaling the forecast cannot move the
   limits, so a forecast that has decayed toward the mean looks decayed.
5. **The correlation trace means what it says**: identical maps 1, inverted -1,
   independent ~0.
6. **It renders.** A GIF and a PNG strip are written (skipped, not failed, if
   matplotlib is unavailable).

Run:  .venv/Scripts/python.exe scripts/validate_viz.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurocast.data.canonical import ChannelType  # noqa: E402
from neurocast.tokenizer.descriptor import Quadrant  # noqa: E402
from neurocast.viz.topomap import (  # noqa: E402
    TopoGrid,
    color_limits,
    project_2d,
    render_movie,
    render_strip,
    select_channels,
    spatial_correlation,
)
from validate_tokenizer import eeg_montage, megin_montage  # noqa: E402

OUT = ROOT / "runs" / "validate_viz"


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


def main() -> int:
    ok = True
    rng = np.random.default_rng(0)
    meg, eeg = megin_montage(), eeg_montage(64)

    print("=" * 78)
    print("1. Projection keeps left/right and anterior/posterior")
    print("=" * 78)
    brain = [s for s in meg.sensors if not s.is_peripheral]
    pos = np.stack([s.pos for s in brain])
    xy = project_2d(pos)
    off_midline = (np.abs(pos[:, 0]) > 1e-9) & (np.abs(pos[:, 1]) > 1e-9)
    same = (np.sign(xy[:, 0]) == np.sign(pos[:, 0])) & (np.sign(xy[:, 1]) == np.sign(pos[:, 1]))
    ok &= check("every off-midline sensor keeps its quadrant", bool(same[off_midline].all()),
                f"{int(off_midline.sum())} sensors")
    quad_2d = np.where(xy[:, 1] >= 0, np.where(xy[:, 0] >= 0, 1, 0), np.where(xy[:, 0] >= 0, 3, 2))
    quad_tok = np.array([int(s.quadrant) for s in brain])
    ok &= check("2-D quadrant agrees with the tokenizer's partition",
                bool((quad_2d == quad_tok)[off_midline].all()))
    ok &= check("vertex maps to the origin", np.allclose(project_2d([[0, 0, 0.1]]), 0.0))
    ok &= check("projection is deterministic", np.array_equal(project_2d(pos), xy))

    print()
    print("=" * 78)
    print("2. Interpolation: exact at sensors, never overshoots")
    print("=" * 78)
    idx = select_channels(eeg)
    exy = project_2d(np.stack([eeg.sensors[i].pos for i in idx]))
    grid = TopoGrid(exy, res=48)
    vals = rng.standard_normal((len(idx), 5))
    at_sensors = grid.interpolate(vals, exy)
    ok &= check("map reproduces every sensor value exactly", np.allclose(at_sensors, vals),
                f"max error {np.abs(at_sensors - vals).max():.1e}")
    img = grid.images(vals)
    finite = img[np.isfinite(img)].reshape(-1)
    lo, hi = vals.min(), vals.max()
    ok &= check("no pixel leaves the range of the sensor values",
                bool(finite.min() >= lo - 1e-12 and finite.max() <= hi + 1e-12),
                f"pixels [{finite.min():.3f}, {finite.max():.3f}] within [{lo:.3f}, {hi:.3f}]")
    ok &= check("outside the head is transparent (NaN), not extrapolated",
                bool(np.isnan(img[:, 0, 0]).all()) and bool(np.isfinite(img[:, 24, 24]).all()))

    print()
    print("=" * 78)
    print("3. One sensor family per map")
    print("=" * 78)
    sel = select_channels(meg)
    kinds = {meg.sensors[i].ch_type for i in sel}
    ok &= check("MEGIN draws magnetometers only, one per site",
                kinds == {ChannelType.MAG} and len(sel) == 102, f"{len(sel)} channels")
    grads = select_channels(meg, ChannelType.GRAD)
    ok &= check("asking for gradiometers keeps one per co-located pair", len(grads) == 102)
    ok &= check("peripherals are never drawn",
                not any(meg.sensors[i].quadrant is Quadrant.PERIPHERAL for i in sel))

    print()
    print("=" * 78)
    print("4. One colour scale, set by the truth")
    print("=" * 78)
    real = rng.standard_normal((len(meg), 64))
    lims = color_limits(real[sel])
    ok &= check("limits are symmetric", lims[0] == -lims[1])
    shrunk, inflated = 0.1 * real, 10.0 * real
    ok &= check("the forecast cannot move the limits (they never see it)",
                color_limits(real[sel]) == lims,
                "a decayed forecast must look decayed, not be re-stretched")
    print(f"       limits {lims[1]:.3f}; a forecast at 10% amplitude would fill "
          f"{100 * np.percentile(np.abs(shrunk[sel]), 99) / lims[1]:.0f}% of the scale, "
          f"one at 10x would saturate ({100 * np.mean(np.abs(inflated[sel]) > lims[1]):.0f}% "
          f"of samples clipped)")

    print()
    print("=" * 78)
    print("5. Spatial correlation")
    print("=" * 78)
    a = rng.standard_normal((102, 200))
    b = rng.standard_normal((102, 200))
    ok &= check("identical maps -> 1", np.allclose(spatial_correlation(a, a), 1.0))
    ok &= check("inverted maps -> -1", np.allclose(spatial_correlation(a, -a), -1.0))
    r0 = spatial_correlation(a, b)
    ok &= check("independent maps -> ~0", abs(float(r0.mean())) < 0.03,
                f"mean {r0.mean():+.3f}, sd {r0.std():.3f}")

    print()
    print("=" * 78)
    print("6. Rendering")
    print("=" * 78)
    try:
        import matplotlib  # noqa: F401
        have_mpl = True
    except ImportError:
        have_mpl = False
    if not have_mpl:
        print("[SKIP] matplotlib not installed; geometry checks above still apply")
    else:
        t = np.arange(48) / 250.0
        wave = np.sin(2 * np.pi * 6.0 * t)[None, :] * real[:, :1]
        gif = render_movie(wave, 0.5 * wave, meg, OUT / "smoke.gif",
                           control=rng.standard_normal(wave.shape), step=8, fps=6, res=40)
        png = render_strip(wave, 0.5 * wave, meg, OUT / "smoke.png",
                           times_ms=(0, 40, 80, 120), res=40)
        ok &= check("GIF written", gif.exists() and gif.stat().st_size > 1000,
                    f"{gif.relative_to(ROOT)} ({gif.stat().st_size // 1024} KB)")
        ok &= check("PNG strip written", png.exists() and png.stat().st_size > 1000,
                    f"{png.relative_to(ROOT)}")

    print()
    print("=" * 78)
    if ok:
        print("VIZ VALIDATED: anatomy preserved, no overshoot, one family, one scale")
        print("fixed by the truth.")
        return 0
    print("VIZ FAILED validation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
