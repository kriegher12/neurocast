"""The real MEGIN (Neuromag) Vectorview geometry for LibriBrain, as a NeuroCast montage.

pnpl's H5 files carry channel names and types but no sensor geometry, and
NeuroCast identifies every channel by its geometry. LibriBrain ships the scanner's
Maxwell-filter fine-calibration file (``metadata/neo/sss_cal.dat``), which lists
each of the 306 sensors in device coordinates::

    111  x y z   ex_x ex_y ex_z   ey_x ey_y ey_z   ez_x ez_y ez_z   calibration

``111`` is ``MEG0111``. ``ez`` is the coil normal. For a planar gradiometer the
measured gradient lies along ``ex``, and the two gradiometers at a site have
orthogonal ``ex`` -- which is exactly the in-plane baseline direction
:class:`~neurocast.tokenizer.descriptor.SensorDescriptor` needs to tell them apart.

The device frame has +x right, +y anterior, +z up, with its origin near the
centre of the head -- the convention the quadrant assignment expects. Parsed with
NumPy only, so the GPU environment needs no MNE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..tokenizer.descriptor import Montage, SensorDescriptor
from .canonical import ChannelType

__all__ = ["read_fine_calibration", "vectorview_montage", "GRAD_BASELINE_M"]

#: Neuromag planar gradiometer baseline.
GRAD_BASELINE_M = 0.0168


def read_fine_calibration(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """``{"MEG0111": {"pos", "ex", "ey", "ez"}, ...}`` from an ``sss_cal.dat`` file."""
    out = {}
    for line in Path(path).read_text().split("\n"):
        f = line.split()
        if len(f) < 13:
            continue
        v = np.array(f[1:13], dtype=float)
        out[f"MEG{int(f[0]):04d}"] = {"pos": v[0:3], "ex": v[3:6], "ey": v[6:9], "ez": v[9:12]}
    if not out:
        raise ValueError(f"{path}: no sensor lines parsed")
    return out


def vectorview_montage(cal_path: Path, channel_names: list[str] | None = None,
                       name: str = "megin-vectorview") -> Montage:
    """Montage in ``channel_names`` order (default: calibration-file order).

    Channel type follows Neuromag naming: a name ending in 1 is a magnetometer,
    2 or 3 a planar gradiometer.
    """
    cal = read_fine_calibration(cal_path)
    names = list(channel_names) if channel_names is not None else list(cal)
    missing = [n for n in names if n.replace(" ", "") not in cal]
    if missing:
        raise KeyError(f"{len(missing)} channels not in {Path(cal_path).name}: {missing[:4]}")
    sensors = []
    for n in names:
        c = cal[n.replace(" ", "")]
        if n.endswith("1"):
            sensors.append(SensorDescriptor(n, c["pos"], c["ez"], ChannelType.MAG))
        else:
            sensors.append(SensorDescriptor(n, c["pos"], c["ez"], ChannelType.GRAD,
                                            ori_base=c["ex"], baseline_m=GRAD_BASELINE_M))
    return Montage(name, tuple(sensors))
