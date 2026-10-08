"""Topographic maps and the real-vs-forecast movie.

The demo is a split screen: the subject's real sensor activity as a topographic
movie on the left, NeuroCast's forecast of the same interval on the right,
"rendered identically". That phrase carries the honesty requirements, and each
is enforced here rather than left to whoever draws the figure:

* **One colour scale, fixed by the real data.** Limits come from the real panel
  only and are shared. Autoscaling each panel would let a forecast that has
  decayed to near-zero (the honest behaviour past the predictability horizon)
  look as vivid as the truth.
* **No interpolation overshoot.** Maps use inverse-distance weighting, whose
  every pixel is a convex combination of sensor values. Spline or RBF
  interpolation can ring beyond the data, inventing extrema neither panel has.
* **One sensor family per map.** Magnetometers and planar gradiometers measure
  different physical quantities; putting both on one map is meaningless. A MEGIN
  helmet is drawn from its magnetometers by default.
* **A control trace.** Under the movie, the spatial correlation between forecast
  and truth is plotted next to the same correlation for a forecast made from
  *another trial's* context. The real trace is only interesting where it
  separates from that control.

Geometry and interpolation are NumPy only, like the audit path. matplotlib is
imported lazily, only by the functions that draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..data.canonical import ChannelType
from ..tokenizer.descriptor import Montage

__all__ = [
    "project_2d",
    "select_channels",
    "TopoGrid",
    "color_limits",
    "spatial_correlation",
    "render_movie",
    "render_strip",
]

#: Preference order when choosing which sensor family to draw.
_FAMILY_ORDER: tuple[ChannelType, ...] = (
    ChannelType.MAG,
    ChannelType.OPM,
    ChannelType.EEG,
    ChannelType.ECOG,
    ChannelType.SEEG,
    ChannelType.GRAD,
)


def project_2d(pos: np.ndarray) -> np.ndarray:
    """Azimuthal equidistant projection of head-centred RAS positions.

    The vertex (+z) maps to the origin and the equator to radius 1; +x (right)
    and +y (anterior, nose up) keep their signs, so the left/right and
    anterior/posterior halves -- and therefore the quadrants of
    :func:`neurocast.tokenizer.descriptor.assign_quadrant` -- are preserved.
    Equidistant rather than orthographic so that sensors low on the head are not
    crushed against the rim.
    """
    p = np.asarray(pos, dtype=float).reshape(-1, 3)
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("a sensor sits at the head origin; cannot project it")
    u = p / norm
    theta = np.arccos(np.clip(u[:, 2], -1.0, 1.0))      # angle from the vertex
    phi = np.arctan2(u[:, 1], u[:, 0])
    r = theta / (np.pi / 2.0)
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def select_channels(montage: Montage, ch_type: ChannelType | str | None = None) -> np.ndarray:
    """Indices of one sensor family, one channel per physical position.

    Peripherals are never drawn. With ``ch_type=None`` the first family present
    in the order MAG, OPM, EEG, ECoG, sEEG, GRAD is used. Co-located channels
    (the two planar gradiometers at a MEGIN site) keep only the first.
    """
    types = montage.ch_types
    if ch_type is None:
        present = set(types)
        family = next((f for f in _FAMILY_ORDER if f in present), None)
        if family is None:
            raise ValueError(f"montage {montage.name!r} has no drawable brain channels")
    else:
        family = ChannelType(ch_type)
    seen: set[tuple[float, float, float]] = set()
    out = []
    for i, s in enumerate(montage.sensors):
        if s.ch_type is not family or s.is_peripheral:
            continue
        key = tuple(np.round(s.pos, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    if not out:
        raise ValueError(f"montage {montage.name!r} has no {family.value} channels")
    return np.asarray(out, dtype=int)


@dataclass
class TopoGrid:
    """Precomputed inverse-distance interpolation from sensors to an image grid.

    Built once per montage; each frame is then one matrix product.

    Parameters
    ----------
    xy
        ``(n_sensors, 2)`` projected positions, from :func:`project_2d`.
    res
        Image side length in pixels.
    power
        IDW exponent. 2 is the usual choice: smooth, and local enough that one
        sensor's value does not bleed across the head.
    pad
        The drawn disc extends to ``pad`` times the outermost sensor radius.
        Pixels beyond it are NaN (transparent), not extrapolated.
    """

    xy: np.ndarray
    res: int = 64
    power: float = 2.0
    pad: float = 1.08

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=float)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2 or len(self.xy) < 3:
            raise ValueError(f"need (n >= 3, 2) projected positions, got {self.xy.shape}")
        self.radius = float(np.linalg.norm(self.xy, axis=1).max()) * self.pad
        g = np.linspace(-self.radius, self.radius, self.res)
        gx, gy = np.meshgrid(g, g)                       # row 0 = most posterior
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        self.inside = np.hypot(pts[:, 0], pts[:, 1]) <= self.radius
        self.weights = self._weights(pts[self.inside])  # (n_inside, n_sensors)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (-self.radius, self.radius, -self.radius, self.radius)

    def _weights(self, points: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(points[:, None, :] - self.xy[None, :, :], axis=-1)
        exact = d < 1e-12
        w = 1.0 / np.maximum(d, 1e-12) ** self.power
        hit = exact.any(axis=1)
        w[hit] = exact[hit].astype(float)                 # on a sensor: its value, exactly
        return w / w.sum(axis=1, keepdims=True)

    def interpolate(self, values: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Values at arbitrary 2-D ``points``. ``values`` is ``(n_sensors, ...)``."""
        v = np.asarray(values, dtype=float)
        return np.tensordot(self._weights(np.asarray(points, float)), v, axes=(1, 0))

    def images(self, values: np.ndarray) -> np.ndarray:
        """``(n_sensors,)`` or ``(n_sensors, T)`` -> ``(T, res, res)`` maps, NaN outside."""
        v = np.asarray(values, dtype=float)
        if v.ndim == 1:
            v = v[:, None]
        if v.shape[0] != len(self.xy):
            raise ValueError(f"expected {len(self.xy)} sensor rows, got {v.shape[0]}")
        out = np.full((v.shape[1], self.res * self.res), np.nan)
        out[:, self.inside] = (self.weights @ v).T
        return out.reshape(v.shape[1], self.res, self.res)


def color_limits(real: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    """Symmetric colour limits from the **real** data only.

    Shared by both panels. Never compute them from the forecast: a forecast that
    has honestly decayed toward the mean would be stretched to look as vivid as
    the truth.
    """
    a = np.abs(np.asarray(real, dtype=float))
    lim = float(np.percentile(a[np.isfinite(a)], percentile))
    lim = lim if lim > 0 else 1.0
    return -lim, lim


def spatial_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-frame Pearson correlation across sensors. ``(C, T)`` x 2 -> ``(T,)``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    den = np.sqrt((ac**2).sum(0) * (bc**2).sum(0))
    return np.where(den > 0, (ac * bc).sum(0) / np.maximum(den, 1e-300), 0.0)


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "drawing needs matplotlib (and Pillow for GIFs): "
            "pip install matplotlib pillow"
        ) from exc
    return plt


def _head(ax, grid: TopoGrid, xy: np.ndarray) -> None:
    from matplotlib.patches import Circle, Polygon

    r = grid.radius
    ax.add_patch(Circle((0, 0), r, fill=False, lw=1.2, color="0.25"))
    ax.add_patch(Polygon([(-0.08 * r, 0.985 * r), (0, 1.1 * r), (0.08 * r, 0.985 * r)],
                         closed=False, fill=False, lw=1.2, color="0.25"))
    ax.plot(xy[:, 0], xy[:, 1], ".", ms=2, color="0.15", alpha=0.6)
    ax.set_xlim(-1.15 * r, 1.15 * r)
    ax.set_ylim(-1.15 * r, 1.15 * r)
    ax.set_aspect("equal")
    ax.axis("off")


def render_movie(
    real: np.ndarray,
    forecast: np.ndarray,
    montage: Montage,
    path: str | Path,
    *,
    fs: float = 250.0,
    control: np.ndarray | None = None,
    ch_type: ChannelType | str | None = None,
    step: int = 4,
    fps: int = 12,
    res: int = 64,
    title: str = "NeuroCast: real vs forecast",
) -> Path:
    """Write the split-screen movie as a GIF.

    Parameters
    ----------
    real, forecast
        ``(n_channels, T)`` over the full montage, same interval.
    control
        Optional ``(n_channels, T)`` forecast made from a *different* trial's
        context; its correlation with ``real`` is drawn as the null trace.
    step
        Samples between frames (4 at 250 Hz = 16 ms).
    """
    plt = _pyplot()
    from matplotlib.animation import FuncAnimation, PillowWriter

    real = np.asarray(real, float)
    forecast = np.asarray(forecast, float)
    if real.shape != forecast.shape:
        raise ValueError(f"real {real.shape} and forecast {forecast.shape} differ")
    idx = select_channels(montage, ch_type)
    xy = project_2d(np.stack([montage.sensors[i].pos for i in idx]))
    grid = TopoGrid(xy, res=res)
    frames = np.arange(0, real.shape[1], step)
    img_r = grid.images(real[idx][:, frames])
    img_f = grid.images(forecast[idx][:, frames])
    vmin, vmax = color_limits(real[idx])

    t_ms = 1000.0 * np.arange(real.shape[1]) / fs
    corr = spatial_correlation(real[idx], forecast[idx])
    corr_c = None if control is None else spatial_correlation(real[idx], np.asarray(control)[idx])

    fig = plt.figure(figsize=(8.0, 6.2), dpi=90)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.3], hspace=0.28, wspace=0.05)
    ax_r, ax_f = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, :])
    kw = dict(origin="lower", extent=grid.extent, cmap="RdBu_r", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    im_r = ax_r.imshow(img_r[0], **kw)
    im_f = ax_f.imshow(img_f[0], **kw)
    for ax, name in ((ax_r, "real"), (ax_f, "forecast")):
        _head(ax, grid, xy)
        ax.set_title(name, fontsize=11)
    cb = fig.colorbar(im_r, ax=[ax_r, ax_f], shrink=0.75, pad=0.02)
    cb.set_label("canonical units (one scale, set by the real data)", fontsize=8)

    ax_t.axhline(0.0, color="0.6", lw=0.8)
    ax_t.plot(t_ms, corr, color="C3", lw=1.6, label="forecast vs real")
    if corr_c is not None:
        ax_t.plot(t_ms, corr_c, color="0.45", lw=1.2, ls="--",
                  label="control: forecast from another trial's context")
    ax_t.set_xlim(0, t_ms[-1])
    ax_t.set_ylim(-1.0, 1.0)
    ax_t.set_xlabel("ms after forecast onset")
    ax_t.set_ylabel("spatial r")
    ax_t.legend(loc="upper right", fontsize=7, frameon=False)
    cursor = ax_t.axvline(0.0, color="k", lw=0.8)
    stamp = fig.suptitle(f"{title}   t = 0 ms", fontsize=11)

    def update(k: int):
        im_r.set_data(img_r[k])
        im_f.set_data(img_f[k])
        cursor.set_xdata([t_ms[frames[k]]] * 2)
        stamp.set_text(f"{title}   t = {t_ms[frames[k]]:.0f} ms")
        return im_r, im_f, cursor, stamp

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def render_strip(
    real: np.ndarray,
    forecast: np.ndarray,
    montage: Montage,
    path: str | Path,
    *,
    fs: float = 250.0,
    times_ms: Sequence[float] = (0, 32, 64, 128, 256, 512),
    ch_type: ChannelType | str | None = None,
    res: int = 64,
) -> Path:
    """Write a two-row PNG (real over forecast) at a few latencies, same colour scale."""
    plt = _pyplot()
    real = np.asarray(real, float)
    forecast = np.asarray(forecast, float)
    idx = select_channels(montage, ch_type)
    xy = project_2d(np.stack([montage.sensors[i].pos for i in idx]))
    grid = TopoGrid(xy, res=res)
    cols = [int(round(t * fs / 1000.0)) for t in times_ms if t * fs / 1000.0 < real.shape[1]]
    vmin, vmax = color_limits(real[idx])
    fig, axes = plt.subplots(2, len(cols), figsize=(1.9 * len(cols), 4.0), dpi=110,
                             squeeze=False)
    kw = dict(origin="lower", extent=grid.extent, cmap="RdBu_r", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    for j, c in enumerate(cols):
        for i, (name, data) in enumerate((("real", real), ("forecast", forecast))):
            ax = axes[i, j]
            ax.imshow(grid.images(data[idx][:, c])[0], **kw)
            _head(ax, grid, xy)
            if i == 0:
                ax.set_title(f"{1000.0 * c / fs:.0f} ms", fontsize=9)
            if j == 0:
                ax.text(-1.25 * grid.radius, 0, name, rotation=90, va="center",
                        ha="right", fontsize=9)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
