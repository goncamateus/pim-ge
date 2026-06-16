"""Shared matplotlib scaffolding for the `gaussian_3d*` animated-plume examples.

Figure/axes layout, colorbar init, the per-frame scatter-threshold/panel/title
logic, and the save-or-show epilogue all live here. Each script supplies only
what's genuinely its own: grid domain, wind model, camera motion (static vs.
orbiting), and the alpha-ramp/dot-size constants for its scatter cloud.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import Colormap, LogNorm

STABILITY_LABELS = {
    "A": "A — Very unstable",
    "B": "B — Unstable",
    "C": "C — Slightly unstable",
    "D": "D — Neutral",
    "E": "E — Slightly stable",
    "F": "F — Stable",
}


def build_figure():
    """Create the dark-themed 3-panel figure: 3D scatter + footprint + xz cross-section.

    Returns
    -------
    tuple
        `(fig, ax3, ax_xy, ax_xz, title)` — the figure, its three axes
        (3D scatter, ground footprint, vertical cross-section), and the
        top title `Text` artist (initially empty, set per-frame by callers).
    """
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("#0e0e0e")

    ax3 = fig.add_axes([0.0, 0.05, 0.60, 0.88], projection="3d")
    ax_xy = fig.add_axes([0.62, 0.52, 0.34, 0.42])
    ax_xz = fig.add_axes([0.62, 0.06, 0.34, 0.38])

    title = fig.text(
        0.30, 0.97, "", ha="center", va="top", fontsize=10, color="white", fontweight="bold"
    )
    return fig, ax3, ax_xy, ax_xz, title


def init_colorbars(ax_xy, ax_xz, Xg, Yg, Zg, fp0, xz0, norm: LogNorm):
    """Draw each panel's first frame and attach a styled "ppm" colorbar.

    Parameters
    ----------
    ax_xy, ax_xz : matplotlib.axes.Axes
        Ground-footprint and vertical-cross-section axes (from `build_figure`).
    Xg, Yg, Zg : numpy.ndarray
        1D axis coordinate arrays for the footprint (`Xg`, `Yg`) and
        cross-section (`Xg`, `Zg`) panels.
    fp0, xz0 : numpy.ndarray
        First-frame footprint `(NX, NY)` and cross-section `(NX, NZ)` arrays.
    norm : matplotlib.colors.LogNorm
        Shared color normalization for both panels.
    """
    im_xy = ax_xy.pcolormesh(Xg, Yg, fp0.T, cmap="inferno", norm=norm, shading="auto")
    im_xz = ax_xz.pcolormesh(Xg, Zg, xz0.T, cmap="inferno", norm=norm, shading="auto")
    for ax, im, lbl in [
        (ax_xy, im_xy, "Ground footprint (max over z)"),
        (ax_xz, im_xz, "Vertical cross-section y=0"),
    ]:
        ax.set_facecolor("#0e0e0e")
        ax.tick_params(colors="white", labelsize=7)
        ax.set_title(lbl, fontsize=8, color="white")
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.04)
        cb.set_label("ppm", color="white", fontsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)


def save_or_show(anim: FuncAnimation, out_base: str, fps: int, show: bool) -> None:
    """Save `anim` as MP4 (falling back to GIF), then show interactively if requested or needed.

    Parameters
    ----------
    anim : matplotlib.animation.FuncAnimation
        The animation to save/display.
    out_base : str
        Output path without extension; `.mp4`/`.gif` is appended.
    fps : int
        Playback frame rate used by the writer.
    show : bool
        If True, open an interactive window even if a video file was saved.
    """
    saved = False
    for ext, writer_cls, kw in [
        (".mp4", FFMpegWriter, {"fps": fps}),
        (".gif", PillowWriter, {"fps": fps}),
    ]:
        try:
            anim.save(out_base + ext, writer=writer_cls(**kw), dpi=100)
            print(f"Saved {out_base + ext}")
            saved = True
            break
        except Exception as e:
            print(f"Cannot save {ext}: {e}", file=sys.stderr)

    if show or not saved:
        plt.show()


def scatter_mask(conc_flat: np.ndarray, peak: float, core_frac: float, vmin: float):
    """Grid points above the per-frame scatter-cloud cutoff, sorted faint-to-bright.

    Parameters
    ----------
    conc_flat : numpy.ndarray
        Flattened per-frame concentration, one value per grid point.
    peak : float
        This frame's peak concentration.
    core_frac : float
        Fraction of `peak` used as the cutoff (below `vmin` is never shown).
    vmin : float
        Colorbar floor — also the absolute minimum threshold.

    Returns
    -------
    tuple
        `(idx, cm, threshold)` — grid-point indices above threshold (sorted
        ascending by concentration so brighter points draw last/on top),
        their concentrations, and the threshold used.
    """
    threshold = max(peak * core_frac, vmin)
    idx = np.where(conc_flat > threshold)[0]
    idx = idx[np.argsort(conc_flat[idx])]
    return idx, conc_flat[idx], threshold


def cloud_rgba(
    cm: np.ndarray,
    cmap: Colormap,
    norm: LogNorm,
    threshold: float,
    peak: float,
    alpha_lo: float,
    alpha_range: float,
    clip_lo: float,
    clip_hi: float,
) -> np.ndarray:
    r"""RGBA colors for the scatter cloud, alpha ramped log-linearly from threshold to peak.

    Parameters
    ----------
    cm : numpy.ndarray
        Concentrations of the points to color (from `scatter_mask`).
    cmap : matplotlib.colors.Colormap
        Colormap mapping normalized concentration to RGB.
    norm : matplotlib.colors.LogNorm
        Shared color normalization.
    threshold, peak : float
        This frame's scatter-cloud cutoff and peak concentration (from
        `scatter_mask`/its caller), bounding the alpha ramp.
    alpha_lo, alpha_range : float
        Alpha at `threshold` and the additional alpha gained by `peak`:
        `alpha = alpha_lo + alpha_range * frac`, where `frac` is the
        log-linear position of each point between `threshold` and `peak`.
    clip_lo, clip_hi : float
        Final alpha clamp, applied after the ramp above.

    Returns
    -------
    numpy.ndarray
        RGBA array, shape `(len(cm), 4)`.
    """
    rgba = cmap(norm(cm))
    log_t = np.log(max(threshold, 1e-12))
    log_p = np.log(max(peak, 1e-12))
    if log_p > log_t:
        frac = (np.log(np.clip(cm, 1e-12, None)) - log_t) / (log_p - log_t)
        rgba[:, 3] = np.clip(alpha_lo + alpha_range * frac, clip_lo, clip_hi)
    else:
        rgba[:, 3] = np.full(len(cm), 0.5)
    return rgba


def setup_axes3d(ax3, xlim: tuple, ylim: tuple, zlim: tuple, elev: float, azim: float) -> None:
    """Style, limit, and aim the 3D scatter axes (call once per frame, after `ax3.cla()`).

    Parameters
    ----------
    ax3 : mpl_toolkits.mplot3d.Axes3D
        The 3D scatter axes.
    xlim, ylim, zlim : tuple
        `(low, high)` bounds [m] for each axis; their spans also set
        `box_aspect` so the plume's proportions stay true to the physical
        domain (not auto-stretched to fill the axes box).
    elev, azim : float
        Camera elevation/azimuth [deg] for `Axes3D.view_init` — pass fixed
        values for a static camera, or values computed from the frame index
        for an orbiting one.
    """
    ax3.set_facecolor("#0e0e0e")
    ax3.set_xlabel("x (m)", labelpad=4, color="white", fontsize=8)
    ax3.set_ylabel("y (m)", labelpad=4, color="white", fontsize=8)
    ax3.set_zlabel("z (m)", labelpad=4, color="white", fontsize=8)
    ax3.set_xlim(*xlim)
    ax3.set_ylim(*ylim)
    ax3.set_zlim(*zlim)
    ax3.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax3.tick_params(colors="white", labelsize=7)
    ax3.view_init(elev=elev, azim=azim)


def draw_3d_scatter(ax3, source, xp, yp, zp, idx, rgba, scatter_size: int, XXg, YYg, footprint) -> None:
    """Draw the source marker, thresholded scatter cloud, and ground-shadow contour on `ax3`.

    Parameters
    ----------
    ax3 : mpl_toolkits.mplot3d.Axes3D
        The 3D scatter axes (already cleared/styled via `setup_axes3d`).
    source : SourceLocation
        Emission source position — drawn as a cyan star.
    xp, yp, zp : numpy.ndarray
        Flattened grid-point coordinates.
    idx : numpy.ndarray
        Indices into `xp`/`yp`/`zp`/`rgba` for points above the scatter
        threshold (from `scatter_mask`).
    rgba : numpy.ndarray
        Per-point RGBA colors (from `cloud_rgba`), aligned with `idx`.
    scatter_size : int
        Marker size for the scatter-cloud points.
    XXg, YYg : numpy.ndarray
        Ground-plane meshgrid for the footprint contour.
    footprint : numpy.ndarray
        This frame's ground-footprint (max-over-z) concentration.
    """
    ax3.scatter(
        [source.x], [source.y], [source.z], c="cyan", s=200, marker="*", zorder=10, depthshade=False
    )
    if len(idx):
        ax3.scatter(xp[idx], yp[idx], zp[idx], c=rgba, s=scatter_size, depthshade=True)
    ax3.contourf(XXg, YYg, footprint, zdir="z", offset=0.0, levels=20, cmap="Blues", alpha=0.45)


def draw_xy_panel(ax_xy, Xg, Yg, footprint, norm: LogNorm, source_x: float, source_y: float) -> None:
    """Redraw the ground-footprint heatmap panel for one frame.

    Parameters
    ----------
    ax_xy : matplotlib.axes.Axes
        Ground-footprint axes.
    Xg, Yg : numpy.ndarray
        1D ground-plane axis coordinates.
    footprint : numpy.ndarray
        This frame's ground-footprint (max-over-z) concentration.
    norm : matplotlib.colors.LogNorm
        Shared color normalization.
    source_x, source_y : float
        Source position, drawn as a cyan star.
    """
    ax_xy.clear()
    ax_xy.set_facecolor("#0e0e0e")
    ax_xy.pcolormesh(Xg, Yg, footprint.T, cmap="inferno", norm=norm, shading="auto")
    ax_xy.scatter([source_x], [source_y], c="cyan", s=80, marker="*")
    ax_xy.set_xlabel("x (m)", fontsize=8, color="white")
    ax_xy.set_ylabel("y (m)", fontsize=8, color="white")
    ax_xy.set_title("Ground footprint (max over z)", fontsize=8, color="white")
    ax_xy.tick_params(colors="white", labelsize=7)


def draw_xz_panel(ax_xz, Xg, Zg, xz_slice, norm: LogNorm, source_z: float) -> None:
    """Redraw the vertical cross-section (y=0) panel for one frame.

    Parameters
    ----------
    ax_xz : matplotlib.axes.Axes
        Vertical-cross-section axes.
    Xg, Zg : numpy.ndarray
        1D downwind/vertical axis coordinates.
    xz_slice : numpy.ndarray
        This frame's concentration at the y=0 slice, shape `(len(Xg), len(Zg))`.
    norm : matplotlib.colors.LogNorm
        Shared color normalization.
    source_z : float
        Source height [m], drawn as a dashed cyan line.
    """
    ax_xz.clear()
    ax_xz.set_facecolor("#0e0e0e")
    ax_xz.pcolormesh(Xg, Zg, xz_slice.T, cmap="inferno", norm=norm, shading="auto")
    ax_xz.axhline(source_z, color="cyan", lw=1, ls="--")
    ax_xz.set_xlabel("x (m)", fontsize=8, color="white")
    ax_xz.set_ylabel("z (m)", fontsize=8, color="white")
    ax_xz.set_title("Vertical cross-section y=0", fontsize=8, color="white")
    ax_xz.tick_params(colors="white", labelsize=7)


def frame_title(label: str, t: int, T: int, direction_deg: float, speed: float, peak: float) -> str:
    """Format the per-frame title bar: stability class, progress, wind, peak concentration.

    Parameters
    ----------
    label : str
        Stability-class label (e.g. `STABILITY_LABELS[cls]`).
    t, T : int
        Current frame index and total frame count.
    direction_deg : float
        Wind direction [deg] this frame.
    speed : float
        Wind speed [m/s] this frame.
    peak : float
        This frame's peak concentration [ppm].

    Returns
    -------
    str
        `"Class {label}  |  t={t+1}/{T}  dir={deg}°  u={speed} m/s  peak={peak} ppm"`.
    """
    return (
        f"Class {label}  |  "
        f"t={t + 1}/{T}  dir={direction_deg:.0f}°  u={speed:.1f} m/s  peak={peak:.1f} ppm"
    )
