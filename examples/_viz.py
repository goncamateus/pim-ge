"""Shared matplotlib scaffolding for the `gaussian_3d*` animated-plume examples.

Holds only the pieces that are byte-identical across both scripts: figure/axes
layout, colorbar initialization, and the save-or-show epilogue. Per-script
logic (grid domain, wind model, camera motion, frame `update`) stays in each
script since it genuinely differs between them.
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

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
