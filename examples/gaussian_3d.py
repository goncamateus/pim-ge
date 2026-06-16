"""3D Gaussian plume — choose stability class, animate T wind-direction timesteps.

Usage:
    uv run examples/gaussian_3d.py --class D --frames 100 --fps 10
"""

import argparse
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

from pim_ge import SourceLocation, WindField
from pim_ge.forward.plume import temporal_gridfree_coupling_matrix

EMISSION_RATE = 0.1  # [kg/s] source strength multiplied into the unit coupling matrix A
WIND_SPEED = 2.0  # [m/s] constant wind speed for every frame (only direction sweeps)
SOURCE_Z = 5.0  # [m] release height of the point source
MIXING_HEIGHT = 200.0  # [m] boundary-layer ceiling used by the inversion-layer reflection term
CORE_FRAC = 0.04  # fraction of each frame's peak concentration used as the scatter-cloud cutoff
NX = NY = 40  # grid points along x/y (ground plane)
NZ = 30  # grid points along z (height)
Z_MAX = 300.0  # [m] z ceiling — matches the x/y half-range so the cone isn't squashed flat

STABILITY_LABELS = {
    "A": "A — Very unstable",
    "B": "B — Unstable",
    "C": "C — Slightly unstable",
    "D": "D — Neutral",
    "E": "E — Slightly stable",
    "F": "F — Stable",
}


def parse_args():
    """Parse CLI flags for stability class, frame count, playback fps, and display mode.

    Returns
    -------
    argparse.Namespace
        `stability_class` (one of "A"-"F"), `frames` (animation length /
        number of wind directions sampled), `fps` (playback rate, also used
        as the animation interval), `show` (force an interactive window even
        if a video file was saved).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--class", dest="stability_class", default="D", choices=list("ABCDEF"))
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def build_grid():
    """Build the evaluation grid the plume concentration is sampled on.

    Returns
    -------
    tuple
        `(x, y, z, XX, YY, ZZ)` — 1D axis arrays (`NX`, `NY`, `NZ` points)
        and their `(NX, NY, NZ)` meshgrid, `indexing="ij"`. The grid is a
        600x600 m square centred on the (fixed) source so it stays valid as
        wind direction sweeps the full 360 degrees.
    """
    x = jnp.linspace(-300.0, 300.0, NX)
    y = jnp.linspace(-300.0, 300.0, NY)
    z = jnp.linspace(0.2, Z_MAX, NZ)
    XX, YY, ZZ = jnp.meshgrid(x, y, z, indexing="ij")
    return x, y, z, XX, YY, ZZ


def main():
    """Compute the plume over all frames, build the 3-panel figure, and save/show the animation.

    Pipeline: parse args -> build a fixed wind-direction sweep (0 to 2*pi
    over `T` frames at constant `WIND_SPEED`) -> evaluate
    `temporal_gridfree_coupling_matrix` once for the whole grid x all frames
    -> animate a 3D scatter cloud (thresholded by `CORE_FRAC` of each
    frame's peak) alongside a ground-footprint heatmap and a vertical
    cross-section at y=0 -> save as MP4 (falls back to GIF, then to an
    interactive window if neither encoder is available).
    """
    args = parse_args()
    T = args.frames
    cls = args.stability_class

    source = SourceLocation(x=0.0, y=0.0, z=SOURCE_Z)

    # Wind: constant speed, direction rotates 0 → 2π over T frames
    directions = jnp.linspace(0.0, 2 * jnp.pi, T, endpoint=False)
    wind = WindField(
        speed=jnp.full((T,), WIND_SPEED),
        direction=directions,
    )

    x_vals, y_vals, z_vals, XX, YY, ZZ = build_grid()
    points = jnp.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    print(f"Computing {T} timesteps, class {cls}...", flush=True)
    A = temporal_gridfree_coupling_matrix(
        source,
        points,
        wind,
        mixing_height=MIXING_HEIGHT,
        scheme="Briggs",
        stability_class=cls,
    )  # (T, NX*NY*NZ)
    conc_all = np.array(A * EMISSION_RATE)
    print(f"Done. Global peak: {conc_all.max():.1f} ppm")

    global_peak = conc_all.max()
    VMIN = max(global_peak * 0.001, 0.01)
    VMAX = global_peak
    NORM = LogNorm(vmin=VMIN, vmax=VMAX)
    CMAP = plt.colormaps["inferno"]

    Xg = np.array(x_vals)
    Yg = np.array(y_vals)
    Zg = np.array(z_vals)
    XXg, YYg = np.meshgrid(Xg, Yg, indexing="ij")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("#0e0e0e")

    ax3 = fig.add_axes([0.0, 0.05, 0.60, 0.88], projection="3d")
    ax_xy = fig.add_axes([0.62, 0.52, 0.34, 0.42])
    ax_xz = fig.add_axes([0.62, 0.06, 0.34, 0.38])

    title = fig.text(
        0.30, 0.97, "", ha="center", va="top", fontsize=10, color="white", fontweight="bold"
    )

    # Build colorbars from first frame so axes are set up once
    fp0 = conc_all[0].reshape(NX, NY, NZ).max(axis=2)
    xz0 = conc_all[0].reshape(NX, NY, NZ)[:, NY // 2, :]
    im_xy = ax_xy.pcolormesh(Xg, Yg, fp0.T, cmap="inferno", norm=NORM, shading="auto")
    im_xz = ax_xz.pcolormesh(Xg, Zg, xz0.T, cmap="inferno", norm=NORM, shading="auto")
    for ax, im, lbl in [
        (ax_xy, im_xy, "Ground footprint (max over z)"),
        (ax_xz, im_xz, "Vertical cross-section y=0"),
    ]:
        ax.set_facecolor("#0e0e0e")
        ax.tick_params(colors="white", labelsize=7)
        ax.set_title(lbl, fontsize=8, color="white")
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.set_label("ppm", color="white", fontsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    def _setup_ax3():
        """Reset the 3D axes' styling/limits/viewing angle (called after each `ax3.cla()`)."""
        ax3.set_facecolor("#0e0e0e")
        ax3.set_xlabel("x (m)", labelpad=4, color="white", fontsize=8)
        ax3.set_ylabel("y (m)", labelpad=4, color="white", fontsize=8)
        ax3.set_zlabel("z (m)", labelpad=4, color="white", fontsize=8)
        ax3.set_xlim(-300, 300)
        ax3.set_ylim(-300, 300)
        ax3.set_zlim(0, Z_MAX)
        ax3.set_box_aspect((600, 600, Z_MAX))  # true proportions so the cone isn't squashed flat
        ax3.tick_params(colors="white", labelsize=7)
        ax3.view_init(elev=24, azim=-55)

    def update(t):
        """Draw frame `t`: redraw the 3D scatter cloud, ground footprint, and cross-section.

        Parameters
        ----------
        t : int
            Frame index into `conc_all` / `directions`.

        Returns
        -------
        list
            Empty list (artists are redrawn via `cla()`/`clear()`, not blit).
        """
        conc_flat = conc_all[t]
        conc_3d = conc_flat.reshape(NX, NY, NZ)
        peak = conc_flat.max()
        threshold = max(peak * CORE_FRAC, VMIN)

        mask = conc_flat > threshold
        idx = np.where(mask)[0]
        idx = idx[np.argsort(conc_flat[idx])]
        cm = conc_flat[idx]

        rgba = CMAP(NORM(cm))
        log_t = np.log(max(threshold, 1e-12))
        log_p = np.log(max(peak, 1e-12))
        rgba[:, 3] = (
            np.clip(
                0.3 + 0.65 * (np.log(np.clip(cm, 1e-12, None)) - log_t) / (log_p - log_t), 0.1, 0.95
            )
            if log_p > log_t
            else np.full(len(cm), 0.5)
        )

        footprint = conc_3d.max(axis=2)

        # 3D axes — clear and redraw each frame
        ax3.cla()
        _setup_ax3()
        ax3.scatter(
            [source.x],
            [source.y],
            [source.z],
            c="cyan",
            s=200,
            marker="*",
            zorder=10,
            depthshade=False,
        )
        if len(idx):
            xp, yp, zp = np.array(XX.ravel()), np.array(YY.ravel()), np.array(ZZ.ravel())
            ax3.scatter(xp[idx], yp[idx], zp[idx], c=rgba, s=12, depthshade=True)
        ax3.contourf(XXg, YYg, footprint, zdir="z", offset=0.0, levels=20, cmap="Blues", alpha=0.45)

        # Ground footprint
        ax_xy.clear()
        ax_xy.set_facecolor("#0e0e0e")
        ax_xy.pcolormesh(Xg, Yg, footprint.T, cmap="inferno", norm=NORM, shading="auto")
        ax_xy.scatter([source.x], [source.y], c="cyan", s=80, marker="*")
        ax_xy.set_xlabel("x (m)", fontsize=8, color="white")
        ax_xy.set_ylabel("y (m)", fontsize=8, color="white")
        ax_xy.set_title("Ground footprint (max over z)", fontsize=8, color="white")
        ax_xy.tick_params(colors="white", labelsize=7)

        # Vertical cross-section at y=0
        ax_xz.clear()
        ax_xz.set_facecolor("#0e0e0e")
        ax_xz.pcolormesh(
            Xg, Zg, conc_3d[:, NY // 2, :].T, cmap="inferno", norm=NORM, shading="auto"
        )
        ax_xz.axhline(SOURCE_Z, color="cyan", lw=1, ls="--")
        ax_xz.set_xlabel("x (m)", fontsize=8, color="white")
        ax_xz.set_ylabel("z (m)", fontsize=8, color="white")
        ax_xz.set_title("Vertical cross-section y=0", fontsize=8, color="white")
        ax_xz.tick_params(colors="white", labelsize=7)

        deg = np.degrees(float(directions[t])) % 360
        title.set_text(
            f"Class {STABILITY_LABELS[cls]}  |  "
            f"t={t + 1}/{T}  dir={deg:.0f}°  u={WIND_SPEED} m/s  peak={peak:.1f} ppm"
        )
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=max(50, 1000 // args.fps), repeat=True)

    out_base = f"examples/plume_3d_class{cls}"
    saved = False
    for ext, WriterCls, kw in [
        (".mp4", FFMpegWriter, {"fps": args.fps}),
        (".gif", PillowWriter, {"fps": args.fps}),
    ]:
        try:
            anim.save(out_base + ext, writer=WriterCls(**kw), dpi=100)
            print(f"Saved {out_base + ext}")
            saved = True
            break
        except Exception as e:
            print(f"Cannot save {ext}: {e}", file=sys.stderr)

    if args.show or not saved:
        plt.show()


if __name__ == "__main__":
    main()
