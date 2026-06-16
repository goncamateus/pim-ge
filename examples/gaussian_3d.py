"""3D Gaussian plume — choose stability class, animate T wind-direction timesteps.

Usage:
    uv run examples/gaussian_3d.py --class D --frames 100 --fps 10
    uv run examples/gaussian_3d.py --class D --jet-speed 10 --jet-angle 60
"""

import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from _viz import (
    STABILITY_LABELS,
    build_figure,
    cloud_rgba,
    draw_3d_scatter,
    draw_xy_panel,
    draw_xz_panel,
    frame_title,
    init_colorbars,
    save_or_show,
    scatter_mask,
    setup_axes3d,
)
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

from pim_ge import SourceLocation, WindField
from pim_ge.forward.momentum import JetSource
from pim_ge.forward.plume import temporal_gridfree_coupling_matrix

EMISSION_RATE = 0.1  # [kg/s] source strength multiplied into the unit coupling matrix A
WIND_SPEED = 2.0  # [m/s] constant wind speed for every frame (only direction sweeps)
SOURCE_Z = 5.0  # [m] release height of the point source
MIXING_HEIGHT = 200.0  # [m] boundary-layer ceiling used by the inversion-layer reflection term
CORE_FRAC = 0.04  # fraction of each frame's peak concentration used as the scatter-cloud cutoff
NX = NY = 40  # grid points along x/y (ground plane)
NZ = 30  # grid points along z (height)
Z_MAX = 300.0  # [m] z ceiling — matches the x/y half-range so the cone isn't squashed flat


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
    p.add_argument("--jet-speed", type=float, default=0.0, help="source exit speed [m/s]; 0 = passive")
    p.add_argument("--jet-angle", type=float, default=0.0, help="exit direction [deg, world frame]")
    p.add_argument("--jet-diameter", type=float, default=4.0, help="source diameter [m] for L_relax")
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

    # Optional momentum-carrying source: fixed exit direction in the world frame.
    jet = None
    if args.jet_speed > 0.0:
        ang = jnp.deg2rad(args.jet_angle)
        jet = JetSource(
            source,
            vx=float(args.jet_speed * jnp.cos(ang)),
            vy=float(args.jet_speed * jnp.sin(ang)),
            diameter=args.jet_diameter,
        )

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
        jet=jet,
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
    fig, ax3, ax_xy, ax_xz, title = build_figure()

    # Build colorbars from first frame so axes are set up once
    fp0 = conc_all[0].reshape(NX, NY, NZ).max(axis=2)
    xz0 = conc_all[0].reshape(NX, NY, NZ)[:, NY // 2, :]
    init_colorbars(ax_xy, ax_xz, Xg, Yg, Zg, fp0, xz0, NORM)

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
        idx, cm, threshold = scatter_mask(conc_flat, peak, CORE_FRAC, VMIN)
        rgba = cloud_rgba(
            cm, CMAP, NORM, threshold, peak, alpha_lo=0.3, alpha_range=0.65, clip_lo=0.1, clip_hi=0.95
        )
        footprint = conc_3d.max(axis=2)

        # 3D axes — clear and redraw each frame
        ax3.cla()
        setup_axes3d(ax3, xlim=(-300, 300), ylim=(-300, 300), zlim=(0, Z_MAX), elev=24, azim=-55)
        xp, yp, zp = np.array(XX.ravel()), np.array(YY.ravel()), np.array(ZZ.ravel())
        draw_3d_scatter(ax3, source, xp, yp, zp, idx, rgba, 12, XXg, YYg, footprint)

        draw_xy_panel(ax_xy, Xg, Yg, footprint, NORM, source.x, source.y)
        draw_xz_panel(ax_xz, Xg, Zg, conc_3d[:, NY // 2, :], NORM, source.z)

        deg = np.degrees(float(directions[t])) % 360
        title.set_text(frame_title(STABILITY_LABELS[cls], t, T, deg, WIND_SPEED, peak))
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=max(50, 1000 // args.fps), repeat=True)

    out_base = f"examples/plume_3d_class{cls}" + (f"_jet{int(args.jet_speed)}" if jet else "")
    save_or_show(anim, out_base, args.fps, args.show)


if __name__ == "__main__":
    main()
