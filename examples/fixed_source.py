"""3D Gaussian plume — fixed source, unstable wind (OU speed + OU direction).

Wind speed and direction both evolve as Ornstein-Uhlenbeck processes, so the
plume meanders and pulses instead of sweeping a clean circle.

Usage:
    uv run examples/gaussian_3d_unstable_wind.py --class D --frames 100 --fps 10
"""

import argparse

import jax
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
from pim_ge.forward.wind import wind_direction, wind_speed


def parse_args():
    """Parse CLI flags: stability class, frame/playback/RNG controls, and all physical/grid/OU/jet parameters.

    Returns
    -------
    argparse.Namespace
        `stability_class`, `frames`, `fps`, `seed`, `show`, plus
        `emission_rate`, `source_z`, `mixing_height`, `core_frac`; grid
        bounds/resolution `start_x`/`end_x`/`nx`, `start_y`/`end_y`/`ny`,
        `start_z`/`end_z`/`nz`; OU wind parameters `speed_mean`/`speed_std`/
        `speed_theta`, `dir_mean`/`dir_std`/`dir_theta`; and jet parameters
        `jet_speed`/`jet_angle`/`jet_diameter`.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--class", dest="stability_class", default="A", choices=list("ABCDEF"))
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show", action="store_true")

    p.add_argument(
        "--emission-rate",
        type=float,
        default=0.9,
        help="source strength s [kg/s], multiplied into the unit coupling matrix A",
    )
    p.add_argument(
        "--source-z", type=float, default=25.0, help="release height [m] of the point source"
    )
    p.add_argument(
        "--mixing-height",
        type=float,
        default=300.0,
        help="boundary-layer ceiling [m] — keep well above --source-z for an offshore chimney",
    )
    p.add_argument(
        "--core-frac",
        type=float,
        default=0.01,
        help="fraction of each frame's peak concentration used as the scatter-cloud cutoff",
    )

    p.add_argument(
        "--start-x", type=float, default=0.0, help="grid lower x bound [m] (downwind of the source)"
    )
    p.add_argument("--end-x", type=float, default=50.0, help="grid upper x bound [m]")
    p.add_argument("--nx", type=int, default=40, help="grid points along x")

    p.add_argument(
        "--start-y", type=float, default=-25.0, help="grid lower y bound [m] (crosswind)"
    )
    p.add_argument("--end-y", type=float, default=25.0, help="grid upper y bound [m]")
    p.add_argument("--ny", type=int, default=40, help="grid points along y")

    p.add_argument(
        "--start-z",
        type=float,
        default=0.0,
        help="grid lower z bound [m] (sea/ground level — plume can spread below the source)",
    )
    p.add_argument("--end-z", type=float, default=50.0, help="grid upper z bound [m]")
    p.add_argument(
        "--nz", type=int, default=35, help="grid points along z (denser -> fuller-looking 3D cloud)"
    )

    p.add_argument(
        "--speed-mean", type=float, default=2.0, help="OU mean-reversion level for wind speed [m/s]"
    )
    p.add_argument(
        "--speed-std",
        type=float,
        default=2.0,
        help="OU diffusion std for wind speed [m/s] (large relative to mean -> bursty)",
    )
    p.add_argument(
        "--speed-theta",
        type=float,
        default=0.5,
        help="OU mean-reversion rate for wind speed (large -> fast relaxation/noisy)",
    )
    p.add_argument(
        "--dir-mean",
        type=float,
        default=0.0,
        help="OU mean-reversion level for wind direction [rad]",
    )
    p.add_argument(
        "--dir-std",
        type=float,
        default=0.05,
        help="OU diffusion std for wind direction [rad] (small -> slow meander)",
    )
    p.add_argument(
        "--dir-theta",
        type=float,
        default=0.01,
        help="OU mean-reversion rate for wind direction (small -> long, smooth drifts)",
    )

    p.add_argument(
        "--jet-speed", type=float, default=20.0, help="jet exit speed [m/s]; 0 = passive (no jet)"
    )
    p.add_argument(
        "--jet-angle", type=float, default=15.0, help="jet exit direction [deg, world frame]"
    )
    p.add_argument("--jet-diameter", type=float, default=0.2, help="jet diameter [m] for L_relax")
    return p.parse_args()


def build_grid(args):
    """Build the evaluation grid the plume concentration is sampled on.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI flags (from `parse_args`); uses `start_x`/`end_x`/`nx`,
        `start_y`/`end_y`/`ny`, `start_z`/`end_z`/`nz`.

    Returns
    -------
    tuple
        `(x, y, z, XX, YY, ZZ)` — 1D axis arrays (`nx`, `ny`, `nz` points)
        and their `(nx, ny, nz)` meshgrid, `indexing="ij"`. Unlike
        `gaussian_3d.py`'s grid (centred at the source, valid for any wind
        direction), this grid spans only `[start_x, end_x]` downwind of the
        source — wind direction here only meanders slightly around 0
        (`--dir-std`/`--dir-theta` are small by default), so the plume never
        needs to be rendered behind the source.
    """
    x = jnp.linspace(args.start_x, args.end_x, args.nx)
    y = jnp.linspace(args.start_y, args.end_y, args.ny)
    z = jnp.linspace(args.start_z, args.end_z, args.nz)
    XX, YY, ZZ = jnp.meshgrid(x, y, z, indexing="ij")
    return x, y, z, XX, YY, ZZ


def build_animation(conc_all, x_vals, y_vals, z_vals, source, speeds, directions,
                     fps, stability_class, core_frac=0.01):
    """Build the 3-panel (3D scatter / footprint / xz cross-section) animation.

    Parameters
    ----------
    conc_all : numpy.ndarray, shape (T, nx, ny, nz)
        Precomputed concentration field (e.g. from export_plume_npz.compute()).
    x_vals, y_vals, z_vals : numpy.ndarray
        1D grid axis coordinates (nx, ny, nz points respectively).
    source : SourceLocation
    speeds, directions : numpy.ndarray, shape (T,)
        Per-frame wind speed [m/s] / direction [rad], for the frame title.
    fps : int
        Only affects on-screen playback timing (`FuncAnimation`'s `interval`);
        the export fps is passed separately to `save_or_show`.
    stability_class : str
        One of "A".."F" — looked up in STABILITY_LABELS for the title.
    core_frac : float
        Scatter-cloud cutoff fraction (same meaning as the --core-frac CLI flag).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    T, nx, ny, nz = conc_all.shape
    cls = stability_class

    global_peak = conc_all.max()
    VMIN = max(global_peak * 0.001, 0.01)
    VMAX = global_peak
    NORM = LogNorm(vmin=VMIN, vmax=VMAX)
    CMAP = plt.colormaps["inferno"]

    Xg = np.asarray(x_vals)
    Yg = np.asarray(y_vals)
    Zg = np.asarray(z_vals)
    XXg, YYg = np.meshgrid(Xg, Yg, indexing="ij")
    XX, YY, ZZ = np.meshgrid(Xg, Yg, Zg, indexing="ij")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax3, ax_xy, ax_xz, title = build_figure()

    # Build colorbars from first frame so axes are set up once
    fp0 = conc_all[0].max(axis=2)
    xz0 = conc_all[0][:, ny // 2, :]
    init_colorbars(ax_xy, ax_xz, Xg, Yg, Zg, fp0, xz0, NORM)

    def update(t):
        """Draw frame `t`: redraw the 3D scatter cloud, ground footprint, and cross-section.

        Camera orbits smoothly over the animation (via `setup_axes3d`'s
        `elev`/`azim`) so both crosswind (-y/+y) and vertical (-z/+z) sides
        of the plume are visible across frames.

        Parameters
        ----------
        t : int
            Frame index into `conc_all` / `speeds` / `directions`.

        Returns
        -------
        list
            Empty list (artists are redrawn via `cla()`/`clear()`, not blit).
        """
        conc_3d = conc_all[t]
        conc_flat = conc_3d.ravel()
        peak = conc_flat.max()
        idx, cm, threshold = scatter_mask(conc_flat, peak, core_frac, VMIN)
        rgba = cloud_rgba(
            cm,
            CMAP,
            NORM,
            threshold,
            peak,
            alpha_lo=0.45,
            alpha_range=0.5,
            clip_lo=0.25,
            clip_hi=0.95,
        )
        footprint = conc_3d.max(axis=2)

        # 3D axes — clear and redraw each frame
        ax3.cla()
        azim = -180 * (t / max(T - 1, 1)) - 30  # full half-turn -> see both +y/-y sides
        elev = source.z + 10 * np.sin(
            2 * np.pi * 2 * t / max(T - 1, 1)
        )  # sweep -> see both +z/-z sides
        setup_axes3d(
            ax3,
            xlim=(Xg[0] - 10, Xg[-1] + 10),
            ylim=(Yg[0] - 10, Yg[-1] + 10),
            zlim=(Zg[0] - 10, Zg[-1] + 10),
            elev=elev,
            azim=azim,
        )
        xp, yp, zp = XX.ravel(), YY.ravel(), ZZ.ravel()
        draw_3d_scatter(ax3, source, xp, yp, zp, idx, rgba, 20, XXg, YYg, footprint)

        draw_xy_panel(ax_xy, Xg, Yg, footprint, NORM, source.x, source.y)
        draw_xz_panel(ax_xz, Xg, Zg, conc_3d[:, ny // 2, :], NORM, source.z)

        deg = np.degrees(float(directions[t])) % 360
        spd = float(speeds[t])
        title.set_text(frame_title(STABILITY_LABELS[cls], t, T, deg, spd, peak))
        return []

    return FuncAnimation(fig, update, frames=T, interval=max(50, 1000 // fps), repeat=True)


def main():
    """Simulate an OU wind realization, compute the plume over all frames, animate it.

    Pipeline: parse args -> simulate wind speed and direction as independent
    Ornstein-Uhlenbeck processes (`forward.wind.wind_speed`,
    `forward.wind.wind_direction`) using the `--speed-*`/`--dir-*` CLI flags
    -> evaluate `temporal_gridfree_coupling_matrix` once for the whole grid x
    all frames -> animate a 3D scatter cloud (thresholded by `--core-frac` of
    each frame's peak) alongside a ground-footprint heatmap and a vertical
    cross-section at y=0 -> save as MP4 (falls back to GIF, then to an
    interactive window if neither encoder is available). Unlike
    `gaussian_3d.py`'s deterministic direction sweep, both wind speed and
    direction meander and pulse here, so the plume drifts and pulses instead
    of sweeping a clean circle.
    """
    args = parse_args()
    T = args.frames
    cls = args.stability_class

    source = SourceLocation(x=0.0, y=0.0, z=args.source_z)

    # Wind: OU speed and OU direction — independent keys, unstable in both
    key = jax.random.PRNGKey(args.seed)
    key_speed, key_dir = jax.random.split(key)
    speeds = wind_speed(
        key_speed, T, mean=args.speed_mean, std=args.speed_std, theta=args.speed_theta
    )
    directions = wind_direction(
        key_dir, T, mean=args.dir_mean, std=args.dir_std, theta=args.dir_theta
    )
    wind = WindField(speed=speeds, direction=directions)

    x_vals, y_vals, z_vals, XX, YY, ZZ = build_grid(args)
    points = jnp.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

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

    print(f"Computing {T} timesteps, class {cls}...", flush=True)
    A = temporal_gridfree_coupling_matrix(
        source,
        points,
        wind,
        mixing_height=args.mixing_height,
        scheme="Briggs",
        stability_class=cls,
        jet=jet,
    )  # (T, NX*NY*NZ)
    conc_all = np.array(A * args.emission_rate).reshape(T, args.nx, args.ny, args.nz)
    print(f"Done. Global peak: {conc_all.max():.1f} ppm")

    speeds_np = np.array(speeds)
    directions_np = np.array(directions)

    anim = build_animation(
        conc_all,
        np.array(x_vals),
        np.array(y_vals),
        np.array(z_vals),
        source,
        speeds_np,
        directions_np,
        args.fps,
        cls,
        args.core_frac,
    )

    out_base = f"examples/plume_3d_unstable_wind_class{cls}"
    save_or_show(anim, out_base, args.fps, args.show)


if __name__ == "__main__":
    main()
