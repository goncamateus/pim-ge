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
from _viz import STABILITY_LABELS, build_figure, init_colorbars, save_or_show
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
    p.add_argument("--source-z", type=float, default=25.0, help="release height [m] of the point source")
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

    p.add_argument("--start-y", type=float, default=-25.0, help="grid lower y bound [m] (crosswind)")
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
        "--dir-mean", type=float, default=0.0, help="OU mean-reversion level for wind direction [rad]"
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
    p.add_argument("--jet-angle", type=float, default=15.0, help="jet exit direction [deg, world frame]")
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
    conc_all = np.array(A * args.emission_rate)
    print(f"Done. Global peak: {conc_all.max():.1f} ppm")

    speeds_np = np.array(speeds)
    directions_np = np.array(directions)

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
    fp0 = conc_all[0].reshape(args.nx, args.ny, args.nz).max(axis=2)
    xz0 = conc_all[0].reshape(args.nx, args.ny, args.nz)[:, args.ny // 2, :]
    init_colorbars(ax_xy, ax_xz, Xg, Yg, Zg, fp0, xz0, NORM)

    def _setup_ax3(t):
        """Reset the 3D axes' styling/limits/viewing angle (called after each `ax3.cla()`).

        Camera orbits smoothly over the animation so both crosswind (-y/+y)
        and vertical (-z/+z) sides of the plume are visible across frames.
        """
        ax3.set_facecolor("#0e0e0e")
        ax3.set_xlabel("x (m)", labelpad=4, color="white", fontsize=8)
        ax3.set_ylabel("y (m)", labelpad=4, color="white", fontsize=8)
        ax3.set_zlabel("z (m)", labelpad=4, color="white", fontsize=8)
        ax3.set_xlim(args.start_x - 10, args.end_x + 10)
        ax3.set_ylim(args.start_y - 10, args.end_y + 10)
        ax3.set_zlim(args.start_z - 10, args.end_z + 10)
        ax3.set_box_aspect(
            (
                args.end_x - args.start_x + 20,
                args.end_y - args.start_y + 20,
                args.end_z - args.start_z + 20,
            )
        )  # true proportions, not auto-stretched
        ax3.tick_params(colors="white", labelsize=7)
        azim = -180 * (t / max(T - 1, 1)) - 30  # full half-turn -> see both +y/-y sides
        elev = source.z + 10 * np.sin(
            2 * np.pi * 2 * t / max(T - 1, 1)
        )  # sweep -> see both +z/-z sides
        ax3.view_init(elev=elev, azim=azim)

    def update(t):
        """Draw frame `t`: redraw the 3D scatter cloud, ground footprint, and cross-section.

        Parameters
        ----------
        t : int
            Frame index into `conc_all` / `speeds_np` / `directions_np`.

        Returns
        -------
        list
            Empty list (artists are redrawn via `cla()`/`clear()`, not blit).
        """
        conc_flat = conc_all[t]
        conc_3d = conc_flat.reshape(args.nx, args.ny, args.nz)
        peak = conc_flat.max()
        threshold = max(peak * args.core_frac, VMIN)

        mask = conc_flat > threshold
        idx = np.where(mask)[0]
        idx = idx[np.argsort(conc_flat[idx])]
        cm = conc_flat[idx]

        rgba = CMAP(NORM(cm))
        log_t = np.log(max(threshold, 1e-12))
        log_p = np.log(max(peak, 1e-12))
        rgba[:, 3] = (
            np.clip(
                0.45 + 0.5 * (np.log(np.clip(cm, 1e-12, None)) - log_t) / (log_p - log_t),
                0.25,
                0.95,
            )
            if log_p > log_t
            else np.full(len(cm), 0.5)
        )

        footprint = conc_3d.max(axis=2)

        # 3D axes — clear and redraw each frame
        ax3.cla()
        _setup_ax3(t)
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
            ax3.scatter(xp[idx], yp[idx], zp[idx], c=rgba, s=20, depthshade=True)
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
            Xg, Zg, conc_3d[:, args.ny // 2, :].T, cmap="inferno", norm=NORM, shading="auto"
        )
        ax_xz.axhline(source.z, color="cyan", lw=1, ls="--")
        ax_xz.set_xlabel("x (m)", fontsize=8, color="white")
        ax_xz.set_ylabel("z (m)", fontsize=8, color="white")
        ax_xz.set_title("Vertical cross-section y=0", fontsize=8, color="white")
        ax_xz.tick_params(colors="white", labelsize=7)

        deg = np.degrees(float(directions_np[t])) % 360
        spd = float(speeds_np[t])
        title.set_text(
            f"Class {STABILITY_LABELS[cls]}  |  "
            f"t={t + 1}/{T}  dir={deg:.0f}°  u={spd:.1f} m/s  peak={peak:.1f} ppm"
        )
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=max(50, 1000 // args.fps), repeat=True)

    out_base = f"examples/plume_3d_unstable_wind_class{cls}"
    save_or_show(anim, out_base, args.fps, args.show)


if __name__ == "__main__":
    main()
