"""3D Gaussian plume — fixed source, unstable wind (OU speed + OU direction).

Wind speed and direction both evolve as Ornstein-Uhlenbeck processes, so the
plume meanders and pulses instead of sweeping a clean circle.

Usage:
    uv run examples/gaussian_3d_unstable_wind.py --class D --frames 100 --fps 10
"""

import argparse
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

from pim_ge import SourceLocation, WindField
from pim_ge.forward.plume import temporal_gridfree_coupling_matrix
from pim_ge.forward.wind import wind_direction, wind_speed

EMISSION_RATE = 0.9  # [kg/s] source strength multiplied into the unit coupling matrix A
SOURCE_Z = 25.0  # [m] release height of the point source
MIXING_HEIGHT = 300.0  # [m] boundary-layer ceiling — well above SOURCE_Z for offshore chimney
CORE_FRAC = 0.01  # fraction of each frame's peak concentration used as the scatter-cloud cutoff

START_X = 0  # [m] grid lower x bound (plume only evaluated downwind of the source)
END_X = 600.0  # [m] grid upper x bound
NX = 40  # grid points along x

START_Y = -200  # [m] grid lower y bound (crosswind)
END_Y = 200.0  # [m] grid upper y bound
NY = 40  # grid points along y

START_Z = 0.0  # [m] grid lower z bound (sea/ground level) — plume can spread below the source
END_Z = 100  # [m] grid upper z bound
NZ = 35  # grid points along z (denser -> fuller-looking 3D cloud)

# OU wind parameters — unstable in both scale and direction
SPEED_MEAN = 2.0  # [m/s] OU mean-reversion level for wind speed
SPEED_STD = 2.0  # [m/s] OU diffusion std for wind speed (large relative to mean -> bursty)
SPEED_THETA = 0.5  # OU mean-reversion rate for wind speed (large -> fast relaxation/noisy)
DIR_MEAN = 0.0  # [rad] OU mean-reversion level for wind direction
DIR_STD = 0.05  # [rad] OU diffusion std for wind direction (small -> slow meander)
DIR_THETA = 0.01  # OU mean-reversion rate for wind direction (small -> long, smooth drifts)

STABILITY_LABELS = {
    "A": "A — Very unstable",
    "B": "B — Unstable",
    "C": "C — Slightly unstable",
    "D": "D — Neutral",
    "E": "E — Slightly stable",
    "F": "F — Stable",
}


def parse_args():
    """Parse CLI flags for stability class, frame count, playback fps, RNG seed, display mode.

    Returns
    -------
    argparse.Namespace
        `stability_class` (one of "A"-"F"), `frames` (number of OU
        timesteps simulated), `fps` (playback rate / animation interval),
        `seed` (PRNG seed for the wind realization), `show` (force an
        interactive window even if a video file was saved).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--class", dest="stability_class", default="A", choices=list("ABCDEF"))
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def build_grid():
    """Build the evaluation grid the plume concentration is sampled on.

    Returns
    -------
    tuple
        `(x, y, z, XX, YY, ZZ)` — 1D axis arrays (`NX`, `NY`, `NZ` points)
        and their `(NX, NY, NZ)` meshgrid, `indexing="ij"`. Unlike
        `gaussian_3d.py`'s grid (centred at the source, valid for any wind
        direction), this grid spans only `[START_X, END_X]` downwind of the
        source — wind direction here only meanders slightly around 0
        (`DIR_STD`/`DIR_THETA` are small), so the plume never needs to be
        rendered behind the source.
    """
    x = jnp.linspace(START_X, END_X, NX)
    y = jnp.linspace(START_Y, END_Y, NY)
    z = jnp.linspace(START_Z, END_Z, NZ)
    XX, YY, ZZ = jnp.meshgrid(x, y, z, indexing="ij")
    return x, y, z, XX, YY, ZZ


def main():
    """Simulate an OU wind realization, compute the plume over all frames, animate it.

    Pipeline: parse args -> simulate wind speed and direction as independent
    Ornstein-Uhlenbeck processes (`forward.wind.wind_speed`,
    `forward.wind.wind_direction`) using the `SPEED_*`/`DIR_*` constants ->
    evaluate `temporal_gridfree_coupling_matrix` once for the whole grid x
    all frames -> animate a 3D scatter cloud (thresholded by `CORE_FRAC` of
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

    source = SourceLocation(x=0.0, y=0.0, z=SOURCE_Z)

    # Wind: OU speed and OU direction — independent keys, unstable in both
    key = jax.random.PRNGKey(args.seed)
    key_speed, key_dir = jax.random.split(key)
    speeds = wind_speed(key_speed, T, mean=SPEED_MEAN, std=SPEED_STD, theta=SPEED_THETA)
    directions = wind_direction(key_dir, T, mean=DIR_MEAN, std=DIR_STD, theta=DIR_THETA)
    wind = WindField(speed=speeds, direction=directions)

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

    def _setup_ax3(t):
        """Reset the 3D axes' styling/limits/viewing angle (called after each `ax3.cla()`).

        Camera orbits smoothly over the animation so both crosswind (-y/+y)
        and vertical (-z/+z) sides of the plume are visible across frames.
        """
        ax3.set_facecolor("#0e0e0e")
        ax3.set_xlabel("x (m)", labelpad=4, color="white", fontsize=8)
        ax3.set_ylabel("y (m)", labelpad=4, color="white", fontsize=8)
        ax3.set_zlabel("z (m)", labelpad=4, color="white", fontsize=8)
        ax3.set_xlim(START_X - 10, END_X + 10)
        ax3.set_ylim(START_Y - 10, END_Y + 10)
        ax3.set_zlim(START_Z - 10, END_Z + 10)
        ax3.set_box_aspect(
            (END_X - START_X + 20, END_Y - START_Y + 20, END_Z - START_Z + 20)
        )  # true proportions, not auto-stretched
        ax3.tick_params(colors="white", labelsize=7)
        azim = -180 * (t / max(T - 1, 1)) - 30  # full half-turn -> see both +y/-y sides
        elev = SOURCE_Z + 10 * np.sin(
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
            Xg, Zg, conc_3d[:, NY // 2, :].T, cmap="inferno", norm=NORM, shading="auto"
        )
        ax_xz.axhline(SOURCE_Z, color="cyan", lw=1, ls="--")
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
