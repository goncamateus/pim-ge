"""Reproduce §4 simulation study figures of Newman et al. (2024).

Figures produced:
    Figure 3  — simulation setup: sensor measurements vs wind direction,
                plume map at wind=0°, sensor grid layout
    Figure 4  — main effects boxplots: dispersion params + emission rate
                (6 factors × 5 rows)
    Figure 5  — main effects boxplots: source location
                (6 factors × 2 rows)
    Figure 6  — dispersion misspecification impact on source estimates

Usage:
    uv run reproduction/section4_simulation_study.py            # fast (N_REPS=5, ITERS=300)
    uv run reproduction/section4_simulation_study.py --paper    # paper quality
    uv run reproduction/section4_simulation_study.py --fig3     # Figure 3 only
"""

import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.colors import LogNorm

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from reproduction.utils import (
    BL,
    DPV_PARAMS,
    FACTOR_LEVELS,
    MEAS_ERROR_VAR,
    MISSPEC_RANGE,
    MISSPEC_TRUE_VAL,
    OUT_DIR,
    S4_KEY,
    S4_MIXING_HEIGHT,
    SENSOR_HEIGHT,
    TRUE_SRC_X,
    TRUE_SRC_Y,
    TRUE_SRC_Z,
    SourceLocation,
    WindField,
    generate_data,
    make_log_params,
    make_sensors,
    make_wind,
    run_factor_sweeps,
    run_misspec_study,
    temporal_gridfree_coupling_matrix,
)

if HAS_MPL:
    FORMATTER = matplotlib.ticker.ScalarFormatter(useMathText=True)
    FORMATTER.set_scientific(True)
    FORMATTER.set_powerlimits((-1, 1))

PARAM_LABELS = {
    "aH": r"$a_H$",
    "aV": r"$a_V$",
    "bH": r"$b_H$",
    "bV": r"$b_V$",
    "s": r"$s$ (kg/s)",
    "src_x": r"$x_{src}$ (m)",
    "src_y": r"$y_{src}$ (m)",
    "sigma2": r"$\sigma^2$",
}

LEVEL_COLORS = ["steelblue", "indianred", "darkorange"]


# ── Figure 3: simulation setup ────────────────────────────────────────────────


def plot_figure3(iters: int, burn_in: int):
    """Render Figure 3: polar sensor traces, plume map, 3D sensor layout."""
    if not HAS_MPL:
        return
    k_w, k_d, k_m = jax.random.split(S4_KEY, 3)
    sensors = make_sensors(BL["layout"], BL["dts"])
    wind = make_wind(k_w, BL["ops"], BL["wdc_degrees"])
    data = generate_data(k_d, sensors, wind, BL["dpv_case"], BL["ser"])

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Figure 3 — Simulation Setup (Level M)", fontsize=11, fontweight="bold")

    ax0 = fig.add_subplot(131, projection="polar")
    wind_rad = np.array(wind.direction)
    order = np.argsort(wind_rad)
    for n_i in range(data.shape[1]):
        ax0.plot(wind_rad[order], np.array(data[:, n_i])[order], lw=0.8, alpha=0.6)
    ax0.set_title("Sensor measurements vs wind direction", pad=12)
    ax0.set_xlabel("Wind direction (rad)", labelpad=10)

    ax1 = fig.add_subplot(132)
    NX, NY = 70, 60
    xg = np.linspace(-300, 400, NX)
    yg = np.linspace(-250, 250, NY)
    XX, YY = np.meshgrid(xg, yg, indexing="ij")
    pts = jnp.stack(
        [jnp.array(XX.ravel()), jnp.array(YY.ravel()), jnp.full(NX * NY, SENSOR_HEIGHT)], axis=1
    )
    w0 = WindField(speed=jnp.array([2.5]), direction=jnp.array([0.0]))
    aH, aV, bH, bV = DPV_PARAMS[BL["dpv_case"]]
    src = SourceLocation(x=TRUE_SRC_X, y=TRUE_SRC_Y, z=TRUE_SRC_Z)
    A0 = temporal_gridfree_coupling_matrix(
        src,
        pts,
        w0,
        mixing_height=S4_MIXING_HEIGHT,
        scheme="SMITH",
        estimated=True,
        log_params=make_log_params(aH, aV, bH, bV),
    )
    conc = np.array(A0[0] * BL["ser"]).reshape(NX, NY)
    peak = conc.max()
    im = ax1.pcolormesh(
        xg,
        yg,
        conc.T,
        cmap="RdBu_r",
        norm=LogNorm(vmin=max(peak * 0.001, 1e-6), vmax=peak),
        shading="auto",
    )
    fig.colorbar(im, ax=ax1, label="ppm", fraction=0.04)
    sx, sy = np.array(sensors[:, 0]), np.array(sensors[:, 1])
    ax1.scatter(sx, sy, c="cyan", s=40, zorder=5, label="Sensors")
    ax1.plot(TRUE_SRC_X, TRUE_SRC_Y, "r*", markersize=10, label="Source")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Plume at wind dir = 0°")
    ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(133, projection="3d")
    grid_sensors = make_sensors("grid", BL["dts"])
    gx = np.array(grid_sensors[:, 0])
    gy = np.array(grid_sensors[:, 1])
    gz = np.array(grid_sensors[:, 2])
    ax2.scatter(gx, gy, gz, c="steelblue", s=40, label="Sensors")
    ax2.scatter(
        [TRUE_SRC_X], [TRUE_SRC_Y], [TRUE_SRC_Z], c="red", marker="*", s=150, label="Source"
    )
    ax2.set_xlabel("x (m)", fontsize=7)
    ax2.set_ylabel("y (m)", fontsize=7)
    ax2.set_zlabel("z (m)", fontsize=7)
    ax2.set_title("6×6 grid sensor layout (3D)")
    ax2.legend(fontsize=7)

    out = OUT_DIR / "fig3_simulation_setup.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── Figures 4 & 5: main effects boxplots ─────────────────────────────────────


def _true_val(param: str, factor: str, level) -> float:
    if factor == "DPV":
        aH, aV, bH, bV = DPV_PARAMS[level]
        return {
            "aH": aH,
            "aV": aV,
            "bH": bH,
            "bV": bV,
            "s": BL["ser"],
            "src_x": TRUE_SRC_X,
            "src_y": TRUE_SRC_Y,
            "sigma2": MEAS_ERROR_VAR,
        }[param]
    if factor == "SER" and param == "s":
        return level
    aH, aV, bH, bV = DPV_PARAMS[BL["dpv_case"]]
    return {
        "aH": aH,
        "aV": aV,
        "bH": bH,
        "bV": bV,
        "s": BL["ser"],
        "src_x": TRUE_SRC_X,
        "src_y": TRUE_SRC_Y,
        "sigma2": MEAS_ERROR_VAR,
    }[param]


def plot_figures_4_5(all_results: dict):
    """Render Figures 4 & 5: main-effects boxplots for all six factors."""
    if not HAS_MPL:
        return
    factors = list(FACTOR_LEVELS.keys())

    for fig_idx, params in enumerate(
        [["s", "src_x", "src_y", "sigma2"], ["aH", "bH", "aV", "bV"]], start=4
    ):
        fig, axes = plt.subplots(
            len(params), len(factors), figsize=(18, 3 * len(params) + 1), squeeze=False
        )
        fig.suptitle(
            f"Figure {fig_idx} — Main Effects: Parameter Estimation", fontsize=11, fontweight="bold"
        )

        for col, factor in enumerate(factors):
            levels = FACTOR_LEVELS[factor]
            sweep = all_results[factor]

            for row, param in enumerate(params):
                ax = axes[row, col]
                ax.yaxis.set_major_formatter(FORMATTER)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
                boxes = [[r[param] for r in sweep[lv]] for lv in levels]
                labels = [
                    str(lv)
                    if isinstance(lv, str)
                    else (f"{lv:.2e}" if abs(lv) < 0.01 else f"{lv:g}")
                    for lv in levels
                ]

                bp = ax.boxplot(
                    boxes,
                    labels=labels,
                    patch_artist=True,
                    boxprops=dict(alpha=0.8),
                    medianprops=dict(color="navy", lw=2),
                    flierprops=dict(marker=".", markersize=3),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2),
                )
                for patch, c in zip(bp["boxes"], LEVEL_COLORS, strict=False):
                    patch.set_facecolor(c)

                if factor == "DPV" and param in ("aH", "aV", "bH", "bV"):
                    for xi, lv in enumerate(levels, 1):
                        tv = _true_val(param, factor, lv)
                        ax.plot(xi, tv, "r_", markersize=14, markeredgewidth=2.5)
                elif factor == "SER" and param == "s":
                    for xi, lv in enumerate(levels, 1):
                        ax.plot(xi, lv, "r_", markersize=14, markeredgewidth=2.5)
                else:
                    tv = _true_val(param, factor, levels[0])
                    ax.axhline(tv, color="red", ls="--", lw=1.5, alpha=0.85)

                ax.tick_params(labelsize=7)
                ax.set_xlabel(factor, fontsize=8)
                if col == 0:
                    ax.set_ylabel(PARAM_LABELS.get(param, param), fontsize=8)
                if row == 0:
                    ax.set_title(factor, fontsize=9, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = OUT_DIR / f"fig{fig_idx}_main_effects.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close(fig)


# ── Figure 6: misspecification impact ─────────────────────────────────────────


def plot_figure6(misspec_results: dict):
    """Render Figure 6: dispersion-misspecification impact on source estimates."""
    if not HAS_MPL:
        return
    source_params = ["s", "src_x", "src_y"]
    source_labels = [r"$s$ (kg/s)", r"$x_{src}$ (m)", r"$y_{src}$ (m)"]
    source_true_vals = [BL["ser"], TRUE_SRC_X, TRUE_SRC_Y]
    disp_params = list(MISSPEC_RANGE.keys())

    fig, axes = plt.subplots(
        len(source_params), len(disp_params), figsize=(14, 9), squeeze=False, sharey="row"
    )
    fig.suptitle("Figure 6 — Dispersion Misspecification Impact", fontsize=11, fontweight="bold")

    for col, dp in enumerate(disp_params):
        vals = MISSPEC_RANGE[dp]

        for row, (sp, slabel, strue) in enumerate(
            zip(source_params, source_labels, source_true_vals, strict=False)
        ):
            ax = axes[row, col]

            true_val = MISSPEC_TRUE_VAL[dp]
            truth_slot = vals.index(true_val) + 1
            est_slot = len(vals) + 1

            misspec_boxes, misspec_pos = [], []
            for i, v in enumerate(vals, 1):
                if i == truth_slot:
                    continue
                misspec_boxes.append([r[sp] for r in misspec_results[dp][v]["misspec"]])
                misspec_pos.append(i)
            truth_box = [r[sp] for r in misspec_results[dp][true_val]["truth"]]
            est_box = [r[sp] for r in misspec_results[dp][true_val]["est"]]

            common = dict(
                widths=0.45,
                patch_artist=True,
                flierprops=dict(marker=".", markersize=3),
                whiskerprops=dict(lw=1.2),
                capprops=dict(lw=1.2),
            )
            bp_m = ax.boxplot(
                misspec_boxes,
                positions=misspec_pos,
                boxprops=dict(facecolor="lightsalmon", alpha=0.8),
                medianprops=dict(color="darkred", lw=2),
                **common,
            )
            bp_t = ax.boxplot(
                [truth_box],
                positions=[truth_slot],
                boxprops=dict(facecolor="steelblue", alpha=0.85),
                medianprops=dict(color="navy", lw=2),
                **common,
            )
            bp_e = ax.boxplot(
                [est_box],
                positions=[est_slot],
                boxprops=dict(facecolor="seagreen", alpha=0.85),
                medianprops=dict(color="darkgreen", lw=2),
                **common,
            )

            ax.axhline(strue, color="red", lw=1.2, ls=":", alpha=0.7)

            labels = [("truth" if i == truth_slot else f"{v:.2f}") for i, v in enumerate(vals, 1)]
            labels.append("est")
            ax.set_xticks(list(range(1, est_slot + 1)))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_xlim(0.4, est_slot + 0.6)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(f"Misspecified: {dp}", fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(slabel, fontsize=8)
            if col > 0:
                ax.tick_params(labelleft=False)
                ax.set_ylabel("")
            ax.yaxis.set_major_formatter(FORMATTER)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            if row == 0 and col == 0:
                ax.legend(
                    [bp_m["boxes"][0], bp_t["boxes"][0], bp_e["boxes"][0]],
                    ["misspec", "truth (fixed)", "est. (inferred)"],
                    fontsize=6,
                    loc="upper left",
                )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "fig6_misspecification.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paper", action="store_true", help="N_REPS=50, ITERS=5000")
    p.add_argument("--fig3", action="store_true", help="Figure 3 only (no MCMC)")
    p.add_argument("--no-fig6", action="store_true", help="Skip Figure 6 (expensive)")
    return p.parse_args()


def main():
    args = parse_args()
    n_reps = 50 if args.paper else 5
    iters = 5000 if args.paper else 300
    burn_in = 1000 if args.paper else 100

    print(f"N_REPS={n_reps}  ITERS={iters}  BURN_IN={burn_in}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_figure3(iters, burn_in)
    if args.fig3:
        return

    print("\n── Factor sweeps (Figures 4 & 5) ──")
    t0 = time.time()
    all_results = run_factor_sweeps(n_reps, iters, burn_in)
    print(f"Factor sweeps done in {time.time() - t0:.0f}s")

    dataset = {"factor_sweeps": all_results}

    if not args.no_fig6:
        print("\n── Misspecification study (Figure 6) ──")
        t0 = time.time()
        misspec = run_misspec_study(n_reps, iters, burn_in)
        print(f"Misspec study done in {time.time() - t0:.0f}s")
        dataset["misspec"] = misspec

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_pkl = Path("Data") / f"artificial-{ts}.pkl"
    with open(out_pkl, "wb") as fh:
        pickle.dump(dataset, fh)
    print(f"Dataset saved → {out_pkl}")

    plot_figures_4_5(all_results)
    if not args.no_fig6:
        plot_figure6(misspec)


if __name__ == "__main__":
    main()
