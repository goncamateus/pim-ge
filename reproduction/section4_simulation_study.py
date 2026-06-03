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
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.ticker
    from mpl_toolkits.mplot3d import Axes3D

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from pim_ge import GibbsSamplers, Priors, SourceLocation, WindField, mwg_scan
from pim_ge.forward.plume import temporal_gridfree_coupling_matrix
from pim_ge.forward.sensors import circle_of_sensors, temporal_sensors_measurements
from pim_ge.forward.wind import wind_speed, wind_direction_sinusoidal

# ── Constants ─────────────────────────────────────────────────────────────────
KEY = jax.random.PRNGKey(42)
TRUE_SRC_X = 50.0
TRUE_SRC_Y = 50.0  # paper level-M source at (50, 50, 5) in a 110x110 m square
TRUE_SRC_Z = 5.0
SENSOR_HEIGHT = 1.0
MIXING_HEIGHT = 500.0
NOISE_STD = math.sqrt(1e-6)  # measurement error variance 1e-6 PPM (paper level M)
OUT_DIR = Path("reproduction")

# Baseline conditions (level M)
BL = dict(wdc_degrees=140, dpv_case=2, ser=0.00039, dts=50.0, ops=100, layout="grid")

# DPV cases: (aH, aV, bH, bV) — order matches x[:4] after exp transform
DPV_PARAMS = {
    1: (1.4, 1.2, 0.9, 0.95),
    2: (1.0, 1.0, 1.0, 1.0),
    3: (0.9, 0.7, 0.8, 0.85),
}

# Paper main-effects design: exactly 3 levels (Low / M=middle / High) per factor.
FACTOR_LEVELS = {
    "WDC": [60, 140, 360],  # wind-direction coverage (degrees)
    "DPV": [1, 2, 3],
    "SER": [0.000195, 0.00039, 0.00078],
    "DTS": [30, 50, 70],  # distance source<->sensors (m)
    "OPS": [10, 100, 1000],  # observations per sensor
    "SL": ["line", "grid", "sline"],  # 36x1 line, 6x6 grid, 6x1 sparse line
}

# True params for Figure 6 misspecification: (aH, aV, bH, bV)
MISSPEC_TRUE = (1.0, 1.0, 0.8, 0.8)
# Paper column order aH, bH, aV, bV; values centred on truth (truth sits at the middle slot)
MISSPEC_RANGE = {
    "aH": [0.6, 0.8, 1.0, 1.2, 1.4],
    "bH": [0.6, 0.7, 0.8, 0.9, 1.0],
    "aV": [0.6, 0.8, 1.0, 1.2, 1.4],
    "bV": [0.6, 0.7, 0.8, 0.9, 1.0],
}
# True value of each misspecified dispersion parameter (for the truth-box x-position)
MISSPEC_TRUE_VAL = {"aH": 1.0, "bH": 0.8, "aV": 1.0, "bV": 0.8}

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

# Measurement error variance (level M) — true value for the sigma2 row in Figure 4
MEAS_ERROR_VAR = 1e-6

# Per-level box colours (Low / M / High) for Figures 4 & 5
LEVEL_COLORS = ["steelblue", "indianred", "darkorange"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_sensors(layout: str, dts: float) -> jnp.ndarray:
    # Centre the array on the source so it observes the plume for all wind directions
    # (paper level M: source at (50, 50), grid spans ~DTS around it). cx/cy keep the
    # source inside coverage; without this the source falls outside the array and the
    # location posterior collapses to the prior mean.
    cx, cy = TRUE_SRC_X, TRUE_SRC_Y
    if layout == "circle":
        return circle_of_sensors(cx, cy, dts, 8, SENSOR_HEIGHT)
    elif layout == "grid":
        xs = cx + jnp.linspace(-dts / 2, dts / 2, 6)
        ys = cy + jnp.linspace(-dts / 2, dts / 2, 6)
        XX, YY = jnp.meshgrid(xs, ys)
        return jnp.stack([XX.ravel(), YY.ravel(), jnp.full(36, SENSOR_HEIGHT)], axis=1)
    elif layout == "sline":  # 6x1 sparse line
        xs = cx + jnp.linspace(-dts / 2, dts / 2, 6)
        return jnp.stack([xs, jnp.full(6, cy), jnp.full(6, SENSOR_HEIGHT)], axis=1)
    else:  # "line" — 36x1 line
        xs = cx + jnp.linspace(-dts / 2, dts / 2, 36)
        return jnp.stack([xs, jnp.full(36, cy), jnp.full(36, SENSOR_HEIGHT)], axis=1)


def make_wind(key, T: int, wdc_degrees: float) -> WindField:
    k_s, k_d = jax.random.split(key)
    std_rad = (wdc_degrees * math.pi) / 360.0
    return WindField(
        speed=wind_speed(k_s, T, mean=6.0, std=0.3, theta=0.1),
        direction=wind_direction_sinusoidal(
            k_d, T, mean=0.0, std=std_rad, theta=0.05, num_periods=2.0
        ),
    )


def make_log_params(aH, aV, bH, bV) -> jnp.ndarray:
    return jnp.array([math.log(aH), math.log(aV), math.log(bH), math.log(bV)])


def generate_data(key, sensors, wind, dpv_case: int, ser: float) -> jnp.ndarray:
    aH, aV, bH, bV = DPV_PARAMS[dpv_case]
    src = SourceLocation(x=TRUE_SRC_X, y=TRUE_SRC_Y, z=TRUE_SRC_Z)
    A = temporal_gridfree_coupling_matrix(
        src,
        sensors,
        wind,
        mixing_height=MIXING_HEIGHT,
        scheme="SMITH",
        estimated=True,
        log_params=make_log_params(aH, aV, bH, bV),
    )
    bg = jnp.full(sensors.shape[0], 0.5)
    return temporal_sensors_measurements(A, ser, bg, NOISE_STD, key)


def make_estimated_coupling(sensors, wind):
    def fn(x):
        src = SourceLocation(x=x[5], y=x[6], z=TRUE_SRC_Z)
        return temporal_gridfree_coupling_matrix(
            src,
            sensors,
            wind,
            mixing_height=MIXING_HEIGHT,
            scheme="SMITH",
            estimated=True,
            log_params=x[:4],
        )

    return fn


def make_fixed_coupling(sensors, wind, aH, aV, bH, bV):
    lp = make_log_params(aH, aV, bH, bV)

    def fn(x):
        src = SourceLocation(x=x[5], y=x[6], z=TRUE_SRC_Z)
        return temporal_gridfree_coupling_matrix(
            src,
            sensors,
            wind,
            mixing_height=MIXING_HEIGHT,
            scheme="SMITH",
            estimated=True,
            log_params=lp,
        )

    return fn


def make_priors(ser: float) -> Priors:
    return Priors(
        log_a_H_mean=0.0,
        log_a_H_std=2.0,
        log_a_V_mean=0.0,
        log_a_V_std=2.0,
        log_b_H_mean=0.0,
        log_b_H_std=1.0,
        log_b_V_mean=0.0,
        log_b_V_std=1.0,
        log_s_mean=math.log(ser),
        log_s_std=3.0,
        source_x_mean=0.0,
        source_x_std=300.0,
        source_y_mean=0.0,
        source_y_std=300.0,
        sigma2_alpha=2.0,
        sigma2_beta=1.0,
        background_std=2.0,
    )


def run_mcmc(key, sensors, wind, data, ser: float, coupling_fn, iters: int, burn_in: int) -> dict:
    priors = make_priors(ser)
    gibbs = GibbsSamplers(priors)
    n = sensors.shape[0]
    # Initialise the source guess at the sensor-array centroid (sensors are deployed
    # around the suspected source). With the sharp likelihood from low measurement
    # noise, a (0,0) start leaves the chain stuck far from the true location.
    cx0 = float(jnp.mean(sensors[:, 0]))
    cy0 = float(jnp.mean(sensors[:, 1]))
    x_init = jnp.array([0.0, 0.0, 0.0, 0.0, math.log(ser), cx0, cy0])
    chains = mwg_scan(
        key,
        x_init=x_init,
        sigma2_init=1.0,
        background_init=jnp.zeros(n),
        data=data,
        coupling_fn=coupling_fn,
        priors=priors,
        gibbs=gibbs,
        step_size_init=0.01,
        adaptation="Optimal",
        target_accept=0.574,
        iters=iters,
    )
    xp = chains["x_chain"][burn_in:]
    return {
        "aH": float(jnp.median(jnp.exp(xp[:, 0]))),
        "aV": float(jnp.median(jnp.exp(xp[:, 1]))),
        "bH": float(jnp.median(jnp.exp(xp[:, 2]))),
        "bV": float(jnp.median(jnp.exp(xp[:, 3]))),
        "s": float(jnp.median(jnp.exp(xp[:, 4]))),
        "src_x": float(jnp.median(xp[:, 5])),
        "src_y": float(jnp.median(xp[:, 6])),
        "sigma2": float(jnp.median(chains["sigma2_chain"][burn_in:])),
        "accept": float(jnp.mean(chains["accept_chain"])),
    }


# ── Factor sweep ──────────────────────────────────────────────────────────────


def build_scenario(factor: str, level) -> dict:
    s = dict(BL)
    key = {
        "WDC": "wdc_degrees",
        "DPV": "dpv_case",
        "SER": "ser",
        "DTS": "dts",
        "OPS": "ops",
        "SL": "layout",
    }[factor]
    s[key] = level
    return s


def run_factor_sweeps(n_reps: int, iters: int, burn_in: int) -> dict:
    """Returns {factor: {level: [n_reps result dicts]}}."""
    all_results = {}
    total = sum(len(lvls) for lvls in FACTOR_LEVELS.values()) * n_reps
    done = 0

    for factor, levels in FACTOR_LEVELS.items():
        all_results[factor] = {}
        for level in levels:
            all_results[factor][level] = []
            sc = build_scenario(factor, level)
            for rep in range(n_reps):
                k = jax.random.fold_in(KEY, done)
                k_w, k_d, k_m = jax.random.split(k, 3)
                sensors = make_sensors(sc["layout"], sc["dts"])
                wind = make_wind(k_w, sc["ops"], sc["wdc_degrees"])
                data = generate_data(k_d, sensors, wind, sc["dpv_case"], sc["ser"])
                cfn = make_estimated_coupling(sensors, wind)
                t0 = time.time()
                r = run_mcmc(k_m, sensors, wind, data, sc["ser"], cfn, iters, burn_in)
                elapsed = time.time() - t0
                r.update(factor=factor, level=level, rep=rep)
                all_results[factor][level].append(r)
                done += 1
                print(
                    f"  [{done:3d}/{total}] {factor}={level} rep={rep}  "
                    f"src=({r['src_x']:.1f},{r['src_y']:.1f})  "
                    f"accept={r['accept']:.2f}  {elapsed:.1f}s"
                )
    return all_results


# ── Misspecification study (Figure 6) ─────────────────────────────────────────


def run_misspec_study(n_reps: int, iters: int, burn_in: int) -> dict:
    """Returns {param_name: {val: {model: [n_reps dicts]}}} plus 'truth' and 'est' keys."""
    sensors = make_sensors(BL["layout"], BL["dts"])
    true_aH, true_aV, true_bH, true_bV = MISSPEC_TRUE
    results = {
        p: {v: {"truth": [], "est": [], "misspec": []} for v in vals}
        for p, vals in MISSPEC_RANGE.items()
    }

    total_per_param = n_reps * len(next(iter(MISSPEC_RANGE.values())))
    total = len(MISSPEC_RANGE) * total_per_param
    done = 0

    for param, vals in MISSPEC_RANGE.items():
        for vi, val in enumerate(vals):
            for rep in range(n_reps):
                k = jax.random.fold_in(KEY, 10_000 + done)
                k_w, k_d, k_truth, k_est, k_mis = jax.random.split(k, 5)

                wind = make_wind(k_w, BL["ops"], BL["wdc_degrees"])
                src = SourceLocation(x=TRUE_SRC_X, y=TRUE_SRC_Y, z=TRUE_SRC_Z)
                A = temporal_gridfree_coupling_matrix(
                    src,
                    sensors,
                    wind,
                    mixing_height=MIXING_HEIGHT,
                    scheme="SMITH",
                    estimated=True,
                    log_params=make_log_params(true_aH, true_aV, true_bH, true_bV),
                )
                bg = jnp.full(sensors.shape[0], 0.5)
                data = temporal_sensors_measurements(A, BL["ser"], bg, NOISE_STD, k_d)

                # truth: all fixed at true values
                cfn_truth = make_fixed_coupling(sensors, wind, true_aH, true_aV, true_bH, true_bV)
                results[param][val]["truth"].append(
                    run_mcmc(k_truth, sensors, wind, data, BL["ser"], cfn_truth, iters, burn_in)
                )

                # est: estimate all dispersion params
                cfn_est = make_estimated_coupling(sensors, wind)
                results[param][val]["est"].append(
                    run_mcmc(k_est, sensors, wind, data, BL["ser"], cfn_est, iters, burn_in)
                )

                # misspec: fix the one param to wrong val, others to true
                fixed = dict(aH=true_aH, aV=true_aV, bH=true_bH, bV=true_bV)
                fixed[param] = val
                cfn_mis = make_fixed_coupling(sensors, wind, **fixed)
                results[param][val]["misspec"].append(
                    run_mcmc(k_mis, sensors, wind, data, BL["ser"], cfn_mis, iters, burn_in)
                )

                done += 1
                print(f"  [misspec {done:3d}/{total}] {param}={val:.3f} rep={rep}")
    return results


# ── Figure 3: simulation setup ────────────────────────────────────────────────


def plot_figure3(iters: int, burn_in: int):
    if not HAS_MPL:
        return
    k_w, k_d, k_m = jax.random.split(KEY, 3)
    sensors = make_sensors(BL["layout"], BL["dts"])
    wind = make_wind(k_w, BL["ops"], BL["wdc_degrees"])
    data = generate_data(k_d, sensors, wind, BL["dpv_case"], BL["ser"])

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Figure 3 — Simulation Setup (Level M)", fontsize=11, fontweight="bold")

    # Panel a: polar plot — concentration vs wind direction, one line per sensor
    ax0 = fig.add_subplot(131, projection="polar")
    wind_rad = np.array(wind.direction)
    order = np.argsort(wind_rad)
    for n_i in range(data.shape[1]):
        ax0.plot(wind_rad[order], np.array(data[:, n_i])[order], lw=0.8, alpha=0.6)
    ax0.set_title("Sensor measurements vs wind direction", pad=12)
    ax0.set_xlabel("Wind direction (rad)", labelpad=10)

    # Panel b: plume map at wind_dir = 0°, colormap dark-blue → dark-red
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
        mixing_height=MIXING_HEIGHT,
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

    # Panel c: 3-D view of the 6×6 grid sensor layout
    ax2 = fig.add_subplot(133, projection="3d")
    grid_sensors = make_sensors("grid", BL["dts"])
    gx = np.array(grid_sensors[:, 0])
    gy = np.array(grid_sensors[:, 1])
    gz = np.array(grid_sensors[:, 2])
    ax2.scatter(gx, gy, gz, c="steelblue", s=40, label="Sensors")
    ax2.scatter([TRUE_SRC_X], [TRUE_SRC_Y], [TRUE_SRC_Z], c="red", marker="*", s=150, label="Source")
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
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
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
                # Colour the 3 boxes by level: Low / M (baseline) / High
                for patch, c in zip(bp["boxes"], LEVEL_COLORS):
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
    if not HAS_MPL:
        return
    source_params = ["s", "src_x", "src_y"]
    source_labels = [r"$s$ (kg/s)", r"$x_{src}$ (m)", r"$y_{src}$ (m)"]
    source_true_vals = [BL["ser"], TRUE_SRC_X, TRUE_SRC_Y]
    disp_params = list(MISSPEC_RANGE.keys())

    fig, axes = plt.subplots(len(source_params), len(disp_params), figsize=(14, 9), squeeze=False, sharey="row")
    fig.suptitle("Figure 6 — Dispersion Misspecification Impact", fontsize=11, fontweight="bold")

    for col, dp in enumerate(disp_params):
        vals = MISSPEC_RANGE[dp]

        for row, (sp, slabel, strue) in enumerate(
            zip(source_params, source_labels, source_true_vals)
        ):
            ax = axes[row, col]

            true_val = MISSPEC_TRUE_VAL[dp]
            truth_slot = vals.index(true_val) + 1  # 1-based x-position of the truth box
            est_slot = len(vals) + 1

            # misspec boxes at every slot except the truth slot (which gets the truth box)
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

            # paper's thin red dashed reference at the true source value
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
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
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
    OUT_DIR.mkdir(exist_ok=True)

    plot_figure3(iters, burn_in)
    if args.fig3:
        return

    print("\n── Factor sweeps (Figures 4 & 5) ──")
    t0 = time.time()
    all_results = run_factor_sweeps(n_reps, iters, burn_in)
    print(f"Factor sweeps done in {time.time() - t0:.0f}s")
    plot_figures_4_5(all_results)

    if not args.no_fig6:
        print("\n── Misspecification study (Figure 6) ──")
        t0 = time.time()
        misspec = run_misspec_study(n_reps, iters, burn_in)
        print(f"Misspec study done in {time.time() - t0:.0f}s")
        plot_figure6(misspec)


if __name__ == "__main__":
    main()
