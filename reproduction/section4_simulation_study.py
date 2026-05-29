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
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from pim_ge import GibbsSamplers, Priors, SourceLocation, WindField, mwg_scan
from pim_ge.forward.plume import temporal_gridfree_coupling_matrix
from pim_ge.forward.sensors import circle_of_sensors, temporal_sensors_measurements
from pim_ge.forward.wind import wind_speed, wind_direction_sinusoidal

# ── Constants ─────────────────────────────────────────────────────────────────
KEY           = jax.random.PRNGKey(42)
TRUE_SRC_X    = 50.0
TRUE_SRC_Y    = 30.0
TRUE_SRC_Z    = 1.5
SENSOR_HEIGHT = 1.0
MIXING_HEIGHT = 500.0
NOISE_STD     = 0.5     # ppm background noise
OUT_DIR       = Path("reproduction")

# Baseline conditions (level M)
BL = dict(wdc_degrees=180, dpv_case=2, ser=0.00039, dts=200.0, ops=100, layout="circle")

# DPV cases: (aH, aV, bH, bV) — order matches x[:4] after exp transform
DPV_PARAMS = {
    1: (1.4, 1.2, 0.9, 0.95),
    2: (1.0, 1.0, 1.0, 1.0),
    3: (0.9, 0.7, 0.8, 0.85),
}

FACTOR_LEVELS = {
    "WDC": [45, 90, 135, 180, 270],
    "DPV": [1, 2, 3],
    "SER": [0.000195, 0.00039, 0.00078],
    "DTS": [100, 200, 400],
    "OPS": [50, 100, 200],
    "SL":  ["circle", "grid", "line"],
}

# True params for Figure 6 misspecification: (aH, aV, bH, bV)
MISSPEC_TRUE = (1.0, 1.0, 0.8, 0.8)
MISSPEC_RANGE = {
    "aH": [0.3, 0.6, 1.0, 2.0, 4.0],
    "aV": [0.3, 0.6, 1.0, 2.0, 4.0],
    "bH": [0.5, 0.65, 0.8, 1.0, 1.2],
    "bV": [0.5, 0.65, 0.8, 1.0, 1.2],
}

PARAM_LABELS = {"aH": r"$a_H$", "aV": r"$a_V$", "bH": r"$b_H$", "bV": r"$b_V$",
                "s": r"$s$ (kg/s)", "src_x": r"$x_{src}$ (m)", "src_y": r"$y_{src}$ (m)"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_sensors(layout: str, dts: float) -> jnp.ndarray:
    if layout == "circle":
        return circle_of_sensors(0.0, 0.0, dts, 8, SENSOR_HEIGHT)
    elif layout == "grid":
        xs = jnp.linspace(-dts / 2, dts / 2, 6)
        ys = jnp.linspace(-dts / 2, dts / 2, 6)
        XX, YY = jnp.meshgrid(xs, ys)
        return jnp.stack([XX.ravel(), YY.ravel(), jnp.full(36, SENSOR_HEIGHT)], axis=1)
    else:  # "line"
        xs = jnp.linspace(-dts / 2, dts / 2, 8)
        return jnp.stack([xs, jnp.zeros(8), jnp.full(8, SENSOR_HEIGHT)], axis=1)


def make_wind(key, T: int, wdc_degrees: float) -> WindField:
    k_s, k_d = jax.random.split(key)
    std_rad = (wdc_degrees * math.pi) / 360.0
    return WindField(
        speed     = wind_speed(k_s, T, mean=2.5, std=0.3, theta=0.1),
        direction = wind_direction_sinusoidal(k_d, T, mean=0.0, std=std_rad,
                                              theta=0.05, num_periods=2.0),
    )


def make_log_params(aH, aV, bH, bV) -> jnp.ndarray:
    return jnp.array([math.log(aH), math.log(aV), math.log(bH), math.log(bV)])


def generate_data(key, sensors, wind, dpv_case: int, ser: float) -> jnp.ndarray:
    aH, aV, bH, bV = DPV_PARAMS[dpv_case]
    src = SourceLocation(x=TRUE_SRC_X, y=TRUE_SRC_Y, z=TRUE_SRC_Z)
    A = temporal_gridfree_coupling_matrix(
        src, sensors, wind,
        mixing_height=MIXING_HEIGHT, scheme="SMITH",
        estimated=True, log_params=make_log_params(aH, aV, bH, bV),
    )
    bg = jnp.full(sensors.shape[0], 0.5)
    return temporal_sensors_measurements(A, ser, bg, NOISE_STD, key)


def make_estimated_coupling(sensors, wind):
    def fn(x):
        src = SourceLocation(x=x[5], y=x[6], z=TRUE_SRC_Z)
        return temporal_gridfree_coupling_matrix(
            src, sensors, wind,
            mixing_height=MIXING_HEIGHT, scheme="SMITH",
            estimated=True, log_params=x[:4],
        )
    return fn


def make_fixed_coupling(sensors, wind, aH, aV, bH, bV):
    lp = make_log_params(aH, aV, bH, bV)
    def fn(x):
        src = SourceLocation(x=x[5], y=x[6], z=TRUE_SRC_Z)
        return temporal_gridfree_coupling_matrix(
            src, sensors, wind,
            mixing_height=MIXING_HEIGHT, scheme="SMITH",
            estimated=True, log_params=lp,
        )
    return fn


def make_priors(ser: float) -> Priors:
    return Priors(
        log_a_H_mean=0.0, log_a_H_std=2.0,
        log_a_V_mean=0.0, log_a_V_std=2.0,
        log_b_H_mean=0.0, log_b_H_std=1.0,
        log_b_V_mean=0.0, log_b_V_std=1.0,
        log_s_mean=math.log(ser), log_s_std=3.0,
        source_x_mean=0.0, source_x_std=300.0,
        source_y_mean=0.0, source_y_std=300.0,
        sigma2_alpha=2.0, sigma2_beta=1.0,
        background_std=2.0,
    )


def run_mcmc(key, sensors, wind, data, ser: float, coupling_fn,
             iters: int, burn_in: int) -> dict:
    priors = make_priors(ser)
    gibbs  = GibbsSamplers(priors)
    n      = sensors.shape[0]
    x_init = jnp.array([0.0, 0.0, 0.0, 0.0, math.log(ser), 0.0, 0.0])
    chains = mwg_scan(
        key, x_init=x_init, sigma2_init=1.0,
        background_init=jnp.zeros(n), data=data,
        coupling_fn=coupling_fn, priors=priors, gibbs=gibbs,
        step_size_init=0.01, adaptation="Optimal",
        target_accept=0.574, iters=iters,
    )
    xp = chains["x_chain"][burn_in:]
    return {
        "aH":    float(jnp.median(jnp.exp(xp[:, 0]))),
        "aV":    float(jnp.median(jnp.exp(xp[:, 1]))),
        "bH":    float(jnp.median(jnp.exp(xp[:, 2]))),
        "bV":    float(jnp.median(jnp.exp(xp[:, 3]))),
        "s":     float(jnp.median(jnp.exp(xp[:, 4]))),
        "src_x": float(jnp.median(xp[:, 5])),
        "src_y": float(jnp.median(xp[:, 6])),
        "accept": float(jnp.mean(chains["accept_chain"])),
    }


# ── Factor sweep ──────────────────────────────────────────────────────────────

def build_scenario(factor: str, level) -> dict:
    s = dict(BL)
    key = {"WDC": "wdc_degrees", "DPV": "dpv_case", "SER": "ser",
           "DTS": "dts", "OPS": "ops", "SL": "layout"}[factor]
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
                wind    = make_wind(k_w, sc["ops"], sc["wdc_degrees"])
                data    = generate_data(k_d, sensors, wind, sc["dpv_case"], sc["ser"])
                cfn     = make_estimated_coupling(sensors, wind)
                t0 = time.time()
                r = run_mcmc(k_m, sensors, wind, data, sc["ser"], cfn, iters, burn_in)
                elapsed = time.time() - t0
                r.update(factor=factor, level=level, rep=rep)
                all_results[factor][level].append(r)
                done += 1
                print(f"  [{done:3d}/{total}] {factor}={level} rep={rep}  "
                      f"src=({r['src_x']:.1f},{r['src_y']:.1f})  "
                      f"accept={r['accept']:.2f}  {elapsed:.1f}s")
    return all_results


# ── Misspecification study (Figure 6) ─────────────────────────────────────────

def run_misspec_study(n_reps: int, iters: int, burn_in: int) -> dict:
    """Returns {param_name: {val: {model: [n_reps dicts]}}} plus 'truth' and 'est' keys."""
    sensors = make_sensors(BL["layout"], BL["dts"])
    true_aH, true_aV, true_bH, true_bV = MISSPEC_TRUE
    results = {p: {v: {"truth": [], "est": [], "misspec": []}
                   for v in vals}
               for p, vals in MISSPEC_RANGE.items()}

    total_per_param = n_reps * len(next(iter(MISSPEC_RANGE.values())))
    total = len(MISSPEC_RANGE) * total_per_param
    done = 0

    for param, vals in MISSPEC_RANGE.items():
        for vi, val in enumerate(vals):
            for rep in range(n_reps):
                k = jax.random.fold_in(KEY, 10_000 + done)
                k_w, k_d, k_truth, k_est, k_mis = jax.random.split(k, 5)

                wind = make_wind(k_w, BL["ops"], BL["wdc_degrees"])
                src  = SourceLocation(x=TRUE_SRC_X, y=TRUE_SRC_Y, z=TRUE_SRC_Z)
                A    = temporal_gridfree_coupling_matrix(
                    src, sensors, wind, mixing_height=MIXING_HEIGHT,
                    scheme="SMITH", estimated=True,
                    log_params=make_log_params(true_aH, true_aV, true_bH, true_bV),
                )
                bg   = jnp.full(sensors.shape[0], 0.5)
                data = temporal_sensors_measurements(A, BL["ser"], bg, NOISE_STD, k_d)

                # truth: all fixed at true values
                cfn_truth = make_fixed_coupling(sensors, wind, true_aH, true_aV, true_bH, true_bV)
                results[param][val]["truth"].append(
                    run_mcmc(k_truth, sensors, wind, data, BL["ser"], cfn_truth, iters, burn_in))

                # est: estimate all dispersion params
                cfn_est = make_estimated_coupling(sensors, wind)
                results[param][val]["est"].append(
                    run_mcmc(k_est, sensors, wind, data, BL["ser"], cfn_est, iters, burn_in))

                # misspec: fix the one param to wrong val, others to true
                fixed = dict(aH=true_aH, aV=true_aV, bH=true_bH, bV=true_bV)
                fixed[param] = val
                cfn_mis = make_fixed_coupling(sensors, wind, **fixed)
                results[param][val]["misspec"].append(
                    run_mcmc(k_mis, sensors, wind, data, BL["ser"], cfn_mis, iters, burn_in))

                done += 1
                print(f"  [misspec {done:3d}/{total}] {param}={val:.3f} rep={rep}")
    return results


# ── Figure 3: simulation setup ────────────────────────────────────────────────

def plot_figure3(iters: int, burn_in: int):
    if not HAS_MPL:
        return
    k_w, k_d, k_m = jax.random.split(KEY, 3)
    sensors = make_sensors(BL["layout"], BL["dts"])
    wind    = make_wind(k_w, BL["ops"], BL["wdc_degrees"])
    data    = generate_data(k_d, sensors, wind, BL["dpv_case"], BL["ser"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Figure 3 — Simulation Setup (Level M)", fontsize=11, fontweight="bold")

    # Left: measurements vs wind direction for each sensor
    ax = axes[0]
    wind_deg = np.degrees(np.array(wind.direction))
    for n_i in range(data.shape[1]):
        ax.scatter(wind_deg, np.array(data[:, n_i]), s=6, alpha=0.5, label=f"S{n_i+1}")
    ax.set_xlabel("Wind direction (°)")
    ax.set_ylabel("CH₄ measurement (ppm)")
    ax.set_title("Sensor measurements vs wind direction")
    ax.legend(fontsize=6, ncol=2)

    # Middle: plume map at wind_dir = 0°, z = SENSOR_HEIGHT
    ax = axes[1]
    NX, NY = 70, 60
    xg = np.linspace(-300, 400, NX)
    yg = np.linspace(-250, 250, NY)
    XX, YY = np.meshgrid(xg, yg, indexing="ij")
    pts = jnp.stack([jnp.array(XX.ravel()), jnp.array(YY.ravel()),
                     jnp.full(NX * NY, SENSOR_HEIGHT)], axis=1)
    w0   = WindField(speed=jnp.array([2.5]), direction=jnp.array([0.0]))
    aH, aV, bH, bV = DPV_PARAMS[BL["dpv_case"]]
    src  = SourceLocation(x=TRUE_SRC_X, y=TRUE_SRC_Y, z=TRUE_SRC_Z)
    A0   = temporal_gridfree_coupling_matrix(src, pts, w0, mixing_height=MIXING_HEIGHT,
                                             scheme="SMITH", estimated=True,
                                             log_params=make_log_params(aH, aV, bH, bV))
    conc = np.array(A0[0] * BL["ser"]).reshape(NX, NY)
    peak = conc.max()
    im = ax.pcolormesh(xg, yg, conc.T, cmap="inferno",
                       norm=LogNorm(vmin=max(peak * 0.001, 1e-6), vmax=peak),
                       shading="auto")
    fig.colorbar(im, ax=ax, label="ppm", fraction=0.04)
    sx, sy = np.array(sensors[:, 0]), np.array(sensors[:, 1])
    ax.scatter(sx, sy, c="cyan", s=40, zorder=5, label="Sensors")
    ax.plot(TRUE_SRC_X, TRUE_SRC_Y, "r*", markersize=10, label="Source")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Plume at wind dir = 0°")
    ax.legend(fontsize=7)

    # Right: 6×6 grid sensor layout
    ax = axes[2]
    grid_sensors = make_sensors("grid", BL["dts"])
    gx, gy = np.array(grid_sensors[:, 0]), np.array(grid_sensors[:, 1])
    ax.scatter(gx, gy, c="steelblue", s=60)
    ax.plot(TRUE_SRC_X, TRUE_SRC_Y, "r*", markersize=12, label="Source")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("6×6 grid sensor layout")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    out = OUT_DIR / "fig3_simulation_setup.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── Figures 4 & 5: main effects boxplots ─────────────────────────────────────

def _true_val(param: str, factor: str, level) -> float:
    if factor == "DPV":
        aH, aV, bH, bV = DPV_PARAMS[level]
        return {"aH": aH, "aV": aV, "bH": bH, "bV": bV,
                "s": BL["ser"], "src_x": TRUE_SRC_X, "src_y": TRUE_SRC_Y}[param]
    if factor == "SER" and param == "s":
        return level
    aH, aV, bH, bV = DPV_PARAMS[BL["dpv_case"]]
    return {"aH": aH, "aV": aV, "bH": bH, "bV": bV,
            "s": BL["ser"], "src_x": TRUE_SRC_X, "src_y": TRUE_SRC_Y}[param]


def plot_figures_4_5(all_results: dict):
    if not HAS_MPL:
        return
    factors = list(FACTOR_LEVELS.keys())

    for fig_idx, params in enumerate([["aH", "aV", "bH", "bV", "s"],
                                       ["src_x", "src_y"]], start=4):
        fig, axes = plt.subplots(len(params), len(factors),
                                 figsize=(18, 3 * len(params) + 1),
                                 squeeze=False)
        fig.suptitle(f"Figure {fig_idx} — Main Effects: Parameter Estimation",
                     fontsize=11, fontweight="bold")

        for col, factor in enumerate(factors):
            levels  = FACTOR_LEVELS[factor]
            sweep   = all_results[factor]

            for row, param in enumerate(params):
                ax = axes[row, col]
                boxes  = [[r[param] for r in sweep[lv]] for lv in levels]
                labels = [str(lv) if isinstance(lv, str) else
                          (f"{lv:.2e}" if abs(lv) < 0.01 else f"{lv:g}") for lv in levels]

                ax.boxplot(boxes, labels=labels, patch_artist=True,
                           boxprops=dict(facecolor="lightsteelblue", alpha=0.8),
                           medianprops=dict(color="navy", lw=2),
                           flierprops=dict(marker=".", markersize=3),
                           whiskerprops=dict(lw=1.2),
                           capprops=dict(lw=1.2))

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
    source_params   = ["s", "src_x", "src_y"]
    source_labels   = [r"$s$ (kg/s)", r"$x_{src}$ (m)", r"$y_{src}$ (m)"]
    source_true_vals = [BL["ser"], TRUE_SRC_X, TRUE_SRC_Y]
    disp_params     = list(MISSPEC_RANGE.keys())

    fig, axes = plt.subplots(len(source_params), len(disp_params),
                              figsize=(14, 9), squeeze=False)
    fig.suptitle("Figure 6 — Dispersion Misspecification Impact", fontsize=11, fontweight="bold")

    for col, dp in enumerate(disp_params):
        vals = MISSPEC_RANGE[dp]

        for row, (sp, slabel, strue) in enumerate(zip(source_params, source_labels, source_true_vals)):
            ax = axes[row, col]

            # boxes for misspec at each value
            boxes_m  = [[r[sp] for r in misspec_results[dp][v]["misspec"]] for v in vals]
            truth_m  = [r[sp] for r in misspec_results[dp][vals[0]]["truth"]]
            est_m    = [r[sp] for r in misspec_results[dp][vals[0]]["est"]]

            positions = list(range(1, len(vals) + 1))
            ax.boxplot(boxes_m, positions=positions, widths=0.45, patch_artist=True,
                       boxprops=dict(facecolor="lightsalmon", alpha=0.8),
                       medianprops=dict(color="darkred", lw=2),
                       flierprops=dict(marker=".", markersize=3),
                       whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                       label="misspec")

            ax.axhline(np.median(truth_m), color="steelblue", lw=2, ls="-", alpha=0.9,
                       label="truth (fixed, correct)")
            ax.axhline(np.median(est_m),   color="seagreen",   lw=2, ls="--", alpha=0.9,
                       label="est. (all inferred)")
            ax.axhline(strue, color="red", lw=1.2, ls=":", alpha=0.7, label="true value")

            ax.set_xticks(positions)
            ax.set_xticklabels([f"{v:.2f}" for v in vals], fontsize=7)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(f"Misspecified: {dp}", fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(slabel, fontsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "fig6_misspecification.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paper",  action="store_true", help="N_REPS=50, ITERS=5000")
    p.add_argument("--fig3",   action="store_true", help="Figure 3 only (no MCMC)")
    p.add_argument("--no-fig6", action="store_true", help="Skip Figure 6 (expensive)")
    return p.parse_args()


def main():
    args    = parse_args()
    n_reps  = 50    if args.paper else 5
    iters   = 5000  if args.paper else 300
    burn_in = 1000  if args.paper else 100

    print(f"N_REPS={n_reps}  ITERS={iters}  BURN_IN={burn_in}")
    OUT_DIR.mkdir(exist_ok=True)

    plot_figure3(iters, burn_in)
    if args.fig3:
        return

    print("\n── Factor sweeps (Figures 4 & 5) ──")
    t0 = time.time()
    all_results = run_factor_sweeps(n_reps, iters, burn_in)
    print(f"Factor sweeps done in {time.time()-t0:.0f}s")
    plot_figures_4_5(all_results)

    if not args.no_fig6:
        print("\n── Misspecification study (Figure 6) ──")
        t0 = time.time()
        misspec = run_misspec_study(n_reps, iters, burn_in)
        print(f"Misspec study done in {time.time()-t0:.0f}s")
        plot_figure6(misspec)


if __name__ == "__main__":
    main()
