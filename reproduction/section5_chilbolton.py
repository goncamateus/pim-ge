"""Reproduce §5 Chilbolton real-data case study of Newman et al. (2024).

Chilbolton uses Open-Path FTIR beam sensors with 7 reflector paths.
This script inverts Source 1 with 4 dispersion models:
    1. Fixed Briggs class D
    2. Fixed Smith class D
    3. Estimated Draxler (a/b sampled)
    4. Estimated Smith (a/b sampled)

Data availability
-----------------
Data must be present at:

    Data/Chilbolton_data_files/Postprocessed/
        Source_1/Chilbolton_CH4_measurements_source_1.pkl
        Source_1/Chilbolton_windfield_source_1.pkl
        Sensor_reflector_locations/Chilbolton_instruments_location.pkl
        Source_locations_and_emission_rates/Chilbolton_sources_locations_and_emission_rates.pkl

Download from:
    https://github.com/NewmanTHP/Probabilistic-Inversion-Modeling-of-Gas-Emissions
"""
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pim_ge import GibbsSamplers, Priors, SourceLocation, WindField, mwg_scan
from pim_ge.forward.plume import beam_path_coupling_matrix

# --- Configuration ------------------------------------------------------------
_POST = Path("Data/Chilbolton_data_files/Postprocessed")
_MEAS_FILE  = _POST / "Source_1/Chilbolton_CH4_measurements_source_1.pkl"
_WIND_FILE  = _POST / "Source_1/Chilbolton_windfield_source_1.pkl"
_LOCS_FILE  = _POST / "Sensor_reflector_locations/Chilbolton_instruments_location.pkl"
_SRCS_FILE  = _POST / "Source_locations_and_emission_rates/Chilbolton_sources_locations_and_emission_rates.pkl"

MODELS        = ["Briggs_fixed", "Smith_fixed", "Draxler_estimated", "Smith_estimated"]
ITERS         = 5000
BURN_IN       = 1000
MIXING_HEIGHT = 200.0
KEY           = jax.random.PRNGKey(0)
N_BEAMS       = 7
SOURCE_Z      = 0.3   # [m] — release height at Chilbolton


def check_data():
    missing = [f for f in (_MEAS_FILE, _WIND_FILE, _LOCS_FILE, _SRCS_FILE) if not f.exists()]
    if missing:
        print("=" * 70)
        print("DATA NOT FOUND")
        print("=" * 70)
        for f in missing:
            print(f"  missing: {f}")
        print(__doc__)
        sys.exit(1)


def load_data() -> dict:
    def _pkl(p):
        with open(p, "rb") as fh:
            return pickle.load(fh)

    meas_df = _pkl(_MEAS_FILE)
    wind_df = _pkl(_WIND_FILE)
    locs    = _pkl(_LOCS_FILE)
    srcs    = _pkl(_SRCS_FILE)

    T = len(wind_df)

    # measurements: (973,) → (T, N_BEAMS)
    measurements = meas_df["Measurements"].values.reshape(T, N_BEAMS).astype(np.float32)

    # beam geometry: sensor is start for all beams; reflectors are ends
    sensor = np.array(locs["line_of_sight_sensor"], dtype=np.float32)  # (3,)
    beam_starts = np.tile(sensor, (N_BEAMS, 1))                         # (7, 3)
    beam_ends   = np.array(
        [locs[f"reflector_{i}"] for i in range(1, N_BEAMS + 1)],
        dtype=np.float32,
    )  # (7, 3)

    # wind: direction in degrees → radians
    wind_speed     = wind_df["Average Speed"].values.astype(np.float32)
    wind_direction = np.deg2rad(wind_df["Average Direction"].values).astype(np.float32)
    tan_gamma_H    = wind_df["Average Tan_gamma Horizontal"].values.astype(np.float32)
    tan_gamma_V    = wind_df["Average Tan_gamma Vertical"].values.astype(np.float32)

    src1 = srcs["source_1_location"]   # [x, y, z]
    return {
        "measurements":  jnp.array(measurements),
        "beam_starts":   jnp.array(beam_starts),
        "beam_ends":     jnp.array(beam_ends),
        "wind_speed":    jnp.array(wind_speed),
        "wind_direction": jnp.array(wind_direction),
        "tan_gamma_H":   float(np.mean(tan_gamma_H)),
        "tan_gamma_V":   float(np.mean(tan_gamma_V)),
        "release_x":     float(src1[0]),
        "release_y":     float(src1[1]),
        "release_z":     float(src1[2]),
        "release_rate":  float(srcs["source_1_emission_rate"]),
    }


def make_coupling_fn(beam_starts, beam_ends, wind, model: str, tan_gamma_H: float, tan_gamma_V: float):
    """Return coupling_fn(x) -> A (T, N_beams) [ppm·m per kg/s]."""
    def coupling_fn_fixed_briggs(x):
        src = SourceLocation(x=x[5], y=x[6], z=SOURCE_Z)
        return beam_path_coupling_matrix(
            src, beam_starts, beam_ends, wind,
            mixing_height=MIXING_HEIGHT,
            scheme="Briggs", stability_class="D", estimated=False,
        )

    def coupling_fn_fixed_smith(x):
        src = SourceLocation(x=x[5], y=x[6], z=SOURCE_Z)
        return beam_path_coupling_matrix(
            src, beam_starts, beam_ends, wind,
            mixing_height=MIXING_HEIGHT,
            scheme="SMITH", stability_class="D", estimated=False,
        )

    def coupling_fn_est_draxler(x):
        src = SourceLocation(x=x[5], y=x[6], z=SOURCE_Z)
        return beam_path_coupling_matrix(
            src, beam_starts, beam_ends, wind,
            mixing_height=MIXING_HEIGHT,
            scheme="Draxler", estimated=True, log_params=x[:4],
            tan_gamma_H=tan_gamma_H, tan_gamma_V=tan_gamma_V,
        )

    def coupling_fn_est_smith(x):
        src = SourceLocation(x=x[5], y=x[6], z=SOURCE_Z)
        return beam_path_coupling_matrix(
            src, beam_starts, beam_ends, wind,
            mixing_height=MIXING_HEIGHT,
            scheme="SMITH", estimated=True, log_params=x[:4],
        )

    return {
        "Briggs_fixed": coupling_fn_fixed_briggs,
        "Smith_fixed": coupling_fn_fixed_smith,
        "Draxler_estimated": coupling_fn_est_draxler,
        "Smith_estimated": coupling_fn_est_smith,
    }[model]


def run_inversion(data: dict, model: str, key) -> dict:
    wind = WindField(speed=data["wind_speed"], direction=data["wind_direction"])
    coupling_fn = make_coupling_fn(
        data["beam_starts"], data["beam_ends"], wind, model,
        tan_gamma_H=data["tan_gamma_H"], tan_gamma_V=data["tan_gamma_V"],
    )

    priors = Priors(
        log_a_H_std=2.0, log_a_V_std=2.0, log_b_H_std=1.0, log_b_V_std=1.0,
        log_s_mean=-2.0, log_s_std=3.0,
        source_x_mean=0.0, source_x_std=300.0,
        source_y_mean=0.0, source_y_std=300.0,
        sigma2_alpha=2.0, sigma2_beta=1.0,
        background_std=5.0,
    )
    gibbs = GibbsSamplers(priors)

    n_beams = data["beam_starts"].shape[0]
    x_init = jnp.zeros(7)
    bg_init = jnp.zeros(n_beams)

    chains = mwg_scan(
        key,
        x_init=x_init,
        sigma2_init=1.0,
        background_init=bg_init,
        data=data["measurements"],
        coupling_fn=coupling_fn,
        priors=priors,
        gibbs=gibbs,
        step_size_init=0.01,
        adaptation="Optimal",
        iters=ITERS,
    )
    x_post = chains["x_chain"][BURN_IN:]
    return {
        "model": model,
        "src_x_median":   float(jnp.median(x_post[:, 5])),
        "src_y_median":   float(jnp.median(x_post[:, 6])),
        "log_s_median":   float(jnp.median(x_post[:, 4])),
        "src_x_q05":      float(jnp.quantile(x_post[:, 5], 0.05)),
        "src_x_q95":      float(jnp.quantile(x_post[:, 5], 0.95)),
        "src_y_q05":      float(jnp.quantile(x_post[:, 6], 0.05)),
        "src_y_q95":      float(jnp.quantile(x_post[:, 6], 0.95)),
        "accept_rate":    float(jnp.mean(chains["accept_chain"])),
    }


def plot_posteriors(results: list[dict], data: dict, out="reproduction/section5_posteriors.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot (uv sync --extra reproduction)")
        return

    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4), squeeze=False)
    for ax, r in zip(axes[0], results, strict=True):
        ax.set_title(r["model"].replace("_", "\n"))
        ax.errorbar(
            r["src_x_median"], r["src_y_median"],
            xerr=[[r["src_x_median"] - r["src_x_q05"]], [r["src_x_q95"] - r["src_x_median"]]],
            yerr=[[r["src_y_median"] - r["src_y_q05"]], [r["src_y_q95"] - r["src_y_median"]]],
            fmt="o", label="Posterior median ± 90%CI",
        )
        if "release_x" in data and "release_y" in data:
            ax.plot(float(data["release_x"]), float(data["release_y"]), "r*", markersize=12, label="True")
        ax.set_xlabel("Source x (m)")
        ax.set_ylabel("Source y (m)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    check_data()
    data = load_data()
    print(f"Loaded Source 1: {data['measurements'].shape} measurements")

    results = []
    for i, model in enumerate(MODELS):
        key_i = jax.random.fold_in(KEY, i)
        print(f"\n[{i+1}/{len(MODELS)}] Running {model} …")
        r = run_inversion(data, model, key_i)
        results.append(r)
        print(f"  src=({r['src_x_median']:.1f}, {r['src_y_median']:.1f}) m  "
              f"log_s={r['log_s_median']:.3f}  accept={r['accept_rate']:.3f}")

    print("\n--- Summary ---")
    for r in results:
        print(f"  {r['model']:22}  src=({r['src_x_median']:.1f}, {r['src_y_median']:.1f}) m")

    plot_posteriors(results, data)


if __name__ == "__main__":
    main()
