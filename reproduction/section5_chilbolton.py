"""Reproduce §5 Chilbolton real-data case study of Newman et al. (2024).

Chilbolton uses Open-Path FTIR beam sensors with 7 reflector paths (40 cm spacing).
This script inverts Source 1 with 4 dispersion models:
    1. Fixed Briggs class D
    2. Fixed Smith class D
    3. Estimated Draxler (a/b sampled)
    4. Estimated Smith (a/b sampled)

Data availability
-----------------
The preprocessed Chilbolton data is NOT included in this repository.
Download the original data from:

    https://github.com/NewmanTHP/Probabilistic-Inversion-Modeling-of-Gas-Emissions

Place the files under `Data/` in the project root:
    Data/
        Chilbolton_Source1_preprocessed.npz   (or equivalent)
        chilbolton_beam_geometry.npz

The npz files should contain:
    measurements : (T, N_beams) float32 [ppm·m]  path-integrated concentrations
    beam_starts  : (N_beams, 3) float32 [m]      beam start coordinates
    beam_ends    : (N_beams, 3) float32 [m]      beam end coordinates
    wind_speed   : (T,) float32 [m/s]
    wind_direction : (T,) float32 [rad]
    release_rate : float  true emission rate [kg/s]
    release_x    : float  true source x [m]
    release_y    : float  true source y [m]
    release_z    : float  true source z [m]
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from pim_ge import GibbsSamplers, Priors, SourceLocation, WindField, mwg_scan
from pim_ge.forward.plume import beam_path_coupling_matrix

# --- Configuration ------------------------------------------------------------
DATA_DIR     = Path("Data")
SOURCE1_FILE = DATA_DIR / "Chilbolton_Source1_preprocessed.npz"
MODELS       = ["Briggs_fixed", "Smith_fixed", "Draxler_estimated", "Smith_estimated"]
ITERS        = 5000
BURN_IN      = 1000
MIXING_HEIGHT = 200.0   # [m] — Chilbolton experiment
KEY           = jax.random.PRNGKey(0)


def check_data():
    if not SOURCE1_FILE.exists():
        print("=" * 70)
        print("DATA NOT FOUND")
        print("=" * 70)
        print(__doc__)
        sys.exit(1)


def load_data(path: Path) -> dict:
    import numpy as np
    d = np.load(path)
    return {k: jnp.array(d[k]) for k in d.files}


def make_coupling_fn(beam_starts, beam_ends, wind, model: str):
    """Return coupling_fn(x) -> A (T, N_beams) [ppm·m per kg/s]."""
    def coupling_fn_fixed_briggs(x):
        src = SourceLocation(x=x[5], y=x[6], z=float(x[7]) if x.shape[0] > 7 else 1.0)
        return beam_path_coupling_matrix(
            src, beam_starts, beam_ends, wind,
            mixing_height=MIXING_HEIGHT,
            scheme="Briggs", stability_class="D", estimated=False,
        )

    def coupling_fn_fixed_smith(x):
        src = SourceLocation(x=x[5], y=x[6], z=1.0)
        return beam_path_coupling_matrix(
            src, beam_starts, beam_ends, wind,
            mixing_height=MIXING_HEIGHT,
            scheme="SMITH", stability_class="D", estimated=False,
        )

    def coupling_fn_est_draxler(x):
        src = SourceLocation(x=x[5], y=x[6], z=1.0)
        return beam_path_coupling_matrix(
            src, beam_starts, beam_ends, wind,
            mixing_height=MIXING_HEIGHT,
            scheme="Draxler", estimated=True, log_params=x[:4],
        )

    def coupling_fn_est_smith(x):
        src = SourceLocation(x=x[5], y=x[6], z=1.0)
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
    coupling_fn = make_coupling_fn(data["beam_starts"], data["beam_ends"], wind, model)

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
    data = load_data(SOURCE1_FILE)
    print(f"Loaded {SOURCE1_FILE}: {data['measurements'].shape} measurements")

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
