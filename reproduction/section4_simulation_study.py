"""Reproduce §4 simulation study of Newman et al. (2024).

3-factor sweep: DPV × WDC × SER (12 scenarios).

DPV — dispersion parameter variant:
    "fixed"     : Briggs class D, parameters fixed
    "estimated" : Smith class D, a/b inferred from data (x[0:4])

WDC — wind direction condition:
    "constant"     : OU around fixed mean direction
    "sinusoidal"   : mean direction follows a sinusoidal sweep

SER — source emission rate [kg/s]: {0.001, 0.01, 0.1}

Outputs: reproduction/section4_results.csv, printed RMSE table.

Set ITERS=5000, BURN_IN=1000 for paper-quality results (much slower).
"""
import csv
import math
import time

import jax
import jax.numpy as jnp

from pim_ge import GibbsSamplers, Priors, SourceLocation, WindField, mwg_scan
from pim_ge.forward.plume import temporal_gridfree_coupling_matrix
from pim_ge.forward.sensors import circle_of_sensors, temporal_sensors_measurements
from pim_ge.forward.wind import wind_direction, wind_direction_sinusoidal, wind_speed

# --- Configuration -----------------------------------------------------------
KEY_BASE      = jax.random.PRNGKey(42)
ITERS         = 2000    # increase to 5000 for paper quality
BURN_IN       = 500     # increase to 1000 for paper quality
TRUE_SRC_X    = 50.0    # [m]
TRUE_SRC_Y    = 30.0    # [m]
TRUE_SRC_Z    = 1.5     # [m]
T             = 100     # timesteps
N_SENSORS     = 8
SENSOR_RADIUS = 200.0   # [m] circle radius
SENSOR_HEIGHT = 1.0     # [m]
MIXING_HEIGHT = 500.0   # [m]
NOISE_STD     = 0.5     # [ppm]
OUT_CSV       = "reproduction/section4_results.csv"

DPV_VARIANTS = ["fixed", "estimated"]
WDC_VARIANTS = ["constant", "sinusoidal"]
SER_VALUES   = [0.001, 0.01, 0.1]

# True Smith D log-params: log([a_H, a_V, b_H, b_V]) = log([0.32, 0.22, 0.78, 0.78])
TRUE_LOG_PARAMS = jnp.array([
    math.log(0.32), math.log(0.22), math.log(0.78), math.log(0.78),
])


def make_wind(key, wdc: str) -> WindField:
    k_s, k_d = jax.random.split(key)
    speed = wind_speed(k_s, T, mean=2.5, std=0.3, theta=0.1)
    if wdc == "sinusoidal":
        direction = wind_direction_sinusoidal(k_d, T, mean=0.0, std=0.4, theta=0.05, num_periods=2.0)
    else:
        direction = wind_direction(k_d, T, mean=0.0, std=0.3, theta=0.05)
    return WindField(speed=speed, direction=direction)


def make_coupling_fn(sensors, wind, dpv: str):
    """Return coupling_fn(x) -> A (T, N) [ppm per kg/s]."""
    if dpv == "fixed":
        def coupling_fn(x):
            src = SourceLocation(x=x[5], y=x[6], z=TRUE_SRC_Z)
            return temporal_gridfree_coupling_matrix(
                src, sensors, wind,
                mixing_height=MIXING_HEIGHT,
                scheme="Briggs", stability_class="D",
                estimated=False,
            )
    else:  # estimated Smith
        def coupling_fn(x):
            src = SourceLocation(x=x[5], y=x[6], z=TRUE_SRC_Z)
            return temporal_gridfree_coupling_matrix(
                src, sensors, wind,
                mixing_height=MIXING_HEIGHT,
                scheme="SMITH", stability_class="D",
                estimated=True, log_params=x[:4],
            )
    return coupling_fn


def generate_data(key, sensors, wind, dpv: str, ser: float) -> jnp.ndarray:
    """Synthetic data using the true source and true dispersion params."""
    src_true = SourceLocation(x=TRUE_SRC_X, y=TRUE_SRC_Y, z=TRUE_SRC_Z)
    if dpv == "fixed":
        A_true = temporal_gridfree_coupling_matrix(
            src_true, sensors, wind,
            mixing_height=MIXING_HEIGHT,
            scheme="Briggs", stability_class="D",
        )
    else:
        A_true = temporal_gridfree_coupling_matrix(
            src_true, sensors, wind,
            mixing_height=MIXING_HEIGHT,
            scheme="SMITH", stability_class="D",
            estimated=True, log_params=TRUE_LOG_PARAMS,
        )
    background = jnp.full((sensors.shape[0],), 0.5)  # 0.5 ppm constant bg
    return temporal_sensors_measurements(A_true, ser, background, NOISE_STD, key)


def make_priors(dpv: str) -> Priors:
    """Priors on x = [log_a_H, log_a_V, log_b_H, log_b_V, log_s, src_x, src_y]."""
    return Priors(
        log_a_H_mean=0.0, log_a_H_std=2.0,
        log_a_V_mean=0.0, log_a_V_std=2.0,
        log_b_H_mean=0.0, log_b_H_std=1.0,
        log_b_V_mean=0.0, log_b_V_std=1.0,
        log_s_mean=-2.0, log_s_std=3.0,
        source_x_mean=0.0, source_x_std=200.0,
        source_y_mean=0.0, source_y_std=200.0,
        sigma2_alpha=2.0, sigma2_beta=1.0,
        background_std=2.0,
    )


def run_scenario(key, dpv: str, wdc: str, ser: float) -> dict:
    k_wind, k_data, k_mcmc = jax.random.split(key, 3)

    sensors = circle_of_sensors(0.0, 0.0, SENSOR_RADIUS, N_SENSORS, SENSOR_HEIGHT)
    wind = make_wind(k_wind, wdc)
    data = generate_data(k_data, sensors, wind, dpv, ser)

    coupling_fn = make_coupling_fn(sensors, wind, dpv)
    priors = make_priors(dpv)
    gibbs = GibbsSamplers(priors)

    # Initial x: dispersion at zeros (exp=1), source at grid center, log_s near truth
    x_init = jnp.array([0.0, 0.0, 0.0, 0.0, math.log(ser), 0.0, 0.0])
    sigma2_init = 1.0
    bg_init = jnp.zeros(N_SENSORS)

    t0 = time.time()
    chains = mwg_scan(
        k_mcmc,
        x_init=x_init,
        sigma2_init=sigma2_init,
        background_init=bg_init,
        data=data,
        coupling_fn=coupling_fn,
        priors=priors,
        gibbs=gibbs,
        step_size_init=0.01,
        adaptation="Optimal",
        target_accept=0.574,
        iters=ITERS,
    )
    elapsed = time.time() - t0

    # Post-burnin samples
    x_post = chains["x_chain"][BURN_IN:]    # (ITERS-BURN_IN, 7)
    est_log_s  = float(jnp.median(x_post[:, 4]))
    est_src_x  = float(jnp.median(x_post[:, 5]))
    est_src_y  = float(jnp.median(x_post[:, 6]))
    accept_rate = float(jnp.mean(chains["accept_chain"]))

    loc_error = math.sqrt((est_src_x - TRUE_SRC_X)**2 + (est_src_y - TRUE_SRC_Y)**2)
    s_error   = abs(est_log_s - math.log(ser))

    return {
        "dpv": dpv, "wdc": wdc, "ser": ser,
        "true_src_x": TRUE_SRC_X, "true_src_y": TRUE_SRC_Y,
        "true_log_s": round(math.log(ser), 4),
        "est_src_x": round(est_src_x, 2), "est_src_y": round(est_src_y, 2),
        "est_log_s": round(est_log_s, 4),
        "loc_error_m": round(loc_error, 2),
        "log_s_error": round(s_error, 4),
        "accept_rate": round(accept_rate, 3),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    scenarios = [
        (dpv, wdc, ser)
        for dpv in DPV_VARIANTS
        for wdc in WDC_VARIANTS
        for ser in SER_VALUES
    ]

    print(f"Running {len(scenarios)} scenarios, {ITERS} iters each …")
    print(f"{'DPV':12} {'WDC':12} {'SER':8} {'loc_err(m)':12} {'|Δlog_s|':10} {'accept':8} {'t(s)':6}")
    print("-" * 72)

    results = []
    for i, (dpv, wdc, ser) in enumerate(scenarios):
        key = jax.random.fold_in(KEY_BASE, i)
        r = run_scenario(key, dpv, wdc, ser)
        results.append(r)
        print(f"{dpv:12} {wdc:12} {ser:<8.3f} {r['loc_error_m']:<12.2f} "
              f"{r['log_s_error']:<10.4f} {r['accept_rate']:<8.3f} {r['elapsed_s']:.1f}")

    # RMSE summary per DPV
    print("\n--- RMSE by DPV ---")
    for dpv in DPV_VARIANTS:
        sub = [r for r in results if r["dpv"] == dpv]
        rmse_loc = math.sqrt(sum(r["loc_error_m"]**2 for r in sub) / len(sub))
        rmse_s   = math.sqrt(sum(r["log_s_error"]**2 for r in sub) / len(sub))
        print(f"  {dpv:12}  RMSE loc={rmse_loc:.2f} m   RMSE |Δlog_s|={rmse_s:.4f}")

    # Write CSV
    fieldnames = list(results[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
