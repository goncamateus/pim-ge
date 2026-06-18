"""Shared utilities for §4 and §5 reproduction scripts.

Non-plotting code: constants, data generation, coupling functions,
MCMC runners, Chilbolton data loading.
"""

import math
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pim_ge import GibbsSamplers, Priors, SourceLocation, WindField, mwg_scan
from pim_ge.forward.plume import beam_path_coupling_matrix, temporal_gridfree_coupling_matrix
from pim_ge.forward.sensors import circle_of_sensors, temporal_sensors_measurements
from pim_ge.forward.wind import wind_direction_sinusoidal, wind_speed

# ── §4 constants ───────────────────────────────────────────────────────────────

S4_KEY = jax.random.PRNGKey(42)
TRUE_SRC_X = 50.0
TRUE_SRC_Y = 50.0
TRUE_SRC_Z = 5.0
SENSOR_HEIGHT = 1.0
S4_MIXING_HEIGHT = 500.0
NOISE_STD = math.sqrt(1e-6)
OUT_DIR = Path("reproduction/figures")
MEAS_ERROR_VAR = 1e-6

BL = dict(wdc_degrees=140, dpv_case=2, ser=0.00039, dts=50.0, ops=100, layout="grid")

DPV_PARAMS = {
    1: (1.4, 1.2, 0.9, 0.95),
    2: (1.0, 1.0, 1.0, 1.0),
    3: (0.9, 0.7, 0.8, 0.85),
}

FACTOR_LEVELS = {
    "WDC": [60, 140, 360],
    "DPV": [1, 2, 3],
    "SER": [0.000195, 0.00039, 0.00078],
    "DTS": [30, 50, 70],
    "OPS": [10, 100, 1000],
    "SL": ["line", "grid", "sline"],
}

MISSPEC_TRUE = (1.0, 1.0, 0.8, 0.8)
MISSPEC_RANGE = {
    "aH": [0.6, 0.8, 1.0, 1.2, 1.4],
    "bH": [0.6, 0.7, 0.8, 0.9, 1.0],
    "aV": [0.6, 0.8, 1.0, 1.2, 1.4],
    "bV": [0.6, 0.7, 0.8, 0.9, 1.0],
}
MISSPEC_TRUE_VAL = {"aH": 1.0, "bH": 0.8, "aV": 1.0, "bV": 0.8}

# ── §5 constants ───────────────────────────────────────────────────────────────

S5_KEY = jax.random.PRNGKey(0)
N_BEAMS = 7
SOURCE_Z = 0.3
S5_MIXING_HEIGHT = 200.0
S5_ITERS = 3000
S5_BURN_IN = 500

_POST = Path("Data/Chilbolton_data_files/Postprocessed")
_LOCS_FILE = _POST / "Sensor_reflector_locations/Chilbolton_instruments_location.pkl"
_SRCS_FILE = (
    _POST
    / "Source_locations_and_emission_rates/Chilbolton_sources_locations_and_emission_rates.pkl"
)

MODELS = [
    ("Briggs A", "Briggs", "A", False),
    ("Briggs B", "Briggs", "B", False),
    ("Briggs C", "Briggs", "C", False),
    ("Briggs D", "Briggs", "D", False),
    ("Briggs E", "Briggs", "E", False),
    ("Briggs F", "Briggs", "F", False),
    ("Smith B", "SMITH", "B", False),
    ("Smith C", "SMITH", "C", False),
    ("Smith D", "SMITH", "D", False),
    ("Smith est", "SMITH", "D", True),
    ("Draxler est", "Draxler", "D", True),
]


# ── §4 helpers ────────────────────────────────────────────────────────────────


def make_sensors(layout: str, dts: float) -> jnp.ndarray:
    """Build a sensor layout centred on the true source."""
    cx, cy = TRUE_SRC_X, TRUE_SRC_Y
    if layout == "circle":
        return circle_of_sensors(cx, cy, dts, 8, SENSOR_HEIGHT)
    elif layout == "grid":
        xs = cx + jnp.linspace(-dts / 2, dts / 2, 6)
        ys = cy + jnp.linspace(-dts / 2, dts / 2, 6)
        XX, YY = jnp.meshgrid(xs, ys)
        return jnp.stack([XX.ravel(), YY.ravel(), jnp.full(36, SENSOR_HEIGHT)], axis=1)
    elif layout == "sline":
        xs = cx + jnp.linspace(-dts / 2, dts / 2, 6)
        return jnp.stack([xs, jnp.full(6, cy), jnp.full(6, SENSOR_HEIGHT)], axis=1)
    else:  # "line"
        xs = cx + jnp.linspace(-dts / 2, dts / 2, 36)
        return jnp.stack([xs, jnp.full(36, cy), jnp.full(36, SENSOR_HEIGHT)], axis=1)


def make_wind(key, T: int, wdc_degrees: float) -> WindField:
    """Simulate a wind realization with a given WDC factor."""
    k_s, k_d = jax.random.split(key)
    std_rad = (wdc_degrees * math.pi) / 360.0
    return WindField(
        speed=wind_speed(k_s, T, mean=6.0, std=0.3, theta=0.1),
        direction=wind_direction_sinusoidal(
            k_d, T, mean=0.0, std=std_rad, theta=0.05, num_periods=2.0
        ),
    )


def make_log_params(aH, aV, bH, bV) -> jnp.ndarray:
    """Pack dispersion coefficients into log_params vector."""
    return jnp.array([math.log(aH), math.log(aV), math.log(bH), math.log(bV)])


def generate_data(key, sensors, wind, dpv_case: int, ser: float) -> jnp.ndarray:
    """Simulate synthetic sensor measurements for one DPV case."""
    aH, aV, bH, bV = DPV_PARAMS[dpv_case]
    src = SourceLocation(x=TRUE_SRC_X, y=TRUE_SRC_Y, z=TRUE_SRC_Z)
    A = temporal_gridfree_coupling_matrix(
        src,
        sensors,
        wind,
        mixing_height=S4_MIXING_HEIGHT,
        scheme="SMITH",
        estimated=True,
        log_params=make_log_params(aH, aV, bH, bV),
    )
    bg = jnp.full(sensors.shape[0], 0.5)
    return temporal_sensors_measurements(A, ser, bg, NOISE_STD, key)


def make_estimated_coupling(sensors, wind):
    """Build coupling_fn(x)->A that infers all dispersion params from x."""

    def fn(x):
        src = SourceLocation(x=x[5], y=x[6], z=TRUE_SRC_Z)
        return temporal_gridfree_coupling_matrix(
            src,
            sensors,
            wind,
            mixing_height=S4_MIXING_HEIGHT,
            scheme="SMITH",
            estimated=True,
            log_params=x[:4],
        )

    return fn


def make_fixed_coupling(sensors, wind, aH, aV, bH, bV):
    """Build coupling_fn(x)->A with dispersion params fixed."""
    lp = make_log_params(aH, aV, bH, bV)

    def fn(x):
        src = SourceLocation(x=x[5], y=x[6], z=TRUE_SRC_Z)
        return temporal_gridfree_coupling_matrix(
            src,
            sensors,
            wind,
            mixing_height=S4_MIXING_HEIGHT,
            scheme="SMITH",
            estimated=True,
            log_params=lp,
        )

    return fn


def make_priors(ser: float) -> Priors:
    """Build Priors for the §4 simulation study."""
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
    """Run mwg_scan and return post-burn-in posterior medians."""
    priors = make_priors(ser)
    gibbs = GibbsSamplers(priors)
    n = sensors.shape[0]
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


def build_scenario(factor: str, level) -> dict:
    """Build a BL scenario dict with one factor overridden."""
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
    """Run the full six-factor, three-level main-effects sweep with replication."""
    all_results = {}
    total = sum(len(lvls) for lvls in FACTOR_LEVELS.values()) * n_reps
    done = 0

    for factor, levels in FACTOR_LEVELS.items():
        all_results[factor] = {}
        for level in levels:
            all_results[factor][level] = []
            sc = build_scenario(factor, level)
            for rep in range(n_reps):
                k = jax.random.fold_in(S4_KEY, done)
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


def run_misspec_study(n_reps: int, iters: int, burn_in: int) -> dict:
    """Run the dispersion-parameter misspecification study (Figure 6)."""
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
        for val in vals:
            for rep in range(n_reps):
                k = jax.random.fold_in(S4_KEY, 10_000 + done)
                k_w, k_d, k_truth, k_est, k_mis = jax.random.split(k, 5)

                wind = make_wind(k_w, BL["ops"], BL["wdc_degrees"])
                src = SourceLocation(x=TRUE_SRC_X, y=TRUE_SRC_Y, z=TRUE_SRC_Z)
                A = temporal_gridfree_coupling_matrix(
                    src,
                    sensors,
                    wind,
                    mixing_height=S4_MIXING_HEIGHT,
                    scheme="SMITH",
                    estimated=True,
                    log_params=make_log_params(true_aH, true_aV, true_bH, true_bV),
                )
                bg = jnp.full(sensors.shape[0], 0.5)
                data = temporal_sensors_measurements(A, BL["ser"], bg, NOISE_STD, k_d)

                cfn_truth = make_fixed_coupling(sensors, wind, true_aH, true_aV, true_bH, true_bV)
                results[param][val]["truth"].append(
                    run_mcmc(k_truth, sensors, wind, data, BL["ser"], cfn_truth, iters, burn_in)
                )

                cfn_est = make_estimated_coupling(sensors, wind)
                results[param][val]["est"].append(
                    run_mcmc(k_est, sensors, wind, data, BL["ser"], cfn_est, iters, burn_in)
                )

                fixed = dict(aH=true_aH, aV=true_aV, bH=true_bH, bV=true_bV)
                fixed[param] = val
                cfn_mis = make_fixed_coupling(sensors, wind, **fixed)
                results[param][val]["misspec"].append(
                    run_mcmc(k_mis, sensors, wind, data, BL["ser"], cfn_mis, iters, burn_in)
                )

                done += 1
                print(f"  [misspec {done:3d}/{total}] {param}={val:.3f} rep={rep}")
    return results


# ── §5 helpers ────────────────────────────────────────────────────────────────


def _meas_file(src: int) -> Path:
    d = {
        1: "Source_1/Chilbolton_CH4_measurements_source_1.pkl",
        2: "Source_2/Chilbolton_CH4_measurements_source_2.pkl",
    }
    return _POST / d[src]


def _wind_file(src: int) -> Path:
    d = {
        1: "Source_1/Chilbolton_windfield_source_1.pkl",
        2: "Source_2/Chilbolton_windfield_source_2.pkl",
    }
    return _POST / d[src]


def check_data():
    """Verify required Chilbolton data files exist, exiting if not."""
    needed = [_LOCS_FILE, _SRCS_FILE, _meas_file(1), _wind_file(1), _meas_file(2), _wind_file(2)]
    missing = [f for f in needed if not f.exists()]
    if missing:
        print("=" * 70)
        print("DATA NOT FOUND")
        for f in missing:
            print(f"  missing: {f}")
        sys.exit(1)


def _pkl(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_data(source_num: int) -> dict:
    """Load Chilbolton measurements, wind field, and beam/source geometry."""
    meas_df = _pkl(_meas_file(source_num))
    wind_df = _pkl(_wind_file(source_num))
    locs = _pkl(_LOCS_FILE)
    srcs = _pkl(_SRCS_FILE)

    T = len(wind_df)
    arr = meas_df["Measurements"].values
    if arr.ndim > 1:
        arr = arr.squeeze()
    measurements = arr.reshape(T, N_BEAMS).astype(np.float32)

    sensor = np.array(locs["line_of_sight_sensor"], dtype=np.float32)
    beam_starts = np.tile(sensor, (N_BEAMS, 1))
    beam_ends = np.array([locs[f"reflector_{i}"] for i in range(1, N_BEAMS + 1)], dtype=np.float32)

    wind_speed_arr = wind_df["Average Speed"].values.astype(np.float32)
    wind_direction_arr = np.deg2rad(wind_df["Average Direction"].values).astype(np.float32)
    tan_gamma_H = float(wind_df["Average Tan_gamma Horizontal"].mean())
    tan_gamma_V = float(wind_df["Average Tan_gamma Vertical"].mean())

    src = srcs[f"source_{source_num}_location"]
    return {
        "source_num": source_num,
        "measurements": jnp.array(measurements),
        "beam_starts": jnp.array(beam_starts),
        "beam_ends": jnp.array(beam_ends),
        "wind_speed": jnp.array(wind_speed_arr),
        "wind_direction": jnp.array(wind_direction_arr),
        "tan_gamma_H": tan_gamma_H,
        "tan_gamma_V": tan_gamma_V,
        "release_x": float(src[0]),
        "release_y": float(src[1]),
        "release_z": float(src[2]),
        "release_rate": float(srcs[f"source_{source_num}_emission_rate"]),
    }


def make_coupling_fn(data: dict, label: str, scheme: str, stability_class: str, estimated: bool):
    """Build coupling_fn(x)->A for one Chilbolton dispersion model."""
    beam_starts = data["beam_starts"]
    beam_ends = data["beam_ends"]
    wind = WindField(speed=data["wind_speed"], direction=data["wind_direction"])
    tan_gamma_H = data["tan_gamma_H"]
    tan_gamma_V = data["tan_gamma_V"]

    if not estimated:

        def fn(x):
            src = SourceLocation(x=x[5], y=x[6], z=SOURCE_Z)
            return beam_path_coupling_matrix(
                src,
                beam_starts,
                beam_ends,
                wind,
                mixing_height=S5_MIXING_HEIGHT,
                scheme=scheme,
                stability_class=stability_class,
                estimated=False,
            )

    elif scheme == "Draxler":

        def fn(x):
            src = SourceLocation(x=x[5], y=x[6], z=SOURCE_Z)
            return beam_path_coupling_matrix(
                src,
                beam_starts,
                beam_ends,
                wind,
                mixing_height=S5_MIXING_HEIGHT,
                scheme="Draxler",
                estimated=True,
                log_params=x[:4],
                tan_gamma_H=tan_gamma_H,
                tan_gamma_V=tan_gamma_V,
            )

    else:

        def fn(x):
            src = SourceLocation(x=x[5], y=x[6], z=SOURCE_Z)
            return beam_path_coupling_matrix(
                src,
                beam_starts,
                beam_ends,
                wind,
                mixing_height=S5_MIXING_HEIGHT,
                scheme=scheme,
                estimated=True,
                log_params=x[:4],
            )

    return fn


def run_inversion(
    data: dict, label: str, scheme: str, stability_class: str, estimated: bool, key
) -> dict:
    """Run mwg_scan for one Chilbolton (data, model) pair and summarize the chain."""
    priors = Priors(
        log_a_H_std=2.0,
        log_a_V_std=2.0,
        log_b_H_std=1.0,
        log_b_V_std=1.0,
        log_s_mean=-4.0,
        log_s_std=3.0,
        source_x_mean=60.0,
        source_x_std=60.0,
        source_y_mean=60.0,
        source_y_std=60.0,
        sigma2_alpha=2.0,
        sigma2_beta=1.0,
        background_std=5.0,
    )
    gibbs = GibbsSamplers(priors)
    cfn = make_coupling_fn(data, label, scheme, stability_class, estimated)

    chains = mwg_scan(
        key,
        x_init=jnp.zeros(7),
        sigma2_init=1.0,
        background_init=jnp.zeros(N_BEAMS),
        data=data["measurements"],
        coupling_fn=cfn,
        priors=priors,
        gibbs=gibbs,
        step_size_init=0.01,
        adaptation="Optimal",
        iters=S5_ITERS,
    )
    xp = chains["x_chain"][S5_BURN_IN:]
    return {
        "label": label,
        "s_samples": np.array(jnp.exp(xp[:, 4])),
        "src_x_samples": np.array(xp[:, 5]),
        "src_y_samples": np.array(xp[:, 6]),
        "s_median": float(jnp.median(jnp.exp(xp[:, 4]))),
        "src_x_median": float(jnp.median(xp[:, 5])),
        "src_y_median": float(jnp.median(xp[:, 6])),
        "accept_rate": float(jnp.mean(chains["accept_chain"])),
    }
