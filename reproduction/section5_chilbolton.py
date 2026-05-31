"""Reproduce §5 Chilbolton real-data case study of Newman et al. (2024).

Figures produced:
    Figure 7 — beam paths and source positions (spatial layout)
    Figure 8 — posterior emission rate + source location by model (boxplots)
    Figure 9 — source location density contours (2D KDE, Source 1 & 2)

Models compared:
    Briggs A/B/C/D/E/F (fixed stability class)
    Smith  B/C/D       (fixed stability class)
    Smith  estimated   (a/b inferred)
    Draxler estimated  (a/b inferred, uses data tan_gamma)

Data expected at:
    Data/Chilbolton_data_files/Postprocessed/
        Source_1/Chilbolton_CH4_measurements_source_1.pkl
        Source_1/Chilbolton_windfield_source_1.pkl
        Source_2/Chilbolton_CH4_measurements_source_2.pkl
        Source_2/Chilbolton_windfield_source_2.pkl
        Sensor_reflector_locations/Chilbolton_instruments_location.pkl
        Source_locations_and_emission_rates/...pkl

Download from:
    https://github.com/NewmanTHP/Probabilistic-Inversion-Modeling-of-Gas-Emissions
"""

import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.stats import gaussian_kde

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from pim_ge import GibbsSamplers, Priors, SourceLocation, WindField, mwg_scan
from pim_ge.forward.plume import beam_path_coupling_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
_POST = Path("Data/Chilbolton_data_files/Postprocessed")
_LOCS_FILE = _POST / "Sensor_reflector_locations/Chilbolton_instruments_location.pkl"
_SRCS_FILE = (
    _POST
    / "Source_locations_and_emission_rates/Chilbolton_sources_locations_and_emission_rates.pkl"
)


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


# ── Configuration ─────────────────────────────────────────────────────────────
N_BEAMS = 7
SOURCE_Z = 0.3  # [m] release height
MIXING_HEIGHT = 200.0
ITERS = 3000
BURN_IN = 500
KEY = jax.random.PRNGKey(0)

# All models: (label, scheme, stability_class, estimated)
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

# Distinct colours per model (tab10 + tab20 fallback)
_CMAP = plt.cm.get_cmap("tab10") if HAS_MPL else None
MODEL_COLORS = {m[0]: f"C{i % 10}" for i, m in enumerate(MODELS)}


# ── Data loading ──────────────────────────────────────────────────────────────


def check_data():
    needed = [_LOCS_FILE, _SRCS_FILE, _meas_file(1), _wind_file(1), _meas_file(2), _wind_file(2)]
    missing = [f for f in needed if not f.exists()]
    if missing:
        print("=" * 70)
        print("DATA NOT FOUND")
        for f in missing:
            print(f"  missing: {f}")
        print(__doc__)
        sys.exit(1)


def _pkl(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_data(source_num: int) -> dict:
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

    wind_speed = wind_df["Average Speed"].values.astype(np.float32)
    wind_direction = np.deg2rad(wind_df["Average Direction"].values).astype(np.float32)
    tan_gamma_H = float(wind_df["Average Tan_gamma Horizontal"].mean())
    tan_gamma_V = float(wind_df["Average Tan_gamma Vertical"].mean())

    src = srcs[f"source_{source_num}_location"]
    return {
        "source_num": source_num,
        "measurements": jnp.array(measurements),
        "beam_starts": jnp.array(beam_starts),
        "beam_ends": jnp.array(beam_ends),
        "wind_speed": jnp.array(wind_speed),
        "wind_direction": jnp.array(wind_direction),
        "tan_gamma_H": tan_gamma_H,
        "tan_gamma_V": tan_gamma_V,
        "release_x": float(src[0]),
        "release_y": float(src[1]),
        "release_z": float(src[2]),
        "release_rate": float(srcs[f"source_{source_num}_emission_rate"]),
    }


# ── Coupling functions ────────────────────────────────────────────────────────


def make_coupling_fn(data: dict, label: str, scheme: str, stability_class: str, estimated: bool):
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
                mixing_height=MIXING_HEIGHT,
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
                mixing_height=MIXING_HEIGHT,
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
                mixing_height=MIXING_HEIGHT,
                scheme=scheme,
                estimated=True,
                log_params=x[:4],
            )

    return fn


# ── Inversion ─────────────────────────────────────────────────────────────────


def run_inversion(
    data: dict, label: str, scheme: str, stability_class: str, estimated: bool, key
) -> dict:
    n_beams = N_BEAMS
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
        background_init=jnp.zeros(n_beams),
        data=data["measurements"],
        coupling_fn=cfn,
        priors=priors,
        gibbs=gibbs,
        step_size_init=0.01,
        adaptation="Optimal",
        iters=ITERS,
    )
    xp = chains["x_chain"][BURN_IN:]  # (N_post, 7)
    return {
        "label": label,
        "s_samples": np.array(jnp.exp(xp[:, 4])),  # emission rate kg/s
        "src_x_samples": np.array(xp[:, 5]),
        "src_y_samples": np.array(xp[:, 6]),
        "s_median": float(jnp.median(jnp.exp(xp[:, 4]))),
        "src_x_median": float(jnp.median(xp[:, 5])),
        "src_y_median": float(jnp.median(xp[:, 6])),
        "accept_rate": float(jnp.mean(chains["accept_chain"])),
    }


# ── Figure 7: beam geometry ───────────────────────────────────────────────────


def plot_figure7(data1: dict, data2: dict):
    if not HAS_MPL:
        return
    locs = _pkl(_LOCS_FILE)
    srcs = _pkl(_SRCS_FILE)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(
        "Figure 7 — Chilbolton Sensor / Beam / Source Layout", fontsize=11, fontweight="bold"
    )

    cmap = plt.cm.tab10(np.linspace(0, 1, N_BEAMS))
    sensor = np.array(locs["line_of_sight_sensor"])
    for i in range(N_BEAMS):
        ref = np.array(locs[f"reflector_{i + 1}"])
        ax.plot(
            [sensor[0], ref[0]],
            [sensor[1], ref[1]],
            color=cmap[i],
            lw=2,
            label=f"Beam {i + 1}",
            alpha=0.85,
        )
        ax.plot(ref[0], ref[1], "o", color=cmap[i], markersize=7)

    ax.plot(sensor[0], sensor[1], "ks", markersize=10, zorder=6, label="Sensor")

    for sn, marker, colour in [
        (1, "*", "red"),
        (2, "*", "blue"),
        (3, "^", "orange"),
        (4, "^", "purple"),
    ]:
        loc = srcs[f"source_{sn}_location"]
        ax.plot(
            loc[0],
            loc[1],
            marker=marker,
            color=colour,
            markersize=12,
            zorder=7,
            label=f"Source {sn}",
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_aspect("equal")
    fig.tight_layout()
    out = "reproduction/fig7_chilbolton_layout.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── Figure 8: posterior comparison boxplots ───────────────────────────────────


def plot_figure8(results1: list, results2: list, data1: dict, data2: dict):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    fig.suptitle(
        "Figure 8 — Source Estimation: All Models (Source 1 left, Source 2 right)",
        fontsize=11,
        fontweight="bold",
    )

    row_keys = ["s_samples", "src_x_samples", "src_y_samples"]
    row_labels = [r"$s$ (kg/s)", r"$x_{src}$ (m)", r"$y_{src}$ (m)"]
    true_keys = ["release_rate", "release_x", "release_y"]

    for col, (results, data) in enumerate([(results1, data1), (results2, data2)]):
        labels = [r["label"] for r in results]
        for row, (rk, rl, tk) in enumerate(zip(row_keys, row_labels, true_keys)):
            ax = axes[row, col]
            boxes = [r[rk] for r in results]
            bp = ax.boxplot(
                boxes,
                labels=labels,
                patch_artist=True,
                boxprops=dict(facecolor="lightsteelblue", alpha=0.75),
                medianprops=dict(color="navy", lw=2),
                flierprops=dict(marker=".", markersize=2),
                whiskerprops=dict(lw=1),
                capprops=dict(lw=1),
            )

            for patch, r in zip(bp["boxes"], results):
                patch.set_facecolor(MODEL_COLORS[r["label"]])
                patch.set_alpha(0.65)

            ax.axhline(data[tk], color="red", ls="--", lw=1.5, alpha=0.9, label="True value")
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_ylabel(rl, fontsize=8)
            if row == 0:
                ax.set_title(f"Source {data['source_num']}", fontsize=9, fontweight="bold")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "reproduction/fig8_source_estimation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── Figure 9: 2D KDE density contours ────────────────────────────────────────


def plot_figure9(results1: list, results2: list, data1: dict, data2: dict):
    if not HAS_MPL:
        return
    locs = _pkl(_LOCS_FILE)
    sensor = np.array(locs["line_of_sight_sensor"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Figure 9 — Source Location Density Contours", fontsize=11, fontweight="bold")

    for ax, results, data in [(axes[0], results1, data1), (axes[1], results2, data2)]:
        # beam paths
        cmap_b = plt.cm.tab10(np.linspace(0, 1, N_BEAMS))
        for i in range(N_BEAMS):
            ref = np.array(locs[f"reflector_{i + 1}"])
            ax.plot([sensor[0], ref[0]], [sensor[1], ref[1]], color=cmap_b[i], lw=1, alpha=0.4)

        ax.plot(sensor[0], sensor[1], "ks", markersize=8, zorder=6)

        # KDE contours per model
        legend_patches = []
        for r in results:
            xs = r["src_x_samples"]
            ys = r["src_y_samples"]
            try:
                kde = gaussian_kde(np.vstack([xs, ys]))
                # grid for contour
                xlo, xhi = xs.min() - 5, xs.max() + 5
                ylo, yhi = ys.min() - 5, ys.max() + 5
                xg = np.linspace(xlo, xhi, 60)
                yg = np.linspace(ylo, yhi, 60)
                XX, YY = np.meshgrid(xg, yg)
                Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(60, 60)
                # 50% and 90% credible contours
                levels_pct = [0.50, 0.90]
                Z_sorted = np.sort(Z.ravel())[::-1]
                cdf = np.cumsum(Z_sorted) / Z_sorted.sum()
                level_vals = [Z_sorted[np.searchsorted(cdf, p)] for p in levels_pct]
                color = MODEL_COLORS[r["label"]]
                ax.contour(
                    xg,
                    yg,
                    Z,
                    levels=level_vals[::-1],
                    colors=[color],
                    linewidths=[0.8, 1.5],
                    alpha=0.85,
                )
                legend_patches.append(mpatches.Patch(color=color, label=r["label"], alpha=0.8))
            except Exception:
                pass

        # True source
        ax.plot(
            data["release_x"], data["release_y"], "r*", markersize=14, zorder=8, label="True source"
        )
        legend_patches.append(mpatches.Patch(color="red", label="True source"))

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Source {data['source_num']} (contours: 50% & 90% CI)", fontsize=9)
        ax.legend(handles=legend_patches, fontsize=6, loc="upper left", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "reproduction/fig9_location_contours.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    check_data()
    data1 = load_data(1)
    data2 = load_data(2)
    print(f"Source 1: {data1['measurements'].shape}  Source 2: {data2['measurements'].shape}")

    plot_figure7(data1, data2)

    all_results = {1: [], 2: []}
    for src_num, data in [(1, data1), (2, data2)]:
        print(f"\n── Source {src_num} ──")
        for i, (label, scheme, cls, estimated) in enumerate(MODELS):
            key_i = jax.random.fold_in(KEY, (src_num - 1) * 100 + i)
            print(f"  [{i + 1}/{len(MODELS)}] {label} ...", end=" ", flush=True)
            r = run_inversion(data, label, scheme, cls, estimated, key_i)
            all_results[src_num].append(r)
            print(
                f"src=({r['src_x_median']:.1f}, {r['src_y_median']:.1f})  "
                f"s={r['s_median']:.2e}  accept={r['accept_rate']:.2f}"
            )

    print("\n── Plotting ──")
    plot_figure8(all_results[1], all_results[2], data1, data2)
    plot_figure9(all_results[1], all_results[2], data1, data2)


if __name__ == "__main__":
    main()
