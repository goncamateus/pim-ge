"""Interactive tuner for the 3D Gaussian plume — live preview, then export.

Drag sidebar parameters (mirrors the CLI flags of fixed_source.py /
export_plume_npz.py), watch the 3-panel preview update, then hit Export to
write the same .npz (Isaac Sim bridge artifact) and .mp4 (animation) using
the exact parameters shown.

Usage:
    uv run streamlit run examples/plume_ui.py
"""

import argparse
import time

import export_plume_npz
import fixed_source
import jax
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from _viz import (
    STABILITY_LABELS,
    build_figure,
    cloud_rgba,
    draw_3d_scatter,
    draw_xy_panel,
    draw_xz_panel,
    frame_title,
    init_colorbars,
    save_or_show,
    scatter_mask,
    setup_axes3d,
)
from matplotlib.colors import LogNorm

from pim_ge.forward.wind import wind_direction, wind_speed

st.set_page_config(page_title="Plume Tuner", layout="wide")


@st.cache_data(show_spinner="Computing plume...")
def compute_cached(
    stability_class,
    frames,
    fps,
    seed,
    emission_rate,
    source_z,
    mixing_height,
    start_x,
    end_x,
    nx,
    start_y,
    end_y,
    ny,
    start_z,
    end_z,
    nz,
    speed_mean,
    speed_std,
    speed_theta,
    dir_mean,
    dir_std,
    dir_theta,
    jet_speed,
    jet_angle,
    jet_diameter,
):
    """Cached wrapper around export_plume_npz.compute(), keyed on every scalar param.

    argparse.Namespace isn't hashable, so it can't be the cache key itself —
    the individual scalars are, and the Namespace is rebuilt fresh each call.
    """
    args = argparse.Namespace(
        stability_class=stability_class,
        frames=frames,
        fps=fps,
        seed=seed,
        emission_rate=emission_rate,
        source_z=source_z,
        mixing_height=mixing_height,
        start_x=start_x,
        end_x=end_x,
        nx=nx,
        start_y=start_y,
        end_y=end_y,
        ny=ny,
        start_z=start_z,
        end_z=end_z,
        nz=nz,
        speed_mean=speed_mean,
        speed_std=speed_std,
        speed_theta=speed_theta,
        dir_mean=dir_mean,
        dir_std=dir_std,
        dir_theta=dir_theta,
        jet_speed=jet_speed,
        jet_angle=jet_angle,
        jet_diameter=jet_diameter,
    )
    conc, x, y, z, source, peak = export_plume_npz.compute(args)

    key = jax.random.PRNGKey(seed)
    key_speed, key_dir = jax.random.split(key)
    speeds = np.asarray(
        wind_speed(key_speed, frames, mean=speed_mean, std=speed_std, theta=speed_theta)
    )
    directions = np.asarray(
        wind_direction(key_dir, frames, mean=dir_mean, std=dir_std, theta=dir_theta)
    )
    return conc, x, y, z, source, peak, speeds, directions


def build_preview_scaffold(conc_all, x, y, z):
    """Build the figure/axes/colorbars/norm once, reused across every drawn frame.

    Rebuilding this per frame (as a naive per-frame render would) redraws the
    colorbars every time, which is too slow for a Play loop — build once here,
    then only redraw the parts `draw_preview_frame` touches.
    """
    _, _, ny_, _ = conc_all.shape
    global_peak = conc_all.max()
    VMIN = max(global_peak * 0.001, 0.01)
    NORM = LogNorm(vmin=VMIN, vmax=global_peak)
    CMAP = plt.colormaps["inferno"]

    Xg, Yg, Zg = np.asarray(x), np.asarray(y), np.asarray(z)
    XXg, YYg = np.meshgrid(Xg, Yg, indexing="ij")
    XX, YY, ZZ = np.meshgrid(Xg, Yg, Zg, indexing="ij")

    fig, ax3, ax_xy, ax_xz, title = build_figure()
    fp0 = conc_all[0].max(axis=2)
    xz0 = conc_all[0][:, ny_ // 2, :]
    init_colorbars(ax_xy, ax_xz, Xg, Yg, Zg, fp0, xz0, NORM)

    return {
        "fig": fig,
        "ax3": ax3,
        "ax_xy": ax_xy,
        "ax_xz": ax_xz,
        "title": title,
        "NORM": NORM,
        "CMAP": CMAP,
        "VMIN": VMIN,
        "Xg": Xg,
        "Yg": Yg,
        "Zg": Zg,
        "XXg": XXg,
        "YYg": YYg,
        "XX": XX,
        "YY": YY,
        "ZZ": ZZ,
        "ny": ny_,
    }


def draw_preview_frame(scaffold, conc_all, source, speeds, directions, t, stability_class, core_frac):
    """Redraw frame `t` onto an existing scaffold's axes (mirrors build_animation's update(t))."""
    Xg, Yg, Zg = scaffold["Xg"], scaffold["Yg"], scaffold["Zg"]
    NORM, CMAP, VMIN = scaffold["NORM"], scaffold["CMAP"], scaffold["VMIN"]

    conc_3d = conc_all[t]
    conc_flat = conc_3d.ravel()
    peak = conc_flat.max()
    idx, cm, threshold = scatter_mask(conc_flat, peak, core_frac, VMIN)
    rgba = cloud_rgba(
        cm, CMAP, NORM, threshold, peak, alpha_lo=0.45, alpha_range=0.5, clip_lo=0.25, clip_hi=0.95
    )
    footprint = conc_3d.max(axis=2)

    ax3 = scaffold["ax3"]
    ax3.cla()
    setup_axes3d(
        ax3,
        xlim=(Xg[0] - 10, Xg[-1] + 10),
        ylim=(Yg[0] - 10, Yg[-1] + 10),
        zlim=(Zg[0] - 10, Zg[-1] + 10),
        elev=24,
        azim=-55,
    )
    xp, yp, zp = scaffold["XX"].ravel(), scaffold["YY"].ravel(), scaffold["ZZ"].ravel()
    draw_3d_scatter(ax3, source, xp, yp, zp, idx, rgba, 20, scaffold["XXg"], scaffold["YYg"], footprint)
    draw_xy_panel(scaffold["ax_xy"], Xg, Yg, footprint, NORM, source.x, source.y)
    draw_xz_panel(scaffold["ax_xz"], Xg, Zg, conc_3d[:, scaffold["ny"] // 2, :], NORM, source.z)

    deg = np.degrees(float(directions[t])) % 360
    spd = float(speeds[t])
    scaffold["title"].set_text(
        frame_title(STABILITY_LABELS[stability_class], t, len(conc_all), deg, spd, peak)
    )


class _Source:
    """Plain (x, y, z) holder matching SourceLocation's attribute access."""

    def __init__(self, xyz):
        self.x, self.y, self.z = float(xyz[0]), float(xyz[1]), float(xyz[2])


st.title("Plume Tuner")

with st.sidebar:
    st.header("Simulation")
    stability_class = st.selectbox(
        "Stability class", list("ABCDEF"), format_func=lambda c: STABILITY_LABELS[c]
    )
    frames = st.number_input("Frames", min_value=2, max_value=1000, value=100)
    fps = st.number_input("FPS", min_value=1, max_value=60, value=10)
    seed = st.number_input("Seed", min_value=0, value=0)

    st.header("Source & Emission")
    emission_rate = st.number_input("Emission rate [kg/s]", value=0.9, format="%.3f")
    source_z = st.number_input("Source height z [m]", value=25.0)
    mixing_height = st.number_input("Mixing height [m]", value=300.0)

    st.header("Grid — x")
    start_x = st.number_input("start_x", value=0.0)
    end_x = st.number_input("end_x", value=50.0)
    nx = st.number_input("nx", min_value=2, max_value=100, value=40)

    st.header("Grid — y")
    start_y = st.number_input("start_y", value=-25.0)
    end_y = st.number_input("end_y", value=25.0)
    ny = st.number_input("ny", min_value=2, max_value=100, value=40)

    st.header("Grid — z")
    start_z = st.number_input("start_z", value=0.0)
    end_z = st.number_input("end_z", value=50.0)
    nz = st.number_input("nz", min_value=2, max_value=100, value=35)

    st.header("OU wind — speed")
    speed_mean = st.number_input("speed_mean", value=2.0)
    speed_std = st.number_input("speed_std", value=2.0)
    speed_theta = st.number_input("speed_theta", value=0.5)

    st.header("OU wind — direction")
    dir_mean = st.number_input("dir_mean", value=0.0)
    dir_std = st.number_input("dir_std", value=0.05, format="%.3f")
    dir_theta = st.number_input("dir_theta", value=0.01, format="%.3f")

    st.header("Jet")
    enable_jet = st.checkbox("Enable jet", value=True)
    if enable_jet:
        jet_speed = st.number_input("jet_speed", value=20.0)
        jet_angle = st.number_input("jet_angle [deg]", value=15.0)
        jet_diameter = st.number_input("jet_diameter", value=0.2)
    else:
        jet_speed, jet_angle, jet_diameter = 0.0, 0.0, 0.0

    st.header("Preview")
    core_frac = st.number_input("core_frac", value=0.01, format="%.3f")

conc, x, y, z, source_xyz, peak, speeds, directions = compute_cached(
    stability_class,
    int(frames),
    int(fps),
    int(seed),
    emission_rate,
    source_z,
    mixing_height,
    start_x,
    end_x,
    int(nx),
    start_y,
    end_y,
    int(ny),
    start_z,
    end_z,
    int(nz),
    speed_mean,
    speed_std,
    speed_theta,
    dir_mean,
    dir_std,
    dir_theta,
    jet_speed,
    jet_angle,
    jet_diameter,
)
source = _Source(source_xyz)

st.caption(f"Global peak: {peak:.1f} ppm")
t = st.slider("Frame", 0, int(frames) - 1, 0)
play = st.button("▶ Play")

scaffold = build_preview_scaffold(conc, x, y, z)
placeholder = st.empty()

if play:
    for tt in range(int(frames)):
        draw_preview_frame(scaffold, conc, source, speeds, directions, tt, stability_class, core_frac)
        placeholder.pyplot(scaffold["fig"])
        time.sleep(1.0 / max(int(fps), 1))
else:
    draw_preview_frame(scaffold, conc, source, speeds, directions, t, stability_class, core_frac)
    placeholder.pyplot(scaffold["fig"])

plt.close(scaffold["fig"])

st.header("Export")
basename = st.text_input("Output basename (no extension)", value="examples/plume_ui_export")
if st.button("Export .npz + .mp4"):
    with st.spinner("Writing .npz..."):
        export_plume_npz.save_npz(
            f"{basename}.npz", conc, x, y, z, source_xyz, emission_rate, stability_class, int(fps)
        )
    with st.spinner("Rendering .mp4..."):
        anim = fixed_source.build_animation(
            conc, x, y, z, source, speeds, directions, int(fps), stability_class, core_frac
        )
        save_or_show(anim, basename, int(fps), show=False)
    st.success(f"Wrote {basename}.npz and {basename}.mp4")
