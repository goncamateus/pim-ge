"""3D Gaussian plume — scatter cloud colored by concentration (ppm).

High-concentration core shown as 3D scatter.
Footprint (max over z) projected onto ground plane as heatmap.
Side panel shows xz cross-section through y=0.
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pim_ge import SourceLocation, WindField
from pim_ge.forward.plume import temporal_gridfree_coupling_matrix

EMISSION_RATE = 0.1    # kg/s
WIND_SPEED    = 2.0    # m/s
SOURCE_Z      = 5.0    # m
MIXING_HEIGHT = 200.0  # m
CORE_FRAC     = 0.04   # show points with conc > CORE_FRAC * peak

NX, NY, NZ = 40, 30, 20


def build_grid():
    x = jnp.linspace(5.0, 400.0, NX)
    y = jnp.linspace(-120.0, 120.0, NY)
    z = jnp.linspace(0.2, 60.0, NZ)
    XX, YY, ZZ = jnp.meshgrid(x, y, z, indexing="ij")
    return x, y, z, XX, YY, ZZ


def main():
    source = SourceLocation(x=0.0, y=0.0, z=SOURCE_Z)
    wind   = WindField(speed=jnp.array([WIND_SPEED]), direction=jnp.array([0.0]))

    x_vals, y_vals, z_vals, XX, YY, ZZ = build_grid()
    points = jnp.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    A    = temporal_gridfree_coupling_matrix(
        source, points, wind,
        mixing_height=MIXING_HEIGHT, scheme="Briggs", stability_class="D",
    )
    conc_flat = np.array(A[0] * EMISSION_RATE)            # (NX*NY*NZ,)
    conc_3d   = conc_flat.reshape(NX, NY, NZ)             # (x, y, z)

    peak      = conc_flat.max()
    threshold = peak * CORE_FRAC

    # --- scatter core points -------------------------------------------------
    xp = np.array(XX.ravel())
    yp = np.array(YY.ravel())
    zp = np.array(ZZ.ravel())
    cp = conc_flat

    mask = cp > threshold
    # sort ascending → high-conc renders last (on top)
    idx  = np.where(mask)[0]
    idx  = idx[np.argsort(cp[idx])]
    xm, ym, zm, cm = xp[idx], yp[idx], zp[idx], cp[idx]

    print(f"Core points: {len(idx)}  (>{threshold:.1f} ppm)")
    print(f"Peak: {peak:.1f} ppm  at s={EMISSION_RATE} kg/s")

    # --- ground footprint = max over z --------------------------------------
    footprint = conc_3d.max(axis=2)   # (NX, NY)
    Xg  = np.array(x_vals)
    Yg  = np.array(y_vals)
    XXg, YYg = np.meshgrid(Xg, Yg, indexing="ij")

    # ---- figure: 3D scatter + two inset panels ------------------------------
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        f"3D Gaussian Plume — Briggs D, s={EMISSION_RATE} kg/s, "
        f"u={WIND_SPEED} m/s, H={MIXING_HEIGHT} m",
        fontsize=11,
    )

    # Main 3D axes
    ax3 = fig.add_axes([0.0, 0.05, 0.62, 0.90], projection="3d")

    from matplotlib.colors import LogNorm  # noqa: PLC0415
    norm = LogNorm(vmin=max(threshold, 0.5), vmax=peak)
    cmap = plt.colormaps["inferno"]
    rgba = cmap(norm(cm))
    rgba[:, 3] = np.clip(0.3 + 0.65 * (np.log(cm) - np.log(threshold)) /
                         (np.log(peak) - np.log(threshold)), 0.1, 0.95)

    ax3.scatter(xm, ym, zm, c=rgba, s=18, depthshade=True)

    # Ground footprint contour
    ax3.contourf(
        XXg, YYg, footprint,
        zdir="z", offset=0.0,
        levels=20, cmap="Blues", alpha=0.55,
    )

    ax3.scatter([source.x], [source.y], [source.z],
                c="cyan", s=200, marker="*", zorder=10, label="Source", depthshade=False)

    ax3.set_xlabel("Downwind x (m)", labelpad=6)
    ax3.set_ylabel("Crosswind y (m)", labelpad=6)
    ax3.set_zlabel("Height z (m)", labelpad=6)
    ax3.set_zlim(0, 60)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.view_init(elev=24, azim=-55)

    # --- Inset 1: ground footprint (xy max-projection) ----------------------
    ax_xy = fig.add_axes([0.63, 0.52, 0.34, 0.42])
    im_xy = ax_xy.pcolormesh(Xg, Yg, footprint.T, cmap="inferno",
                              norm=LogNorm(vmin=max(0.1, peak * 0.001), vmax=peak),
                              shading="auto")
    ax_xy.scatter([source.x], [source.y], c="cyan", s=80, marker="*")
    ax_xy.set_xlabel("x (m)", fontsize=8)
    ax_xy.set_ylabel("y (m)", fontsize=8)
    ax_xy.set_title("Ground footprint (max over z)", fontsize=8)
    fig.colorbar(im_xy, ax=ax_xy, label="ppm", fraction=0.04)

    # --- Inset 2: vertical cross-section at y=0 (xz slice) -----------------
    ax_xz = fig.add_axes([0.63, 0.06, 0.34, 0.38])
    y0_idx = int(NY // 2)
    xz_slice = conc_3d[:, y0_idx, :]   # (NX, NZ)
    im_xz = ax_xz.pcolormesh(Xg, np.array(z_vals), xz_slice.T, cmap="inferno",
                               norm=LogNorm(vmin=max(0.1, peak * 0.001), vmax=peak),
                               shading="auto")
    ax_xz.axhline(SOURCE_Z, color="cyan", lw=1, ls="--", label=f"Source z={SOURCE_Z}m")
    ax_xz.set_xlabel("x (m)", fontsize=8)
    ax_xz.set_ylabel("z (m)", fontsize=8)
    ax_xz.set_title("Vertical cross-section y=0", fontsize=8)
    ax_xz.legend(fontsize=7)
    fig.colorbar(im_xz, ax=ax_xz, label="ppm", fraction=0.04)

    out = "examples/plume_3d_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
