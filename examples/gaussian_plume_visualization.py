"""Visualize a single Gaussian plume realization (Briggs class D, ppm units)."""
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pim_ge import Grid, SourceLocation, WindField
from pim_ge.forward.plume import temporal_gridfree_coupling_matrix
from pim_ge.forward.sensors import grid_of_sensors

EMISSION_RATE = 0.1   # kg/s
WIND_SPEED    = 2.0   # m/s
GRID_SIDE     = 30    # sensors per axis → 30×30 = 900 total
MIXING_HEIGHT = 200.0 # m


def main():
    source = SourceLocation(x=0.0, y=0.0, z=1.5)

    # 1-step constant wind: use T=1 for a static snapshot
    wind = WindField(
        speed=jnp.array([WIND_SPEED]),
        direction=jnp.array([0.0]),  # blowing east
    )

    # Dense sensor grid downwind of source
    grid = Grid.uniform(
        x_range=(1.0, 500.0),
        y_range=(-200.0, 200.0),
        z_range=(1.5, 1.5),
        n=GRID_SIDE,
    )
    n_sensors = GRID_SIDE * GRID_SIDE
    sensors = grid_of_sensors(grid, n_sensors)  # (900, 3)

    # Coupling matrix (ppm per kg/s)
    A = temporal_gridfree_coupling_matrix(
        source, sensors, wind,
        mixing_height=MIXING_HEIGHT,
        scheme="Briggs",
        stability_class="D",
    )  # (T=1, N=900)

    concentration = A[0] * EMISSION_RATE   # (900,) ppm
    Z = concentration.reshape(GRID_SIDE, GRID_SIDE)  # row=y, col=x

    xi = jnp.linspace(1.0, 500.0, GRID_SIDE)
    yi = jnp.linspace(-200.0, 200.0, GRID_SIDE)
    XX, YY = jnp.meshgrid(xi, yi)

    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(XX, YY, Z, shading="auto", cmap="inferno")
    fig.colorbar(pcm, ax=ax, label="Concentration (ppm)")

    ax.plot(source.x, source.y, "w*", markersize=14, label="Source")
    # wind arrow at top-left
    ax.annotate(
        "", xy=(60, 160), xytext=(10, 160),
        arrowprops=dict(arrowstyle="->", color="white", lw=2),
    )
    ax.text(35, 170, f"{WIND_SPEED} m/s", color="white", ha="center", fontsize=9)

    ax.set_xlabel("Downwind distance (m)")
    ax.set_ylabel("Crosswind distance (m)")
    ax.set_title(
        f"Gaussian plume — Briggs class D, s={EMISSION_RATE} kg/s, "
        f"u={WIND_SPEED} m/s, H={MIXING_HEIGHT} m"
    )
    ax.legend(loc="upper right")

    out = "examples/plume_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"Peak concentration: {float(Z.max()):.2f} ppm at {EMISSION_RATE} kg/s")


if __name__ == "__main__":
    main()
