"""Source momentum: near-field jet relaxing to far-field passive advection.

A source expels gas at V_source at an angle to the wind. Near the release the
parcel keeps its injection momentum; entrainment relaxes it so the wind takes
over downwind. Plots (1) the parcel speed and heading vs downwind distance and
(2) the bent centreline next to the passive (straight) wind axis.

Usage:
    uv run examples/jet_momentum.py
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pim_ge.forward.momentum import JetSource, centerline, parcel_velocity
from pim_ge.utils.types import SourceLocation

U = 2.0  # [m/s] wind speed
WIND_DIR = 0.0  # [rad] wind blows toward +x
SRC = SourceLocation(0.0, 0.0, 5.0)

# Exit velocity: fast jet (R = |V_source|/U ~ 5) aimed 60 deg off the wind axis.
V_SRC = 10.0
ANGLE = jnp.deg2rad(60.0)
jet = JetSource(
    SRC,
    vx=float(V_SRC * jnp.cos(ANGLE)),
    vy=float(V_SRC * jnp.sin(ANGLE)),
    diameter=4.0,  # L_relax derived as D * |V_source| / U
)

t = jnp.linspace(0.0, 300.0, 400)
v = parcel_velocity(jet, jnp.asarray(U), jnp.asarray(WIND_DIR), t)
r = centerline(jet, jnp.asarray(U), jnp.asarray(WIND_DIR), t)

speed = jnp.linalg.norm(v, axis=1)
heading = jnp.rad2deg(jnp.arctan2(v[:, 1], v[:, 0]))
downwind = r[:, 0] - SRC.x  # DIR = 0 -> downwind distance is +x offset

t_np, speed_np, heading_np = np.array(t), np.array(speed), np.array(heading)
down_np, r_np = np.array(downwind), np.array(r)

print(f"R = |V_source|/U = {V_SRC / U:.1f}")
print(f"t=0   : speed={speed_np[0]:.2f} m/s  heading={heading_np[0]:.0f} deg (= V_source)")
print(f"t=end : speed={speed_np[-1]:.2f} m/s  heading={heading_np[-1]:.0f} deg (-> V_wind)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1b = ax1.twinx()
ax1.plot(down_np, speed_np, color="tab:red", label="speed")
ax1b.plot(down_np, heading_np, color="tab:blue", ls="--", label="heading")
ax1.axhline(U, color="tab:red", lw=0.8, ls=":", alpha=0.6)
ax1b.axhline(np.degrees(WIND_DIR), color="tab:blue", lw=0.8, ls=":", alpha=0.6)
ax1.set_xlabel("downwind distance (m)")
ax1.set_ylabel("parcel speed (m/s)", color="tab:red")
ax1b.set_ylabel("parcel heading (deg)", color="tab:blue")
ax1.set_title("Near-field jet -> far-field wind")

ax2.plot(r_np[:, 0], r_np[:, 1], color="tab:green", lw=2, label="bent centreline")
ax2.plot(
    [SRC.x, SRC.x + float(downwind[-1])],
    [SRC.y, SRC.y],
    color="gray",
    ls="--",
    label="passive wind axis",
)
ax2.scatter([SRC.x], [SRC.y], c="k", marker="*", s=120, zorder=5, label="source")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_title("Plume centreline")
ax2.legend(fontsize=8)
ax2.set_aspect("equal", adjustable="datalim")

fig.tight_layout()
out = "examples/jet_momentum.png"
fig.savefig(out, dpi=120)
print(f"Saved {out}")
