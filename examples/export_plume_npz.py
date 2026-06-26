"""Export a 3D Gaussian plume (fixed source, OU wind) to a plain-numpy .npz.

Bridge artifact for Isaac Sim: no JAX needed on the consumer side. The compute
pipeline is lifted from examples/fixed_source.py (matplotlib/animation dropped).

Usage:
    uv run examples/export_plume_npz.py --out ../exxon/plume.npz --selfcheck
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from pim_ge import SourceLocation, WindField
from pim_ge.forward.momentum import JetSource
from pim_ge.forward.plume import temporal_gridfree_coupling_matrix
from pim_ge.forward.wind import wind_direction, wind_speed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="../exxon/plume.npz", help="output .npz path")
    p.add_argument("--selfcheck", action="store_true", help="assert output sanity then exit 0")
    p.add_argument("--class", dest="stability_class", default="A", choices=list("ABCDEF"))
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--emission-rate", type=float, default=0.9, help="source strength s [kg/s]")
    p.add_argument("--source-z", type=float, default=25.0, help="release height [m]")
    p.add_argument("--mixing-height", type=float, default=300.0, help="boundary-layer ceiling [m]")

    p.add_argument("--start-x", type=float, default=0.0)
    p.add_argument("--end-x", type=float, default=50.0)
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--start-y", type=float, default=-25.0)
    p.add_argument("--end-y", type=float, default=25.0)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--start-z", type=float, default=0.0)
    p.add_argument("--end-z", type=float, default=50.0)
    p.add_argument("--nz", type=int, default=35)

    p.add_argument("--speed-mean", type=float, default=2.0)
    p.add_argument("--speed-std", type=float, default=2.0)
    p.add_argument("--speed-theta", type=float, default=0.5)
    p.add_argument("--dir-mean", type=float, default=0.0)
    p.add_argument("--dir-std", type=float, default=0.05)
    p.add_argument("--dir-theta", type=float, default=0.01)

    p.add_argument("--jet-speed", type=float, default=20.0, help="0 = passive (no jet)")
    p.add_argument("--jet-angle", type=float, default=15.0)
    p.add_argument("--jet-diameter", type=float, default=0.2)
    return p.parse_args()


def compute(args):
    """Run the OU-wind plume forward model. Returns (conc, x, y, z, source_xyz, peak).

    conc: float32 (T, nx, ny, nz) ppm. Pipeline mirrors fixed_source.py:189-231.
    """
    T = args.frames
    source = SourceLocation(x=0.0, y=0.0, z=args.source_z)

    key = jax.random.PRNGKey(args.seed)
    key_speed, key_dir = jax.random.split(key)
    speeds = wind_speed(key_speed, T, mean=args.speed_mean, std=args.speed_std, theta=args.speed_theta)
    directions = wind_direction(key_dir, T, mean=args.dir_mean, std=args.dir_std, theta=args.dir_theta)
    wind = WindField(speed=speeds, direction=directions)

    x = jnp.linspace(args.start_x, args.end_x, args.nx)
    y = jnp.linspace(args.start_y, args.end_y, args.ny)
    z = jnp.linspace(args.start_z, args.end_z, args.nz)
    XX, YY, ZZ = jnp.meshgrid(x, y, z, indexing="ij")
    points = jnp.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    jet = None
    if args.jet_speed > 0.0:
        ang = jnp.deg2rad(args.jet_angle)
        jet = JetSource(
            source,
            vx=float(args.jet_speed * jnp.cos(ang)),
            vy=float(args.jet_speed * jnp.sin(ang)),
            diameter=args.jet_diameter,
        )

    print(f"Computing {T} timesteps, class {args.stability_class}...", flush=True)
    A = temporal_gridfree_coupling_matrix(
        source,
        points,
        wind,
        mixing_height=args.mixing_height,
        scheme="Briggs",
        stability_class=args.stability_class,
        jet=jet,
    )  # (T, nx*ny*nz)
    conc_flat = np.asarray(A * args.emission_rate, dtype=np.float32)  # (T, N)
    conc = conc_flat.reshape(T, args.nx, args.ny, args.nz)
    print(f"Done. Global peak: {conc.max():.1f} ppm")

    return (
        conc,
        np.asarray(x, np.float32),
        np.asarray(y, np.float32),
        np.asarray(z, np.float32),
        np.array([source.x, source.y, source.z], np.float32),
        float(conc.max()),
    )


def main():
    args = parse_args()
    conc, x, y, z, source, peak = compute(args)

    if args.selfcheck:
        # reshape/units sanity — fails loudly if axis order is wrong
        assert conc.shape == (args.frames, args.nx, args.ny, args.nz), conc.shape
        assert conc.min() >= 0.0, conc.min()
        assert conc.max() > 0.0, "plume is all-zero"
        # peak voxel should sit downwind (x>=source.x) and within a few sigma of source line
        t_peak, ix, iy, iz = np.unravel_index(int(conc.argmax()), conc.shape)
        assert x[ix] >= source[0] - 1e-3, f"peak upwind of source: x={x[ix]}"
        print(f"selfcheck OK: shape={conc.shape} peak={peak:.1f}ppm at "
              f"(x={x[ix]:.1f}, y={y[iy]:.1f}, z={z[iz]:.1f})")

    np.savez_compressed(
        args.out,
        conc=conc,
        x=x,
        y=y,
        z=z,
        source=source,
        emission_rate=np.float32(args.emission_rate),
        stability_class=args.stability_class,
        fps=np.int32(args.fps),
    )
    print(f"Wrote {args.out}  ({conc.nbytes / 1e6:.1f} MB uncompressed)")


if __name__ == "__main__":
    main()
