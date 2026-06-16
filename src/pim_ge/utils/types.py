from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


@dataclass
class Grid:
    r"""Regular 3D spatial domain used to lay out grid-based sensors.

    Parameters
    ----------
    x : Array, shape (Nx,)
        East-west coordinates [m].
    y : Array, shape (Ny,)
        North-south coordinates [m].
    z : Array, shape (Nz,)
        Vertical coordinates [m].

    Notes
    -----
    Paper Mapping: extension beyond Newman et al. (2024). The paper's grid-free
    inversion (§3) does not require a spatial grid; this dataclass only backs the
    optional grid-based sensor-layout helpers in ``forward/sensors.py``
    (``grid_of_sensors``, ``random_sensor_locations``).
    """

    x: Array  # (Nx,) east-west [m]
    y: Array  # (Ny,) north-south [m]
    z: Array  # (Nz,) vertical [m]

    @classmethod
    def uniform(cls, x_range: tuple, y_range: tuple, z_range: tuple, n: int = 50) -> "Grid":
        """Build a `Grid` with `n` evenly spaced points along each axis.

        Parameters
        ----------
        x_range : tuple
            `(start, stop)` bounds [m] for the x-axis.
        y_range : tuple
            `(start, stop)` bounds [m] for the y-axis.
        z_range : tuple
            `(start, stop)` bounds [m] for the z-axis.
        n : int, default 50
            Number of points per axis.

        Returns
        -------
        Grid
            Grid with `x`, `y`, `z` each of shape `(n,)`.

        Notes
        -----
        Paper Mapping: extension beyond Newman et al. (2024); convenience
        constructor, no paper equation.
        """
        return cls(
            x=jnp.linspace(*x_range, n),
            y=jnp.linspace(*y_range, n),
            z=jnp.linspace(*z_range, n),
        )


@dataclass
class SourceLocation:
    r"""Point emission source position :math:`(\tilde{x}, \tilde{y}, \tilde{z})`.

    Parameters
    ----------
    x : float
        Source east-west coordinate [m].
    y : float
        Source north-south coordinate [m].
    z : float
        Source height [m].

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (2), §2.1 — the source location
    :math:`(\tilde{x}, \tilde{y}, \tilde{z})` appearing in the Gaussian plume
    concentration formula `c(x, y, z; \tilde{x}, \tilde{y}, \tilde{z})`. In the
    sampled parameter vector (§3, `inverse/mcmc.py`), `x` and `y` are the
    `source_x`/`source_y` entries of `x[5:7]`; `z` is treated as a known/fixed
    release height (e.g. stack height), not sampled.
    """

    x: float  # [m]
    y: float  # [m]
    z: float  # [m]
