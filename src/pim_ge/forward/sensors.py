"""Sensor layout and measurement generation — §2.4 of Newman et al. (2024)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from pim_ge.utils.types import Grid


@dataclass
class SensorsSettings:
    """Configuration for a sensor network.

    Parameters
    ----------
    n_sensors : int
        Number of sensors.
    obs_per_sensor : int
        Number of temporal observations recorded per sensor.

    Notes
    -----
    Paper Mapping: extension beyond Newman et al. (2024); plain configuration
    container, not a paper equation.
    """

    n_sensors: int
    obs_per_sensor: int


@dataclass
class Sensors:
    """A sensor network's fixed positions and recorded measurements.

    Parameters
    ----------
    positions : Array, shape (N_sensors, 3)
        Sensor `(x, y, z)` coordinates [m].
    measurements : Array, shape (T, N_sensors)
        Recorded concentration time series [ppm or kg/m^3].

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (5), §2.4 — bundles the sensor
    geometry behind the coupling matrix `A` together with the observed data
    `d` it generates.
    """

    positions: Array  # (N_sensors, 3) [m]
    measurements: Array  # (T, N_sensors) [ppm or kg/m^3]


def grid_of_sensors(grid: Grid, n_sensors: int) -> Array:
    """Place sensors on a regular grid in the x-y plane at `z = grid.z[0]`.

    Parameters
    ----------
    grid : Grid
        Spatial domain bounding the sensor layout.
    n_sensors : int
        Desired number of sensors; the grid side length is
        `ceil(sqrt(n_sensors))` and the result is truncated to `n_sensors`.

    Returns
    -------
    Array, shape (n_sensors, 3)
        Sensor `(x, y, z)` coordinates [m].

    Notes
    -----
    Paper Mapping: extension beyond Newman et al. (2024); deterministic
    grid-based sensor layout helper, not specified by the paper (which treats
    sensor positions as given input data).
    """
    side = int(jnp.ceil(jnp.sqrt(n_sensors)))
    xi = jnp.linspace(grid.x[0], grid.x[-1], side)
    yi = jnp.linspace(grid.y[0], grid.y[-1], side)
    xx, yy = jnp.meshgrid(xi, yi)
    z = jnp.full_like(xx.ravel(), float(grid.z[0]))
    positions = jnp.stack([xx.ravel(), yy.ravel(), z], axis=1)
    return positions[:n_sensors]


def random_sensor_locations(key: Array, grid: Grid, n_sensors: int) -> Array:
    """Uniformly random sensor positions within the grid bounding box.

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    grid : Grid
        Spatial domain bounding the sensor layout (z fixed at `grid.z[0]`).
    n_sensors : int
        Number of sensors to place.

    Returns
    -------
    Array, shape (n_sensors, 3)
        Sensor `(x, y, z)` coordinates [m].

    Notes
    -----
    Paper Mapping: extension beyond Newman et al. (2024); not specified by
    the paper.
    """
    key_x, key_y = jax.random.split(key)
    xs = jax.random.uniform(key_x, (n_sensors,), minval=grid.x[0], maxval=grid.x[-1])
    ys = jax.random.uniform(key_y, (n_sensors,), minval=grid.y[0], maxval=grid.y[-1])
    zs = jnp.full((n_sensors,), float(grid.z[0]))
    return jnp.stack([xs, ys, zs], axis=1)


def circle_of_sensors(
    center_x: float,
    center_y: float,
    radius: float,
    n_sensors: int,
    height: float = 1.0,
) -> Array:
    """Evenly spaced sensors on a horizontal circle.

    Parameters
    ----------
    center_x, center_y : float
        Circle center coordinates [m].
    radius : float
        Circle radius [m].
    n_sensors : int
        Number of sensors, evenly spaced around the circle.
    height : float, default 1.0
        Sensor height [m].

    Returns
    -------
    Array, shape (n_sensors, 3)
        Sensor `(x, y, z)` coordinates [m].

    Notes
    -----
    Paper Mapping: extension beyond Newman et al. (2024); a convenient
    synthetic sensor layout used by examples/quickstart code, not from the
    paper.
    """
    angles = jnp.linspace(0.0, 2 * jnp.pi, n_sensors, endpoint=False)
    xs = center_x + radius * jnp.cos(angles)
    ys = center_y + radius * jnp.sin(angles)
    zs = jnp.full((n_sensors,), height)
    return jnp.stack([xs, ys, zs], axis=1)


def temporal_sensors_measurements(
    coupling: Array,  # (T, N_sensors)
    emission_rate: float,
    background: Array,  # (N_sensors,)
    noise_std: float,
    key: Array,
) -> Array:
    r"""Simulate sensor measurements `d = A*s + beta + noise`.

    Parameters
    ----------
    coupling : Array, shape (T, N_sensors)
        Coupling matrix `A` [ppm per kg/s], e.g. from
        `forward.plume.temporal_gridfree_coupling_matrix`.
    emission_rate : float
        Source emission rate `s` [kg/s].
    background : Array, shape (N_sensors,)
        Per-sensor background offset `beta` [ppm].
    noise_std : float
        Measurement noise standard deviation `sigma` [ppm].
    key : Array
        JAX PRNG key.

    Returns
    -------
    Array, shape (T, N_sensors)
        Simulated measurements `d`.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (5), §2.4.

    .. math::
        \mathbf{d} = \mathbf{A}\mathbf{s} + \boldsymbol{\beta} + \boldsymbol{\epsilon},
        \qquad \epsilon_k \overset{iid}{\sim} \mathcal{N}(0, \sigma^2)
    """
    T, N = coupling.shape
    mean = coupling * emission_rate + background[None, :]
    noise = jax.random.normal(key, shape=(T, N)) * noise_std
    return mean + noise
