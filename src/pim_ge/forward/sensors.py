"""Sensor layout and measurement generation — §2 of Newman et al. (2024)."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from pim_ge.utils.types import Grid


@dataclass
class SensorsSettings:
    n_sensors: int
    obs_per_sensor: int


@dataclass
class Sensors:
    positions: Array  # (N_sensors, 3) [m]
    measurements: Array  # (T, N_sensors) [ppm or kg/m^3]


def grid_of_sensors(grid: Grid, n_sensors: int) -> Array:
    """Place sensors on a regular grid in the x-y plane at z=grid.z[0]."""
    side = int(jnp.ceil(jnp.sqrt(n_sensors)))
    xi = jnp.linspace(grid.x[0], grid.x[-1], side)
    yi = jnp.linspace(grid.y[0], grid.y[-1], side)
    xx, yy = jnp.meshgrid(xi, yi)
    z = jnp.full_like(xx.ravel(), float(grid.z[0]))
    positions = jnp.stack([xx.ravel(), yy.ravel(), z], axis=1)
    return positions[:n_sensors]


def random_sensor_locations(key: Array, grid: Grid, n_sensors: int) -> Array:
    """Uniformly random sensor positions within the grid bounding box."""
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
    """Evenly spaced sensors on a horizontal circle."""
    angles = jnp.linspace(0.0, 2 * jnp.pi, n_sensors, endpoint=False)
    xs = center_x + radius * jnp.cos(angles)
    ys = center_y + radius * jnp.sin(angles)
    zs = jnp.full((n_sensors,), height)
    return jnp.stack([xs, ys, zs], axis=1)


def temporal_sensors_measurements(
    coupling: Array,    # (T, N_sensors)
    emission_rate: float,
    background: Array,  # (N_sensors,)
    noise_std: float,
    key: Array,
) -> Array:
    """Simulate data = A * s + beta + noise, shape (T, N_sensors)."""
    T, N = coupling.shape
    mean = coupling * emission_rate + background[None, :]
    noise = jax.random.normal(key, shape=(T, N)) * noise_std
    return mean + noise
