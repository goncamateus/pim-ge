import jax
import jax.numpy as jnp
import pytest
from pim_ge.forward.sensors import (
    Sensors,
    SensorsSettings,
    circle_of_sensors,
    grid_of_sensors,
    random_sensor_locations,
    temporal_sensors_measurements,
)
from pim_ge.utils.types import Grid

KEY = jax.random.PRNGKey(7)
GRID = Grid.uniform((0, 500), (-200, 200), (0, 10), n=20)


def test_grid_sensors_shape():
    pos = grid_of_sensors(GRID, 9)
    assert pos.shape == (9, 3)


def test_grid_sensors_within_bounds():
    pos = grid_of_sensors(GRID, 6)
    assert jnp.all(pos[:, 0] >= GRID.x[0]) and jnp.all(pos[:, 0] <= GRID.x[-1])
    assert jnp.all(pos[:, 1] >= GRID.y[0]) and jnp.all(pos[:, 1] <= GRID.y[-1])


def test_random_sensors_shape():
    pos = random_sensor_locations(KEY, GRID, 12)
    assert pos.shape == (12, 3)


def test_circle_sensors_shape():
    pos = circle_of_sensors(0.0, 0.0, 100.0, 8, height=2.0)
    assert pos.shape == (8, 3)


def test_circle_sensors_radius():
    pos = circle_of_sensors(0.0, 0.0, 100.0, 8)
    r = jnp.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    assert jnp.allclose(r, 100.0, atol=1e-4)


def test_measurements_shape():
    T, N = 20, 5
    coupling = jnp.ones((T, N)) * 1e-4
    background = jnp.zeros(N)
    data = temporal_sensors_measurements(coupling, 10.0, background, 0.01, KEY)
    assert data.shape == (T, N)


def test_measurements_mean():
    T, N = 5000, 3
    coupling = jnp.ones((T, N)) * 1e-3
    background = jnp.array([1.0, 2.0, 3.0])
    emission = 100.0
    data = temporal_sensors_measurements(coupling, emission, background, 0.001, KEY)
    expected_mean = 1e-3 * emission + background
    assert jnp.allclose(data.mean(axis=0), expected_mean, atol=0.05)
