"""Wind field simulation — §2 of Newman et al. (2024).

Implements time-varying wind speed + direction as Ornstein-Uhlenbeck processes.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


@dataclass
class WindField:
    """Container for a wind realization over T timesteps."""

    speed: Array  # (T,) [m/s]
    direction: Array  # (T,) [rad], meteorological convention


def generate_ornstein_uhlenbeck(
    key: Array,
    n_steps: int,
    mean: float,
    std: float,
    theta: float,
    dt: float = 1.0,
) -> Array:
    """Simulate scalar OU process: dX = theta*(mean - X)dt + std*dW."""

    def step(x, noise):
        x_next = x + theta * (mean - x) * dt + std * jnp.sqrt(dt) * noise
        return x_next, x_next

    key, subkey = jax.random.split(key)
    noises = jax.random.normal(subkey, shape=(n_steps,))
    x0 = jnp.array(mean)
    _, xs = jax.lax.scan(step, x0, noises)
    return xs


def wind_speed(
    key: Array,
    n_steps: int,
    mean: float = 2.0,
    std: float = 0.5,
    theta: float = 0.1,
) -> Array:
    """OU wind speed, clipped to [1.0, inf) to avoid non-physical near-zero speeds."""
    raw = generate_ornstein_uhlenbeck(key, n_steps, mean, std, theta)
    return jnp.clip(raw, 1.0)


def wind_direction(
    key: Array,
    n_steps: int,
    mean: float = 0.0,
    std: float = 0.3,
    theta: float = 0.05,
) -> Array:
    """OU wind direction in radians. No wrapping — slow drift matches paper."""
    return generate_ornstein_uhlenbeck(key, n_steps, mean, std, theta)


def wind_direction_linear(
    n_steps: int,
    start_deg: float,
    end_deg: float,
) -> Array:
    """Linear wind direction sweep from start_deg to end_deg, returned in radians."""
    degrees = jnp.linspace(start_deg, end_deg, n_steps)
    return jnp.deg2rad(degrees)


def wind_direction_sinusoidal(
    key: Array,
    n_steps: int,
    mean: float = 0.0,
    std: float = 0.3,
    theta: float = 0.05,
    num_periods: float = 1.0,
) -> Array:
    """OU wind direction with sinusoidally varying mean (WDC mode).

    Mean oscillates as mean + std * sin(2π * num_periods * t/n_steps).
    OU noise is run around that time-varying mean.
    """
    t = jnp.arange(n_steps, dtype=jnp.float32)
    sin_mean = mean + std * jnp.sin(2.0 * jnp.pi * num_periods * t / n_steps)

    def step(x, inputs):
        noise, m = inputs
        x_next = x + theta * (m - x) + std * noise
        return x_next, x_next

    key, subkey = jax.random.split(key)
    noises = jax.random.normal(subkey, shape=(n_steps,))
    x0 = jnp.array(mean)
    _, xs = jax.lax.scan(step, x0, (noises, sin_mean))
    return xs
