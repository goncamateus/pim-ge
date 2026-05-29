import jax
import jax.numpy as jnp
import pytest
from pim_ge.forward.wind import (
    WindField,
    generate_ornstein_uhlenbeck,
    wind_direction,
    wind_direction_linear,
    wind_direction_sinusoidal,
    wind_speed,
)


KEY = jax.random.PRNGKey(42)


def test_ou_shape():
    x = generate_ornstein_uhlenbeck(KEY, 100, mean=2.0, std=0.5, theta=0.1)
    assert x.shape == (100,)


def test_ou_mean_reversion():
    x = generate_ornstein_uhlenbeck(KEY, 10_000, mean=3.0, std=0.2, theta=0.5)
    assert float(jnp.mean(x)) == pytest.approx(3.0, abs=0.2)


def test_wind_speed_positive():
    s = wind_speed(KEY, 200)
    assert float(jnp.min(s)) >= 1.0


def test_wind_speed_shape():
    s = wind_speed(KEY, 50)
    assert s.shape == (50,)


def test_wind_direction_shape():
    d = wind_direction(KEY, 50)
    assert d.shape == (50,)


def test_wind_field_dataclass():
    s = wind_speed(KEY, 30)
    d = wind_direction(KEY, 30)
    wf = WindField(speed=s, direction=d)
    assert wf.speed.shape == (30,)
    assert wf.direction.shape == (30,)


def test_wind_direction_linear_monotone():
    d = wind_direction_linear(100, start_deg=0.0, end_deg=90.0)
    assert d.shape == (100,)
    assert jnp.all(jnp.diff(d) >= 0)
    assert float(d[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(d[-1]) == pytest.approx(jnp.pi / 2, abs=1e-5)


def test_wind_direction_sinusoidal_shape():
    d = wind_direction_sinusoidal(KEY, 200, mean=0.0, std=0.3, theta=0.05, num_periods=2.0)
    assert d.shape == (200,)
    assert jnp.all(jnp.isfinite(d))
