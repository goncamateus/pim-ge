import jax.numpy as jnp
import pytest

from pim_ge.forward.momentum import JetSource, centerline, parcel_velocity
from pim_ge.utils.types import SourceLocation

SRC = SourceLocation(0.0, 0.0, 5.0)
U = 2.0
DIR = 0.0  # wind blows toward +x
WIND = (jnp.asarray(U), jnp.asarray(DIR))


def _wind_vec(speed=U, direction=DIR):
    return jnp.array([speed * jnp.cos(direction), speed * jnp.sin(direction)])


def test_passive_reduction_velocity():
    # V_source == V_wind -> velocity is the wind vector for all t (fp tol).
    jet = JetSource(SRC, vx=U, vy=0.0, l_relax=37.0)
    t = jnp.linspace(0.0, 500.0, 20)
    v = parcel_velocity(jet, *WIND, t)
    wind = _wind_vec()
    assert jnp.allclose(v, wind[None, :], atol=1e-9)


def test_passive_reduction_centerline():
    # V_source == V_wind -> centreline is the straight wind axis: src + U*t*w_hat.
    jet = JetSource(SRC, vx=U, vy=0.0, l_relax=37.0)
    t = jnp.linspace(0.0, 500.0, 20)
    r = centerline(jet, *WIND, t)
    expected_x = SRC.x + U * t  # DIR = 0
    assert jnp.allclose(r[:, 0], expected_x, atol=1e-9)
    assert jnp.allclose(r[:, 1], SRC.y, atol=1e-9)


def test_velocity_at_source_is_v_source():
    # At t=0 parcel velocity == V_source (jet dominates).
    jet = JetSource(SRC, vx=1.0, vy=6.0, l_relax=50.0)
    v0 = parcel_velocity(jet, *WIND, jnp.asarray(0.0))
    assert jnp.allclose(v0, jnp.array([1.0, 6.0]), atol=1e-9)


def test_velocity_far_field_converges_to_wind():
    # At large t parcel velocity -> V_wind.
    jet = JetSource(SRC, vx=1.0, vy=6.0, l_relax=50.0)
    v_far = parcel_velocity(jet, *WIND, jnp.asarray(1e6))
    assert jnp.allclose(v_far, _wind_vec(), atol=1e-6)


def test_transition_monotonic():
    # |v(t) - V_wind| decreases monotonically (no oscillation) for a strong jet.
    jet = JetSource(SRC, vx=-3.0, vy=8.0, l_relax=40.0)  # R = sqrt(73)/2 ~ 4.3
    t = jnp.linspace(0.0, 400.0, 200)
    v = parcel_velocity(jet, *WIND, t)
    dist = jnp.linalg.norm(v - _wind_vec()[None, :], axis=1)
    assert jnp.all(jnp.diff(dist) <= 1e-9)


def test_large_velocity_ratio_no_blowup():
    # Huge R must stay finite and bounded everywhere.
    jet = JetSource(SRC, vx=200.0, vy=-150.0, l_relax=30.0)
    t = jnp.linspace(0.0, 1000.0, 100)
    v = parcel_velocity(jet, *WIND, t)
    r = centerline(jet, *WIND, t)
    assert jnp.all(jnp.isfinite(v))
    assert jnp.all(jnp.isfinite(r))


def test_l_relax_derived_from_diameter():
    # Default L_relax = D * |V_source| / U; tau = L/U. At t = tau the decay is
    # 1/e, so v_par = U + (a-U)/e along the wind axis (DIR=0 -> a = vx).
    D, vx = 4.0, 10.0
    jet = JetSource(SRC, vx=vx, vy=0.0, diameter=D)
    L = D * vx / U
    tau = L / U
    v = parcel_velocity(jet, *WIND, jnp.asarray(tau))
    expected_vx = U + (vx - U) * jnp.exp(-1.0)
    assert float(v[0]) == pytest.approx(float(expected_vx), rel=1e-6)
    assert float(v[1]) == pytest.approx(0.0, abs=1e-9)


def test_centerline_starts_at_source():
    jet = JetSource(SRC, vx=3.0, vy=4.0, l_relax=25.0)
    r0 = centerline(jet, *WIND, jnp.asarray(0.0))
    assert float(r0[0]) == pytest.approx(SRC.x, abs=1e-9)
    assert float(r0[1]) == pytest.approx(SRC.y, abs=1e-9)


def test_cross_wind_deflection_bounded():
    # Far-field cross-wind offset converges to b*tau (here b = vy, DIR=0).
    vy, L = 6.0, 50.0
    jet = JetSource(SRC, vx=U, vy=vy, l_relax=L)
    tau = L / U
    r_far = centerline(jet, *WIND, jnp.asarray(1e7))
    assert float(r_far[1]) == pytest.approx(vy * tau, rel=1e-5)
