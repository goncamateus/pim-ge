import jax.numpy as jnp
import pytest
from pim_ge.inverse.priors import Priors


P = Priors()


def test_log_prior_scalar():
    x = jnp.zeros(7)
    val = P.log_prior(x)
    assert val.shape == ()


def test_log_prior_mode():
    # at prior mean log_prior should be > any off-center value
    x_mode = jnp.array(
        [
            P.log_a_H_mean,
            P.log_a_V_mean,
            P.log_b_H_mean,
            P.log_b_V_mean,
            P.log_s_mean,
            P.source_x_mean,
            P.source_y_mean,
        ]
    )
    x_off = x_mode + 5.0
    assert float(P.log_prior(x_mode)) > float(P.log_prior(x_off))


def test_log_prior_sigma2_decreasing():
    # Inverse-Gamma: log density should peak around beta/(alpha+1)
    vals = jnp.array([0.1, 0.5, 1.0, 5.0, 20.0])
    lp = jnp.array([float(P.log_prior_sigma2(jnp.array(v))) for v in vals])
    assert float(lp[0]) < float(lp[1])  # rising from 0


def test_log_prior_background():
    bg = jnp.zeros(5)
    val = P.log_prior_background(bg)
    assert val.shape == ()
    bg_off = jnp.ones(5) * 10.0
    assert float(P.log_prior_background(bg)) > float(P.log_prior_background(bg_off))
