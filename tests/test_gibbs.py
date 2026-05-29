import jax
import jax.numpy as jnp
import pytest
from pim_ge.inverse.gibbs import GibbsSamplers
from pim_ge.inverse.priors import Priors

KEY = jax.random.PRNGKey(13)
P = Priors(background_std=10.0, sigma2_alpha=2.0, sigma2_beta=1.0)
G = GibbsSamplers(P)


def test_background_sample_shape():
    T, N = 50, 4
    coupling = jnp.ones((T, N)) * 1e-3
    data = coupling * 10.0 + 0.5  # background ≈ 0.5
    bg = G.background_conditional_posterior(KEY, data, coupling, 10.0, 0.01)
    assert bg.shape == (N,)


def test_background_mean_near_truth():
    T, N = 2000, 3
    s = 100.0
    sigma2 = 0.001
    coupling = jnp.ones((T, N)) * 5e-4
    true_bg = jnp.array([1.0, 2.0, 3.0])
    data = coupling * s + true_bg[None, :]
    bg = G.background_conditional_posterior(KEY, data, coupling, s, sigma2)
    assert jnp.allclose(bg, true_bg, atol=0.1)


def test_sigma2_sample_positive():
    residuals = jax.random.normal(KEY, (30, 5))
    s2 = G.measurement_error_var_conditional_posterior(KEY, residuals)
    assert float(s2) > 0.0


def test_sigma2_mean_near_truth():
    # generate many samples and check mean is near the posterior mean
    true_sigma2 = 0.5
    T, N = 100, 5
    residuals = jax.random.normal(KEY, (T, N)) * jnp.sqrt(true_sigma2)
    n = T * N
    post_alpha = P.sigma2_alpha + n / 2.0
    post_beta = P.sigma2_beta + 0.5 * jnp.sum(residuals**2)
    expected_mean = post_beta / (post_alpha - 1)  # IG mean = beta/(alpha-1)
    keys = jax.random.split(KEY, 500)
    samples = jax.vmap(lambda k: G.measurement_error_var_conditional_posterior(k, residuals))(keys)
    assert float(jnp.mean(samples)) == pytest.approx(float(expected_mean), rel=0.05)
