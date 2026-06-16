"""Tests for M-MALA-within-Gibbs sampler.

Structural tests (shapes, finite values) run fast.
The synthetic recovery test uses a very simple 1D-style posterior
(no plume physics) to verify the sampler explores and accepts proposals.
"""

import jax
import jax.numpy as jnp

from pim_ge.inverse.gibbs import GibbsSamplers
from pim_ge.inverse.mcmc import build_log_posterior, inverse_hessian, mwg_scan, sqrt_inv_hess
from pim_ge.inverse.priors import Priors

KEY = jax.random.PRNGKey(99)


def _priors():
    return Priors(
        source_x_std=50.0,
        source_y_std=50.0,
        log_s_std=2.0,
        background_std=5.0,
        sigma2_alpha=2.0,
        sigma2_beta=0.5,
    )


def _gibbs():
    return GibbsSamplers(_priors())


def _toy_coupling(T=20, N=3):
    """Fixed coupling matrix — no plume physics, just a constant A for testing."""
    A = jnp.ones((T, N)) * 1e-3

    def coupling_fn(x):  # noqa: ignore x, return fixed A
        return A

    return coupling_fn, T, N


def test_inverse_hessian_shape():
    P = _priors()
    x = jnp.zeros(7)
    bg = jnp.zeros(3)
    s2 = jnp.array(0.1)

    coupling_fn, T, N = _toy_coupling()
    data = jnp.zeros((T, N))
    lp_fn = build_log_posterior(data, coupling_fn, P)
    H_inv = inverse_hessian(x, lambda xi: lp_fn(xi, bg, s2))
    assert H_inv.shape == (7, 7)


def test_sqrt_inv_hess_shape():
    M = jnp.eye(7) * 0.5
    S = sqrt_inv_hess(M)
    assert S.shape == (7, 7)


def test_sqrt_inv_hess_positive():
    M = jnp.eye(7) * 2.0
    S = sqrt_inv_hess(M)
    assert jnp.allclose(S @ S, M, atol=1e-5)


def test_build_log_posterior_scalar():
    P = _priors()
    coupling_fn, T, N = _toy_coupling()
    data = jnp.zeros((T, N))
    lp_fn = build_log_posterior(data, coupling_fn, P)
    x = jnp.zeros(7)
    bg = jnp.zeros(N)
    s2 = jnp.array(0.1)
    val = lp_fn(x, bg, s2)
    assert val.shape == ()
    assert jnp.isfinite(val)


def test_mwg_scan_shapes():
    P = _priors()
    G = _gibbs()
    coupling_fn, T, N = _toy_coupling(T=10, N=3)

    true_s = 50.0
    A = coupling_fn(jnp.zeros(7))
    data = A * true_s + 0.5 + jax.random.normal(KEY, (T, N)) * 0.05

    chains = mwg_scan(
        KEY,
        x_init=jnp.zeros(7),
        sigma2_init=0.1,
        background_init=jnp.zeros(N),
        data=data,
        coupling_fn=coupling_fn,
        priors=P,
        gibbs=G,
        step_size_init=0.05,
        iters=20,
    )
    assert chains["x_chain"].shape == (20, 7)
    assert chains["sigma2_chain"].shape == (20,)
    assert chains["background_chain"].shape == (20, N)
    assert chains["log_posterior_chain"].shape == (20,)
    assert chains["accept_chain"].shape == (20,)
    assert chains["accept_rate_chain"].shape == (20,)


def test_mwg_scan_finite():
    P = _priors()
    G = _gibbs()
    coupling_fn, T, N = _toy_coupling(T=10, N=3)
    A = coupling_fn(jnp.zeros(7))
    data = A * 50.0 + 0.5 + jax.random.normal(KEY, (T, N)) * 0.05

    chains = mwg_scan(
        KEY,
        x_init=jnp.zeros(7),
        sigma2_init=0.1,
        background_init=jnp.zeros(N),
        data=data,
        coupling_fn=coupling_fn,
        priors=P,
        gibbs=G,
        step_size_init=0.01,
        iters=30,
    )
    assert jnp.all(jnp.isfinite(chains["x_chain"]))
    assert jnp.all(jnp.isfinite(chains["sigma2_chain"]))
    assert jnp.all(chains["sigma2_chain"] > 0)


def test_mwg_accepts_nonzero():
    P = _priors()
    G = _gibbs()
    coupling_fn, T, N = _toy_coupling(T=10, N=3)
    A = coupling_fn(jnp.zeros(7))
    data = A * 50.0 + 0.5 + jax.random.normal(KEY, (T, N)) * 0.05

    chains = mwg_scan(
        KEY,
        x_init=jnp.zeros(7),
        sigma2_init=0.1,
        background_init=jnp.zeros(N),
        data=data,
        coupling_fn=coupling_fn,
        priors=P,
        gibbs=G,
        step_size_init=0.01,
        iters=50,
    )
    assert float(chains["accept_chain"].sum()) > 0


def test_mwg_step_size_adapts():
    P = _priors()
    G = _gibbs()
    coupling_fn, T, N = _toy_coupling(T=10, N=3)
    A = coupling_fn(jnp.zeros(7))
    data = A * 50.0 + 0.5 + jax.random.normal(KEY, (T, N)) * 0.05

    # large step size → low acceptance → cumulative rate < target → step size shrinks
    chains = mwg_scan(
        KEY,
        x_init=jnp.zeros(7),
        sigma2_init=0.1,
        background_init=jnp.zeros(N),
        data=data,
        coupling_fn=coupling_fn,
        priors=P,
        gibbs=G,
        step_size_init=10.0,
        iters=100,
    )
    assert float(chains["step_size_chain"][-1]) < 10.0
