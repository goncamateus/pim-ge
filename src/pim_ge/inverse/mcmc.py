r"""M-MALA-within-Gibbs sampler — §3.2/§3.3, Supp. A.3 of Newman et al. (2024).

Sampled vector: x = [log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y]
beta and sigma^2 handled by exact Gibbs steps (Eqs. 8-9, see `inverse/gibbs.py`).
"""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from pim_ge.inverse.gibbs import GibbsSamplers
from pim_ge.inverse.priors import Priors

StepSizeAdaptation = Literal["DOG", "Optimal", "False"]


def inverse_hessian(x: Array, log_posterior_fn: Callable) -> Array:
    r"""Positive-definite Riemannian metric inverse `G^{-1}` at `x`.

    Parameters
    ----------
    x : Array, shape (7,)
        Point at which to evaluate the metric.
    log_posterior_fn : Callable
        `(x) -> scalar` log-posterior, holding `background`/`sigma2` fixed.

    Returns
    -------
    Array, shape (7, 7)
        `inv(G) = -inv(H)`, where `H` is the Hessian of the log-posterior,
        built from `|eigenvalue|` so the result stays positive-definite
        even where the true Hessian is not.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (10), §3.2 — `G` is the
    Riemannian metric (here taken as the negative Hessian) defining the
    Manifold MALA proposal covariance `zeta * G^{-1}`.
    """
    H = jax.jacfwd(jax.jacrev(log_posterior_fn))(x)
    vals, vecs = jnp.linalg.eigh(-H)
    inv_H = vecs @ jnp.diag(1.0 / jnp.abs(vals)) @ vecs.T
    return inv_H


def sqrt_inv_hess(inv_hess: Array) -> Array:
    """Symmetric matrix square root of `inv_hess` via eigendecomposition.

    Parameters
    ----------
    inv_hess : Array, shape (7, 7)
        Positive-definite matrix (typically `inverse_hessian`'s output).

    Returns
    -------
    Array, shape (7, 7)
        `sqrt_G` such that `sqrt_G @ sqrt_G == inv_hess` (up to the
        `|eigenvalue|` clamp).

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (10), §3.2 — supplies the
    `G^{-1/2}` factor used to draw the M-MALA proposal noise term
    `zeta * G^{-1/2} @ noise`.
    """
    vals, vecs = jnp.linalg.eigh(inv_hess)
    return vecs @ jnp.diag(jnp.sqrt(jnp.abs(vals))) @ vecs.T


def _log_proposal(x_prop: Array, x_curr: Array, grad: Array, sqrt_G: Array, eps: float) -> Array:
    r"""Log density of the M-MALA proposal `q(x_prop | x_curr)`.

    Parameters
    ----------
    x_prop : Array, shape (7,)
        Proposed point.
    x_curr : Array, shape (7,)
        Current point.
    grad : Array, shape (7,)
        Gradient of the log-posterior at `x_curr`.
    sqrt_G : Array, shape (7, 7)
        `G^{-1/2}` at `x_curr` (from `sqrt_inv_hess`).
    eps : float
        Step size `zeta`.

    Returns
    -------
    Array
        Scalar log proposal density, up to an additive normalizing
        constant (sufficient for the Metropolis-Hastings ratio).

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (10), §3.2.

    .. math::
        \theta^* \sim \mathcal{N}_n\!\left(
            \theta^{(l-1)} + \tfrac{1}{2}\zeta^{(l-1)} G^{-1}
            \nabla\log p(\theta^{(l-1)}\mid \mathbf{d}),\;
            \zeta^{(l-1)} G^{-1}\right)

    Note the implementation's `eps**2` scaling is an equivalent
    reparametrization of the paper's `zeta` (step size in "standard
    deviation" units rather than variance units).
    """
    mu = x_curr + 0.5 * eps**2 * (sqrt_G @ sqrt_G @ grad)
    diff = x_prop - mu
    G = jnp.linalg.inv(sqrt_G @ sqrt_G)
    return -0.5 / eps**2 * diff @ G @ diff


def build_log_posterior(
    data: Array,
    coupling_fn: Callable,
    priors: Priors,
) -> Callable:
    r"""Build the closure `log_posterior(x, background, sigma2) -> scalar`.

    Parameters
    ----------
    data : Array, shape (T, N_sensors)
        Observed measurements `d`.
    coupling_fn : Callable
        `(x) -> A`, coupling matrix `(T, N_sensors)` [ppm per kg/s], e.g. from
        `forward.plume.temporal_gridfree_coupling_matrix`.
    priors : Priors
        Prior hyperparameters for `x`.

    Returns
    -------
    Callable
        `log_posterior(x, background, sigma2) -> Array` (scalar).

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (6), §3 (posterior), composed
    with the Gaussian likelihood of Eq. (5):
    `d ~ Normal(A * exp(x[4]) + background, sigma^2)`, where `x[4] = log_s`
    so the emission rate is `s = exp(x[4])` (log-space keeps `s` positive).

    .. math::
        p(\boldsymbol{\lambda}\mid\mathbf{d}) \propto
        p(\mathbf{d}\mid\boldsymbol{\lambda})\,p(\boldsymbol{\lambda})
    """

    def log_posterior(x: Array, background: Array, sigma2: Array) -> Array:
        A = coupling_fn(x)
        s = jnp.exp(x[4])
        predicted = A * s + background[None, :]
        T, N = data.shape
        n = T * N
        ll = -0.5 * n * jnp.log(2 * jnp.pi * sigma2) - 0.5 / sigma2 * jnp.sum(
            (data - predicted) ** 2
        )
        lp = priors.log_prior(x)
        return ll + lp

    return log_posterior


def mwg_scan(
    key: Array,
    x_init: Array,
    sigma2_init: Array,
    background_init: Array,
    data: Array,
    coupling_fn: Callable,  # (x) -> A (T, N)
    priors: Priors,
    gibbs: GibbsSamplers,
    step_size_init: float = 0.01,
    adaptation: StepSizeAdaptation = "Optimal",
    target_accept: float = 0.574,
    iters: int = 10_000,
) -> dict:
    """Run the full M-MALA-within-Gibbs MCMC loop via `jax.lax.scan`.

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    x_init : Array, shape (7,)
        Initial parameter vector.
    sigma2_init : Array
        Initial measurement-error variance.
    background_init : Array, shape (N_sensors,)
        Initial per-sensor background.
    data : Array, shape (T, N_sensors)
        Observed measurements `d`.
    coupling_fn : Callable
        `(x) -> A`, coupling matrix `(T, N)`.
    priors : Priors
        Prior hyperparameters.
    gibbs : GibbsSamplers
        Exact conjugate Gibbs updates.
    step_size_init : float, default 0.01
        Initial M-MALA step size.
    adaptation : {"DOG", "Optimal", "False"}, default "Optimal"
        Step-size adaptation scheme.
    target_accept : float, default 0.574
        Target acceptance rate for adaptation.
    iters : int, default 10_000
        Number of MCMC iterations.

    Returns
    -------
    dict
        `x_chain`, `sigma2_chain`, `background_chain`, `log_posterior_chain`,
        `step_size_chain`, `accept_chain`, `accept_rate_chain` — each an
        Array with leading dimension `iters`.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Algorithm pseudocode, Supp. A.3 —
    `jax.lax.scan`-vectorized M-MALA-within-Gibbs sweep: M-MALA proposal/accept
    for `x` (Eq. 10, via `inverse_hessian`/`sqrt_inv_hess`/`_log_proposal`)
    followed by exact Gibbs updates for `sigma2` (Eq. 8) and `background`
    (Eq. 9), with `"Optimal"` adaptation nudging `zeta` toward `target_accept`
    using the running acceptance rate.
    """
    log_posterior_fn = build_log_posterior(data, coupling_fn, priors)

    def body(carry, key_t):
        x, sigma2, background, step_size, sum_accept, iteration = carry
        key_t, key_bg, key_s2 = jax.random.split(key_t, 3)

        lp_curr, grad_curr = jax.value_and_grad(log_posterior_fn)(x, background, sigma2)

        inv_H = inverse_hessian(x, lambda xi: log_posterior_fn(xi, background, sigma2))
        sqrt_G = sqrt_inv_hess(inv_H)

        key_t, key_prop, key_acc = jax.random.split(key_t, 3)
        mu_fwd = x + 0.5 * step_size**2 * (inv_H @ grad_curr)
        noise = jax.random.normal(key_prop, shape=x.shape)
        x_prop = mu_fwd + step_size * sqrt_G @ noise

        lp_prop, grad_prop = jax.value_and_grad(log_posterior_fn)(x_prop, background, sigma2)
        inv_H_prop = inverse_hessian(x_prop, lambda xi: log_posterior_fn(xi, background, sigma2))
        sqrt_G_prop = sqrt_inv_hess(inv_H_prop)

        log_q_fwd = _log_proposal(x_prop, x, grad_curr, sqrt_G, step_size)
        log_q_rev = _log_proposal(x, x_prop, grad_prop, sqrt_G_prop, step_size)
        log_alpha = (lp_prop + log_q_rev) - (lp_curr + log_q_fwd)

        log_u = jnp.log(jax.random.uniform(key_acc))
        accepted = (log_u < log_alpha).astype(jnp.float32)
        x_new = jnp.where(accepted, x_prop, x)

        # cumulative step-size adaptation (baked in at trace time — no lax.cond on Python string)
        sum_accept_new = sum_accept + accepted
        iteration_new = iteration + 1.0
        if adaptation == "Optimal":
            step_size_new = step_size * (
                1.0 + 0.1 * (sum_accept_new / iteration_new - target_accept)
            )
        else:
            step_size_new = step_size

        # Gibbs: sigma2
        A = coupling_fn(x_new)
        s = jnp.exp(x_new[4])
        residuals = data - A * s - background[None, :]
        sigma2_new = gibbs.measurement_error_var_conditional_posterior(key_s2, residuals)

        # Gibbs: background
        background_new = gibbs.background_conditional_posterior(key_bg, data, A, s, sigma2_new)

        lp_out = log_posterior_fn(x_new, background_new, sigma2_new)
        carry_new = (
            x_new,
            sigma2_new,
            background_new,
            step_size_new,
            sum_accept_new,
            iteration_new,
        )
        output = (
            x_new,
            sigma2_new,
            background_new,
            lp_out,
            step_size_new,
            accepted,
            sum_accept_new / iteration_new,
        )
        return carry_new, output

    keys = jax.random.split(key, iters)
    carry_init = (
        x_init,
        jnp.array(sigma2_init),
        background_init,
        jnp.array(step_size_init),
        jnp.array(0.0),
        jnp.array(0.0),
    )
    _, outputs = jax.lax.scan(body, carry_init, keys)

    x_chain, sigma2_chain, bg_chain, lp_chain, eps_chain, acc_chain, rate_chain = outputs
    return {
        "x_chain": x_chain,
        "sigma2_chain": sigma2_chain,
        "background_chain": bg_chain,
        "log_posterior_chain": lp_chain,
        "step_size_chain": eps_chain,
        "accept_chain": acc_chain,
        "accept_rate_chain": rate_chain,
    }
