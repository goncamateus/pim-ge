"""M-MALA-within-Gibbs sampler — §3 of Newman et al. (2024).

Sampled vector: x = [log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y]
beta and sigma^2 handled by exact Gibbs steps.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from pim_ge.inverse.gibbs import GibbsSamplers
from pim_ge.inverse.priors import Priors

StepSizeAdaptation = Literal["DOG", "Optimal", "False"]


@dataclass
class ManifoldMALAWithinGibbs:
    """M-MALA-within-Gibbs sampler (Algorithm 1 in Newman et al. 2024)."""
    priors: Priors
    gibbs: GibbsSamplers
    step_size: float = 0.01
    adaptation: StepSizeAdaptation = "Optimal"
    target_accept: float = 0.574

    def inverse_hessian(self, x: Array, log_posterior_fn: Callable) -> Array:
        """Hessian preconditioner: -inv(H), where H = jacfwd(jacrev(log_post)).

        Uses |lambda| eigendecomposition so result stays PD even if H is not.
        """
        H = jax.jacfwd(jax.jacrev(log_posterior_fn))(x)
        vals, vecs = jnp.linalg.eigh(-H)
        inv_H = vecs @ jnp.diag(1.0 / jnp.abs(vals)) @ vecs.T
        return inv_H

    def sqrt_inv_hess(self, inv_hess: Array) -> Array:
        """Matrix square root of inv_hess via eigendecomposition of |lambda|."""
        vals, vecs = jnp.linalg.eigh(inv_hess)
        return vecs @ jnp.diag(jnp.sqrt(jnp.abs(vals))) @ vecs.T

    def _log_proposal(self, x_prop: Array, x_curr: Array, grad: Array, sqrt_G: Array, eps: float) -> Array:
        """Log density of M-MALA proposal q(x_prop | x_curr)."""
        mu = x_curr + 0.5 * eps**2 * (sqrt_G @ sqrt_G @ grad)
        diff = x_prop - mu
        G = jnp.linalg.inv(sqrt_G @ sqrt_G)
        return -0.5 / eps**2 * diff @ G @ diff

    def manifold_mala_step(
        self,
        key: Array,
        x: Array,
        sigma2: Array,
        background: Array,
        data: Array,
        log_posterior_fn: Callable,
        step_size: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """One M-MALA M-H step + Gibbs sweeps for sigma2 and background.

        log_posterior_fn signature: (x, background, sigma2) -> scalar log-prob.
        Returns: (x_new, sigma2_new, background_new, step_size_new, accepted).
        """
        key_prop, key_acc, key_bg, key_s2 = jax.random.split(key, 4)

        lp_curr, grad_curr = jax.value_and_grad(log_posterior_fn)(x, background, sigma2)

        inv_H = self.inverse_hessian(x, lambda xi: log_posterior_fn(xi, background, sigma2))
        sqrt_G = self.sqrt_inv_hess(inv_H)

        eps = step_size
        mu_fwd = x + 0.5 * eps**2 * (inv_H @ grad_curr)
        noise = jax.random.normal(key_prop, shape=x.shape)
        x_prop = mu_fwd + eps * sqrt_G @ noise

        lp_prop, grad_prop = jax.value_and_grad(log_posterior_fn)(x_prop, background, sigma2)
        inv_H_prop = self.inverse_hessian(x_prop, lambda xi: log_posterior_fn(xi, background, sigma2))
        sqrt_G_prop = self.sqrt_inv_hess(inv_H_prop)

        log_q_fwd = self._log_proposal(x_prop, x, grad_curr, sqrt_G, eps)
        log_q_rev = self._log_proposal(x, x_prop, grad_prop, sqrt_G_prop, eps)
        log_alpha = (lp_prop + log_q_rev) - (lp_curr + log_q_fwd)

        log_u = jnp.log(jax.random.uniform(key_acc))
        accepted = log_u < log_alpha
        x_new = jnp.where(accepted, x_prop, x)

        accept_rate = jnp.minimum(1.0, jnp.exp(log_alpha))
        if self.adaptation == "Optimal":
            step_size_new = step_size * jnp.exp(0.1 * (accept_rate - self.target_accept))
        else:
            step_size_new = step_size

        residuals_simple = data - background[None, :]
        sigma2_new = self.gibbs.measurement_error_var_conditional_posterior(key_s2, residuals_simple)
        background_new = self.gibbs.background_conditional_posterior(
            key_bg, data, jnp.ones_like(data) * 1e-10, 0.0, sigma2_new
        )

        return x_new, sigma2_new, background_new, step_size_new, accepted.astype(jnp.float32)


def build_log_posterior(
    data: Array,
    coupling_fn: Callable,
    priors: Priors,
) -> Callable:
    """Build log_posterior(x, background, sigma2) -> scalar.

    coupling_fn: (x) -> A (T, N_sensors) coupling matrix [ppm per kg/s].
    Gaussian likelihood: data ~ Normal(A * exp(x[4]) + background, sqrt(sigma2)).
    x[4] = log_s, so emission rate = exp(x[4]).
    """
    def log_posterior(x: Array, background: Array, sigma2: Array) -> Array:
        A = coupling_fn(x)
        s = jnp.exp(x[4])
        predicted = A * s + background[None, :]
        T, N = data.shape
        n = T * N
        ll = -0.5 * n * jnp.log(2 * jnp.pi * sigma2) \
             - 0.5 / sigma2 * jnp.sum((data - predicted) ** 2)
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
    """Full MCMC loop via jax.lax.scan.

    Returns dict: x_chain, sigma2_chain, background_chain, log_posterior_chain,
                  step_size_chain, accept_chain, accept_rate_chain.
    """
    sampler = ManifoldMALAWithinGibbs(priors, gibbs, step_size_init, adaptation, target_accept)
    log_posterior_fn = build_log_posterior(data, coupling_fn, priors)

    def body(carry, key_t):
        x, sigma2, background, step_size, sum_accept, iteration = carry
        key_t, key_bg, key_s2 = jax.random.split(key_t, 3)

        lp_curr, grad_curr = jax.value_and_grad(log_posterior_fn)(x, background, sigma2)

        inv_H = sampler.inverse_hessian(x, lambda xi: log_posterior_fn(xi, background, sigma2))
        sqrt_G = sampler.sqrt_inv_hess(inv_H)

        key_t, key_prop, key_acc = jax.random.split(key_t, 3)
        mu_fwd = x + 0.5 * step_size**2 * (inv_H @ grad_curr)
        noise = jax.random.normal(key_prop, shape=x.shape)
        x_prop = mu_fwd + step_size * sqrt_G @ noise

        lp_prop, grad_prop = jax.value_and_grad(log_posterior_fn)(x_prop, background, sigma2)
        inv_H_prop = sampler.inverse_hessian(x_prop, lambda xi: log_posterior_fn(xi, background, sigma2))
        sqrt_G_prop = sampler.sqrt_inv_hess(inv_H_prop)

        log_q_fwd = sampler._log_proposal(x_prop, x, grad_curr, sqrt_G, step_size)
        log_q_rev = sampler._log_proposal(x, x_prop, grad_prop, sqrt_G_prop, step_size)
        log_alpha = (lp_prop + log_q_rev) - (lp_curr + log_q_fwd)

        log_u = jnp.log(jax.random.uniform(key_acc))
        accepted = (log_u < log_alpha).astype(jnp.float32)
        x_new = jnp.where(accepted, x_prop, x)

        # cumulative step-size adaptation (baked in at trace time — no lax.cond on Python string)
        sum_accept_new = sum_accept + accepted
        iteration_new = iteration + 1.0
        if adaptation == "Optimal":
            step_size_new = step_size * (1.0 + 0.1 * (sum_accept_new / iteration_new - target_accept))
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
        carry_new = (x_new, sigma2_new, background_new, step_size_new, sum_accept_new, iteration_new)
        output = (x_new, sigma2_new, background_new, lp_out, step_size_new, accepted,
                  sum_accept_new / iteration_new)
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
