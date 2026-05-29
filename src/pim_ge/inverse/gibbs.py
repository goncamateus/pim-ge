"""Closed-form Gibbs updates for conjugate parameters — §3 of Newman et al. (2024)."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from pim_ge.inverse.priors import Priors


@dataclass
class GibbsSamplers:
    priors: Priors

    def background_conditional_posterior(
        self,
        key: Array,
        data: Array,       # (T, N)
        coupling: Array,   # (T, N)
        emission_rate: float,
        sigma2: float,
    ) -> Array:
        """MVN conjugate update for per-sensor background beta ~ Normal(mu_post, Sigma_post).

        Prior: beta ~ Normal(0, sigma_bg^2 * I).
        Likelihood: data[t,n] ~ Normal(A[t,n]*s + beta[n], sigma^2).
        Posterior per sensor is independent (diagonal prior + diagonal likelihood per sensor).
        """
        T = data.shape[0]
        residuals = data - coupling * emission_rate   # (T, N)
        sigma_bg2 = self.priors.background_std ** 2
        # posterior precision = T/sigma^2 + 1/sigma_bg^2
        post_prec = T / sigma2 + 1.0 / sigma_bg2
        post_var = 1.0 / post_prec
        # posterior mean = post_var * (sum_t residuals[t,n]) / sigma^2
        post_mean = post_var * jnp.sum(residuals, axis=0) / sigma2
        return post_mean + jnp.sqrt(post_var) * jax.random.normal(key, shape=post_mean.shape)

    def measurement_error_var_conditional_posterior(
        self,
        key: Array,
        residuals: Array,  # (T, N) or flat (T*N,)
    ) -> Array:
        """Inverse-Gamma conjugate update for sigma^2.

        Prior: sigma^2 ~ IG(alpha, beta).
        Posterior: sigma^2 ~ IG(alpha + n/2, beta + sum(residuals^2)/2).
        """
        n = residuals.size
        post_alpha = self.priors.sigma2_alpha + n / 2.0
        post_beta = self.priors.sigma2_beta + 0.5 * jnp.sum(residuals ** 2)
        # Sample from IG(a, b) = 1 / Gamma(a, 1/b)
        gamma_sample = jax.random.gamma(key, post_alpha)
        return post_beta / gamma_sample

    def binary_indicator_conditional_posterior(
        self,
        key: Array,
        log_s: float,
        data: Array,
        coupling: Array,
        background: Array,
        sigma2: float,
    ) -> Array:
        """Binomial spike-and-slab update for Z_i (grid-based only).

        Not used in the grid-free inversion; included for completeness.
        """
        raise NotImplementedError("Spike-and-slab only applies to grid-based source search.")
