"""Prior distributions — §3 of Newman et al. (2024)."""

from dataclasses import dataclass

import jax.numpy as jnp
import jax.scipy.special as jss
from jax import Array


def _log_normal_pdf(x: Array, mean: float, std: float) -> Array:
    return -0.5 * ((x - mean) / std) ** 2 - jnp.log(std)


@dataclass
class Priors:
    # log dispersion params: Normal(mean, std)
    log_a_H_mean: float = 0.0
    log_a_H_std: float = 1.0
    log_a_V_mean: float = 0.0
    log_a_V_std: float = 1.0
    log_b_H_mean: float = 0.0
    log_b_H_std: float = 1.0
    log_b_V_mean: float = 0.0
    log_b_V_std: float = 1.0
    # emission rate: Normal on log_s
    log_s_mean: float = 0.0
    log_s_std: float = 2.0
    # source location: Normal
    source_x_mean: float = 0.0
    source_x_std: float = 100.0
    source_y_mean: float = 0.0
    source_y_std: float = 100.0
    # noise variance: Inverse-Gamma(alpha, beta)
    sigma2_alpha: float = 2.0
    sigma2_beta: float = 1.0
    # background per sensor: Normal(0, sigma_bg)
    background_std: float = 1.0

    def log_prior(self, x: Array) -> Array:
        """x = [log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y]."""
        log_a_H, log_a_V, log_b_H, log_b_V, log_s, src_x, src_y = (x[i] for i in range(7))
        return (
            _log_normal_pdf(log_a_H, self.log_a_H_mean, self.log_a_H_std)
            + _log_normal_pdf(log_a_V, self.log_a_V_mean, self.log_a_V_std)
            + _log_normal_pdf(log_b_H, self.log_b_H_mean, self.log_b_H_std)
            + _log_normal_pdf(log_b_V, self.log_b_V_mean, self.log_b_V_std)
            + _log_normal_pdf(log_s, self.log_s_mean, self.log_s_std)
            + _log_normal_pdf(src_x, self.source_x_mean, self.source_x_std)
            + _log_normal_pdf(src_y, self.source_y_mean, self.source_y_std)
        )

    def log_prior_background(self, background: Array) -> Array:
        """background shape (N_sensors,). Independent Normal(0, sigma_bg)."""
        return jnp.sum(_log_normal_pdf(background, 0.0, self.background_std))

    def log_prior_sigma2(self, sigma2: Array) -> Array:
        """Inverse-Gamma(alpha, beta) log density."""
        a, b = self.sigma2_alpha, self.sigma2_beta
        return (a + 1) * jnp.log(b) - jss.gammaln(a) - (a + 1) * jnp.log(sigma2) - b / sigma2
