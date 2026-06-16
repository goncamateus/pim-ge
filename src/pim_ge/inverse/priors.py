r"""Prior distributions — §3 / Supp. B.3 of Newman et al. (2024)."""

from dataclasses import dataclass

import jax.numpy as jnp
import jax.scipy.special as jss
from jax import Array


def _log_normal_pdf(x: Array, mean: float, std: float) -> Array:
    r"""Log density of a univariate Normal, up to an additive constant.

    Parameters
    ----------
    x : Array
        Point(s) at which to evaluate the log density.
    mean : float
        Distribution mean.
    std : float
        Distribution standard deviation.

    Returns
    -------
    Array
        :math:`\log\mathcal{N}(x;\,\text{mean},\,\text{std}^2)`, dropping the
        constant :math:`-\tfrac{1}{2}\log(2\pi)` term (irrelevant to MCMC
        acceptance ratios).

    Notes
    -----
    Paper Mapping: Newman et al. (2024), §3 / Supp. B.3 — building block for
    every Normal-distributed component of the prior `p(lambda)` in Eq. (6).
    """
    return -0.5 * ((x - mean) / std) ** 2 - jnp.log(std)


@dataclass
class Priors:
    r"""Prior hyperparameters for the sampled parameter vector and Gibbs blocks.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (7), §3.1, with default values
    per Supp. B.3 — independent Normal priors on the log-dispersion
    parameters, log emission rate, and source location; Inverse-Gamma prior
    on the measurement-error variance `sigma^2`; Normal prior on the
    per-sensor background `beta`. See `log_prior`, `log_prior_sigma2`, and
    `log_prior_background` for the corresponding densities.
    """

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
        r"""Joint log prior density `p(x)` of the sampled parameter vector.

        Parameters
        ----------
        x : Array, shape (7,)
            `[log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y]`.

        Returns
        -------
        Array
            Scalar log prior density, summed over the 7 independent Normal
            components.

        Notes
        -----
        Paper Mapping: Newman et al. (2024), Eq. (7), §3.1 — the Normal-prior
        part of `p(lambda)` for the components sampled by M-MALA (`background`
        and `sigma2` have their own priors, evaluated separately by
        `log_prior_background` / `log_prior_sigma2` and updated by exact
        Gibbs steps rather than M-MALA).
        """
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
        r"""Log prior density of the per-sensor background `beta`.

        Parameters
        ----------
        background : Array, shape (N_sensors,)
            Per-sensor background offsets [ppm].

        Returns
        -------
        Array
            Scalar log density, summing independent
            :math:`\beta_n \sim \mathcal{N}(0, \sigma_{bg}^2)` terms.

        Notes
        -----
        Paper Mapping: Newman et al. (2024), Eq. (7), §3.1 —
        :math:`\boldsymbol{\beta} \sim \mathcal{N}(\mu_\beta, \Sigma_\beta)`
        with diagonal :math:`\Sigma_\beta = \sigma_{bg}^2 I` and
        :math:`\mu_\beta = 0` (the diagonal case used by this implementation).
        """
        return jnp.sum(_log_normal_pdf(background, 0.0, self.background_std))

    def log_prior_sigma2(self, sigma2: Array) -> Array:
        r"""Log prior density of the measurement-error variance `sigma^2`.

        Parameters
        ----------
        sigma2 : Array
            Measurement-error variance.

        Returns
        -------
        Array
            Log density of :math:`\sigma^2 \sim \text{Inv-Gamma}(a, b)`.

        Notes
        -----
        Paper Mapping: Newman et al. (2024), Eq. (7), §3.1.

        .. math::
            \sigma^2 \sim \text{Inv-Gamma}(a, b)
        """
        a, b = self.sigma2_alpha, self.sigma2_beta
        return (a + 1) * jnp.log(b) - jss.gammaln(a) - (a + 1) * jnp.log(sigma2) - b / sigma2
