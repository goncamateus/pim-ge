r"""Closed-form Gibbs updates for conjugate parameters — §3.1 of Newman et al. (2024)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from pim_ge.inverse.priors import Priors


@dataclass
class GibbsSamplers:
    """Exact conjugate Gibbs updates for `background` (`beta`) and `sigma^2`.

    Parameters
    ----------
    priors : Priors
        Prior hyperparameters used to form the conjugate posteriors.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), §3.1 — `beta` and `sigma^2` are
    blocked out of the M-MALA proposal (`inverse/mcmc.py`) and updated exactly
    each iteration via their conjugate full-conditional posteriors (Eqs. 8-9),
    following the within-Gibbs scheme of Algorithm A.3.
    """

    priors: Priors

    def background_conditional_posterior(
        self,
        key: Array,
        data: Array,  # (T, N)
        coupling: Array,  # (T, N)
        emission_rate: float,
        sigma2: float,
    ) -> Array:
        r"""Sample `beta` from its conjugate Normal full-conditional posterior.

        Parameters
        ----------
        key : Array
            JAX PRNG key.
        data : Array, shape (T, N)
            Observed measurements `d`.
        coupling : Array, shape (T, N)
            Coupling matrix `A`.
        emission_rate : float
            Current emission rate `s` [kg/s].
        sigma2 : float
            Current measurement-error variance `sigma^2`.

        Returns
        -------
        Array, shape (N,)
            One draw of the per-sensor background `beta` from its posterior.

        Notes
        -----
        Paper Mapping: Newman et al. (2024), Eq. (9), §3.1 — conjugate Normal
        update for `beta` given prior
        :math:`\boldsymbol{\beta}\sim\mathcal{N}(\mathbf{0}, \sigma_{bg}^2 I)`
        and likelihood :math:`d_{t,n}\sim\mathcal{N}(A_{t,n}s+\beta_n,\sigma^2)`.

        .. math::
            \boldsymbol{\beta} \mid \boldsymbol{\lambda}\setminus\{\boldsymbol{\beta}\}
            \sim \mathcal{N}\!\left(
            \left(\tfrac{1}{\sigma^2}I + \Sigma_\beta^{-1}\right)^{-1}
            \left(\tfrac{1}{\sigma^2}(\mathbf{d}-\mathbf{A}\mathbf{s}) + \Sigma_\beta^{-1}\mu_\beta\right),
            \left(\tfrac{1}{\sigma^2}I + \Sigma_\beta^{-1}\right)^{-1}\right)

        With diagonal :math:`\Sigma_\beta=\sigma_{bg}^2 I` and
        :math:`\mu_\beta=0`, the posterior factorizes per sensor, which is
        exactly the scalar precision/mean update implemented below.
        """
        T = data.shape[0]
        residuals = data - coupling * emission_rate  # (T, N)
        sigma_bg2 = self.priors.background_std**2
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
        r"""Sample `sigma^2` from its conjugate Inverse-Gamma full-conditional posterior.

        Parameters
        ----------
        key : Array
            JAX PRNG key.
        residuals : Array, shape (T, N) or (T*N,)
            Residuals `d - A*s - beta` used to form the sum of squares.

        Returns
        -------
        Array
            One draw of `sigma^2` from its posterior.

        Notes
        -----
        Paper Mapping: Newman et al. (2024), Eq. (8), §3.1 — conjugate
        Inverse-Gamma update for `sigma^2` given prior `sigma^2 ~ IG(a, b)`.

        .. math::
            \sigma^2 \mid \boldsymbol{\lambda}\setminus\{\sigma^2\}
            \sim \text{Inv-Gamma}\!\left(\frac{n_{obs}}{2}+a,\;
            b + \frac{1}{2}\sum(\mathbf{d}-\boldsymbol{\beta}-\mathbf{A}\mathbf{s})^2\right)

        Sampled here via `IG(a, b) = 1 / Gamma(a, rate=1/b)`, i.e.
        `b / Gamma(a, 1)`.
        """
        n = residuals.size
        post_alpha = self.priors.sigma2_alpha + n / 2.0
        post_beta = self.priors.sigma2_beta + 0.5 * jnp.sum(residuals**2)
        # Sample from IG(a, b) = 1 / Gamma(a, 1/b)
        gamma_sample = jax.random.gamma(key, post_alpha)
        return post_beta / gamma_sample
