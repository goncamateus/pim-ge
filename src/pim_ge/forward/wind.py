r"""Wind field simulation — §2.3 of Newman et al. (2024).

Implements time-varying wind speed + direction as Ornstein-Uhlenbeck (OU)
processes, simulated by Euler-Maruyama discretization (Eq. 4).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


@dataclass
class WindField:
    """Container for a simulated wind realization over `T` timesteps.

    Parameters
    ----------
    speed : Array, shape (T,)
        Wind speed `u` [m/s] at each timestep.
    direction : Array, shape (T,)
        Wind direction [rad], meteorological convention (direction the wind
        blows *toward*), at each timestep.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), §2.3. Holds the per-timestep wind
    speed and direction series used to build the time-varying coupling matrix
    (`forward/plume.py::temporal_gridfree_coupling_matrix`).
    """

    speed: Array  # (T,) [m/s]
    direction: Array  # (T,) [rad], meteorological convention


def generate_ornstein_uhlenbeck(
    key: Array,
    n_steps: int,
    mean: float,
    std: float,
    theta: float,
    dt: float = 1.0,
) -> Array:
    r"""Simulate a scalar Ornstein-Uhlenbeck process via Euler-Maruyama.

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    n_steps : int
        Number of timesteps to simulate.
    mean : float
        Long-run mean (mean-reversion level) of the process.
    std : float
        Diffusion/noise standard deviation `ξ`.
    theta : float
        Mean-reversion rate `Θ` (correlation-time parameter).
    dt : float, default 1.0
        Discretization timestep.

    Returns
    -------
    Array, shape (n_steps,)
        Simulated process trajectory, starting at `x0 = mean`.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (4), §2.3 — Euler-Maruyama
    discretization of the OU process used for both wind speed and direction.

    .. math::
        \eta(t + dt) = \eta(t) - \Theta\, dt\, \eta(t)
            + \xi\, \zeta \sqrt{2\, dt\, \Theta}

    where :math:`\zeta \sim \mathcal{N}(0, 1)`. This implementation uses the
    equivalent mean-reverting form
    :math:`X_{t+dt} = X_t + \Theta(\text{mean} - X_t)\,dt + \text{std}\sqrt{dt}\,\zeta`.
    """

    def step(x, noise):
        x_next = x + theta * (mean - x) * dt + std * jnp.sqrt(dt) * noise
        return x_next, x_next

    key, subkey = jax.random.split(key)
    noises = jax.random.normal(subkey, shape=(n_steps,))
    x0 = jnp.array(mean)
    _, xs = jax.lax.scan(step, x0, noises)
    return xs


def wind_speed(
    key: Array,
    n_steps: int,
    mean: float = 2.0,
    std: float = 0.5,
    theta: float = 0.1,
) -> Array:
    """Simulate OU wind speed, clipped to avoid non-physical near-zero speeds.

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    n_steps : int
        Number of timesteps `T`.
    mean : float, default 2.0
        Mean wind speed [m/s].
    std : float, default 0.5
        Wind speed diffusion std [m/s].
    theta : float, default 0.1
        Mean-reversion rate.

    Returns
    -------
    Array, shape (n_steps,)
        Wind speed series [m/s], clipped to `[1.0, inf)`.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (4), §2.3 — OU process for wind
    speed `u`. The lower clip at 1.0 m/s is an implementation safeguard (not
    in the paper) to keep the plume denominator `2*pi*u*sigma_H*sigma_V`
    (Eq. 2) well away from zero.
    """
    raw = generate_ornstein_uhlenbeck(key, n_steps, mean, std, theta)
    return jnp.clip(raw, 1.0)


def wind_direction(
    key: Array,
    n_steps: int,
    mean: float = 0.0,
    std: float = 0.3,
    theta: float = 0.05,
) -> Array:
    """Simulate OU wind direction in radians.

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    n_steps : int
        Number of timesteps `T`.
    mean : float, default 0.0
        Mean wind direction [rad].
    std : float, default 0.3
        Direction diffusion std [rad].
    theta : float, default 0.05
        Mean-reversion rate.

    Returns
    -------
    Array, shape (n_steps,)
        Wind direction series [rad]. Not wrapped to `[-pi, pi)` — slow,
        unwrapped drift matches the paper's wind realizations.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (4), §2.3 — OU process for wind
    direction.
    """
    return generate_ornstein_uhlenbeck(key, n_steps, mean, std, theta)


def wind_direction_linear(
    n_steps: int,
    start_deg: float,
    end_deg: float,
) -> Array:
    """Deterministic linear wind direction sweep.

    Parameters
    ----------
    n_steps : int
        Number of timesteps `T`.
    start_deg : float
        Starting direction [degrees].
    end_deg : float
        Ending direction [degrees].

    Returns
    -------
    Array, shape (n_steps,)
        Direction series [rad], linearly interpolated from `start_deg` to
        `end_deg`.

    Notes
    -----
    Paper Mapping: extension beyond Newman et al. (2024); deterministic
    direction sweep used for the 3D animated-plume example, not a wind model
    from the paper.
    """
    degrees = jnp.linspace(start_deg, end_deg, n_steps)
    return jnp.deg2rad(degrees)


def wind_direction_sinusoidal(
    key: Array,
    n_steps: int,
    mean: float = 0.0,
    std: float = 0.3,
    theta: float = 0.05,
    num_periods: float = 1.0,
) -> Array:
    r"""OU wind direction driven around a sinusoidally varying mean (WDC mode).

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    n_steps : int
        Number of timesteps `T`.
    mean : float, default 0.0
        Base direction [rad] about which the sinusoidal mean oscillates.
    std : float, default 0.3
        Oscillation amplitude and OU diffusion std [rad].
    theta : float, default 0.05
        Mean-reversion rate toward the time-varying mean.
    num_periods : float, default 1.0
        Number of full sine periods over the `n_steps` window.

    Returns
    -------
    Array, shape (n_steps,)
        Direction series [rad].

    Notes
    -----
    Paper Mapping: extension beyond Newman et al. (2024); used to drive the
    "Wind Direction Change" (WDC) factor of the §4 simulation study
    (`reproduction/section4_simulation_study.py`), but the time-varying-mean
    OU formulation itself is not given as an equation in the paper.

    The mean is driven by

    .. math::
        m(t) = \text{mean} + \text{std} \cdot
            \sin\!\left(2\pi \cdot \text{num\_periods} \cdot t / n_{\text{steps}}\right)

    and OU noise (Eq. 4 form) is run around `m(t)` instead of a constant mean.
    """
    t = jnp.arange(n_steps, dtype=jnp.float32)
    sin_mean = mean + std * jnp.sin(2.0 * jnp.pi * num_periods * t / n_steps)

    def step(x, inputs):
        noise, m = inputs
        x_next = x + theta * (m - x) + std * noise
        return x_next, x_next

    key, subkey = jax.random.split(key)
    noises = jax.random.normal(subkey, shape=(n_steps,))
    x0 = jnp.array(mean)
    _, xs = jax.lax.scan(step, x0, (noises, sin_mean))
    return xs
