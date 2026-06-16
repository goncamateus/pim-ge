r"""Source-momentum (jet-in-crossflow) extension to the passive plume.

Standalone add-on to the steady-state Gaussian plume (``forward/plume.py``),
which assumes a *passive* release: every parcel instantly adopts the ambient
wind vector, so the plume centreline is the straight wind axis through the
source. That is the zero-injection-pressure limit (``V_source = 0`` relaxed to
``V_wind`` immediately).

Here the source expels gas with its own exit velocity ``V_source``. Near the
source the parcel keeps that injection momentum (jet-like, resisting the wind);
entrainment of ambient air relaxes the excess momentum so the wind progressively
takes over and the centreline bends to align with ``V_wind`` far downwind.

Model — exponential momentum relaxation (Option 1), neutral buoyancy, steady
wind. Working in the wind frame (``a``/``b`` = along-/cross-wind components of
``V_source``, ``U`` = wind speed, ``tau`` = momentum relaxation time):

.. math::
    v_\parallel(t) = U + (a - U)\,e^{-t/\tau}, \qquad
    v_\perp(t)     = b\,e^{-t/\tau}

Integrating once gives the closed-form bent centreline

.. math::
    x_\parallel(t) = U t + (a - U)\,\tau\,(1 - e^{-t/\tau}), \qquad
    x_\perp(t)     = b\,\tau\,(1 - e^{-t/\tau})

Limiting cases:

* ``t -> 0``  : velocity -> ``V_source`` (jet dominates).
* ``t -> inf``: velocity -> ``V_wind``, centreline parallel to the wind axis.
* ``V_source = V_wind`` (a = U, b = 0): reduces *exactly* to the passive
  straight wind axis for all ``t`` — a strict superset of the passive model.

The relaxation length ``L_relax = U * tau`` may be passed directly, or derived
from the source exit diameter ``D`` and velocity ratio ``R = |V_source| / U`` as
the jet momentum length ``L_relax ~ D * R`` (so ``tau = D * R / U``).

This module is deliberately decoupled from ``temporal_gridfree_coupling_matrix``
and the inverse model: importing it changes nothing about existing results.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from pim_ge.utils.types import SourceLocation

jax.config.update("jax_enable_x64", True)


@dataclass
class JetSource:
    r"""Momentum-carrying point source: position + horizontal exit velocity.

    Parameters
    ----------
    location : SourceLocation
        Emission point :math:`(\tilde{x}, \tilde{y}, \tilde{z})` [m].
    vx, vy : float
        Horizontal exit-velocity components [m/s] in the world frame (``vx``
        east, ``vy`` north). Setting ``(vx, vy)`` equal to the wind vector
        ``(U cos dir, U sin dir)`` recovers the passive plume exactly.
    diameter : float, default 0.0
        Source/stack exit diameter ``D`` [m]. Only used to derive ``l_relax``
        when it is not given explicitly.
    l_relax : float or None, default None
        Momentum relaxation length [m]. If ``None``, derived at call time as the
        jet momentum length ``D * |V_source| / U``.

    Notes
    -----
    Density / temperature are intentionally absent: this is the neutral-buoyancy
    momentum model. Buoyant plume rise (Briggs) is out of scope.
    """

    location: SourceLocation
    vx: float
    vy: float
    diameter: float = 0.0
    l_relax: float | None = None


def relaxation_time(
    jet: JetSource,
    wind_speed: Array,
    wind_dir: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    r"""Resolve the wind-frame jet parameters used by both velocity and centreline.

    Parameters
    ----------
    jet : JetSource
        Momentum-carrying source.
    wind_speed : Array
        Wind speed ``U`` [m/s].
    wind_dir : Array
        Wind direction [rad] (blowing *toward*, matching ``WindField``).

    Returns
    -------
    tuple
        ``(a, b, tau, wx, wy)`` where ``a``/``b`` are the along-/cross-wind
        components of ``V_source`` [m/s], ``tau`` the momentum relaxation time
        [s] (floored at a tiny epsilon to stay finite at ``D = 0``), and
        ``(wx, wy)`` the along-wind unit vector. The cross-wind unit vector is
        ``(-wy, wx)``.
    """
    U = wind_speed
    wx, wy = jnp.cos(wind_dir), jnp.sin(wind_dir)  # along-wind unit vector
    # cross-wind unit vector is (-wy, wx)
    a = jet.vx * wx + jet.vy * wy
    b = -jet.vx * wy + jet.vy * wx
    speed_src = jnp.hypot(jet.vx, jet.vy)

    if jet.l_relax is not None:
        L = jnp.asarray(jet.l_relax, dtype=U.dtype)
    else:
        L = jet.diameter * speed_src / U  # D * R, with R = |V_source| / U
    # tau = L / U. Floor keeps t/tau finite when L = 0 (instant-passive limit);
    # exactness when V_source = V_wind holds regardless since a - U = 0, b = 0.
    tau = jnp.maximum(L / U, 1e-12)
    return a, b, tau, wx, wy


def parcel_velocity(
    jet: JetSource,
    wind_speed: Array,
    wind_dir: Array,
    t: Array,
) -> Array:
    r"""Horizontal parcel velocity ``(vx, vy)`` [m/s] at elapsed time ``t`` [s].

    Decays monotonically from ``V_source`` at ``t = 0`` to ``V_wind`` as
    ``t -> inf`` (exponential momentum relaxation).

    Parameters
    ----------
    jet : JetSource
        Momentum-carrying source.
    wind_speed, wind_dir : Array
        Wind speed [m/s] and direction [rad].
    t : Array
        Elapsed time(s) since release [s]. Any broadcastable shape.

    Returns
    -------
    Array, shape ``(..., 2)``
        World-frame velocity ``(vx, vy)`` [m/s].
    """
    a, b, tau, wx, wy = relaxation_time(jet, wind_speed, wind_dir)
    U = wind_speed
    decay = jnp.exp(-t / tau)
    v_par = U + (a - U) * decay
    v_perp = b * decay
    vx = v_par * wx - v_perp * wy
    vy = v_par * wy + v_perp * wx
    return jnp.stack([vx, vy], axis=-1)


def centerline(
    jet: JetSource,
    wind_speed: Array,
    wind_dir: Array,
    t: Array,
) -> Array:
    r"""Bent plume centreline position ``(x, y)`` [m] at elapsed time ``t`` [s].

    Closed-form integral of :func:`parcel_velocity`. Starts at the source and
    asymptotes to a straight line parallel to the wind axis, offset by the
    accumulated cross-wind deflection ``b * tau``.

    Parameters
    ----------
    jet : JetSource
        Momentum-carrying source.
    wind_speed, wind_dir : Array
        Wind speed [m/s] and direction [rad].
    t : Array
        Elapsed time(s) since release [s]. Any broadcastable shape.

    Returns
    -------
    Array, shape ``(..., 2)``
        World-frame centreline position ``(x, y)`` [m].
    """
    a, b, tau, wx, wy = relaxation_time(jet, wind_speed, wind_dir)
    U = wind_speed
    decay = jnp.exp(-t / tau)
    x_par = U * t + (a - U) * tau * (1.0 - decay)
    x_perp = b * tau * (1.0 - decay)
    x = jet.location.x + x_par * wx - x_perp * wy
    y = jet.location.y + x_par * wy + x_perp * wx
    return jnp.stack([x, y], axis=-1)
