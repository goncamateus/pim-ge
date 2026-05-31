"""Gaussian plume forward model — §2 of Newman et al. (2024).

Core output: coupling matrix A such that data = A @ s + beta + noise.
"""

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from pim_ge.forward.wind import WindField
from pim_ge.utils.types import SourceLocation

jax.config.update("jax_enable_x64", True)

DispersionScheme = Literal["Briggs", "SMITH", "Draxler"]

# Briggs open-country: (a, c, exp) where σ = a * x * (1 + c*x)^exp
_BRIGGS_Y = {
    "A": (0.22, 0.0001, -0.5),
    "B": (0.16, 0.0001, -0.5),
    "C": (0.11, 0.0002, -0.5),
    "D": (0.08, 0.0015, -0.5),
    "E": (0.06, 0.0003, -1.0),
    "F": (0.04, 0.0003, -1.0),
}
# c=0 entries use σ_z = a * x (linear)
_BRIGGS_Z = {
    "A": (0.20, 0.0, 1.0),
    "B": (0.12, 0.0, 1.0),
    "C": (0.08, 0.0002, -0.5),
    "D": (0.06, 0.0015, -0.5),
    "E": (0.03, 0.0003, -1.0),
    "F": (0.016, 0.0003, -1.0),
}

# Smith power-law: (a_y, b_y, a_z, b_z)
_SMITH = {
    "B": (0.40, 0.91, 0.41, 0.91),
    "C": (0.36, 0.86, 0.33, 0.86),
    "D": (0.32, 0.78, 0.22, 0.78),
}


def downwind_distance(
    sensor_x: Array,
    sensor_y: Array,
    source: SourceLocation,
    wind_dir: Array,
) -> Array:
    """Scalar projection of (sensor − source) vector onto wind axis.

    wind_dir is the direction the wind is *blowing toward* [rad].
    Returns (T,) or scalar depending on wind_dir shape.
    """
    dx = sensor_x - source.x
    dy = sensor_y - source.y
    ux = jnp.cos(wind_dir)
    uy = jnp.sin(wind_dir)
    return dx * ux + dy * uy


def horizontal_offset(
    sensor_x: Array,
    sensor_y: Array,
    source: SourceLocation,
    wind_dir: Array,
) -> Array:
    """Cross-wind offset (perpendicular to wind axis, signed)."""
    dx = sensor_x - source.x
    dy = sensor_y - source.y
    px = -jnp.sin(wind_dir)
    py = jnp.cos(wind_dir)
    return dx * px + dy * py


def vertical_offset(sensor_z: Array, source: SourceLocation) -> Array:
    return sensor_z - source.z


def horizontal_stddev(
    x_down: Array,
    scheme: DispersionScheme = "Briggs",
    stability_class: str = "D",
    estimated: bool = False,
    a_H: float | None = None,
    b_H: float | None = None,
    tan_gamma_H: float = 1.0,
    source_half_width: float = 0.0,
) -> Array:
    """sigma_y [m]."""
    if estimated:
        assert a_H is not None and b_H is not None
        if scheme == "Draxler":
            return a_H * jnp.power(tan_gamma_H * x_down, b_H) + source_half_width
        return a_H * jnp.power(x_down, b_H)
    if scheme == "Briggs":
        a, c, exp = _BRIGGS_Y[stability_class]
        return a * x_down * jnp.power(1.0 + c * x_down, exp)
    if scheme == "SMITH":
        a, b, _, _ = _SMITH[stability_class]
        return a * jnp.power(x_down, b)
    # Draxler fixed
    a, b = 0.22, 0.5
    return a * jnp.power(tan_gamma_H * x_down, b) + source_half_width


def vertical_stddev(
    x_down: Array,
    scheme: DispersionScheme = "Briggs",
    stability_class: str = "D",
    estimated: bool = False,
    a_V: float | None = None,
    b_V: float | None = None,
    tan_gamma_V: float = 1.0,
) -> Array:
    """sigma_z [m]."""
    if estimated:
        assert a_V is not None and b_V is not None
        if scheme == "Draxler":
            return a_V * jnp.power(tan_gamma_V * x_down, b_V)
        return a_V * jnp.power(x_down, b_V)
    if scheme == "Briggs":
        a, c, exp = _BRIGGS_Z[stability_class]
        if c == 0.0:
            return a * x_down
        return a * x_down * jnp.power(1.0 + c * x_down, exp)
    if scheme == "SMITH":
        _, _, a, b = _SMITH[stability_class]
        return a * jnp.power(x_down, b)
    # Draxler fixed
    a, b = 0.12, 0.5
    return a * jnp.power(tan_gamma_V * x_down, b)


def temporal_gridfree_coupling_matrix(
    source: SourceLocation,
    sensor_positions: Array,  # (N_sensors, 3)
    wind: WindField,
    mixing_height: float = 500.0,
    scheme: DispersionScheme = "Briggs",
    stability_class: str = "D",
    estimated: bool = False,
    log_params: Array | None = None,  # [log_a_H, log_a_V, log_b_H, log_b_V]
    tan_gamma_H: float = 1.0,
    tan_gamma_V: float = 1.0,
    source_half_width: float = 0.0,
) -> Array:
    """Build A ∈ R^{T, N_sensors} [ppm per kg/s].

    4-term Gaussian plume: direct + ground reflection + inversion-layer reflection
    + 2nd-order ceiling reflection. Output converted to ppm.
    """
    T = wind.speed.shape[0]
    N = sensor_positions.shape[0]

    sx, sy, sz = sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2]

    # (T, N) downwind distances and offsets
    x_down = (
        jnp.outer(jnp.ones(T), sx - source.x) * jnp.cos(wind.direction)[:, None]
        + jnp.outer(jnp.ones(T), sy - source.y) * jnp.sin(wind.direction)[:, None]
    )
    y_cross = (
        -jnp.outer(jnp.ones(T), sx - source.x) * jnp.sin(wind.direction)[:, None]
        + jnp.outer(jnp.ones(T), sy - source.y) * jnp.cos(wind.direction)[:, None]
    )
    dz = sz - source.z  # (N,)

    downwind_mask = (x_down > 0.0).astype(jnp.float32)
    x_safe = jnp.where(x_down > 0.0, x_down, jnp.ones_like(x_down))

    if estimated and log_params is not None:
        a_H = jnp.exp(log_params[0])
        a_V = jnp.exp(log_params[1])
        b_H = jnp.exp(log_params[2])
        b_V = jnp.exp(log_params[3])
        sig_y = horizontal_stddev(
            x_safe,
            scheme=scheme,
            estimated=True,
            a_H=a_H,
            b_H=b_H,
            tan_gamma_H=tan_gamma_H,
            source_half_width=source_half_width,
        )
        sig_z = vertical_stddev(
            x_safe,
            scheme=scheme,
            estimated=True,
            a_V=a_V,
            b_V=b_V,
            tan_gamma_V=tan_gamma_V,
        )
    else:
        sig_y = horizontal_stddev(
            x_safe,
            scheme=scheme,
            stability_class=stability_class,
            tan_gamma_H=tan_gamma_H,
            source_half_width=source_half_width,
        )
        sig_z = vertical_stddev(
            x_safe,
            scheme=scheme,
            stability_class=stability_class,
            tan_gamma_V=tan_gamma_V,
        )

    u = wind.speed[:, None]  # (T, 1)

    exp_y = jnp.exp(-0.5 * (y_cross / sig_y) ** 2)

    dz_t = jnp.broadcast_to(dz[None, :], (T, N))
    h = source.z
    H = mixing_height
    exp_z_direct = jnp.exp(-0.5 * (dz_t / sig_z) ** 2)
    exp_z_ground = jnp.exp(-0.5 * ((dz_t + 2 * h) / sig_z) ** 2)
    exp_z_inversion = jnp.exp(-0.5 * ((dz_t - 2 * (H - h)) / sig_z) ** 2)
    exp_z_ceiling2 = jnp.exp(-0.5 * ((dz_t + 2 * H) / sig_z) ** 2)
    exp_z = exp_z_direct + exp_z_ground + exp_z_inversion + exp_z_ceiling2

    coupling = downwind_mask / (2 * jnp.pi * u * sig_y * sig_z) * exp_y * exp_z
    return methane_kg_m3_to_ppm(coupling)


def beam_path_coupling_matrix(
    source: SourceLocation,
    beam_starts: Array,  # (N_beams, 3)
    beam_ends: Array,  # (N_beams, 3)
    wind: WindField,
    n_samples: int = 50,
    mixing_height: float = 500.0,
    scheme: DispersionScheme = "Briggs",
    stability_class: str = "D",
    estimated: bool = False,
    log_params: Array | None = None,
    tan_gamma_H: float = 1.0,
    tan_gamma_V: float = 1.0,
) -> Array:
    """Path-integrated coupling [ppm·m per kg/s] for line-of-sight beam sensors.

    Samples the Gaussian plume at n_samples points along each beam and integrates
    using the trapezoid rule. Divide by beam length to obtain path-average [ppm per kg/s].
    """
    N_beams = beam_starts.shape[0]
    t_samp = jnp.linspace(0.0, 1.0, n_samples)
    directions = beam_ends - beam_starts  # (N_beams, 3)
    # (N_beams, n_samples, 3)
    beam_points = beam_starts[:, None, :] + t_samp[None, :, None] * directions[:, None, :]
    flat_positions = beam_points.reshape(-1, 3)  # (N_beams * n_samples, 3)

    A_flat = temporal_gridfree_coupling_matrix(
        source,
        flat_positions,
        wind,
        mixing_height=mixing_height,
        scheme=scheme,
        stability_class=stability_class,
        estimated=estimated,
        log_params=log_params,
        tan_gamma_H=tan_gamma_H,
        tan_gamma_V=tan_gamma_V,
    )  # (T, N_beams * n_samples)

    A_beams = A_flat.reshape(wind.speed.shape[0], N_beams, n_samples)  # (T, N_beams, n_samples)
    beam_lengths = jnp.linalg.norm(directions, axis=-1)  # (N_beams,)
    dl = beam_lengths / (n_samples - 1)  # [m] per sample interval

    # trapezoid: 0.5*(y[i] + y[i+1]) summed over n_samples-1 intervals
    A_integrated = 0.5 * (A_beams[..., :-1] + A_beams[..., 1:]).sum(axis=-1) * dl[None, :]
    return A_integrated  # (T, N_beams) [ppm·m per kg/s]


def methane_kg_m3_to_ppm(concentration: Array) -> Array:
    """Convert kg/m^3 methane → ppm at 15°C, 1 atm."""
    return concentration * 1e6 / 0.671
