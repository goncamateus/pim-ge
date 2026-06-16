r"""Gaussian plume forward model — §2 of Newman et al. (2024).

Core output: coupling matrix A such that data = A @ s + beta + noise (Eq. 5).
The plume concentration itself (Eq. 2, §2.1) is the steady-state solution of
the advection-diffusion equation (Eq. 1, §2.1) with image-source reflections
off the ground and the mixing-layer ceiling.
"""

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from pim_ge.forward.wind import WindField
from pim_ge.utils.types import SourceLocation

jax.config.update("jax_enable_x64", True)

DispersionScheme = Literal["Briggs", "SMITH", "Draxler"]

# Paper Mapping: extension beyond Newman et al. (2024). The paper parametrizes
# dispersion only via the estimated power law of Eq. (3); these fixed-coefficient
# Pasquill-Gifford tables (Briggs open-country, Smith power-law) are standard
# atmospheric-dispersion references used here to generate synthetic "true" sigma
# values for the simulation study, not equations from the paper.

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
    r"""Scalar projection of the (sensor - source) vector onto the wind axis.

    Parameters
    ----------
    sensor_x : Array
        Sensor east-west coordinate(s) [m].
    sensor_y : Array
        Sensor north-south coordinate(s) [m].
    source : SourceLocation
        Emission source position.
    wind_dir : Array
        Direction the wind is *blowing toward* [rad].

    Returns
    -------
    Array
        Downwind distance `delta_R`, shape `(T,)` or scalar depending on the
        broadcast shape of `wind_dir`.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), §2.1/§2.2 — the downwind distance
    :math:`\delta_R` used as the argument of the dispersion power law (Eq. 3)
    and to rotate sensor offsets into the plume-centered (downwind,
    crosswind) frame for Eq. (2).
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
    r"""Signed cross-wind offset, perpendicular to the wind axis.

    Parameters
    ----------
    sensor_x : Array
        Sensor east-west coordinate(s) [m].
    sensor_y : Array
        Sensor north-south coordinate(s) [m].
    source : SourceLocation
        Emission source position.
    wind_dir : Array
        Direction the wind is *blowing toward* [rad].

    Returns
    -------
    Array
        Crosswind offset `delta_H`, same broadcast shape as `downwind_distance`.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (2), §2.1 — the horizontal offset
    :math:`\delta_H` entering the crosswind Gaussian term
    :math:`\exp(-\delta_H^2 / 2\sigma_H^2)`.
    """
    dx = sensor_x - source.x
    dy = sensor_y - source.y
    px = -jnp.sin(wind_dir)
    py = jnp.cos(wind_dir)
    return dx * px + dy * py


def vertical_offset(sensor_z: Array, source: SourceLocation) -> Array:
    r"""Vertical offset between sensor height and source height.

    Parameters
    ----------
    sensor_z : Array
        Sensor height(s) [m].
    source : SourceLocation
        Emission source position.

    Returns
    -------
    Array
        Vertical offset `delta_V = sensor_z - source.z` [m].

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (2), §2.1 — the vertical offset
    :math:`\delta_V` (here the direct, unreflected term; ground/inversion-layer
    reflections add image-source offsets in
    `temporal_gridfree_coupling_matrix`).
    """
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
    r"""Horizontal (crosswind) dispersion coefficient :math:`\sigma_H` [m].

    Parameters
    ----------
    x_down : Array
        Downwind distance(s) `delta_R` [m] (must be > 0; caller masks/clamps
        non-downwind sensors before calling).
    scheme : {"Briggs", "SMITH", "Draxler"}, default "Briggs"
        Dispersion parametrization. Only consulted when `estimated=False`.
    stability_class : str, default "D"
        Pasquill-Gifford stability class "A"-"F", used by the fixed `"Briggs"`
        / `"SMITH"` lookup tables.
    estimated : bool, default False
        If True, use the inferred power-law form with `a_H`, `b_H` instead of
        a fixed-table scheme.
    a_H, b_H : float, optional
        Power-law coefficient/exponent (required if `estimated=True`).
    tan_gamma_H : float, default 1.0
        `tan(gamma_H)`, the horizontal wind-direction roughness term (used by
        `"Draxler"` and the estimated form).
    source_half_width : float, default 0.0
        Virtual source offset `w` added to the estimated/Draxler power law.

    Returns
    -------
    Array
        :math:`\sigma_H` [m], same shape as `x_down`.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (3), §2.2 — the estimated branch
    (`estimated=True` or `scheme="Draxler"`) implements

    .. math::
        \sigma_H = a_H(\delta_R \tan\gamma_H)^{b_H} + w

    The `"Briggs"`/`"SMITH"` fixed-table branches are extensions beyond the
    paper (standard Pasquill-Gifford open-country curves), used only to
    generate synthetic "true" dispersion for the simulation study.
    """
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
    r"""Vertical dispersion coefficient :math:`\sigma_V` [m].

    Parameters
    ----------
    x_down : Array
        Downwind distance(s) `delta_R` [m] (must be > 0).
    scheme : {"Briggs", "SMITH", "Draxler"}, default "Briggs"
        Dispersion parametrization. Only consulted when `estimated=False`.
    stability_class : str, default "D"
        Pasquill-Gifford stability class "A"-"F", used by the fixed `"Briggs"`
        / `"SMITH"` lookup tables.
    estimated : bool, default False
        If True, use the inferred power-law form with `a_V`, `b_V`.
    a_V, b_V : float, optional
        Power-law coefficient/exponent (required if `estimated=True`).
    tan_gamma_V : float, default 1.0
        `tan(gamma_V)`, the vertical wind-direction roughness term (used by
        `"Draxler"` and the estimated form).

    Returns
    -------
    Array
        :math:`\sigma_V` [m], same shape as `x_down`.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (3), §2.2 — the estimated branch
    implements

    .. math::
        \sigma_V = a_V(\delta_R \tan\gamma_V)^{b_V} + h

    (the `+h` source-height offset is folded into the reflection terms of
    `temporal_gridfree_coupling_matrix` rather than added here). The
    `"Briggs"`/`"SMITH"` fixed-table branches are extensions beyond the paper,
    as in `horizontal_stddev`.
    """
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
    r"""Build the source-sensor coupling matrix `A` [ppm per kg/s].

    Parameters
    ----------
    source : SourceLocation
        Emission source position :math:`(\tilde{x}, \tilde{y}, \tilde{z})`.
    sensor_positions : Array, shape (N_sensors, 3)
        Sensor `(x, y, z)` coordinates [m].
    wind : WindField
        Per-timestep wind speed/direction, length `T`.
    mixing_height : float, default 500.0
        Boundary-layer mixing height `H` [m], used by the inversion-layer
        reflection term.
    scheme : {"Briggs", "SMITH", "Draxler"}, default "Briggs"
        Dispersion parametrization (ignored if `estimated=True`).
    stability_class : str, default "D"
        Pasquill-Gifford stability class, used by fixed-table schemes.
    estimated : bool, default False
        If True, infer `a_H, b_H, a_V, b_V` from `log_params` instead of a
        fixed-table scheme.
    log_params : Array, optional
        `[log_a_H, log_a_V, log_b_H, log_b_V]`, exponentiated to recover the
        positive dispersion coefficients (required if `estimated=True`).
    tan_gamma_H, tan_gamma_V : float, default 1.0
        Wind-direction roughness terms for the horizontal/vertical power law.
    source_half_width : float, default 0.0
        Virtual source offset `w` for the horizontal dispersion term.

    Returns
    -------
    Array, shape (T, N_sensors)
        Coupling matrix `A` [ppm per kg/s], such that `data = A @ s + beta + noise`
        (Eq. 5). Entries are 0 for sensors upwind of the source (`x_down <= 0`).

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (2), §2.1 — steady-state Gaussian
    plume solution of the advection-diffusion equation (Eq. 1) with
    image-source reflections, scaled to ppm via `methane_kg_m3_to_ppm`
    (the `1e6/rho_CH4` factor in Eq. 2).

    .. math::
        c = \frac{10^6}{\rho_{CH_4}}\,
            \frac{s}{2\pi u\,\sigma_H\sigma_V}\,
            \exp\!\left(-\frac{\delta_H^2}{2\sigma_H^2}\right)
            \sum_{j} \exp\!\left(-\frac{\delta_{V,j}^2}{2\sigma_V^2}\right)

    The implementation truncates the image-source sum over `j` to four terms:
    direct (`delta_V`), ground reflection (`delta_V + 2h`), inversion-layer
    reflection (`delta_V - 2(H - h)`), and one second-order ceiling reflection
    (`delta_V + 2H`), where `h = source.z` and `H = mixing_height`.
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
    """Path-integrated coupling [ppm*m per kg/s] for line-of-sight beam sensors.

    Samples the Gaussian plume at `n_samples` points along each beam and
    integrates with the trapezoid rule. Divide by beam length to obtain a
    path-average coupling [ppm per kg/s].

    Parameters
    ----------
    source : SourceLocation
        Emission source position.
    beam_starts : Array, shape (N_beams, 3)
        Beam start coordinates [m].
    beam_ends : Array, shape (N_beams, 3)
        Beam end coordinates [m].
    wind : WindField
        Per-timestep wind speed/direction, length `T`.
    n_samples : int, default 50
        Number of integration points per beam.
    mixing_height : float, default 500.0
        Boundary-layer mixing height `H` [m].
    scheme : {"Briggs", "SMITH", "Draxler"}, default "Briggs"
        Dispersion parametrization (ignored if `estimated=True`).
    stability_class : str, default "D"
        Pasquill-Gifford stability class.
    estimated : bool, default False
        If True, infer dispersion coefficients from `log_params`.
    log_params : Array, optional
        `[log_a_H, log_a_V, log_b_H, log_b_V]`.
    tan_gamma_H, tan_gamma_V : float, default 1.0
        Wind-direction roughness terms.

    Returns
    -------
    Array, shape (T, N_beams)
        Path-integrated coupling [ppm*m per kg/s].

    Notes
    -----
    Paper Mapping: Newman et al. (2024), §5 (Chilbolton case study) — the
    paper's §5 analysis uses open-path FTIR beam sensors, which observe a
    path-integral of the point concentration (Eq. 2) rather than a point
    value. The paper does not give an explicit integral formula or specify a
    numerical quadrature; the trapezoid-rule sampling here
    (`temporal_gridfree_coupling_matrix` evaluated at `n_samples` points along
    each beam) is this implementation's numerical realization of that
    path-integral concept.
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
    r"""Convert methane mass concentration [kg/m^3] to mixing ratio [ppm].

    Parameters
    ----------
    concentration : Array
        Methane concentration [kg/m^3].

    Returns
    -------
    Array
        Concentration in ppm, same shape as `concentration`.

    Notes
    -----
    Paper Mapping: Newman et al. (2024), Eq. (2), §2.1 — the
    :math:`10^6/\rho_{CH_4}` conversion factor applied to the Gaussian plume
    solution to express output in ppm. Uses :math:`\rho_{CH_4} = 0.671` kg/m^3
    (methane density at 15 deg C, 1 atm).
    """
    return concentration * 1e6 / 0.671
