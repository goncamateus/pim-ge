"""pim-ge: JAX reimplementation of Newman et al. (2024), arXiv:2408.01298.

Public API for the forward Gaussian plume model (§2) and the inverse
M-MALA-within-Gibbs sampler (§3).
"""

from pim_ge.forward.plume import temporal_gridfree_coupling_matrix
from pim_ge.forward.sensors import Sensors, SensorsSettings
from pim_ge.forward.wind import WindField
from pim_ge.inverse.gibbs import GibbsSamplers
from pim_ge.inverse.mcmc import inverse_hessian, mwg_scan, sqrt_inv_hess
from pim_ge.inverse.priors import Priors
from pim_ge.utils.types import Grid, SourceLocation

__all__ = [
    "Grid",
    "SourceLocation",
    "WindField",
    "SensorsSettings",
    "Sensors",
    "temporal_gridfree_coupling_matrix",
    "Priors",
    "GibbsSamplers",
    "inverse_hessian",
    "sqrt_inv_hess",
    "mwg_scan",
]
