from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


@dataclass
class Grid:
    """Spatial domain for plume evaluation."""

    x: Array  # (Nx,) east-west [m]
    y: Array  # (Ny,) north-south [m]
    z: Array  # (Nz,) vertical [m]

    @classmethod
    def uniform(cls, x_range: tuple, y_range: tuple, z_range: tuple, n: int = 50) -> "Grid":
        return cls(
            x=jnp.linspace(*x_range, n),
            y=jnp.linspace(*y_range, n),
            z=jnp.linspace(*z_range, n),
        )


@dataclass
class SourceLocation:
    """Point source position."""

    x: float  # [m]
    y: float  # [m]
    z: float  # [m]
