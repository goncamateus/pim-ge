# pim-ge

Reimplementation of **Newman et al. (2024)** — *Probabilistic Inversion Modeling of Gas Emissions: A Gradient-Based MCMC Estimation of Gaussian Plume Parameters* (arXiv:2408.01298 / Ann. Appl. Stat. 19(4), 2025).

JAX-based package. Forward model builds the Gaussian plume coupling matrix (ppm per kg/s); inverse model runs Manifold-MALA-within-Gibbs to recover source location, emission rate, and dispersion parameters from sensor data.

## Install

```bash
uv sync                          # core (jax, numpy, scipy)
uv sync --extra examples         # + matplotlib + jupyter
uv sync --extra reproduction     # + matplotlib + jupyter + pandas
uv sync --extra dev              # + ruff + pytest
uv sync --extra gpu              # + jax[cuda12] for GPU acceleration
```

## Package layout

```
src/pim_ge/
├── forward/
│   ├── plume.py      # Gaussian plume → coupling matrix A   (paper §2)
│   ├── wind.py       # Ornstein-Uhlenbeck wind field         (paper §2)
│   └── sensors.py    # sensor layouts + data generation      (paper §2)
├── inverse/
│   ├── priors.py     # prior specification                   (paper §3)
│   ├── gibbs.py      # conjugate Gibbs updates               (paper §3)
│   └── mcmc.py       # M-MALA-within-Gibbs + mwg_scan        (paper §3)
└── utils/
    └── types.py      # Grid, SourceLocation dataclasses

examples/
├── rotating_source.py                 # deterministic wind sweep: scatter cloud + ground footprint + xz cross-section
├── fixed_source.py   # same 3-panel animation, OU (stochastic) wind
└── _viz.py                        # shared figure/colorbar/save-or-show scaffolding for the two scripts above

reproduction/
├── section4_simulation_study.py      # §4 — DPV × WDC × SER sweep (12 scenarios)
└── section5_chilbolton.py            # §5 — Chilbolton beam-sensor case study (data req.)
```

## Core concept

Measurement model:

```
data[t, n] = A[t, n] * s + beta[n] + noise[t, n]
```

- **A** `(T, N_sensors)` — coupling matrix in **ppm per kg/s**, built from plume physics
- **s** — scalar emission rate [kg/s]; sampled as `x[4] = log_s`
- **beta** `(N_sensors,)` — per-sensor background [ppm]; Gibbs-updated exactly
- **noise** — Gaussian with variance σ²; σ² Gibbs-updated exactly

## Sampled parameter vector

```
x = [log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y]   # dim 7
```

Log-space keeps positivity automatic. `beta` and `sigma²` outside `x` — exact conjugate Gibbs.

## Dispersion schemes

All schemes take `stability_class` (Pasquill–Gifford A–F, default `"D"` = neutral).

| `scheme=`    | `estimated=` | Formula                                              |
|-------------|-------------|------------------------------------------------------|
| `"Briggs"`  | `False`      | `a·x·(1+c·x)^exp` — per-class coefficients          |
| `"SMITH"`   | `False`      | `a·x^b` — per-class power law                       |
| `"Draxler"` | `False`      | `a·(tan_γ·x)^b + source_half_width`                 |
| any         | `True`       | Same formula; `a_H, b_H, a_V, b_V` inferred from data |

## Wind field

```python
from pim_ge.forward.wind import (
    wind_speed,                 # OU speed, clipped ≥ 1.0 m/s
    wind_direction,             # OU direction [rad]
    wind_direction_linear,      # linear sweep start_deg → end_deg
    wind_direction_sinusoidal,  # OU around sinusoidal mean (WDC mode)
)
```

## Quick usage

```python
import jax
import jax.numpy as jnp
from pim_ge import (
    SourceLocation, WindField,
    temporal_gridfree_coupling_matrix,
    Priors, GibbsSamplers, mwg_scan,
)
from pim_ge.forward.sensors import circle_of_sensors
from pim_ge.forward.wind import wind_speed, wind_direction

key = jax.random.PRNGKey(0)
T, N = 100, 8

# wind
wind = WindField(
    speed=wind_speed(key, T, mean=2.5),
    direction=wind_direction(key, T, mean=0.0),
)

# sensors on circle, source at origin
sensors = circle_of_sensors(0.0, 0.0, radius=200.0, n_sensors=N)
source_z = 1.5

# coupling function for inversion (source location comes from x[5:7])
def coupling_fn(x):
    src = SourceLocation(x=x[5], y=x[6], z=source_z)
    return temporal_gridfree_coupling_matrix(
        src, sensors, wind,
        scheme="SMITH", stability_class="D",
        estimated=True, log_params=x[:4],
    )

# run sampler
priors = Priors(log_s_mean=-2.0, log_s_std=3.0, source_x_std=200.0, source_y_std=200.0)
gibbs  = GibbsSamplers(priors)

chains = mwg_scan(
    key,
    x_init=jnp.zeros(7),
    sigma2_init=1.0,
    background_init=jnp.zeros(N),
    data=data,                   # (T, N) measured ppm
    coupling_fn=coupling_fn,
    priors=priors,
    gibbs=gibbs,
    step_size_init=0.01,
    adaptation="Optimal",
    iters=5_000,
)

# posterior samples after burn-in
x_post  = chains["x_chain"][1000:]
src_x   = jnp.median(x_post[:, 5])
src_y   = jnp.median(x_post[:, 6])
log_s   = jnp.median(x_post[:, 4])
```

Return keys: `x_chain`, `sigma2_chain`, `background_chain`, `log_posterior_chain`,
`step_size_chain`, `accept_chain`, `accept_rate_chain`.

## Beam sensors (line-of-sight FTIR)

```python
from pim_ge.forward.plume import beam_path_coupling_matrix

A_beam = beam_path_coupling_matrix(
    source, beam_starts, beam_ends, wind,   # beam_starts/ends: (N_beams, 3) [m]
    n_samples=50,                           # integration points per beam
)  # → (T, N_beams) [ppm·m per kg/s]
```

Divide by beam length to convert to path-average [ppm per kg/s].

## Run examples and reproduction

```bash
# 3D animated plume — wind direction sweeps 0→360° over T frames
# args: --class [A-F]  --frames N  --fps N  --show
uv run --extra examples examples/rotating_source.py --class D --frames 100 --fps 10
# → examples/plume_3d_classD.mp4  (or .gif if ffmpeg absent)

# same 3-panel animation, stochastic (Ornstein-Uhlenbeck) wind instead of a fixed sweep
uv run --extra examples examples/fixed_source.py --class A --frames 100 --fps 10

# Section 4 simulation study (12 scenarios, ~10 min at ITERS=2000)
uv run python reproduction/section4_simulation_study.py

# Section 5 Chilbolton (data download instructions printed if Data/ absent)
uv run python reproduction/section5_chilbolton.py
```

## Development

```bash
uv run pytest                        # 59 tests
uv run ruff check src/               # lint (ruff in dev extras)
uv run ruff check --fix src/
```

## API docs

Docstrings carry full NumPy-style signatures plus a "Paper Mapping" note
(equation/section/figure in Newman et al. 2024) and LaTeX where an equation
exists. Build static HTML with [pdoc](https://pdoc.dev):

```bash
uv run pdoc --math -o docs pim_ge    # → docs/pim_ge.html
uv run pdoc --math pim_ge            # serve locally instead of writing files
```

## Reference

Newman, T., Sherlock, C., Whittle, M., & Gałkowski, M. (2024).
*Probabilistic Inversion Modeling of Gas Emissions: A Gradient-Based MCMC Estimation of Gaussian Plume Parameters.*
Ann. Appl. Stat. 19(4). arXiv:2408.01298.
