# AGENTS.md — pim-ge

Guide for AI agents working on this codebase.

## What this project is

JAX reimplementation of Newman et al. (2024) — probabilistic inversion of gas emissions via Gaussian plume + Manifold-MALA-within-Gibbs MCMC. Paper: arXiv:2408.01298.

All logic lives in `src/pim_ge/`. `examples/` and `reproduction/` are runnable scripts that import from it.

---

## Implementation status

**Complete.** All modules are implemented and tested (48 tests pass). Do not re-stub any function.

| Module | Status |
|--------|--------|
| `utils/types.py` | ✅ `Grid`, `SourceLocation` |
| `forward/wind.py` | ✅ OU speed/direction, linear sweep, sinusoidal WDC mode |
| `forward/plume.py` | ✅ dispersion (Briggs/Smith/Draxler per class), 4-term coupling, ppm output, beam integration |
| `forward/sensors.py` | ✅ grid, circle, random layouts + measurement generation |
| `inverse/priors.py` | ✅ log-prior for all 7 params + background + sigma² |
| `inverse/gibbs.py` | ✅ conjugate updates for background (MVN) and sigma² (IG) |
| `inverse/mcmc.py` | ✅ M-MALA step, `mwg_scan` via `jax.lax.scan`, cumulative step-size adaptation |
| `examples/` | ✅ `gaussian_plume_visualization.py` (2D heatmap), `gaussian_plume_3d_visualization.py` (3D scatter + panels) |
| `reproduction/` | ✅ section4 (12-scenario sweep), section5 (stub, needs Chilbolton data) |

---

## Critical implementation details

### Precision

`forward/plume.py` calls `jax.config.update("jax_enable_x64", True)` at module level — matches the original author's code. This enables 64-bit floats globally once the module is imported. Do not remove it; Hessian computations become numerically unstable in 32-bit.

### Coupling matrix (`temporal_gridfree_coupling_matrix`)

- Output shape: `(T, N_sensors)` — NOT flattened
- **Units: ppm per kg/s** (converted via `methane_kg_m3_to_ppm` before return)
- **4 vertical reflection terms**: direct + ground image + inversion-layer image + 2nd-order ceiling image
  ```python
  exp_z = exp_z_direct + exp_z_ground + exp_z_inversion + exp_z_ceiling2
  # ceiling2 = exp(-0.5 * ((dz + 2*H) / sig_z)^2)
  ```
- Negative downwind distance → zero (via `downwind_mask`, not NaN)
- Source location comes from `x[5]` and `x[6]` inside `coupling_fn` closure

### Dispersion formulas

**Briggs** — `a·x·(1+c·x)^exp`, NOT a power law:
```python
_BRIGGS_Y = {
    "D": (0.08, 0.0015, -0.5),   # σ_y = 0.08·x·(1+0.0015·x)^-0.5
    ...
}
```

**Smith** — simple power law per class:
```python
_SMITH = {"D": (0.32, 0.78, 0.22, 0.78)}  # (a_y, b_y, a_z, b_z)
```

**Draxler** — needs `tan_gamma_H/V` parameters:
```python
σ_y = a_H * (tan_gamma_H * x)^b_H + source_half_width
σ_z = a_V * (tan_gamma_V * x)^b_V
```

### Sampled parameter vector

```
x = [log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y]   # length 7
```

`beta` (background per sensor) and `sigma²` are NOT in `x` — exact conjugate Gibbs.

### `mwg_scan`

- Uses `jax.lax.scan` over `iters` keys — not a Python loop
- Takes `coupling_fn: (x) -> A (T, N)`, NOT a pre-built log_posterior_fn
- **Step-size adaptation**: cumulative ratio form — NOT instantaneous exp:
  ```python
  sum_accept_new = sum_accept + accepted
  iteration_new  = iteration + 1.0
  step_size_new  = step_size * (1.0 + 0.1 * (sum_accept_new / iteration_new - target_accept))
  ```
- `adaptation` is a Python string → use Python `if/elif` inside body closure, **NOT** `jax.lax.cond`
- Carry: `(x, sigma2, background, step_size, sum_accept, iteration)`
- Return keys: `x_chain`, `sigma2_chain`, `background_chain`, `log_posterior_chain`, `step_size_chain`, `accept_chain`, `accept_rate_chain`

### Hessian preconditioning

```python
H    = jax.jacfwd(jax.jacrev(log_posterior_fn))(x)
vals, vecs = jnp.linalg.eigh(-H)
inv_H = vecs @ jnp.diag(1.0 / jnp.abs(vals)) @ vecs.T  # |λ| keeps PD
```

Use `|λ|` eigendecomposition — raw Hessian may not be PD for poorly-conditioned posteriors.

### Wind field

- `wind_speed`: OU clipped at **1.0 m/s** (not 0.1)
- `wind_direction`: plain OU for constant-mean mode
- `wind_direction_linear(n_steps, start_deg, end_deg)`: linear sweep, degrees → radians
- `wind_direction_sinusoidal(key, n_steps, mean, std, theta, num_periods)`: OU around `mean + std·sin(2π·num_periods·t/n_steps)`

### Beam sensors

`beam_path_coupling_matrix` in `forward/plume.py`:
- Integrates Gaussian plume along each beam path via trapezoid rule
- Output: `(T, N_beams)` in **ppm·m per kg/s**
- Divide by beam length for path-average [ppm per kg/s]

---

## Key invariants (do not break)

| Invariant | Where enforced |
|-----------|---------------|
| Positivity via log-space | `x` vector definition |
| Output units ppm, not kg/m³ | `methane_kg_m3_to_ppm` called inside `temporal_gridfree_coupling_matrix` |
| 4 vertical reflection terms | `exp_z` sum in `temporal_gridfree_coupling_matrix` |
| Briggs formula `a·x·(1+c·x)^exp` (not power law) | `_BRIGGS_Y`, `_BRIGGS_Z` dicts |
| Cumulative step-size (not instantaneous exp) | `mwg_scan` body |
| No `jax.lax.cond` on Python `adaptation` string | `mwg_scan` body — use Python `if` |
| `jax.lax.scan` for MCMC loop | `mwg_scan` |
| Wind speed ≥ 1.0 m/s | `jnp.clip(raw, 1.0)` in `wind_speed` |
| 64-bit precision enabled | `jax.config.update("jax_enable_x64", True)` in `plume.py` |

---

## JAX conventions

- All arrays are JAX arrays (`jax.Array`). No NumPy inside JIT-compiled functions.
- Use `jax.jacfwd(jax.jacrev(...))` for the Hessian.
- Random keys: always split, never reuse. Pattern: `key, subkey = jax.random.split(key)`.
- Functional style: no in-place mutation. Scan carry is a tuple of JAX scalars/arrays.
- Python-level control flow (`if`, `for`) is fine for compile-time constants (scheme names, adaptation strings). Use `jax.lax.cond`/`jax.lax.switch` only for runtime JAX booleans.
- Closures over JAX arrays (e.g. `wind.speed` in `coupling_fn`) are concrete values baked into JIT. Different wind realizations → different JIT compilations. Accept this cost or pass wind as explicit argument.

---

## Testing

```bash
uv run pytest -v       # 48 tests
uv run pytest -q       # brief
```

Tests live in `tests/`. Coverage:
- `test_wind.py` — OU shape, mean reversion, speed ≥ 1.0, linear sweep monotone, sinusoidal shape
- `test_plume.py` — downwind/crosswind geometry, Briggs D formula, Smith D formula, coupling shape/upwind/ppm units/4th-term
- `test_gibbs.py` — background and sigma² conjugate posteriors
- `test_mcmc.py` — Hessian shape, log-posterior scalar, scan shapes, finite values, acceptance, step-size shrinks
- `test_priors.py` — prior scalar, mode, sigma² decreasing, background
- `test_sensors.py` — grid/circle/random shapes, measurements mean
- `test_types.py` — Grid uniform, SourceLocation

---

## Lint / format

```bash
uv run ruff check src/ examples/ reproduction/   # lint all Python
uv run ruff check --fix src/ examples/ reproduction/
```

Rules: E, F, I (isort), UP (pyupgrade), B (bugbear), SIM. Line length 100.

---

## File map (paper → code)

| Paper section | Module | Key symbol |
|--------------|--------|-----------|
| §2 wind field | `forward/wind.py` | `WindField`, `generate_ornstein_uhlenbeck`, `wind_direction_sinusoidal` |
| §2 dispersion | `forward/plume.py` | `horizontal_stddev`, `vertical_stddev`, `_BRIGGS_Y`, `_BRIGGS_Z`, `_SMITH` |
| §2 coupling matrix | `forward/plume.py` | `temporal_gridfree_coupling_matrix` |
| §2 beam sensors | `forward/plume.py` | `beam_path_coupling_matrix` |
| §2 sensors | `forward/sensors.py` | `Sensors`, `circle_of_sensors`, `temporal_sensors_measurements` |
| §3 priors | `inverse/priors.py` | `Priors.log_prior`, `log_prior_background`, `log_prior_sigma2` |
| §3 Gibbs | `inverse/gibbs.py` | `GibbsSamplers.background_conditional_posterior`, `measurement_error_var_conditional_posterior` |
| §3 M-MALA | `inverse/mcmc.py` | `ManifoldMALAWithinGibbs.manifold_mala_step` |
| §3 full loop | `inverse/mcmc.py` | `mwg_scan`, `build_log_posterior` |
| §4 sim study | `reproduction/section4_simulation_study.py` | DPV × WDC × SER sweep |
| §5 Chilbolton | `reproduction/section5_chilbolton.py` | beam-sensor inversion (needs data) |
| examples | `examples/gaussian_plume_visualization.py` | 2D ppm heatmap (T=1 snapshot) |
| examples | `examples/gaussian_plume_3d_visualization.py` | 3D scatter core + ground footprint + xz cross-section |

---

## Chilbolton data (§5)

No data bundled. `section5_chilbolton.py` prints download instructions and exits cleanly if `Data/` absent.

Expected files under `Data/`:
```
Data/
    Chilbolton_Source1_preprocessed.npz
    chilbolton_beam_geometry.npz
```

NPZ keys: `measurements (T, N_beams)`, `beam_starts (N_beams, 3)`, `beam_ends (N_beams, 3)`,
`wind_speed (T,)`, `wind_direction (T,)`, `release_rate`, `release_x`, `release_y`, `release_z`.
