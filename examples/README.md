# Examples

Two animated 3D plume visualizations built on the forward model
(`pim_ge.forward.plume.temporal_gridfree_coupling_matrix`). Both render the
same 3-panel layout — 3D scatter cloud, ground footprint (max over z), and a
vertical cross-section at y=0 — over an evolving wind field, and save the
result as MP4 (falling back to GIF, then to an interactive window).

That shared figure/colorbar/save-or-show scaffolding lives in `_viz.py`
(`build_figure`, `init_colorbars`, `save_or_show`, `STABILITY_LABELS`) so
each script below only contains what's actually specific to it: grid
domain, wind model, camera motion, and the per-frame `update()`.

## `gaussian_3d.py` — deterministic wind sweep

Wind speed is constant; wind direction sweeps a full 0→360° circle over the
animation. Demonstrates the plume's shape for a chosen Pasquill–Gifford
stability class as it rotates around a fixed source.

```bash
uv run --extra examples examples/gaussian_3d.py --class D --frames 100 --fps 10
```

### Constants

| Constant | Meaning |
|---|---|
| `EMISSION_RATE` | Source strength `s` [kg/s], multiplied into the unit coupling matrix `A` to get concentration. |
| `WIND_SPEED` | Constant wind speed [m/s] for every frame — only direction varies. |
| `SOURCE_Z` | Release height [m] of the point source. |
| `MIXING_HEIGHT` | Boundary-layer ceiling [m] used by the plume's inversion-layer reflection term. |
| `CORE_FRAC` | Fraction of each frame's peak concentration used as the cutoff for which grid points get drawn in the 3D scatter cloud (lower = more sparse points shown). |
| `NX`, `NY`, `NZ` | Grid resolution along x/y (ground plane) and z (height). |
| `STABILITY_LABELS` | Human-readable name for each Pasquill–Gifford class A–F (e.g. "D — Neutral"). |

### Functions

| Function | What it does |
|---|---|
| `parse_args()` | Reads `--class`, `--frames`, `--fps`, `--show` from the CLI. |
| `build_grid()` | Builds the 600×600 m evaluation grid (centred on the source, valid for any wind direction) the plume is sampled on. |
| `main()` | Simulates the wind sweep, evaluates the plume for all frames at once, builds the figure, and saves/shows the animation. |
| `main._setup_ax3()` | Resets the 3D axes' styling/limits/camera angle after each per-frame `cla()`. |
| `main.update(t)` | Redraws frame `t`: 3D scatter cloud (thresholded by `CORE_FRAC`), ground-footprint heatmap, vertical cross-section, and the title bar (direction/speed/peak). |

## `gaussian_3d_unstable_wind.py` — stochastic wind (Ornstein-Uhlenbeck)

Both wind speed and direction evolve as independent Ornstein-Uhlenbeck (OU)
processes (`pim_ge.forward.wind.wind_speed`, `wind_direction`), so the
plume meanders and pulses instead of sweeping a clean circle. Grid is
narrower and downwind-only since direction barely drifts from 0.

```bash
uv run --extra examples examples/gaussian_3d_unstable_wind.py --class A --frames 100 --fps 10 --seed 0
```

### Constants

| Constant | Meaning |
|---|---|
| `EMISSION_RATE` | Source strength `s` [kg/s]. |
| `SOURCE_Z` | Release height [m] of the point source. |
| `MIXING_HEIGHT` | Boundary-layer ceiling [m] — set low here so ground/ceiling reflections dominate close to the source. |
| `CORE_FRAC` | Same scatter-cloud cutoff fraction as in `gaussian_3d.py`. |
| `START_X`/`END_X`, `NX` | Downwind grid bounds [m] and resolution along x. |
| `START_Y`/`END_Y`, `NY` | Crosswind grid bounds [m] and resolution along y. |
| `START_Z`/`END_Z`, `NZ` | Vertical grid bounds [m] (centred on `SOURCE_Z`) and resolution along z. |
| `SPEED_MEAN`, `SPEED_STD`, `SPEED_THETA` | OU mean, diffusion std, and mean-reversion rate for wind **speed** [m/s]. Higher `SPEED_STD`/`SPEED_THETA` → burstier speed. |
| `DIR_MEAN`, `DIR_STD`, `DIR_THETA` | OU mean, diffusion std, and mean-reversion rate for wind **direction** [rad]. Small `DIR_STD`/`DIR_THETA` → slow, smooth meander rather than full rotation. |
| `STABILITY_LABELS` | Same as in `gaussian_3d.py`. |

### Functions

| Function | What it does |
|---|---|
| `parse_args()` | Reads `--class`, `--frames`, `--fps`, `--seed`, `--show` from the CLI. |
| `build_grid()` | Builds the downwind-only evaluation grid (`START_*`/`END_*`/`N*` constants). |
| `main()` | Simulates the OU wind realization, evaluates the plume for all frames at once, builds the figure, and saves/shows the animation. |
| `main._setup_ax3()` | Resets the 3D axes' styling/limits/camera angle after each per-frame `cla()`. |
| `main.update(t)` | Redraws frame `t`: 3D scatter cloud, ground-footprint heatmap, vertical cross-section, and the title bar (current sampled direction/speed/peak). |
