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

Every physical/grid/OU/jet parameter is a CLI flag (no module constants) —
run `--help` for the full list with defaults. The notable ones:

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--emission-rate` | `0.9` | Source strength `s` [kg/s]. |
| `--source-z` | `25.0` | Release height [m] of the point source. |
| `--mixing-height` | `300.0` | Boundary-layer ceiling [m] — kept low so ground/ceiling reflections dominate close to the source. |
| `--core-frac` | `0.01` | Same scatter-cloud cutoff fraction as in `gaussian_3d.py`. |
| `--start-x`/`--end-x`, `--nx` | `0.0`/`50.0`, `40` | Downwind grid bounds [m] and resolution along x. |
| `--start-y`/`--end-y`, `--ny` | `-25.0`/`25.0`, `40` | Crosswind grid bounds [m] and resolution along y. |
| `--start-z`/`--end-z`, `--nz` | `0.0`/`50.0`, `35` | Vertical grid bounds [m] and resolution along z. |
| `--speed-mean`/`--speed-std`/`--speed-theta` | `2.0`/`2.0`/`0.5` | OU mean, diffusion std, and mean-reversion rate for wind **speed** [m/s]. Higher std/theta → burstier speed. |
| `--dir-mean`/`--dir-std`/`--dir-theta` | `0.0`/`0.05`/`0.01` | OU mean, diffusion std, and mean-reversion rate for wind **direction** [rad]. Small std/theta → slow, smooth meander rather than full rotation. |
| `--jet-speed`/`--jet-angle`/`--jet-diameter` | `20.0`/`15.0`/`0.2` | Optional momentum-carrying source: exit speed [m/s], exit direction [deg, world frame], diameter [m] for `L_relax`. `--jet-speed 0` disables the jet. |

`STABILITY_LABELS` (imported from `_viz.py`) is the same Pasquill–Gifford
class-name lookup as in `gaussian_3d.py`.

### Functions

| Function | What it does |
|---|---|
| `parse_args()` | Reads `--class`, `--frames`, `--fps`, `--seed`, `--show`, and the full set of CLI flags above. |
| `build_grid(args)` | Builds the downwind-only evaluation grid from `args.start_*`/`args.end_*`/`args.n*`. |
| `main()` | Simulates the OU wind realization, evaluates the plume for all frames at once, builds the figure, and saves/shows the animation. |
| `main._setup_ax3(t)` | Resets the 3D axes' styling/limits/camera angle after each per-frame `cla()`. |
| `main.update(t)` | Redraws frame `t`: 3D scatter cloud, ground-footprint heatmap, vertical cross-section, and the title bar (current sampled direction/speed/peak). |
