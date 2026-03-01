# Gaussian Process Surrogate for Thetis West UK Tidal Model

A 2D shallow-water tidal model of the West UK (Bristol Channel and Irish Sea) built on [Thetis](https://thetisproject.org/) / [Firedrake](https://www.firedrakeproject.org/), coupled with a Gaussian Process (GP) surrogate for uncertainty quantification (UQ) of bathymetric and friction parameters.

---

## Overview

This repository implements a workflow for assessing the sensitivity of tidal dynamics in the West UK to two key uncertain parameters:

- **Bathymetric error** (`bath_error`): a uniform perturbation applied to the interpolated bathymetry.
- **Manning friction coefficient** (`manning_bkg`): the background bottom friction.

For each parameter, a Latin Hypercube Sampling (LHS) ensemble of simulations is run. Post-processing extracts the **mean tidal range** and **theoretical tidal energy** at detector sites, and a **Gaussian Process surrogate** with a Matérn kernel is fitted to relate the uncertain input to the model output.

---

## Project Structure

```
.
├── inputs/
│   ├── simulation_parameters.py   # Central configuration file
│   ├── bathymetry_error_LHS.py    # LHS generator for bathymetric error
│   ├── manning_LHS.py             # LHS generator for Manning coefficient
│   └── ...                        # Mesh, gauge, detector, and forcing data files
│
├── modules/
│   └── functions.py               # Tidal analysis utilities (peaks, ranges, energy)
│
├── tools/
│   ├── bathymetry.py              # Bathymetry interpolation and smoothing
│   ├── tidal_forcing.py           # TPXO tidal boundary condition updater
│   ├── tidal_amplitude.py         # Lowest Astronomical Tide (LAT) computation
│   ├── thetis_support_scripts.py  # Coriolis and helper utilities
│   ├── detectors.py               # Tide gauge / detector loading
│   ├── field_tools.py             # Eikonal solver, transition fields, Gaussian humps
│   └── utm.py                     # UTM coordinate utilities
│
├── outputs/                       # Generated at runtime (gitignored)
│
├── preprocessing.py               # Phase 4: mesh setup, bathymetry, friction, viscosity
├── ramp.py                        # Phase 5: spin-up simulation (2 days)
├── run.py                         # Phase 6: main tidal simulation (15 days)
├── run_bathymetry_ensemble.sh     # Phase 7a: ensemble driver for bathymetric UQ
├── run_manning_ensemble.sh        # Phase 7b: ensemble driver for Manning UQ
├── calculate_tidal_range_and_energy.py  # Phase 8: scalar output extraction
├── GP_multiple.py                 # Phase 9: GP surrogate fitting and plotting
└── .gitignore
```

---

## Dependencies

The model requires a working [Firedrake](https://www.firedrakeproject.org/) installation with Thetis. Additional Python packages are needed for post-processing.

**Core (inside Firedrake venv):**
- [Firedrake](https://www.firedrakeproject.org/) — FEM framework
- [Thetis](https://thetisproject.org/) — 2D/3D ocean model built on Firedrake
- [uptide](https://github.com/stephankramer/uptide) — tidal harmonic analysis

**Post-processing:**
```
pip install h5py numpy scipy matplotlib scikit-learn pandas
```

All scripts assume the Firedrake virtual environment is activated:
```bash
source ~/firedrake/bin/activate
```

---

## Configuration

All parameters are centralised in `inputs/simulation_parameters.py`. Key settings:

| Parameter | Description | Default |
|---|---|---|
| `mesh_file` | Path to the `.msh` mesh file | `inputs/west_uk_mesh.msh` |
| `bath_error` | Uniform bathymetric perturbation (m) | `0.63` |
| `manning_bkg` | Background Manning coefficient | `0.024` |
| `i_dt` | Crank-Nicolson timestep (s) | `100` |
| `i_ramptime` | Spin-up duration (s) | `2 × 86400` |
| `i_t_end` | Main run duration (s) | `15 × 86400` |
| `i_lat_cor` | Latitude for Coriolis (°) | `51` |
| `open_bnd` | Open boundary tags | `[4, 5, 6]` |
| `incl_harmonic_analysis` | Export elevation fields for HA | `True` |

Paths to external data (bathymetry grids, TPXO forcing files, gauge CSV) are set via `model_data_dir`.

---

## Preprocessing

```bash
mpirun.mpich -np 1 python preprocessing.py
```

This script:
1. Loads the mesh from the `.msh` file.
2. Interpolates bathymetry from multiple sources (DigiMap 1 arc-sec + GEBCO), applying the configured `bath_error` perturbation and LAT correction. A `RuntimeError` is raised immediately if no bathymetry source produces a valid field.
3. Solves the Eikonal equation to compute distance fields from boundaries, used to construct the **horizontal viscosity sponge**.
4. Builds the **Manning friction field**, optionally reading bed-classification data or applying Gaussian patches (e.g. Cardigan Bay).
5. Saves all fields (mesh, bathymetry, viscosity, Manning) to `inputs/preprocessing.h5`.

---

## Ramp (Spin-up) Simulation

```bash
mpirun.mpich -np 6 python ramp.py
```

A 2-day spin-up run that starts from rest and ramps up tidal forcing. Uses the same Crank-Nicolson / DG-DG solver configuration as the main run. At completion it saves the flow state to `inputs/export_-1.h5`, which initialises the main simulation.

**Solver settings:** CrankNicolson θ=0.75, wetting-and-drying (α=1.5), MUMPS direct solver, MPI-parallel.

---

## Main Tidal Simulation

```bash
mpirun.mpich -np 6 python run.py
```

A 15-day tidal simulation initialised from the ramp state. Tidal boundary conditions are updated at every timestep from TPXO harmonic constituents via `tools/tidal_forcing.py`. The Gloucester river boundary is imposed as a fixed flux.

**Outputs (written to `outputs/outputs_run/`):**
- `diagnostic_detectors_TRS.hdf5` — time series of elevation and velocity at tidal range / energy detector sites.
- `diagnostic_detectors_gauges.hdf5` — time series at BODC tide gauge locations.
- Periodic elevation field snapshots (`elev_XXXXXXX`) for harmonic analysis.
- Final state checkpoint (`inputs/run_export`) for restart.

---

## Ensemble Runs (UQ)

Two ensemble drivers are provided, each reading LHS sample files and looping over simulations. Both scripts use `set -euo pipefail` so any failed pipeline step (preprocessing, ramp, or run) immediately aborts the ensemble with a non-zero exit code rather than silently continuing to the next sample.

### Bathymetric uncertainty
```bash
bash run_bathymetry_ensemble.sh
```
Reads `inputs/bath_samples_LHS.txt` (LHS samples of the uniform bathymetric perturbation). The script validates that this file exists before starting. For each sample:
1. Creates the per-sample output directory (`outputs/outputs_run/H=<value>/`).
2. Patches `bath_error` and `run_output_folder` in `simulation_parameters.py`.
3. Runs the full `preprocessing.py` → `ramp.py` → `run.py` pipeline (a new ramp is required for each sample since the bathymetry changes).
4. Logs the wall-clock time taken per sample.

Each ensemble member produces its own HDF5 diagnostic file, which is later read by `GP_multiple.py` in bathymetry mode to extract mean tidal range at all detector sites.

### Manning uncertainty
```bash
bash run_manning_ensemble.sh
```
Reads `inputs/manning_samples_LHS.txt` (LHS samples of the background Manning coefficient). The script validates that this file exists before starting. For each sample:
1. Patches `manning_bkg` in `simulation_parameters.py`.
2. Runs `preprocessing.py` → `run.py` (no new ramp — the existing ramp state is reused since bathymetry is unchanged).
3. Calls `calculate_tidal_range_and_energy.py` to extract mean tidal range and theoretical energy at the SW detector.
4. Appends `(manning, R_mean, E_mean)` to `manning_results.txt`.

The final `manning_results.txt` is a CSV with columns `Manning, R_mean, E_mean` and is the direct input to `GP_multiple.py` in Manning mode.

---

## Post-processing and GP Surrogate

### Scalar extraction
```bash
python calculate_tidal_range_and_energy.py
```
Reads `diagnostic_detectors_TRS.hdf5` and prints the mean tidal range and theoretical tidal energy at the SW detector (used by the ensemble scripts to capture results).

### GP surrogate fitting
```bash
python GP_multiple.py --mode bathymetry   # or --mode manning
```

`GP_multiple.py` builds a GP surrogate for either the bathymetric or Manning ensemble. The mode is selected via the `--mode` command-line argument (default: `bathymetry`).

---

**Bathymetry mode (`--mode bathymetry`)**

For each detector site (SW, CA, WA, CO, LI, BL, SO, Outer Severn Barrage), the script loops over all bathymetric error samples and reads `diagnostic_detectors_TRS.hdf5` from the corresponding output folder (`outputs/outputs_run/H=<value>/`). Any missing HDF5 file produces a warning and skips that detector rather than aborting. The elevation time series at the detector is extracted and passed to `functions.mean_tidal_range_and_theoretical_energy()`, which identifies HW and LW peaks, computes tidal ranges between consecutive pairs, and returns the mean over the full 15-day record. A separate GP is fitted to the `(bath_error, R_mean)` pairs for each detector. The baseline test point is set at `bath_error = 0`, with the corresponding R_mean interpolated from the ensemble data. All detectors are overlaid on the same plot, allowing direct comparison of spatial sensitivity across the domain.

---

**Manning mode (`--mode manning`)**

Reads `manning_results.txt` — the CSV produced by `run_manning_ensemble.sh` with columns `Manning, R_mean, E_mean`. The script exits with an error if the file is not found. It fits a GP to the `(manning, R_mean)` pairs for the SW detector. The baseline test point is set to the default Manning coefficient (`n = 0.024`), with the corresponding R_mean interpolated from the ensemble data. The GP is evaluated over the full range of sampled Manning values and the plot shows how mean tidal range at SW responds to friction uncertainty.

---

**GP regression (shared)**

Both modes use a `GaussianProcessRegressor` with a **Matérn kernel** (length scale = 1.5, ν = 2.5). The ν=2.5 Matérn is twice differentiable, appropriate for smooth physical responses. The surrogate is evaluated on a dense grid of 100 points spanning the input range, and a held-out test point is used to report test MSE.

**Output (both modes)**

Each plot shows:
- Scatter points of the ensemble simulation results.
- The GP mean prediction curve over the full input range.
- A shaded ±1σ uncertainty band (light steel blue) reflecting GP posterior variance.
- A dashed vertical line marking the unperturbed baseline.


