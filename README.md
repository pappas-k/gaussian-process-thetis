# Gaussian Process Surrogate for Thetis West UK Tidal Model

A 2D shallow-water tidal model of the West UK (Bristol Channel and Irish Sea) built on [Thetis](https://thetisproject.org/) / [Firedrake](https://www.firedrakeproject.org/), coupled with a Gaussian Process (GP) surrogate for uncertainty quantification (UQ) of bathymetric and friction parameters.

---

## Phase 1 — Overview

This repository implements a workflow for assessing the sensitivity of tidal dynamics in the West UK to two key uncertain parameters:

- **Bathymetric error** (`bath_error`): a uniform perturbation applied to the interpolated bathymetry.
- **Manning friction coefficient** (`manning_bkg`): the background bottom friction.

For each parameter, a Latin Hypercube Sampling (LHS) ensemble of simulations is run. Post-processing extracts the **mean tidal range** and **theoretical tidal energy** at detector sites, and a **Gaussian Process surrogate** with a Matérn kernel is fitted to relate the uncertain input to the model output.

---

## Phase 2 — Project Structure

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

## Phase 3 — Dependencies

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

## Phase 4 — Configuration

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

## Phase 5 — Preprocessing

```bash
mpirun.mpich -np 1 python preprocessing.py
```

This script:
1. Loads the mesh from the `.msh` file.
2. Interpolates bathymetry from multiple sources (DigiMap 1 arc-sec + GEBCO), applying the configured `bath_error` perturbation and LAT correction.
3. Solves the Eikonal equation to compute distance fields from boundaries, used to construct the **horizontal viscosity sponge**.
4. Builds the **Manning friction field**, optionally reading bed-classification data or applying Gaussian patches (e.g. Cardigan Bay).
5. Saves all fields (mesh, bathymetry, viscosity, Manning) to `inputs/preprocessing.h5`.

---

## Phase 6 — Ramp (Spin-up) Simulation

```bash
mpirun.mpich -np 6 python ramp.py
```

A 2-day spin-up run that starts from rest and ramps up tidal forcing. Uses the same Crank-Nicolson / DG-DG solver configuration as the main run. At completion it saves the flow state to `inputs/export_-1.h5`, which initialises the main simulation.

**Solver settings:** CrankNicolson θ=0.75, wetting-and-drying (α=1.5), MUMPS direct solver, MPI-parallel.

---

## Phase 7 — Main Tidal Simulation

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

## Phase 8 — Ensemble Runs (UQ)

Two ensemble drivers are provided, each reading LHS sample files and looping over simulations.

### Bathymetric uncertainty
```bash
bash run_bathymetry_ensemble.sh
```
Reads `inputs/bath_samples_LHS.txt`. For each sample, patches `simulation_parameters.py` and runs the full preprocessing → ramp → run pipeline. Output goes to `outputs/outputs_run/H=<value>/`.

### Manning uncertainty
```bash
bash run_manning_ensemble.sh
```
Reads `inputs/manning_samples_LHS.txt`. For each sample, runs preprocessing → run (no new ramp) and appends `(manning, R_mean, E_mean)` to `manning_results.txt`.

---

## Phase 9 — Post-processing and GP Surrogate

### Scalar extraction
```bash
python calculate_tidal_range_and_energy.py
```
Reads `diagnostic_detectors_TRS.hdf5` and prints the mean tidal range and theoretical tidal energy at the SW detector (used by the ensemble scripts to capture results).

### GP surrogate fitting
```bash
python GP_multiple.py
```
For each detector site:
1. Loads ensemble HDF5 outputs and computes mean tidal range per sample.
2. Fits a **Gaussian Process** (Matérn ν=2.5 kernel, sklearn) to the (bath_error, R_mean) pairs.
3. Plots the GP mean prediction with ±1σ uncertainty band against the scatter of ensemble samples.


