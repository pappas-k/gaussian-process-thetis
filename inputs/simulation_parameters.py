"""
Central configuration file for the West UK tidal UQ study.
All simulation parameters are defined here and imported by the other scripts.
"""
import numpy as np

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
model_data_dir = '../../Dropbox/0_PhD_projects/model_data'

mesh_file = "inputs/west_uk_mesh.msh"

# Bathymetry files — listed highest to lowest resolution
bathymetry_file_0 = f"{model_data_dir}/digimap_West_UK_1_arc_seconds.nc"
bathymetry_file_1 = f"{model_data_dir}/GEBCO_West_UK.nc"
bathymetries = [
    (bathymetry_file_0, 'Band1',    'LAT'),
    (bathymetry_file_1, 'elevation', 'MWL'),
]

# Tidal forcing (TPXO)
grid_forcing_file    = f'{model_data_dir}/gridES2008.nc'
hf_forcing_file      = f'{model_data_dir}/hf.ES2008.nc'
range_forcing_coords = ((-12., -2.), (48., 59.))

# Detector / gauge files
i_tidegauge_file        = 'inputs/useful_gauges_BODC.csv'
additional_detector_files = ['inputs/extra_detectors_TRS']
max_dist                = 5e3   # maximum distance to snap detector to mesh (m)

# Friction
friction_data      = "inputs/n_max_125.npy"
use_friction_data  = False      # True: use bed-class data, False: use uniform manning_bkg
bed_classification_file = f'{model_data_dir}/BGS_data/bed_class_pentland_rev.nc'

# Output folders
ramp_output_folder = 'outputs/outputs_ramp'
run_output_folder  = 'outputs/outputs_run/H=0.63'   # updated by ensemble scripts

# ---------------------------------------------------------------------------
# UTM projection parameters (Irish Sea: zone 30U)
# ---------------------------------------------------------------------------
i_zone = 30
i_band = 'U'

# ---------------------------------------------------------------------------
# Simulation start time (used by TPXO tidal forcing)
# ---------------------------------------------------------------------------
s_year, s_month, s_day, s_hour, s_min = 2002, 10, 20, 0, 0

# ---------------------------------------------------------------------------
# Preprocessing / Eikonal parameters
# ---------------------------------------------------------------------------
i_L    = 1e3          # characteristic length for Eikonal equation (m)
i_epss = [100000., 10000., 5000., 2500., 1500., 1000.]  # solver accuracy levels (m)

# Boundary tags (from mesh)
open_bnd = [4, 5, 6]
land_bnd = 2

# ---------------------------------------------------------------------------
# Bathymetry parameters
# ---------------------------------------------------------------------------
i_min_depth = -10.0   # minimum allowed depth (m, negative = above datum)
add_amps    = False   # if True, adjust bathymetry by adding M2+S2 amplitudes
bath_error  = 0.63    # uniform bathymetric perturbation (m) — varied in ensemble

# ---------------------------------------------------------------------------
# Friction parameters
# ---------------------------------------------------------------------------
manning_bkg  = 0.024   # background Manning coefficient — varied in ensemble
manning_peak = 0.06    # peak Manning coefficient (Cardigan Bay ridge)

# Gaussian Manning patches: (peak, (x0, y0, angle), (sd1, sd2), base, (r1, r2))
manning_gauss = [
    (manning_peak, (346_300, 5.83e6, 90), (np.inf, 0.027e6), manning_bkg, (78_600, None)),
]

# ---------------------------------------------------------------------------
# Timestepping and solver parameters
# ---------------------------------------------------------------------------
i_ramptime = 2 * 24 * 3600    # ramp duration (s) — 2 days
i_t_end    = 15 * 24 * 3600   # run duration (s)  — 15 days
i_dt       = 100               # Crank-Nicolson timestep (s)
i_alpha    = 1.5               # wetting-and-drying parameter
i_lat_cor  = 51                # latitude for Coriolis (degrees)

# ---------------------------------------------------------------------------
# Output / harmonic analysis
# ---------------------------------------------------------------------------
incl_harmonic_analysis = True
ramp_exp_interval      = 1000.   # ramp export interval (s)
run_exp_interval       = 10000.  # run export interval (s)
run_exp_elev_interval  = 500     # elevation field export interval for harmonic analysis (s)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
grav_acc = 9.807
density  = 1025

# Tidal constituents for harmonic analysis
i_constituents = ['M2', 'S2']
