"""
Post-processing script: calculates mean tidal range and theoretical energy
at the SW detector from the model diagnostic output.

Prints: R_mean E_mean  (used by the ensemble shell scripts to capture results)
"""
import h5py
import numpy as np
from modules import functions

DEFAULT_DIAGNOSTIC_FILE = 'outputs/outputs_run/diagnostic_detectors_TRS.hdf5'

with h5py.File(DEFAULT_DIAGNOSTIC_FILE, 'r') as df:
    t = df['time'][:]
    data = df['SW'][:, 0]

signal = np.column_stack((t, data))

R_mean, E_mean = functions.mean_tidal_range_and_theoretical_energy(signal)

print(f"{R_mean:.3f} {E_mean:.3f}")
