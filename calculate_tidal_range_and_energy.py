"""
Post-processing script: calculates mean tidal range and theoretical energy
at the SW detector from the model diagnostic output.

Prints: R_mean E_mean  (used by the ensemble shell scripts to capture results)
"""
import os
import sys

import h5py
import numpy as np
from modules import functions

DEFAULT_DIAGNOSTIC_FILE = 'outputs/outputs_run/diagnostic_detectors_TRS.hdf5'
DEFAULT_DETECTOR        = 'SW'


def load_signal(diagnostic_file, detector):
    if not os.path.isfile(diagnostic_file):
        print(f"ERROR: diagnostic file not found: {diagnostic_file}", file=sys.stderr)
        sys.exit(1)
    with h5py.File(diagnostic_file, 'r') as df:
        t    = df['time'][:]
        elev = df[detector][:, 0]
    return np.column_stack((t, elev))


signal = load_signal(DEFAULT_DIAGNOSTIC_FILE, DEFAULT_DETECTOR)

R_mean, E_mean = functions.mean_tidal_range_and_theoretical_energy(signal)

print(f"{R_mean:.3f} {E_mean:.3f}")
