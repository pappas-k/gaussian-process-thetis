"""
Post-processing script: fits a Gaussian Process surrogate to ensemble results
and plots mean tidal range vs. the uncertain parameter.

Usage
-----
    python GP_multiple.py --mode bathymetry
    python GP_multiple.py --mode manning

Modes
-----
bathymetry
    Reads HDF5 diagnostics from outputs/outputs_run/H=<value>/ for each
    bathymetric error sample; supports multiple detector sites.
manning
    Reads manning_results.txt (CSV produced by run_manning_ensemble.sh)
    for each Manning coefficient sample; SW detector only.
"""
import argparse
import csv
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error
from modules import functions

plt.rc('font', family='serif', **{'serif': ['Helvetica'], 'size': 15})
plt.rc('text', usetex=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Bathymetry mode settings
BATH_DETECTOR_NAMES = ['SW', 'CA', 'WA', 'CO', 'LI', 'BL', 'SO', 'Outer_Severn_Barrage']
BATH_ERRORS = [
     0.04,  0.06,  0.36,  0.63,  0.81,  0.87,  1.28,  1.34,  1.56,  1.59,
     1.72,  2.02,  2.08,  2.11,  2.26,  2.34,  2.40,  2.48,  2.52,  2.59,  2.87,
    -0.05, -0.39, -0.59, -1.17, -1.42, -1.69, -1.79, -1.81, -2.21, -2.23,
    -2.31, -2.35, -2.48, -2.57, -2.74, -2.77,
]

# Manning mode settings
MANNING_RESULTS_FILE = 'manning_results.txt'
MANNING_BASELINE     = 0.024   # default Manning coefficient (used as test point)


# ---------------------------------------------------------------------------
# Shared GP regression
# ---------------------------------------------------------------------------

def gp_regression(x_train, y_train, x_test, y_test, x_values):
    """
    Fit a GP with a Matérn-2.5 kernel and return mean and std predictions.

    Parameters
    ----------
    x_train : array, shape (n, 1)
    y_train : array, shape (n,)
    x_test  : array, shape (m, 1)  — held-out test points
    y_test  : array, shape (m,)    — true values at test points
    x_values : array, shape (k,)   — dense grid for prediction

    Returns
    -------
    mean_prediction : array, shape (k,)
    std_prediction  : array, shape (k,)
    """
    kernel = Matern(length_scale=1.5, nu=2.5)
    gp_model = GaussianProcessRegressor(kernel=kernel, alpha=0.0, random_state=0)
    gp_model.fit(x_train, y_train)

    mean_prediction, std_prediction = gp_model.predict(x_values.reshape(-1, 1), return_std=True)
    y_test_pred, _ = gp_model.predict(x_test, return_std=True)

    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"Mean Squared Error (Test Data): {mse_test:.4f}")

    return mean_prediction, std_prediction


# ---------------------------------------------------------------------------
# Bathymetry mode
# ---------------------------------------------------------------------------

def run_bathymetry_mode(detector_names, bath_errors):
    """
    For each detector, load HDF5 ensemble outputs, compute mean tidal range,
    fit a GP surrogate, and plot results.
    """
    bath_errors_fmt = [f"{e:.2f}" for e in bath_errors]
    x_values = np.linspace(-3, 3, num=100)
    bath_error_values = np.array(bath_errors, dtype=float).reshape(-1, 1)

    # Baseline test point: no bathymetric error; reference range interpolated from ensemble
    x_test = np.array([[0.0]])

    for detector_name in detector_names:
        R_means = []
        for bath_error in bath_errors_fmt:
            diagnostic_file = f'outputs/outputs_run/H={bath_error}/diagnostic_detectors_TRS.hdf5'
            try:
                with h5py.File(diagnostic_file, 'r') as df:
                    signal = np.column_stack((df['time'][:], df[detector_name][:, 0]))
            except FileNotFoundError:
                print(f"WARNING: diagnostic file not found: {diagnostic_file}", file=sys.stderr)
                continue
            R_mean, _ = functions.mean_tidal_range_and_theoretical_energy(signal)
            R_means.append(R_mean)

        if len(R_means) != len(bath_errors_fmt):
            print(f"WARNING: skipping {detector_name} — missing ensemble members.", file=sys.stderr)
            continue

        # Interpolate baseline tidal range from ensemble rather than using a fixed guess
        sorted_idx = np.argsort(bath_errors)
        y_test = np.array([np.interp(0.0,
                                     np.array(bath_errors)[sorted_idx],
                                     np.array(R_means)[sorted_idx])])

        mean_prediction, std_prediction = gp_regression(
            bath_error_values, R_means, x_test, y_test, x_values
        )

        plt.scatter(bath_errors, R_means, marker='o', s=40,
                    label=detector_name.replace('_', ' '),
                    edgecolors='black', zorder=2)
        plt.plot(x_values, mean_prediction, '-', zorder=1)
        plt.fill_between(x_values,
                         mean_prediction - std_prediction,
                         mean_prediction + std_prediction,
                         alpha=0.6, color='lightsteelblue')

        print(f"Bathymetric errors: {bath_errors}")
        print(f"Mean Tidal Ranges for {detector_name}: {R_means}")

    plt.axvline(x=0.0, color='navy', linestyle='--', label=r'Baseline $\Delta h=0$')
    plt.xlabel(r'Bathymetric error $\Delta h$ (m)')
    plt.ylabel(r'Mean Tidal Range $\overline{R}$ (m)')
    plt.title('Mean Tidal Range vs. Bathymetric Uncertainty')
    plt.legend(ncol=4, fontsize=10)
    plt.ylim(5, 10.8)
    plt.show()


# ---------------------------------------------------------------------------
# Manning mode
# ---------------------------------------------------------------------------

def run_manning_mode(results_file, baseline):
    """
    Load Manning ensemble results from CSV, fit a GP surrogate, and plot.

    Expects a CSV with columns: Manning, R_mean, E_mean
    (as produced by run_manning_ensemble.sh).
    """
    if not __import__('os').path.isfile(results_file):
        print(f"ERROR: results file not found: {results_file}", file=sys.stderr)
        sys.exit(1)

    manning_vals, R_means = [], []
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            manning_vals.append(float(row['Manning']))
            R_means.append(float(row['R_mean']))

    manning_vals = np.array(manning_vals)
    R_means      = np.array(R_means)

    x_test   = np.array([[baseline]])
    y_test   = np.array([np.interp(baseline, np.sort(manning_vals),
                                   R_means[np.argsort(manning_vals)])])
    x_values = np.linspace(manning_vals.min(), manning_vals.max(), num=100)

    mean_prediction, std_prediction = gp_regression(
        manning_vals.reshape(-1, 1), R_means, x_test, y_test, x_values
    )

    plt.scatter(manning_vals, R_means, marker='o', s=40,
                label='SW', edgecolors='black', zorder=2)
    plt.plot(x_values, mean_prediction, '-', zorder=1)
    plt.fill_between(x_values,
                     mean_prediction - std_prediction,
                     mean_prediction + std_prediction,
                     alpha=0.6, color='lightsteelblue')

    plt.axvline(x=baseline, color='navy', linestyle='--',
                label=rf'Baseline $n={baseline}$')
    plt.xlabel(r'Manning coefficient $n$')
    plt.ylabel(r'Mean Tidal Range $\overline{R}$ (m)')
    plt.title('Mean Tidal Range vs. Manning Uncertainty')
    plt.legend(fontsize=10)
    plt.show()

    print(f"Manning values: {manning_vals.tolist()}")
    print(f"Mean Tidal Ranges (SW): {R_means.tolist()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a GP surrogate to tidal ensemble results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--mode', choices=['bathymetry', 'manning'], default='bathymetry',
        help="Ensemble type to analyse (default: bathymetry).",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'bathymetry':
        run_bathymetry_mode(BATH_DETECTOR_NAMES, BATH_ERRORS)
    else:
        run_manning_mode(MANNING_RESULTS_FILE, MANNING_BASELINE)
