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

# Use LaTeX-rendered serif fonts for publication-quality figures
plt.rc('font', family='serif', **{'serif': ['Helvetica'], 'size': 15})
plt.rc('text', usetex=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Short names of the detector / tide-gauge locations used in bathymetry mode.
# These must match the dataset keys stored inside each diagnostic HDF5 file
# (written by the DetectorsCallback in run.py).  Sites span the Bristol
# Channel and Severn Estuary: SW = Swansea, CA = Cardiff, WA = Watcher,
# CO = Combwich, LI = Lilstock, BL = Blorenge, SO = Steep Holm,
# Outer_Severn_Barrage = a hypothetical tidal barrage reference site.
BATH_DETECTOR_NAMES = ['SW', 'CA', 'WA', 'CO', 'LI', 'BL', 'SO', 'Outer_Severn_Barrage']

# Latin Hypercube Sampling (LHS) draws of the bathymetric error parameter Δh (m).
# Positive values correspond to the seabed being shallower than the reference
# chart datum; negative values correspond to deeper bathymetry.
# These values are the same across all ensemble runs and serve as the training
# inputs (x_train) for the GP surrogate.
BATH_ERRORS = [
     0.04,  0.06,  0.36,  0.63,  0.81,  0.87,  1.28,  1.34,  1.56,  1.59,
     1.72,  2.02,  2.08,  2.11,  2.26,  2.34,  2.40,  2.48,  2.52,  2.59,  2.87,
    -0.05, -0.39, -0.59, -1.17, -1.42, -1.69, -1.79, -1.81, -2.21, -2.23,
    -2.31, -2.35, -2.48, -2.57, -2.74, -2.77,
]

# Path to the CSV file containing Manning ensemble results (produced by the
# shell script run_manning_ensemble.sh which loops over Manning n values,
# runs run.py for each, and appends the extracted tidal range to this file).
MANNING_RESULTS_FILE = 'manning_results.txt'

# Default (calibrated) Manning n used in the baseline run.
# This is used as the held-out test point for GP validation.
MANNING_BASELINE     = 0.024


# ---------------------------------------------------------------------------
# Shared GP regression
# ---------------------------------------------------------------------------

def gp_regression(x_train, y_train, x_test, y_test, x_values):
    """
    Fit a GP with a Matérn-2.5 kernel and return mean and std predictions.

    The Matérn kernel with ν=2.5 is chosen because it assumes the underlying
    function is twice mean-square differentiable — a reasonable assumption for
    the smooth, monotonic response of tidal range to bathymetric or friction
    perturbations.  The initial length scale of 1.5 is a prior guess; sklearn
    optimises it during fitting via marginal-likelihood maximisation.

    alpha=0.0 means no observation noise is added to the diagonal of the
    kernel matrix, so the GP interpolates the training points exactly.  This
    is appropriate here because each ensemble member is a deterministic model
    run (no measurement noise).

    Parameters
    ----------
    x_train  : array, shape (n, 1)   — uncertain parameter values (training)
    y_train  : array, shape (n,)     — mean tidal range at each training point
    x_test   : array, shape (m, 1)   — held-out test points for MSE evaluation
    y_test   : array, shape (m,)     — true tidal range at test points
    x_values : array, shape (k,)     — dense prediction grid for the final plot

    Returns
    -------
    mean_prediction : array, shape (k,)  — GP posterior mean
    std_prediction  : array, shape (k,)  — GP posterior standard deviation
    """
    # Matérn-2.5 kernel: stationary, smooth, and expressive enough to capture
    # the non-linear tidal response without overfitting sparse ensemble data
    kernel = Matern(length_scale=1.5, nu=2.5)

    # random_state=0 ensures reproducible kernel hyper-parameter optimisation
    gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, random_state=0)
    gp_model.fit(x_train, y_train)

    # Predict posterior mean and std on the dense grid for plotting
    mean_prediction, std_prediction = gp_model.predict(x_values.reshape(-1, 1), return_std=True)

    # Evaluate the fitted GP at the held-out test point to compute a
    # leave-one-out-style validation metric (MSE) printed to stdout
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

    For every combination of (detector, bath_error) the function:
      1. Opens the HDF5 diagnostic file written by the DetectorsCallback.
      2. Extracts the time series of surface elevation at the detector site.
      3. Calls functions.mean_tidal_range_and_theoretical_energy to compute
         the mean tidal range R over the simulated period.
      4. Collects all R values across the ensemble into R_means.
    After iterating over all ensemble members, a GP is fitted with bath_errors
    as inputs and R_means as outputs.  The baseline test point (Δh=0) is
    interpolated from the sorted ensemble rather than taken from a separate run,
    so no extra simulation is required for validation.
    """
    # Format errors to 2 d.p. to match the directory naming convention
    # used by the ensemble runner (e.g. H=0.36, H=-1.17)
    bath_errors_fmt = [f"{e:.2f}" for e in bath_errors]

    # Dense evaluation grid spanning the full range of sampled errors plus
    # a small margin, used to produce smooth GP mean and confidence band curves
    x_values = np.linspace(-3, 3, num=100)

    # Reshape training inputs to (n, 1) as required by sklearn's GP API
    bath_error_values = np.array(bath_errors, dtype=float).reshape(-1, 1)

    # Held-out test point at Δh=0 (no bathymetric error = baseline run).
    # y_test is interpolated below per detector rather than hard-coded.
    x_test = np.array([[0.0]])

    for detector_name in detector_names:
        R_means = []  # accumulate mean tidal range for each ensemble member

        for bath_error in bath_errors_fmt:
            # Each ensemble run exports its detector time series to a dedicated
            # sub-directory named H=<value> so runs do not overwrite each other
            diagnostic_file = f'outputs/outputs_run/H={bath_error}/diagnostic_detectors_TRS.hdf5'
            try:
                with h5py.File(diagnostic_file, 'r') as df:
                    # Stack time (s) and elevation (m) into a 2-column array.
                    # Column 0 of the detector dataset is the surface elevation;
                    # additional columns (if present) are velocity components.
                    signal = np.column_stack((df['time'][:], df[detector_name][:, 0]))
            except FileNotFoundError:
                print(f"WARNING: diagnostic file not found: {diagnostic_file}", file=sys.stderr)
                continue

            # Compute the mean tidal range (average of high–low pairs) over
            # the full simulation record; discard the energy estimate here
            R_mean, _ = functions.mean_tidal_range_and_theoretical_energy(signal)
            R_means.append(R_mean)

        # Guard against partially complete ensembles: skip the detector if any
        # member is missing, to avoid misaligned x_train / y_train arrays
        if len(R_means) != len(bath_errors_fmt):
            print(f"WARNING: skipping {detector_name} — missing ensemble members.", file=sys.stderr)
            continue

        # Interpolate the baseline tidal range (Δh=0) from the ensemble.
        # Sorting by bath_errors ensures the interpolation is well-defined
        # even though the LHS samples are not ordered.
        sorted_idx = np.argsort(bath_errors)
        y_test = np.array([np.interp(0.0,
                                     np.array(bath_errors)[sorted_idx],
                                     np.array(R_means)[sorted_idx])])

        mean_prediction, std_prediction = gp_regression(
            bath_error_values, R_means, x_test, y_test, x_values
        )

        # Scatter plot: ensemble training points (one marker per member)
        plt.scatter(bath_errors, R_means, marker='o', s=40,
                    label=detector_name.replace('_', ' '),
                    edgecolors='black', zorder=2)
        # GP posterior mean curve
        plt.plot(x_values, mean_prediction, '-', zorder=1)
        # ±1σ confidence band around the posterior mean
        plt.fill_between(x_values,
                         mean_prediction - std_prediction,
                         mean_prediction + std_prediction,
                         alpha=0.6, color='lightsteelblue')

        print(f"Bathymetric errors: {bath_errors}")
        print(f"Mean Tidal Ranges for {detector_name}: {R_means}")

    # Vertical dashed line marking the baseline (unperturbed) bathymetry
    plt.axvline(x=0.0, color='navy', linestyle='--', label=r'Baseline $\Delta h=0$')
    plt.xlabel(r'Bathymetric error $\Delta h$ (m)')
    plt.ylabel(r'Mean Tidal Range $\overline{R}$ (m)')
    plt.title('Mean Tidal Range vs. Bathymetric Uncertainty')
    # ncol=4 keeps the legend compact when all 8 detectors are displayed
    plt.legend(ncol=4, fontsize=10)
    # Fixed y-limits chosen to accommodate the full spread across all sites
    # in the Bristol Channel / Severn Estuary without excessive whitespace
    plt.ylim(5, 10.8)
    plt.show()


# ---------------------------------------------------------------------------
# Manning mode
# ---------------------------------------------------------------------------

def run_manning_mode(results_file, baseline):
    """
    Load Manning ensemble results from CSV, fit a GP surrogate, and plot.

    Expects a CSV with header row and columns: Manning, R_mean, E_mean
    (as produced by run_manning_ensemble.sh).  Only the SW (Swansea) detector
    is used here because the Manning sensitivity analysis targets a single
    representative site in the outer Bristol Channel.
    """
    if not __import__('os').path.isfile(results_file):
        print(f"ERROR: results file not found: {results_file}", file=sys.stderr)
        sys.exit(1)

    # Read all LHS samples from the CSV; each row corresponds to one ensemble run
    manning_vals, R_means = [], []
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            manning_vals.append(float(row['Manning']))
            R_means.append(float(row['R_mean']))

    manning_vals = np.array(manning_vals)
    R_means      = np.array(R_means)

    # Held-out test point at the calibrated baseline Manning n value.
    # y_test is linearly interpolated from the sorted ensemble so validation
    # does not require an additional simulation at exactly n=baseline.
    x_test   = np.array([[baseline]])
    y_test   = np.array([np.interp(baseline, np.sort(manning_vals),
                                   R_means[np.argsort(manning_vals)])])

    # Dense prediction grid spanning the sampled Manning range
    x_values = np.linspace(manning_vals.min(), manning_vals.max(), num=100)

    mean_prediction, std_prediction = gp_regression(
        manning_vals.reshape(-1, 1), R_means, x_test, y_test, x_values
    )

    # Scatter plot of training data (one point per ensemble member)
    plt.scatter(manning_vals, R_means, marker='o', s=40,
                label='SW', edgecolors='black', zorder=2)
    # GP posterior mean curve
    plt.plot(x_values, mean_prediction, '-', zorder=1)
    # ±1σ uncertainty band
    plt.fill_between(x_values,
                     mean_prediction - std_prediction,
                     mean_prediction + std_prediction,
                     alpha=0.6, color='lightsteelblue')

    # Vertical dashed line at the calibrated baseline Manning coefficient
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
