"""
Post-processing script: fits a Gaussian Process surrogate to the ensemble results
and plots mean tidal range vs. bathymetric error for all detector sites.

Reads diagnostic HDF5 files from outputs/outputs_run/H=<value>/ for each
bathymetric error sample, extracts mean tidal range, fits a GP (Matérn kernel),
and plots the mean prediction with uncertainty band.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error
from modules import functions

plt.rc('font', family='serif', **{'serif': ['Helvetica'], 'size': 15})
plt.rc('text', usetex=True)


def gp_regression(x_train, y_train, X_test, y_test, x_values):
    """Fit a GP with a Matérn kernel and return mean and std predictions."""
    kernel = Matern(length_scale=1.5, nu=2.5)
    gp_model = GaussianProcessRegressor(kernel=kernel, alpha=0.0, random_state=0)
    gp_model.fit(x_train, y_train)

    mean_prediction, std_prediction = gp_model.predict(x_values.reshape(-1, 1), return_std=True)
    y_test_pred, _ = gp_model.predict(X_test, return_std=True)

    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"Mean Squared Error (Test Data): {mse_test:.4f}")

    return mean_prediction, std_prediction


def calculate_and_plot_mean_tidal_range_energy(detector_names, bath_errors):
    """
    For each detector, load ensemble outputs, compute mean tidal range,
    fit a GP surrogate, and plot results.
    """
    bath_errors_fmt = [f"{e:.2f}" for e in bath_errors]

    x_test = np.array([[0.0]])     # baseline: no bathymetric error
    y_test = np.array([8.5])       # approximate baseline tidal range
    x_values = np.linspace(-3, 3, num=100)
    bath_error_values = np.array([float(e) for e in bath_errors]).reshape(-1, 1)

    for detector_name in detector_names:
        R_means = []
        for bath_error in bath_errors_fmt:
            diagnostic_file = f'outputs/outputs_run/H={bath_error}/diagnostic_detectors_TRS.hdf5'
            df = h5py.File(diagnostic_file, 'r')
            signal = np.column_stack((df['time'][:], df[detector_name][:, 0]))
            R_mean, _ = functions.mean_tidal_range_and_theoretical_energy(signal)
            R_means.append(R_mean)

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

    plt.axvline(x=0.0, color='navy', linestyle='--', label=r'Baseline $\Delta h=0$')
    plt.xlabel(r'Bathymetric error $\Delta h$ (m)')
    plt.ylabel(r'Mean Tidal Range $\overline{R}$ (m)')
    plt.title('Mean Tidal Range vs. Bathymetric Uncertainty')
    plt.legend(ncol=4, fontsize=10)
    plt.ylim(5, 10.8)
    plt.show()

    print("Bathymetric errors:", bath_errors)
    for detector_name in detector_names:
        print(f"Mean Tidal Ranges for {detector_name}:", R_means)


detector_names = ['SW', 'CA', 'WA', 'CO', 'LI', 'BL', 'SO', 'Outer_Severn_Barrage']

bath_errors = [
     0.04,  0.06,  0.36,  0.63,  0.81,  0.87,  1.28,  1.34,  1.56,  1.59,
     1.72,  2.02,  2.08,  2.11,  2.26,  2.34,  2.40,  2.48,  2.52,  2.59,  2.87,
    -0.05, -0.39, -0.59, -1.17, -1.42, -1.69, -1.79, -1.81, -2.21, -2.23,
    -2.31, -2.35, -2.48, -2.57, -2.74, -2.77,
]

calculate_and_plot_mean_tidal_range_energy(detector_names, bath_errors)
