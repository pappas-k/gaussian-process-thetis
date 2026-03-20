import uptide
import numpy as np
import datetime
import pandas as pd


def extract_signal_from_recordings(tidegauge_file, start_date, end_date, plot=False):
    """
    Extracts tidal elevation signal from recordings within a specified date range.

    Parameters
    ----------
    tidegauge_file : str
        Path to the CSV file containing the tidal recordings.
    start_date : datetime.datetime
        Start date of the desired date range.
    end_date : datetime.datetime
        End date of the desired date range.
    plot : bool, optional
        Whether to plot the extracted signal. Default is False.

    Returns
    -------
    numpy.ndarray
        Array containing the extracted tidal signal, with columns [time, elevation].
    """
    date = np.loadtxt(tidegauge_file, skiprows=2, usecols=(9,), dtype=str, delimiter=',')
    date = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date]
    eta = np.loadtxt(tidegauge_file, skiprows=2, usecols=(11,), dtype=float, delimiter=',')
    QC_flag = np.loadtxt(tidegauge_file, skiprows=2, usecols=(12,), dtype=str, delimiter=',')

    pd0 = pd.DataFrame({"Time": date, "Elevation": eta, "QC flag": QC_flag})
    pd0_filtered = pd0[
        (~pd0['QC flag'].str.contains('N')) &
        (~pd0['QC flag'].str.contains('M')) &
        (~pd0['QC flag'].str.contains('T'))
    ]

    date = pd.to_datetime(pd0_filtered['Time']).tolist()
    eta = pd0_filtered['Elevation'].values
    eta = eta - np.mean(eta)

    start_index = np.where(np.array(date) >= start_date)[0][0]
    end_index = np.where(np.array(date) <= end_date)[0][-1]

    subset_date = date[start_index:end_index + 1]
    subset_eta = eta[start_index:end_index + 1]

    time_diff = [(d - subset_date[0]).total_seconds() for d in subset_date]

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(time_diff, subset_eta)
        plt.xlabel('Time')
        plt.ylabel('Elevation')
        plt.title('Tidal Elevation Data')
        plt.xticks(rotation=45)
        plt.show()

    return np.column_stack((time_diff, subset_eta))


def extract_constituents_from_tidegauge_file(tidegauge_file='inputs/Mumbles_20010101_20010131.csv',
                                              start_date=datetime.datetime(2001, 1, 1, 0, 0, 0)):
    """
    Performs harmonic analysis on a tidal gauge CSV file and returns constituent amplitudes/phases.

    Parameters
    ----------
    tidegauge_file : str
        Path to the BODC-format CSV file.
    start_date : datetime.datetime
        Reference start time for harmonic analysis.

    Returns
    -------
    pd.DataFrame, numpy.ndarray, numpy.ndarray
        DataFrame of constituents sorted by amplitude, time array (s), elevation array (m).
    """
    with open(tidegauge_file) as f:
        numline = len(f.readlines())

    t = np.arange(0, (numline - 2) * 15 * 60, 15 * 60)
    eta = np.loadtxt(tidegauge_file, skiprows=2, usecols=(11,), dtype=float, delimiter=',')
    QC_flag = np.loadtxt(tidegauge_file, skiprows=2, usecols=(12,), dtype=str, delimiter=',')

    pd0 = pd.DataFrame({"Time": t, "Elevation": eta, "QC flag": QC_flag})
    pd0_filtered = pd0[
        (~pd0['QC flag'].str.contains('N')) &
        (~pd0['QC flag'].str.contains('M')) &
        (~pd0['QC flag'].str.contains('T'))
    ]
    pd0_filtered = pd0_filtered[pd0_filtered["Elevation"] > -15]
    pd0_filtered = pd0_filtered[pd0_filtered["Elevation"] < 15]

    t = pd0_filtered['Time'].values
    eta = pd0_filtered['Elevation'].values

    constituents = [
        'Q1', 'O1', 'P1', 'S1', 'K1', 'J1', 'M1',
        '2N2', 'MU2', 'N2', 'NU2', 'M2', 'L2', 'T2', 'S2', 'K2',
        'LAMBDA2', 'EPS2', 'R2', 'ETA2', 'MSN2', 'MNS2', '3M2S2', '2SM2', 'MKS2',
        'MK3', 'MO3', 'MS4', 'MN4', 'N4', 'M4', 'S4', '2MK6', '2MS6',
        'M3', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12',
    ]
    tide = uptide.Tides(constituents)
    tide.set_initial_time(start_date)
    thetis_amplitudes, thetis_phases = uptide.analysis.harmonic_analysis(tide, eta, t)

    pd1 = pd.DataFrame({
        "Constituents": constituents,
        "Amplitude": thetis_amplitudes,
        "Phase": thetis_phases,
    })
    pd1 = pd1.sort_values(by=['Amplitude'], ignore_index=True, ascending=False)
    print(pd1)
    print('Minimum Rayleigh Period =', tide.get_minimum_Rayleigh_period() / 86400.)

    eta = eta - np.mean(eta)
    return pd1, t, eta


def signal_reconstruction(amplitudes, phases,
                           dt=108,
                           constituents=("M2", "S2"),
                           signal_duration=365.25 * 24 * 3600,
                           start_date=datetime.datetime(2002, 1, 1, 0, 0, 0),
                           time_series_start_time=0):
    """
    Reconstructs a tidal signal from harmonic constituents.

    Parameters
    ----------
    amplitudes : array-like
        Amplitudes of the constituents.
    phases : array-like
        Phases of the constituents (radians).
    dt : float, optional
        Time step in seconds. Default is 108 s.
    constituents : sequence of str, optional
        Tidal constituent names. Default is ('M2', 'S2').
    signal_duration : float, optional
        Duration of the reconstructed signal in seconds. Default is 365.25 days.
    start_date : datetime.datetime, optional
        Reference epoch for reconstruction.
    time_series_start_time : float, optional
        Start time offset in seconds. Default is 0.

    Returns
    -------
    numpy.ndarray
        Array with columns [time (s), elevation (m)].
    """
    time_series = np.arange(time_series_start_time, signal_duration, dt)
    amplitudes = amplitudes[:len(constituents)]
    phases = phases[:len(constituents)]

    tide = uptide.Tides(list(constituents))
    tide.set_initial_time(start_date)
    tide_elevs = tide.from_amplitude_phase(amplitudes, phases, time_series)
    return np.column_stack((time_series, tide_elevs))


def find_tidal_peaks(rel_times, tide_elevs, peak_type):
    """
    Finds HW or LW peaks in a tidal elevation signal.

    Parameters
    ----------
    rel_times : numpy.ndarray
        Time array in seconds.
    tide_elevs : numpy.ndarray
        Elevation array in metres.
    peak_type : str
        'HW' for high water, 'LW' for low water.

    Returns
    -------
    peak_rel_times : numpy.ndarray
    peak_elevs : numpy.ndarray
    """
    from scipy import signal
    multiplier = -1 if peak_type == 'LW' else 1
    peak_idx = signal.find_peaks(tide_elevs * multiplier)[0]
    return rel_times[peak_idx], tide_elevs[peak_idx]


def tidal_ranges_from_peaks(peak_real_times_HW, peak_real_times_LW,
                             peak_elevs_HW, peak_elevs_LW):
    """
    Computes tidal ranges from paired HW and LW peaks.

    Returns
    -------
    tidal_ranges_all : list of float
    rel_times_all : list of float
    """
    if peak_real_times_HW[0] < peak_real_times_LW[0]:
        # HW occurs first
        try:
            tidal_ranges_from_HW = peak_elevs_HW - peak_elevs_LW
            tidal_ranges_from_LW = abs(peak_elevs_LW[:-1] - peak_elevs_HW[1:])
        except (IndexError, ValueError):
            tidal_ranges_from_HW = peak_elevs_HW[:-1] - peak_elevs_LW
            tidal_ranges_from_LW = abs(peak_elevs_LW[:] - peak_elevs_HW[1:])
        tidal_ranges_all, rel_times_all = [], []
        for i in range(len(tidal_ranges_from_HW)):
            tidal_ranges_all.append(tidal_ranges_from_HW[i])
            rel_times_all.append(peak_real_times_HW[i])
            if i < len(tidal_ranges_from_LW):
                tidal_ranges_all.append(tidal_ranges_from_LW[i])
                rel_times_all.append(peak_real_times_LW[i])
    else:
        # LW occurs first
        try:
            tidal_ranges_from_LW = abs(peak_elevs_LW[:] - peak_elevs_HW[:])
            tidal_ranges_from_HW = peak_elevs_HW[:-1] - peak_elevs_LW[1:]
        except (IndexError, ValueError):
            tidal_ranges_from_LW = abs(peak_elevs_LW[:-1] - peak_elevs_HW[:])
            tidal_ranges_from_HW = peak_elevs_HW[:] - peak_elevs_LW[1:]
        tidal_ranges_all, rel_times_all = [], []
        for i in range(len(tidal_ranges_from_LW)):
            tidal_ranges_all.append(tidal_ranges_from_LW[i])
            rel_times_all.append(peak_real_times_LW[i])
            if i < len(tidal_ranges_from_HW):
                tidal_ranges_all.append(tidal_ranges_from_HW[i])
                rel_times_all.append(peak_real_times_HW[i])

    return tidal_ranges_all, rel_times_all


def ranges(signal):
    """Return list of tidal ranges for a [time, elevation] signal array."""
    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges, _ = tidal_ranges_from_peaks(
        peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW
    )
    return tidal_ranges


def theoretical_energy(signal):
    """
    Calculates the total theoretical tidal energy from a tidal elevation signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Array with columns [time (s), elevation (m)].

    Returns
    -------
    float
        Total theoretical tidal energy in kWh.
    """
    rho = 1021   # seawater density, kg/m³
    grav = 9.81  # gravitational acceleration, m/s²

    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges, _ = tidal_ranges_from_peaks(
        peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW
    )

    emax = 0.5 * rho * grav * np.square(tidal_ranges) / 3.6e6  # kWh/m² per tidal cycle
    return np.sum(emax)


def mean_tidal_range_and_theoretical_energy(signal):
    """
    Returns the mean tidal range and total theoretical tidal energy for a signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Array with columns [time (s), elevation (m)].

    Returns
    -------
    R : float
        Mean tidal range (m).
    E : float
        Total theoretical tidal energy (MWh).
    """
    rho = 1021   # seawater density, kg/m³
    grav = 9.81  # gravitational acceleration, m/s²

    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges, _ = tidal_ranges_from_peaks(
        peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW
    )

    emax = 0.5 * rho * grav * np.square(tidal_ranges) / 3.6e6  # kWh/m² per tidal cycle
    E = np.sum(emax) / 1e3  # MWh/m²
    R = np.mean(tidal_ranges)

    return R, E
