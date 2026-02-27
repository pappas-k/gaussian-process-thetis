import uptide
import numpy as np
from statistics import mean
import datetime
import pandas as pd
import matplotlib.pyplot as plt


def extract_signal_from_recordings(tidegauge_file, start_date, end_date, plot=False ):
    """
     Extracts tidal elevation signal from recordings within a specified date range.
     --------------
     params: tidegauge_file (str): Path to the CSV file containing the tidal recordings.
             start_date (datetime): Start date of the desired date range.
             end_date (datetime): End date of the desired date range.
             plot (bool, optional): Whether to plot the extracted signal. Default is False.
     --------------
     Returns: numpy.ndarray: Array containing the extracted tidal signal, with columns [time, elevation]
     """

    date = np.loadtxt(tidegauge_file, skiprows=2, usecols=(9,), dtype=str, delimiter=',')

    # Convert the date array to a datetime format:
    date = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date]
    #print(date)
    eta = np.loadtxt(tidegauge_file, skiprows=2, usecols=(11,), dtype=float, delimiter=',')

    QC_flag = np.loadtxt(tidegauge_file, skiprows=2, usecols=(12,), dtype=str, delimiter=',') #quality control flag of data, if non empty data entries must be skipped
    pd0 = pd.DataFrame({"Time": date, "Elevation": eta, "QC flag": QC_flag})
    # filter all rows for which the elevation has errors, (drop invalid values)
    pd0_filtered = pd0[ (~pd0['QC flag'].str.contains('N')) & (~pd0['QC flag'].str.contains('M')) & (~pd0['QC flag'].str.contains('T'))]
    #pd0_filtered = pd0_filtered[pd0_filtered["Elevation"] > -15]
    #pd0_filtered = pd0_filtered[pd0_filtered["Elevation"] < 15]

    date = pd.to_datetime(pd0_filtered['Time']).tolist()
    #date = pd0_filtered['Time'].values
    eta = pd0_filtered['Elevation'].values
    eta = eta - mean(eta)

    #Find the indices corresponding to the desired date range:
    #start_date = datetime.datetime(2002, 10, 20)
    #end_date = datetime.datetime(2002, 11, 19)
    start_index = np.where(np.array(date) >= start_date)[0][0]
    end_index = np.where(np.array(date) <= end_date)[0][-1]

    #Extract the subset of dates and elevations within the specified range:
    subset_date = date[start_index:end_index+1]
    subset_eta = eta[start_index:end_index+1]

    # Calculate the time difference in seconds
    time_diff = [(d - subset_date[0]).total_seconds() for d in subset_date]

    if plot:
        plt.plot(time_diff, subset_eta)
        plt.xlabel('Time')
        plt.ylabel('Elevation')
        plt.title('Tidal Elevation Data')
        plt.xticks(rotation=45)
        plt.show()

    signal = np.column_stack((time_diff, subset_eta))
    return signal


def extract_constituents_from_tidegauge_file(tidegauge_file='inputs/Mumbles_20010101_20010131.csv', start_date=datetime.datetime(2001, 1, 1, 0, 0, 0)):
    file = open(tidegauge_file)
    print(file)
    numline = len(file.readlines()) #number of rows of the tidegauge_file
    #start_date = datetime.datetime(2002, 1, 1, 0, 0, 0)

    # a=np.loadtxt(tidegauge_file, skiprows=13, usecols=(0,), dtype='str', delimiter=';')
    # dates = []
    # t=[]
    # for i in range(0,len(a)):
    #    dates.append(a[i])
    # from datetime import datetime
    # for d in dates:
    #     date = datetime.strptime(d, '%d/%m/%Y %H:%M:%S')
    #     #print(type(date))
    #     #print(date)
    #     time= date.timestamp() - datetime(2019, 1, 1, 0, 0).timestamp()
    #     t.append(time)
    # print("time",t)


    t = np.arange(0, (numline - 2) * 15 * 60, 15 * 60)
    eta = np.loadtxt(tidegauge_file, skiprows=2, usecols=(11,), dtype=float, delimiter=',')
    QC_flag = np.loadtxt(tidegauge_file, skiprows=2, usecols=(12,), dtype=str, delimiter=',') #quality control flag of data, if non empty data entries must be skipped
    #print("ETA=",eta)
    #print("QC=",QC_flag)
    #print("------------------------------------------------------")
    pd0 = pd.DataFrame({"Time": t, "Elevation": eta, "QC flag": QC_flag  })
    # filter all rows for which the elevation has errors, (drop negative-wrong values)
    pd0_filtered = pd0[(~pd0['QC flag'].str.contains('N')) & (~pd0['QC flag'].str.contains('M')) & (~pd0['QC flag'].str.contains('T')) ]
    pd0_filtered = pd0_filtered[pd0_filtered["Elevation"] > -15]
    pd0_filtered = pd0_filtered[pd0_filtered["Elevation"] < 15]
    #print(pd0)
    #print(pd0_filtered)

    t=pd0_filtered['Time'].values
    eta = pd0_filtered['Elevation'].values

    #print("Availability of signal over a nodal cycle (%):", int(len(eta)*100 / ((int(18.61 * 365.25 * 24 * 60 * 60 / 900)))))

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
    #thetis_phases=np.degrees(thetis_phases)
    #thetis_phases = np.remainder(thetis_phases,2*math.pi)*360/(2*math.pi)
    #thetis_phases = np.remainder(thetis_phases, 2*math.pi) * 360 / (2 * math.pi)
    #print(constituents,thetis_amplitudes,thetis_phases)
    pd1 = pd.DataFrame({"Constituents": constituents,"Amplitude":thetis_amplitudes , "Phase": thetis_phases, })
    pd1 = pd1.sort_values(by=['Amplitude'], ignore_index=True, ascending=False)  # rearange dataframe in descending order of amplitude
    print(pd1)
    print('Minimum Rayleigh Period=', tide.get_minimum_Rayleigh_period() / 86400.)
    #print("Constituents to be resolved in ONE year of data=", uptide.select_constituents(constituents, 365 * 86400))
    eta = eta - mean(eta)
    return pd1, t , eta


def signal_reconstruction(amplitudes,phases,
                          dt=108,
                          constituents=["M2", "S2"],
                          signal_duration=365.25 * 24 * 3600,
                          start_date=datetime.datetime(2002, 1, 1, 0, 0, 0),
                          time_series_start_time=0):
    """
      Reconstructs a tidal signal based on given amplitudes and phases of constituents.
      ------------
      params:
          amplitudes (array-like): Amplitudes of the constituents.
          phases (array-like): Phases of the constituents.
          dt (float, optional): Time step for the reconstructed signal in seconds. Default is 108 seconds.
          constituents (list, optional): List of tidal constituents to consider. Default is ["M2", "S2"].
          signal_duration (float, optional): Duration of the reconstructed signal in seconds. Default is 365.25 days.
          start_date (datetime, optional): Start date for the reconstructed signal. Default is January 1, 2002 at 00:00:00.
          time_series_start_time (float, optional): Start time for the time series. Default is 0.
      -------------
      Returns: numpy.ndarray: Array containing the reconstructed tidal signal, with columns [time, elevation].
      """
    # Create a time series
    time_series = np.arange(time_series_start_time, signal_duration, dt)
    amplitudes, phases= amplitudes[:len(constituents)], phases[:len(constituents)]
    # Conduct signal reconstruction
    tide = uptide.Tides(constituents)
    tide.set_initial_time(start_date)
    tide_elevs = tide.from_amplitude_phase(amplitudes, phases, time_series)
    return np.column_stack((time_series, tide_elevs))





def find_tidal_peaks( rel_times, tide_elevs, peak_type):
    """"
    Function which finds HW or LW peaks and their locations. Averages every two as per the methodology from NOAA,
    rel_times is the time in seconds that the tidal elevations tide_elevs occurs. peak_type = HW for high water,
    and = LW for low water. rel_times is the time in s.
    rel_times:
    """
    from scipy import signal
    # if peak_type is low water (LW), will make tidal elevations negative to fine 'peaks' using negative multiplier
    if peak_type == 'LW':
        multiplier = -1
    # if peak_type is high water (HW), will use original elevation data i.e. multiply by 1
    else:
        multiplier = 1
    # determine index at which the peaks occur - multiplier determines if HW (1) or LW (-1)
    peak_idx = signal.find_peaks(tide_elevs * multiplier)[0]
    peak_rel_times = rel_times[peak_idx]
    peak_elevs = tide_elevs[peak_idx]
    return peak_rel_times, peak_elevs


def tidal_ranges_from_peaks(peak_real_times_HW, peak_real_times_LW, peak_elevs_HW, peak_elevs_LW):
    """"
    Function which finds tidal ranges based on peaks.
    """
    if peak_real_times_HW[0] < peak_real_times_LW[0]:
        try:
            tidal_ranges_from_HW = peak_elevs_HW - peak_elevs_LW
            tidal_ranges_from_LW = abs(peak_elevs_LW[:-1] - peak_elevs_HW[1:])
        except (IndexError, ValueError):
            tidal_ranges_from_HW = peak_elevs_HW[:-1] - peak_elevs_LW
            tidal_ranges_from_LW = abs(peak_elevs_LW[:] - peak_elevs_HW[1:])
        tidal_ranges_all = []
        rel_times_all = []
        # Since HW occurs first we append to a list he HW value and the LW one and so on
        for i in range(0, len(tidal_ranges_from_LW)):
            tidal_ranges_all.append(tidal_ranges_from_HW[i])
            tidal_ranges_all.append(tidal_ranges_from_LW[i])
            rel_times_all.append(peak_real_times_HW[i])
            rel_times_all.append(peak_real_times_LW[i])
    elif peak_real_times_HW[0] > peak_real_times_LW[0]:
        try:
            tidal_ranges_from_LW = abs(peak_elevs_LW[:] - peak_elevs_HW[:])
            tidal_ranges_from_HW = peak_elevs_HW[:-1] - peak_elevs_LW[1:]
        except (IndexError, ValueError):
            tidal_ranges_from_LW = abs(peak_elevs_LW[:-1] - peak_elevs_HW[:])
            tidal_ranges_from_HW = peak_elevs_HW[:] - peak_elevs_LW[1:]
        tidal_ranges_all = []
        rel_times_all = []
        for i in range(0, len(tidal_ranges_from_HW)):
            tidal_ranges_all.append(tidal_ranges_from_LW[i])
            tidal_ranges_all.append(tidal_ranges_from_HW[i])
            rel_times_all.append(peak_real_times_LW[i])
            rel_times_all.append(peak_real_times_HW[i])

    return tidal_ranges_all, rel_times_all




def ranges(signal):
    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW,peak_elevs_LW)
    return tidal_ranges



def theoretical_energy(signal):
    """
       Calculates the theoretical tidal energy from a given tidal elevation signal.
       -----------------
       param: signal (numpy.ndarray): Array containing the tidal elevation signal, with columns [time, elevation].
       -----------------
       Returns: float: The total theoretical tidal energy in kilowatt-hours (kWh).
       """
    rho=1021     # Density of seawater in kg/m^3
    grav= 9.81   # Acceleration due to gravity in m/s^2

    # Extract time and elevation from the signal:
    rel_time, tide_elevs = signal[:,0], signal[:,1]

    # Find peak high water (HW) and low water (LW) points:
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')

    # Calculate tidal ranges and corresponding relative times:
    tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
    # Calculate maximum energy for each tidal range:
    emax = 0.5 * rho * grav * np.square(tidal_ranges) / 3.6

    # Sum up the maximum energy values:
    E= np.sum(emax)

    return E

def mean_tidal_range_and_theoretical_energy(signal):
    """
       Calculates the theoretical tidal energy from a given tidal elevation signal.
       -----------------
       param: signal (numpy.ndarray): Array containing the tidal elevation signal, with columns [time, elevation].
       -----------------
       Returns: float: The total theoretical tidal energy in kilowatt-hours (kWh).
       """
    rho=1021     # Density of seawater in kg/m^3
    grav= 9.81   # Acceleration due to gravity in m/s^2

    # Extract time and elevation from the signal:
    rel_time, tide_elevs = signal[:,0], signal[:,1]

    # Find peak high water (HW) and low water (LW) points:
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')

    # Calculate tidal ranges and corresponding relative times:
    tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
    #print(f"tidal ranges : {tidal_ranges}")
    #print(f"len tidal ranges : {len(tidal_ranges)}")
    # Calculate maximum energy for each tidal range:
    emax = 0.5 * rho * grav * np.square(tidal_ranges) / 3.6 #e+6

    # Sum up the maximum energy values:
    E= np.sum(emax) /1e+6  #KWh

    R = np.mean(tidal_ranges)

    return R, E
