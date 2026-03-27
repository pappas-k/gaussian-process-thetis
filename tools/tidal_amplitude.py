import datetime

import numpy as np
import scipy.interpolate
import uptide
import uptide.tidal_netcdf
from inputs.simulation_parameters import *
from tools import utm

utm_zone = i_zone
utm_band = i_band

# def initial_forcing(t_start):
tide = uptide.Tides(i_constituents)
tide.set_initial_time(datetime.datetime(s_year,s_month,s_day,s_hour,s_min))
tnci = uptide.tidal_netcdf.OTPSncTidalInterpolator(tide,grid_forcing_file,hf_forcing_file,
                                                   ranges=range_forcing_coords)

def get_lowest_astronomical_tide(elev):
  mesh2d = elev.function_space().mesh()
  xvector = mesh2d.coordinates.dat.data
  evector = elev.dat.data
  data = np.loadtxt('inputs/lat.txt')

  intp = scipy.interpolate.NearestNDInterpolator(data[:,0:2], data[:,2])
  for i,xy in enumerate(xvector):
    evector[i] = intp(xy)
