import datetime
from math import tanh

import uptide
import uptide.tidal_netcdf
from inputs.simulation_parameters import *
from tools import utm

utm_zone = i_zone
utm_band = i_band

tide = uptide.Tides(i_constituents)
tide.set_initial_time(datetime.datetime(s_year,s_month,s_day,s_hour,s_min))
tnci = uptide.tidal_netcdf.OTPSncTidalInterpolator(tide,grid_forcing_file,
     hf_forcing_file, ranges=range_forcing_coords)

def set_tidal_field(elev, t, t_start=None, ramptime=i_ramptime/4.):
    tnci.set_time(t)
    mesh2d = elev.function_space().mesh()
    xvector = mesh2d.coordinates.dat.data
    evector = elev.dat.data
    ramp = tanh((t - t_start) / ramptime) if t_start is not None else 1.

    for i, xy in enumerate(xvector):
        lat, lon = utm.to_latlon(xy[0], xy[1], utm_zone, utm_band)
        try:
            evector[i] = tnci.get_val((lon, lat)) * ramp
        except uptide.netcdf_reader.CoordinateError:
            evector[i] = 0.
