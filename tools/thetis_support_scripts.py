import pickle

from thetis import *
from tools import utm


def coriolis(mesh, lat,):
    """
    Adds coriolis term based on latitude (simplistic formulation but sufficient in regional models)
    :param mesh: Mesh
    :param lat: average latitude value
    :return:
    """
    R = 6371e3
    Omega = 7.292e-5
    lat_r = lat * pi / 180.
    f0 = 2 * Omega * sin(lat_r)
    beta = (1 / R) * 2 * Omega * cos(lat_r)
    x = SpatialCoordinate(mesh)
    x_0, y_0, utm_zone, zone_letter = utm.from_latlon(lat, 0)
    coriolis_2d = Function(FunctionSpace(mesh, 'CG', 1), name="coriolis_2d")
    coriolis_2d.interpolate(f0 + beta * (x[1] - y_0))
    return coriolis_2d


def export_final_state(inputdir, identifier, uv, elev, lagoon = None):
    """
    Exporting final state for subsequent simulations
    :param inputdir: input file directory
    :param identifier: simulation identifier
    :param uv: velocity field
    :param elev: elevation field
    :param lagoon: flag if tidal range operation is included.
    :return:
    """
    print_output("Exporting fields for subsequent simulation")

    chk = DumbCheckpoint(inputdir + "/velocity" + str(identifier + 1), mode=FILE_CREATE)
    chk.store(uv, name="velocity")
    File('outputs/velocityout.pvd').write(uv)
    chk.close()
    chk = DumbCheckpoint(inputdir + "/elevation" + str(identifier + 1), mode=FILE_CREATE)
    chk.store(elev, name="elevation")
    File('outputs/elevationout.pvd').write(elev)
    chk.close()

    if lagoon is not None:
        output_status = []
        for i in range(len(lagoon)):
            output_status.append(lagoon[i])
        pickle.dump(output_status, open(inputdir+"/barrage_status_"+str(identifier+1)+".p", "wb"))
