from thetis import *
import pyproj
import numpy as np
from inputs.simulation_parameters import *
from firedrake.petsc import PETSc

tidegauge_file = i_tidegauge_file

UTM_ZONE50 = pyproj.Proj(
        proj='utm',
        zone= i_zone,
        datum='WGS84',
        units='m',
        errcheck=True)
LL_WGS84 = pyproj.Proj(proj='latlong', datum='WGS84', errcheck=True)


def get_detectors(mesh2d, maximum_distance=max_dist, gauge_file=tidegauge_file):
    if gauge_file==tidegauge_file:
        gauge_names = np.loadtxt(gauge_file, skiprows=1, usecols=(0,), dtype=str, delimiter=',')
        gauge_xy = np.loadtxt(gauge_file, skiprows=1, usecols=(3,4), delimiter=',')

    ind = np.argsort(gauge_names)
    gauge_names = list(gauge_names[ind])
    gauge_xy = list(gauge_xy[ind])

    return select_and_move_detectors(mesh2d, gauge_xy, gauge_names, maximum_distance=maximum_distance)

if __name__ == "__main__":
    os.chdir('../')
    mesh2d = Mesh(mesh_file)

    locations, names = get_detectors(mesh2d)
    if mesh2d.comm.rank == 0: # only processor

        PETSc.Sys.Print("Found detectors: {}".format(names))    #af
        import shapely.geometry
        import fiona
        import fiona.crs

        if not os.path.exists('data'):
            os.makedirs("data")
        schema = {'geometry': 'Point', 'properties': {'name': 'str'}}
        crs = fiona.crs.from_string(UTM_ZONE50.srs)
        with fiona.collection("data/detectors.shp", "w", "ESRI Shapefile", schema, crs=crs) as output:
            for xy, name in zip(locations, names):
                point = shapely.geometry.Point(xy[0], xy[1])
                PETSc.Sys.Print({'properties': {'name': name}, 'geometry': shapely.geometry.mapping(point)})    #af
