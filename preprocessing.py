"""
Pre-processing script
"""
from thetis import *
import inputs.simulation_parameters as inputs
from tools import bathymetry, tidal_amplitude, field_tools
from firedrake.petsc import PETSc

import os
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d

starttime = datetime.now()

if MPI.COMM_WORLD.rank == 0:
    print('Start time:', starttime.strftime("%d/%m/%Y %H:%M:%S"))
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
outputdir = "outputs/outputs"
inputdir = "inputs"
mesh = Mesh(inputs.mesh_file)

V = FunctionSpace(mesh, 'CG', 1)
diff = Function(V)

# Step 0 - Adjust bathymetry based on precalculated M2+S2 amplitudes
if inputs.add_amps:
    data = numpy.loadtxt('inputs/M2S2.txt')
    diff.dat.data[:] = data[:,2]
else:
    tidal_amplitude.get_lowest_astronomical_tide(diff)
    File(os.path.join(outputdir, 'lat.pvd')).write(diff)

# Step 1 - Calculate distance for viscosity
PETSc.Sys.Print("Calculate distance for viscosity")

L = inputs.i_L
u = field_tools.eik(V, inputs.open_bnd,
                    outfilename=os.path.join(outputdir, "dist.pvd"))
ue = field_tools.eik(V, 100,
                     outfilename=os.path.join(outputdir, "dist.pvd"))
ui = field_tools.eik(V, 101,
                     outfilename=os.path.join(outputdir, "dist.pvd"))

# Adding viscosity sponge
h_viscosity = Function(V, name="viscosity")
h_viscosity.interpolate(max_value(1., 1000 * (1. - u / 2e4)))
File(os.path.join(outputdir, 'viscosity.pvd')).write(h_viscosity)

# Creating a Manning/Quadratic drag/other type friction field to be used in the simulations

if inputs.use_friction_data:
    manning_data = np.load(inputs.friction_data)
    interpolator = np.vectorize(interp1d(manning_data[0, :], manning_data[1, :],
                                         fill_value=(manning_data[1, 0], manning_data[1, -1]),
                                         bounds_error=False))
    manning_2d = bathymetry.get_manning_class(inputs.bed_classification_file, mesh, interpolator)
    bathymetry.smoothen_bathymetry(manning_2d)
else:
    manning_2d = Function(V, name='manning').assign(inputs.manning_bkg)
    fac1 = field_tools.transition_field(1.25, 1, ue, 180e3, 200e3)
    fac2 = field_tools.transition_field(0.75, 1, ui, 120e3, 150e3)
    manning_2d.interpolate(manning_2d * fac1 * fac2)


# patch Manning in Cardigan bay to reduce flux
x, y = SpatialCoordinate(mesh)

for gauss in inputs.manning_gauss:
    print(gauss)
    peak, (x0, y0, ang), (sd1, sd2), base, (r1, r2) = gauss
    manning_2d.assign(manning_2d +
                      field_tools.gaussian_hump(V, x, y,
                                                x0, y0, ang,
                                                peak - base, sd1, sd2,
                                                0, r1, r2))

File(os.path.join(outputdir, 'manning.pvd')).write(manning_2d)
print_output('Exported manning')


print(f"Bathymetric error = {inputs.bath_error} m")
# Interpolating bathymetry
bath = None
for i, (f, source, datum) in enumerate(inputs.bathymetries):
    bath = bathymetry.get_bathymetry(f, mesh, source=source, bathymetry_function=bath, bathy_name="", h=inputs.bath_error)
    if datum=='LAT':
        bath.assign(bath + diff)

bathymetry.smoothen_bathymetry(bath)

# Applying bathymetry correction at the boundary
# (e.g. when wetting and drying needs to be avoided)
# based on distance u from particular boundary determined by the Eikonal eq
bath.interpolate(max_value(
        (lambda bathy, dist: conditional(ge(bathy, 25 * (1. - dist / 15000)),
                                bathy, 25 * (1. - dist / 15000))) (bath, u),
                                inputs.i_min_depth))

File(os.path.join(outputdir, 'bath.pvd')).write(bath)

with CheckpointFile(os.path.join(inputdir, "preprocessing.h5"), "w") as CF:
    CF.save_mesh(mesh)
    CF.save_function(h_viscosity, name="viscosity")
    CF.save_function(bath, name="bathymetry")
    CF.save_function(manning_2d, name='manning')

endtime = datetime.now()
simulationtime = endtime - starttime

if MPI.COMM_WORLD.rank == 0:
    print('End time:', endtime.strftime("%d/%m/%Y %H:%M:%S"))
    print('Simulation time =', simulationtime)
