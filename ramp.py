import sys
import time
import warnings
from datetime import datetime

from thetis import *
from firedrake.petsc import PETSc
import tools.tidal_forcing
import tools.thetis_support_scripts
from inputs.simulation_parameters import *

warnings.simplefilter(action="ignore", category=DeprecationWarning)

start_time = time.time()

starttime = datetime.now()
if MPI.COMM_WORLD.rank == 0:
    print('Start time:', starttime.strftime("%d/%m/%Y %H:%M:%S"))

inputdir = 'inputs'
outputdir = ramp_output_folder

with CheckpointFile(os.path.join(inputdir, "preprocessing.h5"), 'r') as CF:
    mesh2d = CF.load_mesh()
    bathymetry_2d = CF.load_function(mesh2d, name="bathymetry")
    h_viscosity = CF.load_function(mesh2d, name="viscosity")
    mu_manning = CF.load_function(mesh2d, name='manning')

PETSc.Sys.Print(f'Loaded mesh {mesh2d.name}')
PETSc.Sys.Print(f'Exporting to {outputdir}')

# simulation ID
identifier = -1
PETSc.Sys.Print(f'Simulation identifier : {identifier}')

ramptime = i_ramptime
t_start = - ramptime  # Simulation start time relative to tidal_forcing
t_end = ramptime + t_start  # Simulation duration in sec
Dt = i_dt  # Time integrator timestep
t_export = ramp_exp_interval  # Export time if necessary
wd_alpha = i_alpha  # Wetting and drying

lat_coriolis = i_lat_cor  # Coriolis calculation parameters
CG_2d = FunctionSpace(mesh2d, 'CG', 1)
coriolis_2d = tools.thetis_support_scripts.coriolis(mesh2d, lat_coriolis)

with timed_stage('initialisation'):
    # --- create solver ---
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.cfl_2d = 1.0
    options.use_nonlinear_equations = True
    options.simulation_export_time = t_export
    options.simulation_end_time = ramptime
    options.coriolis_frequency = coriolis_2d
    options.output_directory = outputdir
    options.check_volume_conservation_2d = True
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.fields_to_export_hdf5 = []
    options.element_family = "dg-dg"
    options.swe_timestepper_type = 'CrankNicolson'
    options.swe_timestepper_options.implicitness_theta = 0.75
    options.swe_timestepper_options.use_semi_implicit_linearization = True
    options.use_wetting_and_drying = True
    options.wetting_and_drying_alpha = Constant(wd_alpha)
    options.manning_drag_coefficient = mu_manning
    options.horizontal_viscosity = h_viscosity
    options.use_grad_div_viscosity_term = True
    options.use_grad_depth_viscosity_term = False
    options.timestep = Dt  # override dt for CrankNicolson (semi-implicit)
    options.swe_timestepper_options.solver_parameters = {
        'snes_rtol': 1e-3,
        'snes_max_it': 20,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_package': 'mumps',
    }

# function to be used to update elevation boundary conditions
tidal_elev = Function(bathymetry_2d.function_space())
Gloucester = Constant(-50.)

# dictionary of boundary conditions
bnd_dictionary = {}
for i in open_bnd:
    bnd_dictionary[i] = {'elev': tidal_elev}
bnd_dictionary[7] = {'flux': Gloucester}
solver_obj.bnd_functions['shallow_water'] = bnd_dictionary

# inital equilibrium state for elevation
elev_init = Function(CG_2d).assign(0.0)

# initial conditions in terms of velocity and elevation
solver_obj.assign_initial_conditions(uv=as_vector((1e-3, 0.0)), elev=elev_init)

# Splitting solution
uv, elev = solver_obj.timestepper.solution.subfunctions


def intermediate_steps(t):
    # Exporting to data file - useful for quick sampling etc.
    if incl_harmonic_analysis and t % ramp_exp_interval == 0:
        PETSc.Sys.Print("Exporting elevation field for harmonic analysis")
        elev_CG = Function(CG_2d, name='elev_CG').project(elev)
        checkpoint_file = checkpointing.DumbCheckpoint(os.path.join(outputdir, f'elev_{t}'))
        checkpoint_file.store(elev_CG)
        checkpoint_file.close()

    # Export final state that can be picked up later
    # (exporting each function individually may cause issues with the mesh)
    if t == t_end:
        with CheckpointFile(os.path.join(inputdir, f'export_{identifier}.h5'), 'w') as f:
            f.save_mesh(mesh2d)
            f.save_function(bathymetry_2d, name="bathymetry")
            f.save_function(h_viscosity, name="viscosity")
            f.save_function(mu_manning, name="manning")
            f.save_function(uv, name="velocity")
            f.save_function(elev, name="elevation")

        File('outputs/velocityout.pvd').write(uv)
        File('outputs/elevationout.pvd').write(elev)

        end_time = time.time()  # Record the end time
        elapsed_time = (end_time - start_time) / 3600  # Calculate the elapsed time

        PETSc.Sys.Print(
            f'Simulation completed in {elapsed_time:.2f} hours.')  # Print the time required for the simulation


def update_forcings(t):
    completion_percentage = 100 + (t + t_start) / ramptime * 100

    intermediate_steps(float(t + t_start))
    PETSc.Sys.Print("Updating tidal field at t={} - Simulation Progress: {:.2f}%".format(t_start + t, completion_percentage))
    tools.tidal_forcing.set_tidal_field(tidal_elev, t + int(t_start), t_start)


solver_obj.iterate(update_forcings=update_forcings)
