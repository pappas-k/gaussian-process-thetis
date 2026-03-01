"""
Spin-up (ramp) script — 2-day tidal ramp from rest.

Loads the preprocessed mesh and fields from inputs/preprocessing.h5,
runs a short spin-up with a linearly ramped tidal forcing, and saves
the final state to inputs/export_-1.h5 for use by run.py.
"""
import os
import time
import warnings
from datetime import datetime

from thetis import *
from firedrake.petsc import PETSc
import tools.tidal_forcing
import tools.thetis_support_scripts
from inputs.simulation_parameters import *

warnings.simplefilter(action="ignore", category=DeprecationWarning)


def main():
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

    # Simulation identifier for this ramp stage
    identifier = -1
    PETSc.Sys.Print(f'Simulation identifier: {identifier}')

    # Temporal setup: ramp spans [-ramptime, 0] in the tidal-forcing epoch so
    # that the main simulation (identifier=0) starts at t=0 with a spun-up state.
    ramptime = i_ramptime
    t_start  = -ramptime   # epoch time at start of ramp
    t_end    = 0.0         # epoch time at end of ramp (= ramptime of integration)
    Dt       = i_dt
    t_export = ramp_exp_interval
    wd_alpha = i_alpha

    lat_coriolis = i_lat_cor
    CG_2d = FunctionSpace(mesh2d, 'CG', 1)
    coriolis_2d = tools.thetis_support_scripts.coriolis(mesh2d, lat_coriolis)

    with timed_stage('initialisation'):
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
        options.timestep = Dt
        options.swe_timestepper_options.solver_parameters = {
            'snes_rtol': 1e-3,
            'snes_max_it': 20,
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_package': 'mumps',
        }

    # Boundary conditions
    tidal_elev = Function(bathymetry_2d.function_space())
    Gloucester = Constant(-50.)   # prescribed river flux at boundary 7 (m³/s)

    bnd_dictionary = {}
    for i in open_bnd:
        bnd_dictionary[i] = {'elev': tidal_elev}
    bnd_dictionary[7] = {'flux': Gloucester}
    solver_obj.bnd_functions['shallow_water'] = bnd_dictionary

    # Start from rest
    elev_init = Function(CG_2d).assign(0.0)
    solver_obj.assign_initial_conditions(uv=as_vector((1e-3, 0.0)), elev=elev_init)

    uv, elev = solver_obj.timestepper.solution.subfunctions

    def intermediate_steps(t):
        """Export state at end of ramp for downstream use by run.py."""
        if incl_harmonic_analysis and t % ramp_exp_interval == 0:
            PETSc.Sys.Print("Exporting elevation field for harmonic analysis")
            elev_CG = Function(CG_2d, name='elev_CG').project(elev)
            checkpoint_file = checkpointing.DumbCheckpoint(
                os.path.join(outputdir, f'elev_{t}')
            )
            checkpoint_file.store(elev_CG)
            checkpoint_file.close()

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

            elapsed = (time.time() - start_time) / 3600
            PETSc.Sys.Print(f'Simulation completed in {elapsed:.2f} hours.')

    def update_forcings(t):
        epoch_t = t + t_start
        completion = 100.0 + epoch_t / ramptime * 100.0
        intermediate_steps(float(epoch_t))
        PETSc.Sys.Print(
            f"Updating tidal field at t={epoch_t:.0f} — "
            f"Simulation Progress: {completion:.2f}%"
        )
        tools.tidal_forcing.set_tidal_field(tidal_elev, t + int(t_start), t_start)

    solver_obj.iterate(update_forcings=update_forcings)


if __name__ == '__main__':
    main()
