import sys
import time
import warnings
import os
from datetime import datetime

import numpy as np
from thetis import *
from firedrake.petsc import PETSc
import tools.tidal_forcing
import tools.detectors
import tools.thetis_support_scripts
from inputs.simulation_parameters import *

warnings.simplefilter(action="ignore", category=DeprecationWarning)


def run_model():
    start_time = time.time()

    starttime = datetime.now()
    if MPI.COMM_WORLD.rank == 0:
        print('Start time:', starttime.strftime("%d/%m/%Y %H:%M:%S"))

    inputdir = 'inputs'
    outputdir = run_output_folder


    # simulation ID
    identifier = 0
    PETSc.Sys.Print('Simulation identifier: ' + str(identifier))

    # import fields from previous simulation or ramp
    with CheckpointFile(os.path.join(inputdir, f'export_{int(identifier-1)}.h5'), 'r') as CF:
        mesh2d = CF.load_mesh()
        bathymetry_2d = CF.load_function(mesh2d, name="bathymetry")
        h_viscosity = CF.load_function(mesh2d, name="viscosity")
        mu_manning = CF.load_function(mesh2d, name='manning')
        uv_init = CF.load_function(mesh2d, name="velocity")
        elev_init = CF.load_function(mesh2d, name="elevation")
        CF.close()

    PETSc.Sys.Print(f'Loaded mesh {mesh2d.name}')
    PETSc.Sys.Print(f'Exporting to {outputdir}')

    t_end = i_t_end                 # Simulation duration in sec
    t_start = identifier * t_end    # Simulation start time relative to tidal_forcing
    Dt = i_dt                       # Crank Nicolson timestep
    t_export = run_exp_interval     # Export time if necessary
    wd_alpha = i_alpha              # Wetting and drying
    lat_coriolis = i_lat_cor        # Coriolis calculation parameters

    # Create Firedrake Function Space as Continuous Galerkin of the first order based on the mesh that has been created
    # then create bathymetry and turbine density firedrake Functions.
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)

    # Coriolis forcing
    coriolis_2d = tools.thetis_support_scripts.coriolis(mesh2d, lat_coriolis)

    # --- create the solver object for the mesh and bathymetry, then configure it ---
    with timed_stage('initialisation'):
        # --- create solver ---
        solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
        options = solver_obj.options
        options.cfl_2d = 1.0
        options.use_nonlinear_equations = True
        options.simulation_export_time = t_export
        options.simulation_end_time = t_end
        options.coriolis_frequency = coriolis_2d
        options.output_directory = outputdir
        options.check_volume_conservation_2d = True
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

    # Boundary conditions
    tidal_elev = Function(bathymetry_2d.function_space())
    Gloucester = Constant(-50.)

    # dictionary of boundary conditions
    bnd_dictionary = {}
    for i in open_bnd:
        bnd_dictionary[i] = {'elev': tidal_elev}
    bnd_dictionary[7] = {'flux': Gloucester}
    solver_obj.bnd_functions['shallow_water'] = bnd_dictionary


    extra_detectors = list(np.load(additional_detector_files[0] + '.npy'))
    extra_detector_names = list(np.load(additional_detector_files[0] + '_names.npy'))

    # Simulation preliminaries
    solver_obj.assign_initial_conditions(elev=elev_init, uv=uv_init)
    det_xy, det_names = tools.detectors.get_detectors(mesh2d, maximum_distance=max_dist)

    # Adding detectors callback for monitor points
    cb = DetectorsCallback(solver_obj, extra_detectors, ['elev_2d', 'uv_2d'], name='detectors_TRS',
                           detector_names=extra_detector_names)

    cb2 = DetectorsCallback(solver_obj, det_xy, ['elev_2d', 'uv_2d'], name='detectors_gauges',
                           detector_names=det_names)

    # Adding callback objects to solver
    solver_obj.add_callback(cb, 'timestep')
    solver_obj.add_callback(cb2, 'timestep')

    uv, elev = solver_obj.timestepper.solution.subfunctions


    def intermediate_steps(t):
        """
        Function that is used here to export hdf5 fields for further processing.
        :param t: time
        :return: None
        """
        # Exporting to data file - useful for quick sampling etc.
        if incl_harmonic_analysis and (t % run_exp_elev_interval) == 0:
            PETSc.Sys.Print("Exporting elevation field for harmonic analysis")
            elev_CG = Function(P1_2d, name='elev_CG').project(elev)
            with CheckpointFile(os.path.join(outputdir, f"elev_{t:07d}"), "w") as checkpoint_file:
                checkpoint_file.save_function(elev_CG)

        # Export final state that can be picked up later
        # (exporting each function individually may cause issues with the mesh)
        if t == t_end:
            with CheckpointFile(os.path.join(inputdir, "run_export"), 'w') as f:
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
        completion_percentage = (t + t_start) / t_end * 100
        PETSc.Sys.Print("Updating tidal field at t={} - Simulation Progress: {:.2f}%".format(t_start + t, completion_percentage))
        tools.tidal_forcing.set_tidal_field(tidal_elev, t + int(t_start))


    solver_obj.iterate(update_forcings=update_forcings)

run_model()
