"""
Main simulation script — 15-day tidal run.

Loads spin-up state from inputs/export_-1.h5 (produced by ramp.py),
runs the full tidal simulation with detector callbacks, and writes
final state to inputs/run_export.h5.
"""
import os
import sys
import time
import warnings
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
    # run_output_folder is defined in simulation_parameters and points to
    # the directory where PVD snapshots and detector CSV files are written.
    outputdir = run_output_folder

    # Simulation identifier: 0 means this run starts at t=0 in the tidal epoch.
    # If running sequential segments (e.g. identifier=1 picks up from identifier=0),
    # the checkpoint filename is constructed from identifier-1 so each segment
    # hot-starts from the previous one's final state.
    identifier = 0
    PETSc.Sys.Print('Simulation identifier: ' + str(identifier))

    # Load the spun-up model state produced at the end of ramp.py.
    # All fields (mesh, bathymetry, viscosity, manning, velocity, elevation)
    # are stored together in a single HDF5 checkpoint to ensure consistency.
    with CheckpointFile(os.path.join(inputdir, f'export_{int(identifier - 1)}.h5'), 'r') as CF:
        mesh2d        = CF.load_mesh()
        bathymetry_2d = CF.load_function(mesh2d, name="bathymetry")
        h_viscosity   = CF.load_function(mesh2d, name="viscosity")
        mu_manning    = CF.load_function(mesh2d, name='manning')
        uv_init       = CF.load_function(mesh2d, name="velocity")
        elev_init     = CF.load_function(mesh2d, name="elevation")

    PETSc.Sys.Print(f'Loaded mesh {mesh2d.name}')
    PETSc.Sys.Print(f'Exporting to {outputdir}')

    # ------------------------------------------------------------------
    # Temporal setup
    # ------------------------------------------------------------------
    # t_end    — total model run duration (s), e.g. 15 days = 1 296 000 s
    # t_start  — tidal epoch time at which this segment begins.
    #            For identifier=0 this is 0; for identifier=1 it would be t_end, etc.
    #            This value is passed to the tidal-forcing module so constituents
    #            are evaluated at the correct phase in the tidal record.
    t_end    = i_t_end               # simulation duration (s)
    t_start  = identifier * t_end    # start time in tidal epoch (s)
    Dt       = i_dt                  # model time step (s)
    t_export = run_exp_interval      # PVD export interval (s)
    wd_alpha = i_alpha               # wetting-and-drying alpha (m)
    lat_coriolis = i_lat_cor         # reference latitude for Coriolis (°N)

    # CG1 function space for projected fields (elevation snapshots, Coriolis)
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    # Coriolis field computed from a beta-plane approximation at lat_coriolis
    coriolis_2d = tools.thetis_support_scripts.coriolis(mesh2d, lat_coriolis)

    # ------------------------------------------------------------------
    # Solver setup
    # ------------------------------------------------------------------
    with timed_stage('initialisation'):
        solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
        options = solver_obj.options

        # CFL number used internally by Thetis for diagnostics only
        options.cfl_2d = 1.0

        # Solve the full nonlinear shallow-water equations
        options.use_nonlinear_equations = True

        # Export a PVD snapshot every t_export seconds for visualisation
        options.simulation_export_time = t_export

        # Run the model for the full segment duration
        options.simulation_end_time = t_end

        options.coriolis_frequency = coriolis_2d
        options.output_directory = outputdir

        # Track global volume conservation as a diagnostic printed each export
        options.check_volume_conservation_2d = True

        # DG-DG element pair for mass-conservative, wetting-/drying-compatible
        # discretisation of velocity and surface elevation
        options.element_family = "dg-dg"

        # Crank–Nicolson with theta=0.75: second-order accurate and slightly
        # dissipative, which suppresses high-frequency spurious oscillations
        # without significantly affecting resolved tidal dynamics.
        options.swe_timestepper_type = 'CrankNicolson'
        options.swe_timestepper_options.implicitness_theta = 0.75

        # Semi-implicit linearisation provides a better Newton initial guess,
        # reducing the number of SNES iterations per time step.
        options.swe_timestepper_options.use_semi_implicit_linearization = True

        # Wetting-and-drying scheme (Kärnä et al. 2011).
        # wd_alpha smooths the effective bathymetry in very shallow cells;
        # the value is chosen to balance stability against accuracy in
        # intertidal regions such as the Severn Estuary mudflats.
        options.use_wetting_and_drying = True
        options.wetting_and_drying_alpha = Constant(wd_alpha)

        options.manning_drag_coefficient = mu_manning
        options.horizontal_viscosity = h_viscosity

        # Include the symmetric grad-div viscosity term for stability near
        # sharp bathymetric gradients (headlands, channels).
        options.use_grad_div_viscosity_term = True
        # Depth-gradient viscosity term is disabled to avoid spurious forcing
        # in steep-slope regions where the depth gradient is large.
        options.use_grad_depth_viscosity_term = False

        options.timestep = Dt

        # SNES/KSP settings — same as ramp.py for consistency:
        #   SNES: relative tolerance 1e-3, max 20 Newton iterations
        #   KSP:  direct solve with MUMPS LU factorisation (robust, scalable
        #         up to ~O(10^6) DOFs on distributed memory architectures)
        options.swe_timestepper_options.solver_parameters = {
            'snes_rtol': 1e-3,
            'snes_max_it': 20,
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_package': 'mumps',
        }

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    # tidal_elev is updated at every time step by update_forcings to hold
    # the current tidal surface elevation evaluated from harmonic constituents.
    tidal_elev = Function(bathymetry_2d.function_space())

    # Constant river flux at boundary tag 7 (River Severn near Gloucester).
    # Negative sign: flux leaving the domain (freshwater into the estuary).
    Gloucester = Constant(-50.)   # prescribed river flux at boundary 7 (m³/s)

    # Apply the tidal elevation boundary condition to all open-sea boundaries,
    # then override tag 7 with the prescribed river flux.
    bnd_dictionary = {}
    for i in open_bnd:
        bnd_dictionary[i] = {'elev': tidal_elev}
    bnd_dictionary[7] = {'flux': Gloucester}
    solver_obj.bnd_functions['shallow_water'] = bnd_dictionary

    # ------------------------------------------------------------------
    # Detector / gauge setup
    # ------------------------------------------------------------------
    # Load extra detector locations (e.g. tidal energy resource sites or
    # synthetic monitoring points) from a pre-computed numpy array.
    # The companion _names array provides human-readable labels for the CSV output.
    extra_detectors      = list(np.load(additional_detector_files[0] + '.npy'))
    extra_detector_names = list(np.load(additional_detector_files[0] + '_names.npy'))

    # Assign initial conditions from the ramp checkpoint.
    # The solver will continue from the spun-up state seamlessly.
    solver_obj.assign_initial_conditions(elev=elev_init, uv=uv_init)

    # Build the list of BODC/NTSLF tide-gauge detector locations that fall
    # within max_dist metres of a mesh node (removes gauges on dry land or
    # outside the model domain).
    det_xy, det_names = tools.detectors.get_detectors(mesh2d, maximum_distance=max_dist)

    # Register two DetectorsCallback objects:
    #   cb  — samples at tidal resource site locations (extra_detectors)
    #   cb2 — samples at observational tide-gauge locations (det_xy)
    # Both record elevation and depth-averaged velocity at every time step,
    # producing CSV files that can be compared directly with observations.
    cb = DetectorsCallback(solver_obj, extra_detectors, ['elev_2d', 'uv_2d'],
                           name='detectors_TRS', detector_names=extra_detector_names)
    cb2 = DetectorsCallback(solver_obj, det_xy, ['elev_2d', 'uv_2d'],
                            name='detectors_gauges', detector_names=det_names)
    solver_obj.add_callback(cb,  'timestep')
    solver_obj.add_callback(cb2, 'timestep')

    uv, elev = solver_obj.timestepper.solution.subfunctions

    def intermediate_steps(t):
        """Export elevation snapshots and final state checkpoint.

        t — internal model time (s) within the current segment (0 … t_end).
        """
        # Periodically export the CG-projected elevation for offline harmonic
        # analysis (e.g. UTide) at a user-defined sub-interval.
        if incl_harmonic_analysis and (t % run_exp_elev_interval) == 0:
            PETSc.Sys.Print("Exporting elevation field for harmonic analysis")
            # Project from the DG solution space to CG1 to produce a
            # continuous field suitable for harmonic constituent fitting.
            elev_CG = Function(P1_2d, name='elev_CG').project(elev)
            with CheckpointFile(os.path.join(outputdir, f"elev_{t:07d}.h5"), "w") as cf:
                cf.save_function(elev_CG)

        # At the end of the segment, save the full model state.
        # This checkpoint enables a subsequent segment (identifier=1, 2, …)
        # to hot-start without repeating the spin-up.
        if t == t_end:
            with CheckpointFile(os.path.join(inputdir, "run_export.h5"), 'w') as f:
                f.save_mesh(mesh2d)
                f.save_function(bathymetry_2d, name="bathymetry")
                f.save_function(h_viscosity,   name="viscosity")
                f.save_function(mu_manning,    name="manning")
                f.save_function(uv,            name="velocity")
                f.save_function(elev,          name="elevation")

            # Write final PVD snapshots for quick visual inspection
            File('outputs/velocityout.pvd').write(uv)
            File('outputs/elevationout.pvd').write(elev)

            elapsed = (time.time() - start_time) / 3600
            PETSc.Sys.Print(f'Simulation completed in {elapsed:.2f} hours.')

    def update_forcings(t):
        """Update tidal boundary elevation and log progress.

        t        — Thetis internal time within segment (0 … t_end, s)
        epoch_t  — corresponding absolute tidal epoch time (t_start + t, s)
        completion — fraction of the segment completed (0 … 100 %)
        """
        # Compute the absolute tidal epoch time for this time step.
        # This is the time passed to the harmonic constituent database
        # to look up the correct tidal elevation at the open boundaries.
        epoch_t = t_start + t
        # Percentage of the current segment that has been simulated
        completion = epoch_t / t_end * 100.0
        intermediate_steps(t)
        PETSc.Sys.Print(
            f"Updating tidal field at t={epoch_t:.0f} — "
            f"Simulation Progress: {completion:.2f}%"
        )
        # Evaluate harmonic tidal constituents at epoch_t and update
        # tidal_elev in-place.  The function reads constituent amplitudes
        # and phases from the pre-loaded atlas and applies nodal corrections.
        tools.tidal_forcing.set_tidal_field(tidal_elev, t + int(t_start))

    solver_obj.iterate(update_forcings=update_forcings)


if __name__ == '__main__':
    run_model()
