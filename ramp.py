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
    # ramp_output_folder is defined in simulation_parameters and typically
    # points to outputs/ramp/ so PVD and HDF5 exports are kept separate
    # from the main run outputs.
    outputdir = ramp_output_folder

    # Load the preprocessed mesh and all associated fields from the HDF5
    # checkpoint produced by preprocessing.py.  Loading them here rather
    # than recomputing avoids re-running the expensive Eikonal solves.
    with CheckpointFile(os.path.join(inputdir, "preprocessing.h5"), 'r') as CF:
        mesh2d = CF.load_mesh()
        bathymetry_2d = CF.load_function(mesh2d, name="bathymetry")
        h_viscosity   = CF.load_function(mesh2d, name="viscosity")
        mu_manning    = CF.load_function(mesh2d, name='manning')

    PETSc.Sys.Print(f'Loaded mesh {mesh2d.name}')
    PETSc.Sys.Print(f'Exporting to {outputdir}')

    # Simulation identifier for this ramp stage.
    # Using -1 distinguishes the spin-up checkpoint (export_-1.h5) from
    # the main run checkpoints (export_0.h5, export_1.h5, …).
    identifier = -1
    PETSc.Sys.Print(f'Simulation identifier: {identifier}')

    # ------------------------------------------------------------------
    # Temporal setup
    # ------------------------------------------------------------------
    # The ramp spans a virtual time window [-ramptime, 0] in the tidal-
    # forcing epoch.  Running from -ramptime to 0 means the tidal boundary
    # condition is evaluated at negative epoch times, where the ramp factor
    # smoothly grows from 0 to 1, ensuring the model starts from rest and
    # reaches a realistic tidal state by t=0.  The main run (run.py) then
    # picks up from t=0 with identifier=0.
    ramptime = i_ramptime       # total ramp duration (s), e.g. 2 days
    t_start  = -ramptime        # epoch time at the start of the ramp
    t_end    = 0.0              # epoch time at the end of the ramp
    Dt       = i_dt             # model time step (s)
    t_export = ramp_exp_interval  # interval between PVD snapshot exports (s)
    wd_alpha = i_alpha          # wetting-and-drying smoothing parameter (m)

    lat_coriolis = i_lat_cor    # reference latitude for Coriolis (degrees N)
    CG_2d = FunctionSpace(mesh2d, 'CG', 1)
    # Build a spatially varying Coriolis field from a beta-plane approximation
    # centred on lat_coriolis.
    coriolis_2d = tools.thetis_support_scripts.coriolis(mesh2d, lat_coriolis)

    # ------------------------------------------------------------------
    # Solver setup
    # ------------------------------------------------------------------
    with timed_stage('initialisation'):
        solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
        options = solver_obj.options

        # CFL number used internally by Thetis for stability diagnostics only
        # (the time step is fixed, so this does not control adaptivity).
        options.cfl_2d = 1.0

        # Solve the full nonlinear shallow-water equations (no linearisation).
        options.use_nonlinear_equations = True

        # Export a PVD snapshot every t_export seconds for visualisation.
        options.simulation_export_time = t_export

        # The solver advances from 0 to ramptime; the tidal-forcing callback
        # maps this internal clock to the epoch window [t_start, 0].
        options.simulation_end_time = ramptime

        options.coriolis_frequency = coriolis_2d
        options.output_directory = outputdir

        # Monitor global volume conservation as a sanity check at each export.
        options.check_volume_conservation_2d = True

        # Export velocity and surface elevation fields to PVD for visualisation.
        options.fields_to_export = ['uv_2d', 'elev_2d']
        # No HDF5 field exports during ramp (only the final checkpoint matters).
        options.fields_to_export_hdf5 = []

        # DG-DG element pair: discontinuous Galerkin for both velocity and
        # elevation.  This is mass-conserving and handles wetting/drying well.
        options.element_family = "dg-dg"

        # Crank–Nicolson time integration with implicitness_theta=0.75 gives a
        # stable, second-order-accurate scheme that is slightly dissipative
        # (theta>0.5 adds numerical damping to suppress high-frequency noise).
        options.swe_timestepper_type = 'CrankNicolson'
        options.swe_timestepper_options.implicitness_theta = 0.75

        # Semi-implicit linearisation accelerates convergence of the nonlinear
        # solver by providing a better initial guess for each Newton iteration.
        options.swe_timestepper_options.use_semi_implicit_linearization = True

        # Enable the wetting-and-drying scheme (Kärnä et al. 2011).
        # wd_alpha (metres) controls the steepness of the bathymetry smoothing
        # in very shallow regions; larger values are more stable but less sharp.
        options.use_wetting_and_drying = True
        options.wetting_and_drying_alpha = Constant(wd_alpha)

        options.manning_drag_coefficient = mu_manning
        options.horizontal_viscosity = h_viscosity

        # Include the grad-div viscosity term for improved numerical stability
        # in regions with strong velocity gradients (e.g. near headlands).
        options.use_grad_div_viscosity_term = True
        # Depth-dependent viscosity term is disabled: it can destabilise the
        # solver in very shallow cells during spin-up.
        options.use_grad_depth_viscosity_term = False

        options.timestep = Dt

        # PETSc nonlinear/linear solver settings:
        #   SNES (Newton) converges when residual drops by factor 1e-3, with
        #   a maximum of 20 Newton iterations per time step.
        #   KSP uses a direct LU factorisation (MUMPS) inside each Newton step,
        #   which is robust for the moderate problem sizes typical of 2-D shelf
        #   models and avoids the need to tune iterative-solver preconditioners.
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
    # tidal_elev holds the tidal surface-elevation signal interpolated from
    # a harmonic constituents database; it is updated at every time step by
    # the update_forcings callback below.
    tidal_elev = Function(bathymetry_2d.function_space())

    # Prescribed freshwater flux at boundary tag 7 (River Severn / Gloucester).
    # Negative sign follows Thetis convention: flux into the domain is negative.
    Gloucester = Constant(-50.)   # prescribed river flux at boundary 7 (m³/s)

    # Apply the tidal elevation condition to every open-sea boundary tag,
    # then override tag 7 with the river flux.
    bnd_dictionary = {}
    for i in open_bnd:
        bnd_dictionary[i] = {'elev': tidal_elev}
    bnd_dictionary[7] = {'flux': Gloucester}
    solver_obj.bnd_functions['shallow_water'] = bnd_dictionary

    # ------------------------------------------------------------------
    # Initial conditions — start from rest
    # ------------------------------------------------------------------
    # Elevation is set to zero (flat sea surface).
    # A small non-zero velocity (1 mm/s) is used to avoid a degenerate
    # zero-velocity initial state which can slow SNES convergence.
    elev_init = Function(CG_2d).assign(0.0)
    solver_obj.assign_initial_conditions(uv=as_vector((1e-3, 0.0)), elev=elev_init)

    uv, elev = solver_obj.timestepper.solution.subfunctions

    def intermediate_steps(t):
        """Export state at end of ramp for downstream use by run.py.

        Called at every tidal-forcing update; only performs I/O when the
        epoch time matches a scheduled export or the end of the ramp.
        """
        # Optionally export elevation snapshots at regular intervals for
        # offline harmonic analysis (e.g. T_TIDE or UTide).
        if incl_harmonic_analysis and t % ramp_exp_interval == 0:
            PETSc.Sys.Print("Exporting elevation field for harmonic analysis")
            # Project DG elevation onto CG space for cleaner interpolation
            elev_CG = Function(CG_2d, name='elev_CG').project(elev)
            checkpoint_file = checkpointing.DumbCheckpoint(
                os.path.join(outputdir, f'elev_{t}')
            )
            checkpoint_file.store(elev_CG)
            checkpoint_file.close()

        # At the end of the ramp, save the full model state so that run.py
        # can hot-start the main simulation from a spun-up initial condition.
        if t == t_end:
            with CheckpointFile(os.path.join(inputdir, f'export_{identifier}.h5'), 'w') as f:
                f.save_mesh(mesh2d)
                f.save_function(bathymetry_2d, name="bathymetry")
                f.save_function(h_viscosity,   name="viscosity")
                f.save_function(mu_manning,    name="manning")
                f.save_function(uv,            name="velocity")
                f.save_function(elev,          name="elevation")

            # Also write PVD snapshots for quick visual inspection
            File('outputs/velocityout.pvd').write(uv)
            File('outputs/elevationout.pvd').write(elev)

            elapsed = (time.time() - start_time) / 3600
            PETSc.Sys.Print(f'Simulation completed in {elapsed:.2f} hours.')

    def update_forcings(t):
        """Update tidal boundary elevation and print progress.

        t        — Thetis internal clock time (0 … ramptime)
        epoch_t  — corresponding tidal epoch time (-ramptime … 0)
        completion — percentage of the ramp completed (0 … 100 %)
        """
        # Map the internal model clock to the tidal epoch.
        # During the ramp epoch_t ∈ [-ramptime, 0]; the tidal forcing
        # module uses this to look up constituent amplitudes and phases.
        epoch_t = t + t_start
        # Progress expressed as a fraction of the full ramp window.
        completion = 100.0 + epoch_t / ramptime * 100.0
        intermediate_steps(float(epoch_t))
        PETSc.Sys.Print(
            f"Updating tidal field at t={epoch_t:.0f} — "
            f"Simulation Progress: {completion:.2f}%"
        )
        # set_tidal_field evaluates harmonic constituents at the given epoch
        # time and writes the result into tidal_elev in-place.
        tools.tidal_forcing.set_tidal_field(tidal_elev, t + int(t_start), t_start)

    solver_obj.iterate(update_forcings=update_forcings)


if __name__ == '__main__':
    main()
