"""
Pre-processing script.

Builds and saves the mesh, bathymetry, viscosity sponge, and Manning
friction fields to inputs/preprocessing.h5 for use by ramp.py and run.py.

Steps
-----
1. Load mesh.
2. Compute LAT correction (or load pre-computed M2+S2 amplitudes).
3. Solve Eikonal equation for distance fields used in the viscosity sponge.
4. Build Manning friction field (uniform background + Gaussian patches).
5. Interpolate and smooth bathymetry; apply open-boundary depth correction.
6. Save all fields to checkpoint.
"""
import os
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d

from thetis import *
from firedrake.petsc import PETSc
import inputs.simulation_parameters as inputs
from tools import bathymetry, tidal_amplitude, field_tools


def main():
    starttime = datetime.now()
    if MPI.COMM_WORLD.rank == 0:
        print('Start time:', starttime.strftime("%d/%m/%Y %H:%M:%S"))
        # Create the output directory if it does not already exist
        if not os.path.exists('outputs'):
            os.makedirs('outputs')

    outputdir = "outputs/outputs"
    inputdir  = "inputs"

    # -----------------------------------------------------------------
    # Mesh
    # -----------------------------------------------------------------
    # Load the unstructured mesh from the file specified in simulation_parameters.
    # The mesh covers the West UK continental shelf and Bristol Channel region.
    mesh = Mesh(inputs.mesh_file)
    # CG1 function space used for all scalar fields (bathymetry, viscosity, etc.)
    V = FunctionSpace(mesh, 'CG', 1)

    # -----------------------------------------------------------------
    # Step 0 — LAT correction field
    # -----------------------------------------------------------------
    # The Lowest Astronomical Tide (LAT) correction converts chart datum depths
    # (referenced to LAT) to mean sea level (MSL) depths used in the model.
    # Two paths:
    #   a) add_amps=True  → read pre-computed M2+S2 amplitudes from file
    #      (faster; avoids running a harmonic solve during preprocessing).
    #   b) add_amps=False → compute LAT on the fly via tidal_amplitude module
    #      (more accurate but slower; also exports a PVD for inspection).
    diff = Function(V)
    if inputs.add_amps:
        # Column 2 of M2S2.txt contains the combined M2+S2 tidal amplitude
        # summed at each mesh node, which approximates the LAT correction.
        data = np.loadtxt('inputs/M2S2.txt')
        diff.dat.data[:] = data[:, 2]
    else:
        tidal_amplitude.get_lowest_astronomical_tide(diff)
        File(os.path.join(outputdir, 'lat.pvd')).write(diff)

    # -----------------------------------------------------------------
    # Step 1 — Distance fields for viscosity sponge (Eikonal equation)
    # -----------------------------------------------------------------
    # Solve the Eikonal equation to obtain geodesic distance from each
    # mesh node to a specified boundary tag.  Three distance fields are
    # computed:
    #   u  — distance to the tidal open boundary (open_bnd tag)
    #   ue — distance to the eastern internal boundary (tag 100)
    #   ui — distance to the internal island boundary (tag 101)
    # These are used below to construct a smooth viscosity sponge that
    # absorbs reflected waves near artificial boundaries.
    PETSc.Sys.Print("Computing distance fields for viscosity sponge")
    u  = field_tools.eik(V, inputs.open_bnd,
                         outfilename=os.path.join(outputdir, "dist_open.pvd"))
    ue = field_tools.eik(V, 100,
                         outfilename=os.path.join(outputdir, "dist_east.pvd"))
    ui = field_tools.eik(V, 101,
                         outfilename=os.path.join(outputdir, "dist_island.pvd"))

    # Sponge: background viscosity is 1 Pa·s throughout the domain.
    # Within 20 km of the open boundary it ramps up linearly to 1000 Pa·s
    # to damp outgoing waves and prevent spurious reflections.
    h_viscosity = Function(V, name="viscosity")
    h_viscosity.interpolate(max_value(1., 1000. * (1. - u / 2e4)))
    File(os.path.join(outputdir, 'viscosity.pvd')).write(h_viscosity)

    # -----------------------------------------------------------------
    # Step 2 — Manning friction field
    # -----------------------------------------------------------------
    # Two options controlled by inputs.use_friction_data:
    #   True  → read bed-classification data and map seabed type to a
    #            Manning n value via a lookup table (interp1d).
    #   False → use a spatially uniform background value (manning_bkg)
    #            and apply smooth tapering near artificial boundaries
    #            so the friction does not produce artefacts there.
    if inputs.use_friction_data:
        # Load the Manning lookup table: row 0 = class index, row 1 = n value
        manning_data = np.load(inputs.friction_data)
        interpolator = np.vectorize(interp1d(
            manning_data[0, :], manning_data[1, :],
            fill_value=(manning_data[1, 0], manning_data[1, -1]),
            bounds_error=False,
        ))
        manning_2d = bathymetry.get_manning_class(
            inputs.bed_classification_file, mesh, interpolator
        )
        # Smooth out sharp transitions between seabed classes
        bathymetry.smoothen_bathymetry(manning_2d)
    else:
        # Uniform Manning coefficient across the whole domain
        manning_2d = Function(V, name='manning').assign(inputs.manning_bkg)
        # Taper Manning toward 1.0 near the eastern open boundary (tags 100)
        # over the range 180–200 km to avoid discontinuities at the edge.
        fac1 = field_tools.transition_field(1.25, 1, ue, 180e3, 200e3)
        # Taper toward 1.0 near internal islands (tag 101) over 120–150 km
        fac2 = field_tools.transition_field(0.75, 1, ui, 120e3, 150e3)
        manning_2d.interpolate(manning_2d * fac1 * fac2)

    # Add localised Gaussian Manning bumps in Cardigan Bay.
    # These elevated friction patches suppress a known spurious tidal flux
    # that develops in the shallow, irregular bathymetry of Cardigan Bay.
    # Each entry in inputs.manning_gauss encodes:
    #   peak           — peak Manning n of the Gaussian
    #   (x0, y0, ang)  — centre coordinates (m, OSGB36) and rotation angle
    #   (sd1, sd2)     — standard deviations along principal axes (m)
    #   base           — baseline Manning value subtracted before adding
    #   (r1, r2)       — inner/outer taper radii for the Gaussian envelope
    x, y = SpatialCoordinate(mesh)
    for gauss in inputs.manning_gauss:
        peak, (x0, y0, ang), (sd1, sd2), base, (r1, r2) = gauss
        manning_2d.assign(manning_2d + field_tools.gaussian_hump(
            V, x, y, x0, y0, ang, peak - base, sd1, sd2, 0, r1, r2
        ))

    File(os.path.join(outputdir, 'manning.pvd')).write(manning_2d)
    print_output('Exported manning')

    # -----------------------------------------------------------------
    # Step 3 — Bathymetry interpolation
    # -----------------------------------------------------------------
    # Loop over the ordered list of bathymetric data sources.
    # Each entry is (filename, source_type, vertical_datum).
    # Later sources overwrite earlier ones where data overlap, allowing
    # high-resolution nearshore data to supersede coarser offshore grids.
    # If the datum is 'LAT', the pre-computed LAT-to-MSL correction (diff)
    # is added so all depths end up referenced to mean sea level.
    PETSc.Sys.Print(f"Bathymetric error = {inputs.bath_error} m")

    bath = None
    for f, source, datum in inputs.bathymetries:
        bath = bathymetry.get_bathymetry(
            f, mesh, source=source, bathymetry_function=bath,
            bathy_name="", h=inputs.bath_error,
        )
        if datum == 'LAT':
            # Convert LAT-referenced depths to MSL by adding the correction field
            bath.assign(bath + diff)

    if bath is None:
        raise RuntimeError("No bathymetry sources were provided in inputs.bathymetries.")

    # Smooth the interpolated bathymetry to remove point-scale noise that
    # could trigger instabilities in the finite-element solver.
    bathymetry.smoothen_bathymetry(bath)

    # Enforce a minimum water depth near the open boundary to prevent
    # wetting-and-drying oscillations in the forcing zone.
    # The threshold ramps linearly from 25 m at the boundary (u=0)
    # down to 0 m at 15 km inland (u=15000 m), then the hard floor
    # inputs.i_min_depth takes over everywhere else in the domain.
    bath.interpolate(max_value(
        (lambda bathy, dist: conditional(
            ge(bathy, 25. * (1. - dist / 15000.)),
            bathy, 25. * (1. - dist / 15000.)
        ))(bath, u),
        inputs.i_min_depth,
    ))

    File(os.path.join(outputdir, 'bath.pvd')).write(bath)

    # -----------------------------------------------------------------
    # Step 4 — Save checkpoint
    # -----------------------------------------------------------------
    # Write all preprocessed fields to a single HDF5 checkpoint file.
    # ramp.py and run.py read this file to initialise the flow solver,
    # so the mesh topology and all spatial fields are stored together.
    with CheckpointFile(os.path.join(inputdir, "preprocessing.h5"), "w") as CF:
        CF.save_mesh(mesh)
        CF.save_function(h_viscosity, name="viscosity")
        CF.save_function(bath, name="bathymetry")
        CF.save_function(manning_2d, name='manning')

    endtime = datetime.now()
    if MPI.COMM_WORLD.rank == 0:
        print('End time:', endtime.strftime("%d/%m/%Y %H:%M:%S"))
        print('Preprocessing time =', endtime - starttime)


if __name__ == '__main__':
    main()
