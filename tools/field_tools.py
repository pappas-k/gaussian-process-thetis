# Auxiliary functions to generate fields in Thetis.

import numpy as np

from thetis import *
import inputs.simulation_parameters as inputs


def transition_field(v1, v2, dist, r1, r2):
    """ Create a transition field from one extreme value `v1` 
        at the edge `r1` to another extreme value `v2` at the edge `r2`. 
        `r1` and `r2` conform to the distance metric `dist` and
        `r1` < `r2`.
        :param v1: first extreme value (float)
        :param v2: second extreme value (float)
        :param dist: distance metric (float)
        :param r1: first distance (float)
        :param r2: second distance (float) > r1
        :return field: the transition field created
    """
    assert r1 < r2, f"Distance {r1} must be smaller than {r2}."
    return conditional( le(dist, r1), v1, 
                        conditional( ge(dist, r2), v2,
                                     v2 + (v1 - v2) * (r2 - dist) / (r2 - r1) ) )


def gaussian_patch_ridge_x(fnsp, x, y, y0, peak, std, base, xmin, xmax):
    """ Create a Gaussian patch with a ridge parallel to x-axis (y = const = y0),
        limited between `xmin` and `xmax`. 
        The ridge has a maximum value `peak` and minimum value
        cut-off at `base`. Its shape is Gaussian with a standard deviation
        (`std`). The formula is:
        f = exp[-0.5*(y-y0)^2/std^2] / [sqrt(2pi) * std]
        :param fnsp: function space (Thetis)
        :param x: x SpatialCoordinate of the mesh (Thetis)
        :param y: y SpatialCoordinate of the mesh (Thetis)
        :param y0: centre coordinate of the ridge (float)
        :param peak: maximum of Gaussian ridge (float)
        :param std: standard deviation of the Gaussian ridge (float)
        :param base: minimum level of Gaussian ridge (float)
        :param xmin: left/west bound of the ridge (float)
        :param xmax: right/east bound of the ridge (float)
        :return field: the patch field
    """
    assert xmin < xmax, f"Coordinate {xmin} must be smaller than {xmax}."
    assert base < peak, f"Base level must be lower than peak."
    field = Function(fnsp, name="gauss").assign(0.)
    scale = (peak - base) * (std * sqrt(2*np.pi))
    spread = std * sqrt(2*np.pi)
    norm = exp(-0.5*(((y - y0) / std)**2))
    field.interpolate(base + scale * norm / spread)
    field.interpolate(conditional(And(ge(x, xmin), le(x, xmax)), field, base) )
    return field

def gaussian_hump(fnsp, x, y, x0, y0, ang, peak, sd1, sd2, base=0, r1=None, r2=None):
    """ Create a 2-D Gaussian hump given a peak value, a base value,
        its centre location, rotation angle of the main axes,
        standard deviations along these main axes (`sd1` and `sd2`),
        and optionally, the ranges along these axes (`r1` and `r2`)
        that this Gaussian hump applied (hence cut-off beyond).

        f' = exp[-(x-x0)^2/2sd1^2-(y-y0)^2/2sd2^2] / [sqrt(2pi * (sd1^2 + sd2^2)) ]
        f = rotate(f', ang)

        Note that if either `sd1` or `sd2` = inf then
        the hump becomes a ridge.

        :param fnsp: function space (Thetis)
        :param x: x SpatialCoordinate of the mesh (Thetis)
        :param y: y SpatialCoordinate of the mesh (Thetis)
        :param x0: centre coordinate of the hump (float)
        :param y0: centre coordinate of the hump (float)
        :param ang: rotation angle of the hump (float)
        :param peak: maximum of Gaussian hump (float)
        :param sd1: standard deviation of the hump along 1st axis (float)
        :param sd2: standard deviation of the hump along 2nd axis (float)
        :param base: minimum level of Gaussian hump (float)
        :param r1: cut-off range along 1st axis (float)
        :param r2: cut-off range along 2nd axis (float)
        :return field: the Gaussian field (Thetis)
    """
    assert sd1 > 0 and sd2 > 0, f"Negative value not allowed among {sd1} {sd2}."
    assert base < peak, f"Base level must be lower than peak."
    field = Function(fnsp).assign(0.)
    if ang == 0 and sd2 == np.inf:  # ridge along y-direction
        scale = (peak - base) * (sd1 * sqrt(2*np.pi))
        spread = sd1 * sqrt(2*np.pi)
        norm = exp(-0.5*(((x - x0) / sd1)**2))
        field.interpolate(base + scale * norm / spread)
        field.interpolate(conditional(And(ge(y, y0-r2), le(y, y0+r2)), field, base) )
    elif ang == 90 and sd1 == np.inf:  # ridge along x-direction
        scale = (peak - base) * (sd2 * sqrt(2*np.pi))
        spread = sd2 * sqrt(2*np.pi)
        norm = exp(-0.5*(((y - y0) / sd2)**2))
        field.interpolate(base + scale * norm / spread)
        field.interpolate(conditional(And(ge(x, x0-r1), le(x, x0+r1)), field, base) )
    else:   # finite-extent hump
        x -= x0
        y -= y0
        ang_rad = np.radians(ang)
        x_ =  x*np.cos(ang_rad) + y*np.sin(ang_rad)
        y_ = -x*np.sin(ang_rad) + y*np.cos(ang_rad)
        x_ += x0
        y_ += y0
        sd = np.sqrt(sd1*sd1 + sd2*sd2)
        scale = (peak - base) * (sd * sqrt(2*np.pi))
        spread = sd * sqrt(2*np.pi)
        norm = exp(-(((x_ - x0) / 2 / sd1)**2) - (((y_ - y0) / 2 / sd2)**2))
        field.interpolate(base + scale * norm / spread)
        field.interpolate(conditional(And(ge(x_, x0-r1), le(x_, x0+r1)), field, base) )
        field.interpolate(conditional(And(ge(y_, y0-r2), le(y_, y0+r2)), field, base) )

    return field


def rect_patch(field, val, x, y, xmin, xmax, ymin, ymax):
    """ Create a rectangular patch with sides parallel to the axes. 
        :param field: input field (Thetis)
        :param val: value of the patch (float)
        :param x: x SpatialCoordinate of the mesh (Thetis)
        :param y: y SpatialCoordinate of the mesh (Thetis)
        :param xmin: left/west bound of the patch (float)
        :param xmax: right/east bound of the patch (float)
        :param ymin: lower/south bound of the patch (float)
        :param ymax: upper/north bound of the patch (float)
        :return: the patch field
    """
    field.interpolate(conditional(And(
                                  And( ge(x, xmin), le(x, xmax)), 
                                  And( ge(y, ymin), le(y, ymax) )),
                          val, field) )
    return field


def eik(fnsp, bnd_code, tol=1E-4, outfilename=None):
    """ Solve the Eikonal equation
        The solution can be used as a distance metric.
        :param fnsp: function space (Thetis)
        :param bnd_code: boundary code (int or list<int>)
        :param tol: tolerance of solution (float)
        :param outfilename: HDF5 file of Eikonal solution (str)
        :return: root of the Eikonal equation
    """
    u = Function(fnsp)
    v = TestFunction(fnsp)
    L = inputs.i_L
    if type(bnd_code) is int:
        bc = [DirichletBC(fnsp, 0.0, bnd_code)]
    elif type(bnd_code) is list:
        bc = [DirichletBC(fnsp, 0.0, i) for i in bnd_code]  # boundary conditions

    solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_rtol': tol,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_packages': 'mumps',
    }

    # Solve a Laplace eq to generate an initial guess for Eikonal eq
    F = L**2*(inner(grad(u), grad(v))) * dx - v * dx
    solve(F == 0, u, bc, solver_parameters=solver_parameters)
    solver_parameters = {
        'snes_type': 'newtonls',
        'ksp_rtol': tol,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_packages': 'mumps',
        }

    epss = inputs.i_epss
    for eps in epss:
        PETSc.Sys.Print("Solving Eikonal with eps == ", float(eps))
        F = inner(sqrt(inner(grad(u), grad(u))), v) * dx - v * dx + eps*inner(grad(u), grad(v)) * dx
        solve(F == 0, u, bc, solver_parameters=solver_parameters)
    
    if outfilename is not None:
        File(outfilename).write(u)
    
    return u
