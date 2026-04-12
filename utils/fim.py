"""FIM and Jacobian helper routines."""

import hippylib as hp
import geometric_mcmc as gmc
import numpy as np

def get_global_mv(comm, jacobian_mv):
    return np.vstack([gmc.get_global(comm, jacobian_mv[ii]) for ii in range(jacobian_mv.nvec())])

def compute_pto_map_jacobian(model, x):
    comm = model.problem.Vh[hp.STATE].mesh().mpi_comm()
    pto_jac = gmc.PtOMapJacobian(model.problem, model.misfit.observable)
    pto_jac.setLinearizationPoint(x)
    yhelp = model.problem.generate_parameter()
    out = np.zeros((model.misfit.observable.dim(), yhelp.size())) # Initialize the output
    for ii in range(yhelp.size()): # Loop over the parameter dimension
        unit_vec = np.zeros(yhelp.size()) # Initialize the unit vector
        unit_vec[ii] = 1. # Set the unit vector
        gmc.set_global(comm, unit_vec, yhelp) # Set the unit vector to a dolfin vector
        out[:, ii] = pto_jac.mult(yhelp) # Compute the Jacobian action
    return out

def compute_eigenvalues(model, x):
    model.misfit.setLinearizationPoint(x, gauss_newton_approx=True)
    jacobian = compute_pto_map_jacobian(model, x)
    H = np.einsum("ji, j, jk->ik", jacobian, model.misfit.W, jacobian)
    H = (H + H.T) / 2  # Make sure H is symmetric
    eigenvalues, _ = np.linalg.eigh(H)
    return eigenvalues

def compute_fim(model, x):
    model.misfit.setLinearizationPoint(x, gauss_newton_approx=True)
    jacobian = compute_pto_map_jacobian(model, x)
    FIM = np.einsum("ji, j, jk->ik", jacobian, model.misfit.W, jacobian)
    return FIM