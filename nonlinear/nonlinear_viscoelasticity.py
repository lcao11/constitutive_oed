"""Nonlinear viscoelastic model definitions and utilities.

Usage:
    from nonlinear_viscoelasticity import NonlinearViscoElasticModel, NonlinearViscoElasticSettings
"""

import numpy as np
import dolfin as dl
import ufl
import os, sys
hippy_path = os.environ.get('HIPPYLIB_PATH')
if hippy_path and hippy_path not in sys.path:
    sys.path.append(hippy_path)
import hippylib as hp
gmc_path = os.environ.get('GMC_PATH')
if gmc_path and gmc_path not in sys.path:
    sys.path.append(gmc_path)
import geometric_mcmc as gmc
from utils import TimeDependentBoundaryCondition, ImageObservable, MaskedObservableMisfit, check_inclusion
from scipy.sparse import diags
import math

def NonlinearViscoElasticSettings():
    """
    Returns a dictionary with default settings for the nonlinear viscoelastic model.
    """
    settings = {}
    settings["cell_density"] = 36
    settings["aspect_ratio"] = 2.0
    settings["n_time_steps"] = 200
    settings["total_time"] = 1.0
    settings["n_prony"] = 2
    settings["seed"] = 0
    settings["pixel_density"] = 500
    settings["window_size"] = 5
    settings["high_resolution_factor"] = 1
    settings["n_image_snapshots"] = 20
    settings["n_force_data"] = 100
    settings["force_noise_std"] = 0.005
    settings["image_noise_std"] = 5.0
    settings["mask_threshold"] = 0.05
    settings["mask_steepness"] = 10
    settings["max_strain"] = 0.5
    settings["n_control_points"] = 10
    
    # Parameter ranges define the inverse-problem domain for the transformed
    # parameter vector m (used by set_bounds_for_parameters).
    # ensure Average Force ~ 1.0
    settings["log_shear_range"] = [-1.5, -0.5]
    
    settings["poisson_range"] = [0.45, 0.48]
    
    settings["eq_stiffness_log_ratio_range"] = [0.0, 3.0] 
    
    settings["eq_log_stiffening_range"] = [-1.0, 1.0] 
    
    settings["fiber_angle_range"] = [-0.25*math.pi, 0.25*math.pi]
    
    settings["viscous_shear_log_ratio_range"] = [-1.5, 0.7]
    settings["viscous_stiffness_log_ratio_range"] = [-1.5, 0.7]
    
    settings["viscous_stiffening_log_ratio_range"] = [-1.5, 0.7]
    settings["rate_exponent_range"] = [1.5, 3]
    settings["log_relaxation_increment"] = 1.5
    settings["log_relaxation_min"] = -3.0
    
    # Numerical regularization for fibers in compression (fraction of tensile stiffness)
    settings["compressive_regularization"] = 0.1

    # Control quadrature to avoid auto blow-up
    settings["quadrature_degree"] = 12
    return settings

def set_bounds_for_parameters(settings):
    """Sets the bounds for the model parameters based on the settings.

    These bounds define the inverse-problem search box for the transformed
    parameter vector m (before the erf-based mapping to physical variables).
    """
    dim_param = 5 + 5*settings["n_prony"]
    bounds = np.zeros((dim_param, 2))
    
    # Bounds for hyperelastic parameters
    bounds[0, :] = settings["log_shear_range"]
    bounds[1, :] = settings["poisson_range"]
    bounds[2, :] = settings["eq_stiffness_log_ratio_range"]
    bounds[3, :] = settings["eq_log_stiffening_range"]
    bounds[4, :] = settings["fiber_angle_range"]

    # Bounds for viscoelastic parameters (Prony series)
    lower_relaxation = settings["log_relaxation_min"]
    for ii in range(settings["n_prony"]):
        base_idx = 5 + ii * 5
        bounds[base_idx + 0, :] = settings["viscous_shear_log_ratio_range"]
        bounds[base_idx + 1, :] = settings["viscous_stiffness_log_ratio_range"]
        bounds[base_idx + 2, :] = settings["viscous_stiffening_log_ratio_range"]
        bounds[base_idx + 3, :] = settings["rate_exponent_range"]
        bounds[base_idx + 4, 0] = lower_relaxation
        lower_relaxation += settings["log_relaxation_increment"]
        bounds[base_idx + 4, 1] = lower_relaxation
    return bounds

def transform_parameter(m_input, bounds):
    """ Transform the input parameter to fit within specified bounds."""
    m = [None]*bounds.shape[0]
    for ii in range(bounds.shape[0]):
        m[ii] = dl.Constant(0.5*(bounds[ii, 1] - bounds[ii, 0]))*ufl.erf(m_input[ii]/dl.Constant(math.sqrt(2.0))) + dl.Constant(0.5*(bounds[ii, 1] + bounds[ii, 0]))
    return m

def extract_parameters(Vm, m_input, bounds):
    """ Extracts the parameters from the input vector `m_input` and transforms them"""
    m = transform_parameter(hp.vector2Function(m_input, Vm), bounds)
    m_vec = ufl.as_vector(m)
    return dl.project(m_vec, Vm).vector()

def macaulay(x, alpha=1e-4):
    """
    Creates a smooth, differentiable approximation of the Macaulay bracket,
    max(0, x), suitable for a Newton solver.
    Args:
        x: The UFL expression.
        alpha: A small parameter controlling the smoothness of the corner
               (a smaller alpha makes the corner sharper).
    """
    alpha_constant = dl.Constant(alpha)
    return dl.Constant(0.5) * (x + ufl.sqrt(x**dl.Constant(2.0) + dl.Constant(4.0) * alpha_constant**2))

class Energy:
    """ A class to compute the Equilibrium Energy and stress."""
    def __init__(self, shear, stiffness, stiffening, fiber_angle=None, bulk=None, structure_tensor=None, eta=0.05):
        self.shear = shear
        self.stiffness = stiffness
        self.stiffening = stiffening
        self.fiber_angle = fiber_angle
        self.bulk = bulk
        self.eta = eta # Regularization factor

        if structure_tensor is not None:
            self.structure_tensor = structure_tensor
        else:
            assert fiber_angle is not None, "Either fiber_angle or structure_tensor must be provided."
            a = ufl.as_vector((ufl.cos(fiber_angle), ufl.sin(fiber_angle)))
            self.structure_tensor = ufl.outer(a, a)

    def __call__(self, C):
        """
        Computes the energy as a sum of the isochoric, volumetric parts (if bulks is not None), and fiber stiffness part
        Parameters:
        -----------
        F : ufl.Form
            The deformation gradient tensor.
        Returns:
        --------
        ufl.Form
            The hyperelastic energy form.
        """
        dim = C.geometric_dimension()
        dim_constant = dl.Constant(dim)
        J   = ufl.sqrt(ufl.det(C))
        C_bar = J**(-dl.Constant(2.0/dim)) * C

        I1_bar = ufl.tr(C_bar)
        I4_bar = ufl.inner(C_bar, self.structure_tensor)

        psi_iso = (self.shear / dl.Constant(2.0)) * (I1_bar - dim_constant)

        if self.bulk is not None:
            psi_vol = (self.bulk / dl.Constant(2.0)) * (ufl.ln(J))**dl.Constant(2.0)
        else:
            psi_vol = 0.0

        # Tension part: Exponential stiffening (Standard HGO)
        I4_minus_1 = I4_bar - dl.Constant(1.0)
        psi_fiber_tension = (self.stiffness / (dl.Constant(2.0) * self.stiffening)) * (
            ufl.exp(self.stiffening*(macaulay(I4_minus_1))**dl.Constant(2.0)) - dl.Constant(1.0)
        )

        # Compression part: Linear stiffness (Regularization)
        # Use the parameterized eta (default 0.05)
        k_comp = self.stiffness * dl.Constant(self.eta) 
        psi_fiber_compression = (k_comp / dl.Constant(2.0)) * (macaulay(-I4_minus_1))**dl.Constant(2.0)

        return psi_iso + psi_vol + psi_fiber_tension + psi_fiber_compression

    def stress(self, C):
        """
        The stress for the hyperelastic energy
        Parameters:
        -----------
        u : A ufl vector for the displacement field.
        Returns:
        --------
        ufl.Form
            The force vector form.
        """
        C = ufl.variable(C)
        energy = self.__call__(C)
        return ufl.diff(energy, C)

class DualDissipativePotential:
    """ A class to compute the potential."""
    # Increased eps to 1e-8 to prevent Jacobian blow-up for p < 2 near zero stress
    def __init__(self, relaxation, shear, exponent, eps=1e-8):
        self.tau = relaxation
        self.mu_v = shear
        self.p = exponent
        self._eps = dl.Constant(eps)

    def __call__(self, Mv):
        A = ufl.dev(ufl.sym(Mv))
        nA = ufl.sqrt(ufl.inner(A, A) + self._eps)
        q = self.p + dl.Constant(1.0)
        gamma = dl.Constant(1.0)/(self.tau * self.mu_v)
        return gamma/q * nA**q

    def rate(self, Mv):
        A = ufl.dev(ufl.sym(Mv))
        nA = ufl.sqrt(ufl.inner(A, A) + self._eps)
        gamma = dl.Constant(1.0)/(self.tau * self.mu_v)
        return gamma * nA**(self.p - dl.Constant(1.0)) * A

def scalar_to_tensor(alphas, num_tensors):
    assert len(alphas) == num_tensors * 3, "The number of alphas must be a multiple of 3."
    tensors = []

    for ii in range(num_tensors):
        theta = alphas[3*ii]
        gamma = alphas[3*ii + 1]
        s = alphas[3*ii + 2]

        R = ufl.as_tensor(((ufl.cos(theta), -ufl.sin(theta)),
                   (ufl.sin(theta),  ufl.cos(theta))))
        S = ufl.as_tensor(((dl.Constant(1.0), gamma),
                            (dl.Constant(0.0), dl.Constant(1.0))))
        D = ufl.as_tensor(((ufl.exp(s), dl.Constant(0.0)),
                   (dl.Constant(0.0), ufl.exp(-s))))
        tensors.append(ufl.dot(ufl.transpose(R), ufl.dot(S, D)))
    return tensors

def parse_parameters(m, n_prony):
    """Parses the model parameters from the input vector `m`.

    Physics mapping:
    - Equilibrium response uses shear_eq, bulk_eq, stiffness_eq, stiffening_eq.
    - Viscous branches (Prony series) scale these with log ratios per branch.
    - Relaxation times are exponentiated to ensure positivity.
    """
    shear_eq = ufl.exp(m[0])
    # Minor Constant consistency
    bulk_eq = ufl.exp(m[0]) * dl.Constant(2.0) * (dl.Constant(1.0) + m[1]) / (
        dl.Constant(3.0) * (dl.Constant(1.0) - dl.Constant(2.0) * m[1])
    )
    stiffness_eq = ufl.exp(m[0] + m[2])
    stiffening_eq = ufl.exp(m[3])
    fiber_angle = m[4]
    relaxation = []
    shear_viscous = []
    stiffness_viscous = []
    stiffening_viscous = []
    rate_exponent = []
    for ii in range(n_prony):
        base_idx = 5 + ii * 5
        shear_viscous.append(ufl.exp(m[0] + m[base_idx]))
        stiffness_viscous.append(ufl.exp(m[0] + m[2] + m[base_idx + 1]))
        stiffening_viscous.append(ufl.exp(m[3] + m[base_idx + 2]))
        rate_exponent.append(m[base_idx + 3])
        relaxation.append(ufl.exp(m[base_idx + 4]))
    return (
        shear_eq, bulk_eq, stiffness_eq, stiffening_eq, fiber_angle,
        shear_viscous, stiffness_viscous, stiffening_viscous, rate_exponent, relaxation
    )

class nonlinear_viscoelastic_variational_form:
    """ 
    A class to define the variational form for the viscohyperelastic model
    """
    def __init__(self, settings, bounds):
        self.n_prony = settings["n_prony"]
        self.dt = settings["total_time"]/settings["n_time_steps"]
        self.dt_inv = dl.Constant(1.0/self.dt)
        self.bounds = bounds
        self.eta = settings.get("compressive_regularization", 0.05) # Get eta from settings
        qdeg = settings.get("quadrature_degree", 6)
        self.dx = dl.dx(metadata={"quadrature_degree": qdeg})
    
    def __call__(self, u_mixed, u_mixed_old, m, p_mixed, t):
        # Split state and test
        u, alphas         = dl.split(u_mixed)
        u_old, alphas_old = dl.split(u_mixed_old)
        p_u, p_alphas     = dl.split(p_mixed)

        # Parameters: transform from unconstrained to physical domain, then
        # assemble equilibrium and viscous parameters for each branch.
        m_transformed = transform_parameter(m, self.bounds)
        (shear_eq, bulk_eq, stiffness_eq, stiffening_eq, fiber_angle,
         shear_viscous, stiffness_viscous, stiffening_viscous,
         rate_exponent, relaxation) = parse_parameters(m_transformed, self.n_prony)

        # Kinematics: finite strain with deformation gradient F and right Cauchy-Green C.
        F = ufl.Identity(u.geometric_dimension()) + ufl.grad(u)
        C = ufl.dot(ufl.transpose(F), F)

        Fv_list     = scalar_to_tensor(alphas, self.n_prony)
        Fv_old_list = scalar_to_tensor(alphas_old, self.n_prony)
        p_Fv_list   = [ufl.derivative(Fv, alphas, p_alphas) for Fv in Fv_list]

        Fv_inv_list = [ufl.inv(Fv) for Fv in Fv_list]
        Ce_list     = [ufl.dot(ufl.transpose(Fvi), ufl.dot(C, Fvi)) for Fvi in Fv_inv_list]

        # Equilibrium energy: isotropic + fiber reinforcement (HGO-like).
        eq_energy = Energy(shear_eq, stiffness_eq, stiffening_eq, fiber_angle, bulk=bulk_eq, eta=self.eta)

        # Convected viscous fiber directions: transport fiber directions by Fv
        # to compute anisotropic viscous response in the current configuration.
        a0 = ufl.as_vector((ufl.cos(fiber_angle), ufl.sin(fiber_angle)))
        A_v_list = []
        for ii in range(self.n_prony):
            av_i_unnorm = ufl.dot(Fv_list[ii], a0)
            av_i_norm   = ufl.sqrt(ufl.inner(av_i_unnorm, av_i_unnorm) + dl.Constant(1e-12))
            av_i        = av_i_unnorm / av_i_norm
            A_v_i       = ufl.outer(av_i, av_i)
            A_v_list.append(A_v_i)

        neq_energy = [Energy(sh, ss, st, structure_tensor=A_v_i, eta=self.eta)
                      for sh, ss, st, A_v_i in
                      zip(shear_viscous, stiffness_viscous, stiffening_viscous, A_v_list)]

        # Dual dissipative potentials: define rate-dependent flow in viscous branches.
        dual_potential = [DualDissipativePotential(relax, sh, re)
                          for relax, sh, re in
                          zip(relaxation, shear_viscous, rate_exponent)]

        # Total PK2 stress: equilibrium part + viscous branch stresses pulled back
        # through the internal variables Fv.
        PK2 = 2 * eq_energy.stress(C)

        rate_list = []
        for ii in range(self.n_prony):
            # dΨ_i^neq/dC_i^e
            neq_stress = neq_energy[ii].stress(Ce_list[ii])

            # Add viscous branch ii to PK2:  S_neq^i = 2 Fv^{-1} (dW/dCe) Fv^{-T}
            PK2 += 2 * ufl.dot(Fv_inv_list[ii],
                               ufl.dot(neq_stress, ufl.transpose(Fv_inv_list[ii])))

            # Mandel stress and viscous velocity gradient: drive viscous flow.
            Mv_i = 2 * ufl.dot(Ce_list[ii], neq_stress)
            Lv_i = dual_potential[ii].rate(Mv_i)

            # Evolution rate: \dot{Fv_i} = Lv_i Fv_i
            rate_list.append(ufl.dot(Lv_i, Fv_list[ii]))

        PK1 = ufl.dot(F, PK2)

        # Residual form: momentum balance in weak form (1st Piola vs. grad test).
        varf = ufl.inner(PK1, ufl.grad(p_u)) * self.dx

        # Residuals for internal variables (backward Euler):
        # Fv_{n+1} - Fv_n - dt * Lv * Fv_{n+1} = 0
        dt = dl.Constant(self.dt)
        for ii in range(self.n_prony):
            res_alpha_ii = (Fv_list[ii] - Fv_old_list[ii]) - dt * rate_list[ii]
            varf += ufl.inner(res_alpha_ii, p_Fv_list[ii]) * self.dx

        return varf

class CustomNonlinearViscoElasticModel(hp.TimeDependentPDEVariationalProblem):
    """ A class to define the time-dependent model for the nonlinear viscoelastic problem."""
    def __init__(self, Vh, settings, loading_position, bounds):
        Vh = Vh
        settings = settings
        loading_position = loading_position
        bounds = bounds
        varf_handler = nonlinear_viscoelastic_variational_form(settings, bounds)
        right_boundary = dl.AutoSubDomain(lambda x: dl.near(x[0], 0.5*settings["aspect_ratio"]))
        left_boundary = dl.AutoSubDomain(lambda x: dl.near(x[0], -0.5*settings["aspect_ratio"]))
        bc_r = TimeDependentBoundaryCondition(Vh[hp.STATE].sub(0), loading_position, right_boundary)
        bc_r_0 = dl.DirichletBC(Vh[hp.STATE].sub(0), dl.Constant([0.0, 0.0]), right_boundary)
        bc_l_0 = dl.DirichletBC(Vh[hp.STATE].sub(0), dl.Constant([0.0, 0.0]), left_boundary)
        self.bc = [bc_r, bc_l_0]
        self.bc0 = [bc_r_0, bc_l_0]
        super().__init__(Vh, varf_handler, self.bc, self.bc0, dl.Function(Vh[hp.STATE]), 0, settings["total_time"], is_fwd_linear=False)
        self.parameters['nonlinear_solver'] = 'snes'
        self.parameters['snes_solver']['line_search'] = 'bt'
        self.parameters['snes_solver']['linear_solver'] = 'lu'
        self.parameters['snes_solver']['report'] = False
        self.parameters['snes_solver']['error_on_nonconvergence'] = True
        self.parameters['snes_solver']['absolute_tolerance'] = 1E-10
        self.parameters['snes_solver']['relative_tolerance'] = 1E-8
        self.parameters['snes_solver']['maximum_iterations'] = 1000
    
    def get_bc(self):
        """ Returns the initial boundary conditions for the problem."""
        return self.bc, self.bc0


class force_variational_form:
    def __init__(self, Vh, settings, bounds):
        self.Vh = Vh
        self.dt = settings["total_time"]/settings["n_time_steps"]
        self.n_prony = settings["n_prony"]
        self.dt_inv = dl.Constant(1.0/self.dt)
        self.eta = settings.get("compressive_regularization", 0.05) # Get eta from settings
        if isinstance(bounds, list):
            self.bounds = np.array(bounds)
        else:
            self.bounds = bounds
        right_boundary = dl.AutoSubDomain(lambda x: dl.near(x[0], 0.5*settings["aspect_ratio"]))
        mesh = Vh[hp.STATE].mesh()
        boundaries = dl.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        right_boundary.mark(boundaries, 1)
        qdeg = settings.get("quadrature_degree", 6)
        self.ds = dl.ds(subdomain_data=boundaries, metadata={"quadrature_degree": qdeg})
        self.normal = dl.FacetNormal(Vh[hp.STATE].mesh())
    
    def PK1(self, u_mixed, m):
        # Recompute the first Piola-Kirchhoff stress for traction output.
        # This mirrors the PDE constitutive evaluation but isolates traction on
        # the right boundary for force observations.
        u, alphas       = dl.split(u_mixed)

        m_transformed = transform_parameter(m, self.bounds)
        (shear_eq, bulk_eq, stiffness_eq, stiffening_eq, fiber_angle, \
        shear_viscous, stiffness_viscous, stiffening_viscous, rate_exponent, relaxation) = \
            parse_parameters(m_transformed, self.n_prony)

        F = ufl.Identity(u.geometric_dimension()) + ufl.grad(u)
        C = ufl.dot(ufl.transpose(F), F)

        Fv_list = scalar_to_tensor(alphas, self.n_prony)

        Fv_inv_list = [ufl.inv(Fv) for Fv in Fv_list]
        Ce_list     = [ufl.dot(ufl.transpose(Fvi), ufl.dot(C, Fvi)) for Fvi in Fv_inv_list]

        eq_energy = Energy(shear_eq, stiffness_eq, stiffening_eq, fiber_angle, bulk=bulk_eq, eta=self.eta)

        a0 = ufl.as_vector((ufl.cos(fiber_angle), ufl.sin(fiber_angle)))
        A_v_list = []
        for ii in range(self.n_prony):
            av_i_unnorm = ufl.dot(Fv_list[ii], a0)
            av_i_norm = ufl.sqrt(ufl.inner(av_i_unnorm, av_i_unnorm) + dl.Constant(1e-12))
            av_i = av_i_unnorm / av_i_norm
            A_v_i = ufl.outer(av_i, av_i)
            A_v_list.append(A_v_i)

        neq_energy = [Energy(sh, ss, st, structure_tensor=A_v_i, eta=self.eta)
                      for sh, ss, st, A_v_i in
                      zip(shear_viscous, stiffness_viscous, stiffening_viscous, A_v_list)]

        # PK2 stress is the energetic conjugate to C. Add viscous contributions
        # similarly to the PDE residual.
        PK2 = 2 * eq_energy.stress(C)

        for ii in range(self.n_prony):
            neq_stress = neq_energy[ii].stress(Ce_list[ii])
            PK2 += 2 * ufl.dot(Fv_inv_list[ii],
                               ufl.dot(neq_stress, ufl.transpose(Fv_inv_list[ii])))
        
        return ufl.dot(F, PK2)

    def __call__(self, u_mixed, m):
        PK1 = self.PK1(u_mixed, m)
        traction_vector = dl.dot(PK1, self.normal)
        return traction_vector[0] * self.ds(1)
    
def NonlinearViscoElasticFunctionSpace(mesh, settings):
    VCG_disp = dl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    VDG_int = dl.VectorElement("DG", mesh.ufl_cell(), 0, dim=settings["n_prony"]*3)
    ME = dl.MixedElement([VCG_disp, VDG_int])
    Vh_STATE = dl.FunctionSpace(mesh, ME)
    dim_param = 5*settings["n_prony"] + 5
    Vh_PARAMETER = dl.VectorFunctionSpace(mesh, 'R', 0, dim = dim_param)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE]
    return Vh

def generate_observables(Vh, settings, bounds, image_corners_coords, reference_image, reference_mask, targets, bc0):
    if isinstance(bounds, list):
        bounds = np.array(bounds)
    else:
        bounds = bounds
    assert bounds.shape[1] == 2 and bounds.shape[0] == 5 + 5*settings["n_prony"]
    force_varf = force_variational_form(Vh, settings, bounds)
    force_observables = gmc.VariationalQoiObservation(Vh, force_varf, bc0=bc0)
    image_observables = ImageObservable(Vh[hp.STATE], image_corners_coords, reference_image, reference_mask, 
                                        targets, components=np.array([0, 1]), bc0=bc0, 
                                        downsampling_factor=settings["high_resolution_factor"],
                                        window_size=settings["window_size"])
    return force_observables, gmc.MultipleObservations([force_observables, image_observables])

def NonlinearViscoElasticModel(mesh, settings, loading_position, image_corner_coords, reference_image, reference_mask, targets, prior_covariance=None):
    Vh = NonlinearViscoElasticFunctionSpace(mesh, settings)
    bounds = set_bounds_for_parameters(settings)
    # PDE operator with time-dependent loading; nonlinear solve uses SNES.
    pde = CustomNonlinearViscoElasticModel(Vh, settings, loading_position, bounds)
    bc, bc0 = pde.get_bc()
    if prior_covariance is None:
        prior_covariance =  np.diag(np.ones(Vh[hp.PARAMETER].dim()))
    # Gaussian prior over material parameters in the transformed space.
    # This prior couples with the PDE to define the Bayesian inverse problem.
    prior = hp.GaussianRealPrior(Vh[hp.PARAMETER], prior_covariance)

    assert settings["n_force_data"] % settings["n_image_snapshots"] == 0, "Number of force data points must be divisible by number of image snapshots."
    observation_times = np.linspace(0, settings["total_time"], settings["n_force_data"]+1)[1:]
    force_observable, joint_observable = generate_observables(Vh, settings, bounds, image_corner_coords, reference_image, reference_mask, targets, bc0=bc0)
    observables_list = []
    spacing = settings["n_force_data"] // settings["n_image_snapshots"]
    for ii, _ in enumerate(observation_times):
        if (ii + 1) % spacing == 0:
            observables_list.append(joint_observable)
        else:
            observables_list.append(force_observable)
    # Observations combine force data with image snapshots on a shared time grid.
    observables = gmc.TimeDependentObservations(observables_list, observation_times)
    # Misfit applies a soft image mask and diagonal noise model in data space.
    # This defines the likelihood term in the inverse problem.
    misfit = MaskedObservableMisfit(Vh, observables, mask_threshold=settings["mask_threshold"], steepness=settings["mask_steepness"])
    return hp.Model(pde, prior, misfit), observation_times, bc, bc0

def generate_data_idx(settings, reference_image_shape):
    """Generate indices matching the observable structure"""
    H, W = reference_image_shape
    f = settings.get("high_resolution_factor", 1)
    H_ds, W_ds = H // f, W // f
    image_size = H_ds * W_ds
    n_image_snapshots = settings["n_image_snapshots"]
    n_force_data = settings["n_force_data"]
    
    spacing = n_force_data // n_image_snapshots
    
    force_idx = []
    image_idx = []
    current_pos = 0
    
    for i in range(n_force_data):
        force_idx.append(current_pos)
        current_pos += 1
        
        if (i + 1) % spacing == 0:
            image_idx_start = current_pos
            image_idx_end = current_pos + image_size
            image_idx.extend(range(image_idx_start, image_idx_end))
            current_pos += image_size
    
    return np.array(force_idx), np.array(image_idx)

def generate_noise_model(settings, force_idx, image_idx):
    # Diagonal noise model for the inverse problem likelihood:
    # force and image channels can have different noise scales.
    noise_std_force = settings["force_noise_std"]
    noise_std_image = settings["image_noise_std"]
    noise_std_diag = np.zeros(force_idx.shape[0] + image_idx.shape[0])
    noise_std_diag[force_idx] = noise_std_force
    noise_std_diag[image_idx] = noise_std_image
    noise_precision = diags(1./noise_std_diag**2)
    return noise_precision

class CheckInsideImage:
    def __init__(self, aspect_ratio, stretch, rotation):
        """
        This function checks if a point is inside the reference material domain.
        Aspect ratio defines the aspect ratio of the material domain.
        Stretch and Rotation define the elliptical hole.
        image_length_y is the physical height of the image domain.
        """
        self.stretch = stretch
        self.rotation = rotation
        self.aspect_ratio = aspect_ratio

    def __call__(self, points):
        """
        Checks if the points are inside the reference material domain.
        """
        # Check if the translated points are within the mesh domain boundaries
        inside_full_domain = (points[:, 0] >= -0.5*self.aspect_ratio) & (points[:, 0] <= 0.5*self.aspect_ratio) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
        
        # Create a boolean mask. True means the point is in the material.
        # Initially, assume all points are in the material.
        inside_material_domain = np.ones(points.shape[0], dtype=bool)
        
        # Get indices of points that are inside the inclusion (the hole).
        indices_in_inclusion = check_inclusion(points, self.stretch, self.rotation)

        # Set these points to False, as they are not in the material.
        if indices_in_inclusion.size > 0:
            inside_material_domain[indices_in_inclusion] = False

        return inside_full_domain & inside_material_domain