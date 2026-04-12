"""Linear viscoelastic model definitions and utilities."""

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
from utils import TimeDependentBoundaryCondition, LinearTimeDependentPDEVariationalProblem, \
    ImageObservable, MaskedObservableMisfit, check_inclusion
from scipy.sparse import diags
import math

def ViscoElasticSettings():
    """
    Returns a dictionary with default settings for the viscoelastic model.
    """
    settings = {}
    settings["cell_density"] = 40
    settings["aspect_ratio"] = 2.0
    settings["n_time_steps"] = 200
    settings["total_time"] = 1.0
    settings["n_parameters"] = 11
    settings["seed"] = 0
    settings["pixel_density"] = 500
    settings["window_size"] = 3
    settings["high_resolution_factor"] = 1
    settings["n_image_snapshots"] = 20
    settings["n_force_data"] = 100
    settings["force_noise_std"] = 0.005
    settings["image_noise_std"] = 5.0
    settings["mask_threshold"] = 0.05
    settings["mask_steepness"] = 100
    settings["max_strain"] = 0.05
    settings["n_control_points"] = 10

    # Parameter ranges define the admissible inverse-problem domain for
    # the transformed parameter vector (used by set_bounds_for_parameters).
    settings["alpha_range"] = [-np.pi/4.0, np.pi/4.0]  # shared orientation
    settings["log_E1_0_range"] = [2.0, 3.5]            # natural log E1^(0)
    settings["rho_range"] = [0.2, 1.0]                 # E2^(0) / E1^(0)
    settings["g_range"] = [0.2, 0.8]                   # G / sqrt(E1 E2)
    settings["c_range"] = [0.0, 0.2]                   # nu12*nu21
    settings["f_range"] = [0.2, 0.8]                   # total viscous fractions
    settings["w_range"] = [0.0, 1.0]                   # relaxation time fractions
    settings["log_relaxation_min"] = -3.4
    settings["log_relaxation_increment"] = 2.2

    return settings

def set_bounds_for_parameters(settings):
    """Sets the bounds for the model parameters based on the settings.

    These bounds define the inverse-problem search box for the transformed
    parameter vector m (before the erf-based mapping to physical variables).
    """
    bounds = np.zeros((settings["n_parameters"], 2))

    # Bounds for hyperelastic parameters
    bounds[0, :] = settings["log_E1_0_range"]
    bounds[1, :] = settings["rho_range"]
    bounds[2, :] = settings["g_range"]
    bounds[3, :] = settings["c_range"]
    bounds[4, :] = settings["alpha_range"]
    bounds[5, :] = settings["f_range"]
    bounds[6, :] = settings["f_range"]
    bounds[7, :] = settings["w_range"]
    bounds[8, :] = settings["w_range"]


    # Bounds for viscoelastic parameters (Prony series)
    relaxation_time_min = settings["log_relaxation_min"]
    for ii in range(2):
        base_idx = 9 + ii
        bounds[base_idx, 0] = relaxation_time_min
        relaxation_time_min += settings["log_relaxation_increment"]
        bounds[base_idx, 1] = relaxation_time_min
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

def sigma(eps, params):
    """
    Plane-stress orthotropic stress (small strain) with rotation.
    Uses compliance-based construction:
      S = [[1/E1, nu21/E2, 0],
           [nu12/E1, 1/E2, 0],
           [0, 0, 1/G]]
      Q = S^{-1} with entries:
        D = 1 - nu12*nu21
        Q11 = E1 / D
        Q22 = E2 / D
        Q12 = nu12 * E2 / D  (== nu21 * E1 / D by reciprocity)
        Q66 = G

    Args
    ----
    eps : 2x2 UFL tensor
        Small-strain tensor, e.g. sym(grad(u)).
    params : list/tuple of 6 UFL expressions or dolfin.Constant
        [E1, E2, G, nu12, nu21, angle], with angle in radians.

    Returns
    -------
    sigma : 2x2 UFL tensor
        Cauchy stress (same as 2nd PK under small strain).
    """
    # Unpack in the same order produced by parse_parameters.
    # Physical meaning: E1, E2 are orthotropic Young's moduli in material axes,
    # G is in-plane shear modulus, nu12/nu21 are Poisson ratios, angle rotates
    # material axes into the global frame used by the mesh and displacement field.
    E1, E2, G, nu12, nu21, angle = params

    # In-plane rotation matrix (material axes rotated by +angle from global x)
    c, s = ufl.cos(angle), ufl.sin(angle)
    R = ufl.as_matrix(((c, -s),
                       (s,  c)))

    # Strain in material axes: small-strain tensor pulled back by rotation.
    # This aligns the constitutive response with the local fiber/material directions.
    eps_loc = R.T * eps * R
    e11 = eps_loc[0, 0]
    e22 = eps_loc[1, 1]
    e12 = eps_loc[0, 1]  # engineering shear: gamma12 = 2*e12

    # Orthotropic plane-stress stiffness from compliance.
    # We build the stiffness Q = S^{-1} for plane stress with reciprocity
    # (nu12/E1 == nu21/E2). This is the linear elastic constitutive map
    # in the material frame used to compute stress from strain.
    D = 1.0 - nu12*nu21
    Q11 = E1 / D
    Q22 = E2 / D
    Q12 = nu12 * E2 / D
    Q66 = G

    # Constitutive law in material axes (engineering shear convention).
    # For small strain, sigma_loc = Q : eps_loc with gamma12 = 2*e12.
    sigma_loc = ufl.as_matrix(((Q11*e11 + Q12*e22, 2.0*Q66*e12),
                               (2.0*Q66*e12,       Q12*e11 + Q22*e22)))

    # Rotate stress back to global axes for assembly in the global FEM coordinates.
    sigma = R * sigma_loc * R.T
    return ufl.sym(sigma)  # numerically enforce symmetry

def scalar_to_tensor(alphas):
    """
    Converts a long vector function to two symmetric 2nd order tensor functions.
    Parameters:
    -----------
    alphas : ufl.Form
        A vector function of the form [a1, a2].
    num_tensors : int
        The number of tensors to create.
    """
    tensors = []
    for ii in range(2):
        a1 = alphas[3*ii]
        a2 = alphas[3*ii + 1]
        a3 = alphas[3*ii + 2]
        tensor = ufl.as_tensor([[a1, a2], [a2, a3]])
        tensors.append(tensor)
    return tensors

def parse_parameters(m):
    """ Parses the model parameters from the input vector `m`.

    Ordering used everywhere (matches sigma()):
      Compliance = [E_1, E_2, G, nu_12, nu_21, angle]

    Construction:
    """
    # Parse the latent parameter vector into physical parameters.
    # We use exponentials for strictly positive quantities (moduli, time scales),
    # and algebraic transforms for bounded ratios (rho, g, c, w) defined in settings.
    Compliance = []
    Compliance.append(ufl.exp(m[0]))
    Compliance.append(Compliance[0] * m[1])  # E2 = rho * E1
    Compliance.append(m[2]*ufl.sqrt(Compliance[0] * Compliance[1]))  # G = g * sqrt(E1 E2)
    Compliance.append(ufl.sqrt(m[3] * Compliance[0]/Compliance[1]))  # nu12 from c
    Compliance.append(ufl.sqrt(m[3] * Compliance[1]/Compliance[0]))  # nu21 from c
    Compliance.append(m[4])  # material orientation (radians)

    # Two-branch Prony series: split total viscous fractions f and w across branches
    # using weights (m[7], m[8]) so each branch has its own stiffness and relaxation.
    Compliance_visco = []
    relaxation = []
    weights_1 = [m[7], dl.Constant(1.0)-m[7]]
    weights_2 = [m[8], dl.Constant(1.0)-m[8]]
    for ii in range(2):
        base_idx = 9 + ii
        Compliance_i = []
        Compliance_i.append(Compliance[0]*m[5]*weights_1[ii])  # E1^(i)
        Compliance_i.append(Compliance[1]*m[6]*weights_2[ii])  # E2^(i)
        Compliance_i.append(m[2]*ufl.sqrt(Compliance_i[0] * Compliance_i[1]))  # G
        Compliance_i.append(ufl.sqrt(m[3] * Compliance_i[0]/Compliance_i[1]))  # nu_12
        Compliance_i.append(ufl.sqrt(m[3] * Compliance_i[1]/Compliance_i[0]))  # nu_21
        Compliance_i.append(Compliance[5])  # angle (shared across branches)
        Compliance_visco.append(Compliance_i)
        relaxation.append(ufl.exp(m[base_idx]))
    return Compliance, Compliance_visco, relaxation

class viscoelastic_variational_form:
    """ 
    A class to define the variational form for the viscoelastic model
    """
    def __init__(self, settings, bounds):
        self.dt = settings["total_time"]/settings["n_time_steps"]
        self.dt_inv = dl.Constant(1.0/self.dt)
        self.bounds = bounds
    
    def __call__(self, u_mixed, u_mixed_old, m, p_mixed, t):

        # State decomposition: u is displacement, alphas are internal viscoelastic
        # strain-like variables (Prony branches) stored per element.
        u, alphas = dl.split(u_mixed)
        _, alphas_old = dl.split(u_mixed_old)
        p_u, p_alphas = dl.split(p_mixed)

        # Map unconstrained parameter vector to physical bounds.
        m_transformed = transform_parameter(m, self.bounds)
        Compliance, Compliance_visco, relaxation = parse_parameters(m_transformed)

        # Small-strain kinematics (plane stress): symmetric gradient of displacement.
        E = ufl.sym(ufl.grad(u))

        alphas_tensors = scalar_to_tensor(alphas)
        alphas_old_tensors = scalar_to_tensor(alphas_old)
        p_alphas_tensors = scalar_to_tensor(p_alphas)

        # Stress = elastic response + viscoelastic branch contributions.
        # Each branch uses a stress-like response based on E - alpha_i.
        S = sigma(E, Compliance)
        for ii in range(2):
            S += sigma(E - alphas_tensors[ii], Compliance_visco[ii])

        # Weak form (momentum balance): -div(S) = 0 with test function p_u.
        varf = ufl.inner(S, ufl.grad(p_u)) * ufl.dx
        for ii in range(2):
            # Backward-Euler update for internal variables:
            # (alpha_i - alpha_i_old)/dt = (E - alpha_i)/tau_i
            varf += self.dt_inv*ufl.inner(alphas_tensors[ii] - alphas_old_tensors[ii], p_alphas_tensors[ii]) * ufl.dx
            varf -= (1./relaxation[ii])*ufl.inner(E - alphas_tensors[ii], p_alphas_tensors[ii]) * ufl.dx
        return varf

class CustomViscoElasticModel(LinearTimeDependentPDEVariationalProblem):
    """ A class to define the time-dependent model for the viscoelastic problem."""
    def __init__(self, Vh, settings, loading_position, bounds):
        Vh = Vh
        settings = settings
        loading_position = loading_position
        bounds = bounds
        varf_handler = viscoelastic_variational_form(settings, bounds)
        right_boundary = dl.AutoSubDomain(lambda x: dl.near(x[0], 0.5*settings["aspect_ratio"]))
        left_boundary = dl.AutoSubDomain(lambda x: dl.near(x[0], -0.5*settings["aspect_ratio"]))
        bc_r = TimeDependentBoundaryCondition(Vh[hp.STATE].sub(0), loading_position, right_boundary)
        bc_r_0 = dl.DirichletBC(Vh[hp.STATE].sub(0), dl.Constant([0.0, 0.0]), right_boundary)
        bc_l_0 = dl.DirichletBC(Vh[hp.STATE].sub(0), dl.Constant([0.0, 0.0]), left_boundary)
        self.bc = [bc_r, bc_l_0]
        self.bc0 = [bc_r_0, bc_l_0]
        super().__init__(Vh, varf_handler, self.bc, self.bc0, dl.Function(Vh[hp.STATE]).vector(), 0, settings["total_time"])
    
    def get_bc(self):
        """ Returns the initial boundary conditions for the problem."""
        return self.bc, self.bc0

class force_variational_form:
    def __init__(self, Vh, settings, bounds):
        self.Vh = Vh
        self.dt = settings["total_time"]/settings["n_time_steps"]
        self.dt_inv = dl.Constant(1.0/self.dt)
        if isinstance(bounds, list):
            self.bounds = np.array(bounds)
        else:
            self.bounds = bounds
        right_boundary = dl.AutoSubDomain(lambda x: dl.near(x[0], 0.5*settings["aspect_ratio"]))
        mesh = Vh[hp.STATE].mesh()
        boundaries = dl.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        right_boundary.mark(boundaries, 1)
        self.ds = dl.ds(subdomain_data=boundaries)
        self.normal = dl.FacetNormal(Vh[hp.STATE].mesh())
    
    def compute_stress(self, u_mixed, m):

        # Recompute stress from the current state (u, alpha) and parameters m.
        u, alphas = dl.split(u_mixed)

        m_transformed = transform_parameter(m, self.bounds)
        Compliance, Compliance_visco, relaxation = parse_parameters(m_transformed)

        E = ufl.sym(ufl.grad(u))

        alphas_tensors = scalar_to_tensor(alphas)

        # Same constitutive model as the PDE residual: elastic + viscoelastic branches.
        S = sigma(E, Compliance)
        for ii in range(2):
            S += sigma(E - alphas_tensors[ii], Compliance_visco[ii])
        return S

    def __call__(self, u_mixed, m):

        stress_varf = self.compute_stress(u_mixed, m)

        traction_vector = dl.dot(stress_varf, self.normal)
        return traction_vector[0]*self.ds(1)
    
def ViscoelasticFunctionSpace(mesh, settings):
    VCG_disp = dl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    VDG_int = dl.VectorElement("DG", mesh.ufl_cell(), 0, dim=2*3)
    ME = dl.MixedElement([VCG_disp, VDG_int])
    Vh_STATE = dl.FunctionSpace(mesh, ME)
    Vh_PARAMETER = dl.VectorFunctionSpace(mesh, 'R', 0, dim = settings["n_parameters"])
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE]
    return Vh

def generate_observables(Vh, settings, bounds, image_corners_coords, reference_image, reference_mask, targets, bc0):
    if isinstance(bounds, list):
        bounds = np.array(bounds)
    else:
        bounds = bounds
    force_varf = force_variational_form(Vh, settings, bounds)
    # Force observation: traction integral on the loaded boundary.
    force_observables = gmc.VariationalQoiObservation(Vh, force_varf, bc0=bc0)
    # Image observation: maps displacement field to a deformed speckle image.
    image_observables = ImageObservable(Vh[hp.STATE], image_corners_coords, reference_image, reference_mask, targets, 
                                    components=np.array([0, 1]), bc0=bc0, 
                                    downsampling_factor=settings["high_resolution_factor"], 
                                    window_size=settings["window_size"])
    return force_observables, gmc.MultipleObservations([force_observables, image_observables])

def ViscoElasticModel(mesh, settings, loading_position, 
                      image_corner_coords, reference_image, reference_mask, targets, prior_covariance=None):

    Vh = ViscoelasticFunctionSpace(mesh, settings)
    bounds = set_bounds_for_parameters(settings)

    # Build PDE operator with time-dependent Dirichlet loading (right edge) and
    # fixed left edge. The loading_position acts as the imposed displacement history.
    pde = CustomViscoElasticModel(Vh, settings, loading_position, bounds)
    bc, bc0 = pde.get_bc()
    if prior_covariance is None:
        prior_covariance =  np.diag(np.ones(Vh[hp.PARAMETER].dim()))
    # Gaussian prior over parameters (log-moduli, ratios, viscosities, times).
    # This prior couples with the PDE to define the Bayesian inverse problem.
    prior = hp.GaussianRealPrior(Vh[hp.PARAMETER], prior_covariance)

    assert settings["n_force_data"] % settings["n_image_snapshots"] == 0, "Number of force data points must be divisible by number of DIC snapshots."
    observation_times = np.linspace(0, settings["total_time"], settings["n_force_data"]+1)[1:]
    force_observable, joint_observable = generate_observables(Vh, settings, bounds, image_corner_coords, reference_image, reference_mask, targets, bc0=bc0)
    observables_list = []
    spacing = settings["n_force_data"] // settings["n_image_snapshots"]
    for ii, _ in enumerate(observation_times):
        if (ii + 1) % spacing == 0:
            observables_list.append(joint_observable)
        else:
            observables_list.append(force_observable)
    # Observations combine force-only time steps with joint force+image snapshots.
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
    noise_std_diag = np.zeros(len(force_idx) + len(image_idx))
    noise_std_diag[force_idx] = noise_std_force
    noise_std_diag[image_idx] = noise_std_image
    noise_precision = diags(1./noise_std_diag**2)
    return noise_precision

class CheckInsideImage:
    def __init__(self, aspect_ratio, stretch, rotation):
        """Checks if points are inside the material domain."""
        self.stretch = stretch
        self.rotation = rotation
        self.aspect_ratio = aspect_ratio

    def __call__(self, points):
        inside_full_domain = (points[:, 0] >= -0.5*self.aspect_ratio) & (points[:, 0] <= 0.5*self.aspect_ratio) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
        inside_material_domain = np.ones(points.shape[0], dtype=bool)
        indices_in_inclusion = check_inclusion(points, self.stretch, self.rotation)

        if indices_in_inclusion.size > 0:
            inside_material_domain[indices_in_inclusion] = False

        return inside_full_domain & inside_material_domain