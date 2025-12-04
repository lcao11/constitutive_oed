import dolfin as dl
import hippylib as hp
import numpy as np
from scipy.special import expit
from geometric_mcmc.model.observable import Observable
from typing import List, Optional

class MaskedObservableMisfit(hp.Misfit):
    """
    Implements a misfit functional for an observable with a soft mask.
    
    This class handles cases where the data generation process involves a masking operation
    that depends on the observable itself. The data model is assumed to be:
    
        d = mask(u) * (u + eta)
        
    where:
        - u is the observable (prediction).
        - eta is Gaussian noise with precision P.
        - mask(u) is a sigmoid-based soft mask.
        
    The corresponding negative log-likelihood (cost functional) is:
    
        J(u) = 0.5 * || u - d/mask(u) ||_P^2 + sum(log(mask(u)))
        
    This formulation accounts for the signal-dependent noise variance introduced by the mask.
    """

    def __init__(self, 
                 Vh: List[dl.FunctionSpace], 
                 observable: Observable, 
                 dynamical_range: List[float] = [0, 255],
                 mask_threshold: float = 1e-6,
                 steepness: float = 1.e6,
                 data: Optional[np.ndarray] = None, 
                 noise_precision: Optional[np.ndarray] = None, 
                 check_mask_idx: Optional[np.ndarray] = None, 
                 ) -> None:
        """
        Initialize the MaskedObservableMisfit.

        Args:
            Vh (List[dl.FunctionSpace]): A list containing the function spaces for [state, parameter, adjoint].
            observable (Observable): The observable operator mapping state to observation space.
            dynamical_range (List[float], optional): The expected range [min, max] of the observable values, 
                used for normalizing the input to the sigmoid mask. Defaults to [0, 255].
            mask_threshold (float, optional): The normalized threshold value (0 to 1) where the mask 
                transitions from 0 to 1. Defaults to 1e-6.
            steepness (float, optional): Controls the sharpness of the sigmoid transition. 
                Higher values approximate a binary step function. Defaults to 1.e6.
            data (Optional[np.ndarray], optional): The observed data vector. Defaults to None.
            noise_precision (Optional[np.ndarray], optional): The diagonal of the noise precision matrix P 
                (inverse covariance). Defaults to None.
            check_mask_idx (Optional[np.ndarray], optional): Indices of the observable vector to apply 
                masking to. If None, applies to all. Defaults to None.
        """
        self.observable = observable
        self.Vh = Vh
        self.dynamical_range = dynamical_range
        self.mask_threshold = mask_threshold
        self.data = data
        self.noise_precision = noise_precision
        self.check_mask_idx = check_mask_idx
        self.steepness = steepness
        
        # Internal state for linearization
        self.gauss_newton_approx = True
        self.linearization_point = None
        self.W = None
    
    def generate_noisy_data(self, x: List[dl.Vector]) -> np.ndarray:
        """
        Generate synthetic noisy data based on the state x.
        
        Simulates: d = mask(obs(x)) * (obs(x) + noise)

        Args:
            x (List[dl.Vector]): The state list [u, m, p] (state, parameter, adjoint).

        Returns:
            np.ndarray: The generated noisy data vector.

        Raises:
            ValueError: If noise_precision is not set.
        """
        if self.noise_precision is None: 
            raise ValueError("Noise precision must be specified")
        
        obs = self.observable.eval(x)
        # noise_precision is assumed to be a sparse matrix or operator, but here we access diagonal
        # If it's a scipy sparse matrix, .diagonal() works.
        noise_std = 1.0/np.sqrt(self.noise_precision.diagonal())
        noise_sample = np.random.normal(0, noise_std, size=obs.shape)
        mask = self.generate_mask(obs)
        return (obs + noise_sample) * mask

    def cost(self, x: List[dl.Vector]) -> float:
        """
        Compute the negative log-likelihood cost functional.

        J(x) = 0.5 * (u - d/m)^T P (u - d/m) + sum(log(m))
        
        Args:
            x (List[dl.Vector]): The state list [u, m, p].

        Returns:
            float: The value of the cost functional.

        Raises:
            ValueError: If data or noise_precision are not set.
        """
        if self.noise_precision is None: 
            raise ValueError("Noise precision must be specified")
        if self.data is None:
            raise ValueError("Data must be specified")
        
        obs = self.observable.eval(x)  # Evaluate the observable at the state variable
        mask = self.generate_mask(obs)  # Generate the mask based on the observable
        
        # Avoid division by zero
        safe_mask = mask + 1e-12
        
        # Residual: u - d/m
        misfit_vec = obs - self.data / safe_mask
        
        # Quadratic term: 0.5 * r^T P r
        misfit_value = 0.5 * np.inner(misfit_vec, self.noise_precision.dot(misfit_vec))
        
        # Log-determinant term correction: sum(log(m))
        # This comes from the change of variables d -> eta or the effective covariance.
        misfit_value += np.sum(np.log(safe_mask))
        
        return misfit_value

    def generate_mask(self, obs: np.ndarray) -> np.ndarray:
        """
        Generate the soft mask based on the observable values.

        mask(u) = sigmoid( steepness * ( normalize(u) - threshold ) )

        Args:
            obs (np.ndarray): The observable vector.

        Returns:
            np.ndarray: The mask vector (values between 0 and 1).
        """
        if self.check_mask_idx is None:
            self.check_mask_idx = np.arange(obs.size)
        
        full_mask = np.ones(obs.shape, dtype=float)
        
        # Normalize observation to [0, 1] based on dynamical range
        obs_norm = (obs[self.check_mask_idx] - self.dynamical_range[0]) / (self.dynamical_range[1] - self.dynamical_range[0])
        
        # Apply sigmoid function
        z = self.steepness * (obs_norm - self.mask_threshold)
        obs_mask = expit(z)

        full_mask[self.check_mask_idx] = obs_mask
        return full_mask
    
    def grad(self, i: int, x: List[dl.Vector], out: dl.Vector) -> None:
        """
        Compute the gradient of the cost functional with respect to variable i.

        Args:
            i (int): The index of the variable to differentiate w.r.t. (e.g., STATE or PARAMETER).
            x (List[dl.Vector]): The state list [u, m, p].
            out (dl.Vector): The output vector to store the gradient.
        """
        if self.noise_precision is None: 
            raise ValueError("Noise precision must be specified")
        if self.data is None:
            raise ValueError("Data must be specified")
        
        out.zero()
        obs = self.observable.eval(x)
        mask = self.generate_mask(obs)
        
        # Add epsilon for numerical stability
        safe_mask = mask + 1e-12

        # 1. Gradient of the quadratic term w.r.t. obs (ignoring mask dependence for a moment)
        # d/du (0.5 * (u - d/m)^T P (u - d/m)) -> P * (u - d/m)
        direct_term = self.noise_precision.dot(obs - self.data/safe_mask)

        # 2. Gradient contributions from the mask dependence m(u)
        # Chain rule: dm/du = m * (1 - m) * steepness_scaled
        steepness_scaled = self.steepness / (self.dynamical_range[1] - self.dynamical_range[0])
        mprime = mask * (1.0 - mask) * steepness_scaled
        
        # Term from quadratic part: (P * (u - d/m)) * (d * m^{-2}) * m'
        # Term from log-det part: (1/m) * m'
        mask_sensitivity = ((direct_term * (self.data / (safe_mask**2))) + (1.0 / safe_mask)) * mprime

        # Total gradient w.r.t. observable
        grad_obs = direct_term + mask_sensitivity
        
        # Back-propagate through the observable operator: J_obs^T * grad_obs
        self.observable.jacobian_transpmult(x, i, grad_obs, out)
        
    def setLinearizationPoint(self, x: List[dl.Vector], gauss_newton_approx: bool = True) -> None:
        """
        Set the linearization point and precompute the Hessian (Fisher Information) weights.

        Args:
            x (List[dl.Vector]): The state list [u, m, p] at which to linearize.
            gauss_newton_approx (bool, optional): If True, computes the Fisher Information Matrix 
                (expected Hessian) which is guaranteed to be Positive Semi-Definite. 
                If False, would compute the full Hessian (not implemented). Defaults to True.

        Raises:
            NotImplementedError: If gauss_newton_approx is False.
        """
        self.gauss_newton_approx = gauss_newton_approx
        self.observable.setLinearizationPoint(x)
        self.linearization_point = x

        # Predicted observable μ and soft mask m(μ)
        obs = self.observable.eval(x)     # μ
        mask = self.generate_mask(obs)    # m(μ)

        # Numerically safe divisions
        eps = 1e-12
        safe_mask = mask + eps

        # Derivative dm/dμ (logistic with range scaling)
        range_width = (self.dynamical_range[1] - self.dynamical_range[0])
        steepness_scaled = self.steepness / range_width
        mprime = mask * (1.0 - mask) * steepness_scaled

        # Ratio t = (m'/m)
        t = mprime / safe_mask

        # Full Fisher Information weights (diagonal approximation in observable space):
        # The Fisher Information for the model d = m(u)(u + eta) is derived as:
        # F = P * (1 + u * m'/m)^2 + 2 * (m'/m)^2
        
        Pdiag = self.noise_precision.diagonal()
        
        # Contribution from the mean: (d_mean/du)^T P_y (d_mean/du)
        # mean = m*u => d_mean/du = m + u*m' = m(1 + u*t)
        # P_y = P / m^2
        # Term = (m(1+ut)) * (P/m^2) * (m(1+ut)) = P * (1+ut)^2
        W_mean = Pdiag * (1.0 + obs * t)**2
        
        # Contribution from the covariance: 0.5 * Tr( Sigma^-1 dSigma Sigma^-1 dSigma )
        # Sigma = m^2 P^-1 => dSigma/du = 2mm' P^-1
        # Term = 2 * (m'/m)^2 = 2 * t^2
        W_cov  = 2.0 * (t**2)
        
        W_fisher = W_mean + W_cov

        if gauss_newton_approx:
            self.W = W_fisher
        else:
            raise NotImplementedError("Full observed Hessian (non-PSD) is not implemented; use gauss_newton_approx=True for Fisher.")

    def apply_ij(self, i: int, j: int, dir: dl.Vector, out: dl.Vector) -> None:
        """
        Apply the Hessian action to a direction vector.
        
        Computes: out = J_obs^T * diag(W) * J_obs * dir
        
        Args:
            i (int): Row variable index (output space).
            j (int): Column variable index (input space).
            dir (dl.Vector): The direction vector in the input space.
            out (dl.Vector): The output vector in the output space.

        Raises:
            ValueError: If data or noise_precision are not set.
            NotImplementedError: If gauss_newton_approx was set to False.
        """
        if self.noise_precision is None: 
            raise ValueError("Noise precision must be specified")
        if self.data is None:
            raise ValueError("Data must be specified")
        if not getattr(self, "gauss_newton_approx", True):
            raise NotImplementedError("Full observed Hessian action not implemented; set gauss_newton_approx=True.")

        out.zero()

        # Forward sensitivity: J_obs * dir
        obs_help = self.observable.jacobian_mult(self.linearization_point, j, dir)

        # Apply Fisher weights in observable space
        weighted_obs_help = self.W * obs_help

        # Adjoint sensitivity: J_obs^T * (W * J_obs * dir)
        self.observable.jacobian_transpmult(self.linearization_point, i, weighted_obs_help, out)