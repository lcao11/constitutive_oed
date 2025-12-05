"""
Nested Monte Carlo (NMC) estimator for Expected Information Gain.

Seeding Strategy Summary:
-------------------------------------------
1. OUTER PARAMETERS: Single Sobol sequence with seed = base_seed
   - Use fast_forward(process_id * samples_per_process) to get correct slice
   - Same parameters across all designs for fair comparison
   - Maintains QMC properties across multiple process_ids

2. NOISE (generate noisy data): Seed = base_seed + 999999 + global_rank_id
   - Different noise realizations per rank and process
   - Offset to avoid collision with QMC seeds

3. INNER SAMPLES: Seed = base_seed + 2000000 + global_sample_id
   - Unique seed for every inner loop instance across all processes
   - Uses random_base2 for optimal QMC balance
"""

import os
import sys
import time
import pickle
import logging
import argparse

import numpy as np
import dolfin as dl
import jax
from scipy.interpolate import PchipInterpolator
from scipy.stats import qmc, norm
from mpi4py import MPI

# External library paths
hippy_path = os.environ.get('HIPPYLIB_PATH')
if hippy_path and hippy_path not in sys.path:
    sys.path.append(hippy_path)
import hippylib as hp

gmc_path = os.environ.get('GMC_PATH')
if gmc_path and gmc_path not in sys.path:
    sys.path.append(gmc_path)
import geometric_mcmc as gmc

# Project imports
sys.path.append("../../")
from utils import setup_image_observation, speckled_reference, compute_fim

sys.path.append("../")
from linear_viscoelasticity import (
    ViscoElasticModel, CheckInsideImage, generate_data_idx, generate_noise_model
)

# Configuration
jax.config.update("jax_enable_x64", True)
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)


def generate_outer_parameters(seed, param_dim, n_samples, skip=0):
    """
    Generate reproducible parameter samples using Sobol sequence.
    
    Args:
        seed: Random seed for Sobol scrambling
        param_dim: Dimension of parameter space
        n_samples: Number of samples to generate
        skip: Number of initial samples to skip (for distributed generation)
    
    Returns:
        Array of shape (n_samples, param_dim) with standard normal samples
    """
    # Validate and cast to int (scipy.stats.qmc requires int for fast_forward)
    skip = int(skip)
    n_samples = int(n_samples)
    
    if skip < 0:
        raise ValueError(f"skip must be non-negative, got {skip}")
    
    sobol = qmc.Sobol(d=param_dim, scramble=True, seed=seed)
    if skip > 0:
        sobol.fast_forward(skip)
    uniform_samples = sobol.random(n_samples)
    return norm.ppf(uniform_samples)


def fmt_time(seconds):
    """Format seconds as HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class NestedMonteCarlo:
    """Nested Monte Carlo estimator for Expected Information Gain."""
    
    def __init__(self, comm_mesh, comm_sampler, model_settings,
                 speckle_center, speckle_radius, image_corner_coords):
        self.comm_mesh = comm_mesh
        self.comm_sampler = comm_sampler
        self.model_settings = model_settings
        self.speckle_center = speckle_center
        self.speckle_radius = speckle_radius
        self.image_corner_coords = image_corner_coords
        
        # Sampling parameters
        self.outer_samples_per_process = 5
        self.num_inner_samples = 512  # Should be power of 2 for Sobol
        
        # Seeding
        self.base_seed = 0
        self.process_id = 0
        
        # Progress tracking
        self.verbose = True
        self._t_start = None

    def _log(self, msg):
        """Print message on rank 0 only."""
        if self.verbose and self.comm_sampler.rank == 0:
            print(msg, flush=True)

    def _create_model(self, mesh, loading_position, reference_image, reference_mask, targets):
        """Initialize the viscoelastic model."""
        model, _, _, _ = ViscoElasticModel(
            mesh, self.model_settings, loading_position,
            self.image_corner_coords, reference_image, reference_mask, targets
        )
        return model

    def _interpolate_loading(self, control_values):
        """Create smooth loading path from control points."""
        n_pts = self.model_settings["n_control_points"]
        assert len(control_values) == n_pts
        
        times = np.linspace(0.0, self.model_settings["total_time"], n_pts + 1)
        values = np.zeros((n_pts + 1, 2))
        values[1:, 0] = control_values
        return PchipInterpolator(times, values)

    def _inner_loop(self, model, parameter, FIM, seed):
        """
        Importance sampling from approximate posterior q = N(parameter, (I+FIM)^{-1}).
        
        Returns:
            log_likelihoods: Array of log p(y|theta') for each inner sample
            log_weights: Array of log importance weights
        """
        d = FIM.shape[0]
        
        # Posterior precision and its Cholesky factor
        post_prec = np.eye(d) + FIM
        post_prec = 0.5 * (post_prec + post_prec.T)  # Symmetrize
        post_prec_chol = np.linalg.cholesky(post_prec)
        
        _, logdet_post_prec = np.linalg.slogdet(post_prec)
        logdet_post_cov = -logdet_post_prec
        
        # Generate samples via Sobol
        n_samples = self.num_inner_samples
        m_log2 = int(np.ceil(np.log2(n_samples)))
        n_samples = 2 ** m_log2  # Round up to power of 2
        
        sobol = qmc.Sobol(d=d, scramble=True, seed=seed)
        z_all = norm.ppf(sobol.random_base2(m_log2))
        
        log_likelihoods = np.empty(n_samples)
        log_weights = np.empty(n_samples)
        x_sample = model.generate_vector()

        # Progress control (rank 0 only)
        rank = self.comm_sampler.rank
        t0_inner = time.time()
        # Print about 10 updates over the inner loop, at least every sample if very small
        print_every = max(1, n_samples // 100)
        
        for ii in range(n_samples):
            # Sample: theta' = theta + (I + FIM)^{-1/2} * z
            y = np.linalg.solve(post_prec_chol.T, z_all[ii])
            theta_prime = parameter + y
            
            # Evaluate model
            gmc.set_global(self.comm_mesh, theta_prime, x_sample[hp.PARAMETER])
            model.solveFwd(x_sample[hp.STATE], x_sample)
            _, neg_log_prior, neg_log_likelihood = model.cost(x_sample)
            
            # Importance weight: log p(theta') - log q(theta'|theta)
            diff = theta_prime - parameter
            log_q = -0.5 * logdet_post_cov - 0.5 * diff @ (post_prec @ diff)
            
            log_likelihoods[ii] = -neg_log_likelihood
            log_weights[ii] = -neg_log_prior - log_q

            # Rank-0 progress print
            if rank == 0 and ((ii + 1) % print_every == 0 or ii == n_samples - 1):
                elapsed = time.time() - t0_inner
                done = ii + 1
                avg_per = elapsed / done
                eta = avg_per * (n_samples - done)
                self._log(
                    f"[NMC][Inner] {done}/{n_samples} "
                    f"elapsed {fmt_time(elapsed)}, ETA {fmt_time(eta)}"
                )
        
        return log_likelihoods, log_weights

    def run(self, mesh, stretch, rotation, control_values, outer_parameters):
        """
        Run nested Monte Carlo estimation.
        
        Args:
            mesh: FEniCS mesh
            stretch: Tuple (sx, sy) for specimen geometry
            rotation: Rotation angle for specimen
            control_values: Loading control points
            outer_parameters: Pre-generated parameter samples, shape (n_outer, param_dim)
        
        Returns:
            results: Dict with info_gain_gauss, outer_log_likelihood, 
                     inner_log_likelihood, inner_log_weights, outer_parameters
        """
        self._t_start = time.time()
        n_outer = self.outer_samples_per_process
        n_inner = self.num_inner_samples
        
        # Round inner samples to power of 2
        m_log2 = int(np.ceil(np.log2(n_inner)))
        n_inner = 2 ** m_log2
        self.num_inner_samples = n_inner
        
        self._log(f"[NMC] Starting: {n_outer} outer x {n_inner} inner samples")
        
        # Setup observation model
        loading = self._interpolate_loading(control_values)
        inside_check = CheckInsideImage(self.model_settings["aspect_ratio"], stretch, rotation)
        
        reference_mask, targets = setup_image_observation(
            self.image_corner_coords, inside_check,
            self.model_settings["pixel_density"],
            oversampling_factor=self.model_settings["high_resolution_factor"]
        )
        reference_image = speckled_reference(
            reference_mask, self.image_corner_coords,
            self.speckle_center, self.speckle_radius
        )
        
        model = self._create_model(mesh, loading, reference_image, reference_mask, targets)
        
        # Setup noise model
        force_idx, image_idx = generate_data_idx(self.model_settings, reference_image.shape)
        model.misfit.check_mask_idx = image_idx
        model.misfit.noise_precision = generate_noise_model(self.model_settings, force_idx, image_idx)
        
        # Allocate output arrays
        info_gain_gauss = np.empty(n_outer)
        outer_log_likelihood = np.empty(n_outer)
        inner_log_likelihood = np.empty((n_outer, n_inner))
        inner_log_weights = np.empty((n_outer, n_inner))
        
        x_true = model.generate_vector()
        rank = self.comm_sampler.rank
        
        for i in range(n_outer):
            t_outer = time.time()
            
            # Set outer parameter
            theta = outer_parameters[i]
            gmc.set_global(self.comm_mesh, theta, x_true[hp.PARAMETER])
            model.solveFwd(x_true[hp.STATE], x_true)
            
            # Compute FIM
            model.misfit.setLinearizationPoint(x_true, gauss_newton_approx=True)
            FIM = compute_fim(model, x_true)
            FIM = 0.5 * (FIM + FIM.T)
            
            # Gaussian approximation to info gain
            eigenvalues = np.maximum(np.linalg.eigvalsh(FIM), 0.0)
            gauss_val = (np.sum(np.log1p(eigenvalues)) 
                        - np.sum(eigenvalues / (1.0 + eigenvalues))
                        + x_true[hp.PARAMETER].inner(x_true[hp.PARAMETER]))
            info_gain_gauss[i] = 0.5 * gauss_val
            
            # Generate synthetic data
            data = model.misfit.generate_noisy_data(x_true)
            model.misfit.data = data
            outer_log_likelihood[i] = -model.cost(x_true)[2]
            
            # Inner loop with unique seed
            # FIX: Robust seed formula to ensure "QMC within MC" works correctly.
            # 1. Offset by 2000000 to separate from Noise and Outer QMC seeds.
            # 2. Use global linear indexing so EVERY inner loop in the entire job array is unique.
            global_rank_idx = self.process_id * self.comm_sampler.size + rank
            global_sample_idx = global_rank_idx * n_outer + i
            inner_seed = int(self.base_seed + 2000000 + global_sample_idx)
            
            inner_ll, inner_w = self._inner_loop(model, theta, FIM, inner_seed)
            
            inner_log_likelihood[i] = inner_ll
            inner_log_weights[i] = inner_w
            
            elapsed = time.time() - self._t_start
            dt = time.time() - t_outer
            eta = (elapsed / (i + 1)) * (n_outer - i - 1)
            self._log(f"[NMC] Outer {i+1}/{n_outer}: {dt:.1f}s (elapsed {fmt_time(elapsed)}, ETA {fmt_time(eta)})")
        
        return {
            "info_gain_gauss": info_gain_gauss,
            "outer_log_likelihood": outer_log_likelihood,
            "inner_log_likelihood": inner_log_likelihood,
            "inner_log_weights": inner_log_weights,
            "outer_parameters": outer_parameters.copy()
        }


def main():
    parser = argparse.ArgumentParser(description="Nested Monte Carlo for EIG estimation")
    parser.add_argument("--output_path", type=str, default="./nmc_results/")
    parser.add_argument("--design_index", type=int, default=0)
    parser.add_argument("--inner_samples", type=int, default=512)
    parser.add_argument("--outer_samples_per_process", type=int, default=2)
    parser.add_argument("--process_id", type=int, default=0)
    args = parser.parse_args()
    
    # Validate arguments
    if args.process_id < 0:
        raise ValueError(f"--process_id must be non-negative, got {args.process_id}")
    if args.outer_samples_per_process <= 0:
        raise ValueError(f"--outer_samples_per_process must be positive, got {args.outer_samples_per_process}")
    if args.inner_samples <= 0:
        raise ValueError(f"--inner_samples must be positive, got {args.inner_samples}")
    
    # MPI setup
    comm_mesh, comm_sampler = gmc.split_mpi_comm(MPI.COMM_WORLD, 1, MPI.COMM_WORLD.size)
    rank = comm_sampler.rank
    size = comm_sampler.size
    
    # Create output directory
    output_dir = os.path.join(args.output_path, f"design_{args.design_index}")
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Main] Output: {output_dir}", flush=True)
    comm_sampler.Barrier()
    
    # Load configuration
    with open("../model_config.pkl", "rb") as f:
        config = pickle.load(f)
        model_settings = config["model_settings"]
        speckle_center = config["speckle_centers"]
        speckle_radius = config["speckle_radii"]
        image_corner_coords = config["image_corners_coords"]
    
    with open("./settings/settings.pkl", "rb") as f:
        design_samples = pickle.load(f)["design_samples"]
    
    base_seed = model_settings["seed"]
    
    # Initialize estimator
    estimator = NestedMonteCarlo(
        comm_mesh, comm_sampler, model_settings,
        speckle_center, speckle_radius, image_corner_coords
    )
    estimator.outer_samples_per_process = args.outer_samples_per_process
    estimator.num_inner_samples = args.inner_samples
    estimator.base_seed = base_seed
    estimator.process_id = args.process_id
    
    # Seed for noise generation (varies by process, rank)
    # FIX: Use a robust formula to avoid collisions.
    # Offset by 999999 to separate from QMC seeds.
    # Use (process_id * size + rank) to ensure every rank has a unique ID.
    noise_seed = int(base_seed + 999999 + args.process_id * size + rank)
    np.random.seed(noise_seed)
    
    # Generate outer parameters using SINGLE Sobol sequence with fast_forward
    total_outer = size * args.outer_samples_per_process  # Both are ints, so this is int
    param_dim = int(model_settings["n_parameters"])  # Ensure int
    outer_seed = int(base_seed)  # Ensure int
    skip_count = args.process_id * total_outer  # int * int = int
    
    if rank == 0:
        # Validate before calling
        if skip_count < 0:
            raise ValueError(f"skip_count must be non-negative, got {skip_count} "
                             f"(process_id={args.process_id}, total_outer={total_outer})")
        
        all_params = generate_outer_parameters(outer_seed, param_dim, total_outer, skip=skip_count)
        all_params = all_params.reshape(size, args.outer_samples_per_process, param_dim)
        print(f"[Main] Generated outer params [{skip_count}:{skip_count + total_outer}] "
              f"with seed={outer_seed}", flush=True)
    else:
        all_params = None
    
    local_params = np.empty((args.outer_samples_per_process, param_dim))
    comm_sampler.Scatter(all_params, local_params, root=0)
    
    # Load design and mesh
    design = design_samples[args.design_index]
    rotation = design[0]
    stretch = (0.35, design[1])
    control_values = design[2:]
    mesh = dl.Mesh(comm_mesh, f"./settings/mesh_{args.design_index}.xml")
    
    # Run estimation
    local_results = estimator.run(mesh, stretch, rotation, control_values, local_params)
    
    # Gather results
    def gather_array(arr):
        if rank == 0:
            shape = (size,) + arr.shape
            buf = np.empty(shape)
        else:
            buf = None
        comm_sampler.Gather(arr, buf, root=0)
        return buf
    
    all_results = {k: gather_array(v) for k, v in local_results.items()}
    
    # Save results
    if rank == 0:
        results = {
            "design": design,
            "design_index": args.design_index,
            "process_id": args.process_id,
        }
        for k, v in all_results.items():
            if v.ndim == 2:
                results[k] = v.reshape(-1)
            elif v.ndim == 3:
                results[k] = v.reshape(-1, v.shape[-1])
            else:
                results[k] = v
        
        out_file = os.path.join(output_dir, f"results_{args.process_id}.pkl")
        with open(out_file, "wb") as f:
            pickle.dump(results, f)
        
        elapsed = time.time() - estimator._t_start
        print(f"[Main] Done in {fmt_time(elapsed)}. Saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()