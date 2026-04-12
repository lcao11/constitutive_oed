"""
Nested Monte Carlo (NMC) estimator for Expected Information Gain.

Seeding Strategy Summary:
-------------------------------------------
1. OUTER PARAMETERS: Single Sobol sequence with seed = base_seed
   - Use fast_forward(process_id * total_outer) to get correct slice,
    where total_outer = (mpi_size * outer_samples_per_process)
   - Same parameters across all designs for fair comparison
   - Maintains QMC properties across multiple process_ids

2. NOISE (generate noisy data): Seed = base_seed + 999999 + global_rank_id
   - Different noise realizations per rank and process
   - Offset to avoid collision with QMC seeds

3. INNER SAMPLES: Seed = base_seed + 2000000 + global_sample_id
   - Unique seed for every inner loop instance across all processes
   - Uses random_base2 for optimal QMC balance
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ufl")
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
from scipy.stats import qmc, norm, chi2
from scipy.special import gammaln
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
from utils import setup_image_observation, speckled_reference, compute_fim, BFGS, RescaledIdentity

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
    # Sobol can return exact 0.0 (and in some implementations values extremely close to 1.0).
    # Protect against +/-inf from the inverse CDF.
    eps = 1e-12
    uniform_samples = np.clip(uniform_samples, eps, 1.0 - eps)
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
        self.proposal_df = 0  # 0 = Gaussian, >0 = Student-t with this many DOF
        
        # Inner sample chunking (for distributed inner sampling across jobs)
        self.inner_chunk_id = 0
        self.num_inner_chunks = 1
        
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

    def _inner_loop(self, model, parameter, FIM, seed, chunk_id=0, chunk_size=None):
        """
        Importance sampling from Laplace-approximated posterior.

        Proposal: q = N(m_loc, Sigma)       when proposal_df == 0 (Gaussian)
                  q = t_nu(m_loc, Sigma)     when proposal_df > 0 (Student-t)

        where Sigma = (I + FIM)^{-1} and m_loc = theta - Sigma * theta is the
        posterior mode under the Laplace approximation with a N(0, I) prior.
        
        When chunk_size is set, only generates samples [chunk_id*chunk_size : (chunk_id+1)*chunk_size]
        from the full Sobol sequence, using fast_forward for correct positioning.
        """
        d = FIM.shape[0]
        use_t = self.proposal_df > 0
        nu = float(self.proposal_df) if use_t else None

        # Log diagnostics (only on first chunk to avoid spam)
        if self.verbose and self.comm_sampler.rank == 0 and chunk_id == 0:
            eigvals = np.maximum(np.linalg.eigvalsh(FIM), 0.0)
            eig_gauss = 0.5 * np.sum(np.log1p(eigvals))
            proposal_tag = f"t_{int(nu)}" if use_t else "gauss"
            self._log(f"[NMC][Inner] EIG_gauss={eig_gauss:.2f}, proposal={proposal_tag}")

        # Posterior precision and its Cholesky factor
        post_prec = np.eye(d) + FIM
        post_prec = 0.5 * (post_prec + post_prec.T)  # Symmetrize

        try:
            post_prec_chol = np.linalg.cholesky(post_prec)
        except np.linalg.LinAlgError:
            jitter = 1e-10
            for _ in range(8):
                try:
                    post_prec = post_prec + jitter * np.eye(d)
                    post_prec_chol = np.linalg.cholesky(post_prec)
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0
            else:
                raise

        _, logdet_post_prec = np.linalg.slogdet(post_prec)
        logdet_post_cov = -logdet_post_prec  # log|Sigma| = -log|Sigma^{-1}|

        # Mean shift for Laplace approximation
        # The proposal should be centered at the mode of the posterior approximation.
        # m_loc = theta + (FIM + I)^{-1} * grad_log_prior
        # With Standard Normal prior: grad_log_prior = -theta
        # m_loc = theta - (FIM + I)^{-1} * theta
        # This corrects for the prior pulling the estimate towards zero.
        mean_shift = -np.linalg.solve(post_prec, parameter)
        proposal_mean = parameter + mean_shift

        # Generate QMC samples (with optional chunking via generate-and-slice)
        # When chunking, we generate the full Sobol sequence for total_K and slice
        # to the chunk. Memory is trivial (K*d*8 bytes ≈ 1.4 MB for K=16384, d=11).
        total_K = self.num_inner_samples
        m_log2_total = int(np.ceil(np.log2(total_K)))
        if chunk_size is None:
            chunk_size = total_K
        n_samples = chunk_size

        if use_t:
            # Student-t: d+1 Sobol dims (d for normal component, 1 for chi2 scaling)
            sobol = qmc.Sobol(d=d + 1, scramble=True, seed=seed)
            uniform_full = sobol.random_base2(m_log2_total)
            uniform_all = uniform_full[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]
            eps = 1e-12
            uniform_all = np.clip(uniform_all, eps, 1.0 - eps)
            z_all = norm.ppf(uniform_all[:, :d])
            chi2_samples = chi2.ppf(uniform_all[:, d], df=nu)
            scale_all = np.sqrt(nu / chi2_samples)
        else:
            sobol = qmc.Sobol(d=d, scramble=True, seed=seed)
            z_full = norm.ppf(sobol.random_base2(m_log2_total))
            z_all = z_full[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]

        # Precompute normalization constants for correct importance weights.
        # Including full constants (not just kernel) is required so that
        # the SNIS estimator correctly estimates the marginal likelihood.
        log_prior_const = -0.5 * d * np.log(2.0 * np.pi)
        if use_t:
            log_q_const = (gammaln((nu + d) / 2.0) - gammaln(nu / 2.0)
                           - 0.5 * d * np.log(nu * np.pi)
                           + 0.5 * logdet_post_prec)
        else:
            log_q_const = -0.5 * d * np.log(2.0 * np.pi) + 0.5 * logdet_post_prec

        log_likelihoods = np.empty(n_samples)
        log_weights = np.empty(n_samples)
        x_sample = model.generate_vector()

        # Progress control (rank 0 only)
        rank = self.comm_sampler.rank
        t0_inner = time.time()
        print_every = max(1, n_samples // 100)

        for ii in range(n_samples):
            # Sample: theta' = m_loc + [scale_i *] (I + FIM)^{-1/2} z_i
            y = np.linalg.solve(post_prec_chol.T, z_all[ii])
            if use_t:
                theta_prime = proposal_mean + scale_all[ii] * y
            else:
                theta_prime = proposal_mean + y

            # Evaluate model
            gmc.set_global(self.comm_mesh, theta_prime, x_sample[hp.PARAMETER])

            try:
                model.solveFwd(x_sample[hp.STATE], x_sample)
                _, neg_log_prior, neg_log_likelihood = model.cost(x_sample)

                # Importance weight: log p(theta') - log q(theta')
                diff = theta_prime - proposal_mean
                mahal_sq = diff @ (post_prec @ diff)

                log_prior_val = log_prior_const - neg_log_prior
                if use_t:
                    log_q = log_q_const - ((nu + d) / 2.0) * np.log(1.0 + mahal_sq / nu)
                else:
                    log_q = log_q_const - 0.5 * mahal_sq

                log_likelihoods[ii] = -neg_log_likelihood
                log_weights[ii] = log_prior_val - log_q

            except Exception as e:
                # Handle solver failures gracefully
                log_likelihoods[ii] = -1e10
                log_weights[ii] = -1e10
                if self.verbose and rank == 0:
                    print(f"[NMC][Inner] Sample {ii} failed: {e}", flush=True)

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
        t_start_perf = time.perf_counter()
        n_outer = self.outer_samples_per_process
        n_inner = self.num_inner_samples
        
        # Round inner samples to power of 2
        m_log2 = int(np.ceil(np.log2(n_inner)))
        n_inner = 2 ** m_log2
        self.num_inner_samples = n_inner
        
        # Chunk computation for distributed inner sampling across jobs
        chunk_id = self.inner_chunk_id
        num_chunks = self.num_inner_chunks
        chunk_size = n_inner // num_chunks
        assert n_inner % num_chunks == 0, (
            f"n_inner={n_inner} not divisible by num_chunks={num_chunks}")
        assert chunk_size & (chunk_size - 1) == 0, (
            f"chunk_size={chunk_size} must be power of 2 for Sobol")
        
        self._log(f"[NMC] Starting: {n_outer} outer x {chunk_size} inner samples "
                  f"(chunk {chunk_id}/{num_chunks}, total K={n_inner})")
        
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
        inner_log_likelihood = np.empty((n_outer, chunk_size))
        inner_log_weights = np.empty((n_outer, chunk_size))

        # Timing (seconds)
        # - timing_gauss: includes outer forward solve + FIM + Gaussian EIG computation
        # - timing_nmc: includes everything in the outer iteration (gauss + data gen + inner loop, etc.)
        timing_gauss = np.empty(n_outer)
        timing_nmc = np.empty(n_outer)
        
        x_true = model.generate_vector()
        rank = self.comm_sampler.rank
        
        for i in range(n_outer):
            t_outer_perf = time.perf_counter()
            t_gauss_perf = time.perf_counter()
            
            # Set outer parameter
            theta = outer_parameters[i]
            gmc.set_global(self.comm_mesh, theta, x_true[hp.PARAMETER])
            model.solveFwd(x_true[hp.STATE], x_true)
            
            # Compute FIM
            FIM = compute_fim(model, x_true)
            FIM = 0.5 * (FIM + FIM.T)
            
            # Gaussian approximation to info gain
            eigenvalues = np.maximum(np.linalg.eigvalsh(FIM), 0.0)
            gauss_val = np.sum(np.log1p(eigenvalues))
            info_gain_gauss[i] = 0.5 * gauss_val

            # Stop Gaussian timing after fwd + FIM + Gaussian EIG value is computed
            timing_gauss[i] = time.perf_counter() - t_gauss_perf
            
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
            
            inner_ll, inner_w = self._inner_loop(model, theta, FIM, inner_seed,
                                                  chunk_id=chunk_id, chunk_size=chunk_size)
            
            inner_log_likelihood[i] = inner_ll
            inner_log_weights[i] = inner_w

            # Total per-outer timing for NMC (includes everything)
            timing_nmc[i] = time.perf_counter() - t_outer_perf
            
            elapsed = time.time() - self._t_start
            dt = timing_nmc[i]
            eta = (elapsed / (i + 1)) * (n_outer - i - 1)
            self._log(f"[NMC] Outer {i+1}/{n_outer}: {dt:.1f}s (elapsed {fmt_time(elapsed)}, ETA {fmt_time(eta)})")
        
        timing_total = time.perf_counter() - t_start_perf
        return {
            "info_gain_gauss": info_gain_gauss,
            "outer_log_likelihood": outer_log_likelihood,
            "inner_log_likelihood": inner_log_likelihood,
            "inner_log_weights": inner_log_weights,
            "timing_gauss": timing_gauss,
            "timing_nmc": timing_nmc,
            "timing_total": np.array(timing_total),
            "outer_parameters": outer_parameters.copy()
        }


def main():
    parser = argparse.ArgumentParser(description="Nested Monte Carlo for EIG estimation")
    parser.add_argument("--output_path", type=str, default="./nmc_results/")
    parser.add_argument("--design_index", type=int, default=0)
    parser.add_argument("--inner_samples", type=int, default=512)
    parser.add_argument("--outer_samples_per_process", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)
    parser.add_argument("--mask_threshold", type=float, default=0.3,
                        help="Mask threshold (0-1). Higher = more attenuation.")
    parser.add_argument("--mask_steepness", type=float, default=3.0,
                        help="Mask sigmoid steepness. Lower = softer transition.")
    parser.add_argument("--pixel_density", type=int, default=200,
                        help="Image pixel density. Lower = fewer observation pixels.")
    parser.add_argument("--image_noise_std", type=float, default=5.0,
                        help="Image observation noise std (default: 5.0).")
    parser.add_argument("--force_noise_std", type=float, default=0.005,
                        help="Force observation noise std (default: 0.005).")
    parser.add_argument("--proposal_df", type=float, default=0,
                        help="Proposal degrees of freedom. 0=Gaussian (default), >0=Student-t.")
    parser.add_argument("--n_image_snapshots", type=int, default=0,
                        help="Number of DIC image snapshots. 0=use config default (20).")
    parser.add_argument("--n_force_data", type=int, default=0,
                        help="Number of force data points. 0=use config default (100).")
    parser.add_argument("--inner_chunk_id", type=int, default=0,
                        help="Inner sample chunk ID (0 to num_inner_chunks-1). "
                             "Each chunk handles inner_samples/num_inner_chunks samples.")
    parser.add_argument("--num_inner_chunks", type=int, default=1,
                        help="Total number of inner sample chunks. "
                             "K=inner_samples is split across this many jobs.")
    parser.add_argument("--nmc_config", type=str, default="",
                        help="Path to NMC-specific model config (overrides speckle from model_config.pkl). "
                             "If empty, uses ../model_config.pkl for speckle pattern.")
    parser.add_argument("--n_time_steps", type=int, default=0,
                        help="Override number of time steps. 0=use config default.")
    args = parser.parse_args()
    
    # Validate arguments
    if args.process_id < 0:
        raise ValueError(f"--process_id must be non-negative, got {args.process_id}")
    if args.outer_samples_per_process <= 0:
        raise ValueError(f"--outer_samples_per_process must be positive, got {args.outer_samples_per_process}")
    if args.inner_samples <= 0:
        raise ValueError(f"--inner_samples must be positive, got {args.inner_samples}")
    if args.num_inner_chunks < 1:
        raise ValueError(f"--num_inner_chunks must be >= 1, got {args.num_inner_chunks}")
    if args.inner_chunk_id < 0 or args.inner_chunk_id >= args.num_inner_chunks:
        raise ValueError(f"--inner_chunk_id must be in [0, {args.num_inner_chunks}), got {args.inner_chunk_id}")
    
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
    
    # Optionally override speckle pattern from NMC-specific config
    if args.nmc_config:
        with open(args.nmc_config, "rb") as f:
            nmc_config = pickle.load(f)
            speckle_center = nmc_config["speckle_centers"]
            speckle_radius = nmc_config["speckle_radii"]
        if rank == 0:
            print(f"[Main] Using NMC speckle from {args.nmc_config} "
                  f"({len(speckle_radius)} speckles, r={speckle_radius[0]:.4f})", flush=True)
    
    with open("./settings/settings.pkl", "rb") as f:
        design_samples = pickle.load(f)["design_samples"]
    
    base_seed = model_settings["seed"]

    # -------------------------------------------------------------------------
    # NMC-specific model overrides.
    #
    # The design optimization uses the full-resolution model from model_config.pkl:
    #   pixel_density = 500, mask_threshold = 0.05, mask_steepness = 10
    # With those settings, nearly all image pixels are fully observed
    # (mask ≈ 1), producing EIG ≈ 40-50 nats — too high for NMC.
    #
    # mask_threshold and mask_steepness are configurable via CLI to find
    # the regime where NMC is feasible.
    # -------------------------------------------------------------------------
    model_settings["mask_threshold"] = args.mask_threshold
    model_settings["mask_steepness"] = args.mask_steepness
    model_settings["pixel_density"] = args.pixel_density
    model_settings["image_noise_std"] = args.image_noise_std
    model_settings["force_noise_std"] = args.force_noise_std
    if args.n_image_snapshots > 0:
        model_settings["n_image_snapshots"] = args.n_image_snapshots
    if args.n_force_data > 0:
        model_settings["n_force_data"] = args.n_force_data
    if args.n_time_steps > 0:
        model_settings["n_time_steps"] = args.n_time_steps

    if rank == 0:
        print(f"[Main] mask_threshold={args.mask_threshold}, "
              f"mask_steepness={args.mask_steepness}, "
              f"pixel_density={args.pixel_density}, "
              f"proposal_df={args.proposal_df}, "
              f"image_noise_std={args.image_noise_std}, "
              f"force_noise_std={args.force_noise_std}, "
              f"n_image_snapshots={model_settings['n_image_snapshots']}, "
              f"n_force_data={model_settings['n_force_data']}, "
              f"n_time_steps={model_settings['n_time_steps']}", flush=True)

    # Initialize estimator
    estimator = NestedMonteCarlo(
        comm_mesh, comm_sampler, model_settings,
        speckle_center, speckle_radius, image_corner_coords
    )
    estimator.outer_samples_per_process = args.outer_samples_per_process
    estimator.num_inner_samples = args.inner_samples
    estimator.base_seed = base_seed
    estimator.process_id = args.process_id
    estimator.proposal_df = args.proposal_df
    estimator.inner_chunk_id = args.inner_chunk_id
    estimator.num_inner_chunks = args.num_inner_chunks
    
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
    stretch = (design[1], design[2])
    control_values = design[3:]
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
            "inner_chunk_id": args.inner_chunk_id,
            "num_inner_chunks": args.num_inner_chunks,
            "total_inner_samples": estimator.num_inner_samples,
        }
        for k, v in all_results.items():
            if v.ndim == 2:
                results[k] = v.reshape(-1)
            elif v.ndim == 3:
                results[k] = v.reshape(-1, v.shape[-1])
            else:
                results[k] = v
        
        if args.num_inner_chunks > 1:
            out_file = os.path.join(output_dir, f"results_{args.process_id}_chunk_{args.inner_chunk_id}.pkl")
        else:
            out_file = os.path.join(output_dir, f"results_{args.process_id}.pkl")
        with open(out_file, "wb") as f:
            pickle.dump(results, f)
        
        elapsed = time.time() - estimator._t_start
        print(f"[Main] Done in {fmt_time(elapsed)}. Saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()