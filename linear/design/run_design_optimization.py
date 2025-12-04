from mpi4py import MPI
import os
import sys
import argparse
import numpy as np
import math
import csv
import json
import random
import pickle
import time

# --- External Library Paths ---
# Add paths for hippylib and geometric_mcmc if they are in environment variables
hippy_path = os.environ.get('HIPPYLIB_PATH')
if hippy_path and hippy_path not in sys.path:
    sys.path.append(hippy_path)
import hippylib as hp

gmc_path = os.environ.get('GMC_PATH')
if gmc_path and gmc_path not in sys.path:
    sys.path.append(gmc_path)
import geometric_mcmc as gmc

# --- Project Imports ---
sys.path.append("../../")
from utils import compute_eigenvalues, setup_image_observation, speckled_reference, generate_mesh
sys.path.append("../")
from linear_viscoelasticity import (
    ViscoElasticModel, CheckInsideImage, generate_data_idx, generate_noise_model,
)

# --- Optimization & Math Imports ---
from scipy.interpolate import PchipInterpolator
from scipy.stats import qmc, norm
from sklearn.gaussian_process.kernels import Matern
from bayes_opt import BayesianOptimization, acquisition

import jax
jax.config.update("jax_enable_x64", True)  # Use 64-bit precision for JAX

# --- Plotting & Logging Setup ---
import dolfin as dl
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)


class UtilityFunction:
    """
    Handles the evaluation of the OED objective function (Expected Information Gain).
    
    This class manages:
    1. Mesh generation and caching.
    2. Parameter sampling (Sobol sequence distributed across MPI ranks).
    3. Forward model execution and eigenvalue computation for the OED objective.
    """
    def __init__(self, comm_mesh, comm_sampler, model_settings, image_corner_coords, center, radius):
        self.model_settings = model_settings
        self.comm_sampler = comm_sampler  # Communicator for aggregating results across samples
        self.comm_mesh = comm_mesh        # Communicator for parallel mesh/FEM operations
        self.image_corner_coords = image_corner_coords
        self.center = center
        self.radius = radius
        self._mesh_cache = {}
        self.parameter_sample = None

    def generate_parameter_samples(self, seed):
        """
        Generate parameter samples using a Sobol sequence and distribute them across ranks.
        
        Each MPI rank gets ONE fixed parameter sample 'theta'. The OED objective is approximated
        by averaging the utility u(d, theta) over all ranks (Monte Carlo integration).
        """
        rank = self.comm_sampler.rank
        size = self.comm_sampler.size
        n_params = self.model_settings["n_parameters"]

        scatter_data = None
        if rank == 0:
            # 1. Generate uniform samples in [0, 1]^d using Sobol sequence
            sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)
            # Get enough samples for all ranks (next power of 2 for Sobol efficiency)
            m = int(np.ceil(np.log2(size)))
            u_samples = sampler.random_base2(m)[:size]
            
            # 2. Transform to standard normal distribution N(0, I) via inverse CDF (probit)
            # This matches the Gaussian prior assumption on the parameters.
            normal_samples = norm.ppf(u_samples)
            
            # 3. Prepare list of arrays for scattering
            scatter_data = [normal_samples[i, :] for i in range(size)]

        # Distribute one sample vector to each rank
        self.parameter_sample = self.comm_sampler.scatter(scatter_data, root=0)
        self.parameter_sample = np.array(self.parameter_sample)

    def _get_mesh(self, stretch, rotation):
        """Get or generate mesh for specific geometric parameters (cached)."""
        # Round keys to avoid cache misses due to floating point noise
        key = (round(rotation, 8),
               round(float(stretch[0]), 8),
               round(float(stretch[1]), 8))
        
        mesh = self._mesh_cache.get(key)
        if mesh is None:
            mesh = generate_mesh(
                self.comm_mesh,
                rect_width=self.model_settings["aspect_ratio"],
                rect_height=1.0,
                stretch=stretch,
                rotation=rotation,
                density=self.model_settings["cell_density"],
                refine_factor=2.5,
                corridor_refine_factor=1.5
            )
            self._mesh_cache[key] = mesh
        return mesh

    def interpolate_loading_path(self, control_values):
        """Create a smooth loading path from control points."""
        total_time = self.model_settings["total_time"]
        times = np.linspace(0.0, total_time, self.model_settings["n_control_points"] + 1)
        values = np.zeros((times.shape[0], 2))
        values[1:, 0] = control_values # Apply control values to x-displacement
        return PchipInterpolator(times, values)

    def evaluate_design(self, rotation, stretch, control_values):
        """
        Evaluate the design 'd' (geometry + loading) for the OED objective.
        
        Objective: Expected Information Gain (EIG).
        We use the standard Shannon Information Gain (Mutual Information) approximation:
        U(d) = 0.5 * E_theta [ log(det(I + H_misfit(theta))) ]
        
        Returns:
            float: The globally averaged objective value across all MPI ranks.
        """
        # 1. Setup Geometry and Mesh
        mesh = self._get_mesh(stretch, rotation)
        loading_position = self.interpolate_loading_path(control_values)
        inside = CheckInsideImage(self.model_settings["aspect_ratio"], stretch, rotation)

        # 2. Setup Synthetic Image Observations
        reference_mask, targets = setup_image_observation(
            self.image_corner_coords, inside, self.model_settings["pixel_density"],
            oversampling_factor=self.model_settings["high_resolution_factor"]
        )
        reference_image = speckled_reference(
            reference_mask, self.image_corner_coords, self.center, self.radius
        )

        # 3. Initialize Physics Model
        model, _, _, _ = ViscoElasticModel(
            mesh, self.model_settings, loading_position,
            self.image_corner_coords, reference_image, reference_mask, targets
        )

        # 4. Setup Noise Model
        force_idx, image_idx = generate_data_idx(self.model_settings, reference_image.shape)
        model.misfit.check_mask_idx = image_idx
        model.misfit.noise_precision = generate_noise_model(self.model_settings, force_idx, image_idx)

        # 5. Set True Parameters (Sampled via Sobol)
        if self.parameter_sample is None:
             raise RuntimeError("Parameter samples not initialized.")
        
        x_true = model.generate_vector()
        gmc.set_global(self.comm_mesh, self.parameter_sample, x_true[hp.PARAMETER])

        # 6. Solve Forward Problem & Compute Eigenvalues of Hessian/FIM
        model.solveFwd(x_true[hp.STATE], x_true)
        eigenvalues = compute_eigenvalues(model, x_true)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # 7. Compute Local Utility (KL Divergence / Information Gain)
        # Formula: 0.5 * ( -log(det(Gamma_loc)) + tr(Gamma_0^{-1} Gamma_loc) ) + C(theta)
        # With Gamma_0 = I:
        #   -log(det(Gamma_loc)) = sum(log(1 + lambda_i))
        #   tr(Gamma_loc)        = sum(1 / (1 + lambda_i))
        #   C(theta)             = 0.5 * ||theta||^2  (from prior term in KL)
        
        term1 = np.sum(np.log1p(eigenvalues))       # -log(det(Gamma_loc))
        term2 = np.sum(1.0 / (1.0 + eigenvalues))   # tr(Gamma_loc)
        term3 = x_true[hp.PARAMETER].inner(x_true[hp.PARAMETER]) # ||theta||^2
        
        # Note: The standard KL divergence also has a constant -N/2, which we ignore here.
        val = 0.5 * (term1 + term2 + term3)
        
        local_sum = float(val)
        local_count = 1

        # 8. Aggregate Global Mean across all ranks
        global_sum = self.comm_sampler.allreduce(local_sum, op=MPI.SUM)
        global_count = self.comm_sampler.allreduce(local_count, op=MPI.SUM)
        return float(global_sum / max(1, global_count))


def initial_sobol_points(pbounds, n, seed):
    """Generate 'n' initial design points using a Sobol sequence within 'pbounds'."""
    keys = list(pbounds.keys())
    D = len(keys)
    sampler = qmc.Sobol(d=D, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(1, n))))
    U = sampler.random_base2(m)[:n]
    
    # Map [0,1] samples to parameter bounds
    X = {}
    for j, k in enumerate(keys):
        lo, hi = pbounds[k]
        X[k] = lo + (hi - lo) * U[:, j]
        
    # Convert to list of dictionaries
    return [{k: float(X[k][i]) for k in keys} for i in range(n)]


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization for OED Design")
    parser.add_argument("--output_path", type=str, default="./design_result/")
    parser.add_argument("--n_explorations", type=int, default=5, help="Number of initial random designs")
    parser.add_argument("--n_iterations", type=int, default=10, help="Number of BO iterations")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed")
    parser.add_argument("--checkpoint_every", type=int, default=5, help="Checkpoint interval")
    parser.add_argument("--log_file", type=str, default="evaluations.csv", help="CSV log filename")
    parser.add_argument("--xi", type=float, default=0.1, help="Exploration parameter for EI")
    parser.add_argument("--exploration_decay", type=float, default=0.99)
    parser.add_argument("--exploration_delay", type=int, default=5)
    args = parser.parse_args()
    
    # --- MPI Setup ---
    comm_mesh, comm_sampler = gmc.split_mpi_comm(MPI.COMM_WORLD, 1, MPI.COMM_WORLD.size)
    rank = comm_sampler.rank

    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
    comm_sampler.Barrier()

    # --- Load Configuration ---
    with open("../model_config.pkl", "rb") as f:
        config_data = pickle.load(f)
        model_settings = config_data["model_settings"]
        center = config_data["speckle_centers"]
        radius = config_data["speckle_radii"]
        image_corners_coords = config_data["image_corners_coords"]

    # [FIX] Override model seed with command line argument to ensure --seed works
    model_settings["seed"] = args.seed

    # Seed RNGs
    if rank == 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # --- Initialize Utility Function ---
    util = UtilityFunction(comm_mesh, comm_sampler, model_settings, image_corners_coords, center, radius)
    # Generate fixed parameter samples (Truths) for each rank
    # [FIX] Use args.seed
    util.generate_parameter_samples(seed=args.seed + 12345)

    # --- Define Design Space (Physical Bounds) ---
    # We define the physical bounds but will optimize over [0, 1]
    physical_bounds = {
        "x0": (-0.5 * math.pi, 0.5 * math.pi),
        "x1": (0.1, 0.35),
    }
    control_upper = model_settings["max_strain"] * model_settings["aspect_ratio"]
    for i in range(model_settings["n_control_points"]):
        physical_bounds[f"x{2+i}"] = (0.0, control_upper)

    param_order = list(physical_bounds.keys())
    n_dims = len(param_order)

    # --- Normalized Bounds for Optimizer ---
    # Optimize in unit hypercube [0, 1]^d to help L-BFGS-B and ARD
    pbounds_normalized = {k: (0.0, 1.0) for k in param_order}

    # --- Helper Functions ---
    def to_physical(p_norm):
        """Map normalized [0,1] parameters to physical bounds."""
        p_phys = {}
        for k, v in p_norm.items():
            low, high = physical_bounds[k]
            p_phys[k] = low + (high - low) * v
        return p_phys

    def bcast_params(d):
        """Broadcast design parameters from root to all ranks."""
        return comm_sampler.bcast(d, root=0)

    def eval_point(pdict_norm):
        """Unpack normalized dictionary, map to physical, and evaluate."""
        # 1. Map to physical units
        pdict = to_physical(pdict_norm)
        
        # 2. Evaluate
        rotation = pdict["x0"]
        stretch = np.array([0.35, pdict["x1"]], dtype=float)
        controls = [pdict[f"x{2+i}"] for i in range(model_settings["n_control_points"])]
        return util.evaluate_design(rotation, stretch, controls)

    # --- Initialize Bayesian Optimizer (Rank 0 only) ---
    optimizer = None
    if rank == 0:
        # Acquisition Function
        acq = acquisition.ExpectedImprovement(
            xi=args.xi,
            # [FIX] Use args.seed
            random_state=np.random.RandomState(args.seed + 1),
            exploration_decay=args.exploration_decay,
            exploration_decay_delay=args.exploration_delay
        )
        
        optimizer = BayesianOptimization(
            f=None, 
            acquisition_function=acq,
            pbounds=pbounds_normalized, # <--- Use Normalized Bounds
            # [FIX] Use args.seed
            random_state=np.random.RandomState(args.seed + 2),
            verbose=0
        )

        # Gaussian Process Setup
        # Now that we are in [0, 1]^d, a length scale of ~0.2 is reasonable for ALL dimensions.
        # This makes ARD initialization robust.
        initial_ls = np.full(n_dims, 0.2) 
        ls_bounds = (0.01, 2.0) # Bounds relative to unit hypercube
        kernel = Matern(length_scale=initial_ls, length_scale_bounds=ls_bounds, nu=2.5)

        optimizer.set_gp_params(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=20
        )

    # Logging setup
    log_path = None
    if rank == 0:
        log_path = os.path.join(args.output_path, args.log_file)
        if not os.path.exists(log_path):
            # Log PHYSICAL parameters for interpretation
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(["eval_index", "phase", "objective"] + param_order)
    log_path = comm_sampler.bcast(log_path, root=0)

    def log_eval(phase, params_norm, obj, counter):
        """Write evaluation result to CSV (converting to physical units)."""
        if rank != 0: return
        params_phys = to_physical(params_norm)
        row = [counter, phase, float(obj)] + [float(params_phys[k]) for k in param_order]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def save_state(tag, counter):
        """Save optimizer state and progress JSON."""
        if rank != 0: return
        
        # Helper for JSON serialization of numpy types
        def np_encoder(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # 1. Save Optimizer State (for resuming/debugging)
        # We use a simplified serialization here for clarity
        state = {
            "eval_counter": counter,
            "params": [res["params"] for res in optimizer.res], # Normalized
            "target": [res["target"] for res in optimizer.res],
            "gp_params": optimizer._gp.kernel_.get_params()
        }
        
        with open(os.path.join(args.output_path, f"optimizer_state_{tag}.json"), "w") as f:
            json.dump(state, f, indent=2, default=np_encoder)

        # 2. Save Progress Summary (Best so far)
        if len(optimizer.res) > 0:
            best = optimizer.max
            # Convert best params to physical for easy reading
            best_phys = to_physical(best["params"])
            progress = {
                "eval_counter": counter,
                "best_objective": best["target"],
                "best_params_normalized": best["params"],
                "best_params_physical": best_phys
            }
            with open(os.path.join(args.output_path, "optimizer_progress_latest.json"), "w") as f:
                json.dump(progress, f, indent=2)

    # --- Main Optimization Loop ---
    t0 = time.time()
    eval_counter = 0

    # 1. Initial Exploration (Sobol Designs)
    if rank == 0:
        # initial_sobol_points now generates in [0, 1] which matches pbounds_normalized
        # [FIX] Use args.seed
        init_points = initial_sobol_points(pbounds_normalized, args.n_explorations, seed=args.seed)
    else:
        init_points = None
    init_points = bcast_params(init_points)

    for iexp in range(args.n_explorations):
        params = init_points[iexp]
        
        # Evaluate
        y = eval_point(params)
        
        # Register & Log
        if rank == 0:
            optimizer.register(params=params, target=float(y))
            log_eval("exploration", params, y, eval_counter)
            if args.checkpoint_every > 0 and (eval_counter + 1) % args.checkpoint_every == 0:
                save_state(f"eval{eval_counter+1}", eval_counter+1)
        
        eval_counter += 1

    # 2. Bayesian Optimization (Adaptive Designs)
    for it in range(args.n_iterations):
        # Suggest next point (Rank 0)
        if rank == 0:
            params = optimizer.suggest()
        else:
            params = None
        params = bcast_params(params)

        # Evaluate
        y = eval_point(params)

        # Register & Log
        if rank == 0:
            optimizer.register(params=params, target=float(y))
            log_eval("bayes_opt", params, y, eval_counter)
            if args.checkpoint_every > 0 and (eval_counter + 1) % args.checkpoint_every == 0:
                save_state(f"eval{eval_counter+1}", eval_counter+1)
        
        eval_counter += 1

    # Final Save
    if rank == 0:
        save_state("final", eval_counter)
        elapsed = time.time() - t0
        print(f"[Done] Total evaluations={eval_counter} elapsed={elapsed/60:.2f} min", flush=True)

if __name__ == "__main__":
    main()