import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ufl")
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
        # Formula: 0.5 * (-log(det(Gamma_loc)))
        # With Gamma_0 = I:
        #   -log(det(Gamma_loc)) = sum(log(1 + lambda_i))
        
        val = 0.5 * np.sum(np.log1p(eigenvalues))
        
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
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint JSON to resume from")
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

    # Override model seed with command line argument to ensure --seed works
    model_settings["seed"] = args.seed

    # Seed RNGs
    if rank == 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # --- Initialize Utility Function ---
    util = UtilityFunction(comm_mesh, comm_sampler, model_settings, image_corners_coords, center, radius)
    # Generate fixed parameter samples (Truths) for each rank
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
        # Note: random_state is deprecated for acquisition functions
        acq = acquisition.ExpectedImprovement(
            xi=args.xi,
            exploration_decay=args.exploration_decay,
            exploration_decay_delay=args.exploration_delay
        )
        
        optimizer = BayesianOptimization(
            f=None, 
            acquisition_function=acq,
            pbounds=pbounds_normalized,
            random_state=np.random.RandomState(args.seed + 2),
            verbose=0
        )

        # Gaussian Process Setup
        initial_ls = np.full(n_dims, 0.2) 
        ls_bounds = (0.01, 2.0)
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
        
        def np_encoder(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # Get fitted GP kernel params (for ARD)
        # Note: After fitting, kernel_ is a WrappedKernel. We extract the core Matern params.
        gp = optimizer._gp
        if hasattr(gp, "kernel_") and gp.kernel_ is not None:
            # Extract only the Matern-relevant params from the wrapped kernel
            k = gp.kernel_
            gp_kernel_params = {
                "length_scale": getattr(k, "length_scale", None),
                "length_scale_bounds": getattr(k, "length_scale_bounds", None),
                "nu": getattr(k, "nu", None),
            }
        else:
            k = gp.kernel
            gp_kernel_params = {
                "length_scale": getattr(k, "length_scale", None),
                "length_scale_bounds": getattr(k, "length_scale_bounds", None),
                "nu": getattr(k, "nu", None),
            } if k is not None else {}

        # Save random state for reproducibility
        random_state_tuple = optimizer._random_state.get_state()
        random_state = {
            "bit_generator": random_state_tuple[0],
            "state": random_state_tuple[1].tolist(),
            "pos": int(random_state_tuple[2]),
            "has_gauss": int(random_state_tuple[3]),
            "cached_gaussian": float(random_state_tuple[4]),
        }

        # Save acquisition function state (including iteration counter)
        acq_params = optimizer._acquisition_function.get_acquisition_params()
        acq_iteration = optimizer._acquisition_function.i

        state = {
            "eval_counter": counter,
            "params": [res["params"] for res in optimizer.res],
            "target": [res["target"] for res in optimizer.res],
            "gp_kernel_params": gp_kernel_params,
            "gp_alpha": gp.alpha,
            "gp_normalize_y": gp.normalize_y,
            "gp_n_restarts_optimizer": gp.n_restarts_optimizer,
            "random_state": random_state,
            "acquisition_params": acq_params,
            "acquisition_iteration": acq_iteration,  # Save the iteration counter
        }
        
        with open(os.path.join(args.output_path, f"optimizer_state_{tag}.json"), "w") as f:
            json.dump(state, f, indent=2, default=np_encoder)

        # Save progress summary
        if len(optimizer.res) > 0:
            best = optimizer.max
            best_phys = to_physical(best["params"])
            progress = {
                "eval_counter": counter,
                "best_objective": best["target"],
                "best_params_normalized": best["params"],
                "best_params_physical": best_phys
            }
            with open(os.path.join(args.output_path, "optimizer_progress_latest.json"), "w") as f:
                json.dump(progress, f, indent=2)

    def load_state(path):
        """Load optimizer state from JSON and resume."""
        if rank != 0:
            return None
            
        with open(path, "r") as f:
            state = json.load(f)
        
        # Re-register all points
        for params, target in zip(state["params"], state["target"]):
            optimizer.register(params=params, target=float(target))
        
        # Restore GP kernel with learned ARD hyperparameters
        kernel_params = state["gp_kernel_params"]
        # Handle case where length_scale might be a list (ARD) or scalar
        length_scale = kernel_params["length_scale"]
        if isinstance(length_scale, list):
            length_scale = np.array(length_scale)
        
        # Convert length_scale_bounds back to tuple if it's a list
        ls_bounds = kernel_params["length_scale_bounds"]
        if isinstance(ls_bounds, list):
            ls_bounds = tuple(ls_bounds)
        
        kernel = Matern(
            length_scale=length_scale,
            length_scale_bounds=ls_bounds,
            nu=kernel_params["nu"]
        )
        optimizer.set_gp_params(
            kernel=kernel,
            alpha=state["gp_alpha"],
            normalize_y=state["gp_normalize_y"],
            n_restarts_optimizer=state["gp_n_restarts_optimizer"]
        )
        
        # Fit GP with restored data
        if len(optimizer._space):
            optimizer._gp.fit(optimizer._space.params, optimizer._space.target)
        
        # Restore random state
        rs = state["random_state"]
        random_state_tuple = (
            rs["bit_generator"],
            np.array(rs["state"], dtype=np.uint32),
            rs["pos"],
            rs["has_gauss"],
            rs["cached_gaussian"],
        )
        optimizer._random_state.set_state(random_state_tuple)
        
        # Restore acquisition function state
        optimizer._acquisition_function.set_acquisition_params(state["acquisition_params"])
        # Restore the iteration counter (critical for decay delay!)
        optimizer._acquisition_function.i = state["acquisition_iteration"]
        
        return state["eval_counter"]
    
    # --- Main Optimization Loop ---
    t0 = time.time()
    eval_counter = 0
    
    # Handle resume from checkpoint
    if args.resume is not None:
        file_exists = False
        if rank == 0:
            file_exists = os.path.exists(args.resume)
            if file_exists:
                eval_counter = load_state(args.resume)
                print(f"[Resume] Loaded state from {args.resume}, eval_counter={eval_counter}", flush=True)
        file_exists = comm_sampler.bcast(file_exists, root=0)
        if not file_exists:
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        eval_counter = comm_sampler.bcast(eval_counter, root=0)

    # 1. Initial Exploration (Sobol Designs) - skip completed ones if resuming
    if eval_counter < args.n_explorations:
        if rank == 0:
            init_points = initial_sobol_points(pbounds_normalized, args.n_explorations, seed=args.seed)
        else:
            init_points = None
        init_points = bcast_params(init_points)

        for iexp in range(eval_counter, args.n_explorations):
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
    # Calculate how many BO iterations remain
    bo_start = max(0, eval_counter - args.n_explorations)
    for it in range(bo_start, args.n_iterations):
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