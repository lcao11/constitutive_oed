# filepath: /resnick/groups/astuart/lianghao/constitutive_oed/linear/surrogate/run_data_generation.py
import os
import sys
import time
import math
import pickle
import argparse
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import qmc, norm
from mpi4py import MPI

import dolfin as dl
import jax

# --- External Library Paths ---
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
from utils import (
    generate_mesh, 
    setup_image_observation, 
    speckled_reference, 
    compute_pto_map_jacobian
)
sys.path.append("../")
from linear_viscoelasticity import (
    ViscoElasticModel, 
    CheckInsideImage, 
    generate_data_idx, 
    generate_noise_model
)

# --- Configuration ---
jax.config.update("jax_enable_x64", True)
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
except Exception:
    pass


def design_bounds(upper_control_value):
    """Define the bounds for the design parameters."""
    # [rotation, stretch_y, control_0, ..., control_N]
    lower = [-0.5 * math.pi, 0.1] + [0.0] * 10
    upper = [0.5 * math.pi, 0.35] + [upper_control_value] * 10
    return lower, upper


def interpolate_loading_path(control_values):
    """Interpolate control points to create a continuous loading path."""
    times = np.linspace(0, 1.0, len(control_values) + 1)
    values = np.zeros((times.shape[0], 2))
    values[1:, 0] = control_values
    return PchipInterpolator(times, values)


def main():
    parser = argparse.ArgumentParser(description="Generate training data for OED surrogate model.")
    parser.add_argument("--output_path", type=str, default="./data_set/", help="Path to save the results")
    parser.add_argument("--samples_per_process", type=int, default=512, help="Number of samples per process")
    parser.add_argument("--process_id", type=int, default=0, help="Process ID (for parallel job arrays)")
    parser.add_argument("--seed", type=int, default=None, help="Manual seed override for QMC scrambling")
    parser.add_argument("--checkpoint_interval", type=int, default=8, help="Save partial results every N samples")
    args = parser.parse_args()

    # Validate arguments
    if args.process_id < 0:
        raise ValueError(f"--process_id must be non-negative, got {args.process_id}")
    if args.samples_per_process <= 0:
        raise ValueError(f"--samples_per_process must be positive, got {args.samples_per_process}")

    # --- MPI Setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create output directory (Rank 0 only)
    if rank == 0:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)
    comm.Barrier()

    # --- Load Configuration ---
    with open("../model_config.pkl", "rb") as f:
        config_data = pickle.load(f)
        model_settings = config_data["model_settings"]
        center = config_data["speckle_centers"]
        radius = config_data["speckle_radii"]
        image_corner_coords = config_data["image_corners_coords"]

    # Determine Seed
    if args.seed is not None:
        base_seed = args.seed
    else:
        base_seed = int(model_settings["seed"])

    # --- Joint QMC Sampling (Design + Parameters) ---
    # 1. Define Dimensions
    design_lower, design_upper = design_bounds(
        model_settings["max_strain"] * model_settings["aspect_ratio"]
    )
    design_lower = np.asarray(design_lower, dtype=float)
    design_upper = np.asarray(design_upper, dtype=float)
    design_dim = design_lower.size
    
    # We assume n_parameters is fixed and known from settings
    param_dim = model_settings["n_parameters"]
    total_dim = design_dim + param_dim

    # 2. Generate Joint Samples
    # Single Sobol sampler for the joint space
    joint_sampler = qmc.Sobol(d=total_dim, scramble=True, seed=base_seed)
    
    # Advance the sequence to the start of this process's chunk.
    # This allows us to generate the correct slice of the global Sobol sequence
    # without needing to know the total number of processes or generating previous samples.
    start_index = args.process_id * args.samples_per_process
    
    # Validate start_index before fast_forward (scipy requires non-negative int)
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index} "
                         f"(process_id={args.process_id}, samples_per_process={args.samples_per_process})")
    
    if start_index > 0:
        joint_sampler.fast_forward(start_index)
    
    # Generate the samples for this specific process
    joint_u_proc = joint_sampler.random(args.samples_per_process)

    # --- Initialization ---
    parameter_list = []
    design_list = []
    fisher_list = []
    
    time0 = time.time()

    # --- Main Data Generation Loop ---
    for ii in range(args.samples_per_process):
        # Extract unit hypercube samples for this iteration
        u_vec = joint_u_proc[ii]
        u_design = u_vec[:design_dim]
        u_param = u_vec[design_dim:]

        # Transform Design: [0, 1] -> [lower, upper]
        design_samples = design_lower + (design_upper - design_lower) * u_design
        
        # Transform Parameters: [0, 1] -> Gaussian(0, I) via Inverse CDF
        parameter_sample = norm.ppf(u_param)

        # 1. Setup Design & Mesh
        rotation = design_samples[0]
        stretch = np.array([0.35, design_samples[1]])
        
        mesh = generate_mesh(
            comm,
            rect_width=model_settings["aspect_ratio"],
            rect_height=1.0,
            stretch=stretch,
            rotation=rotation,
            density=model_settings["cell_density"],
            refine_factor=2.5,
            corridor_refine_factor=1.5
        )
        
        # 2. Setup Model & Observations
        loading_position = interpolate_loading_path(design_samples[3:])
        inside = CheckInsideImage(model_settings["aspect_ratio"], stretch, rotation)
        
        reference_mask, targets = setup_image_observation(
            image_corner_coords,
            inside,
            model_settings["pixel_density"],
            oversampling_factor=model_settings["high_resolution_factor"]
        )
        reference_image = speckled_reference(reference_mask, image_corner_coords, center, radius)
        
        model, _, _, _ = ViscoElasticModel(
            mesh, model_settings, loading_position, image_corner_coords, 
            reference_image, reference_mask, targets
        )
        
        # 3. Setup Noise & Parameters
        Vh = model.problem.Vh
        current_param_dim = Vh[hp.PARAMETER].dim()
        
        # Safety check: Ensure the mesh-implied dimension matches our sampler dimension
        if current_param_dim != param_dim:
            raise ValueError(f"Model parameter dimension ({current_param_dim}) does not match "
                             f"configured dimension ({param_dim}). Joint sampling requires fixed dimensions.")
        
        force_idx, image_idx = generate_data_idx(model_settings, reference_image.shape)
        model.misfit.check_mask_idx = image_idx
        model.misfit.noise_precision = generate_noise_model(model_settings, force_idx, image_idx)

        x_true = model.generate_vector()

        # 4. Solve Forward
        gmc.set_global(comm, parameter_sample, x_true[hp.PARAMETER])
        model.solveFwd(x_true[hp.STATE], x_true)
        
        # 5. Compute FIM
        model.misfit.setLinearizationPoint(x_true, gauss_newton_approx=True)
        jacobian = compute_pto_map_jacobian(model, x_true)
        fims = np.einsum("ji, j, jk->ik", jacobian, model.misfit.W, jacobian)

        # 6. Store Results
        parameter_list.append(parameter_sample)
        design_list.append(design_samples)
        fisher_list.append(fims)
        
        # 7. Logging & Checkpointing (Rank 0 only)
        if rank == 0:
            per_sample_min = (time.time() - time0) / (ii + 1) / 60.0
            print(f"Finished sample {ii+1}/{args.samples_per_process}, "
                  f"took {per_sample_min:.2f} min/sample", flush=True)

            if (ii + 1) % args.checkpoint_interval == 0:
                temp_data_set = {
                    "process_id": args.process_id,
                    "seed": base_seed,
                    "parameters": np.stack(parameter_list),
                    "designs": np.stack(design_list),
                    "fims": np.stack(fisher_list),
                    "completed_samples": ii + 1
                }
                # Use _partial to distinguish from completed files
                temp_filename = os.path.join(
                    args.output_path, 
                    f"data_{args.process_id}_seed_{base_seed}_partial.pkl"
                )
                with open(temp_filename, "wb") as f:
                    pickle.dump(temp_data_set, f)
                print(f"Saved checkpoint to {temp_filename}", flush=True)

    # --- Final Save (Rank 0 only) ---
    if rank == 0:
        data_set = {
            "process_id": args.process_id,
            "seed": base_seed,  # <--- Added seed here
            "parameters": np.stack(parameter_list),
            "designs": np.stack(design_list),
            "fims": np.stack(fisher_list)
        }

        final_filename = os.path.join(
            args.output_path, 
            f"data_{args.process_id}_seed_{base_seed}.pkl"
        )
        with open(final_filename, "wb") as f:
            pickle.dump(data_set, f)
        print(f"Saved final dataset to {final_filename}", flush=True)


if __name__ == "__main__":
    main()