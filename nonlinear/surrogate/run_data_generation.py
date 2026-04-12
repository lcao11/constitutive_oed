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
hippy_path = os.environ.get("HIPPYLIB_PATH")
if hippy_path and hippy_path not in sys.path:
    sys.path.append(hippy_path)
import hippylib as hp

gmc_path = os.environ.get("GMC_PATH")
if gmc_path and gmc_path not in sys.path:
    sys.path.append(gmc_path)
import geometric_mcmc as gmc

# --- Project Imports ---
sys.path.append("../../")
from utils import (
    generate_mesh,
    setup_image_observation,
    speckled_reference,
    compute_fim,
)
sys.path.append("../")
from nonlinear_viscoelasticity import (
    NonlinearViscoElasticModel,
    CheckInsideImage,
    generate_data_idx,
    generate_noise_model,
)

# --- Configuration ---
jax.config.update("jax_enable_x64", True)
logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
dl.set_log_active(False)

try:
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=20)
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
except Exception:
    pass


def design_bounds(upper_control_value: float):
    """
    Define the bounds for the design parameters.

    Design vector:
        [rotation, stretch_x, stretch_y, control_0, ..., control_9]
    """
    lower = [0.0, 0.1, 0.1] + [0.0] * 10
    upper = [0.5 * math.pi, 0.35, 0.35] + [upper_control_value] * 10
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def interpolate_loading_path(control_values: np.ndarray):
    """Interpolate control points to create a continuous loading path on [0, 1]."""
    times = np.linspace(0.0, 1.0, control_values.size + 1)
    values = np.zeros((times.shape[0], 2))
    values[1:, 0] = control_values
    return PchipInterpolator(times, values)


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for nonlinear OED surrogate model (MPI/QMC)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data_set/",
        help="Path to save the results",
    )
    parser.add_argument(
        "--samples_per_process",
        type=int,
        default=64,
        help="Number of joint (design, parameter) samples per process_id *per rank*",
    )
    parser.add_argument(
        "--process_id",
        type=int,
        default=0,
        help="Process ID (job-array index) selecting a disjoint QMC block",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global seed for QMC scrambling (defaults to model_settings['seed'])",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=2,
        help="Checkpoint every N local samples (rank 0 only)",
    )
    args = parser.parse_args()

    if args.process_id < 0:
        raise ValueError(f"--process_id must be non-negative, got {args.process_id}")
    if args.samples_per_process <= 0:
        raise ValueError(
            f"--samples_per_process must be positive, got {args.samples_per_process}"
        )

    # --- MPI Setup (mirror run_random_design_evaluations.py) ---
    # comm_mesh: size 1, used for FEniCS solves
    # comm_sampler: all ranks, used to coordinate sampling and I/O
    comm_mesh, comm_sampler = gmc.split_mpi_comm(
        MPI.COMM_WORLD, 1, MPI.COMM_WORLD.size
    )
    rank = comm_sampler.Get_rank()
    size = comm_sampler.Get_size()

    if rank == 0 and not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    comm_sampler.Barrier()

    # --- Load Configuration ---
    with open("../model_config.pkl", "rb") as f:
        config_data = pickle.load(f)
    model_settings = config_data["model_settings"]
    center = config_data["speckle_centers"]
    radius = config_data["speckle_radii"]
    image_corner_coords = config_data["image_corners_coords"]

    # Determine seed
    if args.seed is not None:
        base_seed = int(args.seed)
    else:
        base_seed = int(model_settings.get("seed", 0))

    # --- Dimensions ---
    design_lower, design_upper = design_bounds(
        model_settings["max_strain"] * model_settings["aspect_ratio"]
    )
    design_dim = design_lower.size

    param_dim = 15
    total_dim = design_dim + param_dim

    # Total number of joint samples for this process_id across all ranks
    n_local_per_rank = int(args.samples_per_process)
    n_local_block = n_local_per_rank * size  # per process_id

    # Global start index in the Sobol sequence for this process_id
    global_block_start = args.process_id * n_local_block

    # --- Joint QMC Sampling (Design + Parameters) ---
    # Rank 0 in comm_sampler generates the joint block, then scatters
    joint_u_local = None
    if rank == 0:
        sampler = qmc.Sobol(d=total_dim, scramble=True, seed=base_seed)
        if global_block_start > 0:
            sampler.fast_forward(global_block_start)
        joint_block = sampler.random(n_local_block)  # shape (n_local_block, total_dim)

        # Split into size chunks, each of length n_local_per_rank
        joint_chunks = [
            joint_block[i * n_local_per_rank : (i + 1) * n_local_per_rank]
            for i in range(size)
        ]
    else:
        joint_chunks = None

    # Scatter chunks to all ranks
    joint_u_local = comm_sampler.scatter(joint_chunks, root=0)
    assert joint_u_local.shape == (n_local_per_rank, total_dim)

    # --- Storage (local + global) ---
    # Local lists on each rank
    local_parameters = []
    local_designs = []
    local_fims = []

    time_start = time.time()

    # Parameter dimension consistency check will be done per-solve using Vh

    # --- Main loop over local samples on each rank ---
    for j_local in range(n_local_per_rank):
        u_vec = joint_u_local[j_local]
        u_design = u_vec[:design_dim]
        u_param = u_vec[design_dim:]

        # Map design into bounds
        design_samples = design_lower + (design_upper - design_lower) * u_design

        # Map parameters to Gaussian(0, I)
        parameter_sample = norm.ppf(u_param)

        # Extract design components
        rotation = float(design_samples[0])
        stretch_x = float(design_samples[1])
        stretch_y = float(design_samples[2])
        stretch = np.array([stretch_x, stretch_y], dtype=float)
        control_values = design_samples[3:]

        # --- Mesh & model on comm_mesh (size 1) ---
        mesh = generate_mesh(
            comm_mesh,
            rect_width=model_settings["aspect_ratio"],
            rect_height=1.0,
            stretch=stretch,
            rotation=rotation,
            density=model_settings["cell_density"],
            refine_factor=4,
            corridor_refine_factor=2,
        )

        loading_position = interpolate_loading_path(control_values)
        inside = CheckInsideImage(model_settings["aspect_ratio"], stretch, rotation)

        reference_mask, targets = setup_image_observation(
            image_corner_coords,
            inside,
            model_settings["pixel_density"],
            oversampling_factor=model_settings["high_resolution_factor"],
        )
        reference_image = speckled_reference(
            reference_mask, image_corner_coords, center, radius
        )

        model, _, _, _ = NonlinearViscoElasticModel(
            mesh,
            model_settings,
            loading_position,
            image_corner_coords,
            reference_image,
            reference_mask,
            targets,
        )

        Vh = model.problem.Vh
        current_param_dim = Vh[hp.PARAMETER].dim()
        if current_param_dim != param_dim:
            raise ValueError(
                f"Model parameter dimension ({current_param_dim}) does not match "
                f"configured dimension ({param_dim})."
            )

        force_idx, image_idx = generate_data_idx(model_settings, reference_image.shape)
        model.misfit.check_mask_idx = image_idx
        model.misfit.noise_precision = generate_noise_model(
            model_settings, force_idx, image_idx
        )

        x_true = model.generate_vector()
        # Set parameters using comm_mesh
        gmc.set_global(comm_mesh, parameter_sample, x_true[hp.PARAMETER])

        # Solve forward with fail catch
        try:
            model.solveFwd(x_true[hp.STATE], x_true)
            # Compute FIM (Gauss–Newton Hessian) only if solve succeeds
            fim_local = compute_fim(model, x_true)
        except Exception as e:
            # Mark failure: use NaNs (or zeros) so you can filter later
            fim_local = np.full((param_dim, param_dim), np.nan, dtype=float)
            if rank == 0:
                print(f"solveFwd failed at local sample {j_local} on rank {rank}: {e}", flush=True)

        # Store local results regardless, to keep alignment of samples
        local_parameters.append(parameter_sample)
        local_designs.append(design_samples)
        local_fims.append(fim_local)

        # Rank 0 progress summary (based on index in joint block)
        if rank == 0:
            # Global sample index within this process_id block (0-based)
            global_index_within_block = j_local
            total_done = global_index_within_block + 1
            elapsed = time.time() - time_start
            per_sample_min = elapsed / total_done / 60.0
            print(
                f"Rank 0 (local idx {j_local+1}/{n_local_per_rank}) "
                f"avg {per_sample_min:.2f} min/sample",
                flush=True,
            )

        # --- Checkpointing (rank 0 only, gather first) ---
        if (j_local + 1) % max(1, args.checkpoint_interval) == 0:
            # Gather all local lists to rank 0
            all_params = comm_sampler.gather(np.stack(local_parameters), root=0)
            all_designs = comm_sampler.gather(np.stack(local_designs), root=0)
            all_fims = comm_sampler.gather(np.stack(local_fims), root=0)

            if rank == 0:
                # Shape them as (size, n_local_so_far, ...) then stack over first axis
                params_arr = np.concatenate(all_params, axis=0)
                designs_arr = np.concatenate(all_designs, axis=0)
                fims_arr = np.concatenate(all_fims, axis=0)

                checkpoint_path = os.path.join(
                    args.output_path,
                    f"data_{int(args.process_id)}_seed_{base_seed}_checkpoint.pkl",
                )
                checkpoint = {
                    "process_id": int(args.process_id),
                    "seed": base_seed,
                    "parameters": params_arr,
                    "designs": designs_arr,
                    "fims": fims_arr,
                    "samples_per_rank": j_local + 1,
                    "world_size": size,
                }
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved checkpoint to {checkpoint_path}", flush=True)

    # --- Final gather and save (rank 0 only) ---
    all_params = comm_sampler.gather(np.stack(local_parameters), root=0)
    all_designs = comm_sampler.gather(np.stack(local_designs), root=0)
    all_fims = comm_sampler.gather(np.stack(local_fims), root=0)

    if rank == 0:
        params_arr = np.concatenate(all_params, axis=0)   # shape (size * n_local_per_rank, param_dim)
        designs_arr = np.concatenate(all_designs, axis=0) # shape (size * n_local_per_rank, design_dim)
        fims_arr = np.concatenate(all_fims, axis=0)       # shape (size * n_local_per_rank, param_dim, param_dim)

        final_path = os.path.join(
            args.output_path,
            f"data_{int(args.process_id)}_seed_{base_seed}.pkl",
        )
        result = {
            "process_id": int(args.process_id),
            "seed": base_seed,
            "parameters": params_arr,
            "designs": designs_arr,
            "fims": fims_arr,
            "samples_per_rank": n_local_per_rank,
            "world_size": size,
        }
        with open(final_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved final dataset to {final_path}", flush=True)


if __name__ == "__main__":
    main()