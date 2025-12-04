# Move these environment settings to the very top of the file
import os, sys
import jax
import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt
hippy_path = os.environ.get('HIPPYLIB_PATH')
if hippy_path and hippy_path not in sys.path:
    sys.path.append(hippy_path)
import hippylib as hp
gmc_path = os.environ.get('GMC_PATH')
if gmc_path and gmc_path not in sys.path:
    sys.path.append(gmc_path)
import geometric_mcmc as gmc
sys.path.append("../../")
from utils import *
sys.path.append("../")
from linear_viscoelasticity import ViscoElasticModel, CheckInsideImage, generate_data_idx, generate_noise_model, set_bounds_for_parameters, extract_parameters
from scipy.interpolate import PchipInterpolator
jax.config.update("jax_enable_x64", True) # Use 64-bit precision

import matplotlib
try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
except:
    pass
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)
from mpi4py import MPI
import pickle
import time

if __name__=="__main__":

    comm_mesh, comm_sampler = gmc.split_mpi_comm(MPI.COMM_WORLD, 1, MPI.COMM_WORLD.size)

    output_dir = f"./result_mean_start/"
    os.makedirs(output_dir, exist_ok=True)

    with open("../model_config.pkl", "rb") as f:
        config_data = pickle.load(f)
        model_settings = config_data["model_settings"]
        center = config_data["speckle_centers"]
        radius = config_data["speckle_radii"]
        image_corners_coords = config_data["image_corners_coords"]

    np.random.seed(model_settings["seed"])
    scale = model_settings["max_strain"] * model_settings["aspect_ratio"]
    control_time = np.linspace(0, model_settings["total_time"], model_settings["n_control_points"]+1)
    control_value = np.zeros((control_time.shape[0], 2))
    control_value[:, 0] = np.random.uniform(low=0.0, high=scale, size=control_time.shape)
    control_value[0, 0] = 0.0
    loading_position = PchipInterpolator(control_time, control_value)
    plot_time = np.linspace(0, model_settings["total_time"], 1000)
    plt.figure(figsize=(5, 4))
    plt.plot(plot_time, loading_position(plot_time)[:, 0]/model_settings["aspect_ratio"], label="Loading Position", lw = 3, color = "k")
    plt.plot(control_time[1:], control_value[1:, 0]/model_settings["aspect_ratio"], "o", markersize= 10, label="Control Points", color="C3")
    plt.xlabel("Time")
    plt.ylabel("Prescribed Strain")
    plt.ylim(0, model_settings["max_strain"])
    plt.grid(":")
    plt.savefig(f"{output_dir}/loading_position.png", dpi=300, bbox_inches='tight')
    plt.close()
    rotation = np.random.uniform(-np.pi/2, np.pi/2)
    stretch = np.zeros(2)
    stretch[0] = 0.35
    stretch[1] = np.random.uniform(0.1, 0.35)

    inside = CheckInsideImage(model_settings["aspect_ratio"], stretch, rotation)

    reference_mask, targets = setup_image_observation(image_corners_coords, inside, model_settings["pixel_density"], oversampling_factor=model_settings["high_resolution_factor"])
    reference_image = speckled_reference(reference_mask, image_corners_coords, center, radius)

    plt.imshow(reference_mask, origin="lower", cmap="gray")
    plt.axis("off")
    plt.savefig(f"{output_dir}/reference_masks.png", dpi=300, bbox_inches='tight')

    plt.close()
    plt.imshow(reference_image, origin="lower", cmap="gray")
    plt.axis("off")
    plt.savefig(f"{output_dir}/reference_image.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.imshow(np.random.normal(size=reference_mask.shape), origin="lower", cmap="gray")
    plt.axis("off")
    plt.savefig(f"{output_dir}/noise_image.png", dpi=300, bbox_inches='tight')
    plt.close()

    mesh = generate_mesh(
            comm_mesh,
            rect_width=model_settings["aspect_ratio"],
            rect_height=1.0,
            stretch=stretch,
            rotation=rotation,
            density=model_settings["cell_density"],
            refine_factor=2.5, 
            corridor_refine_factor=1.5
        )
    model, observation_times, bc, bc0 = ViscoElasticModel(mesh, model_settings, loading_position, 
                                                          image_corners_coords, reference_image, reference_mask, targets)
    Vh = model.problem.Vh
    print("State space dimension: ", Vh[hp.STATE].dim())
    print("Parameter space dimension: ", Vh[hp.PARAMETER].dim())
    dl.plot(Vh[hp.STATE].mesh())
    plt.savefig(f"{output_dir}/mesh.png", dpi=300, bbox_inches='tight')
    plt.close()

    x_true = model.generate_vector()

    m_true_array = np.random.normal(size=Vh[hp.PARAMETER].dim())
    gmc.set_global(comm_mesh, m_true_array, x_true[hp.PARAMETER])
    bounds = set_bounds_for_parameters(model_settings)
    m_transformed = extract_parameters(Vh[hp.PARAMETER], x_true[hp.PARAMETER], bounds)
    print("Transformed parameters: ", gmc.get_global(comm_mesh, m_transformed))
    time_start = time.time()
    model.solveFwd(x_true[hp.STATE], x_true)
    print("Time to solve forward problem: ", time.time() - time_start)
    plot_time = np.linspace(0, model_settings["total_time"], 21)
    for ii, t in enumerate(plot_time):
        snapshot_func = hp.vector2Function(x_true[hp.STATE].view(t), Vh[hp.STATE])
        cbar = dl.plot(snapshot_func.sub(0), mode="displacement", vmin=0.0, vmax=scale)
        plt.axis("off")
        plt.savefig(f"{output_dir}/fwd_solution_{ii}.png", dpi=300, bbox_inches='tight')
        plt.close()

    force_idx, image_idx = generate_data_idx(model_settings, reference_image.shape)
    model.misfit.check_mask_idx = image_idx
    observable_prediction = model.misfit.observable.eval(x_true)
    noise_precision = generate_noise_model(model_settings, force_idx, image_idx)
    model.misfit.noise_precision = noise_precision
    data = model.misfit.generate_noisy_data(x_true)
    model.misfit.data = data

    plt.figure(figsize=(5, 4))
    plt.plot(observation_times, observable_prediction[force_idx], "-", label="Prediction", lw = 3, color = "k")
    plt.plot(observation_times, data[force_idx], "o", markersize= 6, label="Data", color="C2")
    plt.xlabel("Time")
    plt.ylabel("Force")
    plt.grid(":")
    plt.savefig(f"{output_dir}/force_data.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Use downsampled shape for image snapshots
    f = model_settings["high_resolution_factor"]
    H_ds = reference_mask.shape[0] // f
    W_ds = reference_mask.shape[1] // f
    lowres_size = H_ds * W_ds
    print("Image size: H =", H_ds, "W =", W_ds)

    for ii in range(model_settings["n_image_snapshots"]):
        start_idx = ii * lowres_size
        end_idx = start_idx + lowres_size

        image_indices = image_idx[start_idx:end_idx]
        deformed_image = data[image_indices].reshape((H_ds, W_ds))
        plt.imshow(deformed_image, origin="lower", cmap="gray")
        plt.axis("off")
        plt.savefig(f"{output_dir}/noisy_deformed_image_{ii}.png", dpi=300, bbox_inches='tight')
        plt.close()
    

    # m0 = x_true[hp.PARAMETER].copy()
    m0 = model.generate_vector(hp.PARAMETER)
    m0.zero()
    solver = BFGS(model)
    solver.parameters["rel_tolerance"] = 1e-6
    solver.parameters["abs_tolerance"] = 1e-3
    solver.parameters["LS"]["max_backtracking_iter"] = 50
    solver.parameters["max_iter"] = 500
    solver.parameters["print_level"] = 1 if comm_sampler.rank == 0 else -1
    
    # Enable L-BFGS with memory limit to utilize adaptive scaling
    solver.parameters["BFGS_op"]["memory_limit"] = 10
    
    # Fix initialization (remove incorrect init_vector argument)
    H0inv = RescaledIdentity()
    
    x_MAP, x_history, cost_history, gradnorm_history = solver.solve([None, m0, None], H0inv, out_frequency=5)
    print("MAP point: ", x_MAP[hp.PARAMETER].get_local())
    print("True parameters: ", x_true[hp.PARAMETER].get_local())
    print("Time to compute MAP point: ", time.time() - time_start)

    force_history = []
    param_history = []
    for x_iter in x_history:
        observables = model.misfit.observable.eval(x_iter)
        force_history.append(observables[force_idx])
        param_history.append(x_iter[hp.PARAMETER].get_local())

    model.misfit.setLinearizationPoint(x_MAP, gauss_newton_approx=True)
    jacobian = compute_pto_map_jacobian(model, x_MAP)
    gauss_newton_hessian = np.einsum("ji,j,jk->ik", jacobian, model.misfit.W, jacobian)

    output = {
        "force_history": np.stack(force_history),
        "parameter_history": np.stack(param_history),
        "MAP": x_MAP[hp.PARAMETER].get_local(),
        "cost": np.array(cost_history),
        "gradnorm": np.array(gradnorm_history),
        "hessian": gauss_newton_hessian,
        "true_parameter": x_true[hp.PARAMETER].get_local()
    }

    np.savez(f"{output_dir}/identification_output.npz", **output)
