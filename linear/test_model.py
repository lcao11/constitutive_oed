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
sys.path.append("../")
from utils import *
from linear_viscoelasticity import ViscoElasticSettings, ViscoElasticModel, CheckInsideImage, generate_data_idx, generate_noise_model, set_bounds_for_parameters, extract_parameters
from scipy.interpolate import PchipInterpolator
jax.config.update("jax_enable_x64", True) # Use 64-bit precision

import matplotlib
try:
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif', size=20)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
except:
    pass
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)
from mpi4py import MPI
import time

if __name__=="__main__":

    comm_mesh, comm_sampler = gmc.split_mpi_comm(MPI.COMM_WORLD, 1, MPI.COMM_WORLD.size)

    model_settings = ViscoElasticSettings()
    model_settings["cell_density"] = 32
    model_settings["n_time_steps"] = 100
    model_settings["seed"] = 0
    model_settings["pixel_density"] = 500
    model_settings["window_size"] = 3
    model_settings["max_strain"] = 0.05
    model_settings["n_control_points"] = 10

    visual_dir = f"./test_output/"
    os.makedirs(visual_dir, exist_ok=True)

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
    plt.savefig(f"{visual_dir}/loading_position.png", dpi=300, bbox_inches='tight')
    plt.close()
    rotation = np.random.uniform(-np.pi/2, np.pi/2)
    stretch = np.zeros(2)
    stretch[0] = 0.35
    stretch[1] = np.random.uniform(0.1, 0.35)

    image_corners_coords = np.array([[-0.5*model_settings["aspect_ratio"], -0.53], [0.5*model_settings["aspect_ratio"] + scale, 0.53]])
    inside = CheckInsideImage(model_settings["aspect_ratio"], stretch, rotation)

    reference_mask, targets = setup_image_observation(image_corners_coords, inside, model_settings["pixel_density"], oversampling_factor=model_settings["high_resolution_factor"])
    center, radius = generate_speckle_pattern(image_corners_coords, density=0.5, base_speckle_radius=0.006)
    reference_image = speckled_reference(reference_mask, image_corners_coords, center, radius)

    plt.imshow(reference_mask, origin="lower", cmap="gray")
    plt.axis("off")
    plt.savefig(f"{visual_dir}/reference_masks.png", dpi=300, bbox_inches='tight')

    plt.close()
    plt.imshow(reference_image, origin="lower", cmap="gray")
    plt.axis("off")
    plt.savefig(f"{visual_dir}/reference_image.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.imshow(np.random.normal(size=reference_mask.shape), origin="lower", cmap="gray")
    plt.axis("off")
    plt.savefig(f"{visual_dir}/noise_image.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{visual_dir}/mesh.png", dpi=300, bbox_inches='tight')
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
        plt.savefig(f"{visual_dir}/fwd_solution_{ii}.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{visual_dir}/force_data.png", dpi=300, bbox_inches='tight')
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
        plt.savefig(f"{visual_dir}/noisy_deformed_image_{ii}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    model.misfit.setLinearizationPoint(x_true, gauss_newton_approx=True)
    jacobian = compute_pto_map_jacobian(model, x_true)
    weighted_jacobian = np.einsum("ij,i->ij", jacobian, np.sqrt(model.misfit.W))
    for ii in range(Vh[hp.PARAMETER].dim()):
        sensitivity_dir = f"{visual_dir}/sensitivity_{ii}/"
        os.makedirs(sensitivity_dir, exist_ok=True)
        plt.figure(figsize=(5, 4))
        plt.plot(observation_times, weighted_jacobian[force_idx, ii], "o", markersize= 6, label="Data", color="C2")
        plt.xlabel("Time")
        plt.ylabel("Force Variation")
        plt.grid(":")
        plt.savefig(f"{sensitivity_dir}/force_variation.png", dpi=300, bbox_inches='tight')
        plt.close()
        for jj in range(model_settings["n_image_snapshots"]):
            start_idx = jj * lowres_size
            end_idx = start_idx + lowres_size

            image_indices = image_idx[start_idx:end_idx]
            deformed_image = weighted_jacobian[image_indices, ii].reshape((H_ds, W_ds))
            plt.imshow(deformed_image, origin="lower", cmap="gray")
            plt.axis("off")
            plt.savefig(f"{sensitivity_dir}/image_variation_{jj}.png", dpi=300, bbox_inches='tight')
            plt.close()

    gauss_newton_hessian = np.einsum("ji,jk->ik", weighted_jacobian, weighted_jacobian)

    eigenvalues, eigenvectors = np.linalg.eigh(gauss_newton_hessian)
    post_cov = np.linalg.inv(gauss_newton_hessian + np.diag(np.ones(Vh[hp.PARAMETER].dim())))
    print(eigenvalues)
    ig = 0.5*np.sum((np.log1p(eigenvalues) - eigenvalues/(1.0+eigenvalues))) + 0.5*np.inner(m_true_array, m_true_array)
    print("IG: ", ig)
    m_true_array = gmc.get_global(comm_mesh, x_true[hp.PARAMETER])
    samples = np.random.multivariate_normal(mean=m_true_array, cov=post_cov, size=2500)

    plot_joint_density(samples, reference=m_true_array, scatter=False)
    plt.savefig(f"{visual_dir}/joint_density_true.png", dpi=300, bbox_inches='tight')