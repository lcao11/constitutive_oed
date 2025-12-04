# Move these environment settings to the very top of the file
import math
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
from scipy.stats import qmc, norm
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

def design_bounds(upper_control_value):
    lower = [-0.5*math.pi, 0.1] + [0.0]*10
    upper = [0.5*math.pi, 0.35] + [upper_control_value]*10
    return lower, upper

def interpolate_loading_path(control_values):

    times = np.linspace(0, 1.0, len(control_values)+1)
    values = np.zeros((times.shape[0], 2))
    values[1:, 0] = control_values
    loading_position = PchipInterpolator(times, values)

    return loading_position

if __name__=="__main__":
    import argparse
    import pickle
    import time

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_path", type=str, default="./data_set/", help="Path to save the results of the OED")
    argparser.add_argument("--samples_per_process", type=int, default=100, help="Number of samples per process")
    argparser.add_argument("--process_id", type=int, default=0, help="Process ID for random seed")

    args = argparser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    with open("../model_config.pkl", "rb") as f:
        config_data = pickle.load(f)
        model_settings = config_data["model_settings"]
        center = config_data["speckle_centers"]
        radius = config_data["speckle_radii"]
        image_corner_coords = config_data["image_corners_coords"]

    base_seed = int(model_settings["seed"])

    scale = model_settings["max_strain"] * model_settings["aspect_ratio"]
    
    parameter_list = []
    design_list = []
    fisher_list = []

    # --- Sobol designs ---
    design_lower, design_upper = design_bounds(
        model_settings["max_strain"] * model_settings["aspect_ratio"]
    )
    design_lower = np.asarray(design_lower, dtype=float)
    design_upper = np.asarray(design_upper, dtype=float)
    design_dim = design_lower.size

    total_design_samples = args.samples_per_process * (args.process_id + 1)
    design_sampler = qmc.Sobol(d=design_dim, scramble=True, seed=base_seed)
    m_design = int(np.ceil(np.log2(max(1, total_design_samples))))
    design_u = design_sampler.random_base2(m_design)[:total_design_samples]

    start = args.process_id * args.samples_per_process
    end = start + args.samples_per_process
    design_u_proc = design_u[start:end]

    design_samples_all = design_lower + (design_upper - design_lower) * design_u_proc

    param_seed = base_seed + 12345

    time0 = time.time()
    param_sampler = None
    param_u_proc = None

    for ii in range(args.samples_per_process):
        design_samples = design_samples_all[ii]
        rotation = design_samples[0]
        stretch = np.array([0.35, design_samples[1]])
        mesh = generate_mesh(
                    MPI.COMM_WORLD,
                    rect_width=model_settings["aspect_ratio"],
                    rect_height=1.0,
                    stretch=stretch,
                    rotation=rotation,
                    density=model_settings["cell_density"],
                    refine_factor=2.5,
                    corridor_refine_factor=1.5
                )
        
        loading_position = interpolate_loading_path(design_samples[3:])
        inside = CheckInsideImage(model_settings["aspect_ratio"], stretch, rotation)
        reference_mask, targets = setup_image_observation(
            image_corner_coords,
            inside,
            model_settings["pixel_density"],
            oversampling_factor=model_settings["high_resolution_factor"]
        )
        reference_image = speckled_reference(reference_mask, image_corner_coords, center, radius)
        model, _, _, _ = ViscoElasticModel(mesh, model_settings, 
                                 loading_position, image_corner_coords, 
                                 reference_image, reference_mask, targets)
        Vh = model.problem.Vh
        param_dim = Vh[hp.PARAMETER].dim()
        force_idx, image_idx = generate_data_idx(model_settings, reference_image.shape)  # FIX: was global model_settings
        model.misfit.check_mask_idx = image_idx
        noise_precision = generate_noise_model(model_settings, force_idx, image_idx)
        model.misfit.noise_precision = noise_precision

        x_true = model.generate_vector()

        if param_sampler is None:
            total_param_samples = args.samples_per_process * (args.process_id + 1)
            param_sampler = qmc.Sobol(d=param_dim, scramble=True, seed=param_seed)
            m_param = int(np.ceil(np.log2(max(1, total_param_samples))))
            param_u = param_sampler.random_base2(m_param)[:total_param_samples]

            start_p = args.process_id * args.samples_per_process
            end_p = start_p + args.samples_per_process
            param_u_proc = param_u[start_p:end_p]

        u_vec = param_u_proc[ii]
        parameter_sample = norm.ppf(u_vec)

        gmc.set_global(MPI.COMM_WORLD, parameter_sample, x_true[hp.PARAMETER])
        model.solveFwd(x_true[hp.STATE], x_true)
        model.misfit.setLinearizationPoint(x_true, gauss_newton_approx=True)
        jacobian = compute_pto_map_jacobian(model, x_true)
        fims = np.einsum("ji, j, jk->ik", jacobian, model.misfit.W, jacobian)

        parameter_list.append(parameter_sample)
        design_list.append(design_samples)
        fisher_list.append(fims)
        per_sample_min = (time.time()-time0)/(ii+1)/60.0
        print(f"Finished sample {ii+1}/{args.samples_per_process}, took {per_sample_min:.2f} minutes per sample")

    parameter_array = np.stack(parameter_list)
    design_array = np.stack(design_list)
    fim_array = np.stack(fisher_list)

    data_set = {
        "process_id": args.process_id,
        "parameters": parameter_array,
        "designs": design_array,
        "fims": fim_array
    }

    with open(os.path.join(args.output_path, f"data_{args.process_id}.pkl"), "wb") as f:
        pickle.dump(data_set, f)