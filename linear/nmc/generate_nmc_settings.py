"""Generate NMC mesh, loading plots, and settings samples.

Usage:
    python linear/nmc/generate_nmc_settings.py

Expected output:
    - settings/mesh_*.xml, settings/mesh_*.png, and settings.pkl.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ufl")
import math
import os, sys
import jax
import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt
from scipy.stats import qmc
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
jax.config.update("jax_enable_x64", True)

import matplotlib
try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=30)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
except:
    pass
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)
from mpi4py import MPI
import pickle
from scipy.interpolate import PchipInterpolator

DESIGN_BOUNDS = [
    (-0.5 * math.pi, 0.5 * math.pi),
    (0.1, 0.35),
    (0.1, 0.35),
    (0.0, 0.1),
    (0.0, 0.1),
    (0.0, 0.1),
    (0.0, 0.1),
    (0.0, 0.1),
    (0.0, 0.1),
    (0.0, 0.1),
    (0.0, 0.1),
    (0.0, 0.1),
    (0.0, 0.1),
]

if __name__ == "__main__":

    N_SAMPLES = 4  # Number of design samples to generate
    DIR_NAME = "./settings/"
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME, exist_ok=True)

    with open("../model_config.pkl", "rb") as f:
        config_data = pickle.load(f)
        model_settings = config_data["model_settings"]
        center = config_data["speckle_centers"]
        radius = config_data["speckle_radii"]
        image_corner_coords = config_data["image_corners_coords"]

    model_settings["cell_density"] = 25

    np.random.seed(model_settings["seed"] + 1024)

    scale = model_settings["max_strain"] * model_settings["aspect_ratio"]

    lows = np.array([b[0] for b in DESIGN_BOUNDS], dtype=np.float64)
    highs = np.array([b[1] for b in DESIGN_BOUNDS], dtype=np.float64)
    dim = len(DESIGN_BOUNDS)

    sobol_seed = model_settings["seed"] + 1024
    sobol = qmc.Sobol(d=dim, scramble=True, seed=sobol_seed)
    m = int(np.ceil(np.log2(max(1, N_SAMPLES))))
    u = sobol.random_base2(m)[:N_SAMPLES]

    design_samples = lows + (highs - lows) * u

    for ii in range(N_SAMPLES):
        design = design_samples[ii]
        rotation = design[0]
        stretch = (design[1], design[2])
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
        dl.plot(mesh)
        plt.axis("off")
        plt.savefig(f"{DIR_NAME}/mesh_{ii}.png", dpi=300, bbox_inches='tight')
        plt.close()
        mesh_file = dl.File(f"{DIR_NAME}/nmc_mesh_{ii}.xml")
        mesh_file << mesh

        scale = model_settings["max_strain"] * model_settings["aspect_ratio"]
        control_time = np.linspace(0, model_settings["total_time"], model_settings["n_control_points"]+1)
        control_value = np.zeros((control_time.shape[0], 2))
        control_value[1:, 0] = design[3:]
        loading_position = PchipInterpolator(control_time, control_value)
        plot_time = np.linspace(0, model_settings["total_time"], 1000)
        plt.figure(figsize=(5, 4))
        plt.plot(plot_time, loading_position(plot_time)[:, 0]/model_settings["aspect_ratio"], label="Loading Position", lw = 3, color = "k")
        plt.plot(control_time[1:], control_value[1:, 0]/model_settings["aspect_ratio"], "o", markersize= 10, label="Control Points", color="C3")
        plt.xlabel(r"Time")
        plt.ylabel(r"Imposed Strain ($\%$)")
        plt.ylim(0, model_settings["max_strain"] * 1.05)
        tick_positions = np.linspace(0, model_settings["max_strain"], 3)
        tick_labels = [f"{int(x * 100)}" for x in tick_positions]
        plt.yticks(tick_positions, labels=tick_labels)
        plt.grid(":")
        plt.savefig(f"{DIR_NAME}/nmc_loading_{ii}.png", dpi=200, bbox_inches='tight')
        plt.close()
    
    with open(f"{DIR_NAME}/settings.pkl", "wb") as f:
        import pickle
        pickle.dump({
            "design_samples": design_samples
        }, f)