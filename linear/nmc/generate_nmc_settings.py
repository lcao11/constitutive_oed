# Move these environment settings to the very top of the file
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

import math
# Bounds (low, high) for ONE design vector
DESIGN_BOUNDS = [
    (-0.5 * math.pi, 0.5 * math.pi),
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

    N_SAMPLES = 5  # Number of design samples to generate
    DIR_NAME = "./settings/"
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME, exist_ok=True)

    with open("../model_config.pkl", "rb") as f:
        config_data = pickle.load(f)
        model_settings = config_data["model_settings"]
        center = config_data["speckle_centers"]
        radius = config_data["speckle_radii"]
        image_corner_coords = config_data["image_corners_coords"]

    np.random.seed(model_settings["seed"] + 1024)

    scale = model_settings["max_strain"] * model_settings["aspect_ratio"]

    lows = np.array([b[0] for b in DESIGN_BOUNDS], dtype=np.float64)
    highs = np.array([b[1] for b in DESIGN_BOUNDS], dtype=np.float64)
    dim = len(DESIGN_BOUNDS)

    sobol_seed = model_settings["seed"] + 1024
    sobol = qmc.Sobol(d=dim, scramble=True, seed=sobol_seed)
    # random_base2 requires n_samples to be a power of 2; take first N_SAMPLES
    m = int(np.ceil(np.log2(max(1, N_SAMPLES))))
    u = sobol.random_base2(m)[:N_SAMPLES]

    design_samples = lows + (highs - lows) * u

    for ii in range(N_SAMPLES):
        design = design_samples[ii]
        rotation = design[0]
        stretch = (0.35, design[1])
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
        mesh_file = dl.File(f"{DIR_NAME}/mesh_{ii}.xml")
        mesh_file << mesh
    
    with open(f"{DIR_NAME}/settings.pkl", "wb") as f:
        import pickle
        pickle.dump({
            "design_samples": design_samples
        }, f)