"""Generate linear model_config.pkl and speckle pattern settings.

Usage:
    python linear/generate_settings.py

Expected output:
    - model_config.pkl in the linear directory.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ufl")
import os, sys
import jax
import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt
import pickle
import datetime
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
from linear_viscoelasticity import ViscoElasticSettings
jax.config.update("jax_enable_x64", True)

if __name__=="__main__":

    model_settings = ViscoElasticSettings()
    model_settings["cell_density"] = 32
    model_settings["n_time_steps"] = 100
    model_settings["seed"] = 0
    model_settings["pixel_density"] = 500
    model_settings["window_size"] = 3
    model_settings["max_strain"] = 0.05
    model_settings["n_control_points"] = 10

    scale = model_settings["max_strain"] * model_settings["aspect_ratio"]

    image_corners_coords = np.array([[-0.5*model_settings["aspect_ratio"], -0.53], [0.5*model_settings["aspect_ratio"] + scale, 0.53]])
    center, radius = generate_speckle_pattern(image_corners_coords, density=0.5, base_speckle_radius=0.006)

    config_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_settings": dict(model_settings),
        "speckle_centers": center,
        "speckle_radii": radius,
        "image_corners_coords": image_corners_coords
    }

    config_filename = os.path.join("model_config.pkl")
    with open(config_filename, "wb") as f:
        pickle.dump(config_data, f)
    
    print(f"Configuration saved to {config_filename}")

