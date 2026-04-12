# Move these environment settings to the very top of the file
import os, sys
import jax
import numpy as np
import dolfin as dl
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.append(os.environ.get('GMC_PATH'))
import geometric_mcmc as gmc
sys.path.append("../")
from utils import *
from nonlinear_viscoelasticity import NonlinearViscoElasticSettings
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)
import datetime
import pickle
jax.config.update("jax_enable_x64", True)

if __name__=="__main__":

    model_settings = NonlinearViscoElasticSettings()
    model_settings["cell_density"] = 34
    model_settings["n_time_steps"] = 200
    model_settings["total_time"] = 1.0
    model_settings["high_resolution_factor"] = 1
    model_settings["seed"] = 0

    scale = model_settings["max_strain"] * model_settings["aspect_ratio"]

    image_corners_coords = np.array([[-0.5*model_settings["aspect_ratio"], -0.6], [0.5*model_settings["aspect_ratio"] + scale, 0.6]])
    center, radius = generate_speckle_pattern(image_corners_coords, density=0.5, base_speckle_radius=0.007)

    # Prepare dictionary for saving
    config_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_settings": dict(model_settings), # Convert to standard dict if it's a custom object
        "speckle_centers": center,
        "speckle_radii": radius,
        "image_corners_coords": image_corners_coords
    }

    # Save to a pickle file
    config_filename = os.path.join("model_config.pkl")
    with open(config_filename, "wb") as f:
        pickle.dump(config_data, f)
    
    print(f"Configuration saved to {config_filename}")