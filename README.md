# Constitutive OED
Bayesian optimal experimental design (OED) for learning history-dependent constitutive models.

This repository contains the code for results in the arXiv paper: https://arxiv.org/abs/2603.12365
The mathematical formulation and problem settings are described in the paper.

## Repository Guide
- `linear/`: linear viscoelastic model, OED, surrogate training, and NMC utilities.
- `nonlinear/`: nonlinear counterpart (model, surrogate, and data generation).
- `utils/`: shared FEM, observation, misfit, and plotting helpers.
- `environment.yml`: conda environment used for FEniCS/hippylib workflows.

Key entry points (Python scripts):
- `linear/design/run_design_optimization.py`: Bayesian optimization over design variables.
- `linear/surrogate/run_data_generation.py`: generate surrogate datasets.
- `linear/surrogate/run_surrogate_training.py`: train surrogate models.
- `linear/nmc/run_nmc_comparison.py`: nested Monte Carlo EIG estimates.
- `nonlinear/surrogate/*`: analogous workflows for nonlinear models.

## Installation
### Conda environment
```bash
conda env create -f environment.yml
conda activate fenics-2019
```

### hippylib and geometric_mcmc
These are expected to be available as source trees. Set environment variables to their locations:
```bash
export HIPPYLIB_PATH=/path/to/hippylib
export GMC_PATH=/path/to/geometric_mcmc
```
Most scripts import these paths at runtime.

External repositories:
- [hippylib](https://github.com/hippylib/hippylib)
- [geometric_mcmc](https://github.com/dinoSciML/geometric_mcmc)

## Running the Demos
Many scripts are MPI-enabled. `mpirun -n 4` launches 4 MPI processes (ranks). In this repo that typically means:
- more parallel parameter samples when estimating EIG in the BO loop,
- more parallel samples during data generation for surrogate FIM,
- more parallel work for the nested Monte Carlo estimator.
Adjust `4` to match your available CPU cores or the number of samples you want in parallel.
A typical linear workflow is:
```bash
# 1) Build the model configuration and speckle pattern used by the forward model
python linear/generate_settings.py

# 2) Search for an informative experiment design via Bayesian optimization
mpirun -n 4 python linear/design/run_design_optimization.py --output_path ./design_result/

# 3) Sample many designs/parameters to build training data for the surrogate
mpirun -n 4 python linear/surrogate/run_data_generation.py --output_path ./data_set/
# Train a surrogate model that predicts FIMs quickly for design screening
python linear/surrogate/run_surrogate_training.py --data_path ./data_set --save_dir ./results

# 4) (Optional) NMC evaluation to estimate EIG for selected designs
#    generate_nmc_settings prepares fixed meshes and design settings for NMC runs
python linear/nmc/generate_nmc_settings.py
#    run_nmc_comparison runs the nested Monte Carlo estimator for one design index
mpirun -n 4 python linear/nmc/run_nmc_comparison.py --design_index 0 --outer_samples_per_process 1 --inner_samples 512
```

For nonlinear demos, use the matching scripts under `nonlinear/`.

## Citation
If you use this code, please cite the paper:
```bibtex
@article{bhattacharya2026optimal,
	title={Optimal Experimental Design for Reliable Learning of History-Dependent Constitutive Laws}, 
	author={Kaushik Bhattacharya and Lianghao Cao and Andrew Stuart},
	year={2026},
	journal={arXiv preprint},
	doi={10.48550/arXiv.2603.12365}
}
```

## Notes
- Scripts that write outputs state expected files in their module headers.
- `model_config.pkl` files under `linear/` and `nonlinear/` are tracked and used as defaults.
