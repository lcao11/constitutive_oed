import argparse
import math
import json
import os
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from architecture import load_fim_model

# Design variable bounds based on run_design_optimization.py
# 12 dimensions: [rotation, stretch_y, 10 controls...]
# Note: run_design_optimization.py defines bounds for x0 (rotation), x1 (stretch), and x2..xN (controls)
# We assume the model was trained with these 12 inputs.

def get_design_bounds(n_controls=10, max_strain=0.1, aspect_ratio=1.0):
    """
    Returns the physical bounds for the design variables.
    Matches run_design_optimization.py logic.
    """
    # x0: Rotation [-pi/2, pi/2]
    bounds = [(-0.5 * math.pi, 0.5 * math.pi)]
    
    # x1: Stretch Y [0.1, 0.35]
    bounds.append((0.1, 0.35))
    
    # x2...x11: Controls [0.0, max_strain * aspect_ratio]
    # Default max_strain=0.1, aspect_ratio=1.0 implies upper bound 0.1
    control_upper = max_strain * aspect_ratio
    for _ in range(n_controls):
        bounds.append((0.0, control_upper))
        
    return bounds

def generate_param_samples(n_samples: int, n_params: int, seed: int, device: torch.device) -> torch.Tensor:
    """
    Generates parameter samples from the prior N(0, I).
    Matches the QMC logic in run_design_optimization.py: Sobol -> Inverse CDF.
    """
    # 1. Generate uniform samples in [0, 1]^d using Sobol sequence
    sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)
    
    # Get next power of 2 for Sobol efficiency
    m = int(np.ceil(np.log2(n_samples)))
    u_samples = sampler.random_base2(m)[:n_samples]
    
    # 2. Transform to standard normal distribution N(0, I) via inverse CDF (probit)
    normal_samples = norm.ppf(u_samples)
    
    return torch.tensor(normal_samples, dtype=torch.float32, device=device)

def sobol_design_starts(n: int, bounds: list, seed: int) -> np.ndarray:
    """Generates N initial design points using Sobol sequence within bounds."""
    d = len(bounds)
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(n)))
    u = sampler.random_base2(m)[:n]
    
    lows = np.array([b[0] for b in bounds])[None, :]
    highs = np.array([b[1] for b in bounds])[None, :]
    
    return lows + u * (highs - lows)

def expected_utility(model, params, design_flat, n_designs, design_dim):
    """
    Computes the Expected Utility for a set of designs.
    Matches evaluate_design in run_design_optimization.py:
    U(d) = 0.5 * E_theta [ log(det(I + FIM)) + tr((I + FIM)^-1) + ||theta||^2 ]
    
    Since we optimize w.r.t. d, the ||theta||^2 term is constant and ignored.
    
    Args:
        model: The trained FIMModel.
        params: (M, param_dim) tensor of parameter samples.
        design_flat: (n_designs * design_dim,) tensor of flattened design variables.
        n_designs: Number of designs in the set.
        design_dim: Dimension of a single design.
        
    Returns:
        Scalar tensor representing the objective to MAXIMIZE.
    """
    M = params.size(0)
    D = int(n_designs)
    d = int(design_dim)

    # Reshape designs: (D, d)
    designs = design_flat.view(D, d)
    
    # Expand for batch processing: (D * M, d)
    design_expanded = designs.unsqueeze(1).expand(D, M, d).reshape(D * M, d)
    params_expanded = params.unsqueeze(0).expand(D, M, -1).reshape(D * M, -1)

    # Concatenate: [params, designs]
    x = torch.cat([params_expanded, design_expanded], dim=1)

    # Predict FIMs: (D*M, n, n)
    FIM_all = model.forward_to_FIM(x)

    # Reshape back to (D, M, n, n)
    n_matrix = FIM_all.shape[-1]
    FIM_view = FIM_all.view(D, M, n_matrix, n_matrix)
    
    # Sum FIMs over designs (D dimension) -> (M, n, n)
    # This represents the total information from the set of experiments.
    FIM_total = FIM_view.sum(dim=0)
    
    # Symmetrize
    FIM_total = 0.5 * (FIM_total + FIM_total.transpose(-1, -2))
    
    # Eigenvalues of FIM_total
    # Use float64 for precision
    eigs, _ = torch.linalg.eigh(FIM_total.double())
    eigs = torch.maximum(eigs, torch.tensor(0.0, device=eigs.device, dtype=eigs.dtype))
    
    # Compute Utility Terms
    # 1. log(det(I + FIM)) = sum(log(1 + lambda))
    term1 = torch.log1p(eigs).sum(dim=-1)
    
    # 2. tr((I + FIM)^-1) = sum(1 / (1 + lambda))
    term2 = (1.0 / (1.0 + eigs)).sum(dim=-1)
    
    # Total Utility (ignoring constant ||theta||^2 term)
    utility = 0.5 * (term1 + term2)
    
    return utility.mean()

def make_objective(model, param_samples, device, n_designs, design_dim, shared_geometry=False):
    """Creates the objective function and gradient for scipy.minimize."""
    
    def func(d_np):
        d_tensor = torch.tensor(d_np, dtype=torch.float32, device=device, requires_grad=True)
        
        if shared_geometry:
            # Reconstruct full design tensor (n_designs, design_dim)
            # d_np structure: [rot, stretch, c_1_0..c_1_9, c_2_0..c_2_9, ...]
            n_geo = 2
            n_controls = design_dim - n_geo
            
            rot = d_tensor[0]
            stretch = d_tensor[1]
            controls_flat = d_tensor[2:]
            
            # (n_designs, n_controls)
            controls = controls_flat.view(n_designs, n_controls)
            
            # Expand geo to (n_designs, 1)
            rot_exp = rot.expand(n_designs, 1)
            str_exp = stretch.expand(n_designs, 1)
            
            # Concatenate: [rot, str, controls]
            d_full = torch.cat([rot_exp, str_exp, controls], dim=1)
            
            # Flatten for expected_utility which expects (n_designs * design_dim)
            d_input = d_full.view(-1)
        else:
            d_input = d_tensor

        util = expected_utility(model, param_samples, d_input, n_designs, design_dim)
        
        # Maximize Utility -> Minimize Negative Utility
        loss = -util
        
        if torch.isnan(loss) or torch.isinf(loss):
            return 1e9, np.zeros_like(d_np)
            
        loss.backward()
        
        grad_np = d_tensor.grad.detach().cpu().numpy().astype(np.float64)
        loss_val = loss.item()
        
        return loss_val, grad_np
        
    return func

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing the trained model")
    parser.add_argument('--tag', type=str, default='best', help="Checkpoint tag (best/last)")
    parser.add_argument('--n_designs', type=int, default=3, help="Number of designs to optimize jointly")
    parser.add_argument('--n_starts', type=int, default=512, help="Number of random restarts for optimization")
    parser.add_argument('--mc_samples', type=int, default=1024, help="Number of MC samples for integration")
    parser.add_argument('--max_iter', type=int, default=200, help="Max iterations for L-BFGS-B")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_prefix', type=str, default='optimal_design', help="Prefix for output JSON file")
    parser.add_argument('--shared_geometry', action='store_true', help="Share rotation and stretch across designs")
    
    args = parser.parse_args()

    # Hardcoded Design Parameters
    N_CONTROLS = 10
    MAX_STRAIN = 0.1
    ASPECT_RATIO = 1.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    print(f"Loading model from {args.model_dir} (tag={args.tag})...")
    model, meta = load_fim_model(args.model_dir, tag=args.tag, device=device)
    model.eval()
    
    # Determine Dimensions
    input_size = meta['config']['input_size']
    design_bounds = get_design_bounds(N_CONTROLS, MAX_STRAIN, ASPECT_RATIO)
    design_dim = len(design_bounds)
    param_dim = input_size - design_dim
    
    print(f"Model Input Size: {input_size}")
    print(f"Inferred Design Dim: {design_dim}")
    print(f"Inferred Param Dim: {param_dim}")
    
    if param_dim <= 0:
        raise ValueError("Parameter dimension must be positive. Check n_controls or model config.")

    # Generate Parameter Samples (Fixed for optimization)
    print(f"Generating {args.mc_samples} parameter samples (Sobol)...")
    param_samples = generate_param_samples(args.mc_samples, param_dim, args.seed, device)
    
    # Optimization Setup
    if args.shared_geometry:
        # Bounds: [rot, str] + [c0..c9] * n_designs
        geo_bounds = design_bounds[:2]
        ctrl_bounds = design_bounds[2:]
        bounds_flat = geo_bounds + ctrl_bounds * args.n_designs
        print(f"Shared Geometry Mode: Optimizing {len(bounds_flat)} variables (2 shared + {args.n_designs}x{len(ctrl_bounds)} controls)")
    else:
        bounds_flat = design_bounds * args.n_designs
        print(f"Independent Mode: Optimizing {len(bounds_flat)} variables ({args.n_designs}x{len(design_bounds)})")

    objective_fn = make_objective(model, param_samples, device, args.n_designs, design_dim, args.shared_geometry)
    
    # Random Restarts
    best_val = -np.inf
    best_design = None
    
    starts = sobol_design_starts(args.n_starts, bounds_flat, seed=args.seed + 1)
    
    print(f"Starting optimization with {args.n_starts} restarts...")
    
    for i, x0 in enumerate(starts):
        res = minimize(
            objective_fn, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            bounds=bounds_flat,
            options={'maxiter': args.max_iter, 'disp': False}
        )
        
        val = -res.fun # Convert back to maximization
        
        if val > best_val:
            best_val = val
            best_design = res.x
            print(f"  [Start {i+1}] New best: {best_val:.4f}")
        else:
            if (i+1) % 5 == 0:
                print(f"  [Start {i+1}] Val: {val:.4f}")

    print(f"Optimization complete. Best Value: {best_val:.4f}")
    
    # Reconstruct full design for saving if shared
    if args.shared_geometry:
        n_geo = 2
        n_controls = design_dim - n_geo
        rot = best_design[0]
        stretch = best_design[1]
        controls = best_design[2:].reshape(args.n_designs, n_controls)
        
        best_design_reshaped = np.zeros((args.n_designs, design_dim))
        best_design_reshaped[:, 0] = rot
        best_design_reshaped[:, 1] = stretch
        best_design_reshaped[:, 2:] = controls
        best_design_reshaped_list = best_design_reshaped.tolist()
    else:
        best_design_reshaped_list = best_design.reshape(args.n_designs, design_dim).tolist()

    # Save Results
    results = {
        'best_value': float(best_val),
        'best_design_flat': best_design.tolist(),
        'best_design_reshaped': best_design_reshaped_list,
        'n_designs': args.n_designs,
        'mc_samples': args.mc_samples,
        'model_dir': args.model_dir,
        'design_bounds': design_bounds,
        'shared_geometry': args.shared_geometry
    }
    
    suffix = "_shared" if args.shared_geometry else ""
    output_filename = f"{args.output_prefix}_n{args.n_designs}{suffix}.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_filename}")

if __name__ == '__main__':
    main()
