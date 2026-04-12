"""Optimize designs using a trained surrogate model for EIG.

Usage:
    python linear/surrogate/run_eig_max.py --model_dir ./results --n_designs 1

Expected output:
    - optimal_design_*.json in the current working directory.
"""

import argparse
import json
import math
import os
from typing import List, Tuple, Union, Any

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import norm, qmc

from architecture import load_fim_model

# --- Configuration ---
N_CONTROLS = 10
MAX_STRAIN = 0.1
ASPECT_RATIO = 1.0
MIN_STRETCH = 0.1
MAX_STRETCH = 0.35

def get_design_bounds(n_controls: int = 10, max_strain: float = 0.1, aspect_ratio: float = 1.0) -> List[Tuple[float, float]]:
    """Returns physical bounds for design variables."""
    # x0: Rotation [-pi/2, pi/2]
    bounds = [(-0.5 * math.pi, 0.5 * math.pi)]
    # x1: Stretch X [MIN_STRETCH, MAX_STRETCH]
    bounds.append((MIN_STRETCH, MAX_STRETCH))
    # x2: Stretch Y [MIN_STRETCH, MAX_STRETCH]
    bounds.append((MIN_STRETCH, MAX_STRETCH))
    # x3...x12: Controls
    control_upper = max_strain * aspect_ratio
    for i in range(n_controls):
        # First control point has a minimum of 0.1, others 0.0
        lower = 0.1 if i == 0 else 0.0
        bounds.append((lower, control_upper))
    return bounds

def generate_param_samples(n_samples: int, n_params: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generates parameter samples from prior N(0, I) using Sobol sequence."""
    sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(n_samples)))
    u_samples = sampler.random_base2(m)[:n_samples]
    normal_samples = norm.ppf(u_samples)
    return torch.tensor(normal_samples, dtype=torch.float32, device=device)

def sobol_design_starts(n: int, bounds: List[Tuple[float, float]], seed: int) -> np.ndarray:
    """Generates N initial design points using Sobol sequence."""
    d = len(bounds)
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(n)))
    u = sampler.random_base2(m)[:n]
    
    lows = np.array([b[0] for b in bounds])[None, :]
    highs = np.array([b[1] for b in bounds])[None, :]
    
    return lows + u * (highs - lows)

def expand_shared_design(x: torch.Tensor, n_designs: int, design_dim: int) -> torch.Tensor:
    """
    Expands a compressed design vector (shared geometry) to the full design vector.
    
    Args:
        x: (B, compressed_dim) or (compressed_dim,) tensor.
            Structure: [rot, stretch_x, stretch_y, c_0_0, ..., c_0_{C-1}, c_1_0, ...]
            where rot, stretch_x, and stretch_y are shared.
        n_designs: Number of designs (D).
        design_dim: Dimension of a single design (d).
        
    Returns:
        (B, n_designs * design_dim) or (n_designs * design_dim,) tensor.
    """
    is_batch = x.dim() == 2
    if not is_batch:
        x = x.unsqueeze(0)
    
    B = x.shape[0]
    
    # x structure: [rot, stretch_x, stretch_y, controls_flat]
    rot = x[:, 0:1].unsqueeze(1).expand(B, n_designs, 1) # (B, D, 1)
    str_x = x[:, 1:2].unsqueeze(1).expand(B, n_designs, 1) # (B, D, 1)
    str_y = x[:, 2:3].unsqueeze(1).expand(B, n_designs, 1) # (B, D, 1)
    
    n_geo = 3
    n_controls = design_dim - n_geo
    
    # controls are remaining elements, reshaped to (B, D, C)
    ctrl = x[:, 3:].view(B, n_designs, n_controls) 
    
    full = torch.cat([rot, str_x, str_y, ctrl], dim=2) # (B, D, d)
    full = full.view(B, -1) # (B, D*d)
    
    if not is_batch:
        return full.squeeze(0)
    return full

def compute_utility(model, params: torch.Tensor, designs: torch.Tensor, n_designs: int, design_dim: int) -> torch.Tensor:
    """
    Computes Expected Utility for a batch of designs.
    
    Args:
        model: The FIM surrogate model.
        params: (M, param_dim) Parameter samples.
        designs: (B, n_designs * design_dim) or (n_designs * design_dim,) Design vectors.
        n_designs: Number of designs per experiment.
        design_dim: Dimension of a single design.
        
    Returns:
        (B,) tensor of utilities or scalar if input was 1D.
    """
    is_batch = designs.dim() == 2
    if not is_batch:
        designs = designs.unsqueeze(0) # (1, D*d)
    
    B = designs.shape[0]
    M = params.shape[0]
    D = n_designs
    d = design_dim

    # Reshape: (B, D, d)
    des_view = designs.view(B, D, d)
    
    # Expand for broadcasting: (B, D, M, d)
    des_exp = des_view.unsqueeze(2).expand(B, D, M, d).reshape(B * D * M, d)
    
    # Expand params: (B, D, M, p)
    par_exp = params.unsqueeze(0).unsqueeze(0).expand(B, D, M, -1).reshape(B * D * M, -1)
    
    # Forward pass
    x = torch.cat([par_exp, des_exp], dim=1)
    FIM_all = model.forward_to_FIM(x) # (B*D*M, n, n)
    
    n_matrix = FIM_all.shape[-1]
    FIM_view = FIM_all.view(B, D, M, n_matrix, n_matrix)
    
    # Sum over designs (D) -> (B, M, n, n)
    FIM_total = FIM_view.sum(dim=1)
    
    # Symmetrize
    FIM_total = 0.5 * (FIM_total + FIM_total.transpose(-1, -2))
    
    # Eigenvalues
    eigs, _ = torch.linalg.eigh(FIM_total.double())
    eigs = torch.maximum(eigs, torch.tensor(0.0, device=eigs.device, dtype=eigs.dtype))
    
    # Utility: 0.5 * logdet(FIM + I)
    term1 = torch.log1p(eigs).sum(dim=-1)
    utility_samples = 0.5 * (term1) # (B, M)
    
    # Expectation over parameters
    utility = utility_samples.mean(dim=1) # (B,)
    
    if not is_batch:
        return utility.squeeze(0)
    return utility

def optimize_design(model, param_samples: torch.Tensor, x0: torch.Tensor, bounds: torch.Tensor, 
                   n_designs: int, design_dim: int, shared_geometry: bool, max_iter: int) -> Tuple[torch.Tensor, float]:
    
    lower = bounds[:, 0].cpu().numpy()
    upper = bounds[:, 1].cpu().numpy()
    
    # Ensure start point is strictly within bounds to avoid immediate errors
    x0_np = np.clip(x0.detach().cpu().numpy(), lower, upper)

    def obj_func(x_np):
        # 1. Create tensor with gradient enabled
        x = torch.tensor(x_np, dtype=torch.float32, device=param_samples.device, requires_grad=True)
        
        # 2. Reconstruct input
        if shared_geometry:
            d_input = expand_shared_design(x, n_designs, design_dim)
        else:
            d_input = x
            
        # 3. Compute utility
        util = compute_utility(model, param_samples, d_input, n_designs, design_dim)
        loss = -util # Minimize negative utility
        
        # 4. Compute gradient
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()
        
        # 5. Return value and gradient as float64 for SciPy
        val = loss.item()
        grad = x.grad.cpu().numpy().astype(np.float64)
        
        return val, grad

    res = minimize(
        obj_func,
        x0_np,
        method="L-BFGS-B",
        jac=True, # Tell SciPy we are providing the gradient
        bounds=list(zip(lower, upper)),
        options={"maxiter": max_iter}
    )
    
    x_opt = np.clip(res.x, lower, upper).astype(np.float32)
    return torch.from_numpy(x_opt).to(param_samples.device), -res.fun

def round_floats(obj: Any, decimals: int = 6) -> Any:
    """Recursively rounds floats in a nested structure."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, (np.float32, np.float64)):
        return round(float(obj), decimals)
    if isinstance(obj, list):
        return [round_floats(x, decimals) for x in obj]
    if isinstance(obj, np.ndarray):
        return round_floats(obj.tolist())
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--tag', type=str, default='best')
    parser.add_argument('--n_designs', type=int, default=1)
    parser.add_argument('--n_screen', type=int, default=131072, help="Number of initial screening points")
    parser.add_argument('--n_fine', type=int, default=512, help="Number of points for fine optimization (L-BFGS)")
    parser.add_argument('--mc_samples', type=int, default=256)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1995)
    parser.add_argument('--output_prefix', type=str, default='optimal_design')
    parser.add_argument('--shared_geometry', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {args.model_dir}...")
    model, meta = load_fim_model(args.model_dir, tag=args.tag, device=device)
    model.eval()
    
    input_size = meta['config']['input_size']
    design_bounds = get_design_bounds(N_CONTROLS, MAX_STRAIN, ASPECT_RATIO)
    design_dim = len(design_bounds)
    param_dim = input_size - design_dim
    
    param_samples = generate_param_samples(args.mc_samples, param_dim, args.seed, device)
    param_samples.requires_grad_(False)
    
    if args.shared_geometry:
        bounds_flat = design_bounds[:3] + design_bounds[3:] * args.n_designs
    else:
        bounds_flat = design_bounds * args.n_designs
    
    bounds_tensor = torch.tensor(bounds_flat, dtype=torch.float32, device=device)
    bounds_tensor.requires_grad_(False)
    
    print(f"Screening {args.n_screen} starts on GPU...")
    starts_np = sobol_design_starts(args.n_screen, bounds_flat, seed=args.seed + 1)
    starts_tensor = torch.tensor(starts_np, dtype=torch.float32, device=device)
    
    if args.shared_geometry:
        rot = starts_tensor[:, 0:1].unsqueeze(1).expand(-1, args.n_designs, 1)
        str_x = starts_tensor[:, 1:2].unsqueeze(1).expand(-1, args.n_designs, 1)
        str_y = starts_tensor[:, 2:3].unsqueeze(1).expand(-1, args.n_designs, 1)
        ctrl = starts_tensor[:, 3:].view(args.n_screen, args.n_designs, -1)
        full_designs = torch.cat([rot, str_x, str_y, ctrl], dim=2).view(args.n_screen, -1)
    else:
        full_designs = starts_tensor

    chunk_size = 256
    utilities = []
    with torch.no_grad():
        for i in range(0, args.n_screen, chunk_size):
            batch = full_designs[i:i+chunk_size]
            utilities.append(compute_utility(model, param_samples, batch, args.n_designs, design_dim))
    start_utilities = torch.cat(utilities)
    
    top_vals, top_idxs = torch.topk(start_utilities, min(args.n_fine, args.n_screen))
    print(f"Best start utility (Screening): {top_vals[0]:.4f}")
    
    print(f"Running Fine Optimization (L-BFGS) on {len(top_idxs)} designs...")
    
    best_val = -np.inf
    best_design = None
    
    fine_starts = starts_tensor[top_idxs]
    
    for i, x0 in enumerate(fine_starts):
        opt_design, val = optimize_design(
            model, param_samples, x0, bounds_tensor, 
            args.n_designs, design_dim, args.shared_geometry, args.max_iter
        )
        
        if val > best_val:
            best_val = val
            best_design = opt_design.cpu().numpy()
            
            lower_bounds_np = bounds_tensor[:, 0].cpu().numpy()
            upper_bounds_np = bounds_tensor[:, 1].cpu().numpy()
            best_design = np.clip(best_design, lower_bounds_np, upper_bounds_np)
            best_design = np.round(best_design, 6)

            print(f"  [Fine Opt {i+1}/{len(fine_starts)}] New Best: {best_val:.4f}")
            
            if args.shared_geometry:
                n_geo = 3
                n_controls = design_dim - n_geo
                rot = best_design[0]
                stretch_x = best_design[1]
                stretch_y = best_design[2]
                controls = best_design[3:].reshape(args.n_designs, n_controls)
                
                best_design_reshaped = np.zeros((args.n_designs, design_dim))
                best_design_reshaped[:, 0] = rot
                best_design_reshaped[:, 1] = stretch_x
                best_design_reshaped[:, 2] = stretch_y
                best_design_reshaped[:, 3:] = controls
                best_design_list = best_design_reshaped.tolist()
            else:
                best_design_list = best_design.reshape(args.n_designs, design_dim).tolist()

            results = {
                'best_value': float(best_val),
                'best_design_flat': round_floats(best_design.tolist()),
                'best_design_reshaped': round_floats(best_design_list),
                'n_designs': args.n_designs,
                'shared_geometry': args.shared_geometry
            }
            suffix = "_shared" if args.shared_geometry else ""
            output_filename = f"{args.output_prefix}_n{args.n_designs}{suffix}.json"
            with open(output_filename, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            if (i+1) % 4 == 0 or i == 0:
                print(f"  [Fine Opt {i+1}/{len(fine_starts)}] Val: {val:.4f}")

    print(f"Done. Best Value: {best_val:.4f}")

if __name__ == '__main__':
    main()
