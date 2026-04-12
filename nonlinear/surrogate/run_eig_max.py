import argparse
import json
import math
import os
import itertools
from typing import List, Tuple, Union, Any

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import norm, qmc

from architecture import load_fim_model

# --- Configuration ---
N_CONTROLS = 10
MAX_STRAIN = 1.0
ASPECT_RATIO = 1.0
MIN_STRETCH = 0.1
MAX_STRETCH = 0.35

def compute_pchip_h1_penalty(controls: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared H1 semi-norm (integral of derivative squared) of the PCHIP trajectory.
    controls: (B, N_CONTROLS)
    Returns the MEAN penalty across the batch B (designs).
    """
    B, N = controls.shape
    # 1. Construct full y vector: (B, N+1)
    zeros = torch.zeros((B, 1), device=controls.device, dtype=controls.dtype)
    y = torch.cat([zeros, controls], dim=1)
    
    n_points = y.shape[1]
    h = 1.0 / (n_points - 1)
    
    # 2. Compute slopes delta: (B, N)
    delta = (y[:, 1:] - y[:, :-1]) / h
    
    # 3. Compute node derivatives d: (B, N+1)
    d = torch.zeros_like(y)
    
    # Internal points
    delta_L = delta[:, :-1]
    delta_R = delta[:, 1:]
    
    mask = (delta_L * delta_R) > 0
    
    sum_delta = delta_L + delta_R
    valid_denom = torch.abs(sum_delta) > 1e-12
    safe_sum = torch.where(valid_denom, sum_delta, torch.ones_like(sum_delta))
    hm_vals = 2 * delta_L * delta_R / safe_sum
    
    d[:, 1:-1] = torch.where(mask & valid_denom, hm_vals, torch.zeros_like(hm_vals))
    
    # Endpoints (One-sided scheme)
    def edge_deriv(dt0, dt1):
        val = (3 * dt0 - dt1) / 2.0
        mask1 = (val * dt0) <= 0
        val = torch.where(mask1, torch.zeros_like(val), val)
        mask2 = (dt0 * dt1) <= 0
        mask3 = torch.abs(val) > torch.abs(3 * dt0)
        val = torch.where(mask2 & mask3, 3 * dt0, val)
        return val

    d[:, 0] = edge_deriv(delta[:, 0], delta[:, 1])
    d[:, -1] = edge_deriv(delta[:, -1], delta[:, -2])
    
    # 4. Analytical Integration of (f')^2 over intervals
    # f(t) is cubic on [0, h], f'(t) is quadratic.
    # f'(t) = 3at^2 + 2bt + c
    # Integral[(f')^2] = 9/5 a^2 h^5 + 3 ab h^4 + (4b^2 + 6ac)/3 h^3 + 2bc h^2 + c^2 h
    
    y_k = y[:, :-1]
    y_kp1 = y[:, 1:]
    d_k = d[:, :-1]
    d_kp1 = d[:, 1:]
    
    # Coefficients for cubic polynomial f(t) = at^3 + bt^2 + ct + d
    a = (2 * (y_k - y_kp1) + h * (d_k + d_kp1)) / (h**3)
    b = (3 * (y_kp1 - y_k) - h * (2 * d_k + d_kp1)) / (h**2)
    c = d_k
    
    term1 = 1.8 * a**2 * (h**5)
    term2 = 3.0 * a * b * (h**4)
    term3 = (4.0 * b**2 + 6.0 * a * c) * (h**3) / 3.0
    term4 = 2.0 * b * c * (h**2)
    term5 = c**2 * h
    
    integral_per_interval = term1 + term2 + term3 + term4 + term5
    
    # Sum over intervals to get total integral for each design: (B,)
    total_integral = integral_per_interval.sum(dim=1)
    
    # Return MEAN over batch (designs)
    return total_integral.mean()

def get_design_bounds(n_controls: int = 10, max_strain: float = 0.1, aspect_ratio: float = 1.0) -> List[Tuple[float, float]]:
    """Returns physical bounds for design variables."""
    # x0: Rotation [0, pi/2]
    bounds = [(0.0, 0.5 * math.pi)]
    
    # x1: Stretch Y [MIN_STRETCH, MAX_STRETCH]
    # Note: This allows the stretch parameter to vary within bounds.
    bounds.append((MIN_STRETCH, MAX_STRETCH))
    # Re-adding the second stretch bound as requested
    bounds.append((MIN_STRETCH, MAX_STRETCH))
    
    # x2...x11: Controls
    control_upper = max_strain * aspect_ratio
    for i in range(n_controls):
        # First control point has a minimum of 0.1, others 0.0
        bounds.append((0.0, control_upper))
        
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
    """
    is_batch = x.dim() == 2
    if not is_batch:
        x = x.unsqueeze(0)
    
    B = x.shape[0]
    
    # We have 3 geometric parameters: Rot, Stretch1, Stretch2
    n_geo = 3
    n_controls = design_dim - n_geo
    
    # x structure: [geo_1, ..., geo_3, controls_flat]
    geo = x[:, :n_geo].unsqueeze(1).expand(B, n_designs, n_geo) # (B, D, 3)
    
    # controls are remaining elements, reshaped to (B, D, C)
    ctrl = x[:, n_geo:].view(B, n_designs, n_controls) 
    
    full = torch.cat([geo, ctrl], dim=2) # (B, D, d)
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
    
    # Utility: 0.5 * (log(det) + tr(inv))
    term1 = torch.log1p(eigs).sum(dim=-1)
    utility_samples = 0.5 * term1 # (B, M)
    
    # Expectation over parameters
    utility = utility_samples.mean(dim=1) # (B,)
    
    if not is_batch:
        return utility.squeeze(0)
    return utility

def optimize_design(model, param_samples: torch.Tensor, x0: torch.Tensor, bounds: torch.Tensor, 
                   n_designs: int, design_dim: int, shared_geometry: bool, max_iter: int,
                   penalty_weight: float = 0.0) -> Tuple[torch.Tensor, float]:
    
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
        
        # 4. Penalty for large deformation (controls)
        penalty = 0.0
        if penalty_weight > 0.0:
            n_geo = 3
            if shared_geometry:
                # x: [geo..., c_0, ..., c_N]
                n_controls = design_dim - n_geo
                controls = x[n_geo:].view(n_designs, n_controls)
            else:
                # x: [geo..., c_0..., geo..., c_0...]
                x_reshaped = x.view(n_designs, design_dim)
                controls = x_reshaped[:, n_geo:]
            
            # H1 penalty on controls
            penalty = penalty_weight * compute_pchip_h1_penalty(controls)

        loss = -util + penalty # Minimize negative utility + penalty
        
        # 5. Compute gradient
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()
        
        # 6. Return value and gradient as float64 for SciPy
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
    parser.add_argument('--n_designs', type=int, default=3)
    parser.add_argument('--n_screen', type=int, default=131072, help="Number of initial screening points")
    parser.add_argument('--n_fine', type=int, default=256, help="Number of points for fine optimization (L-BFGS)")
    parser.add_argument('--mc_samples', type=int, default=512)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--seed', type=int, default=109)
    parser.add_argument('--output_prefix', type=str, default='optimal_design')
    parser.add_argument('--shared_geometry', action='store_true')
    parser.add_argument('--penalty_weight', type=float, default=0.0, help="Weight for penalty term in optimization")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    print(f"Loading model from {args.model_dir}...")
    model, meta = load_fim_model(args.model_dir, tag=args.tag, device=device)
    model.eval()
    
    # Setup Dimensions
    input_size = meta['config']['input_size']
    design_bounds = get_design_bounds(N_CONTROLS, MAX_STRAIN, ASPECT_RATIO)
    design_dim = len(design_bounds)
    param_dim = input_size - design_dim
    
    # Generate MC Samples
    param_samples = generate_param_samples(args.mc_samples, param_dim, args.seed, device)
    param_samples.requires_grad_(False)
    
    # Setup Bounds
    if args.shared_geometry:
        # [geo...] + [c0..c9]*D
        n_geo = 3
        bounds_flat = design_bounds[:n_geo] + design_bounds[n_geo:] * args.n_designs
    else:
        # [geo..., c0..c9]*D
        bounds_flat = design_bounds * args.n_designs
    
    bounds_tensor = torch.tensor(bounds_flat, dtype=torch.float32, device=device)
    bounds_tensor.requires_grad_(False)
    
    # 1. Screening Step
    print(f"Screening {args.n_screen} starts on GPU...")
    starts_np = sobol_design_starts(args.n_screen, bounds_flat, seed=args.seed + 1)

    starts_tensor = torch.tensor(starts_np, dtype=torch.float32, device=device)
    
    # Prepare batch input for screening
    if args.shared_geometry:
        # Expand shared vars for evaluation
        n_total = starts_tensor.shape[0]
        n_geo = 3
        geo = starts_tensor[:, :n_geo].unsqueeze(1).expand(-1, args.n_designs, n_geo)
        ctrl = starts_tensor[:, n_geo:].view(n_total, args.n_designs, -1)
        full_designs = torch.cat([geo, ctrl], dim=2).view(n_total, -1)
    else:
        full_designs = starts_tensor

    # Evaluate in chunks to avoid OOM
    # Scales: 1->1024, 2->512, 3->256, 4->256, 5->128, 6->64
    chunk_size = max(1, int(2048 / (2 ** args.n_designs)))
    utilities = []
    n_total = full_designs.shape[0]
    with torch.no_grad():
        for i in range(0, n_total, chunk_size):
            batch = full_designs[i:i+chunk_size]
            utilities.append(compute_utility(model, param_samples, batch, args.n_designs, design_dim))
    start_utilities = torch.cat(utilities)
    
    # Select top candidates for fine optimization directly
    top_vals, top_idxs = torch.topk(start_utilities, min(args.n_fine, len(start_utilities)))
    print(f"Best start utility (Screening): {top_vals[0]:.4f}")
    
    # 2. Fine Optimization (Sequential L-BFGS)
    print(f"Running Fine Optimization (L-BFGS) on {len(top_idxs)} designs...")
    
    best_val = -np.inf
    best_design = None
    
    # Use screening results directly
    fine_starts = starts_tensor[top_idxs]
    
    for i, x0 in enumerate(fine_starts):
        opt_design, val = optimize_design(
            model, param_samples, x0, bounds_tensor, 
            args.n_designs, design_dim, args.shared_geometry, args.max_iter,
            penalty_weight=args.penalty_weight
        )
        
        if val > best_val:
            best_val = val
            best_design = opt_design.cpu().numpy()
            
            # Strict cleaning: Clip -> Round -> Clip
            lower_bounds_np = bounds_tensor[:, 0].cpu().numpy()
            upper_bounds_np = bounds_tensor[:, 1].cpu().numpy()
            
            # 1. Clip to physical bounds
            best_design = np.clip(best_design, lower_bounds_np, upper_bounds_np)
            
            # 2. Round to 6 decimal places
            best_design = np.round(best_design, 6)
            
            # 3. Clip again just in case rounding pushed it over
            best_design = np.clip(best_design, lower_bounds_np, upper_bounds_np)

            with torch.no_grad():
                # Reconstruct input for utility calc
                x_opt_tensor = torch.tensor(best_design, dtype=torch.float32, device=device)
                n_geo = 3
                if args.shared_geometry:
                    d_input = expand_shared_design(x_opt_tensor, args.n_designs, design_dim)
                    n_controls = design_dim - n_geo
                    controls = x_opt_tensor[n_geo:].view(args.n_designs, n_controls)
                else:
                    d_input = x_opt_tensor
                    x_reshaped = x_opt_tensor.view(args.n_designs, design_dim)
                    controls = x_reshaped[:, n_geo:]
                
                util_val = compute_utility(model, param_samples, d_input, args.n_designs, design_dim).item()
                penalty_val = args.penalty_weight * compute_pchip_h1_penalty(controls).item()

            print(f"  [Fine Opt {i+1}/{len(fine_starts)}] New Best Total: {best_val:.4f} (Utility: {util_val:.4f}, Penalty: {penalty_val:.4f})")
            
            # Helper to round floats for JSON
            def round_floats(obj):
                if isinstance(obj, float):
                    return round(obj, 6)
                if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                    return round(float(obj), 6)
                if isinstance(obj, list):
                    return [round_floats(x) for x in obj]
                if isinstance(obj, np.ndarray):
                    return round_floats(obj.tolist())
                return obj

            # Reshape design for better readability
            if args.shared_geometry:
                # Reconstruct full design: (n_designs, design_dim)
                n_geo = 3
                n_controls = design_dim - n_geo
                geo = best_design[:n_geo]
                controls = best_design[n_geo:].reshape(args.n_designs, n_controls)
                
                best_design_reshaped = np.zeros((args.n_designs, design_dim))
                best_design_reshaped[:, :n_geo] = geo
                best_design_reshaped[:, n_geo:] = controls
                best_design_list = best_design_reshaped.tolist()
            else:
                # Independent designs: simply reshape (n_designs, design_dim)
                best_design_list = best_design.reshape(args.n_designs, design_dim).tolist()

            # Save immediately with explicit rounding
            results = {
                'best_value': float(best_val),
                'utility_value': float(util_val),
                'penalty_value': float(penalty_val),
                'penalty_weight': args.penalty_weight,
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
