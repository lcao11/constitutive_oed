"""Neural network architectures and utilities for nonlinear FIM surrogate models.

Usage:
    from architecture import build_fim_model, load_fim_model
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

def get_activation(name: str) -> nn.Module:
    """Returns the activation module based on name."""
    return {'gelu': nn.GELU(), 'relu': nn.ReLU()}[name]

class PlainMLP(nn.Module):
    """
    A simple multi-layer perceptron with dropout.
    Structure: [Linear -> Activation -> Dropout] x (depth-1) -> Linear
    """
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int, activation: str = 'gelu', dropout: float = 0.0):
        super().__init__()
        act = get_activation(activation)
        layers = []
        d_in = in_dim
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(d_in, width), act, nn.Dropout(dropout)])
            d_in = width
        layers.append(nn.Linear(d_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResidualBlock(nn.Module):
    """
    A Pre-Activation Residual Block (Pre-LN).
    Structure: x + Linear(Dropout(Activation(Linear(LayerNorm(x)))))
    """
    def __init__(self, dim: int, activation: str = 'gelu', dropout: float = 0.0, expansion: int = 1, residual_scale: float = 1.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.ln = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, hidden)
        self.lin2 = nn.Linear(hidden, dim)
        self.act = get_activation(activation)
        self.drop = nn.Dropout(dropout)
        self.res_scale = residual_scale

        # Zero-init second linear to start near identity
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.lin1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.lin2(h)
        return x + self.res_scale * h

class ResidualMLP(nn.Module):
    """
    A Residual MLP consisting of an input projection followed by a stack of ResidualBlocks.
    """
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int, activation: str = 'gelu', dropout: float = 0.0):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        # depth - 2 because we have input projection and output projection
        self.blocks = nn.ModuleList([
            ResidualBlock(width, activation, dropout) for _ in range(max(0, depth - 2))
        ])
        self.out = nn.Linear(width, out_dim)
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.inp(x))
        for b in self.blocks:
            h = b(h)
        return self.out(h)

def make_backbone(kind: str, in_dim: int, out_dim: int, width: int, depth: int, activation: str = 'gelu', dropout: float = 0.0) -> nn.Module:
    if kind == 'mlp':
        return PlainMLP(in_dim, out_dim, width, depth, activation, dropout)
    if kind == 'residual':
        return ResidualMLP(in_dim, out_dim, width, depth, activation, dropout)
    raise ValueError(f"Unknown backbone {kind}")

class FIMModel(nn.Module):
    """
    Model to predict the Fisher Information Matrix (FIM) from input parameters.
    Predicts the Cholesky-like decomposition or log-matrix components in a latent space.
    """
    def __init__(self, input_dim_raw: int, matrix_size: int,
                 input_norm_mean: torch.Tensor, input_norm_inv_std: torch.Tensor,
                 output_mode_mean: torch.Tensor, output_mode_components: torch.Tensor, output_mode_sqrt_eig: torch.Tensor,
                 output_lower_unscale: Optional[torch.Tensor] = None,
                 backbone_type: str = 'residual',
                 width: int = 128, depth: int = 6, activation: str = 'gelu', dropout: float = 0.05):
        super().__init__()
        self.raw_input_size = input_dim_raw
        self.matrix_size = matrix_size
        self.full_output_size = matrix_size * (matrix_size + 1) // 2

        # Coordinate-wise normalization buffers
        self.register_buffer('input_norm_mean', input_norm_mean)
        self.register_buffer('input_norm_inv_std', input_norm_inv_std)

        # Output rotation-only (PCA components)
        self.register_buffer('output_mode_mean', output_mode_mean)
        self.register_buffer('output_mode_components', output_mode_components)
        self.register_buffer('output_mode_sqrt_eig', output_mode_sqrt_eig)
        
        if output_lower_unscale is None:
            output_lower_unscale = torch.ones(output_mode_components.shape[1], dtype=output_mode_components.dtype)
        self.register_buffer('output_lower_unscale', output_lower_unscale)

        # Backbone works on normalized raw inputs (no PCA), so in_dim = raw_input_size
        self.input_latent_dim = self.raw_input_size
        self.latent_dim = output_mode_components.shape[1]
        self.backbone = make_backbone(backbone_type,
                                      in_dim=self.input_latent_dim,
                                      out_dim=self.latent_dim,
                                      width=width, depth=depth,
                                      activation=activation, dropout=dropout)

        # Indices for lower triangular matrix
        tri = torch.tril_indices(matrix_size, matrix_size)
        self.register_buffer('tril_i', tri[0])
        self.register_buffer('tril_j', tri[1])

    def normalize_inputs(self, x_raw: torch.Tensor) -> torch.Tensor:
        return (x_raw - self.input_norm_mean) * self.input_norm_inv_std

    def denormalize_outputs(self, y_latent: torch.Tensor) -> torch.Tensor:
        y_scaled = y_latent * self.output_mode_sqrt_eig
        full_scaled = y_scaled @ self.output_mode_components.T + self.output_mode_mean
        full_raw = full_scaled * self.output_lower_unscale
        return full_raw

    def lower_vec_to_matrix(self, vec: torch.Tensor) -> torch.Tensor:
        B = vec.size(0)
        n = self.matrix_size
        M = vec.new_zeros(B, n, n)
        M[:, self.tril_i, self.tril_j] = vec
        # Symmetrize: M + M.T - diag(M)
        diag = torch.diagonal(M, dim1=-2, dim2=-1)
        M = M + M.transpose(-1, -2) - torch.diag_embed(diag)
        return M

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        z_in = self.normalize_inputs(x_raw)
        return self.backbone(z_in)

    def latent_to_logFIM(self, latent: torch.Tensor) -> torch.Tensor:
        """Converts latent output to the log-FIM matrix (symmetric)."""
        lower = self.denormalize_outputs(latent)
        R = self.lower_vec_to_matrix(lower)
        return 0.5 * (R + R.transpose(-1, -2))

    def latent_to_log_eigs(self, latent: torch.Tensor, descending: bool = True) -> torch.Tensor:
        """Computes eigenvalues of the log-FIM."""
        R = self.latent_to_logFIM(latent)
        # Use float64 for stability in eigendecomposition
        R64 = 0.5 * (R.to(torch.float64) + R.to(torch.float64).transpose(-1, -2))
        log_eigs, _ = torch.linalg.eigh(R64)
        if descending:
            log_eigs = log_eigs.flip(-1)
        return log_eigs.to(latent.dtype)

    def latent_to_FIM(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstructs the FIM from latent output."""
        R = self.latent_to_logFIM(latent)
        # Use float64 for stability
        R64 = 0.5 * (R.to(torch.float64) + R.to(torch.float64).transpose(-1, -2))
        r, Q = torch.linalg.eigh(R64)
        FIM = Q @ torch.diag_embed(torch.exp(r)) @ Q.transpose(-1, -2)
        FIM = 0.5 * (FIM + FIM.transpose(-1, -2))
        return FIM.to(R.dtype)

    def forward_to_FIM(self, x_raw: torch.Tensor) -> torch.Tensor:
        latent = self.forward(x_raw)
        return self.latent_to_FIM(latent)

def build_fim_model(config: Dict, norm: Dict, device: torch.device) -> FIMModel:
    """Factory function to build FIMModel from config and normalization dicts."""
    return FIMModel(
        input_dim_raw=config['input_size'],
        matrix_size=config['matrix_size'],
        input_norm_mean=norm['input_norm_mean'].to(device),
        input_norm_inv_std=norm['input_norm_inv_std'].to(device),
        output_mode_mean=norm['output_mode_mean'].to(device),
        output_mode_components=norm['output_mode_components'].to(device),
        output_mode_sqrt_eig=norm['output_mode_sqrt_eig'].to(device),
        output_lower_unscale=norm.get('output_lower_unscale', torch.ones(
            config['matrix_size'] * (config['matrix_size'] + 1) // 2, device=device, dtype=torch.float32)),
        backbone_type=config['model_type'],
        width=config['width'],
        depth=config['depth'],
        activation=config.get('activation', 'gelu'),
        dropout=config.get('dropout', 0.05)
    ).to(device)

def load_fim_model(save_dir: str, tag: str = 'best', device: Optional[torch.device] = None) -> Tuple[FIMModel, Dict]:
    """
    Load model + meta + normalization (new-format checkpoint triplet).
    Returns (model, meta_dict).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    meta_path = os.path.join(save_dir, f"meta_{tag}.json")
    norm_path = os.path.join(save_dir, f"norm_{tag}.npz")
    model_path = os.path.join(save_dir, f"model_{tag}.pt")

    if not (os.path.isfile(meta_path) and os.path.isfile(norm_path) and os.path.isfile(model_path)):
        raise FileNotFoundError(f"Missing one of meta/norm/model for tag '{tag}' in {save_dir}")

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    config = meta['config']

    norm_npz = np.load(norm_path)
    # Required keys
    needed = ['input_norm_mean','input_norm_inv_std',
              'output_mode_mean','output_mode_components','output_mode_sqrt_eig','output_lower_unscale']
    for k in needed:
        if k not in norm_npz:
            raise KeyError(f"Normalization key '{k}' not found in {norm_path}")

    norm = {
        'input_norm_mean': torch.tensor(norm_npz['input_norm_mean'], dtype=torch.float32, device=device),
        'input_norm_inv_std': torch.tensor(norm_npz['input_norm_inv_std'], dtype=torch.float32, device=device),
        'output_mode_mean': torch.tensor(norm_npz['output_mode_mean'], dtype=torch.float32, device=device),
        'output_mode_components': torch.tensor(norm_npz['output_mode_components'], dtype=torch.float32, device=device),
        'output_mode_sqrt_eig': torch.tensor(norm_npz['output_mode_sqrt_eig'], dtype=torch.float32, device=device),
        'output_lower_unscale': torch.tensor(norm_npz['output_lower_unscale'], dtype=torch.float32, device=device),
    }

    model = build_fim_model(config, norm, device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, meta
