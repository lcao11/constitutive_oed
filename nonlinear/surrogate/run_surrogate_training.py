import os
import time
import pickle
import argparse
import json
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
try:
    import matplotlib
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
except:
    pass

from architecture import build_fim_model

torch.backends.cudnn.benchmark = True

# --- Data Loading & Processing ---

def load_dataset_group(path: str, seed: int, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a group of pickle files defined by a seed and a list of process IDs.
    Returns stacked parameters, designs, and FIMs.
    """
    params_list, designs_list, fims_list = [], [], []
    print(f"Loading data for seed {seed} (indices {indices})...")
    
    for pid in indices:
        # --- Use Partial Files (Current Default) ---
        filename = f"data_{pid}_seed_{seed}_checkpoint.pkl"
        
        #--- Use Full Files (Future Use) ---
        # filename = f"data_{pid}_seed_{seed}.pkl"
        
        file_path = os.path.join(path, filename)
        
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'rb') as f:
                d = pickle.load(f)
            
            params_list.append(d['parameters'])
            designs_list.append(d['designs'])
            fims_list.append(d['fims'])
            
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
    
    if not params_list:
        raise ValueError(f"No data loaded for seed {seed} in {path}.")
        
    return np.vstack(params_list), np.vstack(designs_list), np.vstack(fims_list)

def filter_invalid_samples(fims: np.ndarray, params: np.ndarray, designs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters out samples where the FIM has NaNs or non-positive eigenvalues.
    This ensures the matrix logarithm is well-defined.
    """
    # 1. Check for NaNs in FIM
    nan_mask = np.isnan(fims).reshape(fims.shape[0], -1).any(axis=1)
    num_nans = nan_mask.sum()
    
    if num_nans > 0:
        print(f"[FILTER] Found {num_nans} / {fims.shape[0]} samples with NaNs in FIM.")
        nan_indices = np.where(nan_mask)[0]
        print(f"         Indices (in current batch): {nan_indices[:10]}{' ...' if num_nans > 10 else ''}")
        print(f"         Designs of first 5 NaN samples:\n{designs[nan_indices[:5]]}")
        
        keep_mask = ~nan_mask
        fims = fims[keep_mask]
        params = params[keep_mask]
        designs = designs[keep_mask]

    if fims.shape[0] == 0:
        return fims, params, designs

    # 2. Symmetrize & Check Eigenvalues
    # Symmetrize
    Fs = 0.5 * (fims + np.swapaxes(fims, -1, -2))
    # Check eigenvalues (use float64 for precision)
    eigs = np.linalg.eigvalsh(Fs.astype(np.float64, copy=False))
    min_eigs = eigs.min(axis=1)
    mask = min_eigs > 0.0
    
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"[FILTER] Dropping {dropped} / {Fs.shape[0]} FIMs with non-positive eigenvalues "
              f"(min kept={min_eigs[mask].min():.3e}, min dropped={min_eigs[~mask].min():.3e})")
    
    return fims[mask], params[mask], designs[mask]

def matrix_log_batch(F: np.ndarray) -> np.ndarray:
    """Computes matrix logarithm for a batch of symmetric matrices."""
    F = 0.5 * (F + np.swapaxes(F, -1, -2))
    evals, evecs = np.linalg.eigh(F.astype(np.float64, copy=False))
    
    # Safety clamp to avoid log(<=0) issues, though filtering should catch most
    evals = np.maximum(evals, 1e-14)
    
    # log(lambda)
    loge = np.log(evals)
    # Reconstruct: V * diag(log(lambda)) * V^T
    return (evecs * loge[..., None, :]) @ np.swapaxes(evecs, -1, -2)

def pca_rotate_full(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes PCA whitening statistics: mean, eigenvectors, and eigenvalues."""
    mean = X.mean(0)
    Xc = X - mean
    # Covariance matrix
    C = (Xc.T @ Xc) / max(1, Xc.shape[0])
    evals, vecs = np.linalg.eigh(C)
    evals = np.maximum(evals, eps)
    return mean, vecs, evals

def coord_norm_stats(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Computes mean and inverse standard deviation for coordinate-wise normalization."""
    mean = X.mean(0)
    std = X.std(0, ddof=0)
    std = np.maximum(std, eps)
    inv_std = 1.0 / std
    return mean, inv_std

def prepare_datasets(args) -> Tuple[DataLoader, DataLoader, DataLoader, Dict, Dict]:
    """
    Loads training, validation, and test sets based on fixed seeds.
    Computes normalization statistics on the training set.
    Returns DataLoaders, normalization dict, and model config.
    """
    
    # 1. Load Data Groups
    # Train: Seed 2025 (0-7), Seed 2026 (2,3), Seed 2027 (2,3)
    X_tr_p1, X_tr_d1, F_tr1 = load_dataset_group(args.data_path, 2025, list(range(8)))
    X_tr_p2, X_tr_d2, F_tr2 = load_dataset_group(args.data_path, 2026, [2,3])
    X_tr_p3, X_tr_d3, F_tr3 = load_dataset_group(args.data_path, 2027, [2,3])
    
    X_tr_p = np.vstack([X_tr_p1, X_tr_p2, X_tr_p3])
    X_tr_d = np.vstack([X_tr_d1, X_tr_d2, X_tr_d3])
    F_tr = np.vstack([F_tr1, F_tr2, F_tr3])

    # Val: Seed 2026 (0-2)
    X_va_p, X_va_d, F_va = load_dataset_group(args.data_path, 2026, [0, 1])
    
    # Test: Seed 2027 (0-2)
    X_te_p, X_te_d, F_te = load_dataset_group(args.data_path, 2027, [0, 1])

    # 2. Filter Invalid Samples (NaNs or Non-Positive FIMs)
    F_tr, X_tr_p, X_tr_d = filter_invalid_samples(F_tr, X_tr_p, X_tr_d)
    F_va, X_va_p, X_va_d = filter_invalid_samples(F_va, X_va_p, X_va_d)
    F_te, X_te_p, X_te_d = filter_invalid_samples(F_te, X_te_p, X_te_d)

    print(f"Dataset sizes: Train={len(F_tr)}, Val={len(F_va)}, Test={len(F_te)}")

    # 3. Process FIMs (Logarithm -> Lower Triangular -> Scaling)
    n = F_tr.shape[1]
    tri = torch.tril_indices(n, n)
    tri_r, tri_c = tri[0].cpu().numpy(), tri[1].cpu().numpy()
    diag_mask = (tri_r == tri_c)
    
    def process_fims(fims):
        logF = matrix_log_batch(fims)
        lower = logF[:, tri_r, tri_c]
        m = lower.shape[1]
        # Scale off-diagonals by sqrt(2) to preserve Frobenius norm in vector space
        w = np.ones(m, dtype=lower.dtype)
        w[~diag_mask] = np.sqrt(2.0)
        return lower * w, logF, w

    L_tr, LogF_tr, w_lower = process_fims(F_tr)
    L_va, LogF_va, _ = process_fims(F_va)
    L_te, LogF_te, _ = process_fims(F_te)

    # 4. Output Normalization (Fit on Train only)
    if args.out_mode == 'full_norm_mse':
        out_mean, out_comps, out_evals = pca_rotate_full(L_tr)
        out_sqrt = np.sqrt(out_evals)
        
        def normalize_out(L):
            return (L - out_mean) @ out_comps / out_sqrt
            
        Y_tr = normalize_out(L_tr)
        Y_va = normalize_out(L_va)
        Y_te = normalize_out(L_te)
    else:
        m = L_tr.shape[1]
        out_mean = np.zeros(m, dtype=L_tr.dtype)
        out_comps = np.eye(m, dtype=L_tr.dtype)
        out_evals = np.ones(m, dtype=L_tr.dtype)
        out_sqrt = np.ones(m, dtype=L_tr.dtype)
        Y_tr, Y_va, Y_te = L_tr, L_va, L_te

    # 5. Input Normalization (Fit on Train only)
    X_tr = np.concatenate([X_tr_p, X_tr_d], axis=1)
    X_va = np.concatenate([X_va_p, X_va_d], axis=1)
    X_te = np.concatenate([X_te_p, X_te_d], axis=1)

    in_mean, in_inv_std = coord_norm_stats(X_tr)

    # 6. Create DataLoaders
    def create_loader(X, Y, LogF, shuffle):
        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        Rt = torch.tensor(LogF, dtype=torch.float32)
        return DataLoader(TensorDataset(Xt, Yt, Rt), batch_size=args.batch_size, shuffle=shuffle)

    train_loader = create_loader(X_tr, Y_tr, LogF_tr, True)
    valid_loader = create_loader(X_va, Y_va, LogF_va, False)
    test_loader = create_loader(X_te, Y_te, LogF_te, False)

    norm_stats = {
        'input_norm_mean': torch.tensor(in_mean, dtype=torch.float32),
        'input_norm_inv_std': torch.tensor(in_inv_std, dtype=torch.float32),
        'output_mode_mean': torch.tensor(out_mean, dtype=torch.float32),
        'output_mode_components': torch.tensor(out_comps, dtype=torch.float32),
        'output_mode_sqrt_eig': torch.tensor(out_sqrt, dtype=torch.float32),
        'output_lower_unscale': torch.tensor(1.0 / w_lower, dtype=torch.float32),
    }
    
    config = {
        'input_size': X_tr.shape[1],
        'matrix_size': n,
        'model_type': args.model_type,
        'width': args.width,
        'depth': args.depth,
        'activation': args.activation,
        'dropout': args.dropout,
        'out_mode': args.out_mode,
    }

    return train_loader, valid_loader, test_loader, norm_stats, config

# --- Metrics & Logging ---

def qoi_from_eigs(eigs: torch.Tensor) -> torch.Tensor:
    """Computes the Quantity of Interest (QoI) from eigenvalues."""
    # Example QoI: Sum of log(1 + lambda)
    return 0.5 * (torch.log1p(eigs)).sum(dim=1)

def save_checkpoint(save_dir: str, tag: str, model: nn.Module, config: Dict, norm_dict: Dict, epoch: int, metrics: Dict):
    os.makedirs(save_dir, exist_ok=True)
    norm_np = {k: v.detach().cpu().numpy() for k, v in norm_dict.items()}
    np.savez(os.path.join(save_dir, f"norm_{tag}.npz"), **norm_np)
    
    meta = {'config': config, 'epoch': epoch, 'metrics': metrics, 'tag': tag}
    with open(os.path.join(save_dir, f"meta_{tag}.json"), 'w') as f:
        json.dump(meta, f, indent=2)
        
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_{tag}.pt"))

def plot_loss_curves(save_dir: str, hist: Dict, loss_label: str):
    if not hist['epoch']:
        return
    epochs = np.array(hist['epoch'])
    tr = np.maximum(np.array(hist['train_loss']), 1e-12)
    va = np.maximum(np.array(hist['val_loss']), 1e-12)
    
    plt.figure(figsize=(5,4))
    plt.semilogy(epochs, tr, label='Training')
    plt.semilogy(epochs, va, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(f'{loss_label} loss')
    plt.grid(True, ls='--', alpha=0.4)
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    plt.savefig(os.path.join(save_dir, f'{loss_label.lower()}_loss_curves.png'), dpi=200)
    plt.close()

# --- Training & Validation Loops ---

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device, args, eps: float = 1e-8) -> Dict:
    model.train()
    tr_latent_mse = 0.0
    tr_grad_norm_sum = 0.0
    tr_rel_logFIM_sum = 0.0
    
    for xb, yb, rtrue in loader:
        xb, yb, rtrue = xb.to(device), yb.to(device), rtrue.to(device)
        
        optimizer.zero_grad()
        pred_latent = model(xb)

        if args.out_mode == 'full_norm_mse':
            loss = ((pred_latent - yb) ** 2).mean()
        else:
            R_pred_loss = model.latent_to_logFIM(pred_latent)
            num_l = torch.linalg.norm(R_pred_loss - rtrue, dim=(1, 2))
            den_l = torch.linalg.norm(rtrue, dim=(1, 2)).clamp_min(eps)
            loss = (num_l / den_l).mean()

        loss.backward()

        # Gradient clipping and norm tracking
        gn = 0.0
        for p in model.parameters():
            if p.grad is not None:
                gn += p.grad.detach().norm(2).item() ** 2
        gn = gn ** 0.5
        tr_grad_norm_sum += gn * xb.size(0)

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Metrics
        tr_latent_mse += ((pred_latent - yb) ** 2).mean().item() * xb.size(0)
        with torch.no_grad():
            R_pred = model.latent_to_logFIM(pred_latent)
            num = torch.linalg.norm(R_pred - rtrue, dim=(1, 2))
            den = torch.linalg.norm(rtrue, dim=(1, 2)).clamp_min(eps)
            tr_rel_logFIM_sum += (num / den).sum().item()

    N = len(loader.dataset)
    return {
        'loss': tr_latent_mse / N if args.out_mode == 'full_norm_mse' else tr_rel_logFIM_sum / N,
        'mse': tr_latent_mse / N,
        'grad_norm': tr_grad_norm_sum / N,
        'rel_logFIM': tr_rel_logFIM_sum / N
    }

def validate(model: nn.Module, loader: DataLoader, device: torch.device, eps: float = 1e-8) -> Dict:
    model.eval()
    va_latent_mse = 0.0
    va_spec = 0.0
    rel_qoi_sum = 0.0
    va_rel_logFIM_sum = 0.0
    count = 0
    latent_dim = 0

    with torch.no_grad():
        for xb, yb, rtrue in loader:
            xb, yb, rtrue = xb.to(device), yb.to(device), rtrue.to(device)
            if latent_dim == 0: latent_dim = yb.shape[1]
            
            pred_latent = model(xb)
            
            # MSE
            va_latent_mse += ((pred_latent - yb) ** 2).sum().item()

            # Relative Log-FIM Error
            R_pred = model.latent_to_logFIM(pred_latent)
            num = torch.linalg.norm(R_pred - rtrue, dim=(1, 2))
            den = torch.linalg.norm(rtrue, dim=(1, 2)).clamp_min(eps)
            va_rel_logFIM_sum += (num / den).sum().item()

            # Eigen-structure
            log_eigs_pred = model.latent_to_log_eigs(pred_latent, descending=True)
            Rt64 = 0.5 * (rtrue.double() + rtrue.double().transpose(-1, -2))
            log_eigs_true, _ = torch.linalg.eigh(Rt64)
            log_eigs_true = log_eigs_true.flip(-1).to(pred_latent.dtype)
            
            rel_spec = ((log_eigs_pred - log_eigs_true).abs() / (log_eigs_true.abs() + eps)).mean(dim=1)
            va_spec += rel_spec.sum().item()

            # QoI
            eigs_pred = torch.exp(log_eigs_pred)
            eigs_true = torch.exp(log_eigs_true)
            pred_qoi = qoi_from_eigs(eigs_pred)
            true_qoi = qoi_from_eigs(eigs_true)
            num_qoi = (pred_qoi - true_qoi).abs().view(-1)
            den_qoi = true_qoi.abs().view(-1).clamp_min(eps)
            rel_qoi_sum += (num_qoi / den_qoi).sum().item()
            
            count += xb.size(0)

    return {
        'mse': va_latent_mse / (len(loader.dataset) * latent_dim),
        'rel_logFIM': va_rel_logFIM_sum / max(1, count),
        'rel_logEig': va_spec / max(1, count),
        'rel_qoi': rel_qoi_sum / max(1, count)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=str, default='./data/')
    ap.add_argument('--model_type', type=str, default='residual', choices=['mlp', 'residual'])
    ap.add_argument('--width', type=int, default=240)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'relu'])
    ap.add_argument('--dropout', type=float, default=0.00)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--grad_clip', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--save_dir', type=str, default='./results')
    ap.add_argument('--out_mode', type=str, default='no_norm_rel', choices=['no_norm_rel', 'full_norm_mse'])
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading and preparing data...")
    train_loader, valid_loader, test_loader, norm, config = prepare_datasets(args)

    print(f"Building model ({args.model_type})...")
    model = build_fim_model(config, norm, device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.001)

    best_val = float('inf')
    start = time.time()
    hist = {'epoch': [], 'train_loss': [], 'val_loss': []}
    loss_label = 'MSE' if args.out_mode == 'full_norm_mse' else 'RelLogFIM'

    print("Starting training...")
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args)
        val_metrics = validate(model, valid_loader, device)
        scheduler.step()

        # Determine primary loss for tracking
        if args.out_mode == 'full_norm_mse':
            train_loss = train_metrics['mse']
            val_loss = val_metrics['mse']
        else:
            train_loss = train_metrics['rel_logFIM']
            val_loss = val_metrics['rel_logFIM']

        # Logging
        hist['epoch'].append(epoch + 1)
        hist['train_loss'].append(train_loss)
        hist['val_loss'].append(val_loss)

        # Checkpointing
        metrics_to_save = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'loss_name': loss_label,
            'val_rel_logeig': val_metrics['rel_logEig'],
            'val_qoi_rel': val_metrics['rel_qoi'],
            'val_rel_logFIM': val_metrics['rel_logFIM'],
        }
        
        save_checkpoint(args.save_dir, 'last', model, config, norm, epoch + 1, metrics_to_save)

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            save_checkpoint(args.save_dir, 'best', model, config, norm, epoch + 1, metrics_to_save)
            print(f"  [Save] Best checkpoint at epoch {epoch+1} (Val {loss_label}: {val_loss:.4e})")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = (time.time() - start) / 60
            print(
                f"Epoch {epoch+1}/{args.epochs} "
                f"Tr{loss_label} {train_loss:.4e} "
                f"Va{loss_label} {val_loss:.4e} "
                f"VaRelLogFIM {val_metrics['rel_logFIM']:.4e} "
                f"VaRelLogEig {val_metrics['rel_logEig']:.4e} "
                f"QoI {val_metrics['rel_qoi']:.4e} "
                f"Grad {train_metrics['grad_norm']:.2e} "
                f"LR {scheduler.get_last_lr()[0]:.2e} "
                f"Time {elapsed:.1f}m"
            )
            plot_loss_curves(args.save_dir, hist, loss_label)

    print("Training complete. Evaluating on Test Set...")
    test_metrics = validate(model, test_loader, device)
    print(f"Test Results: MSE={test_metrics['mse']:.4e}, "
          f"RelLogFIM={test_metrics['rel_logFIM']:.4e}, "
          f"RelLogEig={test_metrics['rel_logEig']:.4e}, "
          f"QoI={test_metrics['rel_qoi']:.4e}")
    print("Done.")

if __name__ == '__main__':
    main()