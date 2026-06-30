"""Loss landscape visualization for PINN optimization analysis.

Computes and plots the 2D loss landscape by perturbing model parameters
along two random directions. This reveals the geometry of the loss surface
and helps diagnose optimization difficulties in PINN training.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Optional, Tuple
from pathlib import Path


def _flatten_params(model: nn.Module) -> Tuple[torch.Tensor, list]:
    """Flatten all model parameters into a single 1D vector.

    Returns:
        (flat_params, shapes): vector and list of (name, shape) for reconstruction.
    """
    tensors = []
    shapes = []
    for name, p in model.named_parameters():
        tensors.append(p.data.view(-1))
        shapes.append((name, p.shape))
    return torch.cat(tensors), shapes


def _unflatten_to_model(model: nn.Module, flat: torch.Tensor, shapes: list):
    """Copy values from a flat vector back into model parameters."""
    offset = 0
    for (name, shape), p in zip(shapes, model.parameters()):
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].view(shape))
        offset += n


def _random_direction(shapes: list, device: torch.device, seed: int = None,
                       filter_norm: bool = True) -> torch.Tensor:
    """Generate a random direction in parameter space.

    Args:
        shapes: List of (name, shape) from _flatten_params.
        device: Torch device.
        seed: Random seed for reproducibility.
        filter_norm: If True, normalize each filter's random values to unit norm
            (following the filter-normalized method of Li et al., 2018).

    Returns:
        Flat direction vector with unit overall norm.
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    parts = []
    for name, shape in shapes:
        if gen is not None:
            r = torch.randn(shape, generator=gen, device=device)
        else:
            r = torch.randn(shape, device=device)

        if filter_norm and r.dim() >= 2:
            # Normalize each filter (row) to unit norm
            for i in range(r.shape[0]):
                norm = r[i].norm()
                if norm > 0:
                    r[i] /= norm
        elif filter_norm and r.dim() == 1:
            norm = r.norm()
            if norm > 0:
                r /= norm

        parts.append(r.view(-1))

    d = torch.cat(parts)
    d /= d.norm()   # normalize the full direction to unit norm
    return d


def compute_loss_landscape(
    model: nn.Module,
    loss_fn: Callable,
    batch: Dict[str, torch.Tensor],
    grid_size: int = 40,
    alpha_range: float = 1.0,
    beta_range: float = 1.0,
    seed: int = 42,
    filter_norm: bool = True,
    verbose: bool = True,
) -> Dict:
    """Compute the 2D loss landscape around current parameters.

    Two random directions d1, d2 are generated. The loss is evaluated on a
    grid of (alpha, beta) perturbations: L(theta + alpha*d1 + beta*d2).

    Args:
        model: Trained neural network.
        loss_fn: Loss function callable(model, batch) -> dict with 'total'.
        batch: Training batch (collocation points, BC targets, etc.).
        grid_size: Number of grid points per axis.
        alpha_range: Range for d1 direction: [-alpha_range, +alpha_range].
        beta_range: Range for d2 direction: [-beta_range, +beta_range].
        seed: Random seed for direction generation (reproducibility).
        filter_norm: Use filter-normalized directions (recommended).
        verbose: Print progress.

    Returns:
        Dictionary with keys:
        - 'landscape': 2D numpy array of losses (grid_size x grid_size)
        - 'alpha': 1D array of alpha values
        - 'beta': 1D array of beta values
        - 'd1_seed': seed used for d1
        - 'd2_seed': seed used for d2
        - 'baseline_loss': loss at the current (unperturbed) parameters
    """
    model.eval()
    device = next(model.parameters()).device

    # Save original parameters
    flat_params, shapes = _flatten_params(model)
    original_params = flat_params.clone()

    # Generate two independent random directions
    d1_seed = seed
    d2_seed = seed + 1000
    d1 = _random_direction(shapes, device, seed=d1_seed, filter_norm=filter_norm)
    d2 = _random_direction(shapes, device, seed=d2_seed, filter_norm=filter_norm)

    alpha_vals = np.linspace(-alpha_range, alpha_range, grid_size)
    beta_vals = np.linspace(-beta_range, beta_range, grid_size)
    landscape = np.full((grid_size, grid_size), np.nan)

    # Baseline loss at current parameters
    baseline_dict = loss_fn(model, batch)
    baseline_loss = baseline_dict["total"].item()
    del baseline_dict  # Free autograd graph

    if verbose:
        print(f"[Landscape] Baseline loss = {baseline_loss:.6e}")
        print(f"[Landscape] Grid: {grid_size}x{grid_size}, "
              f"alpha=[{-alpha_range}, {alpha_range}], beta=[{-beta_range}, {beta_range}]")

    total_points = grid_size * grid_size
    count = 0

    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            # Perturb: theta' = theta* + alpha*d1 + beta*d2
            new_params = original_params + alpha * d1 + beta * d2
            _unflatten_to_model(model, new_params, shapes)

            # Evaluate loss (model.eval() so no dropout/batchnorm effects)
            # Note: loss_fn uses autograd internally for PDE derivatives
            loss_dict = loss_fn(model, batch)
            landscape[i, j] = loss_dict["total"].item()

            # Free the autograd graph to prevent memory buildup
            del loss_dict

            count += 1
            if verbose and count % grid_size == 0:
                print(f"[Landscape] Progress: {count}/{total_points}")

    # Restore original parameters
    _unflatten_to_model(model, original_params, shapes)
    model.train()

    return {
        "landscape": landscape,
        "alpha": alpha_vals,
        "beta": beta_vals,
        "d1_seed": d1_seed,
        "d2_seed": d2_seed,
        "baseline_loss": baseline_loss,
    }


def plot_landscape(
    result: Dict,
    save_path: Optional[str] = None,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot the 2D loss landscape as a filled contour plot.

    Args:
        result: Dictionary returned by compute_loss_landscape().
        save_path: If provided, save figure to this path.
        log_scale: If True, plot log(loss) for better contrast.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.
    """
    landscape = result["landscape"]
    alpha = result["alpha"]
    beta = result["beta"]
    baseline = result["baseline_loss"]

    if log_scale:
        plot_data = np.log10(np.maximum(landscape, 1e-30))
        cb_label = "log10(Loss)"
    else:
        plot_data = landscape
        cb_label = "Loss"

    fig, ax = plt.subplots(figsize=figsize)
    alpha_grid, beta_grid = np.meshgrid(alpha, beta, indexing="ij")

    contour = ax.contourf(alpha_grid, beta_grid, plot_data, levels=40, cmap="viridis")
    fig.colorbar(contour, ax=ax, label=cb_label)

    # Mark the current position (center)
    ax.plot(0, 0, "r*", markersize=15, label="Current params")

    ax.set_xlabel("Direction d1")
    ax.set_ylabel("Direction d2")
    ax.set_title(f"Loss Landscape  (baseline = {baseline:.4e})")
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Landscape plot saved to {save_path}")

    return fig


def plot_landscape_3d(
    result: Dict,
    save_path: Optional[str] = None,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot the 2D loss landscape as a 3D surface plot.

    Args:
        result: Dictionary returned by compute_loss_landscape().
        save_path: If provided, save figure to this path.
        log_scale: If True, plot log(loss) for better contrast.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.
    """
    from mpl_toolkits.mplot3d import Axes3D

    landscape = result["landscape"]
    alpha = result["alpha"]
    beta = result["beta"]
    baseline = result["baseline_loss"]

    if log_scale:
        plot_data = np.log10(np.maximum(landscape, 1e-30))
        z_label = "log10(Loss)"
    else:
        plot_data = landscape
        z_label = "Loss"

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    alpha_grid, beta_grid = np.meshgrid(alpha, beta, indexing="ij")

    surf = ax.plot_surface(alpha_grid, beta_grid, plot_data, cmap="viridis", alpha=0.9)
    fig.colorbar(surf, ax=ax, label=z_label)

    ax.set_xlabel("Direction d1")
    ax.set_ylabel("Direction d2")
    ax.set_zlabel(z_label)
    ax.set_title(f"Loss Landscape 3D  (baseline = {baseline:.4e})")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"3D landscape plot saved to {save_path}")

    return fig
