"""Singular value spectrum analysis for neural network weights.

Provides functions to compute and visualize the singular value decomposition
of weight matrices in a neural network, useful for understanding network
conditioning and expressivity during training.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def compute_weight_spectra(model: nn.Module) -> Dict[str, np.ndarray]:
    """Compute singular value spectra for all Linear layers in the model.
    
    Args:
        model: PyTorch neural network (e.g., SimpleNN).
    
    Returns:
        Dictionary mapping layer names to their singular value arrays.
        Each array is sorted in descending order.
    
    Example:
        >>> spectra = compute_weight_spectra(model)
        >>> for layer_name, svs in spectra.items():
        ...     print(f"{layer_name}: {len(svs)} singular values")
    """
    spectra = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data.cpu()
            # Compute SVD (we only need singular values)
            svs = torch.linalg.svdvals(W).numpy()
            spectra[name] = svs
    return spectra


def plot_spectra(
    spectra: Dict[str, np.ndarray],
    epoch: Optional[int] = None,
    save_path: Optional[str] = None,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot singular value spectra for all layers.
    
    Args:
        spectra: Dictionary from compute_weight_spectra().
        epoch: Current epoch number (for title).
        save_path: If provided, save figure to this path.
        log_scale: If True, use logarithmic y-axis.
        figsize: Figure size in inches.
    
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for layer_name, svs in spectra.items():
        ax.plot(range(len(svs)), svs, marker='o', markersize=3, label=layer_name)
    
    ax.set_xlabel("Singular Value Index")
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel("Singular Value (log scale)" if log_scale else "Singular Value")
    
    title = "Singular Value Spectrum"
    if epoch is not None:
        title += f" — Epoch {epoch}"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Spectra plot saved to {save_path}")
    
    return fig
