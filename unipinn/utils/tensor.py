"""Tensor conversion and autograd utilities."""

import torch
import numpy as np
from typing import Union


def gradients(y: torch.Tensor, x: torch.Tensor):
    """Compute dy/dx with autograd, preserving the computation graph."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        allow_unused=True,
        create_graph=True,
    )


def to_tensor(
    input: Union[np.ndarray, list, torch.Tensor],
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
    device: Union[str, torch.device, None] = None,
) -> torch.Tensor:
    """Convert numpy array or list to a torch Tensor, optionally moving to device."""
    t = torch.tensor(input, dtype=dtype, requires_grad=requires_grad)
    if device is not None:
        t = t.to(device)
    return t


def to_numpy(input: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a torch Tensor to numpy, or return as-is if already numpy."""
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError(
            f"Expected torch.Tensor or np.ndarray, got {type(input)}"
        )


def get_param_num(model: torch.nn.Module) -> int:
    """Return total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def r2_score(val_real: np.ndarray, val_pred: np.ndarray) -> float:
    """Coefficient of determination (R-squared)."""
    val_real = np.asarray(val_real, dtype=float)
    val_pred = np.asarray(val_pred, dtype=float)
    val_mean = np.mean(val_real)
    ss_res = np.sum((val_real - val_pred) ** 2)
    ss_tot = np.sum((val_real - val_mean) ** 2)
    return 1.0 - ss_res / ss_tot


def abs_r2_score(val_real: np.ndarray, val_pred: np.ndarray) -> float:
    """R-squared computed on absolute values."""
    return r2_score(np.abs(val_real), np.abs(val_pred))
