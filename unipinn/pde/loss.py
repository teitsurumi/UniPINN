"""Loss functions for Physics-Informed Neural Networks."""

import torch
from typing import Dict
from unipinn.utils.tensor import gradients


class PINNLossPoisson1D:
    """PDE residual + boundary condition loss for 1D Poisson: -u''(x) = f(x)."""

    def __init__(self, pde_weight: float = 1.0, bc_weight: float = 1.0):
        self.pde_w = pde_weight
        self.bc_w = bc_weight

    def __call__(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]):
        # PDE residual: -u''(x) = f(x)
        x_col = batch["x_col"]
        f_col = batch["f_col"]
        u_col = model(x_col)
        u_xx = gradients(gradients(u_col, x_col)[0], x_col)[0]
        f_pred = -u_xx
        loss_pde = torch.mean((f_pred - f_col) ** 2)

        # Boundary condition loss
        x_bc = batch["x_bc"]
        bc_target = batch["bc_target"]
        u_bc = model(x_bc)
        loss_bc = torch.mean((u_bc - bc_target) ** 2)

        total = self.pde_w * loss_pde + self.bc_w * loss_bc
        return {"total": total, "pde": loss_pde, "bc": loss_bc}
