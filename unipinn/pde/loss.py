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


class PINNLossPoisson2D:
    """PDE residual + BC loss + optional data loss for 2D Poisson: -(u_xx + u_yy) = f(x,y).

    The Laplacian is computed via autograd with per-column slicing of the
    first-derivative tensor, which correctly propagates second-order
    gradients for both input dimensions.
    """

    def __init__(self, pde_weight: float = 1.0, bc_weight: float = 1.0,
                 data_weight: float = 0.0):
        self.pde_w = pde_weight
        self.bc_w = bc_weight
        self.data_w = data_weight

    def __call__(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]):
        x_col = batch["x_col"]       # (n, 2) requires_grad=True
        f_col = batch["f_col"]       # (n, 1)

        u_col = model(x_col)         # (n, 1)

        # First derivatives: (n, 2) — column 0 = du/dx, column 1 = du/dy
        du = gradients(u_col, x_col)[0]

        # Second derivatives via per-column slicing
        u_xx = gradients(du[:, 0:1], x_col)[0][:, 0:1]   # d²u/dx²
        u_yy = gradients(du[:, 1:2], x_col)[0][:, 1:2]   # d²u/dy²

        laplacian = u_xx + u_yy
        f_pred = -laplacian
        loss_pde = torch.mean((f_pred - f_col) ** 2)

        # Boundary condition loss
        x_bc = batch["x_bc"]
        bc_target = batch["bc_target"]
        u_bc = model(x_bc)
        loss_bc = torch.mean((u_bc - bc_target) ** 2)

        total = self.pde_w * loss_pde + self.bc_w * loss_bc

        # Supervised data loss (optional)
        loss_data = torch.tensor(0.0, device=x_col.device)
        if self.data_w > 0 and "x_data" in batch:
            u_data_pred = model(batch["x_data"])
            loss_data = torch.mean((u_data_pred - batch["u_data"]) ** 2)
            total = total + self.data_w * loss_data

        return {"total": total, "pde": loss_pde, "bc": loss_bc, "data": loss_data}
