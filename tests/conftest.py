"""Shared test fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def simple_model():
    from unipinn.nn.architectures import SimpleNN
    config = {
        1: {"n": 1},
        2: {"n": 16, "a": "tanh"},
        3: {"n": 16, "a": "tanh"},
        4: {"n": 1},
    }
    return SimpleNN(config)


@pytest.fixture
def sample_1d_batch(device):
    """Minimal 1D Poisson training batch."""
    x_col = torch.linspace(-1, 1, 50, device=device).reshape(-1, 1).requires_grad_(True)
    f_col = torch.ones_like(x_col)
    x_bc = torch.tensor([[-1.0], [1.0]], device=device).requires_grad_(True)
    bc_target = torch.zeros(2, 1, device=device)
    return {"x_col": x_col, "f_col": f_col, "x_bc": x_bc, "bc_target": bc_target}
