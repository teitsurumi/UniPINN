"""Custom activation functions for Physics-Informed Neural Networks.

Includes adaptive, periodic, polynomial, and spectral activations designed
to improve PINN convergence on problems with multi-scale or high-frequency
solutions.
"""

import torch
from torch import nn
import numpy as np


class AdaptiveTanh(nn.Module):
    """Trainable tanh with a learnable scaling coefficient."""
    def __init__(self, scale_factor: float = 1.0):
        super().__init__()
        self.n = scale_factor
        self.coefficients = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return torch.tanh(self.coefficients[0] * self.n * x)


class DyT(nn.Module):
    """Dynamic Tanh: gamma * tanh(alpha * x) + beta, with trainable alpha, beta, gamma."""
    def __init__(self, input_dim: int, init_alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.beta = nn.Parameter(torch.zeros(input_dim))
        self.gamma = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class SineActivation(nn.Module):
    """Fixed-frequency sine: sin(omega * x) where omega = 2*pi / T."""
    def __init__(self, T: float):
        super().__init__()
        self.omega = 2 * torch.pi / T

    def forward(self, x):
        return torch.sin(self.omega * x)


class SineActivationT(nn.Module):
    """Trainable-amplitude sine: coeff * sin(omega * x)."""
    def __init__(self, T: float):
        super().__init__()
        self.coefficients = nn.Parameter(torch.randn(1))
        self.omega = 2 * torch.pi / T

    def forward(self, x):
        return self.coefficients[0] * torch.sin(self.omega * x)


class SineActivationT3(nn.Module):
    """Three-parameter sine: c0 * sin(c1 * x + c2)."""
    def __init__(self, T: float):
        super().__init__()
        self.coefficients = nn.Parameter(torch.randn(3))
        self.omega = 2 * torch.pi / T

    def forward(self, x):
        return self.coefficients[0] * torch.sin(self.coefficients[1] * x + self.coefficients[2])


class DynSine(nn.Module):
    """Dynamic Sine: gamma * sin(alpha * x) + beta, with trainable parameters."""
    def __init__(self, input_dim: int, init_alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * 2 * torch.pi / init_alpha)
        self.beta = nn.Parameter(torch.zeros(input_dim))
        self.gamma = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        return self.gamma * torch.sin(self.alpha * x) + self.beta


class PolynomialActivationV1(nn.Module):
    """Polynomial activation with trainable coefficients c_0 to c_N."""
    def __init__(self, n_degree: int):
        super().__init__()
        self.coefficients = nn.Parameter(torch.randn(n_degree + 1))

    def forward(self, x):
        poly_output = torch.zeros_like(x)
        for n in range(self.coefficients.size(0)):
            poly_output += self.coefficients[n] * (x ** n)
        return poly_output


class PolynomialActivationV2(nn.Module):
    """Per-neuron polynomial activation: sum_n c_n * x^n, vectorized."""
    def __init__(self, input_dim: int, n_degree: int):
        super().__init__()
        self.d = input_dim
        self.n = n_degree + 1
        self.coefficients = nn.Parameter(torch.zeros(1, self.d, self.n))

    def forward(self, x):
        poly = torch.stack([x ** n for n in range(self.n)], dim=-1)
        return (poly * self.coefficients).sum(dim=-1)


class ChebyshevActivation(nn.Module):
    """Per-neuron Chebyshev polynomial activation."""
    def __init__(self, input_dim: int, n_degree: int):
        super().__init__()
        self.d = input_dim
        self.n = n_degree + 1
        self.weight = nn.Parameter(torch.zeros(1, self.d, self.n))

    def forward(self, X):
        B = self._chebyshev_basis(X, self.n)
        e = len(B.shape) - len(self.weight.shape)
        w = self.weight.view(1, self.d, *([1] * e), self.n)
        L = (w * B).sum(dim=-1)
        assert L.size() == X.size()
        return L

    @staticmethod
    def _chebyshev_basis(X, n_terms):
        """Compute Chebyshev basis T_0(x), T_1(x), ..., T_{n-1}(x) via recurrence."""
        H = [torch.ones_like(X), X]
        for _ in range(2, n_terms):
            H.append(2 * X * H[-1] - H[-2])
        return torch.stack(H[:n_terms], dim=-1)


class FourierActivation(nn.Module):
    """Fourier series activation: sum of cos/sin terms with trainable coefficients."""
    def __init__(self, input_dim: int, grid_size: int,
                 addbias: bool = False, smooth_initialization: bool = False):
        super().__init__()
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.addbias = addbias

        grid_norm = (torch.arange(grid_size) + 1) ** 2 if smooth_initialization else np.sqrt(grid_size)
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, input_dim, grid_size) / (np.sqrt(input_dim) * grid_norm)
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, input_dim))

    def forward(self, x):
        shape = x.shape
        k = torch.reshape(torch.arange(1, self.grid_size + 1, device=x.device), (1, 1, self.grid_size))
        x_r = torch.reshape(x, (shape[0], shape[1], 1))
        c = torch.cos(k * x_r)
        s = torch.sin(k * x_r)
        y = torch.sum(c * self.fouriercoeffs[0:1], dim=-1)
        y += torch.sum(s * self.fouriercoeffs[1:2], dim=-1)
        if self.addbias:
            y += self.bias
        return y
