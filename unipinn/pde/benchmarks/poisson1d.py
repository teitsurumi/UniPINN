"""1D Poisson equation benchmarks with exact solutions.

Problem: -u''(x) = f(x) on [x_min, x_max]
with Dirichlet or Neumann boundary conditions.

References:
    [1] E. Kharazmi et al., hp-VPINNs, CMAME 374 (2021) 113547.
"""

import numpy as np
from abc import ABC
from typing import Callable, Dict, List, Optional, Tuple
from pyDOE3 import lhs

from unipinn.pde.benchmarks.base import BaseBenchmark


# ============================================================================
# Registry
# ============================================================================

class Poisson1DBenchmarkIndex:
    """Central registry for 1D Poisson benchmarks."""
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(bench_cls):
            if name in cls._registry:
                raise ValueError(f"Benchmark name '{name}' already registered.")
            cls._registry[name] = bench_cls
            return bench_cls
        return wrapper

    @classmethod
    def get(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Benchmark '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._registry.keys())

    def __getitem__(self, name: str): return self.get(name)
    def __contains__(self, name: str) -> bool: return name in self._registry


# ============================================================================
# Base class
# ============================================================================

class BasePoisson1DBenchmark(BaseBenchmark, ABC):
    """Shared logic for 1D Poisson benchmarks."""

    def __init__(self, domain: Tuple[float, float], bc_type: str,
                 u: Callable, u_prime: Callable, f: Callable, description: str):
        self.domain = domain
        self.bc_type = bc_type
        self.u = u
        self.u_prime = u_prime
        self.f = f
        self.description = description

    # -- Sampling utilities --

    @staticmethod
    def _sample_points(n: int, domain: Tuple[float, float], method: str,
                       seed: int, lhs_criterion: Optional[str] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x_min, x_max = domain
        if method == "uniform":
            return np.linspace(x_min, x_max, n)
        elif method == "random":
            return rng.uniform(x_min, x_max, n)
        elif method == "chebyshev":
            k = np.arange(n)
            return (0.5 * (x_max + x_min) + 0.5 * (x_max - x_min) * np.cos(np.pi * k / (n - 1)))[::-1]
        elif method == "lhs":
            samples = lhs(1, samples=n, criterion=lhs_criterion)
            return x_min + (x_max - x_min) * samples.flatten()
        else:
            raise ValueError("method must be 'uniform', 'random', 'chebyshev', or 'lhs'")

    @staticmethod
    def _sample_boundary(domain: Tuple[float, float]) -> np.ndarray:
        return np.array([domain[0], domain[1]], dtype=np.float64)

    # -- Core generation --

    def generate(self, n_colloc: int = 1000, n_eval: int = 500,
                 sampling_method: str = "uniform", seed: int = 42,
                 lhs_criterion: Optional[str] = "cm") -> Dict[str, np.ndarray]:
        """Generate training and evaluation data.

        Args:
            n_colloc: number of collocation points.
            n_eval: number of evaluation grid points.
            sampling_method: 'uniform', 'random', 'chebyshev', or 'lhs'.
            seed: random seed.
            lhs_criterion: LHS criterion (e.g., 'cm' for center-maximin).

        Returns:
            Dictionary with case_name, domain, bc_type, collocation, boundary,
            evaluation data, and description.
        """
        x_min, x_max = self.domain
        eps = 1e-6

        # Collocation points (exclude boundary for Dirichlet)
        if self.bc_type == "dirichlet":
            x_col = self._sample_points(n_colloc, [x_min + eps, x_max - eps],
                                        sampling_method, seed=seed, lhs_criterion=lhs_criterion)
        else:
            x_col = self._sample_points(n_colloc, self.domain,
                                        sampling_method, seed=seed, lhs_criterion=lhs_criterion)

        f_col = self.f(x_col)

        # Boundary data
        x_bc = np.array([x_min, x_max], dtype=np.float64)
        if self.bc_type == "dirichlet":
            bc_vals = np.array([self.u(x_min), self.u(x_max)], dtype=np.float64)
            u_bc, du_bc = bc_vals, None
        else:
            du_vals = np.array([self.u_prime(x_min), self.u_prime(x_max)], dtype=np.float64)
            u_bc, du_bc = None, du_vals

        # Evaluation grid
        x_eval = np.linspace(x_min, x_max, n_eval)
        u_eval = self.u(x_eval)
        f_eval = self.f(x_eval)

        return {
            "case_name": self.__class__.__name__.replace("Benchmark", "").lower(),
            "domain": np.array(self.domain),
            "bc_type": self.bc_type,
            "x_colloc": x_col.astype(np.float64),
            "f_colloc": f_col.astype(np.float64),
            "x_bc": x_bc,
            "u_bc": u_bc,
            "du_bc": du_bc,
            "x_eval": x_eval.astype(np.float64),
            "u_eval": u_eval.astype(np.float64),
            "f_eval": f_eval.astype(np.float64),
            "description": self.description,
        }


# ============================================================================
# Concrete benchmarks
# ============================================================================

@Poisson1DBenchmarkIndex.register("steep_solution")
class SteepSolutionBenchmark(BasePoisson1DBenchmark):
    """Steep solution: u = 0.1*sin(a*x) + tanh(b*x). Reference [1]."""
    def __init__(self, a: float = 4 * np.pi, b: float = 20.0,
                 domain: Tuple[float, float] = (-1.0, 1.0), bc_type: str = "dirichlet"):
        u = lambda x: 0.1 * np.sin(a * x) + np.tanh(b * x)
        u_prime = lambda x: 0.1 * a * np.cos(a * x) + b * (1.0 - np.tanh(b * x) ** 2)
        f = lambda x: 0.1 * a ** 2 * np.sin(a * x) + 2 * b ** 2 * np.tanh(b * x) * (1.0 - np.tanh(b * x) ** 2)
        super().__init__(domain, bc_type, u, u_prime, f,
                         description=f"Steep solution (a={a:.2f}, b={b:.2f})")


@Poisson1DBenchmarkIndex.register("boundary_layer_exp")
class BoundaryLayerExpBenchmark(BasePoisson1DBenchmark):
    """Boundary layer with exponential: u = 0.1*sin(a*x) + exp((b-(x+1))/b). Reference [1]."""
    def __init__(self, a: float = 5 * np.pi, b: float = 0.01,
                 domain: Tuple[float, float] = (-1.0, 1.0), bc_type: str = "dirichlet"):
        u = lambda x: 0.1 * np.sin(a * x) + np.exp((b - (x + 1)) / b)
        u_prime = lambda x: 0.1 * a * np.cos(a * x) - np.exp((b - (x + 1)) / b) / b
        f = lambda x: 0.1 * a ** 2 * np.sin(a * x) - np.exp((b - (x + 1)) / b) / b ** 2
        super().__init__(domain, bc_type, u, u_prime, f,
                         description=f"Boundary layer (a={a:.2f}, b={b:.4f})")


@Poisson1DBenchmarkIndex.register("asymmetric_steep")
class AsymmetricSteepBenchmark(BasePoisson1DBenchmark):
    """Asymmetric steep: u = 0.1*sin(a*x) + tanh(b*(x+0.1)). Reference [1]."""
    def __init__(self, a: float = 8 * np.pi, b: float = 0.01,
                 domain: Tuple[float, float] = (-1.0, 1.0), bc_type: str = "dirichlet"):
        u = lambda x: 0.1 * np.sin(a * x) + np.tanh(b * (x + 0.1))
        u_prime = lambda x: 0.1 * a * np.cos(a * x) + b * (1.0 - np.tanh(b * (x + 0.1)) ** 2)
        f = lambda x: (0.1 * a ** 2 * np.sin(a * x)
                       + 2 * b ** 2 * np.tanh(b * (x + 0.1)) * (1.0 - np.tanh(b * (x + 0.1)) ** 2))
        super().__init__(domain, bc_type, u, u_prime, f,
                         description=f"Asymmetric steep (a={a:.2f}, b={b:.4f})")


@Poisson1DBenchmarkIndex.register("discontinuous_jump")
class DiscontinuousFunctionBenchmark(BasePoisson1DBenchmark):
    """Piecewise solution with jump discontinuity at x=0."""
    def __init__(self):
        PI = np.pi

        def u(x):
            return np.where(x < 0,
                            2.0 * np.sin(4.0 * PI * x),
                            6.0 + np.exp(1.2 * x) * np.sin(12.0 * PI * x))

        def u_prime(x):
            return np.where(x < 0,
                            8.0 * PI * np.cos(4.0 * PI * x),
                            np.exp(1.2 * x) * (1.2 * np.sin(12.0 * PI * x) + 12.0 * PI * np.cos(12.0 * PI * x)))

        def f(x):
            A, B = 1.2, 12.0 * PI
            coeff_sin = A ** 2 - B ** 2
            coeff_cos = 2.0 * A * B
            u_dd_pos = np.exp(1.2 * x) * (coeff_sin * np.sin(B * x) + coeff_cos * np.cos(B * x))
            return np.where(x < 0,
                            32.0 * (PI ** 2) * np.sin(4.0 * PI * x),
                            -u_dd_pos)

        super().__init__(
            domain=(-1.0, 1.0), bc_type="dirichlet",
            u=u, u_prime=u_prime, f=f,
            description="Piecewise solution with jump at x=0. u(-1)=0, u(1)=6."
        )
