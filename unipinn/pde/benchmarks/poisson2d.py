"""2D Poisson equation benchmarks with exact solutions.

Three benchmark tiers:
    Case 1: Rectangular domain with analytical solution.
    Case 2: Irregular domain with file-based sampling (collocation, boundary, evaluation).
    Case 3: Irregular domain with mesh-based sampling (T3, Q4, Q8, Q9 elements).

Problem: -Laplacian(u) = f(x, y)
"""

import numpy as np
from abc import ABC
from typing import Callable, Dict, List, Optional, Tuple, Union
from pyDOE3 import lhs

from unipinn.pde.benchmarks.base import BaseBenchmark
from unipinn.geometry.types import MeshData, ShapeFunctions
from unipinn.geometry.io.mesh_format import (
    load_mesh, load_collocation_points, load_evaluation_points,
    load_boundary_sample, load_data_sample,
)


# ============================================================================
# Registry
# ============================================================================

class Poisson2DBenchmarkIndex:
    """Central registry for 2D Poisson benchmarks."""
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

    def __getitem__(self, name: str): return self.__class__.get(name)
    def __contains__(self, name: str) -> bool: return name in self.__class__._registry


# ============================================================================
# Base class (Case 1: rectangular analytical)
# ============================================================================

class BasePoisson2DBenchmark(BaseBenchmark, ABC):
    """Base class for 2D Poisson benchmarks with analytical solutions."""

    def __init__(self, domain: Tuple[Tuple[float, float], Tuple[float, float]],
                 bc_type: str, u: Callable, u_grad: Callable, f: Callable,
                 description: str):
        self.domain = domain
        self.bc_type = bc_type
        self.u = u
        self.u_grad = u_grad
        self.f = f
        self.description = description

    @staticmethod
    def _sample_points_2d(n: int, domain, method: str, seed: int,
                          lhs_criterion: Optional[str] = None,
                          exclude_boundary: bool = False) -> np.ndarray:
        """Sample points in a 2D rectangular domain. Returns (n, 2)."""
        rng = np.random.default_rng(seed)
        (x_min, x_max), (y_min, y_max) = domain
        eps = 1e-6 if exclude_boundary else 0.0

        if method == "uniform":
            nx = int(np.sqrt(n))
            ny = max(n // nx, 1)
            xs = np.linspace(x_min + eps, x_max - eps, nx)
            ys = np.linspace(y_min + eps, y_max - eps, ny)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            return np.column_stack([X.ravel(), Y.ravel()])[:n]
        elif method == "random":
            x = rng.uniform(x_min + eps, x_max - eps, n)
            y = rng.uniform(y_min + eps, y_max - eps, n)
            return np.column_stack([x, y])
        elif method == "lhs":
            samples = lhs(2, samples=n, criterion=lhs_criterion)
            x = x_min + eps + (x_max - x_min - 2 * eps) * samples[:, 0]
            y = y_min + eps + (y_max - y_min - 2 * eps) * samples[:, 1]
            return np.column_stack([x, y])
        else:
            raise ValueError("method must be 'uniform', 'random', or 'lhs'")

    @staticmethod
    def _sample_boundary_2d(domain, n_per_edge: int = 50,
                            exclude_corners: bool = False) -> np.ndarray:
        """Sample on all 4 edges. Returns (4*n_per_edge, 2)."""
        (x_min, x_max), (y_min, y_max) = domain
        eps = 1e-6 if exclude_corners else 0.0
        t = np.linspace(0, 1, n_per_edge)
        edges = [
            np.column_stack([x_min + (x_max - x_min) * t, np.full(n_per_edge, y_min + eps)]),
            np.column_stack([x_min + (x_max - x_min) * t, np.full(n_per_edge, y_max - eps)]),
            np.column_stack([np.full(n_per_edge, x_min + eps), y_min + (y_max - y_min) * t]),
            np.column_stack([np.full(n_per_edge, x_max - eps), y_min + (y_max - y_min) * t]),
        ]
        return np.vstack(edges)

    def generate(self, n_colloc: int = 2500, n_eval: int = 100,
                 sampling_method: str = "lhs", seed: int = 42,
                 lhs_criterion: Optional[str] = "cm",
                 n_bc_per_edge: int = 50) -> Dict[str, np.ndarray]:
        exclude_bc = (self.bc_type == "dirichlet")
        x_col = self._sample_points_2d(
            n_colloc, self.domain, sampling_method, seed=seed,
            lhs_criterion=lhs_criterion, exclude_boundary=exclude_bc,
        )
        f_col = self.f(x_col)

        x_bc = self._sample_boundary_2d(self.domain, n_per_edge=n_bc_per_edge, exclude_corners=exclude_bc)
        u_vals_bc = self.u(x_bc)
        dudx_bc, dudy_bc = self.u_grad(x_bc)
        du_bc_all = np.column_stack([dudx_bc, dudy_bc])

        if self.bc_type == "dirichlet":
            u_bc, du_bc_out = u_vals_bc, None
        else:
            u_bc, du_bc_out = None, du_bc_all

        n_side = n_eval if sampling_method == "uniform" else int(np.sqrt(n_eval))
        xs = np.linspace(self.domain[0][0], self.domain[0][1], n_side)
        ys = np.linspace(self.domain[1][0], self.domain[1][1], n_side)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        x_eval = np.column_stack([X.ravel(), Y.ravel()])
        u_eval = self.u(x_eval)
        f_eval = self.f(x_eval)

        return {
            "case_name": self.__class__.__name__.replace("Benchmark", "").lower(),
            "domain": np.array(self.domain),
            "bc_type": self.bc_type,
            "x_colloc": x_col.astype(np.float64),
            "f_colloc": f_col.astype(np.float64),
            "x_bc": x_bc.astype(np.float64),
            "u_bc": u_bc.astype(np.float64) if u_bc is not None else None,
            "du_bc": du_bc_out.astype(np.float64) if du_bc_out is not None else None,
            "x_eval": x_eval.astype(np.float64),
            "u_eval": u_eval.astype(np.float64),
            "f_eval": f_eval.astype(np.float64),
            "description": self.description,
        }


# ============================================================================
# Case 1: Concrete rectangular benchmarks
# ============================================================================

@Poisson2DBenchmarkIndex.register("harmonic_rational")
class HarmonicRationalBenchmark(BasePoisson2DBenchmark):
    r"""Harmonic function u = 2(1+y) / ((3+x)^2 + (1+y)^2). Reference [1].

    This is the imaginary part of 2/(z + 3 + i), hence satisfies Laplace's
    equation exactly (f = 0). Provides a smooth, non-trivial test for
    Dirichlet boundary enforcement.

    Default domain: [-1, 1] x [-1, 1].
    """
    def __init__(self, domain=((-1, 1), (-1, 1)), bc_type: str = "dirichlet"):
        def u(pts):
            X = 3.0 + pts[:, 0]
            Y = 1.0 + pts[:, 1]
            D = X ** 2 + Y ** 2
            return 2.0 * Y / D

        def u_grad(pts):
            X = 3.0 + pts[:, 0]
            Y = 1.0 + pts[:, 1]
            D = X ** 2 + Y ** 2
            D2 = D ** 2
            dudx = -4.0 * X * Y / D2
            dudy = 2.0 * (X ** 2 - Y ** 2) / D2
            return dudx, dudy

        def f(pts):
            # Harmonic: Laplacian(u) = 0, so f = -Laplacian(u) = 0
            return np.zeros(pts.shape[0])

        super().__init__(domain, bc_type, u, u_grad, f,
                         "Harmonic rational: u=2(1+y)/((3+x)^2+(1+y)^2), f=0. Ref [1]")


@Poisson2DBenchmarkIndex.register("steep_product_2d")
class SteepProduct2DBenchmark(BasePoisson2DBenchmark):
    r"""u = (0.1*sin(wx*x) + tanh(r1*x)) * sin(wy*y). Reference [1].

    Combines high-frequency oscillation in x with a steep tanh transition,
    modulated by a sinusoidal envelope in y.  Challenging for PINNs due to
    the multi-scale structure.

    Default domain: [-1, 1] x [-1, 1], wx = 4*pi, wy = 4*pi, r1 = 10.
    """
    def __init__(self, wx: float = 4 * np.pi, wy: float = 4 * np.pi,
                 r1: float = 10.0,
                 domain=((-1, 1), (-1, 1)), bc_type: str = "dirichlet"):
        def g(x):
            return 0.1 * np.sin(wx * x) + np.tanh(r1 * x)

        def g_pp(x):
            s = np.cosh(r1 * x)
            sech2 = 1.0 / (s ** 2)
            return -0.1 * wx ** 2 * np.sin(wx * x) + 2.0 * r1 ** 2 * np.tanh(r1 * x) * sech2

        def u(pts):
            return g(pts[:, 0]) * np.sin(wy * pts[:, 1])

        def u_grad(pts):
            x, y = pts[:, 0], pts[:, 1]
            # dg/dx
            s = np.cosh(r1 * x)
            sech2 = 1.0 / (s ** 2)
            dg = 0.1 * wx * np.cos(wx * x) + r1 * sech2
            dudx = dg * np.sin(wy * y)
            dudy = g(x) * wy * np.cos(wy * y)
            return dudx, dudy

        def f(pts):
            x, y = pts[:, 0], pts[:, 1]
            h = np.sin(wy * y)
            return -(g_pp(x) * h - wy ** 2 * g(x) * h)

        super().__init__(domain, bc_type, u, u_grad, f,
                         f"Steep product 2D: wx={wx:.2f}, wy={wy:.2f}, r1={r1:.1f}. Ref [1]")


# ============================================================================
# Case 2: Irregular domain with file-based sampling
# ============================================================================

class IrregularDomainBenchmark(BasePoisson2DBenchmark):
    """Irregular domain with data loaded from files.

    Supports collocation, boundary (grouped), evaluation, and data sample files.
    """
    def __init__(self, collocation_file=None, boundary_file=None,
                 evaluation_file=None, data_file=None,
                 u_func=None, f_func=None,
                 domain_description: str = "Irregular domain"):
        self.domain = None
        self.bc_type = "mixed"
        self.u = u_func
        self.f = f_func
        self.description = domain_description
        self.collocation_file = collocation_file
        self.boundary_file = boundary_file
        self.evaluation_file = evaluation_file
        self.data_file = data_file
        self._colloc_data = None
        self._boundary_data = None
        self._eval_data = None
        self._data_sample = None

    def load_collocation(self, file=None):
        if self._colloc_data is not None: return self._colloc_data
        self._colloc_data = load_collocation_points(file or self.collocation_file)
        return self._colloc_data

    def load_boundary(self, file=None):
        if self._boundary_data is not None: return self._boundary_data
        self._boundary_data = load_boundary_sample(file or self.boundary_file)
        return self._boundary_data

    def load_evaluation(self, file=None):
        if self._eval_data is not None: return self._eval_data
        self._eval_data = load_evaluation_points(file or self.evaluation_file)
        return self._eval_data

    def load_data_sample(self, file=None):
        if self._data_sample is not None: return self._data_sample
        self._data_sample = load_data_sample(file or self.data_file)
        return self._data_sample

    def set_collocation(self, points: np.ndarray):
        self._colloc_data = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    def set_boundary(self, groups: Dict[str, dict]):
        self._boundary_data = groups

    def set_evaluation(self, points: np.ndarray):
        self._eval_data = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    def subsample_collocation(self, n: int, method: str = "random", seed: int = 42):
        pts = self.load_collocation()
        if n >= len(pts): return pts.copy()
        rng = np.random.default_rng(seed)
        if method == "random":
            idx = rng.choice(len(pts), size=n, replace=False)
        elif method == "uniform":
            idx = np.linspace(0, len(pts) - 1, n, dtype=int)
        else:
            raise ValueError("method must be 'random' or 'uniform'")
        return pts[idx].copy()

    def generate(self, n_colloc=None, colloc_method="random", seed=42, **kwargs):
        result = {"case_name": "irregular", "domain": None, "bc_type": "mixed",
                  "description": self.description}
        if n_colloc is not None:
            x_col = self.subsample_collocation(n_colloc, colloc_method, seed)
        else:
            x_col = self.load_collocation()
        result["x_colloc"] = x_col
        if self.f is not None:
            result["f_colloc"] = self.f(x_col)
        if self._boundary_data is not None or self.boundary_file:
            bc = self.load_boundary()
            result["x_bc_groups"] = {k: v["points"] for k, v in bc.items()}
            result["u_bc_groups"] = {k: v["values"] for k, v in bc.items()}
            result["normals"] = {k: v.get("normals") for k, v in bc.items()}
        if self._eval_data is not None or self.evaluation_file:
            x_eval = self.load_evaluation()
            result["x_eval"] = x_eval
            if self.u is not None: result["u_eval"] = self.u(x_eval)
            if self.f is not None: result["f_eval"] = self.f(x_eval)
        if self._data_sample is not None or self.data_file:
            result["data_sample"] = self.load_data_sample()
        return result


# ============================================================================
# Case 3: Irregular domain with mesh-based sampling
# ============================================================================

class MeshBasedBenchmark(BasePoisson2DBenchmark):
    """Irregular domain with mesh-based sampling (T3, Q4, Q8, Q9)."""
    def __init__(self, mesh_file=None, mesh_data=None,
                 u_func=None, f_func=None,
                 domain_description: str = "Mesh-defined irregular domain"):
        self.domain = None
        self.bc_type = "mixed"
        self.u = u_func
        self.f = f_func
        self.description = domain_description
        self._mesh = None
        self.mesh_file = mesh_file
        if mesh_data is not None:
            self._mesh = mesh_data

    @property
    def mesh(self) -> MeshData:
        if self._mesh is None:
            if self.mesh_file is None:
                raise ValueError("No mesh file or mesh data provided.")
            self._mesh = load_mesh(self.mesh_file)
        return self._mesh

    def set_mesh(self, mesh_data: MeshData):
        self._mesh = mesh_data

    def sample_collocation_from_mesh(self, n_pts: int, seed: int = 42,
                                     method: str = "element_weighted") -> np.ndarray:
        """Sample collocation points from mesh elements.

        Methods: 'element_weighted' (area-proportional), 'uniform_per_element', 'nodes'.
        """
        rng = np.random.default_rng(seed)
        m = self.mesh
        n_elem = len(m.elements)

        if method == "element_weighted":
            if m.elem_type == "T3" and m.elem_areas is not None:
                weights = m.elem_areas / m.elem_areas.sum()
                counts = np.maximum(1, np.floor(weights * n_pts).astype(int))
                deficit = n_pts - counts.sum()
                if deficit > 0:
                    extra_idx = rng.choice(n_elem, size=deficit, p=weights)
                    for idx in extra_idx:
                        counts[idx] += 1
                elif deficit < 0:
                    sorted_idx = np.argsort(-m.elem_areas)
                    i = 0
                    while counts.sum() > n_pts:
                        if counts[sorted_idx[i]] > 1:
                            counts[sorted_idx[i]] -= 1
                        i += 1
            else:
                n0 = m.nodes[m.elements[:, 0]]
                n2 = m.nodes[m.elements[:, 2]]
                approx_areas = np.abs((n2[:, 0] - n0[:, 0]) * (n2[:, 1] - n0[:, 1]))
                weights = approx_areas / approx_areas.sum()
                elem_indices = rng.choice(n_elem, size=n_pts, p=weights)
                counts = np.array([np.sum(elem_indices == i) for i in range(n_elem)])

            all_pts = []
            for i, cnt in enumerate(counts):
                if cnt == 0: continue
                elem_nodes = m.nodes[m.elements[i]]
                pts = ShapeFunctions.sample_element(m.elem_type, elem_nodes, int(cnt), rng)
                all_pts.append(pts)
            return np.vstack(all_pts)

        elif method == "uniform_per_element":
            pts_per = max(1, n_pts // n_elem)
            remainder = n_pts - pts_per * n_elem
            all_pts = []
            for i in range(n_elem):
                cnt = pts_per + (1 if i < remainder else 0)
                elem_nodes = m.nodes[m.elements[i]]
                pts = ShapeFunctions.sample_element(m.elem_type, elem_nodes, cnt, rng)
                all_pts.append(pts)
            return np.vstack(all_pts)

        elif method == "nodes":
            return m.nodes.copy()
        else:
            raise ValueError("method must be 'element_weighted', 'uniform_per_element', or 'nodes'")

    def sample_boundary_from_mesh(self, n_pts_per_group: int = 50,
                                  seed: int = 42) -> Dict[str, np.ndarray]:
        """Sample boundary points from mesh boundary groups (length-weighted)."""
        rng = np.random.default_rng(seed)
        m = self.mesh
        edge_groups = m.extract_boundary_edges()
        result = {}

        for name, edges in edge_groups.items():
            if not edges:
                result[name] = np.empty((0, 2))
                continue
            n1_nodes = m.nodes[[e[0] for e in edges]]
            n2_nodes = m.nodes[[e[1] for e in edges]]
            lengths = np.sqrt((n2_nodes[:, 0] - n1_nodes[:, 0]) ** 2
                              + (n2_nodes[:, 1] - n1_nodes[:, 1]) ** 2)
            total = lengths.sum()
            if total == 0:
                result[name] = np.empty((0, 2))
                continue
            probs = lengths / total
            counts = np.maximum(1, np.floor(probs * n_pts_per_group).astype(int))
            deficit = n_pts_per_group - counts.sum()
            if deficit > 0:
                extra = rng.choice(len(edges), size=deficit, p=probs)
                for idx in extra:
                    counts[idx] += 1
            all_pts = []
            for i, (n1_idx, n2_idx) in enumerate(edges):
                if counts[i] == 0: continue
                p1, p2 = m.nodes[n1_idx], m.nodes[n2_idx]
                t = np.linspace(0, 1, counts[i])
                pts = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
                all_pts.append(pts)
            result[name] = np.vstack(all_pts) if all_pts else np.empty((0, 2))
        return result

    def sample_evaluation_from_mesh(self, method: str = "dense_nodes",
                                    n_pts: int = 10000, seed: int = 42) -> np.ndarray:
        m = self.mesh
        if method == "dense_nodes":
            return m.nodes.copy()
        elif method == "dense_sample":
            return self.sample_collocation_from_mesh(n_pts, seed=seed)
        elif method == "uniform_grid":
            x_min, y_min = m.nodes.min(axis=0)
            x_max, y_max = m.nodes.max(axis=0)
            n_side = int(np.sqrt(n_pts))
            xs = np.linspace(x_min, x_max, n_side)
            ys = np.linspace(y_min, y_max, n_side)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            return np.column_stack([X.ravel(), Y.ravel()])
        else:
            raise ValueError("method must be 'dense_nodes', 'dense_sample', or 'uniform_grid'")

    def generate(self, n_colloc: int = 5000, n_bc_per_group: int = 100,
                 colloc_method: str = "element_weighted",
                 eval_method: str = "dense_nodes", n_eval: int = 10000,
                 seed: int = 42, **kwargs):
        result = {"case_name": "mesh_based", "domain": None, "bc_type": "mixed",
                  "description": self.description, "mesh": self.mesh}
        x_col = self.sample_collocation_from_mesh(n_colloc, seed=seed, method=colloc_method)
        result["x_colloc"] = x_col
        if self.f is not None:
            result["f_colloc"] = self.f(x_col)
        result["x_bc_groups"] = self.sample_boundary_from_mesh(n_bc_per_group, seed=seed)
        x_eval = self.sample_evaluation_from_mesh(eval_method, n_eval, seed=seed)
        result["x_eval"] = x_eval
        if self.u is not None: result["u_eval"] = self.u(x_eval)
        if self.f is not None: result["f_eval"] = self.f(x_eval)
        return result
