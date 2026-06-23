"""Mesh data structures and shape functions for finite elements.

Defines ``MeshData`` (the canonical mesh container) and ``ShapeFunctions``
(reference-element mappings and sampling for T3, Q4, Q8, Q9).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Shape Functions
# ============================================================================

class ShapeFunctions:
    """Reference-element shape functions and sampling for various element types."""

    # ---- T3 (linear triangle) ----

    @staticmethod
    def t3_map(pts: np.ndarray):
        """Return a closure that maps reference triangle coords to physical space."""
        def evaluate(nodes):
            N1 = 1.0 - pts[:, 0] - pts[:, 1]
            N2 = pts[:, 0]
            N3 = pts[:, 1]
            x = N1[:, None] * nodes[0:1, :] + N2[:, None] * nodes[1:2, :] + N3[:, None] * nodes[2:3, :]
            return x
        return evaluate

    @staticmethod
    def t3_sample(nodes: np.ndarray, n_pts: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform sampling inside a T3 element. Returns (n_pts, 2)."""
        r1 = rng.random(n_pts)
        r2 = rng.random(n_pts)
        sqrt_r1 = np.sqrt(r1)
        xi = 1.0 - sqrt_r1
        eta = r2 * sqrt_r1
        pts = np.column_stack([xi, eta])
        return ShapeFunctions.t3_map(pts)(nodes)

    # ---- Q4 (bilinear quad) ----

    @staticmethod
    def q4_map(pts: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        """Map from reference square [-1,1]^2 to physical Q4."""
        xi, eta = pts[:, 0], pts[:, 1]
        N1 = 0.25 * (1 - xi) * (1 - eta)
        N2 = 0.25 * (1 + xi) * (1 - eta)
        N3 = 0.25 * (1 + xi) * (1 + eta)
        N4 = 0.25 * (1 - xi) * (1 + eta)
        return (N1[:, None] * nodes[0:1, :] + N2[:, None] * nodes[1:2, :]
                + N3[:, None] * nodes[2:3, :] + N4[:, None] * nodes[3:4, :])

    @staticmethod
    def q4_sample(nodes: np.ndarray, n_pts: int, rng: np.random.Generator) -> np.ndarray:
        pts = rng.uniform(-1, 1, (n_pts, 2))
        return ShapeFunctions.q4_map(pts, nodes)

    # ---- Q8 (serendipity quad) ----

    @staticmethod
    def q8_map(pts: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        xi, eta = pts[:, 0], pts[:, 1]
        N = np.zeros((len(xi), 8))
        N[:, 0] = 0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1)
        N[:, 1] = 0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1)
        N[:, 2] = 0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1)
        N[:, 3] = 0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1)
        N[:, 4] = 0.5 * (1 - xi ** 2) * (1 - eta)
        N[:, 5] = 0.5 * (1 + xi) * (1 - eta ** 2)
        N[:, 6] = 0.5 * (1 - xi ** 2) * (1 + eta)
        N[:, 7] = 0.5 * (1 - xi) * (1 - eta ** 2)
        return N @ nodes

    @staticmethod
    def q8_sample(nodes: np.ndarray, n_pts: int, rng: np.random.Generator) -> np.ndarray:
        pts = rng.uniform(-1, 1, (n_pts, 2))
        return ShapeFunctions.q8_map(pts, nodes)

    # ---- Q9 (Lagrange quad) ----

    @staticmethod
    def q9_map(pts: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        xi, eta = pts[:, 0], pts[:, 1]
        N = np.zeros((len(xi), 9))
        # Corners
        N[:, 0] = 0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1)
        N[:, 1] = 0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1)
        N[:, 2] = 0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1)
        N[:, 3] = 0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1)
        # Midsides
        N[:, 4] = 0.5 * (1 - xi ** 2) * (1 - eta)
        N[:, 5] = 0.5 * (1 + xi) * (1 - eta ** 2)
        N[:, 6] = 0.5 * (1 - xi ** 2) * (1 + eta)
        N[:, 7] = 0.5 * (1 - xi) * (1 - eta ** 2)
        # Center
        N[:, 8] = (1 - xi ** 2) * (1 - eta ** 2)
        return N @ nodes

    @staticmethod
    def q9_sample(nodes: np.ndarray, n_pts: int, rng: np.random.Generator) -> np.ndarray:
        pts = rng.uniform(-1, 1, (n_pts, 2))
        return ShapeFunctions.q9_map(pts, nodes)

    # ---- Dispatch ----

    @staticmethod
    def sample_element(elem_type: str, nodes: np.ndarray, n_pts: int,
                       rng: np.random.Generator) -> np.ndarray:
        """Dispatch sampling based on element type string."""
        dispatch = {
            "T3": ShapeFunctions.t3_sample,
            "Q4": ShapeFunctions.q4_sample,
            "Q8": ShapeFunctions.q8_sample,
            "Q9": ShapeFunctions.q9_sample,
        }
        if elem_type not in dispatch:
            raise ValueError(f"Unsupported element type: {elem_type}. Supported: {list(dispatch.keys())}")
        return dispatch[elem_type](nodes, n_pts, rng)


# ============================================================================
# Mesh Data Structure
# ============================================================================

@dataclass
class MeshData:
    """Container for mesh data loaded from .mesh files or constructed programmatically.

    Attributes:
        nodes: Node coordinates, shape (N, 2).
        elements: Element connectivity, shape (M, n_nodes_per_elem).
        elem_type: Element type string ("T3", "Q4", "Q8", "Q9").
        elem_areas: Precomputed element areas, shape (M,). Optional (T3 only by default).
        boundary_groups: Named boundary groups for mixed BCs.
    """
    nodes: np.ndarray
    elements: np.ndarray
    elem_type: str
    elem_areas: Optional[np.ndarray] = None
    boundary_groups: Dict[str, dict] = field(default_factory=dict)

    def extract_boundary_edges(self) -> Dict[str, List[Tuple[int, int]]]:
        """Extract boundary edges from mesh connectivity.

        Boundary edges are those belonging to exactly one element.
        Returns a dict mapping group name to list of (node1, node2) edge index pairs.
        If no groups are defined, all boundary edges go to 'default'.
        """
        edge_count: Dict[Tuple[int, int], int] = {}
        for elem in self.elements:
            if self.elem_type == "T3":
                edge_pairs = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
            else:
                perimeter = elem[:4]
                edge_pairs = [(perimeter[i], perimeter[(i + 1) % 4]) for i in range(4)]
            for n1, n2 in edge_pairs:
                key = (min(n1, n2), max(n1, n2))
                edge_count[key] = edge_count.get(key, 0) + 1

        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

        if not self.boundary_groups:
            return {"default": boundary_edges}

        result = {}
        for name, bgroup in self.boundary_groups.items():
            result[name] = bgroup.get("edges", [])
        if not result:
            result["default"] = boundary_edges
        return result
