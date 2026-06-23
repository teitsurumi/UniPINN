"""Tests for mesh I/O and geometry utilities."""

import numpy as np
import pytest
from pathlib import Path
from unipinn.geometry.types import MeshData, ShapeFunctions
from unipinn.geometry.io.mesh_format import load_mesh
from unipinn.geometry.generation import plane_rectangle_mesh
from unipinn.geometry.quality import shape_quality, condition_number


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "poisson2d"


def test_mesh_data_creation():
    nodes = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
    elements = np.array([[0, 1, 2]], dtype=int)
    mesh = MeshData(nodes=nodes, elements=elements, elem_type="T3")
    assert mesh.elem_type == "T3"
    assert mesh.nodes.shape == (3, 2)


def test_mesh_data_boundary_extraction():
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    elements = np.array([[0, 1, 2, 3]], dtype=int)
    mesh = MeshData(nodes=nodes, elements=elements, elem_type="Q4")
    edges = mesh.extract_boundary_edges()
    assert "default" in edges
    assert len(edges["default"]) == 4  # All 4 edges of a single quad are boundary


def test_shape_functions_q4_sample():
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    rng = np.random.default_rng(42)
    pts = ShapeFunctions.q4_sample(nodes, 100, rng)
    assert pts.shape == (100, 2)
    # All points should be within [-1, 1]^2 mapped to [0, 1]^2
    assert np.all(pts >= -0.01) and np.all(pts <= 1.01)


def test_load_mesh_file():
    """Test loading a .mesh file if data is available."""
    mesh_file = DATA_DIR / "disk.mesh"
    if not mesh_file.exists():
        pytest.skip("Mesh data file not found")
    mesh = load_mesh(str(mesh_file))
    assert mesh.nodes.shape[1] == 2
    assert mesh.elem_type in ("T3", "Q4", "Q8", "Q9")
    assert len(mesh.elements) > 0


def test_plane_rectangle_mesh():
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 3)
    gx, gy = np.meshgrid(x, y)
    nodes = np.stack([gx, gy]).reshape(2, 3, 4)
    elements = plane_rectangle_mesh(nodes)
    assert elements.shape == (6, 4, 2)  # 3*2=6 elements
