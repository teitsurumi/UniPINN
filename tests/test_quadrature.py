"""Tests for quadrature utilities."""

import numpy as np
from unipinn.numerics.quadrature import (
    gauss_jacobi_weights, plane_rectangle_quad, plane_delaunay_quad,
    plane_q4_quad,
)


def test_gauss_legendre_weights():
    nodes, weights = gauss_jacobi_weights(5, 0, 0)
    assert len(nodes) == 5
    assert abs(np.sum(weights) - 2.0) < 1e-12  # Integral of 1 over [-1,1]


def test_plane_rectangle_quad():
    # Single unit square element: [0,1] x [0,1]
    elements = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=float)
    result = plane_rectangle_quad(elements, 3, 3)
    xi, eta, wxi, weta, weights, x, y, J = result
    assert weights.shape == (1, 9, 1)
    assert abs(np.sum(weights * np.abs(J)) - 1.0) < 1e-12  # Area = 1


def test_plane_delaunay_quad():
    # Single right triangle: (0,0), (1,0), (0,1)
    elements = np.array([[[0, 0], [1, 0], [0, 1]]], dtype=float)
    result = plane_delaunay_quad(elements, 3, 3)
    _, _, _, _, weights, _, _, J = result
    assert weights.shape == (1, 9, 1)
    # Triangle area = 0.5
    total_area = np.sum(weights * np.abs(J))
    assert abs(total_area - 0.5) < 1e-10


def test_plane_q4_quad_unit_square():
    # Unit square Q4 element
    elements = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=float)
    result = plane_q4_quad(elements, 3, 3, return_full_jacobian=False)
    _, _, _, _, weights, x, y, Jdet = result
    total_area = np.sum(weights * np.abs(Jdet))
    assert abs(total_area - 1.0) < 1e-10
