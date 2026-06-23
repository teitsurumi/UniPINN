"""Mesh generation utilities for simple 2D geometries.

Provides programmatic mesh construction for:
- Plane rectangle elements (structured grid)
- Delaunay triangle elements (from scattered nodes)
- Q4 elements derived from Delaunay triangulation
"""

import numpy as np
from scipy.spatial import Delaunay


def plane_rectangle_mesh(nodes: np.ndarray) -> np.ndarray:
    """Generate Q4 elements from a structured grid of nodes.

    Args:
        nodes: shape (2, NE_y+1, NE_x+1), node coordinate grid.
            nodes[0] = x-coordinates, nodes[1] = y-coordinates.

    Returns:
        elements: shape (NE_x * NE_y, 4, 2), quadrilateral element vertex coordinates.
    """
    NE_x = nodes.shape[2] - 1
    NE_y = nodes.shape[1] - 1
    elements = np.zeros((NE_x * NE_y, 4, 2), dtype=np.float32)

    for i in range(NE_x):
        for j in range(NE_y):
            elements[j * NE_x + i, 0, :] = nodes[:, j, i].reshape((1, 2))
            elements[j * NE_x + i, 1, :] = nodes[:, j, i + 1].reshape((1, 2))
            elements[j * NE_x + i, 2, :] = nodes[:, j + 1, i + 1].reshape((1, 2))
            elements[j * NE_x + i, 3, :] = nodes[:, j + 1, i].reshape((1, 2))
    return elements


def plane_delaunay_mesh(nodes: np.ndarray):
    """Generate Delaunay triangle elements from scattered node coordinates.

    Args:
        nodes: shape (2, N), where nodes[0] = x, nodes[1] = y.

    Returns:
        Tuple of (tri, delaunay_triangles):
            tri: scipy.spatial.Delaunay object.
            delaunay_triangles: shape (num_triangles, 3, 2), triangle vertex coordinates.
    """
    pts = np.hstack((nodes[0].flatten()[:, None], nodes[1].flatten()[:, None]))
    tri = Delaunay(pts)
    delaunay_triangles = pts[tri.simplices]
    return tri, delaunay_triangles


def plane_delaunay_q4_mesh(nodes: np.ndarray) -> np.ndarray:
    """Generate Q4 elements by splitting each Delaunay triangle into a quad.

    The longest edge of each triangle is split at its midpoint, producing
    a 4-node quadrilateral element.

    Args:
        nodes: shape (2, N), node coordinates.

    Returns:
        elements: shape (num_triangles, 4, 2), Q4 element vertex coordinates.
    """
    pts = np.hstack((nodes[0].flatten()[:, None], nodes[1].flatten()[:, None]))
    tri = Delaunay(pts)
    triangles = pts[tri.simplices]

    # Find longest edge per triangle
    l = np.sum(
        np.concatenate((
            np.square(triangles[:, 1:2, :] - triangles[:, 0:1, :]),
            np.square(triangles[:, 2:3, :] - triangles[:, 1:2, :]),
            np.square(triangles[:, 0:1, :] - triangles[:, 2:3, :]),
        ), axis=1),
        axis=-1,
    )
    max_indices = np.argmax(l, axis=1, keepdims=True)

    q4_elements = np.zeros(
        (triangles.shape[0], triangles.shape[1] + 1, triangles.shape[2])
    )
    for i in range(max_indices.shape[0]):
        idx = max_indices[i][0]
        v = triangles[i, :, :]
        if idx == 0:
            q4_elements[i] = np.insert(v, 1, 0.5 * (v[0:1] + v[1:2]), axis=0)
        elif idx == 1:
            q4_elements[i] = np.insert(v, 2, 0.5 * (v[1:2] + v[2:3]), axis=0)
        elif idx == 2:
            q4_elements[i] = np.insert(v, 3, 0.5 * (v[2:3] + v[0:1]), axis=0)
        else:
            raise ValueError("max_indices >= 3")
    return q4_elements
