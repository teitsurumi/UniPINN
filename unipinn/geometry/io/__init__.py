"""Geometry I/O dispatch and re-exports.

Provides a unified ``read_mesh_file`` dispatcher that routes to the correct
reader based on file extension, plus convenience re-exports.
"""

from unipinn.geometry.io.mesh_format import (
    load_mesh,
    load_collocation_points,
    load_evaluation_points,
    load_boundary_sample,
    load_data_sample,
)
from unipinn.geometry.io.gmsh_io import read_msh_to_numpy, q4_to_q8
from unipinn.geometry.io.abaqus_io import read_inp_to_numpy
from unipinn.geometry.io.vtk_io import read_vtk_to_numpy

import numpy as np


def read_mesh_file(filepath: str, elem_type: str = "Q4",
                   to_q8: bool = False) -> np.ndarray:
    """Read a mesh file and return element coordinate array.

    Automatically selects reader based on file extension:
        .msh  -> Gmsh reader
        .inp  -> Abaqus reader
        .vtk  -> VTK reader
        .mesh -> custom format reader (returns MeshData, not array)

    Args:
        filepath: Path to mesh file.
        elem_type: Element type for Gmsh files ("Q4", "Q8", "Q9").
        to_q8: If True, convert Q4 elements to Q8 by adding midside nodes.

    Returns:
        Element coordinate array of shape (num_elements, n_nodes, 3).
    """
    if filepath.endswith(".msh"):
        _, elements, _ = read_msh_to_numpy(filepath, elem_type=elem_type)
    elif filepath.endswith(".inp"):
        _, elements, _ = read_inp_to_numpy(filepath)
    elif filepath.endswith(".vtk"):
        _, elements, _ = read_vtk_to_numpy(filepath)
    else:
        raise ValueError(f"Unsupported file extension for: {filepath}")

    if elements is None:
        raise RuntimeError(f"Failed to read mesh file: {filepath}")

    if to_q8 and elements.shape[1] == 4:
        elements = q4_to_q8(elements)

    return elements
