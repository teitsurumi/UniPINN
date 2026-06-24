"""Gmsh .msh file reader for Q4, Q8, and Q9 elements.

Uses the Gmsh Python API to read mesh files and extract node coordinates,
element connectivity, and element coordinate arrays.
"""

import numpy as np
from typing import Optional, Tuple

# Gmsh element type IDs and their node counts for 2D quadrilaterals
_GMSH_QUAD_TYPES = {
    "Q4": (3, 4),    # type_id=3,  4 nodes
    "Q8": (16, 8),   # type_id=16, 8 nodes
    "Q9": (10, 9),   # type_id=10, 9 nodes
}


def read_msh_to_numpy(msh_file: str, elem_type: str = "Q4"
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a Gmsh .msh file and extract node/element data.

    Args:
        msh_file: Path to the .msh file.
        elem_type: Element type, one of "Q4", "Q8", "Q9".

    Returns:
        Tuple of (nodes_array, elements_array, connectivity):
            nodes_array: shape (num_nodes, 3)
            elements_array: shape (num_elements, n_nodes_per_elem, 3)
            connectivity: shape (num_elements, n_nodes_per_elem)
    """
    import gmsh

    if elem_type not in _GMSH_QUAD_TYPES:
        raise ValueError(f"Unsupported element type: {elem_type}. Use one of {list(_GMSH_QUAD_TYPES)}")

    gmsh_type_id, n_nodes_per_elem = _GMSH_QUAD_TYPES[elem_type]

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.open(msh_file)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        num_nodes = len(node_tags)
        nodes_array = np.array(node_coords).reshape(num_nodes, 3)

        node_tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

        element_types, element_tags, element_nodes = gmsh.model.mesh.getElements(2)

        quad_elements = None
        for i, et in enumerate(element_types):
            if et == gmsh_type_id:
                quad_elements = {"tags": element_tags[i], "nodes": element_nodes[i]}
                break

        if quad_elements is None:
            raise ValueError(f"No {elem_type} elements (type={gmsh_type_id}) found in file")

        num_elems = len(quad_elements["tags"])
        connectivity = np.array(quad_elements["nodes"]).reshape(num_elems, n_nodes_per_elem)

        elements_array = np.zeros((num_elems, n_nodes_per_elem, 3))
        for i in range(num_elems):
            for j in range(n_nodes_per_elem):
                node_idx = node_tag_to_idx[connectivity[i, j]]
                elements_array[i, j, :] = nodes_array[node_idx, :]

        print(f"Loaded {msh_file}: {num_nodes} nodes, {num_elems} {elem_type} elements")
        return nodes_array, elements_array, connectivity

    except Exception as e:
        print(f"Error reading {msh_file}: {e}")
        return None, None, None
    finally:
        gmsh.finalize()


def q4_to_q8(elements: np.ndarray) -> np.ndarray:
    """Convert Q4 elements to Q8 by adding midside nodes.

    Args:
        elements: shape (N, 4, 3), Q4 element node coordinates.

    Returns:
        shape (N, 8, 3), Q8 element node coordinates.
    """
    mid_nodes = 0.5 * np.stack([
        elements[:, 0] + elements[:, 1],
        elements[:, 1] + elements[:, 2],
        elements[:, 2] + elements[:, 3],
        elements[:, 3] + elements[:, 0],
    ], axis=1)
    return np.concatenate([elements, mid_nodes], axis=1)
