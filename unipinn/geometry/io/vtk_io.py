"""Legacy VTK file reader for Q4 elements."""

import numpy as np
from typing import Tuple


def read_vtk_to_numpy(vtk_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a legacy VTK file and extract Q4 element data.

    Args:
        vtk_file: Path to the VTK file.

    Returns:
        Tuple of (nodes_array, elements_array, connectivity):
            nodes_array: shape (num_nodes, 3)
            elements_array: shape (num_elements, 4, 3)
            connectivity: shape (num_elements, 4)
    """
    try:
        with open(vtk_file, "r") as f:
            lines = f.readlines()

        nodes = []
        elements = []
        reading_points = False
        reading_cells = False

        for line in lines:
            line = line.strip()

            if line.startswith("POINTS"):
                reading_points = True
                reading_cells = False
                continue
            elif line.startswith("CELLS"):
                reading_cells = True
                reading_points = False
                continue
            elif line.startswith("CELL_TYPES"):
                break

            if reading_points and line:
                parts = line.split()
                for j in range(0, len(parts), 3):
                    if len(parts[j:j + 3]) == 3:
                        nodes.append([float(parts[j]), float(parts[j + 1]), float(parts[j + 2])])

            if reading_cells and line:
                parts = line.split()
                if len(parts) >= 5 and parts[0] == "4":  # Quadrilateral cell type
                    node_ids = [int(parts[k]) for k in range(1, 5)]
                    elements.append(node_ids)

        nodes_array = np.array(nodes)
        connectivity = np.array(elements)

        num_elements = len(elements)
        elements_array = np.zeros((num_elements, 4, 3))
        for i in range(num_elements):
            for j in range(4):
                elements_array[i, j, :] = nodes_array[connectivity[i, j], :]

        print(f"Loaded {vtk_file}: {len(nodes)} nodes, {num_elements} Q4 elements")
        return nodes_array, elements_array, connectivity

    except Exception as e:
        print(f"Error reading VTK file: {e}")
        return None, None, None
