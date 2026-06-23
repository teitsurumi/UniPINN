"""Abaqus .inp file reader for Q4 elements."""

import numpy as np
from typing import Tuple


def read_inp_to_numpy(inp_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an Abaqus .inp file and extract Q4 element data.

    Args:
        inp_file: Path to the .inp file.

    Returns:
        Tuple of (nodes_array, elements_array, connectivity):
            nodes_array: shape (num_nodes, 3)
            elements_array: shape (num_elements, 4, 3)
            connectivity: shape (num_elements, 4)
    """
    nodes = []
    elements = []
    reading_nodes = False
    reading_elements = False

    try:
        with open(inp_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith("*Node"):
                reading_nodes = True
                reading_elements = False
                continue
            elif line.startswith("*Element"):
                if "type=CPS4" in line or "type=CPE4" in line or "type=S4" in line:
                    reading_elements = True
                    reading_nodes = False
                continue
            elif line.startswith("*"):
                reading_nodes = False
                reading_elements = False
                continue

            if reading_nodes and line:
                parts = line.split(",")
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3]) if len(parts) > 3 else 0.0
                    while len(nodes) < node_id:
                        nodes.append(None)
                    nodes[node_id - 1] = [x, y, z]

            elif reading_elements and line:
                parts = line.split(",")
                if len(parts) >= 5:
                    elem_id = int(parts[0])
                    node_ids = [int(n) for n in parts[1:5]]
                    elements.append(node_ids)

        nodes_array = np.array(nodes)
        connectivity = np.array(elements) - 1  # Convert to 0-based indexing

        num_elements = len(elements)
        elements_array = np.zeros((num_elements, 4, 3))
        for i in range(num_elements):
            for j in range(4):
                elements_array[i, j, :] = nodes_array[connectivity[i, j], :]

        print(f"Loaded {inp_file}: {len(nodes)} nodes, {num_elements} Q4 elements")
        return nodes_array, elements_array, connectivity

    except Exception as e:
        print(f"Error reading INP file: {e}")
        return None, None, None
