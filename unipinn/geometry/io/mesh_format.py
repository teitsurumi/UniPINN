"""Custom .mesh file format reader/writer and point-cloud data loaders.

File format specification:
    *NODES
    <x1> <y1>
    ...

    *ELEMENTS <TYPE>
    <n1> <n2> <n3> ...
    ...

    *BOUNDARY_GROUP <name>
    <n_segments>
    <n1> <n2> [bc_type] [nx] [ny]
    ...
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from unipinn.geometry.types import MeshData


def load_mesh(filepath: str) -> MeshData:
    """Load mesh from a custom .mesh file.

    Returns:
        MeshData instance with nodes, elements, elem_type, elem_areas, boundary_groups.
    """
    with open(filepath, "r") as f:
        content = f.read()

    nodes: List[List[float]] = []
    elements: List[List[int]] = []
    elem_type = ""
    boundary_groups: Dict[str, dict] = {}

    lines = content.strip().split("\n")
    section = None
    _current_bg = None
    _seg_count = 0
    _seg_idx = 0

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.upper().startswith("*NODES"):
            section = "nodes"
            continue
        if line.upper().startswith("*ELEMENTS"):
            section = "elements"
            parts = line.split()
            if len(parts) >= 2:
                elem_type = parts[1].upper()
            continue
        if line.upper().startswith("*BOUNDARY_GROUP"):
            section = "boundary_group"
            parts = line.split()
            _current_bg = parts[1] if len(parts) > 1 else "default"
            boundary_groups[_current_bg] = {"segments": [], "edges": []}
            _seg_count = 0
            _seg_idx = 0
            continue

        if section == "nodes":
            vals = line.split()
            if len(vals) >= 2:
                nodes.append([float(vals[0]), float(vals[1])])
        elif section == "elements":
            vals = line.split()
            if len(vals) >= 2 and all(v.lstrip("-").isdigit() for v in vals):
                elements.append([int(v) for v in vals])
        elif section == "boundary_group":
            parts = line.split()
            if _seg_idx == 0 and len(parts) == 1 and parts[0].isdigit():
                _seg_count = int(parts[0])
                _seg_idx += 1
                continue
            if len(parts) >= 2 and parts[0].lstrip("-").isdigit() and parts[1].lstrip("-").isdigit():
                n1, n2 = int(parts[0]), int(parts[1])
                bc_type = parts[2] if len(parts) > 2 else None
                nx = float(parts[3]) if len(parts) > 3 else None
                ny = float(parts[4]) if len(parts) > 4 else None
                boundary_groups[_current_bg]["edges"].append((n1, n2))
                boundary_groups[_current_bg]["segments"].append({
                    "n1": n1, "n2": n2, "bc_type": bc_type, "nx": nx, "ny": ny,
                })
                _seg_idx += 1
                if _seg_idx > _seg_count:
                    section = None

    nodes_arr = np.array(nodes, dtype=np.float64)
    elems_arr = np.array(elements, dtype=np.int64)

    # Precompute T3 areas
    elem_areas = None
    if elem_type == "T3" and len(elems_arr) > 0:
        n0 = nodes_arr[elems_arr[:, 0]]
        n1 = nodes_arr[elems_arr[:, 1]]
        n2 = nodes_arr[elems_arr[:, 2]]
        elem_areas = 0.5 * np.abs(
            (n1[:, 0] - n0[:, 0]) * (n2[:, 1] - n0[:, 1])
            - (n2[:, 0] - n0[:, 0]) * (n1[:, 1] - n0[:, 1])
        )

    return MeshData(
        nodes=nodes_arr,
        elements=elems_arr,
        elem_type=elem_type,
        elem_areas=elem_areas,
        boundary_groups=boundary_groups,
    )


def load_collocation_points(filepath: str) -> np.ndarray:
    """Load collocation points from a TXT file (columns 0,1 = X, Y).

    Returns:
        Array of shape (N, 2).
    """
    data = np.loadtxt(filepath, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :2]


def load_evaluation_points(filepath: str) -> np.ndarray:
    """Load evaluation points from a TXT file. Returns (N, 2)."""
    data = np.loadtxt(filepath, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :2]


def load_boundary_sample(filepath: str) -> Dict[str, dict]:
    """Load grouped boundary sample data from a TXT file.

    File format per group:
        BOUNDARY_GROUP <name>
        <n_points>
        <x1> <y1> <value1> [nx1] [ny1]
        ...

    Returns:
        Dict mapping group name to:
            {"points": (N,2), "values": (N,), "normals": (N,2) or None}
    """
    with open(filepath, "r") as f:
        content = f.read()

    result: Dict[str, dict] = {}
    lines = content.strip().split("\n")
    section = None
    _name = ""
    _count = 0
    _idx = 0
    _pts: List[List[float]] = []
    _vals: List[float] = []
    _normals: List[List[float]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        if line.upper().startswith("BOUNDARY_GROUP"):
            if section == "boundary" and _pts:
                result[_name] = {
                    "points": np.array(_pts, dtype=np.float64),
                    "values": np.array(_vals, dtype=np.float64),
                    "normals": np.array(_normals, dtype=np.float64) if _normals else None,
                }
            section = "boundary"
            _name = line.split()[1] if len(line.split()) > 1 else "default"
            _pts, _vals, _normals = [], [], []
            _count = 0
            _idx = 0
            continue

        if section == "boundary":
            if _idx == 0 and line.isdigit():
                _count = int(line)
                _idx += 1
                continue
            parts = line.split()
            if len(parts) >= 3:
                _pts.append([float(parts[0]), float(parts[1])])
                _vals.append(float(parts[2]))
                if len(parts) >= 5:
                    _normals.append([float(parts[3]), float(parts[4])])
                _idx += 1
                if _idx > _count:
                    section = None

    if section == "boundary" and _pts:
        result[_name] = {
            "points": np.array(_pts, dtype=np.float64),
            "values": np.array(_vals, dtype=np.float64),
            "normals": np.array(_normals, dtype=np.float64) if _normals else None,
        }

    return result


def load_data_sample(filepath: str) -> Dict[str, Optional[np.ndarray]]:
    """Load data sample for data-driven loss.

    Returns:
        {"points": (N, 2), "values": (N,) or None}
    """
    data = np.loadtxt(filepath, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        "points": data[:, :2],
        "values": data[:, 2] if data.shape[1] > 2 else None,
    }
