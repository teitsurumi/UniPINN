"""Element quality metrics for finite element meshes.

All functions accept a Jacobian array of shape (num_elements, num_quad, 2, 2)
and return per-element quality measures of shape (num_elements,).
"""

import numpy as np

EPS = 1e-12


def _det(jacobians: np.ndarray) -> np.ndarray:
    """Compute determinant of 2x2 Jacobian matrices."""
    return (jacobians[..., 0, 0] * jacobians[..., 1, 1]
            - jacobians[..., 0, 1] * jacobians[..., 1, 0])


def quality(jdet: np.ndarray):
    """Print a summary quality assessment from Jacobian determinants.

    Args:
        jdet: 1D array of Jacobian determinant values (e.g., per quadrature point).
    """
    J_min = np.min(jdet)
    J_max = np.max(jdet)
    J_mean = np.mean(jdet)
    J_ratio = J_min / J_max
    J_cv = np.std(jdet) / J_mean

    print(f"Min: {J_min:.4f} | Max: {J_max:.4f} | Mean: {J_mean:.4f} | "
          f"Min/Max ratio: {J_ratio:.4f} | CV: {J_cv:.4f} | ", end="")

    if J_min <= 0:
        print("INVALID: non-positive Jacobian determinant detected!")
    elif J_ratio > 0.5:
        print("Good quality")
    elif J_ratio > 0.3:
        print("Acceptable quality")
    else:
        print("Poor quality, consider mesh refinement")


def shape_quality(jacobians: np.ndarray) -> np.ndarray:
    """Shape quality: n / kappa(T), where kappa = |T| * |T^{-1}| (Frobenius).

    Range [0, 1]; 1 = ideal shape, 0 = degenerate.
    """
    n = 2
    det = _det(jacobians)
    norm = np.sqrt(np.sum(jacobians ** 2, axis=(-2, -1)))
    inv_norm = np.sqrt(
        (jacobians[..., 1, 1] ** 2 + jacobians[..., 0, 1] ** 2
         + jacobians[..., 1, 0] ** 2 + jacobians[..., 0, 0] ** 2) / (det ** 2 + EPS)
    )
    cond = norm * inv_norm
    return np.mean(n / (cond + EPS), axis=1)


def volume_quality(jacobians: np.ndarray) -> np.ndarray:
    """Volume quality: min(det, 1/det).

    Range [0, 1]; 1 = reference volume, 0 = degenerate.
    """
    det_abs = np.abs(_det(jacobians)) + EPS
    return np.mean(np.minimum(det_abs, 1.0 / det_abs), axis=1)


def volume_shape_quality(jacobians: np.ndarray) -> np.ndarray:
    """Combined volume-shape quality: min(det, 1/det) * (2/kappa).

    Range [0, 1].
    """
    n = 2
    det_abs = np.abs(_det(jacobians)) + EPS
    vol_part = np.minimum(det_abs, 1.0 / det_abs)
    norm = np.sqrt(np.sum(jacobians ** 2, axis=(-2, -1)))
    inv_norm = np.sqrt(
        (jacobians[..., 1, 1] ** 2 + jacobians[..., 0, 1] ** 2
         + jacobians[..., 1, 0] ** 2 + jacobians[..., 0, 0] ** 2) / (_det(jacobians) ** 2 + EPS)
    )
    cond = norm * inv_norm
    shape_part = n / (cond + EPS)
    return np.mean(vol_part * shape_part, axis=1)


def condition_number(jacobians: np.ndarray) -> np.ndarray:
    """Condition number kappa(T) = |T| * |T^{-1}|.

    Range [2, inf); lower is better.
    """
    det = _det(jacobians)
    norm = np.sqrt(np.sum(jacobians ** 2, axis=(-2, -1)))
    inv_norm = np.sqrt(
        (jacobians[..., 1, 1] ** 2 + jacobians[..., 0, 1] ** 2
         + jacobians[..., 1, 0] ** 2 + jacobians[..., 0, 0] ** 2) / (det ** 2 + EPS)
    )
    return np.mean(norm * inv_norm, axis=1)


def mean_ratio(jacobians: np.ndarray) -> np.ndarray:
    """Mean ratio mu(T) = |T|^2 / det(T).

    Equivalent to condition number in 2D.
    """
    det = _det(jacobians)
    norm2 = np.sum(jacobians ** 2, axis=(-2, -1))
    return np.mean(norm2 / (np.abs(det) + EPS), axis=1)


def frobenius_norm(jacobians: np.ndarray) -> np.ndarray:
    """Frobenius norm |T| averaged over quadrature points.

    Reflects mean element scale.
    """
    return np.mean(np.sqrt(np.sum(jacobians ** 2, axis=(-2, -1))), axis=1)


def determinant(jacobians: np.ndarray) -> np.ndarray:
    """Mean determinant (sign preserved). Positive = correct orientation."""
    return np.mean(_det(jacobians), axis=1)


def trace(jacobians: np.ndarray) -> np.ndarray:
    """Mean trace of the Jacobian."""
    return np.mean(jacobians[..., 0, 0] + jacobians[..., 1, 1], axis=1)


def oddy_metric(jacobians: np.ndarray) -> np.ndarray:
    """Oddy distortion metric (2D).

    Q = (|T^t T|^2 - 0.5 |T|^4) / det^2. Lower is better; ideal = 0.
    """
    det = _det(jacobians)
    TtT = np.einsum("...ji,...jk->...ik", jacobians, jacobians)
    TtT_sq = np.einsum("...ij,...jk->...ik", TtT, TtT)
    norm4_TtT = np.trace(TtT_sq, axis1=-2, axis2=-1)
    norm2_T = np.sum(jacobians ** 2, axis=(-2, -1))
    norm4_T = norm2_T ** 2
    oddy = (norm4_TtT - 0.5 * norm4_T) / (det ** 2 + EPS)
    return np.mean(np.maximum(oddy, 0.0), axis=1)


def winslow_metric(jacobians: np.ndarray) -> np.ndarray:
    """Winslow metric: tau * |T^{-1}|^2.

    Commonly used in mesh smoothing.
    """
    det = _det(jacobians)
    inv_norm2 = (
        (jacobians[..., 1, 1] ** 2 + jacobians[..., 0, 1] ** 2
         + jacobians[..., 1, 0] ** 2 + jacobians[..., 0, 0] ** 2) / (det ** 2 + EPS)
    )
    return np.mean(np.abs(det) * inv_norm2, axis=1)
