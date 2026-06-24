"""Base benchmark interface for PDE problems with known solutions."""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from unipinn.metrics.errors import (
    mape, smape, rmse, l_inf, relative_l2, relative_l_inf,
    gradient_l2_error, spectral_relative_error, relative_energy_error,
    pearson_correlation, robust_median_absolute_error,
)


class BaseBenchmark(ABC):
    """Abstract base class for PDE benchmarks with exact solutions.

    Subclasses must implement ``generate()`` to produce training/evaluation data.
    """

    @abstractmethod
    def generate(self, **kwargs) -> Dict[str, np.ndarray]:
        """Generate training and evaluation data.

        Returns a dictionary containing at minimum:
            - case_name, domain, bc_type, description
            - x_colloc, f_colloc (collocation points and source term)
            - x_bc, u_bc or du_bc (boundary data)
            - x_eval, u_eval, f_eval (evaluation grid and exact values)
        """
        ...

    @staticmethod
    def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Compute the full suite of error metrics."""
        return {
            "mape": mape(actual, predicted),
            "smape": smape(actual, predicted),
            "rmse": rmse(actual, predicted),
            "l_inf": l_inf(actual, predicted),
            "relative_l2": relative_l2(actual, predicted),
            "relative_l_inf": relative_l_inf(actual, predicted),
            "gradient_l2_error": gradient_l2_error(actual, predicted),
            "spectral_relative_error": spectral_relative_error(actual, predicted),
            "relative_energy_error": relative_energy_error(actual, predicted),
            "pearson_correlation": pearson_correlation(actual, predicted),
            "robust_median_absolute_error": robust_median_absolute_error(actual, predicted),
        }
