"""Tests for error metrics."""

import numpy as np
from unipinn.metrics.errors import rmse, l_inf, relative_l2, mape, pearson_correlation


def test_rmse_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert rmse(a, a) == 0.0


def test_rmse_known():
    a = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 2.1, 3.1])
    assert abs(rmse(a, p) - 0.1) < 1e-10


def test_l_inf():
    a = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.5, 3.0])
    assert l_inf(a, p) == 0.5


def test_relative_l2():
    a = np.ones(100)
    p = np.ones(100) * 1.1
    assert relative_l2(a, p) < 0.2


def test_pearson_perfect():
    a = np.linspace(0, 1, 50)
    assert abs(pearson_correlation(a, a) - 1.0) < 1e-10


def test_shape_mismatch_raises():
    import pytest
    a = np.array([1.0, 2.0])
    p = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        rmse(a, p)
