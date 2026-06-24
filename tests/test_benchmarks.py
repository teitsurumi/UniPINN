"""Tests for benchmark data generation."""

import numpy as np
from unipinn.pde.benchmarks.poisson1d import Poisson1DBenchmarkIndex
from unipinn.pde.benchmarks.poisson2d import Poisson2DBenchmarkIndex


def test_poisson1d_registry():
    names = Poisson1DBenchmarkIndex.list()
    assert "steep_solution" in names
    assert "discontinuous_jump" in names


def test_poisson1d_generate():
    bench = Poisson1DBenchmarkIndex.get("steep_solution")
    data = bench.generate(n_colloc=50, n_eval=100, seed=42)
    assert data["x_colloc"].shape == (50,)
    assert data["x_eval"].shape == (100,)
    assert data["u_eval"].shape == (100,)
    assert data["f_colloc"].shape == (50,)
    assert data["x_bc"].shape == (2,)


def test_poisson1d_sampling_methods():
    bench = Poisson1DBenchmarkIndex.get("steep_solution")
    for method in ["uniform", "random", "chebyshev", "lhs"]:
        data = bench.generate(n_colloc=30, sampling_method=method, seed=42)
        assert len(data["x_colloc"]) == 30


def test_poisson2d_registry():
    names = Poisson2DBenchmarkIndex.list()
    assert "harmonic_rational" in names
    assert "steep_product_2d" in names


def test_poisson2d_generate():
    bench = Poisson2DBenchmarkIndex.get("harmonic_rational")
    data = bench.generate(n_colloc=100, n_eval=10, n_bc_per_edge=10, seed=42)
    assert data["x_colloc"].shape[1] == 2
    assert data["x_eval"].shape[1] == 2
    assert data["x_bc"].shape[1] == 2
