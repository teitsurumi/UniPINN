"""Tests for 2D Poisson benchmarks: harmonic_rational and steep_product_2d.

Verifies that both benchmarks produce correct data for:
  1. Unsupervised PINN solving  (collocation + boundary conditions)
  2. Supervised reconstruction   (sparse data + collocation + boundary conditions)
"""

import numpy as np
import pytest
from unipinn.pde.benchmarks.poisson2d import Poisson2DBenchmarkIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["harmonic_rational", "steep_product_2d"])
def bench(request):
    return Poisson2DBenchmarkIndex.get(request.param)


@pytest.fixture(params=["harmonic_rational", "steep_product_2d"])
def bench_name(request):
    return request.param


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_new_benchmarks():
    names = Poisson2DBenchmarkIndex.list()
    assert "harmonic_rational" in names
    assert "steep_product_2d" in names


# ---------------------------------------------------------------------------
# Unsupervised: collocation + boundary conditions
# ---------------------------------------------------------------------------

class TestUnsupervisedData:
    """Verify that generate() produces everything needed for unsupervised PINN."""

    def test_collocation_shape_and_dtype(self, bench):
        data = bench.generate(n_colloc=200, n_eval=10, n_bc_per_edge=10, seed=42)
        assert data["x_colloc"].shape == (200, 2)
        assert data["x_colloc"].dtype == np.float64

    def test_collocation_inside_domain(self, bench):
        data = bench.generate(n_colloc=500, n_eval=5, n_bc_per_edge=5, seed=42)
        domain = data["domain"]  # (2, 2)
        x, y = data["x_colloc"][:, 0], data["x_colloc"][:, 1]
        assert x.min() >= domain[0, 0] - 1e-5
        assert x.max() <= domain[0, 1] + 1e-5
        assert y.min() >= domain[1, 0] - 1e-5
        assert y.max() <= domain[1, 1] + 1e-5

    def test_source_term_shape(self, bench):
        data = bench.generate(n_colloc=100, n_eval=5, n_bc_per_edge=5, seed=42)
        assert data["f_colloc"].shape == (100,)
        assert np.all(np.isfinite(data["f_colloc"]))

    def test_boundary_shape(self, bench):
        data = bench.generate(n_colloc=50, n_eval=5, n_bc_per_edge=20, seed=42)
        # Dirichlet: 4 edges * 20 = 80 points
        assert data["x_bc"].shape == (80, 2)
        assert data["u_bc"].shape == (80,)
        assert data["du_bc"] is None

    def test_boundary_on_domain_edges(self, bench):
        data = bench.generate(n_colloc=50, n_eval=5, n_bc_per_edge=10, seed=42)
        domain = data["domain"]
        x_bc = data["x_bc"]
        x_min, x_max = domain[0]
        y_min, y_max = domain[1]
        # Every BC point should lie on at least one edge
        on_edge = (
            np.isclose(x_bc[:, 0], x_min, atol=1e-5) |
            np.isclose(x_bc[:, 0], x_max, atol=1e-5) |
            np.isclose(x_bc[:, 1], y_min, atol=1e-5) |
            np.isclose(x_bc[:, 1], y_max, atol=1e-5)
        )
        assert on_edge.all()

    def test_bc_values_consistent_with_u(self, bench):
        """u_bc should equal bench.u(x_bc)."""
        data = bench.generate(n_colloc=50, n_eval=5, n_bc_per_edge=10, seed=42)
        u_recomputed = bench.u(data["x_bc"])
        np.testing.assert_allclose(data["u_bc"], u_recomputed, atol=1e-12)

    def test_evaluation_grid(self, bench):
        data = bench.generate(n_colloc=50, n_eval=20, n_bc_per_edge=5, seed=42)
        n_side = int(np.sqrt(20))  # generate uses int(sqrt(n_eval)) for non-uniform
        expected = n_side * n_side
        assert data["x_eval"].shape == (expected, 2)
        assert data["u_eval"].shape == (expected,)
        assert data["f_eval"].shape == (expected,)

    def test_eval_values_consistent(self, bench):
        """u_eval and f_eval should match bench.u() and bench.f()."""
        data = bench.generate(n_colloc=50, n_eval=10, n_bc_per_edge=5, seed=42)
        np.testing.assert_allclose(data["u_eval"], bench.u(data["x_eval"]), atol=1e-12)
        np.testing.assert_allclose(data["f_eval"], bench.f(data["x_eval"]), atol=1e-12)


# ---------------------------------------------------------------------------
# Supervised: sparse interior data sampling
# ---------------------------------------------------------------------------

class TestSupervisedData:
    """Verify that the benchmark supports sparse interior sampling for
    supervised high-resolution reconstruction."""

    def test_sparse_interior_sampling(self, bench):
        """bench._sample_points_2d should produce interior points."""
        pts = bench._sample_points_2d(
            100, bench.domain, "random", seed=42, exclude_boundary=True,
        )
        assert pts.shape == (100, 2)
        domain = bench.domain
        x, y = pts[:, 0], pts[:, 1]
        assert x.min() > domain[0][0]
        assert x.max() < domain[0][1]
        assert y.min() > domain[1][0]
        assert y.max() < domain[1][1]

    def test_sparse_data_evaluation(self, bench):
        """bench.u() should be callable on arbitrary interior points."""
        pts = bench._sample_points_2d(
            50, bench.domain, "random", seed=99, exclude_boundary=True,
        )
        u_vals = bench.u(pts)
        assert u_vals.shape == (50,)
        assert np.all(np.isfinite(u_vals))

    def test_uniform_grid_sampling(self, bench):
        """Uniform sampling should produce a structured grid."""
        pts = bench._sample_points_2d(
            100, bench.domain, "uniform", seed=42, exclude_boundary=False,
        )
        assert pts.shape[0] <= 100
        assert pts.shape[1] == 2

    def test_lhs_sampling(self, bench):
        pts = bench._sample_points_2d(
            80, bench.domain, "lhs", seed=42,
            lhs_criterion="cm", exclude_boundary=True,
        )
        assert pts.shape == (80, 2)


# ---------------------------------------------------------------------------
# Sampling methods (all three)
# ---------------------------------------------------------------------------

class TestSamplingMethods:

    @pytest.mark.parametrize("method", ["uniform", "random", "lhs"])
    def test_all_sampling_methods(self, bench, method):
        data = bench.generate(
            n_colloc=60, n_eval=5, n_bc_per_edge=5,
            sampling_method=method, seed=42,
        )
        # uniform grid may produce floor(sqrt(n))^2 < n points
        assert data["x_colloc"].shape[0] <= 60
        assert data["x_colloc"].shape[0] >= 30
        assert data["x_colloc"].shape[1] == 2


# ---------------------------------------------------------------------------
# Analytical correctness: finite-difference checks
# ---------------------------------------------------------------------------

class TestAnalyticalCorrectness:
    """Verify u, grad(u), and f via finite differences."""

    def test_gradient_finite_difference(self, bench):
        eps = 1e-7
        pts = np.array([[0.3, 0.5], [-0.2, 0.7], [0.0, 0.0], [0.8, -0.3]])
        dudx_an, dudy_an = bench.u_grad(pts)
        for i, p in enumerate(pts):
            px = np.array([[p[0] + eps, p[1]], [p[0] - eps, p[1]]])
            py = np.array([[p[0], p[1] + eps], [p[0], p[1] - eps]])
            dudx_fd = (bench.u(px[:1]) - bench.u(px[1:])) / (2 * eps)
            dudy_fd = (bench.u(py[:1]) - bench.u(py[1:])) / (2 * eps)
            np.testing.assert_allclose(dudx_an[i], dudx_fd[0], atol=1e-4)
            np.testing.assert_allclose(dudy_an[i], dudy_fd[0], atol=1e-4)

    def test_source_finite_difference(self, bench):
        """f = -Laplacian(u) verified via central finite differences."""
        eps = 1e-6
        # Avoid x~0 where tanh(r1*x) has extreme curvature
        pts = np.array([[0.5, 0.3], [-0.7, 0.6], [0.8, -0.2]])
        f_an = bench.f(pts)
        for i, p in enumerate(pts):
            u0 = bench.u(np.array([[p[0], p[1]]]))[0]
            uxp = bench.u(np.array([[p[0] + eps, p[1]]]))[0]
            uxm = bench.u(np.array([[p[0] - eps, p[1]]]))[0]
            uyp = bench.u(np.array([[p[0], p[1] + eps]]))[0]
            uym = bench.u(np.array([[p[0], p[1] - eps]]))[0]
            laplacian = (uxp - 2 * u0 + uxm + uyp - 2 * u0 + uym) / eps ** 2
            f_fd = -laplacian
            np.testing.assert_allclose(f_an[i], f_fd, atol=0.5,
                                       err_msg=f"f mismatch at {p}")


# ---------------------------------------------------------------------------
# Benchmark-specific checks
# ---------------------------------------------------------------------------

def test_harmonic_source_is_zero():
    """harmonic_rational is harmonic -> f = 0 everywhere."""
    bench = Poisson2DBenchmarkIndex.get("harmonic_rational")
    data = bench.generate(n_colloc=200, n_eval=5, n_bc_per_edge=5, seed=42)
    np.testing.assert_allclose(data["f_colloc"], 0.0, atol=1e-14)
    np.testing.assert_allclose(data["f_eval"], 0.0, atol=1e-14)


def test_steep_product_boundary_zeros():
    """steep_product_2d: u = 0 on y = -1 and y = 1 (sin(4pi*y) = 0)."""
    bench = Poisson2DBenchmarkIndex.get("steep_product_2d")
    # Points on y-boundaries
    x_vals = np.linspace(-1, 1, 20)
    pts_bottom = np.column_stack([x_vals, np.full(20, -1.0)])
    pts_top = np.column_stack([x_vals, np.full(20, 1.0)])
    np.testing.assert_allclose(bench.u(pts_bottom), 0.0, atol=1e-12)
    np.testing.assert_allclose(bench.u(pts_top), 0.0, atol=1e-12)


def test_steep_product_custom_params():
    """steep_product_2d accepts custom wx, wy, r1."""
    bench = Poisson2DBenchmarkIndex.get("steep_product_2d", wx=2 * np.pi, wy=2 * np.pi, r1=5.0)
    pts = np.array([[0.5, 0.25]])
    u_val = bench.u(pts)
    expected = (0.1 * np.sin(2 * np.pi * 0.5) + np.tanh(5.0 * 0.5)) * np.sin(2 * np.pi * 0.25)
    np.testing.assert_allclose(u_val[0], expected, rtol=1e-12)
