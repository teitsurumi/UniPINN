"""Benchmark problems with exact solutions."""

from unipinn.pde.benchmarks.poisson1d import Poisson1DBenchmarkIndex, BasePoisson1DBenchmark
from unipinn.pde.benchmarks.poisson2d import (
    Poisson2DBenchmarkIndex, BasePoisson2DBenchmark,
    IrregularDomainBenchmark, MeshBasedBenchmark,
)
