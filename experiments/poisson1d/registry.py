"""Predefined experiment configurations for the 1D Poisson PINN benchmark suite.

This module defines the 14 configurations (cfg_0 .. cfg_13) used in the paper.
Each entry is a ``Poisson1DVanillaPINNConfig`` instance; configs with multiple
random seeds store them as a list in ``seed``.

The ``REGISTRY`` dict maps short names (e.g. ``"cfg_0"``) to config objects,
and is consumed by ``batch.py`` for batched / parallel execution.

Groups
------
SamplingTest   (cfg_0 – cfg_3)  : uniform vs LHS × Adam vs LBFGS
RandomSeedTest (cfg_4 – cfg_5)  : 10 random seeds × Adam vs LBFGS
SampleNumTest  (cfg_6 – cfg_13) : n_colloc ∈ {101, 201, 401} × Adam vs LBFGS,
                                   plus extended LBFGS epoch counts
"""

import numpy as np
from typing import Dict
from unipinn.config.poisson1d import Poisson1DVanillaPINNConfig

# ──────────────────────────────────────────────────────────────
# Shared random-seed list (reproducible; generated once from seed 42)
# ──────────────────────────────────────────────────────────────
_RANDOM_SEEDS_10: list = np.random.default_rng(42).integers(1, 1_000_000, size=(10,)).tolist()
_FIXED_SEED: int = 697_368

# ──────────────────────────────────────────────────────────────
# Group 1 – SamplingTest (cfg_0 – cfg_3)
# Adam vs LBFGS × uniform vs LHS sampling
# ──────────────────────────────────────────────────────────────

cfg_0 = Poisson1DVanillaPINNConfig(
    exp_group=["SamplingTest", "OptimizerTest"],
    optimizer="adam", epochs=15000, precision="float64",
    pde_weight=1.0, bc_weight=50.0, lr=5e-3,
    scheduler_type="cosine", scheduler_step_times=20,
)

cfg_1 = Poisson1DVanillaPINNConfig(
    exp_group=["SamplingTest", "OptimizerTest"],
    optimizer="adam", epochs=15000, precision="float64",
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
    pde_weight=1.0, bc_weight=50.0, lr=5e-3,
    scheduler_type="cosine", scheduler_step_times=20,
)

cfg_2 = Poisson1DVanillaPINNConfig(
    exp_group=["SamplingTest", "OptimizerTest"],
    optimizer="lbfgs", epochs=4000, precision="float64",
    pde_weight=1.0, bc_weight=50.0, lr=1e-2,
    scheduler_type="cosine",
)

cfg_3 = Poisson1DVanillaPINNConfig(
    exp_group=["SamplingTest", "OptimizerTest"],
    optimizer="lbfgs", epochs=4000, precision="float64",
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
    pde_weight=1.0, bc_weight=50.0, lr=1e-2,
    scheduler_type="cosine",
)

# ──────────────────────────────────────────────────────────────
# Group 2 – RandomSeedTest (cfg_4 – cfg_5)
# 10 random seeds × Adam vs LBFGS (both with LHS)
# ──────────────────────────────────────────────────────────────

cfg_4 = Poisson1DVanillaPINNConfig(
    exp_group=["RandomSeedTest"],
    seed=list(_RANDOM_SEEDS_10),
    optimizer="adam", epochs=15000, precision="float64",
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
    pde_weight=1.0, bc_weight=50.0, lr=5e-3,
    scheduler_type="cosine", scheduler_step_times=20,
)

cfg_5 = Poisson1DVanillaPINNConfig(
    exp_group=["RandomSeedTest"],
    seed=list(_RANDOM_SEEDS_10),
    optimizer="lbfgs", epochs=4000, precision="float64",
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
    pde_weight=1.0, bc_weight=50.0, lr=1e-2,
    scheduler_type="cosine",
)

# ──────────────────────────────────────────────────────────────
# Group 3 – SampleNumTest (cfg_6 – cfg_13)
# n_colloc ∈ {101, 201, 401} × Adam vs LBFGS,
# plus extended LBFGS epochs (5k / 8k / 10k)
# ──────────────────────────────────────────────────────────────

cfg_6 = Poisson1DVanillaPINNConfig(
    exp_group=["SampleNumTest"],
    seed=_FIXED_SEED, n_colloc=101,
    optimizer="adam", epochs=20000, precision="float64",
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
    pde_weight=1.0, bc_weight=50.0, lr=5e-3,
    scheduler_type="cosine", scheduler_step_times=25,
)

cfg_7 = Poisson1DVanillaPINNConfig(
    exp_group=["SampleNumTest"],
    seed=_FIXED_SEED, n_colloc=201,
    optimizer="adam", epochs=20000, precision="float64",
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
    pde_weight=1.0, bc_weight=50.0, lr=5e-3,
    scheduler_type="cosine", scheduler_step_times=25,
)

cfg_8 = Poisson1DVanillaPINNConfig(
    exp_group=["SampleNumTest"],
    seed=_FIXED_SEED, n_colloc=401,
    optimizer="adam", epochs=20000, precision="float64",
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
    pde_weight=1.0, bc_weight=50.0, lr=5e-3,
    scheduler_type="cosine", scheduler_step_times=25,
)

cfg_9 = Poisson1DVanillaPINNConfig(
    exp_group=["SampleNumTest", "BFGSTest1"],
    seed=_FIXED_SEED, n_colloc=101,
    optimizer="lbfgs", epochs=5000, precision="float64",
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
    pde_weight=1.0, bc_weight=50.0, lr=1e-2,
    scheduler_type="cosine",
)

cfg_10 = Poisson1DVanillaPINNConfig(
    exp_group=["SampleNumTest", "BFGSTest1"],
    seed=_FIXED_SEED, n_colloc=201,
    optimizer="lbfgs", epochs=5000, precision="float64",
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
    pde_weight=1.0, bc_weight=50.0, lr=1e-2,
    scheduler_type="cosine",
)

cfg_11 = Poisson1DVanillaPINNConfig(
    exp_group=["SampleNumTest", "BFGSTest1"],
    seed=_FIXED_SEED, n_colloc=401,
    optimizer="lbfgs", epochs=5000, precision="float64",
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
    pde_weight=1.0, bc_weight=50.0, lr=1e-2,
    scheduler_type="cosine",
)

cfg_12 = Poisson1DVanillaPINNConfig(
    exp_group=["SampleNumTest", "BFGSTest1"],
    seed=_FIXED_SEED, n_colloc=201,
    optimizer="lbfgs", epochs=8000, precision="float64",
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
    pde_weight=1.0, bc_weight=50.0, lr=1e-2,
    scheduler_type="cosine",
)

cfg_13 = Poisson1DVanillaPINNConfig(
    exp_group=["SampleNumTest", "BFGSTest1"],
    seed=_FIXED_SEED, n_colloc=401,
    optimizer="lbfgs", epochs=10000, precision="float64",
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
    pde_weight=1.0, bc_weight=50.0, lr=1e-2,
    scheduler_type="cosine",
)

# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

REGISTRY: Dict[str, Poisson1DVanillaPINNConfig] = {
    f"cfg_{i}": cfg for i, cfg in enumerate([
        cfg_0, cfg_1, cfg_2, cfg_3,
        cfg_4, cfg_5,
        cfg_6, cfg_7, cfg_8, cfg_9, cfg_10, cfg_11, cfg_12, cfg_13,
    ])
}
"""Maps short name (e.g. ``"cfg_0"``) → ``Poisson1DVanillaPINNConfig``."""


def list_configs() -> list:
    """Return sorted list of registry keys."""
    return sorted(REGISTRY.keys(), key=lambda k: int(k.split("_")[1]))
