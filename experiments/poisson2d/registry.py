"""Predefined experiment configurations for the 2D Poisson PINN benchmark suite.

All configs target the ``steep_product_2d`` benchmark.

Groups
------
UnsupervisedColloc  (cfg_0 – cfg_4) : random vs uniform colloc × loss-weight tuning
SupervisedRecon     (cfg_5 – cfg_7) : sparse data count ∈ {50, 200, 500}
"""

import numpy as np
from typing import Dict
from unipinn.config.poisson2d import Poisson2DVanillaPINNConfig

_FIXED_SEED: int = 697_368

# Shared architecture — 2D input, wider/deeper for the multi-scale 2D problem
_ARCH = {
    1: {"n": 2},
    2: {"n": 48, "a": "dynsine", "p": 1},
    3: {"n": 48, "a": "dynsine", "p": 1},
    4: {"n": 48, "a": "dynsine", "p": 1},
    5: {"n": 48, "a": "dynsine", "p": 1},
    6: {"n": 1},
}

_ARCH_LARGE = {
    1: {"n": 2},
    2: {"n": 64, "a": "dynsine", "p": 1},
    3: {"n": 64, "a": "dynsine", "p": 1},
    4: {"n": 64, "a": "dynsine", "p": 1},
    5: {"n": 64, "a": "dynsine", "p": 1},
    6: {"n": 64, "a": "dynsine", "p": 1},
    7: {"n": 1},
}

# ──────────────────────────────────────────────────────────────
# Group 1 – UnsupervisedColloc (cfg_0 – cfg_4)
# Random vs uniform collocation × Adam/LBFGS × loss-weight tuning
# ──────────────────────────────────────────────────────────────

cfg_0 = Poisson2DVanillaPINNConfig(
    exp_group=["UnsupervisedColloc"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="random",
    n_colloc=4000, n_bc_per_edge=60,
    arch_config=_ARCH,
    optimizer="adam", epochs=15000, precision="float64",
    pde_weight=1.0, bc_weight=50.0, lr=3e-3,
    scheduler_type="cosine", scheduler_step_times=20,
)

cfg_1 = Poisson2DVanillaPINNConfig(
    exp_group=["UnsupervisedColloc"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="uniform",
    n_colloc=4096, n_bc_per_edge=60,      # 64*64 grid
    arch_config=_ARCH,
    optimizer="adam", epochs=15000, precision="float64",
    pde_weight=1.0, bc_weight=50.0, lr=3e-3,
    scheduler_type="cosine", scheduler_step_times=20,
)

cfg_2 = Poisson2DVanillaPINNConfig(
    exp_group=["UnsupervisedColloc", "LossWeightTest"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="random",
    n_colloc=4000, n_bc_per_edge=60,
    arch_config=_ARCH,
    optimizer="adam", epochs=15000, precision="float64",
    pde_weight=1.0, bc_weight=200.0, lr=3e-3,     # heavier BC penalty
    scheduler_type="cosine", scheduler_step_times=20,
)

cfg_3 = Poisson2DVanillaPINNConfig(
    exp_group=["UnsupervisedColloc", "LossWeightTest"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="random",
    n_colloc=4000, n_bc_per_edge=60,
    arch_config=_ARCH,
    optimizer="adam", epochs=15000, precision="float64",
    pde_weight=10.0, bc_weight=50.0, lr=3e-3,     # heavier PDE penalty
    scheduler_type="cosine", scheduler_step_times=20,
)

cfg_4 = Poisson2DVanillaPINNConfig(
    exp_group=["UnsupervisedColloc"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="random",
    n_colloc=4000, n_bc_per_edge=60,
    arch_config=_ARCH,
    optimizer="lbfgs", epochs=5000, precision="float64",
    pde_weight=1.0, bc_weight=50.0, lr=1e-2,
    scheduler_type="cosine",
)

# ──────────────────────────────────────────────────────────────
# Group 2 – SupervisedRecon (cfg_5 – cfg_7)
# Sparse interior data ∈ {50, 200, 500} + collocation + BC
# ──────────────────────────────────────────────────────────────

cfg_5 = Poisson2DVanillaPINNConfig(
    exp_group=["SupervisedRecon"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="random",
    n_colloc=4000, n_bc_per_edge=60,
    n_data=50, data_weight=1.0,
    arch_config=_ARCH,
    optimizer="adam", epochs=15000, precision="float64",
    pde_weight=1.0, bc_weight=50.0, lr=3e-3,
    scheduler_type="cosine", scheduler_step_times=20,
)

cfg_6 = Poisson2DVanillaPINNConfig(
    exp_group=["SupervisedRecon"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="random",
    n_colloc=4000, n_bc_per_edge=60,
    n_data=200, data_weight=1.0,
    arch_config=_ARCH,
    optimizer="adam", epochs=15000, precision="float64",
    pde_weight=1.0, bc_weight=50.0, lr=3e-3,
    scheduler_type="cosine", scheduler_step_times=20,
)

cfg_7 = Poisson2DVanillaPINNConfig(
    exp_group=["SupervisedRecon"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="random",
    n_colloc=4000, n_bc_per_edge=60,
    n_data=500, data_weight=1.0,
    arch_config=_ARCH,
    optimizer="adam", epochs=15000, precision="float64",
    pde_weight=1.0, bc_weight=50.0, lr=3e-3,
    scheduler_type="cosine", scheduler_step_times=20,
)


# ──────────────────────────────────────────────────────────────
# Review
# ──────────────────────────────────────────────────────────────

cfg_8 = Poisson2DVanillaPINNConfig(
    exp_group=["UnsupervisedColloc", "LRSchedulerTest"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="random",
    n_colloc=4000, n_bc_per_edge=60,
    arch_config=_ARCH,
    optimizer="adam", epochs=15000, precision="float64",
    pde_weight=1.0, bc_weight=50.0, lr=6e-3,
    warmup_epochs=2000,
    scheduler_type="steplr", scheduler_step_times=15,
    scheduler_gamma=0.56,
)

cfg_9 = Poisson2DVanillaPINNConfig(
    exp_group=["UnsupervisedColloc", "LRSchedulerTest"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="random",
    n_colloc=4000, n_bc_per_edge=60,
    arch_config=_ARCH,
    optimizer="adam", epochs=20000, precision="float64",
    pde_weight=1.0, bc_weight=50.0, lr=6e-3,
    warmup_epochs=4000,
    scheduler_type="steplr", scheduler_step_times=45,
    scheduler_gamma=0.86,
)

cfg_10 = Poisson2DVanillaPINNConfig(
    exp_group=["UnsupervisedColloc", "LRSchedulerTest"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="uniform",
    n_colloc=3600, n_bc_per_edge=60,
    arch_config=_ARCH,
    optimizer="adam", epochs=20000, precision="float64",
    pde_weight=1.0, bc_weight=64.0, lr=1e-2,
    warmup_epochs=5000,
    scheduler_type="steplr", scheduler_step_times=64,
    scheduler_gamma=0.88,
)

cfg_11 = Poisson2DVanillaPINNConfig(
    exp_group=["UnsupervisedColloc", "LRSchedulerTest"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="lhs",
    n_colloc=6400, n_bc_per_edge=100,
    arch_config=_ARCH_LARGE,
    optimizer="adam", epochs=30000, precision="float64",
    pde_weight=1.0, bc_weight=64.0, lr=5e-3,
    warmup_epochs=8000,
    scheduler_type="steplr", scheduler_step_times=40,
    scheduler_gamma=0.9,
)

cfg_12 = Poisson2DVanillaPINNConfig(
    exp_group=["UnsupervisedColloc", "LRSchedulerTest"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="uniform",   # keep cfg_10's proven sampling
    n_colloc=4900, n_bc_per_edge=70,       # 70×70 grid, slightly more BC
    arch_config={
        1: {"n": 2},
        2: {"n": 64, "a": "dynsine", "p": 1},
        3: {"n": 64, "a": "dynsine", "p": 1},
        4: {"n": 64, "a": "dynsine", "p": 1},
        5: {"n": 64, "a": "dynsine", "p": 1},
        6: {"n": 1},
    },  # 5×64: more capacity
    optimizer="adam", epochs=32000, precision="float64",
    pde_weight=1.0, bc_weight=64.0, lr=1e-2,   # keep cfg_10's lr
    warmup_epochs=5000,
    scheduler_type="steplr", scheduler_step_times=64,
    scheduler_gamma=0.9,                   # gentler decay
)

cfg_13 = Poisson2DVanillaPINNConfig(
    exp_group=["UnsupervisedColloc", "LRSchedulerTest"],
    seed=_FIXED_SEED,
    benchmark_name="steep_product_2d",
    benchmark_sampling_method="uniform",   # keep cfg_10's proven sampling
    n_colloc=5200, n_bc_per_edge=70,       # 70×70 grid, slightly more BC
    arch_config={
        1: {"n": 2},
        2: {"n": 64, "a": "dynsine", "p": 1},
        3: {"n": 64, "a": "dynsine", "p": 1},
        4: {"n": 64, "a": "dynsine", "p": 1},
        5: {"n": 64, "a": "dynsine", "p": 1},
        6: {"n": 1},
    },  # 5×64: more capacity
    optimizer="adam", epochs=36000, precision="float64",
    pde_weight=1.0, bc_weight=64.0, lr=1e-2,   # keep cfg_10's lr
    warmup_epochs=5000,
    scheduler_type="steplr", scheduler_step_times=60,
    scheduler_gamma=0.9,                   # gentler decay
)

# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

REGISTRY: Dict[str, Poisson2DVanillaPINNConfig] = {
    f"cfg_{i}": cfg for i, cfg in enumerate([
        cfg_0, cfg_1, cfg_2, cfg_3, cfg_4,
        cfg_5, cfg_6, cfg_7,
        cfg_8, cfg_9, cfg_10,
        cfg_11, cfg_12, cfg_13
    ])
}


def list_configs() -> list:
    """Return sorted list of registry keys."""
    return sorted(REGISTRY.keys(), key=lambda k: int(k.split("_")[1]))
