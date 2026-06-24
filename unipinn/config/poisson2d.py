"""Configuration for 2D Poisson vanilla PINN experiments.

Defines ``Poisson2DVanillaPINNConfig`` — the single source of truth for
all 2D Poisson PINN experiments.  Supports both unsupervised (PDE + BC only)
and supervised (PDE + BC + sparse data) modes via the ``n_data`` field.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from unipinn.config.base import BaseExperimentConfig


@dataclass
class Poisson2DVanillaPINNConfig(BaseExperimentConfig):
    """Full configuration for 2D Poisson vanilla PINN training.

    Subclasses ``BaseExperimentConfig`` and adds problem-specific fields.
    """
    # Override defaults from base
    exp_name: str = "poisson2d_vanilla"
    epochs: int = 8000

    # Data and benchmarks
    benchmark_name: str = "steep_product_2d"
    benchmark_params: Dict[str, float] = field(default_factory=dict)
    benchmark_sampling_method: str = "lhs"
    benchmark_lhs: Optional[str] = "cm"
    n_colloc: int = 2500
    n_eval: int = 100               # per side -> n_eval^2 total eval points
    n_bc_per_edge: int = 50         # 4 edges * n_bc_per_edge total BC points

    # Supervised mode (n_data=0 -> unsupervised)
    n_data: int = 0
    data_weight: float = 1.0

    # Model architecture (n=2 for 2D input)
    arch_config: Dict[int, Dict[str, Union[int, str]]] = field(default_factory=lambda: {
        1: {"n": 2},
        2: {"n": 32, "a": "dynsine", "p": 1},
        3: {"n": 32, "a": "dynsine", "p": 1},
        4: {"n": 32, "a": "dynsine", "p": 1},
        5: {"n": 1},
    })
    weight_init: str = "default"

    # Loss weights
    pde_weight: float = 1.0
    bc_weight: float = 10.0

    # Optimizer
    optimizer: str = "adam"
    lr: float = 5e-3
    weight_decay: float = 0.0
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    lbfgs_max_iter: int = 5
    lbfgs_history_size: int = 100
    lbfgs_line_search_fn: str = "strong_wolfe"

    # Scheduler
    warmup_epochs: int = 1000
    scheduler_type: str = "steplr"
    scheduler_step_times: int = 12
    scheduler_step_size: Optional[int] = None
    scheduler_gamma: float = 0.8
    scheduler_T_max: Optional[int] = None
    scheduler_factor: float = 0.5

    # Training loop extras
    grad_clip_max_norm: Optional[float] = None
    early_stop_patience: Optional[int] = None

    def __post_init__(self):
        if self.scheduler_type == "steplr" and self.scheduler_step_size is None:
            self.scheduler_step_size = max(1, (self.epochs - self.warmup_epochs) // self.scheduler_step_times)
        if self.scheduler_type == "cosine" and self.scheduler_T_max is None:
            self.scheduler_T_max = self.epochs - self.warmup_epochs
