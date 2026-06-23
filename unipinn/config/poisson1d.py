"""Configuration for 1D Poisson vanilla PINN experiments.

This module defines ``Poisson1DVanillaPINNConfig`` — the single source of truth
for all 1D Poisson PINN experiments. It is imported by both training scripts
and analysis scripts to ensure consistency.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from unipinn.config.base import BaseExperimentConfig


@dataclass
class Poisson1DVanillaPINNConfig(BaseExperimentConfig):
    """Full configuration for 1D Poisson vanilla PINN training.

    Subclasses ``BaseExperimentConfig`` and adds problem-specific fields.
    """
    # Override defaults from base
    exp_name: str = "poisson1d_vanilla"
    epochs: int = 6000

    # Data and benchmarks
    benchmark_name: str = "steep_solution"
    benchmark_params: Dict[str, float] = field(default_factory=lambda: {"a": 4 * np.pi, "b": 20.0})
    benchmark_sampling_method: str = "uniform"
    benchmark_lhs: Optional[str] = None
    n_colloc: int = 101
    n_eval: int = 501

    # Model architecture
    arch_config: Dict[int, Dict[str, Union[int, str]]] = field(default_factory=lambda: {
        1: {"n": 1},
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
