"""Base experiment configuration dataclass.

Provides shared fields and validation common to all experiment configs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class BaseExperimentConfig:
    """Shared configuration fields for all experiments."""
    # Identification
    exp_name: str = "experiment"
    exp_group: Optional[List[str]] = None
    seed: Union[int, List[int]] = 42

    # Device and precision
    device: str = "auto"
    precision: str = "float32"
    deterministic: bool = True

    # Training loop
    epochs: int = 6000
    log_interval: int = 200

    # ============== Optional diagnostic callbacks ==============
    # Set to a list of epoch numbers to activate; None = disabled.
    spectra_epochs: Optional[List[int]] = None
    landscape_epochs: Optional[List[int]] = None
    landscape_grid_size: int = 40
    landscape_alpha_range: float = 1.0
    landscape_beta_range: float = 1.0