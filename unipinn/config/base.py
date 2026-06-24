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
