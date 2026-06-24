"""Neural network architectures for PINNs.

Provides ``SimpleNN``, a configurable feedforward network supporting a variety
of custom activation functions per layer.
"""

import torch
from torch import nn
from typing import Dict, List, Union, TypeAlias

from unipinn.nn.activations import (
    SineActivation, SineActivationT, SineActivationT3, DynSine,
    AdaptiveTanh, DyT, FourierActivation,
    PolynomialActivationV2, ChebyshevActivation,
)

CONFIG_TYPE: TypeAlias = Dict[int, Dict[str, Union[int, float, str]]]


class SimpleNN(nn.Module):
    """Configurable feedforward network with per-layer activation functions.

    Args:
        config_layers: Layer configuration dictionary.

    Configuration format::

        config = {
            1: {"n": 1},                              # input layer (1D)
            2: {"n": 32, "a": "dynsine", "p": 1},     # hidden: 32 neurons, DynSine
            3: {"n": 32, "a": "dynsine", "p": 1},
            4: {"n": 32, "a": "dynsine", "p": 1},
            5: {"n": 1},                              # output layer (1D)
        }

    Activation function keys (``"a"``) and their parameter (``"p"``):
        - ``"relu"``, ``"sigmoid"``, ``"tanh"``: no parameter needed
        - ``"sin"``: ``p`` = period T
        - ``"sinada"``: ``p`` = period T
        - ``"sinadafull"``: ``p`` = period T
        - ``"dynsine"``: ``p`` = init_alpha (related to initial frequency)
        - ``"tanhada"``: ``p`` = scale_factor
        - ``"dyt"``: ``p`` = init_alpha
        - ``"poly"`` or ``"taylor"``: ``p`` = polynomial degree
        - ``"cheby"``: ``p`` = Chebyshev degree
        - ``"fourier"``: ``p`` = grid_size (minimum period)
    """
    def __init__(self, config_layers: CONFIG_TYPE):
        super().__init__()
        self.config = {int(k): v for k, v in config_layers.items()}

        modules = []
        for idx in sorted(self.config.keys())[:-1]:
            cfg = self.config[idx]
            if "a" in cfg:
                modules.append(self._get_activation(cfg, input_dim=cfg["n"]))
            modules.append(nn.Linear(cfg["n"], self.config[idx + 1]["n"]))

        # Output layer activation (if specified)
        last_idx = max(self.config.keys())
        if "a" in self.config[last_idx]:
            modules.append(self._get_activation(self.config[last_idx], input_dim=self.config[last_idx]["n"]))

        self.net = nn.Sequential(*modules)

    def _get_activation(self, cfg: dict, input_dim: int = None) -> nn.Module:
        name = cfg["a"]
        param = cfg.get("p")
        match name:
            case "relu": return nn.ReLU()
            case "sigmoid": return nn.Sigmoid()
            case "tanh": return nn.Tanh()
            case "sin": return SineActivation(T=param)
            case "sinada": return SineActivationT(T=param)
            case "sinadafull": return SineActivationT3(T=param)
            case "dynsine": return DynSine(init_alpha=param, input_dim=input_dim)
            case "tanhada": return AdaptiveTanh(scale_factor=param)
            case "dyt": return DyT(init_alpha=param, input_dim=input_dim)
            case "poly" | "taylor": return PolynomialActivationV2(n_degree=param, input_dim=input_dim)
            case "cheby": return ChebyshevActivation(n_degree=param, input_dim=input_dim)
            case "fourier": return FourierActivation(
                grid_size=param, input_dim=input_dim, addbias=True, smooth_initialization=True
            )
            case _:
                raise ValueError(f'Unknown activation function: "{name}"')

    def forward(self, x):
        return self.net(x)

    def print_params(self, num_only: bool = False, if_return: bool = False):
        """Print parameter statistics."""
        all_params = list(self.net.parameters())
        trainable = [p for p in all_params if p.requires_grad]
        total = sum(p.numel() for p in all_params)
        n_trainable = sum(p.numel() for p in trainable)

        if not num_only:
            print(all_params)
            print(trainable)
        print(f"\nTotal: {total}; Trainable: {n_trainable}")

        if if_return:
            return all_params, trainable, total, n_trainable
