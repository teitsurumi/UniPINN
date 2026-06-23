"""Tests for neural network architectures and activation functions."""

import torch
import pytest
from unipinn.nn.architectures import SimpleNN
from unipinn.nn import activations as act


def test_simple_nn_forward():
    config = {1: {"n": 1}, 2: {"n": 8, "a": "tanh"}, 3: {"n": 1}}
    model = SimpleNN(config)
    x = torch.randn(10, 1)
    y = model(x)
    assert y.shape == (10, 1)


def test_simple_nn_with_dynsine():
    config = {
        1: {"n": 1},
        2: {"n": 16, "a": "dynsine", "p": 1},
        3: {"n": 1},
    }
    model = SimpleNN(config)
    x = torch.linspace(-1, 1, 20).reshape(-1, 1)
    y = model(x)
    assert y.shape == (20, 1)


@pytest.mark.parametrize("cls,kwargs", [
    (act.AdaptiveTanh, {"scale_factor": 2.0}),
    (act.DyT, {"input_dim": 8, "init_alpha": 1.0}),
    (act.SineActivation, {"T": 1.0}),
    (act.SineActivationT, {"T": 1.0}),
    (act.DynSine, {"input_dim": 8, "init_alpha": 1.0}),
    (act.PolynomialActivationV2, {"input_dim": 8, "n_degree": 3}),
    (act.ChebyshevActivation, {"input_dim": 8, "n_degree": 3}),
    (act.FourierActivation, {"input_dim": 8, "grid_size": 5}),
])
def test_activation_forward(cls, kwargs):
    module = cls(**kwargs)
    x = torch.randn(10, 8)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
