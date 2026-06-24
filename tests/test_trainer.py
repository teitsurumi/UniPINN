"""Tests for the training engine."""

import torch
from unipinn.core.trainer import Trainer, Callback
from unipinn.pde.loss import PINNLossPoisson1D


def test_trainer_fit_runs(simple_model, sample_1d_batch):
    """Trainer should complete fit without error."""
    loss_fn = PINNLossPoisson1D()
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    trainer = Trainer(simple_model, loss_fn, optimizer, device="cpu")
    trainer.fit(epochs=10, batch=sample_1d_batch)
    assert len(trainer.history["epoch"]) == 10
    assert "total" in trainer.history["loss"]


def test_trainer_lbfgs(simple_model, sample_1d_batch):
    """Trainer should work with LBFGS optimizer."""
    loss_fn = PINNLossPoisson1D()
    optimizer = torch.optim.LBFGS(simple_model.parameters(), lr=0.1, max_iter=5)
    trainer = Trainer(simple_model, loss_fn, optimizer, device="cpu")
    trainer.fit(epochs=5, batch=sample_1d_batch)
    assert len(trainer.history["epoch"]) == 5


def test_callback_triggered(simple_model, sample_1d_batch):
    """Custom callback should be called during training."""
    class Counter(Callback):
        def __init__(self):
            self.count = 0
        def on_epoch_end(self, **kwargs):
            self.count += 1

    cb = Counter()
    loss_fn = PINNLossPoisson1D()
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    trainer = Trainer(simple_model, loss_fn, optimizer, callbacks=[cb], device="cpu")
    trainer.fit(epochs=5, batch=sample_1d_batch)
    assert cb.count == 5
