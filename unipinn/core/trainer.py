"""Training engine with callback support.

Provides a lightweight ``Trainer`` class for PINN training loops with
LBFGS/Adam support, optimizer switching, and a callback system.
"""

import torch
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class Callback:
    """Base callback class. Override specific event methods as needed."""
    def on_epoch_begin(self, trainer): pass
    def on_epoch_end(self, trainer, epoch: int, loss_dict: Dict[str, float], epoch_delta: float): pass
    def on_optimizer_switch(self, trainer, old_opt, new_opt): pass
    def on_training_end(self, trainer, total_time: float, avg_time_per_step: float): pass


@dataclass
class CallbackManager:
    """Manages and dispatches events to registered callbacks."""
    callbacks: List[Callback] = field(default_factory=list)

    def trigger(self, event: str, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, event, None)
            if fn:
                fn(**kwargs)


class Trainer:
    """PINN training loop with LBFGS/Adam support and callback system.

    Args:
        model: Neural network model.
        loss_fn: Loss function callable(model, batch) -> dict with 'total' key.
        optimizer: PyTorch optimizer.
        scheduler: Optional learning rate scheduler.
        warmup_epochs: Number of epochs before activating the scheduler.
        callbacks: List of Callback instances.
        device: Training device.
    """
    def __init__(self, model: torch.nn.Module, loss_fn, optimizer: torch.optim.Optimizer,
                 scheduler=None, warmup_epochs: int = 0,
                 callbacks: Optional[List[Callback]] = None,
                 device: str = "cpu"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup = warmup_epochs
        self.device = device
        self.callbacks = CallbackManager(callbacks or [])
        self.history: Dict = {"epoch": [], "loss": {}, "metrics": {}}
        self._use_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        self.current_epoch = 0
        self._start_time = None
        self._epoch_start_time = None
        self._total_time = 0.0

    def switch_optimizer(self, new_opt: torch.optim.Optimizer, new_sch=None):
        """Replace the current optimizer (and optionally scheduler) mid-training."""
        old_opt = self.optimizer
        self.optimizer = new_opt
        self.scheduler = new_sch or self.scheduler
        self._use_lbfgs = isinstance(new_opt, torch.optim.LBFGS)
        self.callbacks.trigger("on_optimizer_switch", trainer=self, old_opt=old_opt, new_opt=new_opt)

    def step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute a single training step."""
        self._last_losses = None

        if self._use_lbfgs:
            def closure():
                self.optimizer.zero_grad()
                losses = self.loss_fn(self.model, batch)
                self._last_losses = losses
                losses["total"].backward()
                return losses["total"]
            loss_total = self.optimizer.step(closure)
        else:
            self.optimizer.zero_grad()
            self._last_losses = self.loss_fn(self.model, batch)
            self._last_losses["total"].backward()
            self.optimizer.step()
            loss_total = self._last_losses["total"]

        return loss_total, self._last_losses

    def fit(self, epochs: int, batch: Dict[str, torch.Tensor]):
        """Run the full training loop."""
        self._start_time = time.time()
        self._total_time = 0.0

        for ep in range(epochs):
            self.current_epoch = ep
            self._epoch_start_time = time.time()

            self.callbacks.trigger("on_epoch_begin", trainer=self)

            loss_total, loss_dict = self.step(batch)

            epoch_delta = time.time() - self._epoch_start_time
            self._total_time += epoch_delta

            loss_dict_np = {k: v.item() for k, v in loss_dict.items()}

            self.callbacks.trigger(
                "on_epoch_end",
                trainer=self,
                epoch=ep,
                loss_dict=loss_dict_np,
                epoch_delta=epoch_delta,
            )

            if ep >= self.warmup and self.scheduler:
                self.scheduler.step()

            self.history["epoch"].append(ep)
            for k, v in loss_dict_np.items():
                self.history["loss"].setdefault(k, []).append(v)

        total_time = time.time() - self._start_time
        avg_time_per_step = total_time / epochs

        self.callbacks.trigger(
            "on_training_end",
            trainer=self,
            total_time=total_time,
            avg_time_per_step=avg_time_per_step,
        )

        return self.model
