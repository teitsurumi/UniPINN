"""Training modifier callbacks (optimizer switching, etc.)."""

import torch
from unipinn.core.trainer import Callback


class DynamicSwitchCallback(Callback):
    """Switch optimizer at a specified epoch (e.g., LBFGS -> Adam for fine-tuning)."""
    def __init__(self, switch_epoch: int = 3000,
                 new_opt_cls=torch.optim.Adam, new_lr: float = 1e-4):
        self.switch_epoch = switch_epoch
        self.new_opt_cls = new_opt_cls
        self.new_lr = new_lr

    def on_epoch_end(self, trainer, epoch, **kwargs):
        if epoch + 1 == self.switch_epoch:
            new_opt = self.new_opt_cls(trainer.model.parameters(), lr=self.new_lr)
            trainer.switch_optimizer(new_opt)
            print(f"\nSwitched to {self.new_opt_cls.__name__} at epoch {self.switch_epoch}")
