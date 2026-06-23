"""Logging callbacks for training monitoring."""

import torch
from unipinn.core.trainer import Callback


class ConsoleLogger(Callback):
    """Print loss at fixed epoch intervals."""
    def __init__(self, interval: int = 200,
                 fmt: str = "| Total: {total:.8e} | PDE: {pde:.8e} | BC: {bc:.8e}"):
        self.interval = interval
        self.fmt = fmt

    def on_epoch_end(self, trainer, epoch, loss_dict, **kwargs):
        if (epoch + 1) % self.interval == 0 or epoch == 0:
            print(f"Epoch: {epoch + 1}".ljust(18),
                  self.fmt.format(epoch=epoch + 1, **loss_dict))


class ConsoleLoggerTimeit(Callback):
    """Console logger with accumulated time tracking between prints."""
    def __init__(self, interval: int = 200,
                 fmt: str = "| Total: {total:.8e} | PDE: {pde:.8e} | BC: {bc:.8e} | dT: {interval_time:.4f}s"):
        self.interval = interval
        self.fmt = fmt
        self._interval_time = 0.0

    def on_epoch_end(self, trainer, epoch, loss_dict, epoch_delta, **kwargs):
        self._interval_time += epoch_delta
        if (epoch + 1) % self.interval == 0 or epoch == 0:
            print(f"Epoch: {epoch + 1}".ljust(18),
                  self.fmt.format(epoch=epoch + 1, **loss_dict, interval_time=self._interval_time))
            self._interval_time = 0.0


class TimingSummaryCallback(Callback):
    """Print timing statistics at the end of training."""
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def on_training_end(self, trainer, total_time, avg_time_per_step, **kwargs):
        if not self.verbose:
            return
        print("\n" + "=" * 60)
        print("TRAINING TIME SUMMARY")
        print("=" * 60)
        print(f"  Total Training Time:     {total_time:>10.2f} s  ({total_time / 60:.2f} min)")
        print(f"  Total Epochs:            {trainer.current_epoch + 1:>10d}")
        print(f"  Avg Time per Epoch:      {avg_time_per_step:>10.4f} s")

        if trainer.history.get("time_per_step"):
            times = trainer.history["time_per_step"]
            print(f"  Fastest Step:            {min(times):>10.4f} s")
            print(f"  Slowest Step:            {max(times):>10.4f} s")
            print(f"  Std Dev:                 {torch.std(torch.tensor(times)):>10.4f} s")
        print("=" * 60)
