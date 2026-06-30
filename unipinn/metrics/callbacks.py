"""Callback classes for training diagnostics.

Provides callbacks that integrate with the Trainer to compute and visualize:
- Singular value spectra of network weights at specified epochs
- Loss landscape (2D contour and 3D surface) at specified epochs

These callbacks help analyze the optimization dynamics and network conditioning
during PINN training.
"""

import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import json

from unipinn.core.trainer import Callback
from unipinn.metrics.spectra import compute_weight_spectra, plot_spectra
from unipinn.metrics.landscape import (
    compute_loss_landscape,
    plot_landscape,
    plot_landscape_3d,
)


class SpectraCallback(Callback):
    """Compute and plot singular value spectra at specified epochs.

    This callback analyzes the conditioning of the network's weight matrices
    by computing their singular value decomposition. This can reveal how the
    network's expressivity evolves during training.

    Args:
        output_dir: Directory to save spectrum plots.
        epochs: List of specific epochs to compute spectra (e.g., [0, 1000, 5000]).
            If None, uses every `interval` epochs.
        interval: Compute spectra every N epochs (used if `epochs` is None).
        log_scale: Use logarithmic y-axis for singular values.

    Example:
        >>> # Plot at specific epochs
        >>> spectra_cb = SpectraCallback(
        ...     output_dir="results/spectra",
        ...     epochs=[0, 1000, 5000, 10000]
        ... )
        >>> trainer = Trainer(..., callbacks=[spectra_cb])

        >>> # Plot every 2000 epochs
        >>> spectra_cb = SpectraCallback(
        ...     output_dir="results/spectra",
        ...     interval=2000
        ... )
    """

    def __init__(
        self,
        output_dir: str,
        epochs: Optional[List[int]] = None,
        interval: int = 1000,
        log_scale: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = set(epochs) if epochs is not None else None
        self.interval = interval
        self.log_scale = log_scale

    def _should_compute(self, epoch: int) -> bool:
        """Check if spectra should be computed at this epoch."""
        if self.epochs is not None:
            return epoch in self.epochs
        return epoch % self.interval == 0

    def on_epoch_end(self, trainer, epoch, loss_dict, epoch_delta, **kwargs):
        """Compute spectra at specified epochs."""
        if not self._should_compute(epoch):
            return

        model = trainer.model
        spectra = compute_weight_spectra(model)

        save_path = self.output_dir / f"spectra_epoch_{epoch:06d}.png"
        plot_spectra(spectra, epoch=epoch, save_path=str(save_path), log_scale=self.log_scale)
        plt.close()  # Free memory


class LandscapeCallback(Callback):
    """Compute and plot loss landscape at specified epochs.

    This callback visualizes the optimization landscape by perturbing model
    parameters along two random directions and evaluating the loss on a 2D grid.
    This reveals the geometry of the loss surface and helps diagnose optimization
    difficulties.

    Note: Landscape computation is expensive (grid_size^2 forward passes), so
    use sparingly (e.g., at a few key epochs).

    Args:
        output_dir: Directory to save landscape plots and data.
        epochs: List of specific epochs to compute landscapes.
            If None, uses every `interval` epochs.
        interval: Compute landscape every N epochs (used if `epochs` is None).
        grid_size: Number of grid points per axis (total = grid_size^2).
        alpha_range: Range for d1 direction: [-alpha_range, +alpha_range].
        beta_range: Range for d2 direction: [-beta_range, +beta_range].
        seed: Random seed for direction generation (reproducibility).
        filter_norm: Use filter-normalized directions (recommended).
        plot_3d: Also generate 3D surface plot (slower).
        verbose: Print progress during computation.

    Example:
        >>> # Compute landscape at key epochs
        >>> landscape_cb = LandscapeCallback(
        ...     output_dir="results/landscape",
        ...     epochs=[0, 5000, 10000, 20000],
        ...     grid_size=40,
        ...     alpha_range=0.5,
        ...     beta_range=0.5,
        ... )
        >>> trainer = Trainer(..., callbacks=[landscape_cb])
    """

    def __init__(
        self,
        output_dir: str,
        epochs: Optional[List[int]] = None,
        interval: int = 5000,
        grid_size: int = 40,
        alpha_range: float = 1.0,
        beta_range: float = 1.0,
        seed: int = 42,
        filter_norm: bool = True,
        plot_3d: bool = False,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = set(epochs) if epochs is not None else None
        self.interval = interval
        self.grid_size = grid_size
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.seed = seed
        self.filter_norm = filter_norm
        self.plot_3d = plot_3d
        self.verbose = verbose

    def _should_compute(self, epoch: int) -> bool:
        """Check if landscape should be computed at this epoch."""
        if self.epochs is not None:
            return epoch in self.epochs
        return epoch % self.interval == 0

    def on_epoch_end(self, trainer, epoch, loss_dict, epoch_delta, batch=None, **kwargs):
        """Compute landscape at specified epochs."""
        if not self._should_compute(epoch):
            return

        model = trainer.model
        loss_fn = trainer.loss_fn

        print(f"\n[LandscapeCallback] Computing landscape at epoch {epoch}...")

        result = compute_loss_landscape(
            model=model,
            loss_fn=loss_fn,
            batch=batch,
            grid_size=self.grid_size,
            alpha_range=self.alpha_range,
            beta_range=self.beta_range,
            seed=self.seed,
            filter_norm=self.filter_norm,
            verbose=self.verbose,
        )

        # Save 2D contour plot
        save_path_2d = self.output_dir / f"landscape_epoch_{epoch:06d}_2d.png"
        plot_landscape(result, save_path=str(save_path_2d), log_scale=True)
        plt.close()

        # Save 3D surface plot (optional)
        if self.plot_3d:
            save_path_3d = self.output_dir / f"landscape_epoch_{epoch:06d}_3d.png"
            plot_landscape_3d(result, save_path=str(save_path_3d), log_scale=True)
            plt.close()

        # Save metadata (seeds, baseline loss, etc.)
        metadata = {
            "epoch": epoch,
            "grid_size": self.grid_size,
            "alpha_range": self.alpha_range,
            "beta_range": self.beta_range,
            "d1_seed": result["d1_seed"],
            "d2_seed": result["d2_seed"],
            "baseline_loss": result["baseline_loss"],
            "filter_norm": self.filter_norm,
        }
        metadata_path = self.output_dir / f"landscape_epoch_{epoch:06d}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[LandscapeCallback] Saved to {self.output_dir}")
