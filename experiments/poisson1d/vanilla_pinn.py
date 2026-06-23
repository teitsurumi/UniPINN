"""1D Poisson — Vanilla PINN training experiment.

Usage:
    # Single run (default config)
    python experiments/poisson1d/vanilla_pinn.py

    # Single run with JSON config
    python experiments/poisson1d/vanilla_pinn.py --config configs/my_run.json

    # Batch / parallel run (see experiments/poisson1d/batch.py for full options)
    python experiments/poisson1d/batch.py --parallel 4

The ``run_single_experiment`` function is the main entry-point for programmatic use.
It runs one (config, seed) pair and returns the result directory path.
"""

import sys
import io
import json
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

from unipinn.config.poisson1d import Poisson1DVanillaPINNConfig
from unipinn.nn.architectures import SimpleNN
from unipinn.utils.tensor import to_tensor, to_numpy
from unipinn.io.console import StreamTee
from unipinn.io.artifacts import SafeJSONEncoder, save_models, save_experiment_artifacts
from unipinn.io.workspace import get_next_test_case_dir
from unipinn.pde.benchmarks.poisson1d import Poisson1DBenchmarkIndex, BasePoisson1DBenchmark
from unipinn.core.trainer import Trainer
from unipinn.pde.loss import PINNLossPoisson1D
from unipinn.core.callbacks.loggers import ConsoleLoggerTimeit, TimingSummaryCallback


# ---------------------------------------------------------------------------
# Build helpers (model / loss / optimizer / scheduler)
# ---------------------------------------------------------------------------

def build_experiment(cfg: Poisson1DVanillaPINNConfig):
    """Construct and run a single training experiment from config."""
    # Environment & precision
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    global_dtype = torch.float64 if cfg.precision == "float64" else torch.float32
    torch.set_default_dtype(global_dtype)

    device_str = cfg.device
    if device_str == "auto":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[Device] {device} | [Precision] {cfg.precision}")

    # Data generation
    bench = Poisson1DBenchmarkIndex.get(cfg.benchmark_name, **cfg.benchmark_params)
    data = bench.generate(
        n_colloc=cfg.n_colloc, n_eval=cfg.n_eval,
        sampling_method=cfg.benchmark_sampling_method,
        seed=cfg.seed, lhs_criterion=cfg.benchmark_lhs,
    )
    batch = {
        "x_col": to_tensor(data["x_colloc"], dtype=global_dtype, device=device).reshape(-1, 1).requires_grad_(True),
        "f_col": to_tensor(data["f_colloc"], dtype=global_dtype, device=device).reshape(-1, 1),
        "x_bc": to_tensor(data["x_bc"], dtype=global_dtype, device=device).reshape(-1, 1).requires_grad_(True),
        "bc_target": to_tensor(
            data["u_bc"] if data["u_bc"] is not None else data["du_bc"],
            dtype=global_dtype, device=device,
        ).reshape(-1, 1),
    }

    # Model, loss, optimizer, scheduler, callbacks
    model = SimpleNN(cfg.arch_config).to(device)
    loss_fn = PINNLossPoisson1D(pde_weight=cfg.pde_weight, bc_weight=cfg.bc_weight)

    if cfg.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=cfg.lr, max_iter=cfg.lbfgs_max_iter,
            history_size=cfg.lbfgs_history_size, line_search_fn=cfg.lbfgs_line_search_fn,
        )
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, betas=cfg.adam_betas, weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    if cfg.scheduler_type == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma,
        )
    elif cfg.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.scheduler_T_max, eta_min=cfg.lr * 0.01,
        )
    elif cfg.scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.scheduler_type}")

    callbacks = [
        ConsoleLoggerTimeit(interval=cfg.log_interval),
        TimingSummaryCallback(verbose=True),
    ]

    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        scheduler=scheduler, warmup_epochs=cfg.warmup_epochs,
        callbacks=callbacks, device=device,
    )
    trainer.cfg = cfg
    trainer.fit(cfg.epochs, batch)
    return trainer, batch, data


# ---------------------------------------------------------------------------
# High-level runner: one (config, seed) → artifacts on disk
# ---------------------------------------------------------------------------

def run_single_experiment(
    cfg: Poisson1DVanillaPINNConfig,
    result_dir: Path,
    *,
    suppress_stdout: bool = True,
) -> Path:
    """Run a single (config, seed) experiment and save all artifacts.

    Parameters
    ----------
    cfg : Poisson1DVanillaPINNConfig
        Configuration (seed should already be a scalar int).
    result_dir : Path
        Root result directory. A ``seed_<N>/`` sub-folder will be created.
    suppress_stdout : bool
        If True (default), capture console output into a buffer so that
        parallel workers do not interleave prints.

    Returns
    -------
    Path
        Path to the ``seed_<N>/`` directory containing artifacts.
    """
    seed_dir = result_dir / f"seed_{cfg.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Capture console output
    log_buffer = io.StringIO()
    if suppress_stdout:
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        sys.stdout = StreamTee(sys.__stdout__, log_buffer)
        sys.stderr = StreamTee(sys.__stderr__, log_buffer)

    try:
        trainer, batch, data = build_experiment(cfg)
    finally:
        if suppress_stdout:
            sys.stdout, sys.stderr = saved_stdout, saved_stderr

    console_logs = log_buffer.getvalue()

    # Evaluation and plotting
    global_dtype = torch.float64 if cfg.precision == "float64" else torch.float32
    model = trainer.model
    model.eval()
    with torch.no_grad():
        x_eval = to_tensor(data["x_eval"], dtype=global_dtype, device=trainer.device).reshape(-1, 1)
        u_pred = model(x_eval).cpu().numpy().flatten()

    plt.figure(figsize=(6, 4), layout="constrained")
    plt.plot(data["x_eval"], data["u_eval"].flatten(), linewidth=4, c="#2980b9", alpha=0.5, label="Exact")
    plt.scatter(data["x_eval"], u_pred, s=1, c="#ff4757", label="PINN")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(seed_dir / "prediction_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save artifacts
    save_experiment_artifacts(seed_dir, cfg, trainer, data, u_pred, console_logs)
    print(f"seed_{cfg.seed} artifacts saved to {seed_dir}")
    return seed_dir


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------

def _generate_features_report(cfg: Poisson1DVanillaPINNConfig, seeds: list) -> str:
    """Generate a markdown report comparing config to defaults."""
    try:
        defaults = asdict(Poisson1DVanillaPINNConfig())
    except Exception:
        return "Warning: Could not load default config for comparison."

    current = asdict(cfg)
    features = []
    focus_keys = [
        "optimizer", "lr", "epochs", "precision", "benchmark_sampling_method",
        "benchmark_lhs", "n_colloc", "n_eval", "pde_weight", "bc_weight",
        "scheduler_type", "warmup_epochs", "grad_clip_max_norm", "deterministic",
        "benchmark_params", "arch_config",
    ]
    for key in focus_keys:
        if key in current and key in defaults and current[key] != defaults[key]:
            features.append(f"- **{key}**: `{current[key]}` *(default: `{defaults[key]}`)*")

    if not features:
        features.append("- All core parameters use default values.")

    header = "# Experiment Auto-Summary\n\n## Customized Features\n\n"
    header += "\n".join(features) + "\n\n"
    header += f"## Metadata\n\n"
    header += f"- **Seeds**: `{seeds}`\n"
    header += f"- **Generated**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n"
    header += f"- **Device/Precision**: `{cfg.device}` / `{cfg.precision}`\n"
    return header


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D Poisson Vanilla PINN Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--problem", type=str, default="poisson1d")
    parser.add_argument("--framework", type=str, default="01_pinn")
    args = parser.parse_args()

    # Load or create config
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = Poisson1DVanillaPINNConfig(**cfg_dict)
    else:
        cfg = Poisson1DVanillaPINNConfig()

    # Normalize seed to list
    seeds = [cfg.seed] if isinstance(cfg.seed, int) else list(cfg.seed)

    # Create result directory
    result_dir = get_next_test_case_dir(args.problem, args.framework)
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {result_dir}")

    # Training loop over seeds (sequential, with stdout capture)
    for s in seeds:
        print(f"\n{'=' * 40}\nRunning seed: {s}\n{'=' * 40}")
        cfg_s = replace(cfg, seed=s)
        run_single_experiment(cfg_s, result_dir, suppress_stdout=True)

    # Generate comments
    comments = _generate_features_report(cfg, seeds)
    (result_dir / "COMMENTS.md").write_text(comments, encoding="utf-8")
    print(f"\nAll {len(seeds)} seed(s) completed. Results in: {result_dir}")
