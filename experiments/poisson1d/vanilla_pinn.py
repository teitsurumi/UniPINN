"""1D Poisson — Vanilla PINN training experiment.

Supports both single-config runs and batch / parallel execution of the
pre-defined configuration registry (``registry.py``).

Usage
-----
# Single run with default config
python experiments/poisson1d/vanilla_pinn.py

# Single run with a JSON config file
python experiments/poisson1d/vanilla_pinn.py --config configs/my_run.json

# List all registry configs
python experiments/poisson1d/vanilla_pinn.py --list

# Batch run — all 14 configs (sequential)
python experiments/poisson1d/vanilla_pinn.py --batch

# Batch run — 4 parallel workers
python experiments/poisson1d/vanilla_pinn.py --batch --parallel 4

# Batch run — specific configs
python experiments/poisson1d/vanilla_pinn.py --batch --configs cfg_0 cfg_1 cfg_2

# Batch run — filter by group
python experiments/poisson1d/vanilla_pinn.py --batch --group SampleNumTest --parallel 3

# Batch run — assign GPUs round-robin
python experiments/poisson1d/vanilla_pinn.py --batch --parallel 4 --gpus 0 1
"""

import io
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path so experiments.poisson1d.* is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
from unipinn.metrics.callbacks import SpectraCallback, LandscapeCallback


# ===================================================================
# Training
# ===================================================================

def build_experiment(cfg: Poisson1DVanillaPINNConfig, result_dir: Optional[Path] = None):
    """Construct and run a single training experiment from config."""
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

    # Optional diagnostic callbacks (activated via config fields)
    if cfg.spectra_epochs is not None and result_dir is not None:
        callbacks.append(SpectraCallback(
            output_dir=str(result_dir / "spectra"),
            epochs=cfg.spectra_epochs,
        ))

    if cfg.landscape_epochs is not None and result_dir is not None:
        callbacks.append(LandscapeCallback(
            output_dir=str(result_dir / "landscape"),
            epochs=cfg.landscape_epochs,
            grid_size=cfg.landscape_grid_size,
            alpha_range=cfg.landscape_alpha_range,
            beta_range=cfg.landscape_beta_range,
        ))

    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        scheduler=scheduler, warmup_epochs=cfg.warmup_epochs,
        callbacks=callbacks, device=device,
    )
    trainer.cfg = cfg
    trainer.fit(cfg.epochs, batch)
    return trainer, batch, data


def run_single_experiment(
    cfg: Poisson1DVanillaPINNConfig,
    result_dir: Path,
    *,
    suppress_stdout: bool = True,
) -> Path:
    """Run one (config, seed) experiment and save all artifacts.

    Parameters
    ----------
    cfg : Poisson1DVanillaPINNConfig
        Configuration (seed should already be a scalar int).
    result_dir : Path
        Root result directory.  A ``seed_<N>/`` sub-folder is created.
    suppress_stdout : bool
        Capture console output so parallel workers don't interleave prints.

    Returns
    -------
    Path  — the ``seed_<N>/`` directory containing artifacts.
    """
    seed_dir = result_dir / f"seed_{cfg.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    log_buffer = io.StringIO()
    if suppress_stdout:
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        sys.stdout = StreamTee(sys.__stdout__, log_buffer)
        sys.stderr = StreamTee(sys.__stderr__, log_buffer)

    try:
        trainer, batch, data = build_experiment(cfg, result_dir=seed_dir)
    finally:
        if suppress_stdout:
            sys.stdout, sys.stderr = saved_stdout, saved_stderr

    console_logs = log_buffer.getvalue()

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

    save_experiment_artifacts(seed_dir, cfg, trainer, data, u_pred, console_logs)
    print(f"seed_{cfg.seed} artifacts saved to {seed_dir}")
    return seed_dir


# ===================================================================
# Reports
# ===================================================================

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


# ===================================================================
# Batch / parallel execution
# ===================================================================

def _batch_worker(task: Tuple[str, Poisson1DVanillaPINNConfig, int, Path, Optional[int]]) -> Dict:
    """Run a single (config, seed) task inside a child process."""
    cfg_name, cfg, seed, result_dir, gpu_id = task

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cfg_with_seed = replace(cfg, seed=seed)
    cfg_result_dir = result_dir / cfg_name
    cfg_result_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        # Ensure project root is on sys.path in the spawned process
        _root = str(Path(__file__).resolve().parent.parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)

        from experiments.poisson1d.vanilla_pinn import run_single_experiment
        seed_dir = run_single_experiment(cfg_with_seed, cfg_result_dir, suppress_stdout=True)
        elapsed = time.time() - t0
        return {
            "cfg_name": cfg_name, "seed": seed, "status": "ok",
            "elapsed": elapsed, "result_dir": str(seed_dir), "error": None,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "cfg_name": cfg_name, "seed": seed, "status": "error",
            "elapsed": elapsed, "result_dir": str(cfg_result_dir),
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def _expand_tasks(
    selected: Dict[str, Poisson1DVanillaPINNConfig],
    result_dir: Path,
    gpus: List[int],
) -> List[Tuple[str, Poisson1DVanillaPINNConfig, int, Path, Optional[int]]]:
    """Expand configs into per-seed tasks, assigning GPUs round-robin."""
    tasks, gpu_idx = [], 0
    for cfg_name, cfg in selected.items():
        seeds = [cfg.seed] if isinstance(cfg.seed, int) else list(cfg.seed)
        for s in seeds:
            gpu = gpus[gpu_idx % len(gpus)] if gpus else None
            tasks.append((cfg_name, cfg, s, result_dir, gpu))
            gpu_idx += 1
    return tasks


def _write_batch_summary(result_dir: Path, results: List[Dict], wall_time: float):
    """Write a markdown summary of the batch run."""
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "error"]

    lines = [
        "# Batch Run Summary", "",
        f"- **Total tasks**: {len(results)}",
        f"- **Succeeded**: {len(ok)}",
        f"- **Failed**: {len(failed)}",
        f"- **Wall time**: {wall_time:.1f}s ({wall_time / 60:.1f}min)",
        f"- **Total CPU time**: {sum(r['elapsed'] for r in results):.1f}s",
        f"- **Speedup**: {sum(r['elapsed'] for r in results) / max(wall_time, 1e-3):.2f}x",
        "", "## Results", "",
        "| Config | Seed | Status | Elapsed |",
        "|--------|------|--------|---------|",
    ]
    for r in sorted(results, key=lambda r: (r["cfg_name"], r["seed"])):
        tag = "ok" if r["status"] == "ok" else "FAIL"
        lines.append(f"| {r['cfg_name']} | {r['seed']} | {tag} | {r['elapsed']:.1f}s |")

    if failed:
        lines += ["", "## Errors", ""]
        for r in failed:
            lines += [f"### {r['cfg_name']} / seed {r['seed']}", "", "```", r["error"], "```", ""]

    (result_dir / "BATCH_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _run_batch(args):
    """Execute batch mode: select registry configs, expand seeds, run."""
    from experiments.poisson1d.registry import REGISTRY, list_configs

    # --list: print table and exit
    if args.list:
        print(f"{'Name':<10} {'Optimizer':<10} {'Epochs':>8} {'n_colloc':>9} "
              f"{'Sampling':<10} {'Seeds':>6} {'Groups'}")
        print("-" * 80)
        for name in list_configs():
            cfg = REGISTRY[name]
            seeds = [cfg.seed] if isinstance(cfg.seed, int) else cfg.seed
            groups = ", ".join(cfg.exp_group or [])
            print(f"{name:<10} {cfg.optimizer:<10} {cfg.epochs:>8} {cfg.n_colloc:>9} "
                  f"{cfg.benchmark_sampling_method:<10} {len(seeds):>6} {groups}")
        return

    # Select configs
    if args.configs:
        selected = {k: REGISTRY[k] for k in args.configs if k in REGISTRY}
        missing = set(args.configs) - set(selected)
        if missing:
            print(f"Warning: unknown configs ignored: {missing}")
    elif args.group:
        selected = {
            k: v for k, v in REGISTRY.items()
            if v.exp_group and args.group in v.exp_group
        }
        if not selected:
            print(f"No configs found for group '{args.group}'.")
            print(f"Available groups: "
                  f"{sorted({g for c in REGISTRY.values() for g in (c.exp_group or [])})}")
            return
    else:
        selected = dict(REGISTRY)

    if not selected:
        print("No configs to run.")
        return

    result_dir = get_next_test_case_dir(args.problem, args.framework)
    result_dir.mkdir(parents=True, exist_ok=True)

    gpus = args.gpus if args.gpus is not None else []
    tasks = _expand_tasks(selected, result_dir, gpus)

    print(f"\n{'=' * 60}")
    print(f" Batch run: {len(selected)} config(s), {len(tasks)} task(s)")
    print(f" Parallel workers: {args.parallel}")
    print(f" GPUs: {gpus if gpus else 'auto'}")
    print(f" Results: {result_dir}")
    print(f"{'=' * 60}\n")

    (result_dir / "batch_info.json").write_text(json.dumps({
        "configs": list(selected.keys()),
        "parallel": args.parallel,
        "gpus": gpus,
        "n_tasks": len(tasks),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")

    t_start = time.time()
    results: List[Dict] = []

    if args.parallel <= 1:
        for i, task in enumerate(tasks):
            cfg_name, _, seed, _, _ = task
            print(f"[{i + 1}/{len(tasks)}] {cfg_name} / seed {seed} ...")
            result = _batch_worker(task)
            results.append(result)
            tag = "ok" if result["status"] == "ok" else "FAIL"
            print(f"  -> {tag} ({result['elapsed']:.1f}s)")
    else:
        pool = mp.Pool(processes=args.parallel)
        try:
            for i, result in enumerate(pool.imap_unordered(_batch_worker, tasks)):
                results.append(result)
                tag = "ok" if result["status"] == "ok" else "FAIL"
                print(f"[{i + 1}/{len(tasks)}] {result['cfg_name']} / seed {result['seed']} "
                      f"-> {tag} ({result['elapsed']:.1f}s)")
        finally:
            pool.close()
            pool.join()

    wall_time = time.time() - t_start
    ok_count = sum(1 for r in results if r["status"] == "ok")
    fail_count = len(results) - ok_count

    print(f"\n{'=' * 60}")
    print(f" Batch complete: {ok_count}/{len(results)} succeeded, "
          f"{fail_count} failed, wall time {wall_time:.1f}s")
    print(f" Results: {result_dir}")
    print(f"{'=' * 60}")

    _write_batch_summary(result_dir, results, wall_time)
    print(f" Summary written to: {result_dir / 'BATCH_SUMMARY.md'}")


def _run_single(args):
    """Execute single-config mode: one config, loop over seeds."""
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = Poisson1DVanillaPINNConfig(**cfg_dict)
    else:
        cfg = Poisson1DVanillaPINNConfig()

    seeds = [cfg.seed] if isinstance(cfg.seed, int) else list(cfg.seed)
    result_dir = get_next_test_case_dir(args.problem, args.framework)
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {result_dir}")

    for s in seeds:
        print(f"\n{'=' * 40}\nRunning seed: {s}\n{'=' * 40}")
        cfg_s = replace(cfg, seed=s)
        run_single_experiment(cfg_s, result_dir, suppress_stdout=True)

    comments = _generate_features_report(cfg, seeds)
    (result_dir / "COMMENTS.md").write_text(comments, encoding="utf-8")
    print(f"\nAll {len(seeds)} seed(s) completed. Results in: {result_dir}")


# ===================================================================
# CLI
# ===================================================================

if __name__ == "__main__":
    # spawn must be set before any pool creation
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="1D Poisson Vanilla PINN — single and batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mode
    parser.add_argument("--batch", action="store_true",
                        help="Run in batch mode using the config registry.")
    parser.add_argument("--list", action="store_true",
                        help="(batch) List all registry configs and exit.")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="(batch) Config names to run (e.g. cfg_0 cfg_5).")
    parser.add_argument("--group", type=str, default=None,
                        help="(batch) Run all configs in a group (e.g. SampleNumTest).")
    parser.add_argument("--parallel", "-j", type=int, default=1,
                        help="(batch) Number of parallel workers (default: 1).")
    parser.add_argument("--gpus", nargs="*", type=int, default=None,
                        help="(batch) GPU IDs to assign round-robin.")

    # Single-mode
    parser.add_argument("--config", type=str, default=None,
                        help="(single) Path to a JSON config file.")

    # Common
    parser.add_argument("--problem", type=str, default="poisson1d")
    parser.add_argument("--framework", type=str, default="01_pinn")

    args = parser.parse_args()

    if args.batch or args.list or args.configs or args.group:
        args.batch = True          # --list / --configs / --group imply batch
        _run_batch(args)
    else:
        _run_single(args)
