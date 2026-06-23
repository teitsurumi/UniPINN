"""Batch / parallel runner for 1D Poisson PINN experiments.

Runs multiple pre-defined configurations (see ``registry.py``), optionally
in parallel across multiple processes and/or GPUs.

Usage
-----
# Run all 14 configs sequentially
python experiments/poisson1d/batch.py

# Run all configs with 4 parallel workers
python experiments/poisson1d/batch.py --parallel 4

# Run only specific configs
python experiments/poisson1d/batch.py --configs cfg_0 cfg_1 cfg_2

# Run all configs in a specific group
python experiments/poisson1d/batch.py --group SampleNumTest --parallel 3

# Assign specific GPUs (round-robin across workers)
python experiments/poisson1d/batch.py --parallel 4 --gpus 0 1

# List all available configs and exit
python experiments/poisson1d/batch.py --list
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path so `experiments.poisson1d.*` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure spawn so each worker gets a fresh CUDA context
mp.set_start_method("spawn", force=True)

import numpy as np

from unipinn.config.poisson1d import Poisson1DVanillaPINNConfig
from unipinn.io.artifacts import SafeJSONEncoder
from unipinn.io.workspace import get_next_test_case_dir


# ---------------------------------------------------------------------------
# Worker: executed inside a child process
# ---------------------------------------------------------------------------

def _worker_run(task: Tuple[str, Poisson1DVanillaPINNConfig, int, Path, Optional[int]]) -> Dict:
    """Run a single (config, seed) task in a worker process.

    Parameters
    ----------
    task : (cfg_name, cfg, seed, result_dir, gpu_id)

    Returns
    -------
    dict with keys: cfg_name, seed, status, elapsed, result_dir, error
    """
    cfg_name, cfg, seed, result_dir, gpu_id = task

    # Pin to a specific GPU if provided
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

        # Import inside worker to avoid pickling CUDA state
        from experiments.poisson1d.vanilla_pinn import run_single_experiment
        seed_dir = run_single_experiment(
            cfg_with_seed, cfg_result_dir, suppress_stdout=True,
        )
        elapsed = time.time() - t0
        return {
            "cfg_name": cfg_name,
            "seed": seed,
            "status": "ok",
            "elapsed": elapsed,
            "result_dir": str(seed_dir),
            "error": None,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "cfg_name": cfg_name,
            "seed": seed,
            "status": "error",
            "elapsed": elapsed,
            "result_dir": str(cfg_result_dir),
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


# ---------------------------------------------------------------------------
# Task expansion: flatten configs (including multi-seed) into (cfg, seed) pairs
# ---------------------------------------------------------------------------

def _expand_tasks(
    selected: Dict[str, Poisson1DVanillaPINNConfig],
    result_dir: Path,
    gpus: List[int],
) -> List[Tuple[str, Poisson1DVanillaPINNConfig, int, Path, Optional[int]]]:
    """Expand configs into per-seed tasks, assigning GPUs round-robin."""
    tasks = []
    gpu_idx = 0
    for cfg_name, cfg in selected.items():
        seeds = [cfg.seed] if isinstance(cfg.seed, int) else list(cfg.seed)
        for s in seeds:
            gpu = gpus[gpu_idx % len(gpus)] if gpus else None
            tasks.append((cfg_name, cfg, s, result_dir, gpu))
            gpu_idx += 1
    return tasks


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _write_summary(result_dir: Path, results: List[Dict], wall_time: float):
    """Write a markdown summary of the batch run."""
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "error"]

    lines = [
        "# Batch Run Summary",
        "",
        f"- **Total tasks**: {len(results)}",
        f"- **Succeeded**: {len(ok)}",
        f"- **Failed**: {len(failed)}",
        f"- **Wall time**: {wall_time:.1f}s ({wall_time / 60:.1f}min)",
        f"- **Total CPU time**: {sum(r['elapsed'] for r in results):.1f}s",
        f"- **Speedup**: {sum(r['elapsed'] for r in results) / max(wall_time, 1e-3):.2f}x",
        "",
        "## Results",
        "",
        "| Config | Seed | Status | Elapsed |",
        "|--------|------|--------|---------|",
    ]
    for r in sorted(results, key=lambda r: (r["cfg_name"], r["seed"])):
        status_icon = "ok" if r["status"] == "ok" else "FAIL"
        lines.append(
            f"| {r['cfg_name']} | {r['seed']} | {status_icon} | {r['elapsed']:.1f}s |"
        )

    if failed:
        lines += ["", "## Errors", ""]
        for r in failed:
            lines += [f"### {r['cfg_name']} / seed {r['seed']}", "",
                      f"```", r["error"], f"```", ""]

    (result_dir / "BATCH_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for 1D Poisson PINN experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Config names to run (e.g. cfg_0 cfg_5). Default: all.",
    )
    parser.add_argument(
        "--group", type=str, default=None,
        help="Run all configs belonging to a group (e.g. SampleNumTest).",
    )
    parser.add_argument(
        "--parallel", "-j", type=int, default=1,
        help="Number of parallel worker processes (default: 1 = sequential).",
    )
    parser.add_argument(
        "--gpus", nargs="*", type=int, default=None,
        help="GPU IDs to assign round-robin (e.g. 0 1). Default: auto-detect.",
    )
    parser.add_argument(
        "--problem", type=str, default="poisson1d",
    )
    parser.add_argument(
        "--framework", type=str, default="01_pinn",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available configs and exit.",
    )
    args = parser.parse_args()

    # Import registry
    from experiments.poisson1d.registry import REGISTRY, list_configs

    # --list: print table and exit
    if args.list:
        print(f"{'Name':<10} {'Optimizer':<10} {'Epochs':>8} {'n_colloc':>9} "
              f"{'Sampling':<10} {'Seeds':>6} {'Groups'}")
        print("-" * 80)
        for name in list_configs():
            cfg = REGISTRY[name]
            seeds = [cfg.seed] if isinstance(cfg.seed, int) else cfg.seed
            n_seeds = len(seeds)
            groups = ", ".join(cfg.exp_group or [])
            print(
                f"{name:<10} {cfg.optimizer:<10} {cfg.epochs:>8} {cfg.n_colloc:>9} "
                f"{cfg.benchmark_sampling_method:<10} {n_seeds:>6} {groups}"
            )
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
            print(f"Available groups: {sorted({g for c in REGISTRY.values() for g in (c.exp_group or [])})}")
            return
    else:
        selected = dict(REGISTRY)

    if not selected:
        print("No configs to run.")
        return

    # Result directory (shared test_case for the whole batch)
    result_dir = get_next_test_case_dir(args.problem, args.framework)
    result_dir.mkdir(parents=True, exist_ok=True)

    # GPU list
    gpus = args.gpus if args.gpus is not None else []

    # Expand to per-seed tasks
    tasks = _expand_tasks(selected, result_dir, gpus)

    print(f"\n{'=' * 60}")
    print(f" Batch run: {len(selected)} config(s), {len(tasks)} task(s)")
    print(f" Parallel workers: {args.parallel}")
    print(f" GPUs: {gpus if gpus else 'auto'}")
    print(f" Results: {result_dir}")
    print(f"{'=' * 60}\n")

    # Save batch config (which configs were run)
    batch_info = {
        "configs": list(selected.keys()),
        "parallel": args.parallel,
        "gpus": gpus,
        "n_tasks": len(tasks),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (result_dir / "batch_info.json").write_text(
        json.dumps(batch_info, indent=2), encoding="utf-8"
    )

    # Run
    t_start = time.time()
    results: List[Dict] = []

    if args.parallel <= 1:
        # Sequential: simple loop
        for i, task in enumerate(tasks):
            cfg_name, _, seed, _, _ = task
            print(f"[{i + 1}/{len(tasks)}] {cfg_name} / seed {seed} ...")
            result = _worker_run(task)
            results.append(result)
            status = "ok" if result["status"] == "ok" else "FAIL"
            print(f"  -> {status} ({result['elapsed']:.1f}s)")
    else:
        # Parallel: process pool
        pool = mp.Pool(processes=args.parallel)
        try:
            for i, result in enumerate(pool.imap_unordered(_worker_run, tasks)):
                results.append(result)
                status = "ok" if result["status"] == "ok" else "FAIL"
                print(f"[{i + 1}/{len(tasks)}] {result['cfg_name']} / seed {result['seed']} "
                      f"-> {status} ({result['elapsed']:.1f}s)")
        finally:
            pool.close()
            pool.join()

    wall_time = time.time() - t_start

    # Summary
    ok_count = sum(1 for r in results if r["status"] == "ok")
    fail_count = len(results) - ok_count
    print(f"\n{'=' * 60}")
    print(f" Batch complete: {ok_count}/{len(results)} succeeded, "
          f"{fail_count} failed, wall time {wall_time:.1f}s")
    print(f" Results: {result_dir}")
    print(f"{'=' * 60}")

    _write_summary(result_dir, results, wall_time)
    print(f" Summary written to: {result_dir / 'BATCH_SUMMARY.md'}")


if __name__ == "__main__":
    main()
