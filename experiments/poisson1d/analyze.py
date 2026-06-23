"""1D Poisson — Post-hoc analysis and visualization.

Loads saved experiment results and generates evaluation plots and metrics.

Usage:
    python experiments/poisson1d/analyze.py --case <test_case_index> [--problem poisson1d] [--framework 01_pinn]
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from unipinn.config.poisson1d import Poisson1DVanillaPINNConfig
from unipinn.nn.architectures import SimpleNN
from unipinn.utils.tensor import to_tensor, gradients
from unipinn.io.artifacts import SafeJSONEncoder
from unipinn.io.workspace import get_results_dir
from unipinn.pde.benchmarks.poisson1d import Poisson1DBenchmarkIndex, BasePoisson1DBenchmark

plt.rcParams.update({
    "font.size": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D Poisson Analysis")
    parser.add_argument("--case", type=int, required=True, help="Test case index")
    parser.add_argument("--problem", type=str, default="poisson1d")
    parser.add_argument("--framework", type=str, default="01_pinn")
    args = parser.parse_args()

    results_dir = get_results_dir(args.problem, args.framework)
    result_dir = results_dir / f"test_case_{args.case}"
    print(f"[Results loaded] {result_dir}")

    if not result_dir.exists():
        raise FileNotFoundError(f"Directory not found: {result_dir}")

    # Detect seed subdirectories
    seed_dirs = sorted([d for d in result_dir.glob("seed_*") if d.is_dir()])
    if not seed_dirs:
        print("[Fallback] No seed_* directories found. Treating root as single run.")
        seed_dirs = [result_dir]
    print(f"[Detected {len(seed_dirs)} run(s)]")

    all_seed_metrics = []

    for seed_dir in seed_dirs:
        print(f"\n{'=' * 40}\nProcessing: {seed_dir.name}\n{'=' * 40}")

        # Load config
        config_path = seed_dir / "config.json"
        if not config_path.exists():
            print(f"Skipping {seed_dir.name}: config.json missing.")
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = Poisson1DVanillaPINNConfig(**json.load(f))
        print(f"Config loaded | seed={cfg.seed}, epochs={cfg.epochs}, opt={cfg.optimizer}")

        # Environment
        dtype = torch.float64 if cfg.precision == "float64" else torch.float32
        torch.set_default_dtype(dtype)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        model_path = seed_dir / "model_main.pt"
        if not model_path.exists():
            print(f"Skipping {seed_dir.name}: model_main.pt missing.")
            continue
        model = SimpleNN(cfg.arch_config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        # Load predictions
        pred_data = np.load(seed_dir / "predictions.npz")
        x_eval_np_orig = pred_data["x_eval"].flatten()
        u_exact = pred_data["u_exact"].flatten()
        u_pred_orig = pred_data["u_pred"].flatten()

        # Load history
        history = None
        hist_path = seed_dir / "history.json"
        if hist_path.exists():
            with open(hist_path, "r", encoding="utf-8") as f:
                history = json.load(f)

        # Re-generate benchmark data for residual computation
        bench = Poisson1DBenchmarkIndex.get(cfg.benchmark_name, **cfg.benchmark_params)
        data = bench.generate(
            n_colloc=cfg.n_colloc, n_eval=cfg.n_eval,
            sampling_method=cfg.benchmark_sampling_method,
            seed=cfg.seed, lhs_criterion=cfg.benchmark_lhs,
        )

        x_col_t = to_tensor(np.sort(data["x_colloc"]), dtype=dtype, device=device).reshape(-1, 1).requires_grad_(True)
        x_eval_t = to_tensor(x_eval_np_orig, dtype=dtype, device=device).reshape(-1, 1).requires_grad_(True)

        with torch.no_grad():
            u_pred_col = model(x_col_t).cpu().numpy().flatten()
            u_pred_eval = model(x_eval_t).cpu().numpy().flatten()

        # PDE residual: f(x) = -u''(x)
        u_col_grad = model(x_col_t)
        f_pred_col = -gradients(gradients(u_col_grad, x_col_t)[0], x_col_t)[0].detach().cpu().numpy().flatten()

        u_eval_grad = model(x_eval_t)
        f_pred_eval = -gradients(gradients(u_eval_grad, x_eval_t)[0], x_eval_t)[0].detach().cpu().numpy().flatten()

        x_col_np = x_col_t.detach().cpu().numpy().flatten()
        x_eval_np = x_eval_t.detach().cpu().numpy().flatten()
        u_exact_col = bench.u(x_col_np).flatten()
        u_exact_eval = bench.u(x_eval_np).flatten()
        f_exact_col = bench.f(x_col_np).flatten()
        f_exact_eval = bench.f(x_eval_np).flatten()

        # Plot 1: Solution u(x)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(x_col_np, u_exact_col, linewidth=4, c="#2980b9", alpha=0.6, label="Exact")
        plt.scatter(x_col_np, u_pred_col, s=2, c="#ff4757", label="PINN")
        plt.xlabel("x"); plt.ylabel("u(x)"); plt.title("Solution @ Colloc")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(x_eval_np, u_exact_eval, linewidth=4, c="#2980b9", alpha=0.6, label="Exact")
        plt.scatter(x_eval_np, u_pred_eval, s=2, c="#ff4757", label="PINN")
        plt.xlabel("x"); plt.ylabel("u(x)"); plt.title("Solution @ Eval")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(seed_dir / "solution_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot 2: PDE residual f(x)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(x_col_np, f_exact_col, linewidth=4, c="#2980b9", alpha=0.6, label="Exact")
        plt.scatter(x_col_np, f_pred_col, s=2, c="#ff4757", label="PINN")
        plt.xlabel("x"); plt.ylabel("-d²u/dx²"); plt.title("PDE Residual @ Colloc")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(x_eval_np, f_exact_eval, linewidth=4, c="#2980b9", alpha=0.6, label="Exact")
        plt.scatter(x_eval_np, f_pred_eval, s=2, c="#ff4757", label="PINN")
        plt.xlabel("x"); plt.ylabel("-d²u/dx²"); plt.title("PDE Residual @ Eval")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(seed_dir / "residual_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot 3: Training loss
        if history and "loss" in history:
            plt.figure(figsize=(4, 4))
            loss_dict = history["loss"]
            epochs = history["epoch"]
            plt.semilogy(epochs, loss_dict.get("total", []), "-", linewidth=2, label="Total")
            if "pde" in loss_dict: plt.semilogy(epochs, loss_dict["pde"], "--", alpha=0.8, label="PDE")
            if "bc" in loss_dict: plt.semilogy(epochs, loss_dict["bc"], ":", alpha=0.8, label="BC")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training History")
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(seed_dir / "loss_plot.png", dpi=300, bbox_inches="tight")
            plt.close()

        # Metrics
        err = {
            "col_point": {
                "u": BasePoisson1DBenchmark.compute_metrics(u_exact_col, u_pred_col),
                "f": BasePoisson1DBenchmark.compute_metrics(f_exact_col, f_pred_col),
            },
            "eval_point": {
                "u": BasePoisson1DBenchmark.compute_metrics(u_exact_eval, u_pred_eval),
                "f": BasePoisson1DBenchmark.compute_metrics(f_exact_eval, f_pred_eval),
            },
        }
        with open(seed_dir / "eval.json", "w", encoding="utf-8") as f:
            json.dump(err, f, indent=2, cls=SafeJSONEncoder)
        print(f"Metrics saved to {seed_dir / 'eval.json'}")
        all_seed_metrics.append({"seed": cfg.seed, "metrics": err})

    # Multi-seed comparison
    if len(all_seed_metrics) > 1:
        print("\nGenerating multi-seed comparison...")
        comp_dict = {str(m["seed"]): m["metrics"] for m in all_seed_metrics}
        with open(result_dir / "metrics_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comp_dict, f, indent=2, cls=SafeJSONEncoder)

        if all_seed_metrics:
            metric_names = list(all_seed_metrics[0]["metrics"]["eval_point"]["u"].keys())
            n_cols_per_row = 3
            n_rows = (len(metric_names) + n_cols_per_row - 1) // n_cols_per_row
            n_total_cols = n_cols_per_row * 2
            fig, axes = plt.subplots(n_rows, n_total_cols, figsize=(18, n_rows * 3.2))
            axes = axes.flatten() if hasattr(axes, "flatten") else np.atleast_1d(axes)

            for i, metric in enumerate(metric_names):
                row = i // n_cols_per_row
                col_off = (i % n_cols_per_row) * 2
                ax_u = axes[row * n_total_cols + col_off]
                ax_f = axes[row * n_total_cols + col_off + 1]

                u_col = [m["metrics"]["col_point"]["u"][metric] for m in all_seed_metrics]
                u_eval = [m["metrics"]["eval_point"]["u"][metric] for m in all_seed_metrics]
                f_col = [m["metrics"]["col_point"]["f"][metric] for m in all_seed_metrics]
                f_eval = [m["metrics"]["eval_point"]["f"][metric] for m in all_seed_metrics]

                for ax, data_pairs, titles, colors in [
                    (ax_u, [u_col, u_eval], ["u_col", "u_eval"], ["#2ecc71", "#3498db"]),
                    (ax_f, [f_col, f_eval], ["f_col", "f_eval"], ["#e74c3c", "#9b59b6"]),
                ]:
                    bp = ax.boxplot(data_pairs, tick_labels=titles, patch_artist=True, widths=0.6)
                    for patch, color in zip(bp["boxes"], colors):
                        patch.set_facecolor(color); patch.set_alpha(0.6)
                    ax.set_title(f"{metric.upper()} ({titles[0].split('_')[0]})", fontsize=10, pad=6)
                    ax.grid(True, alpha=0.3, linestyle="--")
                    if metric != "pearson_correlation":
                        ax.set_yscale("log")

            for j in range(len(metric_names) * 2, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.savefig(result_dir / "metrics_boxplot.png", dpi=300, bbox_inches="tight")
            plt.close()

    print(f"\nAll {len(seed_dirs)} run(s) processed successfully.")
