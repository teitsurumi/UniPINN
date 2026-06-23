"""Artifact saving utilities: models, configs, predictions, and history."""

import json
import torch
import numpy as np
from typing import Dict, Union
from pathlib import Path


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and torch types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super().default(obj)


def save_models(model_or_dict: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
                save_dir: Path):
    """Save model state dicts. Supports single model or dict of models.

    Files are named ``model_{key}.pt``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model_or_dict, dict):
        for key, mdl in model_or_dict.items():
            if hasattr(mdl, "state_dict"):
                torch.save(mdl.state_dict(), save_dir / f"model_{key}.pt")
    elif hasattr(model_or_dict, "state_dict"):
        torch.save(model_or_dict.state_dict(), save_dir / "model_main.pt")


def save_experiment_artifacts(save_dir: Path, config, trainer, data: dict,
                              u_pred: np.ndarray, console_logs: str = "",
                              comments: str = ""):
    """Save all experiment artifacts to a directory.

    Saves: config.json, model, predictions.npz, history.json, console_log.txt, COMMENTS.md.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from dataclasses import asdict

    # Config
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, cls=SafeJSONEncoder, ensure_ascii=False)

    # Model
    save_models({"main": trainer.model}, save_dir)

    # Predictions
    np.savez(
        save_dir / "predictions.npz",
        x_eval=data["x_eval"],
        u_exact=data["u_eval"].flatten(),
        u_pred=u_pred,
    )

    # History
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.history, f, indent=2, cls=SafeJSONEncoder)

    # Console log
    if console_logs:
        (save_dir / "console_log.txt").write_text(console_logs, encoding="utf-8")

    # Comments
    if comments:
        (save_dir / "COMMENTS.md").write_text(comments, encoding="utf-8")
