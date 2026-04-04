"""Checkpoint save/load utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    step: int,
    online_net_state: dict[str, Any],
    target_net_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    epsilon: float,
    best_eval_reward: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint.

    Args:
        path: File path to save to.
        step: Current training step.
        online_net_state: Online network state_dict.
        target_net_state: Target network state_dict.
        optimizer_state: Optimizer state_dict.
        epsilon: Current epsilon value.
        best_eval_reward: Best evaluation reward seen so far.
        extra: Any additional data to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "online_net": online_net_state,
        "target_net": target_net_state,
        "optimizer": optimizer_state,
        "epsilon": epsilon,
        "best_eval_reward": best_eval_reward,
    }
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path} (step={step})")


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: File path to load from.
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    logger.info(f"Checkpoint loaded from {path} (step={checkpoint['step']})")
    return checkpoint
