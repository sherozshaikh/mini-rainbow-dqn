"""Weights & Biases logging wrapper with graceful fallback."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class WandbLogger:
    """Wrapper around W&B with graceful fallback.

    If W&B is not configured or the import fails, logs a warning
    and falls back to no-op logging (metrics still printed to console
    via the standard Python logger in the trainer).
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize W&B logger.

        Args:
            cfg: Full Hydra config. Uses cfg.wandb section.
        """
        self._enabled = False
        self._run = None

        wandb_cfg = cfg.wandb

        if not wandb_cfg.enabled:
            logger.info("W&B logging disabled by config")
            return

        try:
            import wandb

            # Load WANDB_API_KEY from .env if not already in environment
            if not os.environ.get("WANDB_API_KEY"):
                env_file = Path.cwd() / ".env"
                if env_file.exists():
                    for line in env_file.read_text().splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, value = line.partition("=")
                            if key.strip() == "WANDB_API_KEY" and value.strip():
                                os.environ["WANDB_API_KEY"] = value.strip()
                                logger.info("Loaded WANDB_API_KEY from .env")
                                break

            # Determine run name
            run_name = cfg.get("run_name") or "mini-rainbow-run"

            # Convert config to plain dict for W&B
            config_dict = OmegaConf.to_container(cfg, resolve=True)

            # Finish any previous run before starting a new one
            if wandb.run is not None:
                wandb.finish()

            self._run = wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.get("entity"),
                name=run_name,
                config=config_dict,
                mode=wandb_cfg.mode,
            )
            self._enabled = True
            logger.info(f"W&B initialized: project={wandb_cfg.project}, run={run_name}")

        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}. Continuing without W&B.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric name -> value.
            step: Global step number.
        """
        if not self._enabled:
            return

        try:
            import wandb

            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"W&B log failed: {e}")

    def log_video(self, video_path: str, caption: str = "", step: int | None = None) -> None:
        """Log a video file to W&B.

        Args:
            video_path: Path to video file.
            caption: Video caption.
            step: Global step number.
        """
        if not self._enabled:
            return

        try:
            import wandb

            wandb.log(
                {"eval/video": wandb.Video(video_path, caption=caption, fps=30)},
                step=step,
            )
        except Exception as e:
            logger.warning(f"W&B video log failed: {e}")

    def finish(self) -> None:
        """Finish the W&B run."""
        if self._enabled and self._run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
