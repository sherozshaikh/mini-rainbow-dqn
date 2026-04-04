"""Weights & Biases logging wrapper with graceful fallback."""

from __future__ import annotations

import logging
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

            # Determine run name
            run_name = cfg.get("run_name") or "mini-rainbow-run"

            # Convert config to plain dict for W&B
            config_dict = OmegaConf.to_container(cfg, resolve=True)

            self._run = wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.get("entity"),
                name=run_name,
                config=config_dict,
                mode=wandb_cfg.mode,
                reinit=True,
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
