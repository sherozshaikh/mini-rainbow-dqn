"""Training entry point. Run with: python -m mini_rainbow.scripts.train"""

from __future__ import annotations

import logging
import sys

try:
    import torch
except ImportError:
    print(
        "\nERROR: PyTorch is not installed.\n"
        "Install it for your CUDA version before running training:\n\n"
        "  uv pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1\n"
        "  uv pip install torch --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8\n"
        "  uv pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU only\n\n"
        "Or use the Makefile shortcut:  make install-torch-cu121\n"
    )
    sys.exit(1)

import hydra
from omegaconf import DictConfig, OmegaConf

from mini_rainbow.src.agents.dqn_agent import DQNAgent
from mini_rainbow.src.envs.atari_env import make_atari_env
from mini_rainbow.src.evaluation.evaluator import Evaluator
from mini_rainbow.src.logging.wandb_logger import WandbLogger
from mini_rainbow.src.replay.prioritized import PrioritizedReplayBuffer
from mini_rainbow.src.replay.uniform import UniformReplayBuffer
from mini_rainbow.src.training.trainer import Trainer
from mini_rainbow.src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device.

    Args:
        device_str: 'auto', 'cpu', 'cuda', or 'cuda:N'.

    Returns:
        torch.device instance.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-detected CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
    else:
        device = torch.device(device_str)
    return device


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Seed
    set_seed(cfg.seed)

    # Device
    device = resolve_device(cfg.device)

    # Environment
    env = make_atari_env(cfg=cfg.env, seed=cfg.seed)
    obs_shape = env.observation_space.shape  # (C, H, W)
    num_actions = env.action_space.n
    logger.info(f"Environment: {cfg.env.name}")
    logger.info(f"  obs_shape={obs_shape}, num_actions={num_actions}")

    # Replay buffer
    if cfg.agent.per:
        replay_buffer = PrioritizedReplayBuffer(
            buffer_size=cfg.replay.buffer_size,
            obs_shape=obs_shape,
            alpha=cfg.replay.alpha,
            beta_start=cfg.replay.beta_start,
            beta_end=cfg.replay.beta_end,
            beta_anneal_steps=cfg.replay.beta_anneal_steps,
            prior_eps=cfg.replay.prior_eps,
        )
        logger.info("Using PrioritizedReplayBuffer")
    else:
        replay_buffer = UniformReplayBuffer(
            buffer_size=cfg.replay.buffer_size,
            obs_shape=obs_shape,
        )
        logger.info("Using UniformReplayBuffer")

    # Agent
    agent = DQNAgent(
        obs_shape=obs_shape,
        num_actions=num_actions,
        replay_buffer=replay_buffer,
        cfg=cfg,
        device=device,
    )

    # Logger
    wandb_logger = WandbLogger(cfg)

    # Evaluator
    evaluator = Evaluator(agent=agent, cfg=cfg)

    # Trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        evaluator=evaluator,
        wandb_logger=wandb_logger,
        cfg=cfg,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        wandb_logger.finish()
        env.close()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
