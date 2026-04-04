"""Deterministic evaluation with optional video recording."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from mini_rainbow.src.agents.dqn_agent import DQNAgent
from mini_rainbow.src.envs.atari_env import make_atari_env

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs deterministic evaluation episodes.

    Creates a separate eval environment (no reward clipping, no episodic life).
    Optionally records videos of evaluation episodes.
    """

    def __init__(self, agent: DQNAgent, cfg: DictConfig) -> None:
        """Initialize evaluator.

        Args:
            agent: Trained DQN agent.
            cfg: Full Hydra config.
        """
        self.agent = agent
        self.cfg = cfg
        self.video_dir = Path(cfg.video_dir) if cfg.training.record_video else None

    def evaluate(self, step: int | None = None) -> float:
        """Run evaluation episodes and return mean reward.

        Args:
            step: Current training step (used for video folder naming).

        Returns:
            Mean reward over evaluation episodes.
        """
        num_episodes = self.cfg.training.eval_episodes

        # Create video subdirectory for this evaluation
        video_dir = None
        if self.video_dir is not None and step is not None:
            video_dir = str(self.video_dir / f"step_{step}")
            Path(video_dir).mkdir(parents=True, exist_ok=True)

        # Create a fresh eval environment
        eval_env = make_atari_env(
            cfg=self.cfg.env,
            seed=self.cfg.seed + 1000,  # different seed from training
            video_dir=video_dir,
            eval_mode=True,
        )

        rewards = []
        for ep in range(num_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                # Greedy action (epsilon=0)
                action = self.agent.act(state, epsilon=0.0)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state

            rewards.append(episode_reward)
            logger.debug(f"Eval episode {ep + 1}/{num_episodes}: reward={episode_reward:.1f}")

        eval_env.close()

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        logger.info(
            f"Evaluation ({num_episodes} episodes): mean={mean_reward:.2f}, std={std_reward:.2f}"
        )

        return mean_reward
