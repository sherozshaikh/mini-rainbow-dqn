"""Training loop orchestrator."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from mini_rainbow.src.agents.dqn_agent import DQNAgent
from mini_rainbow.src.evaluation.evaluator import Evaluator
from mini_rainbow.src.logging.wandb_logger import WandbLogger
from mini_rainbow.src.utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the DQN training loop.

    Responsibilities:
        - Step environment and agent
        - Call learn() on agent
        - Periodically update target network
        - Trigger evaluation
        - Log metrics
        - Save checkpoints
    """

    def __init__(
        self,
        env,
        agent: DQNAgent,
        evaluator: Evaluator,
        wandb_logger: WandbLogger,
        cfg: DictConfig,
    ) -> None:
        """Initialize trainer.

        Args:
            env: Training environment (wrapped Atari).
            agent: DQN agent instance.
            evaluator: Evaluator for periodic evaluation.
            wandb_logger: Weights & Biases logger.
            cfg: Full Hydra config.
        """
        self.env = env
        self.agent = agent
        self.evaluator = evaluator
        self.wandb_logger = wandb_logger
        self.cfg = cfg
        self.training_cfg = cfg.training

        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_eval_reward = -float("inf")

    def train(self) -> None:
        """Run the full training loop."""
        total_steps = self.training_cfg.total_steps
        target_update_freq = self.cfg.agent.target_update_freq
        eval_freq = self.training_cfg.eval_freq
        save_freq = self.training_cfg.save_freq
        log_freq = self.training_cfg.log_freq

        # Episode tracking
        episode_reward = 0.0
        episode_length = 0
        episode_count = 0
        episode_rewards: list[float] = []

        state, _ = self.env.reset()
        start_time = time.time()

        logger.info(f"Starting training for {total_steps} steps")
        logger.info(
            f"  eval_freq={eval_freq}, save_freq={save_freq}, "
            f"target_update_freq={target_update_freq}"
        )

        for step in range(1, total_steps + 1):
            # 1. Select action
            action = self.agent.act(state)

            # 2. Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # 3. Store transition
            self.agent.store(state, action, reward, next_state, done)

            # 4. Learn
            learn_metrics = self.agent.learn()

            # 5. Update target network
            if step % target_update_freq == 0:
                self.agent.update_target_network()

            # Track episode stats
            episode_reward += reward
            episode_length += 1

            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)

                # Log episode metrics
                if learn_metrics is not None:
                    self.wandb_logger.log(
                        {
                            "train/episode_reward": episode_reward,
                            "train/episode_length": episode_length,
                            "train/episode_count": episode_count,
                            "train/epsilon": self.agent.epsilon,
                        },
                        step=step,
                    )

                # Reset
                episode_reward = 0.0
                episode_length = 0
                state, _ = self.env.reset()
            else:
                state = next_state

            # 6. Log training metrics
            if learn_metrics is not None and step % log_freq == 0:
                elapsed = time.time() - start_time
                fps = step / elapsed
                recent_rewards = episode_rewards[-100:] if episode_rewards else [0.0]
                self.wandb_logger.log(
                    {
                        "train/loss": learn_metrics["loss"],
                        "train/mean_q": learn_metrics["mean_q"],
                        "train/max_q": learn_metrics["max_q"],
                        "train/epsilon": learn_metrics["epsilon"],
                        "train/fps": fps,
                        "train/avg_reward_100": np.mean(recent_rewards),
                        "train/step": step,
                    },
                    step=step,
                )
                logger.info(
                    f"Step {step}/{total_steps} | "
                    f"Loss: {learn_metrics['loss']:.4f} | "
                    f"Eps: {learn_metrics['epsilon']:.4f} | "
                    f"Avg100: {np.mean(recent_rewards):.2f} | "
                    f"FPS: {fps:.0f}"
                )

            # 7. Evaluation
            if step % eval_freq == 0:
                eval_reward = self.evaluator.evaluate(step=step)
                self.wandb_logger.log(
                    {"eval/mean_reward": eval_reward},
                    step=step,
                )
                logger.info(f"Eval at step {step}: mean_reward={eval_reward:.2f}")

                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self._save_checkpoint(step, tag="best")

            # 8. Periodic checkpoint
            if step % save_freq == 0:
                self._save_checkpoint(step, tag="latest")

        # Final checkpoint
        self._save_checkpoint(total_steps, tag="final")
        logger.info("Training complete!")

    def _save_checkpoint(self, step: int, tag: str = "latest") -> None:
        """Save a training checkpoint.

        Args:
            step: Current training step.
            tag: Filename tag (e.g., 'latest', 'best', 'final').
        """
        agent_state = self.agent.get_state()
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        save_checkpoint(
            path=path,
            step=step,
            online_net_state=agent_state["online_net"],
            target_net_state=agent_state["target_net"],
            optimizer_state=agent_state["optimizer"],
            epsilon=agent_state["epsilon"],
            best_eval_reward=self.best_eval_reward,
        )
