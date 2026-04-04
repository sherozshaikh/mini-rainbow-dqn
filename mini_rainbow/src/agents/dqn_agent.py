"""DQN Agent supporting standard DQN, Double DQN, Dueling, and PER."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from mini_rainbow.src.networks.dueling_q_network import DuelingQNetwork
from mini_rainbow.src.networks.q_network import QNetwork
from mini_rainbow.src.replay.base import BaseReplayBuffer
from mini_rainbow.src.replay.prioritized import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


class DQNAgent:
    """Unified DQN agent.

    Supports all variants through config flags:
        - Standard DQN (default)
        - Double DQN (agent.double_dqn=true)
        - Dueling architecture (agent.dueling=true)
        - Prioritized replay (agent.per=true, requires PER buffer)

    The agent owns:
        - Online and target networks
        - Epsilon-greedy policy
        - Optimizer
        - Learning logic (loss computation + backprop)
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        num_actions: int,
        replay_buffer: BaseReplayBuffer,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        """Initialize DQN agent.

        Args:
            obs_shape: Observation shape, e.g. (4, 84, 84).
            num_actions: Number of discrete actions.
            replay_buffer: Replay buffer instance (uniform or prioritized).
            cfg: Full Hydra config (needs cfg.agent section).
            device: Torch device.
        """
        self.num_actions = num_actions
        self.replay_buffer = replay_buffer
        self.device = device
        self.agent_cfg = cfg.agent
        self.training_cfg = cfg.training

        # Build networks
        in_channels = obs_shape[0]
        if self.agent_cfg.dueling:
            self.online_net = DuelingQNetwork(in_channels, num_actions).to(device)
        else:
            self.online_net = QNetwork(in_channels, num_actions).to(device)

        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()
        # Freeze target network
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=self.agent_cfg.learning_rate,
            eps=self.agent_cfg.adam_eps,
        )

        # Epsilon schedule
        self.epsilon_start = self.agent_cfg.epsilon_start
        self.epsilon_end = self.agent_cfg.epsilon_end
        self.epsilon_decay_steps = self.agent_cfg.epsilon_decay_steps

        # Loss
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # Huber loss, per-element

        # Step counter for epsilon decay
        self._step = 0

        # Log network type
        net_type = "Dueling" if self.agent_cfg.dueling else "Standard"
        variant = "Double DQN" if self.agent_cfg.double_dqn else "DQN"
        logger.info(f"Agent initialized: {net_type} {variant}, PER={self.agent_cfg.per}")

    @property
    def epsilon(self) -> float:
        """Current epsilon value (linearly decayed)."""
        fraction = min(1.0, self._step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def act(self, state: np.ndarray, epsilon: float | None = None) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Observation array of shape (C, H, W).
            epsilon: Override epsilon. If None, use internal schedule.

        Returns:
            Selected action index.
        """
        eps = epsilon if epsilon is not None else self.epsilon

        if np.random.random() < eps:
            return np.random.randint(self.num_actions)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.uint8, device=self.device)
            state_t = state_t.unsqueeze(0)  # add batch dim
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer.

        Args:
            state: Current observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next observation.
            done: Whether episode ended.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> dict[str, float] | None:
        """Sample batch and perform one gradient step.

        Returns:
            Dictionary with loss and mean Q-value, or None if buffer too small.
        """
        if len(self.replay_buffer) < self.training_cfg.learning_starts:
            return None

        self._step += 1
        batch = self.replay_buffer.sample(self.training_cfg.batch_size)

        # Convert to tensors
        states = torch.as_tensor(batch["states"], dtype=torch.uint8, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.uint8, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)

        # Current Q-values: Q(s, a)
        q_values = self.online_net(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if self.agent_cfg.double_dqn:
                # Double DQN: select action with online net, evaluate with target
                next_q_online = self.online_net(next_states)
                best_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.target_net(next_states)
                next_q = next_q_target.gather(1, best_actions).squeeze(1)
            else:
                # Standard DQN: max Q from target network
                next_q = self.target_net(next_states).max(dim=1).values

            target_q = rewards + self.agent_cfg.gamma * next_q * (1.0 - dones)

        # Compute element-wise Huber loss
        td_errors = current_q - target_q
        element_loss = self.loss_fn(current_q, target_q)

        # Weight by importance-sampling weights (uniform = all ones)
        loss = (element_loss * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.agent_cfg.max_grad_norm)
        self.optimizer.step()

        # Update priorities for PER
        if self.agent_cfg.per and isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            self.replay_buffer.update_priorities(
                batch["indices"],
                td_errors.abs().detach().cpu().numpy(),
            )

        return {
            "loss": loss.item(),
            "mean_q": current_q.mean().item(),
            "max_q": current_q.max().item(),
            "epsilon": self.epsilon,
        }

    def update_target_network(self) -> None:
        """Hard copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        logger.debug("Target network updated")

    def get_state(self) -> dict[str, Any]:
        """Get agent state for checkpointing.

        Returns:
            Dictionary with network states, optimizer state, epsilon info.
        """
        return {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
            "epsilon": self.epsilon,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore agent state from checkpoint.

        Args:
            state: Dictionary from get_state() or load_checkpoint().
        """
        self.online_net.load_state_dict(state["online_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._step = state["step"]
        logger.info(f"Agent state loaded (step={self._step})")
