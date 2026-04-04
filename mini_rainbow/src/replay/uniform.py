"""Uniform (standard) experience replay buffer."""

from __future__ import annotations

from typing import Any

import numpy as np

from mini_rainbow.src.replay.base import BaseReplayBuffer


class UniformReplayBuffer(BaseReplayBuffer):
    """Standard replay buffer with uniform random sampling.

    Stores transitions in pre-allocated numpy arrays for memory efficiency.
    Uses uint8 for states to minimize memory (1M transitions * 4*84*84 = ~28GB
    at float32, but only ~7GB at uint8).
    """

    def __init__(self, buffer_size: int, obs_shape: tuple[int, ...]) -> None:
        """Initialize uniform replay buffer.

        Args:
            buffer_size: Maximum number of transitions.
            obs_shape: Shape of observations, e.g. (4, 84, 84).
        """
        super().__init__(buffer_size)

        self.states = np.zeros((buffer_size, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((buffer_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition.

        Args:
            state: Current observation (C, H, W) uint8.
            action: Action taken.
            reward: Reward received.
            next_state: Next observation (C, H, W) uint8.
            done: Whether episode terminated.
        """
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample a uniform random batch.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with numpy arrays for states, actions, rewards,
            next_states, dones.
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "indices": indices,
            "weights": np.ones(batch_size, dtype=np.float32),  # uniform weights
        }
