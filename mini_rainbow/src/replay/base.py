"""Abstract base class for replay buffers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseReplayBuffer(ABC):
    """Abstract base replay buffer defining the common interface.

    All replay buffer implementations must inherit from this class
    and implement add() and sample().
    """

    def __init__(self, buffer_size: int) -> None:
        """Initialize buffer.

        Args:
            buffer_size: Maximum number of transitions to store.
        """
        self.buffer_size = buffer_size
        self.size = 0
        self.pos = 0

    @abstractmethod
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the buffer.

        Args:
            state: Current observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next observation.
            done: Whether episode terminated.
        """

    @abstractmethod
    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys: states, actions, rewards, next_states, dones.
            May also include: indices, weights (for PER).
        """

    def __len__(self) -> int:
        return self.size
