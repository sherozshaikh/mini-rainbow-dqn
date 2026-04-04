"""Prioritized Experience Replay (PER) buffer (Schaul et al., 2016).

Uses a sum-tree data structure for efficient O(log N) priority sampling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mini_rainbow.src.replay.base import BaseReplayBuffer


class SumTree:
    """Binary sum-tree for efficient priority-based sampling.

    Leaf nodes store priorities. Internal nodes store the sum of their children.
    Allows O(log N) sampling proportional to priority and O(log N) updates.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Tree has (2 * capacity - 1) nodes: capacity leaves + (capacity - 1) internals
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def update(self, leaf_idx: int, priority: float) -> None:
        """Update priority of a leaf node and propagate change up."""
        tree_idx = leaf_idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def sample(self, value: float) -> int:
        """Sample a leaf index proportional to priorities.

        Args:
            value: Random value in [0, total_priority).

        Returns:
            Leaf index (data index, 0-based).
        """
        idx = 0  # root
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                # Reached leaf level
                break
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        leaf_idx = idx - (self.capacity - 1)
        return leaf_idx

    @property
    def total(self) -> float:
        """Total priority (root node value)."""
        return self.tree[0]

    def get_priority(self, leaf_idx: int) -> float:
        """Get priority of a specific leaf."""
        return self.tree[leaf_idx + self.capacity - 1]


class PrioritizedReplayBuffer(BaseReplayBuffer):
    """Prioritized Experience Replay buffer.

    Samples transitions with probability proportional to their TD-error priority.
    Uses importance-sampling weights to correct for the non-uniform sampling bias.

    Args:
        buffer_size: Maximum number of transitions.
        obs_shape: Observation shape, e.g. (4, 84, 84).
        alpha: Priority exponent. 0 = uniform, 1 = full prioritization.
        beta_start: Initial importance-sampling exponent.
        beta_end: Final importance-sampling exponent (annealed linearly).
        beta_anneal_steps: Steps over which to anneal beta.
        prior_eps: Small constant added to priorities to prevent zero probability.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple[int, ...],
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 5_000_000,
        prior_eps: float = 1e-6,
    ) -> None:
        super().__init__(buffer_size)

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.prior_eps = prior_eps
        self._step = 0

        self.tree = SumTree(buffer_size)
        self.max_priority = 1.0  # initial max priority for new transitions

        # Data storage
        self.states = np.zeros((buffer_size, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((buffer_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)

    @property
    def beta(self) -> float:
        """Current importance-sampling exponent (linearly annealed)."""
        fraction = min(1.0, self._step / max(1, self.beta_anneal_steps))
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition with max priority (ensures it gets sampled at least once)."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        # New transitions get max priority
        priority = self.max_priority**self.alpha
        self.tree.update(self.pos, priority)

        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample a batch proportional to priorities.

        Args:
            batch_size: Number of transitions.

        Returns:
            Dictionary with: states, actions, rewards, next_states, dones,
            indices (for priority update), weights (importance-sampling).
        """
        self._step += 1
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        # Divide total priority into equal segments and sample one from each
        segment_size = self.tree.total / batch_size

        for i in range(batch_size):
            low = segment_size * i
            high = segment_size * (i + 1)
            value = np.random.uniform(low, high)
            idx = self.tree.sample(value)
            # Clamp to valid range
            idx = min(idx, self.size - 1)
            indices[i] = idx
            priorities[i] = self.tree.get_priority(idx)

        # Compute importance-sampling weights
        # w_i = (N * P(i))^{-beta} / max_j(w_j)
        probs = priorities / self.tree.total
        probs = np.clip(probs, 1e-10, None)  # avoid division by zero

        weights = (self.size * probs) ** (-self.beta)
        weights = weights / weights.max()  # normalize to [0, 1]
        weights = weights.astype(np.float32)

        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "indices": indices,
            "weights": weights,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on new TD errors.

        Args:
            indices: Buffer indices to update.
            td_errors: Absolute TD errors for those transitions.
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.prior_eps) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, abs(td_error) + self.prior_eps)
