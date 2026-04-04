"""Dueling Q-Network for Atari (Wang et al., 2016)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """Dueling Deep Q-Network.

    Separates state-value and advantage streams:
        Q(s, a) = V(s) + (A(s, a) - mean_a(A(s, a)))

    Uses the same CNN encoder as the standard Nature DQN.
    """

    def __init__(self, in_channels: int, num_actions: int) -> None:
        """Initialize Dueling Q-Network.

        Args:
            in_channels: Number of input channels (e.g., 4 for frame stack).
            num_actions: Number of discrete actions.
        """
        super().__init__()
        self.num_actions = num_actions

        # Shared CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, C, H, W), uint8 or float.

        Returns:
            Q-values of shape (batch, num_actions).
        """
        # Normalize pixel values to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        features = self.encoder(x)

        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, num_actions)

        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
