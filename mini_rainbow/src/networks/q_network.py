"""Standard CNN Q-Network for Atari (Nature DQN architecture)."""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Standard Deep Q-Network with CNN encoder and fully connected head.

    Architecture follows the Nature DQN paper (Mnih et al., 2015):
        Conv2d(in_channels, 32, 8, stride=4) -> ReLU
        Conv2d(32, 64, 4, stride=2) -> ReLU
        Conv2d(64, 64, 3, stride=1) -> ReLU
        Flatten
        Linear(3136, 512) -> ReLU
        Linear(512, num_actions)
    """

    def __init__(self, in_channels: int, num_actions: int) -> None:
        """Initialize Q-Network.

        Args:
            in_channels: Number of input channels (e.g., 4 for frame stack).
            num_actions: Number of discrete actions.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size: for 84x84 input -> 7x7x64 = 3136
        self.head = nn.Sequential(
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
        return self.head(features)
