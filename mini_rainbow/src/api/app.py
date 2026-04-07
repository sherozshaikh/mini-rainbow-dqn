"""FastAPI inference endpoint for trained DQN agent."""

from __future__ import annotations

import logging

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mini_rainbow.src.networks.dueling_q_network import DuelingQNetwork
from mini_rainbow.src.networks.q_network import QNetwork
from mini_rainbow.src.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

app = FastAPI(title="Mini-Rainbow DQN Inference API", version="1.0.0")

# Global model state (loaded on startup)
_model: torch.nn.Module | None = None
_device: torch.device = torch.device("cpu")


class ActRequest(BaseModel):
    """Request body for /act endpoint."""

    state: list[list[list[list[float]]]]  # Shape: (C, H, W) nested lists


class ActResponse(BaseModel):
    """Response body for /act endpoint."""

    action: int
    q_values: list[float]


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    model_loaded: bool


def load_model(
    checkpoint_path: str,
    num_actions: int = 4,
    in_channels: int = 4,
    dueling: bool = False,
    device: str = "cpu",
) -> None:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        num_actions: Number of actions (4 for Breakout).
        in_channels: Input channels (4 for frame stack).
        dueling: Whether to use dueling architecture.
        device: Device string.
    """
    global _model, _device

    _device = torch.device(device)

    if dueling:
        _model = DuelingQNetwork(in_channels, num_actions).to(_device)
    else:
        _model = QNetwork(in_channels, num_actions).to(_device)

    checkpoint = load_checkpoint(checkpoint_path, device=_device)
    _model.load_state_dict(checkpoint["online_net"])
    _model.eval()

    logger.info(f"Model loaded from {checkpoint_path} (step={checkpoint['step']})")


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
    )


@app.post("/act", response_model=ActResponse)
def act(request: ActRequest):
    """Select action for given state.

    The state should be a (C, H, W) array with pixel values in [0, 255].
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        state = np.array(request.state, dtype=np.uint8)
        state_t = torch.as_tensor(state, dtype=torch.uint8, device=_device).unsqueeze(0)

        with torch.no_grad():
            q_values = _model(state_t)

        action = q_values.argmax(dim=1).item()
        q_list = q_values.squeeze(0).cpu().tolist()

        return ActResponse(action=action, q_values=q_list)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
