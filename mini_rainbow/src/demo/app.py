"""Live demo: DQN vs Rainbow-Lite playing Breakout side-by-side.

Two trained agents play simultaneously. Frames are staggered (alternate stepping)
so CPU usage equals one game at 30 FPS. Each game renders at ~15 FPS.
Prometheus-compatible /metrics endpoint included for optional monitoring.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time

import ale_py  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from gymnasium.wrappers import AtariPreprocessing
from PIL import Image
from pydantic import BaseModel

from mini_rainbow.src.demo.page import HTML_PAGE
from mini_rainbow.src.networks.dueling_q_network import DuelingQNetwork
from mini_rainbow.src.networks.q_network import QNetwork
from mini_rainbow.src.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Mini-Rainbow DQN Live Demo", version="0.1.0")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_device: torch.device = torch.device("cpu")
_models: dict[str, torch.nn.Module] = {}
_agents: dict[str, dict] = {}
_lock = threading.Lock()
_running = False


# ---------------------------------------------------------------------------
# Frame-stack helper
# ---------------------------------------------------------------------------
class FrameStack:
    """Minimal frame stacker for inference."""

    def __init__(self, num_stack: int = 4):
        self.num_stack = num_stack
        self.frames: list[np.ndarray] = []

    def reset(self, obs: np.ndarray) -> np.ndarray:
        frame = self._proc(obs)
        self.frames = [frame] * self.num_stack
        return np.stack(self.frames, axis=0)

    def step(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(self._proc(obs))
        if len(self.frames) > self.num_stack:
            self.frames.pop(0)
        return np.stack(self.frames, axis=0)

    @staticmethod
    def _proc(obs: np.ndarray) -> np.ndarray:
        return obs.squeeze(-1).astype(np.uint8) if obs.ndim == 3 else obs.astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-agent state
# ---------------------------------------------------------------------------
def _make_agent_state(name: str) -> dict:
    return {
        "name": name,
        "frame_b64": "",
        "q_values": [0.0, 0.0, 0.0, 0.0],
        "score": 0.0,
        "episode": 0,
        "step": 0,
        "total_episodes": 0,
        "total_steps": 0,
        "total_reward": 0.0,
        "avg_score_10": 0.0,
        "aps": 0.0,
        "recent_scores": [],
        "best_score": 0.0,
    }


MAX_EPISODE_STEPS = 10000  # cap episodes to prevent infinite loops


def _make_env() -> gym.Env:
    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode="rgb_array")
    return AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )


# ---------------------------------------------------------------------------
# Game loop (both agents, staggered)
# ---------------------------------------------------------------------------
def _game_loop() -> None:
    """Run both agents playing Breakout with staggered frame stepping."""
    global _running
    _running = True

    agent_names = list(_models.keys())
    envs = {name: _make_env() for name in agent_names}
    stackers = {name: FrameStack(4) for name in agent_names}
    states = {}
    ep_rewards = {name: 0.0 for name in agent_names}
    ep_steps = {name: 0 for name in agent_names}
    ep_counts = {name: 0 for name in agent_names}
    ep_starts = {name: time.time() for name in agent_names}
    dones = {name: True for name in agent_names}
    prev_lives = {name: 5 for name in agent_names}
    no_reward_steps = {name: 0 for name in agent_names}

    tick = 0
    while _running:
        for i, name in enumerate(agent_names):
            # Stagger: agent 0 on even ticks, agent 1 on odd ticks
            if tick % len(agent_names) != i:
                continue

            model = _models[name]
            env = envs[name]
            stacker = stackers[name]

            # Reset if needed
            if dones[name]:
                if ep_counts[name] > 0:
                    elapsed = max(time.time() - ep_starts[name], 0.01)
                    with _lock:
                        a = _agents[name]
                        a["total_reward"] += ep_rewards[name]
                        a["recent_scores"].append(ep_rewards[name])
                        if len(a["recent_scores"]) > 50:
                            a["recent_scores"] = a["recent_scores"][-50:]
                        a["avg_score_10"] = float(np.mean(a["recent_scores"][-10:]))
                        a["aps"] = ep_steps[name] / elapsed
                        a["best_score"] = max(a["best_score"], ep_rewards[name])
                        logger.info(f"[{name}] ep {ep_counts[name]}: score={ep_rewards[name]:.0f}")

                raw, _ = env.reset()
                states[name] = stacker.reset(raw)
                ep_rewards[name] = 0.0
                ep_steps[name] = 0
                ep_counts[name] += 1
                ep_starts[name] = time.time()
                dones[name] = False
                prev_lives[name] = 5
                no_reward_steps[name] = 0

                # Fire to start the game
                raw, _, _, _, _ = env.step(1)
                states[name] = stacker.step(raw)

            # Act
            with torch.no_grad():
                s = torch.as_tensor(states[name], dtype=torch.uint8, device=_device).unsqueeze(0)
                qv = model(s)
                action = qv.argmax(dim=1).item()
                q_list = qv.squeeze(0).cpu().tolist()

            raw, reward, term, trunc, _ = env.step(action)
            states[name] = stacker.step(raw)

            # Detect life loss -> press FIRE to launch next ball
            current_lives = env.unwrapped.ale.lives()
            if current_lives < prev_lives[name] and not term:
                raw, _, _, _, _ = env.step(1)  # FIRE
                states[name] = stacker.step(raw)
            prev_lives[name] = current_lives

            # Cap episode length to prevent stuck agents
            dones[name] = term or trunc or ep_steps[name] >= MAX_EPISODE_STEPS
            ep_rewards[name] += reward
            ep_steps[name] += 1

            # Render frame
            rgb = env.render()
            if rgb is not None:
                img = Image.fromarray(rgb).resize((240, 320), Image.NEAREST)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=65)
                fb64 = base64.b64encode(buf.getvalue()).decode("ascii")

                with _lock:
                    a = _agents[name]
                    a["frame_b64"] = fb64
                    a["q_values"] = q_list
                    a["score"] = ep_rewards[name]
                    a["episode"] = ep_counts[name]
                    a["step"] = ep_steps[name]
                    a["total_episodes"] = ep_counts[name]
                    a["total_steps"] += 1

        tick += 1
        time.sleep(1 / 30)  # 30 ticks/sec total, 15 per agent

    for env in envs.values():
        env.close()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    global _device

    device_str = os.environ.get("DEVICE", "cpu")
    _device = torch.device(device_str)

    # Load DQN (standard Q-network)
    dqn_path = os.environ.get("DQN_CHECKPOINT", "checkpoints/stage1_dqn_best.pt")
    if os.path.exists(dqn_path):
        net = QNetwork(in_channels=4, num_actions=4).to(_device)
        ckpt = load_checkpoint(dqn_path, device=_device)
        net.load_state_dict(ckpt["online_net"])
        net.eval()
        _models["DQN"] = net
        _agents["DQN"] = _make_agent_state("DQN")
        logger.info(f"Loaded DQN from {dqn_path} (step={ckpt['step']})")

    # Load Rainbow-Lite (dueling network)
    rl_path = os.environ.get("RAINBOW_CHECKPOINT", "checkpoints/stage2_rainbow_lite_best.pt")
    if os.path.exists(rl_path):
        net = DuelingQNetwork(in_channels=4, num_actions=4).to(_device)
        ckpt = load_checkpoint(rl_path, device=_device)
        net.load_state_dict(ckpt["online_net"])
        net.eval()
        _models["Rainbow-Lite"] = net
        _agents["Rainbow-Lite"] = _make_agent_state("Rainbow-Lite")
        logger.info(f"Loaded Rainbow-Lite from {rl_path} (step={ckpt['step']})")

    if not _models:
        logger.error("No checkpoints found. Set DQN_CHECKPOINT / RAINBOW_CHECKPOINT env vars.")
        return

    thread = threading.Thread(target=_game_loop, daemon=True)
    thread.start()
    logger.info(f"Game loop started with {len(_models)} agent(s)")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.get("/stream")
def stream():
    """SSE stream: both agents' frames + metrics."""

    def gen():
        while True:
            with _lock:
                payload = {}
                for name, a in _agents.items():
                    payload[name] = {
                        "frame": a["frame_b64"],
                        "q_values": a["q_values"],
                        "score": a["score"],
                        "episode": a["total_episodes"],
                        "step": a["step"],
                        "total_steps": a["total_steps"],
                        "avg_score": round(a["avg_score_10"], 1),
                        "aps": round(a["aps"], 0),
                        "total_reward": round(a["total_reward"], 0),
                        "best_score": a["best_score"],
                        "recent_scores": list(a["recent_scores"]),
                    }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1 / 15)

    return StreamingResponse(gen(), media_type="text/event-stream")


class HealthResponse(BaseModel):
    status: str
    agents_loaded: list[str]
    total_episodes: dict[str, int]


@app.get("/health", response_model=HealthResponse)
def health():
    with _lock:
        return HealthResponse(
            status="ok",
            agents_loaded=list(_models.keys()),
            total_episodes={n: a["total_episodes"] for n, a in _agents.items()},
        )


@app.get("/metrics")
def metrics():
    """Prometheus-compatible metrics."""
    with _lock:
        lines = []
        for name, a in _agents.items():
            tag = name.lower().replace("-", "_")
            lines += [
                f'dqn_episodes_total{{agent="{name}"}} {a["total_episodes"]}',
                f'dqn_steps_total{{agent="{name}"}} {a["total_steps"]}',
                f'dqn_current_score{{agent="{name}"}} {a["score"]}',
                f'dqn_avg_score_10{{agent="{name}"}} {a["avg_score_10"]:.1f}',
                f'dqn_best_score{{agent="{name}"}} {a["best_score"]}',
                f'dqn_actions_per_second{{agent="{name}"}} {a["aps"]:.1f}',
                f'dqn_total_reward{{agent="{name}"}} {a["total_reward"]:.0f}',
            ]
    return StreamingResponse(
        iter(["\n".join(lines) + "\n"]),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
