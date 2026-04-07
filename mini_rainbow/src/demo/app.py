"""Live demo: trained DQN agent plays Breakout in the browser.

Streams rendered frames over Server-Sent Events (SSE) as base64 JPEG.
Exposes Prometheus metrics at /metrics.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from pathlib import Path

import ale_py  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from gymnasium.wrappers import AtariPreprocessing
from pydantic import BaseModel

from mini_rainbow.src.networks.dueling_q_network import DuelingQNetwork
from mini_rainbow.src.networks.q_network import QNetwork
from mini_rainbow.src.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Mini-Rainbow DQN Live Demo", version="0.1.0")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_model: torch.nn.Module | None = None
_device: torch.device = torch.device("cpu")
_frame_buffer: dict = {"frame_b64": "", "q_values": [], "score": 0, "episode": 0, "step": 0}
_metrics: dict = {
    "total_episodes": 0,
    "total_steps": 0,
    "total_reward": 0.0,
    "current_score": 0.0,
    "avg_score_last_10": 0.0,
    "actions_per_second": 0.0,
    "recent_scores": [],
}
_lock = threading.Lock()
_running = False


# ---------------------------------------------------------------------------
# Frame-stack helper (same as training but minimal)
# ---------------------------------------------------------------------------
class _FrameStack:
    """Simple frame stacker for inference."""

    def __init__(self, num_stack: int = 4):
        self.num_stack = num_stack
        self.frames: list[np.ndarray] = []

    def reset(self, obs: np.ndarray) -> np.ndarray:
        frame = self._process(obs)
        self.frames = [frame] * self.num_stack
        return np.stack(self.frames, axis=0)

    def step(self, obs: np.ndarray) -> np.ndarray:
        frame = self._process(obs)
        self.frames.append(frame)
        if len(self.frames) > self.num_stack:
            self.frames.pop(0)
        return np.stack(self.frames, axis=0)

    @staticmethod
    def _process(obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3:
            obs = obs.squeeze(-1)
        return obs.astype(np.uint8)


# ---------------------------------------------------------------------------
# Game loop (runs in background thread)
# ---------------------------------------------------------------------------
def _game_loop() -> None:
    """Run the agent playing Breakout continuously."""
    global _running

    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode="rgb_array")
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    stacker = _FrameStack(num_stack=4)

    episode = 0
    _running = True

    while _running:
        raw_obs, _ = env.reset()
        state = stacker.reset(raw_obs)
        episode_reward = 0.0
        episode += 1
        step_count = 0
        t_start = time.time()

        done = False
        while not done and _running:
            # Get action from model
            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.uint8, device=_device).unsqueeze(0)
                q_values = _model(state_t)
                action = q_values.argmax(dim=1).item()
                q_list = q_values.squeeze(0).cpu().tolist()

            raw_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = stacker.step(raw_obs)
            episode_reward += reward
            step_count += 1

            # Render RGB frame for browser
            rgb_frame = env.render()
            if rgb_frame is not None:
                # Encode as JPEG base64
                from PIL import Image

                img = Image.fromarray(rgb_frame)
                img = img.resize((320, 420), Image.NEAREST)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                frame_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                with _lock:
                    _frame_buffer["frame_b64"] = frame_b64
                    _frame_buffer["q_values"] = q_list
                    _frame_buffer["score"] = episode_reward
                    _frame_buffer["episode"] = episode
                    _frame_buffer["step"] = step_count

            # Control speed (~30 FPS for smooth viewing)
            time.sleep(1 / 30)

        # Update metrics
        elapsed = time.time() - t_start
        with _lock:
            _metrics["total_episodes"] = episode
            _metrics["total_steps"] += step_count
            _metrics["total_reward"] += episode_reward
            _metrics["current_score"] = episode_reward
            _metrics["actions_per_second"] = step_count / max(elapsed, 0.01)
            _metrics["recent_scores"].append(episode_reward)
            if len(_metrics["recent_scores"]) > 10:
                _metrics["recent_scores"] = _metrics["recent_scores"][-10:]
            _metrics["avg_score_last_10"] = np.mean(_metrics["recent_scores"])

        logger.info(f"Episode {episode}: score={episode_reward:.0f}, steps={step_count}")

    env.close()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    """Load model and start game loop on server startup."""
    global _model, _device

    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "checkpoint_best.pt")
    dueling = os.environ.get("DUELING", "false").lower() == "true"
    device_str = os.environ.get("DEVICE", "cpu")

    _device = torch.device(device_str)

    if dueling:
        _model = DuelingQNetwork(in_channels=4, num_actions=4).to(_device)
    else:
        _model = QNetwork(in_channels=4, num_actions=4).to(_device)

    ckpt = load_checkpoint(checkpoint_path, device=_device)
    _model.load_state_dict(ckpt["online_net"])
    _model.eval()
    logger.info(f"Loaded checkpoint from {checkpoint_path} (step={ckpt['step']})")

    # Start game loop in background
    thread = threading.Thread(target=_game_loop, daemon=True)
    thread.start()
    logger.info("Game loop started")


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the live demo page."""
    return HTML_PAGE


@app.get("/stream")
def stream():
    """SSE stream of game frames."""

    def event_stream():
        while True:
            with _lock:
                data = (
                    f'{{"frame":"{_frame_buffer["frame_b64"]}",'
                    f'"q_values":{_frame_buffer["q_values"]},'
                    f'"score":{_frame_buffer["score"]},'
                    f'"episode":{_frame_buffer["episode"]},'
                    f'"step":{_frame_buffer["step"]}}}'
                )
            yield f"data: {data}\n\n"
            time.sleep(1 / 20)  # 20 FPS to browser

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    episodes_played: int
    avg_score: float


@app.get("/health", response_model=HealthResponse)
def health():
    with _lock:
        return HealthResponse(
            status="ok",
            model_loaded=_model is not None,
            episodes_played=_metrics["total_episodes"],
            avg_score=_metrics["avg_score_last_10"],
        )


@app.get("/metrics")
def metrics():
    """Prometheus-compatible metrics endpoint."""
    with _lock:
        m = _metrics.copy()

    lines = [
        "# HELP dqn_episodes_total Total episodes played",
        "# TYPE dqn_episodes_total counter",
        f"dqn_episodes_total {m['total_episodes']}",
        "",
        "# HELP dqn_steps_total Total environment steps",
        "# TYPE dqn_steps_total counter",
        f"dqn_steps_total {m['total_steps']}",
        "",
        "# HELP dqn_current_score Score of the current/last episode",
        "# TYPE dqn_current_score gauge",
        f"dqn_current_score {m['current_score']}",
        "",
        "# HELP dqn_avg_score_last_10 Average score over last 10 episodes",
        "# TYPE dqn_avg_score_last_10 gauge",
        f"dqn_avg_score_last_10 {m['avg_score_last_10']}",
        "",
        "# HELP dqn_actions_per_second Actions per second",
        "# TYPE dqn_actions_per_second gauge",
        f"dqn_actions_per_second {m['actions_per_second']:.2f}",
        "",
        "# HELP dqn_total_reward Cumulative reward across all episodes",
        "# TYPE dqn_total_reward counter",
        f"dqn_total_reward {m['total_reward']}",
        "",
    ]
    return StreamingResponse(
        iter(["\n".join(lines) + "\n"]),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ---------------------------------------------------------------------------
# HTML page (self-contained, no external deps)
# ---------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <title>Mini-Rainbow DQN — Live Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0a0a0a; color: #e0e0e0; font-family: 'Courier New', monospace; }
        .container { display: flex; height: 100vh; }
        .game-panel { flex: 1; display: flex; align-items: center; justify-content: center; background: #111; }
        .game-panel img { border: 2px solid #333; image-rendering: pixelated; }
        .info-panel { width: 340px; padding: 20px; background: #1a1a1a; border-left: 1px solid #333; overflow-y: auto; }
        h1 { font-size: 16px; color: #4fc3f7; margin-bottom: 15px; }
        h2 { font-size: 13px; color: #81c784; margin: 15px 0 8px 0; text-transform: uppercase; }
        .stat { display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; border-bottom: 1px solid #222; }
        .stat-value { color: #fff; font-weight: bold; }
        .q-bar-container { margin: 3px 0; }
        .q-bar-label { font-size: 11px; color: #888; }
        .q-bar { height: 14px; background: #333; border-radius: 2px; margin: 2px 0; position: relative; }
        .q-bar-fill { height: 100%; border-radius: 2px; transition: width 0.1s; }
        .q-bar-fill.best { background: #4fc3f7; }
        .q-bar-fill.other { background: #37474f; }
        .q-val { position: absolute; right: 4px; top: 0; font-size: 10px; color: #fff; line-height: 14px; }
        .actions { font-size: 11px; color: #666; margin-top: 20px; }
        .footer { margin-top: 20px; font-size: 11px; color: #555; }
        .footer a { color: #4fc3f7; }
    </style>
</head>
<body>
<div class="container">
    <div class="game-panel">
        <img id="gameFrame" width="320" height="420" alt="Breakout">
    </div>
    <div class="info-panel">
        <h1>Mini-Rainbow DQN</h1>
        <p style="font-size:12px; color:#888; margin-bottom:15px;">Trained agent playing Atari Breakout</p>

        <h2>Game Stats</h2>
        <div class="stat"><span>Score</span><span class="stat-value" id="score">0</span></div>
        <div class="stat"><span>Episode</span><span class="stat-value" id="episode">0</span></div>
        <div class="stat"><span>Step</span><span class="stat-value" id="step">0</span></div>

        <h2>Q-Values (per action)</h2>
        <div id="qvalues">
            <div class="q-bar-container">
                <div class="q-bar-label">NOOP</div>
                <div class="q-bar"><div class="q-bar-fill other" id="q0" style="width:0%"><span class="q-val" id="qv0">0</span></div></div>
            </div>
            <div class="q-bar-container">
                <div class="q-bar-label">FIRE</div>
                <div class="q-bar"><div class="q-bar-fill other" id="q1" style="width:0%"><span class="q-val" id="qv1">0</span></div></div>
            </div>
            <div class="q-bar-container">
                <div class="q-bar-label">RIGHT</div>
                <div class="q-bar"><div class="q-bar-fill other" id="q2" style="width:0%"><span class="q-val" id="qv2">0</span></div></div>
            </div>
            <div class="q-bar-container">
                <div class="q-bar-label">LEFT</div>
                <div class="q-bar"><div class="q-bar-fill other" id="q3" style="width:0%"><span class="q-val" id="qv3">0</span></div></div>
            </div>
        </div>

        <div class="actions">
            <h2>Actions</h2>
            <p>0 = NOOP &nbsp; 1 = FIRE &nbsp; 2 = RIGHT &nbsp; 3 = LEFT</p>
        </div>

        <div class="footer">
            <p>Architecture: Nature DQN (CNN)</p>
            <p>Environment: ALE/Breakout-v5</p>
            <p><a href="/health">/health</a> &middot; <a href="/metrics">/metrics</a></p>
        </div>
    </div>
</div>
<script>
const img = document.getElementById('gameFrame');
const evtSource = new EventSource('/stream');
evtSource.onmessage = function(event) {
    const d = JSON.parse(event.data);
    if (d.frame) {
        img.src = 'data:image/jpeg;base64,' + d.frame;
    }
    document.getElementById('score').textContent = d.score;
    document.getElementById('episode').textContent = d.episode;
    document.getElementById('step').textContent = d.step;

    if (d.q_values && d.q_values.length === 4) {
        const qv = d.q_values;
        const maxQ = Math.max(...qv.map(Math.abs), 0.01);
        const bestIdx = qv.indexOf(Math.max(...qv));
        for (let i = 0; i < 4; i++) {
            const pct = Math.max((qv[i] / maxQ) * 50 + 50, 2);
            const bar = document.getElementById('q' + i);
            bar.style.width = pct + '%';
            bar.className = 'q-bar-fill ' + (i === bestIdx ? 'best' : 'other');
            document.getElementById('qv' + i).textContent = qv[i].toFixed(3);
        }
    }
};
</script>
</body>
</html>"""
