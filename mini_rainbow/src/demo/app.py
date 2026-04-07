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
    """SSE stream of game frames + metrics."""

    def event_stream():
        while True:
            with _lock:
                data = (
                    f'{{"frame":"{_frame_buffer["frame_b64"]}",'
                    f'"q_values":{_frame_buffer["q_values"]},'
                    f'"score":{_frame_buffer["score"]},'
                    f'"episode":{_frame_buffer["episode"]},'
                    f'"step":{_frame_buffer["step"]},'
                    f'"total_episodes":{_metrics["total_episodes"]},'
                    f'"total_steps":{_metrics["total_steps"]},'
                    f'"avg_score":{_metrics["avg_score_last_10"]:.2f},'
                    f'"aps":{_metrics["actions_per_second"]:.1f},'
                    f'"total_reward":{_metrics["total_reward"]:.0f},'
                    f'"recent_scores":{list(_metrics["recent_scores"])}}}'
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
        body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; }

        .header { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 24px; display: flex; align-items: center; gap: 12px; }
        .header h1 { font-size: 18px; color: #58a6ff; font-weight: 600; }
        .header .badge { background: #238636; color: #fff; font-size: 11px; padding: 2px 8px; border-radius: 10px; }

        .main { display: grid; grid-template-columns: 340px 1fr; grid-template-rows: 1fr; height: calc(100vh - 49px); }

        /* Left: Game */
        .game-col { background: #0d1117; display: flex; flex-direction: column; align-items: center; justify-content: center; border-right: 1px solid #30363d; }
        .game-col img { border: 2px solid #30363d; border-radius: 4px; image-rendering: pixelated; }
        .game-score { margin-top: 12px; font-size: 28px; font-weight: 700; color: #58a6ff; }
        .game-score-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }

        /* Right: Dashboard */
        .dash-col { overflow-y: auto; padding: 20px; }

        .card-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; }
        .card-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
        .card-value { font-size: 24px; font-weight: 700; color: #c9d1d9; }
        .card-value.blue { color: #58a6ff; }
        .card-value.green { color: #3fb950; }
        .card-value.orange { color: #d29922; }

        .section-title { font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin: 20px 0 10px 0; border-bottom: 1px solid #21262d; padding-bottom: 6px; }

        /* Q-values */
        .q-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
        .q-label { width: 50px; font-size: 12px; color: #8b949e; text-align: right; }
        .q-track { flex: 1; height: 20px; background: #21262d; border-radius: 3px; position: relative; overflow: hidden; }
        .q-fill { height: 100%; border-radius: 3px; transition: width 0.15s ease; }
        .q-fill.best { background: linear-gradient(90deg, #1f6feb, #58a6ff); }
        .q-fill.other { background: #30363d; }
        .q-num { width: 65px; font-size: 12px; color: #c9d1d9; font-family: 'Courier New', monospace; text-align: right; }

        /* Score chart */
        .chart-container { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; margin-top: 12px; }
        .chart-container canvas { width: 100%; height: 120px; }

        /* Footer */
        .footer { margin-top: 20px; font-size: 11px; color: #484f58; display: flex; gap: 16px; }
        .footer a { color: #58a6ff; text-decoration: none; }
        .footer a:hover { text-decoration: underline; }
    </style>
</head>
<body>

<div class="header">
    <h1>Mini-Rainbow DQN</h1>
    <span class="badge">LIVE</span>
    <span style="font-size:13px; color:#8b949e;">Trained agent playing Atari Breakout</span>
</div>

<div class="main">
    <!-- Game column -->
    <div class="game-col">
        <img id="gameFrame" width="280" height="370" alt="Breakout">
        <div class="game-score" id="currentScore">0</div>
        <div class="game-score-label">Current Score</div>
    </div>

    <!-- Dashboard column -->
    <div class="dash-col">

        <!-- Metric cards -->
        <div class="card-grid">
            <div class="card">
                <div class="card-label">Episode</div>
                <div class="card-value blue" id="metEpisode">0</div>
            </div>
            <div class="card">
                <div class="card-label">Avg Score (10 ep)</div>
                <div class="card-value green" id="metAvgScore">0</div>
            </div>
            <div class="card">
                <div class="card-label">Actions / sec</div>
                <div class="card-value orange" id="metAPS">0</div>
            </div>
            <div class="card">
                <div class="card-label">Total Steps</div>
                <div class="card-value" id="metSteps">0</div>
            </div>
            <div class="card">
                <div class="card-label">Total Reward</div>
                <div class="card-value" id="metReward">0</div>
            </div>
            <div class="card">
                <div class="card-label">Episode Step</div>
                <div class="card-value" id="metEpStep">0</div>
            </div>
        </div>

        <!-- Q-values -->
        <div class="section-title">Q-Values (Action Selection)</div>
        <div class="q-row"><span class="q-label">NOOP</span><div class="q-track"><div class="q-fill other" id="q0"></div></div><span class="q-num" id="qv0">0.000</span></div>
        <div class="q-row"><span class="q-label">FIRE</span><div class="q-track"><div class="q-fill other" id="q1"></div></div><span class="q-num" id="qv1">0.000</span></div>
        <div class="q-row"><span class="q-label">RIGHT</span><div class="q-track"><div class="q-fill other" id="q2"></div></div><span class="q-num" id="qv2">0.000</span></div>
        <div class="q-row"><span class="q-label">LEFT</span><div class="q-track"><div class="q-fill other" id="q3"></div></div><span class="q-num" id="qv3">0.000</span></div>

        <!-- Score history chart -->
        <div class="section-title">Score History</div>
        <div class="chart-container">
            <canvas id="scoreChart" height="120"></canvas>
        </div>

        <!-- Info -->
        <div class="section-title">Model Info</div>
        <div style="font-size:13px; color:#8b949e; line-height:1.8;">
            Architecture: Nature DQN (3-layer CNN + 2-layer FC)<br>
            Environment: ALE/Breakout-v5<br>
            Training: 5M steps on NVIDIA RTX A6000<br>
            Policy: Greedy (epsilon = 0)
        </div>

        <div class="footer">
            <a href="/health">/health</a>
            <a href="/metrics">/metrics (Prometheus)</a>
            <span>github.com/sherozshaikh/mini-rainbow-dqn</span>
        </div>
    </div>
</div>

<script>
// Score history for chart
const scoreHistory = [];
const maxHistory = 50;

const canvas = document.getElementById('scoreChart');
const ctx = canvas.getContext('2d');

function drawChart() {
    const w = canvas.width = canvas.offsetWidth;
    const h = canvas.height = 120;
    ctx.clearRect(0, 0, w, h);

    if (scoreHistory.length < 2) return;

    const maxVal = Math.max(...scoreHistory, 1);
    const step = w / (maxHistory - 1);

    // Grid lines
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += 30) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Line
    ctx.beginPath();
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2;
    const startIdx = Math.max(0, scoreHistory.length - maxHistory);
    for (let i = startIdx; i < scoreHistory.length; i++) {
        const x = (i - startIdx) * step;
        const y = h - (scoreHistory[i] / maxVal) * (h - 10) - 5;
        if (i === startIdx) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill under
    ctx.lineTo((scoreHistory.length - 1 - startIdx) * step, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(88,166,255,0.08)';
    ctx.fill();

    // Max label
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px monospace';
    ctx.fillText('max: ' + maxVal.toFixed(0), 4, 12);
}

let lastEpisode = 0;

const evtSource = new EventSource('/stream');
evtSource.onmessage = function(event) {
    const d = JSON.parse(event.data);

    // Game frame
    if (d.frame) {
        document.getElementById('gameFrame').src = 'data:image/jpeg;base64,' + d.frame;
    }

    // Current score
    document.getElementById('currentScore').textContent = d.score;

    // Metric cards
    document.getElementById('metEpisode').textContent = d.total_episodes || d.episode;
    document.getElementById('metAvgScore').textContent = parseFloat(d.avg_score || 0).toFixed(1);
    document.getElementById('metAPS').textContent = parseFloat(d.aps || 0).toFixed(0);
    document.getElementById('metSteps').textContent = (d.total_steps || 0).toLocaleString();
    document.getElementById('metReward').textContent = parseFloat(d.total_reward || 0).toFixed(0);
    document.getElementById('metEpStep').textContent = d.step;

    // Q-values
    if (d.q_values && d.q_values.length === 4) {
        const qv = d.q_values;
        const maxQ = Math.max(...qv.map(Math.abs), 0.01);
        const bestIdx = qv.indexOf(Math.max(...qv));
        for (let i = 0; i < 4; i++) {
            const pct = Math.max((qv[i] / maxQ) * 50 + 50, 2);
            const bar = document.getElementById('q' + i);
            bar.style.width = pct + '%';
            bar.className = 'q-fill ' + (i === bestIdx ? 'best' : 'other');
            document.getElementById('qv' + i).textContent = qv[i].toFixed(3);
        }
    }

    // Score history chart (update on new episode)
    if (d.recent_scores && d.recent_scores.length > 0 && d.total_episodes > lastEpisode) {
        lastEpisode = d.total_episodes;
        // Push the latest score
        scoreHistory.push(d.recent_scores[d.recent_scores.length - 1]);
        drawChart();
    }
};

// Redraw chart on resize
window.addEventListener('resize', drawChart);
</script>
</body>
</html>"""
