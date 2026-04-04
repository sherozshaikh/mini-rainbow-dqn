# Runbook

Operational guide for setting up, training, evaluating, and serving the Mini-Rainbow DQN system.

---

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- NVIDIA GPU + CUDA drivers (for training; check with `nvidia-smi`)
- Docker (optional, for containerized deployment)
- ffmpeg (for evaluation video recording: `sudo apt-get install -y ffmpeg`)

---

## VM Setup (A6000 / CUDA GPU)

```bash
git clone git@github.com:sherozshaikh/mini-rainbow-dqn.git
cd mini-rainbow-dqn

# Create project venv (named .venv_mini_rainbow)
uv venv .venv_mini_rainbow --python 3.11
source .venv_mini_rainbow/bin/activate

# Install project dependencies
uv pip install -e ".[dev]"

# Install PyTorch for your CUDA version (pick one)
make install-torch-cu121    # CUDA 12.1
# make install-torch-cu118  # CUDA 11.8
# make install-torch-cpu    # CPU only (not recommended for training)

# Verify setup
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## Stage 1: Baseline DQN Training

### Smoke Test (verify nothing crashes, ~30 seconds)

```bash
make smoke-test
```

Runs 1000 steps with tiny buffer, no W&B, no video. If this completes without errors, the full setup is correct.

### Full Training

```bash
make train-stage1
```

This runs 5M steps of baseline DQN on BreakoutNoFrameskip-v4 with:
- Standard Q-Network (Nature DQN architecture)
- Uniform replay buffer (1M capacity)
- Epsilon-greedy: 1.0 -> 0.01 over 1M steps
- Target network hard update every 10K steps
- Evaluation every 100K steps (10 episodes, with video)
- Checkpoints saved every 250K steps

### Monitor Progress

Console output logs every 1000 steps:
```
Step 50000/5000000 | Loss: 0.0234 | Eps: 0.9500 | Avg100: 1.20 | FPS: 145
```

Key metrics to watch:
- `Avg100` (mean reward over last 100 episodes): should increase over time
- `Loss`: should decrease and stabilize
- `Eps`: decays from 1.0 to 0.01 over first 1M steps
- `FPS`: ~100-200 on A6000

### Checkpoints

Saved to `outputs/checkpoints/`:
- `checkpoint_latest.pt` -- most recent periodic save
- `checkpoint_best.pt` -- best evaluation reward
- `checkpoint_final.pt` -- end of training

### Evaluation Videos

Saved to `outputs/videos/step_<N>/` as `.mp4` files.

---

## Stage 2: Rainbow-Lite (after Stage 1 is validated)

```bash
make train-stage2
```

Adds Double DQN + Dueling Networks + Prioritized Experience Replay.

---

## Evaluation (standalone)

```bash
make eval CKPT=outputs/checkpoints/checkpoint_best.pt
```

Runs 10 episodes with epsilon=0 and records videos to `eval_videos/`.

---

## API Server

```bash
make serve CKPT=outputs/checkpoints/checkpoint_best.pt
make health   # verify it's running
```

Endpoints:
- `GET /health` -- health check
- `POST /act` -- get action for a given state

---

## W&B Setup (optional)

```bash
pip install wandb
wandb login
# Paste your API key when prompted
```

Get your free API key at https://wandb.ai/authorize

Training will auto-detect W&B. To disable: add `wandb.enabled=false` to any make command.

---

## Custom Training Overrides

```bash
# Train Double DQN only (no dueling, no PER)
make train ARGS="agent=double_dqn"

# Train with fewer steps
make train ARGS="+experiment=stage1_dqn training.total_steps=1000000"

# Train on CPU (not recommended)
make train-stage1 DEVICE=cpu
```

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make setup` | Create venv with uv |
| `make install` | Install project deps |
| `make install-torch-cu121` | Install PyTorch for CUDA 12.1 |
| `make install-torch-cu118` | Install PyTorch for CUDA 11.8 |
| `make install-torch-cpu` | Install PyTorch CPU-only |
| `make smoke-test` | Quick 1000-step sanity check |
| `make train-stage1` | Stage 1: Baseline DQN |
| `make train-stage2` | Stage 2: Rainbow-Lite |
| `make train ARGS='...'` | Train with custom Hydra overrides |
| `make eval CKPT=path` | Evaluate a checkpoint |
| `make serve CKPT=path` | Start inference API server |
| `make health` | Health check running API |
| `make docker-build` | Build Docker image |
| `make docker-train` | Run training in Docker |
| `make docker-push` | Build and push to Docker Hub |
| `make lint` | Lint with ruff |
| `make format` | Format with isort + black + ruff |
| `make test` | Run pytest |
| `make clean` | Remove generated files |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch` | Install PyTorch: `make install-torch-cu121` |
| `ModuleNotFoundError: ale_py` | Run: `uv pip install -e ".[dev]"` (reinstall with extras) |
| CUDA out of memory | Reduce `training.batch_size` or `replay.buffer_size` |
| Slow FPS (<50) | Check GPU is being used: `nvidia-smi` during training |
| `make help` fails | Ensure Makefile uses tabs not spaces (re-pull from git) |
| W&B errors | Disable with `wandb.enabled=false` or run `wandb login` |
| No video files | Install ffmpeg: `sudo apt-get install -y ffmpeg` |
| Port 8000 in use | `lsof -ti:8000 | xargs kill -9` |
