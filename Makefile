DOCKER_HUB_USER := sherozshaikh
PROJECT         := mini-rainbow-dqn
IMAGE           := $(DOCKER_HUB_USER)/$(PROJECT)
TAG             := v0.1.0
PYTHON          := python
DEVICE          := auto

# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

## Create venv and install all deps (run once)
setup:
	uv venv .venv_mini_rainbow --python 3.11
	@echo "Activate with: source .venv_mini_rainbow/bin/activate"
	@echo "Then run: make install"

## Install core deps only (train + eval, no API, no W&B, no video)
install:
	uv pip install -e .
	@echo ""
	@echo "NOTE: Install PyTorch separately for your CUDA version:"
	@echo "  make install-torch-cu126   # CUDA 12.6+ / driver 13.0 (A6000)"
	@echo "  make install-torch-cu124   # CUDA 12.4"
	@echo "  make install-torch-cu121   # CUDA 12.1"
	@echo "  make install-torch-cu118   # CUDA 11.8"
	@echo "  make install-torch-cpu     # CPU only"
	@echo ""
	@echo "Optional extras (install only what you need):"
	@echo "  uv pip install -e '.[wandb]'    # W&B logging"
	@echo "  uv pip install -e '.[video]'    # Eval video recording"
	@echo "  uv pip install -e '.[api]'      # FastAPI inference server"
	@echo "  uv pip install -e '.[all]'      # All of the above"
	@echo "  uv pip install -e '.[dev]'      # All + dev tools"

## Install everything (all extras + dev tools, no PyTorch)
install-all:
	uv pip install -e ".[dev]"

## Install PyTorch for CUDA 12.6 (for CUDA 13.0 driver — backward compatible)
install-torch-cu126:
	uv pip install torch --index-url https://download.pytorch.org/whl/cu126

## Install PyTorch for CUDA 12.4
install-torch-cu124:
	uv pip install torch --index-url https://download.pytorch.org/whl/cu124

## Install PyTorch for CUDA 12.1
install-torch-cu121:
	uv pip install torch --index-url https://download.pytorch.org/whl/cu121

## Install PyTorch for CUDA 11.8
install-torch-cu118:
	uv pip install torch --index-url https://download.pytorch.org/whl/cu118

## Install PyTorch CPU-only
install-torch-cpu:
	uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

## Stage 1: Train baseline DQN (uniform replay, standard Q-network)
train-stage1:
	PYTHONPATH=. $(PYTHON) -m mini_rainbow.scripts.train +experiment=stage1_dqn device=$(DEVICE)

## Stage 2: Train Rainbow-Lite (Double DQN + Dueling + PER)
train-stage2:
	PYTHONPATH=. $(PYTHON) -m mini_rainbow.scripts.train +experiment=stage2_rainbow_lite device=$(DEVICE)

## Train with custom overrides (e.g. make train ARGS="agent=double_dqn training.total_steps=1000000")
train:
	PYTHONPATH=. $(PYTHON) -m mini_rainbow.scripts.train device=$(DEVICE) $(ARGS)

## Quick smoke test: 1000 steps to verify nothing crashes
smoke-test:
	PYTHONPATH=. $(PYTHON) -m mini_rainbow.scripts.train +experiment=stage1_dqn \
		training.total_steps=1000 \
		training.learning_starts=100 \
		training.eval_freq=500 \
		training.eval_episodes=1 \
		training.save_freq=500 \
		training.log_freq=100 \
		replay.buffer_size=1000 \
		wandb.enabled=false \
		training.record_video=false \
		device=$(DEVICE)

## Validate ALL 4 variants (every run tracked in W&B).
## Runs 2000 steps each: DQN, Double DQN, Dueling DDQN, Rainbow-Lite (PER).
## Total time: ~5 minutes on A6000.
validate-all: clean
	@echo "=== [1/4] Validating DQN (baseline) ==="
	PYTHONPATH=. $(PYTHON) -m mini_rainbow.scripts.train \
		+experiment=stage1_dqn \
		run_name=validate_dqn \
		training.total_steps=2000 \
		training.learning_starts=200 \
		training.eval_freq=1000 \
		training.eval_episodes=1 \
		training.save_freq=1000 \
		training.log_freq=500 \
		replay.buffer_size=2000 \
		training.record_video=false \
		device=$(DEVICE)
	@echo ""
	@echo "=== [2/4] Validating Double DQN ==="
	PYTHONPATH=. $(PYTHON) -m mini_rainbow.scripts.train \
		agent=double_dqn \
		run_name=validate_double_dqn \
		training.total_steps=2000 \
		training.learning_starts=200 \
		training.eval_freq=1000 \
		training.eval_episodes=1 \
		training.save_freq=1000 \
		training.log_freq=500 \
		replay.buffer_size=2000 \
		training.record_video=false \
		device=$(DEVICE)
	@echo ""
	@echo "=== [3/4] Validating Dueling DDQN ==="
	PYTHONPATH=. $(PYTHON) -m mini_rainbow.scripts.train \
		agent=dueling_ddqn \
		run_name=validate_dueling_ddqn \
		training.total_steps=2000 \
		training.learning_starts=200 \
		training.eval_freq=1000 \
		training.eval_episodes=1 \
		training.save_freq=1000 \
		training.log_freq=500 \
		replay.buffer_size=2000 \
		training.record_video=false \
		device=$(DEVICE)
	@echo ""
	@echo "=== [4/4] Validating Rainbow-Lite (DDQN + Dueling + PER) ==="
	PYTHONPATH=. $(PYTHON) -m mini_rainbow.scripts.train \
		+experiment=stage2_rainbow_lite \
		run_name=validate_rainbow_lite \
		training.total_steps=2000 \
		training.learning_starts=200 \
		training.eval_freq=1000 \
		training.eval_episodes=1 \
		training.save_freq=1000 \
		training.log_freq=500 \
		replay.buffer_size=2000 \
		training.record_video=false \
		device=$(DEVICE)
	@echo ""
	@echo "=== ALL 4 VALIDATIONS PASSED ==="

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

## Evaluate a trained checkpoint (e.g. make eval CKPT=outputs/checkpoints/checkpoint_best.pt)
eval:
	PYTHONPATH=. $(PYTHON) -m mini_rainbow.scripts.evaluate --checkpoint $(CKPT) --episodes 10

# ---------------------------------------------------------------------------
# API Server
# ---------------------------------------------------------------------------

## Start inference API (e.g. make serve CKPT=outputs/checkpoints/checkpoint_best.pt)
serve:
	PYTHONPATH=. $(PYTHON) -m mini_rainbow.scripts.serve --checkpoint $(CKPT)

## Health check the running API
health:
	@curl -sf http://localhost:8000/health | python3 -m json.tool || echo "API not reachable"

# ---------------------------------------------------------------------------
# Demo (live agent playing in browser + Grafana dashboard)
# ---------------------------------------------------------------------------

## Run demo locally: DQN vs Rainbow-Lite side-by-side (http://localhost:8000)
demo:
	PYTHONPATH=. $(PYTHON) -m mini_rainbow.scripts.demo \
		--dqn-checkpoint gdrive/checkpoints/stage1_dqn_best.pt \
		--rainbow-checkpoint gdrive/checkpoints/stage2_rainbow_lite_best.pt

## Start full demo stack: agent + Prometheus + Grafana (docker-compose)
demo-stack:
	docker compose up --build

## Stop demo stack
demo-stop:
	docker compose down

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

## Build platform image (live agent + metrics, baked checkpoints)
docker-build:
	docker build -t $(IMAGE):$(TAG) -f mini_rainbow/docker/Dockerfile.demo .

## Run platform locally
docker-run:
	docker run --rm -p 8000:8000 $(IMAGE):$(TAG)

## Push image to Docker Hub
docker-push: docker-build
	docker push $(IMAGE):$(TAG)

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------

## Lint code
lint:
	ruff check .

## Format code
format:
	isort . && black . && ruff check --fix . && ruff format .

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

## Run tests
test:
	PYTHONPATH=. pytest tests/ -v

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

## Remove generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".vscode" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".idea" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".tox" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -type f -delete
	find . -name "*.pyo" -type f -delete
	find . -name ".DS_Store" -type f -delete
	rm -rf outputs/ eval_videos/ wandb/ 2>/dev/null || true

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

## Show this help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup                Create venv with uv"
	@echo "    make install              Install core deps only (skinny)"
	@echo "    make install-all          Install all extras + dev tools"
	@echo "    make install-torch-cu126  Install PyTorch for CUDA 12.6+ (A6000)"
	@echo "    make install-torch-cu124  Install PyTorch for CUDA 12.4"
	@echo "    make install-torch-cu121  Install PyTorch for CUDA 12.1"
	@echo "    make install-torch-cu118  Install PyTorch for CUDA 11.8"
	@echo "    make install-torch-cpu    Install PyTorch CPU-only"
	@echo ""
	@echo "  Training:"
	@echo "    make train-stage1         Stage 1: Baseline DQN"
	@echo "    make train-stage2         Stage 2: Rainbow-Lite (DDQN + Dueling + PER)"
	@echo "    make train ARGS='...'     Train with custom Hydra overrides"
	@echo "    make smoke-test           Quick 1000-step sanity check"
	@echo "    make validate-all         Validate all 4 variants + W&B (~5 min)"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make eval CKPT=path       Evaluate a checkpoint"
	@echo ""
	@echo "  Demo:"
	@echo "    make demo                 Run live demo locally (port 8000)"
	@echo "    make demo-stack           Start full stack (demo + Prometheus + Grafana)"
	@echo "    make demo-stop            Stop demo stack"
	@echo ""
	@echo "  API:"
	@echo "    make serve CKPT=path      Start inference API server"
	@echo "    make health               Health check running API"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build         Build platform image"
	@echo "    make docker-run           Run platform locally (port 8000)"
	@echo "    make docker-push          Build and push to Docker Hub"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make lint                 Lint with ruff"
	@echo "    make format               Format with isort + black + ruff"
	@echo "    make test                 Run pytest"
	@echo "    make clean                Remove generated files"

.PHONY: setup install install-all install-torch-cu126 install-torch-cu124 install-torch-cu121 install-torch-cu118 install-torch-cpu \
        train-stage1 train-stage2 train smoke-test validate-all \
        eval serve health \
        demo demo-stack demo-stop \
        docker-build docker-run docker-push \
        lint format test clean help
