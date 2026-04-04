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
	uv venv .venv --python 3.11
	@echo "Activate with: source .venv/bin/activate"
	@echo "Then run: make install"

## Install project + deps into active venv
install:
	uv pip install -e ".[dev]"
	@echo ""
	@echo "NOTE: Install PyTorch separately for your CUDA version:"
	@echo "  uv pip install torch --index-url https://download.pytorch.org/whl/cu121"
	@echo "  (or cu118, cu124, cpu — match your nvidia-smi output)"

## Install PyTorch for CUDA 12.1 (default for A6000)
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
# Docker
# ---------------------------------------------------------------------------

## Build Docker image
docker-build:
	docker build -t $(IMAGE):$(TAG) -f mini_rainbow/docker/Dockerfile .

## Run training in Docker
docker-train:
	docker run --rm --gpus all $(IMAGE):$(TAG) +experiment=stage1_dqn

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
	@echo "    make install              Install project deps (run after activating venv)"
	@echo "    make install-torch-cu121  Install PyTorch for CUDA 12.1"
	@echo "    make install-torch-cu118  Install PyTorch for CUDA 11.8"
	@echo "    make install-torch-cpu    Install PyTorch CPU-only"
	@echo ""
	@echo "  Training:"
	@echo "    make train-stage1         Stage 1: Baseline DQN"
	@echo "    make train-stage2         Stage 2: Rainbow-Lite (DDQN + Dueling + PER)"
	@echo "    make train ARGS='...'     Train with custom Hydra overrides"
	@echo "    make smoke-test           Quick 1000-step sanity check"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make eval CKPT=path       Evaluate a checkpoint"
	@echo ""
	@echo "  API:"
	@echo "    make serve CKPT=path      Start inference API server"
	@echo "    make health               Health check running API"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build         Build Docker image"
	@echo "    make docker-train         Run training in Docker"
	@echo "    make docker-push          Build and push to Docker Hub"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make lint                 Lint with ruff"
    @echo "    make format               Format with isort + black + ruff"
	@echo "    make test                 Run pytest"
	@echo "    make clean                Remove generated files"

.PHONY: setup install install-torch-cu121 install-torch-cu118 install-torch-cpu \
        train-stage1 train-stage2 train smoke-test \
        eval serve health \
        docker-build docker-train docker-push \
        lint format test clean help
