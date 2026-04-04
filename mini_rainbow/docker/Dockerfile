FROM python:3.11-slim

# System deps: ffmpeg for video recording, minimal libs for Atari rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what's needed
COPY pyproject.toml README.md ./
COPY mini_rainbow/ ./mini_rainbow/

# Install PyTorch CPU (smallest footprint; override at build for GPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install core + video (no wandb, no API, no dev tools)
RUN pip install --no-cache-dir -e ".[video]"

# Default: run training
ENTRYPOINT ["python", "-m", "mini_rainbow.scripts.train"]
