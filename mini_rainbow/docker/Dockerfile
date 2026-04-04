FROM python:3.11-slim

# System deps for Atari + video recording
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unrar \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY mini_rainbow/ ./mini_rainbow/
COPY README.md ./

# Install PyTorch (CPU for Docker by default; override for GPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install project
RUN pip install --no-cache-dir -e .

# Default: run training
ENTRYPOINT ["python", "-m", "mini_rainbow.scripts.train"]
