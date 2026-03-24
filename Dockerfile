# ABOUTME: Docker image for HappyAnonymity face anonymization system
# ABOUTME: NVIDIA GPU-enabled, uses uv for Python dependency management

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PYTHON=3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev


FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/pyproject.toml /app/uv.lock ./

# Copy vendored trackers
COPY vendor/ ./vendor/

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

ENTRYPOINT ["uv", "run", "python", "-m", "src.cli"]
