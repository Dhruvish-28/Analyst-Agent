# ==============================================================================
# Text-to-SQL Analyst Agent — Production GPU Container
# Base: NVIDIA CUDA 12.4 Runtime on Ubuntu 22.04
# Target: FastAPI REST Server + Web Dashboard + QLoRA LangGraph Agent
# ==============================================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies & Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    sqlite3 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set working directory
WORKDIR /app

# Copy dependency definition and install via pyproject.toml
COPY pyproject.toml /app/

# Upgrade pip and install PyTorch with CUDA 12.4 wheels first for stability
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -e .

# Copy source code, web UI static assets, configs, databases, and trained LoRA adapter
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY data/spider_db/ /app/data/spider_db/
COPY output/final_adapter/ /app/output/final_adapter/

# Expose port for FastAPI backend & Web UI
EXPOSE 8000

# Health check against /api/v1/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start the unified server on 0.0.0.0:8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
