# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BG Remover — GPU-enabled Multi-stage Dockerfile                        ║
# ║  Stage 1: Build React frontend (Node)                                   ║
# ║  Stage 2: CUDA runtime + Python backend + model from DagsHub            ║
# ║                                                                         ║
# ║  Automatically uses GPU if host has nvidia-container-toolkit installed.  ║
# ║  Falls back to CPU gracefully if no GPU is available.                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# syntax=docker/dockerfile:1

# ── Stage 1: Build Frontend ────────────────────────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Stage 2: CUDA Backend ─────────────────────────────────────────────────────
# CUDA 12.2 runtime — lightweight, has CUDA + cuDNN but not the full toolkit
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 + system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3.11-dev \
    libgl1 libglib2.0-0 git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY bg_remover/ ./bg_remover/
COPY dvc.yaml dvc.lock ./

# Copy built frontend
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Pull ONLY the model from DagsHub using BuildKit secrets
RUN --mount=type=secret,id=dagshub_owner \
    --mount=type=secret,id=dagshub_repo \
    --mount=type=secret,id=dagshub_token \
    git init && \
    OWNER=$(cat /run/secrets/dagshub_owner) && \
    REPO=$(cat /run/secrets/dagshub_repo) && \
    TOKEN=$(cat /run/secrets/dagshub_token) && \
    dvc remote add -f dagshub https://dagshub.com/${OWNER}/${REPO}.dvc && \
    dvc remote modify dagshub auth basic && \
    dvc remote modify dagshub user "${OWNER}" && \
    dvc remote modify dagshub password "${TOKEN}" && \
    dvc pull -r dagshub train && \
    rm -rf .dvc/config .dvc/tmp .git

ENV MODEL_PATH=/app/bg_remover/artifacts/model/my_model.pth
# Let PyTorch auto-detect GPU/CPU — no forced device
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000

CMD ["uvicorn", "bg_remover.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
