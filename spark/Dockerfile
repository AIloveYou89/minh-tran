# spark/Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torchaudio khớp CUDA 12.4
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu124 torchaudio==2.5.1+cu124

# Python deps
COPY spark/requirements.txt .
RUN pip install -r requirements.txt

# Code
COPY spark/rp_handler.py .

# Queue-based: KHÔNG cần expose cổng
CMD ["python", "-u", "rp_handler.py"]
