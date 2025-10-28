# Vietnamese ASR (Sherpa-ONNX) — RunPod Serverless
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg để chuẩn hoá âm thanh
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY rp_handler.py .

# ENV mặc định đồng bộ style Spark
ENV MODEL_ID="hynt/Zipformer-30M-RNNT-6000h" \
    MODEL_DIR="/models/Zipformer-30M-RNNT-6000h" \
    OUT_DIR="/runpod-volume/jobs" \
    NUM_THREADS="1"

CMD ["python", "-u", "rp_handler.py"]
