# --------  tts-runtime.Dockerfile  --------
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 1) Thư viện hệ thống cho soundfile (và ffmpeg nếu cần)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 2) (Tùy) nâng cấp pip
RUN python -m pip install --no-cache-dir --upgrade pip

# 3) Cài các Python-only libs (không đụng đến torch trong base image)
COPY requirements-runtime.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 4) Đặt WORKDIR trùng với mount của Network Storage
WORKDIR /workspace/minh-tran/tts-lates

# 5) Khởi FastAPI
CMD ["uvicorn", "jobs_app_on_demand:app", "--host", "0.0.0.0", "--port", "5300"]

