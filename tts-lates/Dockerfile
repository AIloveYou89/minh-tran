# ---------- Base image (CUDA 12.4, PyTorch 2.4, Python 3.11) ----------
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 1) System libs cho soundfile (ffmpeg optional)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 2) (Tuỳ chọn) nâng cấp pip
RUN python -m pip install --no-cache-dir -U pip

# 3) Đặt thư mục làm việc TRƯỚC khi copy code
WORKDIR /workspace/minh-tran/tts-lates

# 4) Chép requirements + cài Python-only dependencies
COPY tts-lates/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) Chép code & prompt WAV đúng vị trí WORKDIR
COPY tts-lates/handler.py        ./handler.py
COPY tts-lates/preprocess.py     ./preprocess.py
COPY tts-lates/consent_audio.wav ./fixed/consent_audio.wav

# 6) Chạy handler (RunPod Serverless tự khởi lệnh này cho mỗi worker)
CMD ["python", "handler.py"]
