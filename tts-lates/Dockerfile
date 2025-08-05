# --------  tts-runtime.Dockerfile  --------
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 1) System libs for soundfile (ffmpeg only if you really need it)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 2) (Optional) upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# 3) Python-only deps that are **not** in the RunPod PyTorch base
COPY tts-lates/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy the code and prompt WAV **into** the image
#    ─ image will later mount the network-volume at the same path, so
#      the files you copy here will be masked by the volume at runtime.
COPY tts-lates/handler.py         handler.py
COPY tts-lates/preprocess.py      preprocess.py
COPY tts-lates/consent_audio.wav  consent_audio.wav

# 5) The code expects: /workspace/minh-tran/tts-lates
WORKDIR /workspace/minh-tran/tts-lates

CMD ["python", "handler.py"]
