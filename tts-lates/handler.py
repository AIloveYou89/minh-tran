"""
handler.py – Runpod Serverless
──────────────────────────────
• Network-volume mount tại /workspace/minh-tran/tts-lates
• Nếu thiếu consent_audio.wav  ➜ tải từ S3 (dùng ENV)
"""

import uuid, os
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
import torch
import runpod

# ─────────── ĐƯỜNG DẪN CỐ ĐỊNH ───────────
BASE_DIR   = Path("/workspace/minh-tran/tts-lates")
FIXED_DIR  = BASE_DIR / "fixed"
PROMPT_WAV = FIXED_DIR / "consent_audio.wav"
JOBS_DIR   = BASE_DIR / "jobs"

MODEL_ID = "DragonLineageAI/Vi-SparkTTS-0.5B"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TRANSCRIPT = (
    "Tôi là chủ sở hữu giọng nói này và tôi đồng ý cho Google sử dụng "
    "giọng nói này để tạo mô hình giọng nói tổng hợp."
)

# ────────── TẢI PROMPT TỪ S3 (nếu cần) ──────────
def _download_prompt_once() -> None:
    if PROMPT_WAV.exists():
        return

    s3_ep  = os.getenv("S3_ENDPOINT_URL")
    bucket = os.getenv("S3_BUCKET_NAME")
    key    = os.getenv("S3_PROMPT_KEY")  # vd: minh-tran/tts-lates/fixed/consent_audio.wav
    akid   = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not all([s3_ep, bucket, key, akid, secret]):
        raise FileNotFoundError(
            f"{PROMPT_WAV} chưa tồn tại và biến môi trường S3_* thiếu."
        )

    print(f"[PROMPT] downloading s3://{bucket}/{key} …", flush=True)
    import boto3
    from botocore.config import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_ep,
        aws_access_key_id=akid,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )

    PROMPT_WAV.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(PROMPT_WAV))
    print("[PROMPT] download done ✔", flush=True)


_download_prompt_once()  # sẽ raise nếu thất bại

# ────────── LOAD MODEL MỘT LẦN ──────────
print("[INIT] Loading SparkTTS …", flush=True)
from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model     = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()
processor.model = model
print("[INIT] Model ready ✔", flush=True)

# ────────── HÀM ÂM THANH ──────────
def _cross_fade(a, b, sr, sec=0.12):
    n = int(sr * sec)
    if n == 0 or len(a) < n or len(b) < n:
        return np.concatenate([a, b])
    w = np.linspace(1.0, 0.0, n, dtype=np.float32)
    return np.concatenate([a[:-n], a[-n:]*w + b[:n]*w[::-1], b[n:]])

# ────────── CLONE TTS ──────────
def clone_tts(text: str) -> str:
    from preprocess import preprocess_text

    parts = preprocess_text(text.strip())
    if not parts:
        raise ValueError("`text` không được rỗng.")

    sr, merged = None, None
    for seg in parts:
        enc = processor(
            text=seg,
            prompt_speech_path=str(PROMPT_WAV),
            prompt_text=PROMPT_TRANSCRIPT,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        dec   = processor.decode(
            generated_ids=gen,
            global_token_ids_prompt=None,
            input_ids_len=enc["input_ids"].shape[-1],
        )
        audio = np.asarray(dec["audio"], dtype=np.float32)
        sr    = dec["sampling_rate"] if sr is None else sr

        if merged is None:
            merged = audio
        else:
            merged = _cross_fade(
                np.concatenate([merged, np.zeros(int(sr*0.25), np.float32)]),
                audio,
                sr,
            )

    # fade edges
    edge = np.zeros(int(sr*0.12), np.float32)
    merged = _cross_fade(edge, merged, sr, 0.04)
    merged = _cross_fade(merged, edge, sr, 0.04)

    JOBS_DIR.mkdir(exist_ok=True)
    out_path = JOBS_DIR / f"{uuid.uuid4()}.wav"
    sf.write(out_path, merged, sr)
    return str(out_path)

# ────────── RUNPOD HANDLER ──────────
def handler(event: Dict):
    try:
        return {"wav_path": clone_tts(event["input"]["text"])}
    except Exception as exc:
        return {"error": str(exc)}

runpod.serverless.start({"handler": handler})
