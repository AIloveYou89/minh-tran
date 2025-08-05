"""
handler.py
=======
• Load model SparkTTS 1 lần khi container start
• clone_tts(text) sinh file WAV
• handler(event) cho Runpod Serverless
Không cần FastAPI, không mở cổng HTTP.
"""

import os
import uuid
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
import torch
import runpod                            # SDK Runpod Serverless

# ---------- Cấu hình đường dẫn ----------
BASE_DIR   = Path("/workspace/minh-tran/tts-lates")
PROMPT_WAV = BASE_DIR / "consent_audio.wav"
JOBS_DIR   = BASE_DIR / "jobs"
MODEL_ID   = "DragonLineageAI/Vi-SparkTTS-0.5B"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TRANSCRIPT = (
    "Tôi là chủ sở hữu giọng nói này và tôi đồng ý cho Google sử dụng "
    "giọng nói này để tạo mô hình giọng nói tổng hợp."
)

# ---------- Đảm bảo file prompt ----------
if not PROMPT_WAV.exists():
    raise FileNotFoundError(
        f"Prompt WAV thiếu: {PROMPT_WAV}. "
        "Hãy commit file này hoặc cấu hình tải từ S3."
    )

# ---------- Load model ----------
print("[INIT] Loading SparkTTS model …", flush=True)
from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()
processor.model = model
print("[INIT] Model loaded ✔", flush=True)


# ---------- Hàm cross-fade ----------
def _cross_fade(a: np.ndarray, b: np.ndarray, sr: int, sec: float = 0.12):
    n = int(sr * sec)
    if n == 0 or len(a) < n or len(b) < n:
        return np.concatenate([a, b])
    fade_out = np.linspace(1, 0, n, dtype=np.float32)
    fade_in  = fade_out[::-1]
    blend = a[-n:] * fade_out + b[:n] * fade_in
    return np.concatenate([a[:-n], blend, b[n:]])


# ---------- clone_tts ----------
def clone_tts(text: str) -> str:
    """
    Tạo WAV từ đoạn `text`, trả path tuyệt đối tới file.
    Không cần truyền prompt_path / transcript – đã cố định.
    """
    text = text.strip()
    if not text:
        raise ValueError("`text` must không rỗng.")

    from preprocess import preprocess_text   # import chậm

    chunks = preprocess_text(text)
    sr = None
    audio_full = None

    for idx, chunk in enumerate(chunks):
        enc = processor(
            text=chunk,
            prompt_speech_path=str(PROMPT_WAV),
            prompt_text=PROMPT_TRANSCRIPT,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        decoded = processor.decode(
            generated_ids=gen_ids,
            global_token_ids_prompt=None,
            input_ids_len=enc["input_ids"].shape[-1],
        )

        audio = np.array(decoded["audio"], dtype=np.float32)
        rate  = decoded["sampling_rate"]

        if sr is None:
            sr = rate
        if audio_full is None:
            audio_full = audio
        else:
            breath = np.zeros(int(0.25 * sr), dtype=np.float32)
            audio_full = _cross_fade(
                np.concatenate([audio_full, breath]), audio, sr
            )

    # Fade-in/out nhẹ
    edge = np.zeros(int(sr * 0.12), dtype=np.float32)
    audio_full = _cross_fade(edge, audio_full, sr, 0.04)
    audio_full = _cross_fade(audio_full, edge, sr, 0.04)

    JOBS_DIR.mkdir(exist_ok=True)
    out_path = JOBS_DIR / f"{uuid.uuid4()}.wav"
    sf.write(out_path, audio_full, sr)
    print(f"[TTS] Saved {out_path}", flush=True)
    return str(out_path)


# ---------- Runpod handler ----------
def handler(event: Dict):
    """
    event: {"input": {"text": "Xin chào"}}
    Trả:   {"wav_path": "..."} hoặc {"error": "..."}
    """
    try:
        text = event["input"]["text"]
        wav  = clone_tts(text)
        return {"wav_path": wav}
    except Exception as exc:
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
