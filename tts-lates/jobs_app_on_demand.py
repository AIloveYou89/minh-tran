import os
import uuid
import threading
import numpy as np
import torch
import soundfile as sf

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoProcessor, AutoModel
from preprocess import preprocess_text
from typing import Dict, Any

app = FastAPI()

# ================== MODEL SETUP ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "DragonLineageAI/Vi-SparkTTS-0.5B"
print(f"[INIT] Loading model {MODEL_ID} on {DEVICE}", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()
processor.model = model
print("[INIT] Model loaded.", flush=True)

# ================== PROMPT AUDIO (DEFAULT) ==================
# Đường dẫn file WAV đã clone sẵn trong Network Storage
LOCAL_PROMPT_PATH = "/workspace/minh-tran/tts-lates/consent_audio.wav"

DEFAULT_PROMPT_TRANSCRIPT = (
    "Tôi là chủ sở hữu giọng nói này và tôi đồng ý cho Google sử dụng "
    "giọng nói này để tạo mô hình giọng nói tổng hợp."
)

# --- Nếu có cấu hình S3 (tuỳ chọn) sẽ tải file prompt về /tmp/prompt.wav ---
def ensure_prompt_audio() -> str:
    """
    Trả về đường dẫn local tới prompt audio.
    Ưu tiên: 
        1) Nếu đã có ở LOCAL_PROMPT_PATH -> dùng luôn.
        2) Nếu có biến môi trường S3_* thì thử tải từ S3 về /tmp/prompt.wav.
    """
    if os.path.exists(LOCAL_PROMPT_PATH):
        return LOCAL_PROMPT_PATH

    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_key_id   = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret   = os.environ.get("AWS_SECRET_ACCESS_KEY")
    s3_bucket   = os.environ.get("S3_BUCKET_NAME")
    s3_key      = os.environ.get("S3_PROMPT_KEY")  # bạn có thể set biến này, ví dụ: tts-demo/consent_audio.wav

    if all([s3_endpoint, s3_key_id, s3_secret, s3_bucket, s3_key]):
        try:
            import boto3
            from botocore.config import Config
            local_tmp = "/tmp/prompt_clone.wav"
            if not os.path.exists(local_tmp):
                print("[PROMPT] Downloading from S3...", flush=True)
                s3 = boto3.client(
                    "s3",
                    endpoint_url=s3_endpoint,
                    aws_access_key_id=s3_key_id,
                    aws_secret_access_key=s3_secret,
                    config=Config(signature_version="s3v4"),
                )
                s3.download_file(s3_bucket, s3_key, local_tmp)
                print(f"[PROMPT] Downloaded to {local_tmp}", flush=True)
            return local_tmp
        except Exception as e:
            print(f"[PROMPT] S3 download failed: {e}", flush=True)

    # Nếu không tìm thấy file
    raise FileNotFoundError(
        f"Không tìm thấy prompt audio tại {LOCAL_PROMPT_PATH} và không tải được từ S3. "
        "Hãy chắc chắn file WAV tồn tại hoặc cấu hình S3."
    )

PROMPT_PATH = ensure_prompt_audio()
print(f"[PROMPT] Using prompt audio: {PROMPT_PATH}", flush=True)

# ================== JOB STORE ==================
jobs: Dict[str, Dict[str, Any]] = {}

# ================== AUDIO UTILS ==================
def cross_fade(a: np.ndarray, b: np.ndarray, sr: int, fade_duration: float = 0.15) -> np.ndarray:
    fade_samples = int(sr * fade_duration)
    if fade_samples <= 0 or len(a) < fade_samples or len(b) < fade_samples:
        return np.concatenate([a, b], axis=0)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in = fade_out[::-1]
    blended = a[-fade_samples:] * fade_out + b[:fade_samples] * fade_in
    return np.concatenate([a[:-fade_samples], blended, b[fade_samples:]], axis=0)

# ================== WORKER ==================
def process_job(job_id: str, text: str, prompt_path: str, prompt_transcript: str):
    jobs[job_id]["status"] = "running"
    try:
        chunks = preprocess_text(text)
        print(f"[JOB {job_id}] {len(chunks)} chunk(s). Using prompt: {prompt_path}", flush=True)

        full_audio = None
        sr = None
        global_tokens = None
        total_in = total_out = 0

        for idx, chunk in enumerate(chunks):
            proc_args = {"text": chunk, "return_tensors": "pt"}
            # Luôn dùng prompt mặc định (trừ khi người dùng override hợp lệ)
            if prompt_path and os.path.exists(prompt_path):
                proc_args["prompt_speech_path"] = prompt_path
                proc_args["prompt_text"] = prompt_transcript

            inputs = processor(**proc_args).to(DEVICE)
            in_tokens = inputs["input_ids"].shape[-1]
            total_in += in_tokens

            if idx == 0:
                global_tokens = inputs.pop("global_token_ids_prompt", None)
            else:
                inputs.pop("global_token_ids_prompt", None)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            out_tokens = output_ids.shape[-1]
            total_out += out_tokens

            audio_dict = processor.decode(
                generated_ids=output_ids,
                global_token_ids_prompt=global_tokens,
                input_ids_len=in_tokens
            )

            audio = np.array(audio_dict["audio"], dtype=np.float32)
            rate = audio_dict["sampling_rate"]
            if sr is None:
                sr = rate
            elif sr != rate:
                raise RuntimeError("Sampling rate mismatch")

            if full_audio is None:
                full_audio = audio
            else:
                breath = np.zeros(int(0.3 * sr), dtype=np.float32)
                full_audio = cross_fade(
                    np.concatenate([full_audio, breath], axis=0),
                    audio,
                    sr,
                    fade_duration=0.15
                )

            print(f"[JOB {job_id}] Chunk {idx+1}/{len(chunks)} in={in_tokens} out={out_tokens}", flush=True)

        silence = np.zeros(int(sr * 0.15), dtype=np.float32)
        full_audio = cross_fade(silence, full_audio, sr, 0.01)
        full_audio = cross_fade(full_audio, silence, sr, 0.01)

        os.makedirs("jobs", exist_ok=True)
        out_path = f"jobs/{job_id}.wav"
        sf.write(out_path, full_audio, sr)

        jobs[job_id].update({
            "status": "done",
            "path": out_path,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
        })
        print(f"[JOB {job_id}] DONE -> {out_path}", flush=True)

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        print(f"[JOB {job_id}] ERROR: {e}", flush=True)

# ================== ENDPOINTS ==================
@app.post("/clone_tts")
async def clone_tts(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    text = data.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Missing text"}, status_code=400)

    # Cho phép override *tuỳ chọn*, nếu không gửi sẽ dùng PROMPT_PATH & DEFAULT_PROMPT_TRANSCRIPT
    prompt_path = data.get("prompt_path") or PROMPT_PATH
    prompt_transcript = data.get("prompt_transcript") or DEFAULT_PROMPT_TRANSCRIPT

    if not os.path.exists(prompt_path):
        return JSONResponse({"error": f"Prompt file not found: {prompt_path}"}, status_code=400)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    background_tasks.add_task(process_job, job_id, text, prompt_path, prompt_transcript)
    return JSONResponse({"job_id": job_id}, status_code=202)

@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, **job}

@app.get("/result/{job_id}")
async def result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        return JSONResponse({"error": "Result not ready", "status": job["status"]}, status_code=202)
    return FileResponse(job["path"], media_type="audio/wav")

# ================== MAIN (dev) ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("jobs_app_on_demand:app", host="0.0.0.0", port=5300, log_level="info")
