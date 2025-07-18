# jobs_app1_fastapi.py

import os
import uuid
import threading
import numpy as np
import torch
import soundfile as sf
import boto3

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from botocore.config import Config

from transformers import AutoProcessor, AutoModel
from preprocess import preprocess_text
from typing import Dict, Any

app = FastAPI()

# --- Initialize model & processor once ---
device = "cuda"  # hoặc "cpu" nếu bạn không có GPU
MODEL_ID = "DragonLineageAI/Vi-SparkTTS-0.5B"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = (
    AutoModel
    .from_pretrained(MODEL_ID, trust_remote_code=True)
    .to(device)
    .eval()
)
processor.model = model

# --- Global job store ---
JobStore = Dict[str, Any]
jobs: Dict[str, JobStore] = {}

# === Thiết lập S3 client để download prompt audio ===
S3_ENDPOINT    = os.environ["S3_ENDPOINT_URL"]
S3_ACCESS_KEY  = os.environ["AWS_ACCESS_KEY_ID"]
S3_SECRET_KEY  = os.environ["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
PROMPT_S3_KEY  = "tts-demo/consent_audio.wav"  # đường dẫn trong bucket

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version="s3v4"),
)

def fetch_prompt() -> str:
    """
    Tải file consent_audio.wav từ S3 vào /tmp nếu chưa có,
    rồi trả về đường dẫn local.
    """
    local_path = "/tmp/consent_audio.wav"
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(S3_BUCKET_NAME, PROMPT_S3_KEY, local_path)
    return local_path

# --- Cross‑fade helper (giữ nguyên) ---
def cross_fade(a: np.ndarray, b: np.ndarray, sr: int, fade_duration: float = 0.15) -> np.ndarray:
    fade_samples = int(sr * fade_duration)
    if fade_samples <= 0 or len(a) < fade_samples or len(b) < fade_samples:
        return np.concatenate([a, b], axis=0)
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in  = fade_out[::-1]
    a_tail = a[-fade_samples:] * fade_out
    b_head = b[:fade_samples]  * fade_in
    return np.concatenate([a[:-fade_samples], a_tail + b_head, b[fade_samples:]], axis=0)

# --- Worker function (giữ nguyên) ---
def process_job(job_id: str, text: str, prompt_path: str, prompt_transcript: str):
    jobs[job_id]['status'] = 'running'
    try:
        chunks = preprocess_text(text)
        full_audio = None
        sr = None
        global_tokens = None
        total_in, total_out = 0, 0

        for idx, chunk in enumerate(chunks):
            proc_args = {'text': chunk, 'return_tensors': 'pt'}

            # nếu có prompt_path hợp lệ, add vào proc_args
            if prompt_path and os.path.exists(prompt_path):
                proc_args['prompt_speech_path'] = prompt_path
                proc_args['prompt_text']       = prompt_transcript

            inputs = processor(**proc_args).to(device)
            in_toks = inputs['input_ids'].shape[-1]
            total_in += in_toks

            if idx == 0:
                global_tokens = inputs.pop('global_token_ids_prompt', None)
            else:
                inputs.pop('global_token_ids_prompt', None)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    do_sample=True, temperature=0.8,
                    top_k=50, top_p=0.95,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id
                )

            out_toks = output_ids.shape[-1]
            total_out += out_toks

            audio_dict = processor.decode(
                generated_ids=output_ids,
                global_token_ids_prompt=global_tokens,
                input_ids_len=in_toks
            )
            audio = np.array(audio_dict['audio'], dtype=np.float32)
            rate  = audio_dict['sampling_rate']

            if sr is None:
                sr = rate
            elif sr != rate:
                raise RuntimeError('Sampling rate mismatch')

            if full_audio is None:
                full_audio = audio
            else:
                breath = np.zeros(int(0.3 * sr), dtype=np.float32)
                temp   = np.concatenate([full_audio, breath], axis=0)
                full_audio = cross_fade(temp, audio, sr, fade_duration=0.15)

        # Thêm silence đầu/cuối
        silence = np.zeros(int(sr * 0.15), dtype=np.float32)
        full_audio = cross_fade(silence, full_audio, sr, fade_duration=0.01)
        full_audio = cross_fade(full_audio, silence, sr, fade_duration=0.01)

        out_dir = 'jobs'
        os.makedirs(out_dir, exist_ok=True)
        wav_path = os.path.join(out_dir, f'{job_id}.wav')
        sf.write(wav_path, full_audio, sr)

        jobs[job_id].update({
            'status': 'done',
            'path': wav_path,
            'total_input_tokens': total_in,
            'total_output_tokens': total_out
        })

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error']  = str(e)

# --- Endpoint: submit a TTS job ---
@app.post("/clone_tts")
async def submit_tts(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    text = data.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Missing text"}, status_code=400)

    # nếu client không gửi prompt_path thì dùng file mặc định từ S3
    prompt_path = data.get("prompt_path") or fetch_prompt()
    prompt_transcript = data.get(
        "prompt_transcript",
        "Tôi là chủ sở hữu giọng nói này và tôi đồng ý cho Google sử dụng giọng nói này để tạo mô hình giọng nói tổng hợp."
    )

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    background_tasks.add_task(process_job, job_id, text, prompt_path, prompt_transcript)

    return JSONResponse({"job_id": job_id}, status_code=202)

# --- Endpoint: get job status ---
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    resp = {"job_id": job_id, "status": job["status"]}
    if job.get("error"):
        resp["error"] = job["error"]
    return resp

# --- Endpoint: download result ---
@app.get("/result/{job_id}")
async def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        return JSONResponse(
            {"error": "Result not ready", "status": job["status"]},
            status_code=202
        )
    return FileResponse(job["path"], media_type="audio/wav")

# --- Chạy trực tiếp bằng Python ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "jobs_app1_fastapi:app",
        host="0.0.0.0",
        port=5003,
        log_level="info",
        reload=True,    # chỉ dev
    )
