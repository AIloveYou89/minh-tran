# jobs_app_on_demand.py
import os, uuid, numpy as np, torch, soundfile as sf
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoProcessor, AutoModel
from typing import Dict, Any
from preprocess import preprocess_text  # phải trả về list chunks (hoặc (chunks, meta) -> chỉnh bên dưới nếu cần)

MODEL_ID = "DragonLineageAI/Vi-SparkTTS-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prompt / transcript
LOCAL_PROMPT_PATH = "/workspace/minh-tran/tts-lates/fixed/consent_audio.wav"
DEFAULT_PROMPT_TRANSCRIPT = (
    "Tôi là chủ sở hữu giọng nói này và tôi đồng ý cho Google sử dụng "
    "giọng nói này để tạo mô hình giọng nói tổng hợp."
)

# Bắt buộc có prompt
REQUIRE_PROMPT = True

# S3 env (tuỳ chọn)
S3_ENDPOINT_URL      = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME       = os.getenv("S3_BUCKET_NAME")
S3_PROMPT_KEY        = os.getenv("S3_PROMPT_KEY")

app = FastAPI()

print(f"[INIT] Loading model {MODEL_ID} on {DEVICE}", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()
processor.model = model
print("[INIT] Model loaded.", flush=True)

def ensure_prompt_audio() -> str:
    if os.path.exists(LOCAL_PROMPT_PATH):
        print(f"[PROMPT] Local found: {LOCAL_PROMPT_PATH}", flush=True)
        return LOCAL_PROMPT_PATH

    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_key_id   = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret   = os.environ.get("AWS_SECRET_ACCESS_KEY")
    s3_bucket   = os.environ.get("S3_BUCKET_NAME")
    s3_key      = os.environ.get("S3_PROMPT_KEY")

    if all([s3_endpoint, s3_key_id, s3_secret, s3_bucket, s3_key]):
        try:
            import boto3
            from botocore.config import Config
            local_tmp = "/tmp/prompt_clone.wav"
            if not os.path.exists(local_tmp):
                print(f"[PROMPT] Downloading from S3 s3://{s3_bucket}/{s3_key}", flush=True)
                s3 = boto3.client(
                    "s3",
                    endpoint_url=s3_endpoint,
                    aws_access_key_id=s3_key_id,
                    aws_secret_access_key=s3_secret,
                    config=Config(signature_version="s3v4"),
                )
                s3.download_file(s3_bucket, s3_key, local_tmp)
                print(f"[PROMPT] Downloaded: {local_tmp}", flush=True)
            else:
                print(f"[PROMPT] Using cached S3 file: {local_tmp}", flush=True)
            return local_tmp
        except Exception as e:
            print(f"[PROMPT] S3 download failed: {e}", flush=True)

    raise FileNotFoundError(
        f"Prompt audio not found locally ({LOCAL_PROMPT_PATH}) and S3 download failed / not configured."
    )

PROMPT_PATH = ensure_prompt_audio()
print(f"[PROMPT] Using prompt audio: {PROMPT_PATH}", flush=True)

jobs: Dict[str, Dict[str, Any]] = {}

def cross_fade(a: np.ndarray, b: np.ndarray, sr: int, fade_duration: float = 0.15):
    fade_samples = int(sr * fade_duration)
    if fade_samples <= 0 or len(a) < fade_samples or len(b) < fade_samples:
        return np.concatenate([a, b], axis=0)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in  = fade_out[::-1]
    blend    = a[-fade_samples:] * fade_out + b[:fade_samples] * fade_in
    return np.concatenate([a[:-fade_samples], blend, b[fade_samples:]], axis=0)

def process_job(job_id: str, text: str, prompt_path: str, prompt_transcript: str):
    jobs[job_id]["status"] = "running"
    try:
        chunks = preprocess_text(text)
        print(f"[JOB {job_id}] Total chunks: {len(chunks)}", flush=True)
        for i, c in enumerate(chunks):
            print(f"[JOB {job_id}] CHUNK{i}: {repr(c)}", flush=True)

        full_audio = None
        sr = None
        global_tokens = None
        total_in = total_out = 0

        for idx, chunk in enumerate(chunks):
            proc_args = {"text": chunk, "return_tensors": "pt", "prompt_speech_path": prompt_path, "prompt_text": prompt_transcript}
            inputs = processor(**proc_args).to(DEVICE)
            in_tok = inputs["input_ids"].shape[-1]
            total_in += in_tok

            if idx == 0:
                global_tokens = inputs.pop("global_token_ids_prompt", None)
            else:
                _ = inputs.pop("global_token_ids_prompt", None)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
            out_tok = output_ids.shape[-1]
            total_out += out_tok

            audio_dict = processor.decode(
                generated_ids=output_ids,
                global_token_ids_prompt=global_tokens,
                input_ids_len=in_tok
            )
            audio = np.array(audio_dict["audio"], dtype=np.float32)
            rate  = audio_dict["sampling_rate"]

            if sr is None:
                sr = rate
            elif sr != rate:
                raise RuntimeError("Sampling rate mismatch")

            if full_audio is None:
                full_audio = audio
            else:
                breath = np.zeros(int(0.25 * sr), dtype=np.float32)
                full_audio = cross_fade(np.concatenate([full_audio, breath], axis=0), audio, sr, fade_duration=0.12)

            print(f"[JOB {job_id}] Chunk {idx+1}/{len(chunks)} in={in_tok} out={out_tok}", flush=True)

        if full_audio is not None:
            edge = np.zeros(int(sr * 0.12), dtype=np.float32)
            full_audio = cross_fade(edge, full_audio, sr, 0.04)
            full_audio = cross_fade(full_audio, edge, sr, 0.04)

        os.makedirs("jobs", exist_ok=True)
        out_path = f"jobs/{job_id}.wav"
        sf.write(out_path, full_audio, sr)

        jobs[job_id].update({
            "status": "done",
            "path": out_path,
            "num_chunks": len(chunks),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out
        })
        print(f"[JOB {job_id}] DONE -> {out_path}", flush=True)
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        print(f"[JOB {job_id}] ERROR: {e}", flush=True)

@app.post("/clone_tts")
async def clone_tts(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    text = data.get("text","").strip()
    if not text:
        return JSONResponse({"error":"Missing text"}, status_code=400)

    req_prompt = data.get("prompt_path")
    effective_prompt = PROMPT_PATH
    if req_prompt:
        if os.path.exists(req_prompt):
            effective_prompt = req_prompt
            print(f"[PROMPT] Override with request path: {effective_prompt}", flush=True)
        else:
            return JSONResponse({"error": f"Provided prompt_path not found: {req_prompt}"}, status_code=400)

    if not os.path.exists(effective_prompt):
        return JSONResponse({"error": f"Prompt file not found: {effective_prompt}"}, status_code=400)

    prompt_transcript = data.get("prompt_transcript") or DEFAULT_PROMPT_TRANSCRIPT

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "prompt_path": effective_prompt}
    background_tasks.add_task(process_job, job_id, text, effective_prompt, prompt_transcript)
    return JSONResponse({"job_id": job_id, "using_prompt": True}, status_code=202)

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, **job}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        return JSONResponse({"error":"Result not ready", "status":job["status"]}, status_code=202)
    return FileResponse(job["path"], media_type="audio/wav", filename=f"{job_id}.wav")

@app.get("/health")
async def health():
    return {"status":"ok", "model": MODEL_ID, "device": DEVICE, "prompt_exists": os.path.exists(PROMPT_PATH)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("jobs_app_on_demand:app", host="0.0.0.0", port=5300, log_level="info")
