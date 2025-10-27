import os, io, uuid, base64, numpy as np, soundfile as sf, torch, torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoProcessor, AutoModel
from preprocess import preprocess_text

# ===== Config =====
MODEL_ID  = os.getenv("MODEL_ID", "DragonLineageAI/Vi-SparkTTS-0.5B")
HF_TOKEN  = os.getenv("HF_TOKEN")  # token đã cài sẵn khi deploy
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required but not detected.")
DEVICE    = "cuda"
TARGET_SR = int(os.getenv("TARGET_SR", "24000"))

# Network Volume mount (Serverless): /runpod-volume
CONSENT_PATH = os.getenv("CONSENT_LOCAL", "/runpod-volume/minh-tran/tts-lates/fixed/consent_audio.wav")
OUT_DIR      = os.getenv("OUT_DIR", "/runpod-volume/minh-tran/tts-lates/jobs")

app = FastAPI()
jobs = {}

# ===== Helpers =====
def join_with_gap(a: np.ndarray | None, b: np.ndarray,
                  gap_sec=0.2, fade_sec=0.1, sr=TARGET_SR):
    if a is None:
        return b
    gap  = np.zeros(int(gap_sec * sr), dtype=np.float32)
    x    = np.concatenate([a, gap, b], axis=0)
    n    = int(fade_sec * sr)
    if n > 0 and len(a) > n and len(b) > n:
        fo = np.linspace(1, 0, n, dtype=np.float32)
        fi = 1 - fo
        x[len(a)-n:len(a)] *= fo
        x[len(a)+len(gap):len(a)+len(gap)+n] *= fi
    return x

def normalize_peak(x: np.ndarray, peak=0.98):
    if x is None or x.size == 0: return x
    m = float(np.max(np.abs(x)))
    return (x / m * peak).astype(np.float32) if m > 0 else x

def resample_np(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out: return x
    wav = torch.from_numpy(x).unsqueeze(0)
    out = torchaudio.functional.resample(wav, sr_in, sr_out)
    return out.squeeze(0).numpy()

# ===== Load model (GPU only) =====
print(f"[INIT] Loading {MODEL_ID} on {DEVICE}", flush=True)
processor = AutoProcessor.from_pretrained(
    MODEL_ID, trust_remote_code=True, token=HF_TOKEN
)
model = AutoModel.from_pretrained(
    MODEL_ID, trust_remote_code=True, token=HF_TOKEN
).to(DEVICE).eval()

if os.path.exists(CONSENT_PATH):
    print(f"[CONSENT] Using {CONSENT_PATH}", flush=True)
else:
    print("[CONSENT] Not found; run without prompt voice.", flush=True)
    CONSENT_PATH = None

@app.get("/ping")
async def ping():
    return {"status": "healthy"}

@app.post("/tts")
async def tts(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "Missing text"}, status_code=400)

    gap_sec  = float(payload.get("gap_sec", 0.2))
    fade_sec = float(payload.get("fade_sec", 0.1))
    ret_mode = payload.get("return", "path")  # 'path' | 'base64'
    outfile  = payload.get("outfile")

    chunks = [c.strip() for c in preprocess_text(text) if c.strip()]
    full, sr = None, TARGET_SR

    for ck in chunks:
        kw = {"text": ck, "return_tensors": "pt"}
        if CONSENT_PATH and os.path.exists(CONSENT_PATH):
            kw["prompt_speech_path"] = CONSENT_PATH
            kw["prompt_text"] = "Consent voice"
        inputs = processor(**kw).to(DEVICE)
        global_tokens = inputs.pop("global_token_ids_prompt", None)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs, do_sample=True, temperature=0.7, top_p=0.9,
                repetition_penalty=1.05, max_new_tokens=1800,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        audio_dict = processor.decode(
            generated_ids=out_ids,
            global_token_ids_prompt=global_tokens,
            input_ids_len=inputs["input_ids"].shape[-1],
            return_type="np"
        )
        audio = np.asarray(audio_dict["audio"], dtype=np.float32)
        sr    = int(audio_dict.get("sampling_rate", TARGET_SR))
        if sr != TARGET_SR:
            audio = resample_np(audio, sr, TARGET_SR)
            sr = TARGET_SR
        if audio is not None and audio.size:
            full = join_with_gap(full, audio, gap_sec, fade_sec, sr)

    if full is None:
        return JSONResponse({"error": "TTS failed or empty output"}, status_code=500)

    full = normalize_peak(full)
    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())
    name = outfile or f"{job_id}.wav"
    out_path = os.path.join(OUT_DIR, name)
    sf.write(out_path, full, sr)

    if ret_mode == "base64":
        with io.BytesIO() as buf:
            sf.write(buf, full, sr, format="WAV")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"job_id": job_id, "audio_b64": b64, "sample_rate": sr,
                "seconds": round(len(full)/sr, 2)}

    return {"job_id": job_id, "path": out_path, "sample_rate": sr,
            "seconds": round(len(full)/sr, 2)}

@app.get("/result/{job_id}")
async def result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Not found")
    p = job.get("path")
    if not p or not os.path.exists(p):
        raise HTTPException(404, "file missing")
    return FileResponse(p, media_type="audio/wav", filename=os.path.basename(p))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "80"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
