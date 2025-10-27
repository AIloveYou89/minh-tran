# rp_handler.py
import os, io, uuid, base64, numpy as np, soundfile as sf, torch, torchaudio
from transformers import AutoProcessor, AutoModel
import runpod

# ===== Config =====
MODEL_ID  = os.getenv("MODEL_ID", "DragonLineageAI/Vi-SparkTTS-0.5B")
HF_TOKEN  = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Missing HF_TOKEN env."

# GPU-only
assert torch.cuda.is_available(), "CUDA/GPU not available."
DEVICE    = "cuda"
TARGET_SR = int(os.getenv("TARGET_SR", "24000"))

# Network Volume mount: /runpod-volume (Runpod)
NV_ROOT    = os.getenv("NV_ROOT", "/runpod-volume")
CONSENT_PATH = os.getenv("CONSENT_LOCAL", os.path.join(NV_ROOT, "fixed/consent_audio.wav"))
OUT_DIR      = os.getenv("OUT_DIR", os.path.join(NV_ROOT, "jobs"))

# ===== Simple preprocess (như bạn yêu cầu gộp vào 1 file) =====
def preprocess_text(text: str):
    # cắt câu nhẹ nhàng, bỏ khoảng trắng thừa
    seps = ".!?;\n"
    buf, out = "", []
    for ch in text:
        buf += ch
        if ch in seps:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out

# ===== Helpers =====
def join_with_gap(a: np.ndarray | None, b: np.ndarray, gap_sec=0.2, fade_sec=0.1, sr=TARGET_SR):
    if a is None: return b
    gap = np.zeros(int(gap_sec * sr), dtype=np.float32)
    x   = np.concatenate([a, gap, b], axis=0)
    n   = int(fade_sec * sr)
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

# ===== Load model once =====
print(f"[INIT] Loading {MODEL_ID} on {DEVICE}", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_auth_token=HF_TOKEN)
model     = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, use_auth_token=HF_TOKEN).to(DEVICE).eval()

if os.path.exists(CONSENT_PATH):
    print(f"[CONSENT] {CONSENT_PATH}", flush=True)
else:
    print("[CONSENT] Not found; proceed without prompt voice.", flush=True)
    CONSENT_PATH = None

# ===== Runpod handler =====
def handler(job):
    """
    input:
      text: string (bắt buộc)
      gap_sec: float (default 0.2)
      fade_sec: float (default 0.1)
      return: "path" | "base64" (default "path")
      outfile: optional filename
    """
    inp = job["input"] or {}
    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "Missing 'text'."}

    gap_sec  = float(inp.get("gap_sec", 0.2))
    fade_sec = float(inp.get("fade_sec", 0.1))
    ret_mode = inp.get("return", "path")
    outfile  = inp.get("outfile")

    chunks = [c.strip() for c in preprocess_text(text) if c.strip()]
    full, sr = None, TARGET_SR

    for ck in chunks:
        kw = {"text": ck, "return_tensors": "pt"}
        if CONSENT_PATH and os.path.exists(CONSENT_PATH):
            kw["prompt_speech_path"] = CONSENT_PATH
            kw["prompt_text"] = "Consent voice"
        inputs = processor(**kw)
        inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
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
        sr_in = int(audio_dict.get("sampling_rate", TARGET_SR))
        if sr_in != TARGET_SR:
            audio = resample_np(audio, sr_in, TARGET_SR)
        if audio is not None and audio.size:
            full = join_with_gap(full, audio, gap_sec, fade_sec, TARGET_SR)

    if full is None:
        return {"error": "TTS failed or empty output"}

    full = normalize_peak(full)
    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())
    name = outfile or f"{job_id}.wav"
    out_path = os.path.join(OUT_DIR, name)
    sf.write(out_path, full, TARGET_SR)

    if ret_mode == "base64":
        with io.BytesIO() as buf:
            sf.write(buf, full, TARGET_SR, format="WAV")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "job_id": job_id,
            "audio_b64": b64,
            "sample_rate": TARGET_SR,
            "seconds": round(len(full)/TARGET_SR, 2)
        }

    return {
        "job_id": job_id,
        "path": out_path,
        "sample_rate": TARGET_SR,
        "seconds": round(len(full)/TARGET_SR, 2)
    }

# Bắt buộc cho queue-based worker
runpod.serverless.start({"handler": handler})
