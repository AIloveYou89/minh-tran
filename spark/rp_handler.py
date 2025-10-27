import os, io, base64, numpy as np, soundfile as sf, torch, torchaudio
import runpod
from transformers import AutoProcessor, AutoModel
from preprocess import preprocess_text

MODEL_ID  = os.getenv("MODEL_ID", "DragonLineageAI/Vi-SparkTTS-0.5B")
HF_TOKEN  = os.getenv("HUGGING_FACE_HUB_TOKEN") or None
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = int(os.getenv("TARGET_SR", "24000"))

CONSENT_PATH = os.getenv("CONSENT_LOCAL", "/runpod-volume/consent_audio.wav")
OUT_DIR      = os.getenv("OUT_DIR", "/runpod-volume/tts-out")

def join_with_gap(a, b, gap_sec=0.2, fade_sec=0.1, sr=TARGET_SR):
    if a is None: return b
    gap = np.zeros(int(gap_sec * sr), dtype=np.float32)
    x = np.concatenate([a, gap, b], axis=0)
    n = int(fade_sec * sr)
    if n > 0 and len(a) > n and len(b) > n:
        fo = np.linspace(1, 0, n, dtype=np.float32)
        fi = 1 - fo
        x[len(a)-n:len(a)] *= fo
        x[len(a)+len(gap):len(a)+len(gap)+n] *= fi
    return x

def normalize_peak(x, peak=0.98):
    if x is None or x.size == 0: return x
    m = float(np.max(np.abs(x)))
    return (x / m * peak).astype(np.float32) if m > 0 else x

def resample_np(x, sr_in, sr_out):
    if sr_in == sr_out: return x
    wav = torch.from_numpy(x).unsqueeze(0)
    out = torchaudio.functional.resample(wav, sr_in, sr_out)
    return out.squeeze(0).numpy()

print(f"[INIT] Loading {MODEL_ID} on {DEVICE}", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_auth_token=HF_TOKEN)
model     = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, use_auth_token=HF_TOKEN).to(DEVICE).eval()

if os.path.exists(CONSENT_PATH):
    print(f"[CONSENT] Using {CONSENT_PATH}", flush=True)
else:
    print("[CONSENT] Not found; proceed without prompt.", flush=True)
    CONSENT_PATH = None

def _speak_chunk(text, prompt_path):
    kw = {"text": text, "return_tensors": "pt"}
    if prompt_path:
        kw["prompt_speech_path"] = prompt_path
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
    return audio, sr

def synthesize(text, prompt_path, gap_sec=0.2, fade_sec=0.1):
    chunks = [c.strip() for c in preprocess_text(text) if c.strip()]
    full, sr = None, TARGET_SR
    for ck in chunks:
        au, sr = _speak_chunk(ck, prompt_path)
        if au is not None and au.size:
            full = join_with_gap(full, au, gap_sec, fade_sec, sr)
    return (normalize_peak(full), sr) if full is not None else (None, sr)

def handler(job):
    inp = job.get("input", {})
    text = (inp.get("text") or "").strip()
    if not text: return {"error": "Missing input.text"}

    gap_sec  = float(inp.get("gap_sec", 0.2))
    fade_sec = float(inp.get("fade_sec", 0.1))
    ret_mode = inp.get("return", "path")
    outfile  = inp.get("outfile")

    audio, sr = synthesize(text, CONSENT_PATH, gap_sec, fade_sec)
    if audio is None: return {"error": "TTS failed or empty output"}

    os.makedirs(OUT_DIR, exist_ok=True)
    name = outfile or f"{job['id']}.wav"
    out_path = os.path.join(OUT_DIR, name)

    with io.BytesIO() as buf:
        sf.write(buf, audio, sr, format="WAV")
        raw = buf.getvalue()
    with open(out_path, "wb") as f:
        f.write(raw)

    if ret_mode == "base64":
        b64 = base64.b64encode(raw).decode("utf-8")
        return {"audio_b64": b64, "sample_rate": sr, "seconds": round(len(audio)/sr, 2), "path": out_path}

    return {"path": out_path, "sample_rate": sr, "seconds": round(len(audio)/sr, 2)}

runpod.serverless.start({"handler": handler})
