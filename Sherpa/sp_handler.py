# RunPod Serverless: Vietnamese ASR (Sherpa-ONNX, Zipformer-RNNT)
import os, time, uuid, base64, logging, subprocess, json
import runpod
from huggingface_hub import snapshot_download

# ------------------------------
# CONFIG (đồng bộ style Spark)
# ------------------------------
MODEL_ID  = os.getenv("MODEL_ID", "hynt/Zipformer-30M-RNNT-6000h")
MODEL_DIR = os.getenv("MODEL_DIR", "/models/Zipformer-30M-RNNT-6000h")
OUT_DIR   = os.getenv("OUT_DIR", "/runpod-volume/jobs")
NUM_THREADS = os.getenv("NUM_THREADS")  # vd: "1" | "2" | ...
HF_TOKEN = os.getenv("HF_TOKEN")  # nếu repo private thì set thêm

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SherpaASR")

# ------------------------------
# MODEL DOWNLOAD (1 lần)
# ------------------------------
def ensure_model():
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        log.info(f"[MODEL] Downloading {MODEL_ID} → {MODEL_DIR}")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )
    else:
        log.info(f"[MODEL] Found model at {MODEL_DIR}")

def pick_tokens_path(model_dir: str) -> str:
    # Ưu tiên tokens.txt theo khuyến nghị sherpa-onnx
    cand = [
        os.path.join(model_dir, "tokens.txt"),
        os.path.join(model_dir, "config.json"),  # fallback nếu model dùng json
        os.path.join(model_dir, "bpe.model"),    # fallback khác (một số model)
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find tokens file (tokens.txt/config.json/bpe.model) in model dir")

ensure_model()

# ------------------------------
# CORE HANDLER
# ------------------------------
def handler(job):
    """
    Input JSON:
    {
      "audio_path": "/runpod-volume/audio/test.wav",
      "return": "text" | "base64",
      "outfile": "optional_output.txt"
    }
    """
    inp = job.get("input", {})
    audio_path = inp.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        return {"error": f"Audio not found: {audio_path}"}

    job_id = str(uuid.uuid4())
    out_name = inp.get("outfile") or f"{job_id}.txt"
    out_path = os.path.join(OUT_DIR, out_name)

    # Chuẩn hoá audio: mono + 16-bit; sample rate không bắt buộc 16k nhưng an toàn. :contentReference[oaicite:3]{index=3}
    fixed_wav = f"/tmp/{job_id}_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", fixed_wav],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    tokens_path = pick_tokens_path(MODEL_DIR)
    encoder = os.path.join(MODEL_DIR, "encoder-epoch-20-avg-10.onnx")
    decoder = os.path.join(MODEL_DIR, "decoder-epoch-20-avg-10.onnx")
    joiner  = os.path.join(MODEL_DIR, "joiner-epoch-20-avg-10.onnx")

    cmd = [
        "sherpa-onnx-offline",
        f"--tokens={tokens_path}",
        f"--encoder={encoder}",
        f"--decoder={decoder}",
        f"--joiner={joiner}",
        fixed_wav
    ]
    if NUM_THREADS:
        cmd.insert(1, f"--num-threads={NUM_THREADS}")  # CLI có tham số num-threads; ví dụ trong docs. :contentReference[oaicite:4]{index=4}

    log.info(f"[JOB] Decoding {audio_path} (job {job_id})")
    t0 = time.time()
    result = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - t0

    # Trích transcript từ stdout: sherpa-onnx in một dòng JSON có "text"/"timestamps" khi decode offline. :contentReference[oaicite:5]{index=5}
    transcript_lines = []
    for line in result.stdout.splitlines():
        if '"text"' in line or '"result"' in line:
            transcript_lines.append(line.strip())
    transcript = "\n".join(transcript_lines).strip()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    log.info(f"[DONE] {audio_path} → {out_path} ({elapsed:.2f}s)")

    if inp.get("return") == "base64":
        b64 = base64.b64encode(transcript.encode("utf-8")).decode("utf-8")
        return {"job_id": job_id, "elapsed_sec": round(elapsed, 2), "text_b64": b64}
    return {"job_id": job_id, "elapsed_sec": round(elapsed, 2), "path": out_path, "text": transcript}

# ------------------------------
# START WORKER (RunPod handler)
# ------------------------------
# Kiểu handler này là đúng chuẩn serverless của RunPod. :contentReference[oaicite:6]{index=6}
runpod.serverless.start({"handler": handler})
