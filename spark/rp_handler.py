# rp_handler.py - Fixed version based on rendervideo.py
import os, io, uuid, base64, numpy as np, soundfile as sf, torch, torchaudio
from transformers import AutoProcessor, AutoModel
import runpod
import gc
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import preprocess từ file riêng
try:
    from preprocess import preprocess_text
    logger.info("[INIT] preprocess module loaded successfully")
except ImportError:
    logger.error("[INIT] ERROR: preprocess.py is required!")
    raise ImportError("preprocess.py not found! This file is required for text chunking.")

# ===== Config =====
MODEL_ID  = os.getenv("MODEL_ID", "DragonLineageAI/Vi-SparkTTS-0.5B")
HF_TOKEN  = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Missing HF_TOKEN env."

# GPU-only
assert torch.cuda.is_available(), "CUDA/GPU not available."
DEVICE    = "cuda"
TARGET_SR = int(os.getenv("TARGET_SR", "24000"))

# Network Volume mount
NV_ROOT    = os.getenv("NV_ROOT", "/runpod-volume")
CONSENT_PATH = os.getenv("CONSENT_LOCAL", os.path.join(NV_ROOT, "fixed/consent_audio.wav"))
OUT_DIR      = os.getenv("OUT_DIR", os.path.join(NV_ROOT, "jobs"))

# ===== Helper Functions =====
def calc_max_new(text: str, in_tok: int, base_ratio: float = 2.5, cap: int = 1800) -> int:
    """Tính max_new_tokens thông minh dựa vào độ dài text"""
    text_len = len(text.strip())
    
    if text_len < 50:
        ratio = base_ratio * 1.5
    elif text_len < 200:
        ratio = base_ratio * 1.1
    elif text_len < 500:
        ratio = base_ratio
    else:
        ratio = base_ratio * 0.8
    
    estimated_by_text = int(text_len * 0.3 * ratio)
    estimated_by_input = int(in_tok * ratio)
    
    max_new = max(estimated_by_text, estimated_by_input)
    max_new = int(max_new * 1.2)
    
    final_max = min(cap, max(max_new, 600))
    return final_max

def validate_audio_output(audio: np.ndarray, sr: int, chunk_text: str) -> bool:
    """Fast audio validation"""
    if audio is None or len(audio) == 0:
        return False
    
    # Quick amplitude check
    max_amp = np.max(np.abs(audio))
    if max_amp < 0.001:
        logger.warning(f"Audio too quiet: max_amp={max_amp}")
        return False
    
    return True

def join_with_pause(a: np.ndarray, b: np.ndarray, sr: int,
                    gap_sec: float = 0.2, fade_sec: float = 0.1):
    """Ghép audio với pause và fade - tăng gap để tránh overlap"""
    gap_n  = max(int(sr * 0.1), int(sr * gap_sec))  # Minimum 0.1s gap
    fade_n = max(0, int(sr * fade_sec))

    if a.ndim == 2:
        ch = a.shape[1]
        silence = np.zeros((gap_n, ch), dtype=np.float32)
    else:
        silence = np.zeros(gap_n, dtype=np.float32)

    if fade_n <= 0 or len(a) < fade_n or len(b) < fade_n:
        return np.concatenate([a, silence, b], axis=0)

    fade_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    fade_in  = fade_out[::-1]
    if a.ndim == 2:
        fade_out = fade_out[:, None]
        fade_in  = fade_in[:, None]

    a_tail = a[-fade_n:] * fade_out
    b_head = b[:fade_n]  * fade_in

    return np.concatenate([a[:-fade_n], a_tail, silence, b_head, b[fade_n:]], axis=0)

def normalize_peak(x: np.ndarray, peak=0.95):
    """Normalize audio peak"""
    if x is None or x.size == 0: 
        return x
    m = float(np.max(np.abs(x)))
    return (x / m * peak).astype(np.float32) if m > 0 else x

def resample_np(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample audio array"""
    if sr_in == sr_out: 
        return x
    wav = torch.from_numpy(x).unsqueeze(0)
    out = torchaudio.functional.resample(wav, sr_in, sr_out)
    return out.squeeze(0).numpy()

# ===== Load model once =====
logger.info(f"[INIT] Loading {MODEL_ID} on {DEVICE}")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
model     = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN).to(DEVICE).eval()

# Validate consent audio
if os.path.exists(CONSENT_PATH):
    try:
        audio_data, sr = sf.read(CONSENT_PATH)
        if len(audio_data) > 0:
            logger.info(f"[CONSENT] Valid audio: {CONSENT_PATH}, duration: {len(audio_data)/sr:.2f}s")
        else:
            logger.warning(f"[CONSENT] Audio file is empty: {CONSENT_PATH}")
    except Exception as e:
        logger.error(f"[CONSENT] Error reading audio: {e}")
        CONSENT_PATH = None
else:
    logger.warning("[CONSENT] Not found; proceed without prompt voice.")
    CONSENT_PATH = None

# ===== Runpod handler =====
def handler(job):
    """
    Enhanced handler with better error handling and retry logic
    
    input:
      text: string (bắt buộc)
      gap_sec: float (default 0.2)
      fade_sec: float (default 0.1)
      prompt_transcript: string (optional)
      return: "path" | "base64" (default "path")
      outfile: optional filename
    """
    inp = job["input"] or {}
    text = (inp.get("text") or "").strip()
    
    if not text:
        return {"error": "Missing 'text'."}

    # Validate text length
    if len(text) > 10000:
        return {"error": "Text too long (max 10000 characters)"}

    # Parameters với validation
    gap_sec  = max(0.1, min(1.0, float(inp.get("gap_sec", 0.2))))
    fade_sec = max(0.05, min(0.5, float(inp.get("fade_sec", 0.1))))
    ret_mode = inp.get("return", "path")
    outfile  = inp.get("outfile")
    prompt_transcript = inp.get("prompt_transcript", "Tôi là chủ sở hữu giọng nói này.")

    # Preprocess text using imported function
    try:
        chunks = preprocess_text(text)
        logger.info(f"[HANDLER] Preprocessed into {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"[HANDLER] Preprocessing failed: {e}")
        return {"error": f"Text preprocessing failed: {str(e)}"}

    full_audio = None
    sr = TARGET_SR
    global_tokens = None
    total_in = total_out = 0
    successful_chunks = 0

    # Process each chunk with retry logic
    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            logger.warning(f"[HANDLER] Skipping empty chunk {idx+1}")
            continue
            
        logger.info(f"[HANDLER] Processing Chunk {idx+1}/{len(chunks)}: {len(chunk)} chars")
        
        retry_count = 0
        max_retries = 2
        chunk_audio = None
        
        while retry_count <= max_retries and chunk_audio is None:
            try:
                # Clear cache before processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Prepare inputs
                proc_args = {
                    "text": chunk,
                    "return_tensors": "pt"
                }
                
                if CONSENT_PATH and os.path.exists(CONSENT_PATH):
                    proc_args["prompt_speech_path"] = CONSENT_PATH
                    proc_args["prompt_text"] = prompt_transcript
                
                inputs = processor(**proc_args)
                inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
                
                in_tok = inputs["input_ids"].shape[-1]
                total_in += in_tok

                # Handle global tokens (only for first chunk)
                if idx == 0:
                    global_tokens = inputs.pop("global_token_ids_prompt", None)
                    logger.info(f"[HANDLER] Global tokens initialized")
                else:
                    _ = inputs.pop("global_token_ids_prompt", None)

                # Calculate max_new_tokens smartly
                max_new = calc_max_new(chunk, in_tok)
                logger.info(f"[HANDLER] max_new_tokens={max_new} for input={in_tok}")

                # Generate with improved parameters
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.8,
                        repetition_penalty=1.1,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )

                out_tok = output_ids.shape[-1]
                total_out += out_tok
                logger.info(f"[HANDLER] Generated {out_tok} tokens")

                # Decode audio
                audio_dict = processor.decode(
                    generated_ids=output_ids,
                    global_token_ids_prompt=global_tokens,
                    input_ids_len=in_tok,
                    return_type="np"
                )
                
                audio = np.asarray(audio_dict["audio"], dtype=np.float32)
                sr_in = int(audio_dict.get("sampling_rate", TARGET_SR))
                
                # Resample if needed
                if sr_in != TARGET_SR:
                    audio = resample_np(audio, sr_in, TARGET_SR)
                    logger.info(f"[HANDLER] Resampled from {sr_in} to {TARGET_SR}")

                # Validate generated audio
                if validate_audio_output(audio, TARGET_SR, chunk):
                    chunk_audio = audio
                    successful_chunks += 1
                    logger.info(f"[HANDLER] Chunk {idx+1} SUCCESS: {len(audio)/TARGET_SR:.2f}s audio")
                    break
                else:
                    retry_count += 1
                    if retry_count <= max_retries:
                        logger.warning(f"[HANDLER] Chunk {idx+1} failed validation, retrying ({retry_count}/{max_retries})")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    
            except Exception as e:
                retry_count += 1
                logger.error(f"[HANDLER] Chunk {idx+1} error (attempt {retry_count}): {e}")
                if retry_count <= max_retries:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                continue

        # Add valid audio to full_audio
        if chunk_audio is not None:
            if full_audio is None:
                full_audio = chunk_audio
            else:
                full_audio = join_with_pause(full_audio, chunk_audio, TARGET_SR, gap_sec, fade_sec)
                logger.info(f"[HANDLER] Combined audio length: {len(full_audio)/TARGET_SR:.2f}s")
        else:
            logger.error(f"[HANDLER] Failed to generate valid audio for chunk {idx+1} after {max_retries} retries")

        # Memory cleanup after each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Final validation and output
    if full_audio is None or successful_chunks == 0:
        return {
            "error": f"TTS failed. Success: {successful_chunks}/{len(chunks)} chunks",
            "total_chunks": len(chunks),
            "successful_chunks": successful_chunks
        }

    # Normalize audio
    full_audio = normalize_peak(full_audio)
    final_duration = len(full_audio) / TARGET_SR
    
    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())
    name = outfile or f"{job_id}.wav"
    out_path = os.path.join(OUT_DIR, name)
    
    sf.write(out_path, full_audio, TARGET_SR)
    logger.info(f"[HANDLER] ✅ Saved to {out_path}, duration: {final_duration:.2f}s")

    # Return based on mode
    if ret_mode == "base64":
        with io.BytesIO() as buf:
            sf.write(buf, full_audio, TARGET_SR, format="WAV")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "job_id": job_id,
            "audio_b64": b64,
            "sample_rate": TARGET_SR,
            "duration": round(final_duration, 2),
            "total_chunks": len(chunks),
            "successful_chunks": successful_chunks,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out
        }

    return {
        "job_id": job_id,
        "path": out_path,
        "sample_rate": TARGET_SR,
        "duration": round(final_duration, 2),
        "total_chunks": len(chunks),
        "successful_chunks": successful_chunks,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out
    }

# Start serverless worker
runpod.serverless.start({"handler": handler})
