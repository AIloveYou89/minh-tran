# rp_handler_13.py - ANTI CPU FALLBACK VERSION
import os, io, uuid, base64, numpy as np, soundfile as sf, torch, torchaudio
from transformers import AutoProcessor, AutoModel
import runpod
import gc
import logging
import re
from num2words import num2words
from typing import List, Optional
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# EMBEDDED PREPROCESS MODULE
# ============================================================
MIN_CHARS_PER_CHUNK = 80
MAX_CHARS_PER_CHUNK = 200
OPTIMAL_CHUNK_SIZE = 150
PUNCS = r".?!…"

_number_pattern = re.compile(r"(\d{1,3}(?:\.\d{3})*)(?:\s*(%|[^\W\d_]+))?", re.UNICODE)
_whitespace_pattern = re.compile(r"\s+")
_comma_pattern = re.compile(r"\s*,\s*")
_punct_spacing_pattern = re.compile(r"\s+([,;:])")
_repeated_punct_pattern = re.compile(rf"[{PUNCS}]{{2,}}")
_punct_no_space_pattern = re.compile(rf"([{PUNCS}])(?=\S)")

def normalize_text_vn(text: str) -> str:
    text = text.strip()
    text = _whitespace_pattern.sub(" ", text)
    text = _comma_pattern.sub(", ", text)
    text = text.lower()
    
    def repl_number_with_unit(m):
        num_str = m.group(1).replace(".", "")
        unit = m.group(2) or ""
        try:
            return num2words(int(num_str), lang="vi") + (" " + unit if unit else "")
        except:
            return m.group(0)
    
    text = _number_pattern.sub(repl_number_with_unit, text)
    text = _punct_spacing_pattern.sub(r"\1", text)
    text = _repeated_punct_pattern.sub(lambda m: m.group(0)[0], text)
    text = _punct_no_space_pattern.sub(r"\1 ", text)
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    parts = re.split(rf"(?<=[{PUNCS}])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p or re.fullmatch(rf"[{PUNCS}]+", p):
            continue
        out.append(p)
    return out

def ensure_punctuation(s: str) -> str:
    s = s.strip()
    if not s.endswith(tuple(PUNCS)):
        s += "."
    return s

def ensure_leading_dot(s: str) -> str:
    s = s.lstrip()
    if s and s[0] not in PUNCS:
        return ". " + s
    return s

def smart_chunk_split(text: str) -> List[str]:
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word) + 1
        
        if current_length + word_len > MAX_CHARS_PER_CHUNK and current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) < MIN_CHARS_PER_CHUNK and len(chunks) > 0:
                prev_chunk = chunks[-1]
                if len(prev_chunk) + len(chunk_text) + 1 <= MAX_CHARS_PER_CHUNK:
                    chunks[-1] = prev_chunk + " " + chunk_text
                    current_chunk = [word]
                    current_length = word_len
                    continue
            
            chunks.append(chunk_text)
            current_chunk = [word]
            current_length = word_len
        else:
            current_chunk.append(word)
            current_length += word_len
    
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) < MIN_CHARS_PER_CHUNK and chunks:
            chunks[-1] += " " + chunk_text
        else:
            chunks.append(chunk_text)
    
    return chunks

def preprocess_text(text: str) -> List[str]:
    clean = normalize_text_vn(text)
    sentences = split_into_sentences(clean)
    
    if not sentences:
        s = ensure_punctuation(clean)
        return [ensure_leading_dot(s)]
    
    chunks = []
    buffer = ""
    
    for i, sent in enumerate(sentences, 1):
        sent = ensure_punctuation(sent)
        sent = re.sub(r'^([^\w]*\w[^,]{0,10}),\s*', r'\1 ', sent)
        
        if i <= 5:
            if len(sent) > MAX_CHARS_PER_CHUNK:
                chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(sent)])
            else:
                chunks.append(ensure_leading_dot(sent))
        else:
            if len(sent) > MAX_CHARS_PER_CHUNK:
                if buffer:
                    chunks.append(ensure_leading_dot(buffer))
                    buffer = ""
                chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(sent)])
            else:
                if buffer and len(buffer) + len(sent) + 1 > OPTIMAL_CHUNK_SIZE:
                    chunks.append(ensure_leading_dot(buffer))
                    buffer = sent
                elif buffer:
                    buffer += " " + sent
                else:
                    buffer = sent
    
    if buffer:
        if len(buffer) > MAX_CHARS_PER_CHUNK:
            chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(buffer)])
        else:
            chunks.append(ensure_leading_dot(ensure_punctuation(buffer)))
    
    return chunks

# ============================================================
# CONFIG
# ============================================================
MODEL_ID  = os.getenv("MODEL_ID", "DragonLineageAI/Vi-SparkTTS-0.5B")
HF_TOKEN  = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Missing HF_TOKEN env."

assert torch.cuda.is_available(), "CUDA/GPU not available."
DEVICE    = "cuda"
TARGET_SR = int(os.getenv("TARGET_SR", "24000"))

# ============================================================
# NETWORK VOLUME
# ============================================================
def detect_network_volume_path() -> Optional[str]:
    possible_roots = [
        "/runpod-volume",
        os.getenv("RUNPOD_VOLUME_PATH"),
        os.getenv("NV_ROOT"),
    ]
    
    for root in possible_roots:
        if root and os.path.exists(root) and os.path.isdir(root):
            logger.info(f"[VOLUME] ✓ Found Network Volume at: {root}")
            return root
    
    logger.warning("[VOLUME] ⚠️ No Network Volume detected.")
    return None

NV_ROOT = detect_network_volume_path()

def find_prompt_audio() -> Optional[str]:
    if not NV_ROOT:
        logger.warning("[PROMPT] No Network Volume, cannot load prompt audio")
        return None
    
    possible_paths = [
        os.path.join(NV_ROOT, "workspace/consent_audio.wav"),
        os.path.join(NV_ROOT, "consent_audio.wav"),
        os.path.join(NV_ROOT, "audio/consent_audio.wav"),
        os.path.join(NV_ROOT, "prompts/consent_audio.wav"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                audio_data, sr = sf.read(path)
                if len(audio_data) == 0:
                    continue
                
                duration = len(audio_data) / sr
                if duration < 1.0:
                    continue
                
                max_amp = np.max(np.abs(audio_data))
                if max_amp < 0.001:
                    continue
                
                logger.info(f"[PROMPT] ✓ Found valid prompt: {path}")
                logger.info(f"[PROMPT] Duration: {duration:.2f}s, SR: {sr}, Amp: {max_amp:.4f}")
                return path
                
            except Exception as e:
                logger.error(f"[PROMPT] Error reading {path}: {e}")
                continue
    
    logger.warning(f"[PROMPT] ⚠️ No valid prompt audio found")
    return None

OUT_DIR = os.path.join(NV_ROOT, "jobs") if NV_ROOT else "/tmp/jobs"

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def calc_max_new(text: str, in_tok: int, base_ratio: float = 2.5, cap: int = 1800) -> int:
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
    if audio is None or len(audio) == 0:
        return False
    
    max_amp = np.max(np.abs(audio))
    if max_amp < 0.001:
        logger.warning(f"Audio too quiet: max_amp={max_amp}")
        return False
    
    return True

def join_with_pause(a: np.ndarray, b: np.ndarray, sr: int,
                    gap_sec: float = 0.2, fade_sec: float = 0.1):
    gap_n  = max(int(sr * 0.1), int(sr * gap_sec))
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
    if x is None or x.size == 0: 
        return x
    max_val = float(np.max(np.abs(x)))
    if max_val > 0.98:
        return (x * (peak / max_val)).astype(np.float32)
    return x

def resample_np(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out: 
        return x
    wav = torch.from_numpy(x).unsqueeze(0)
    out = torchaudio.functional.resample(wav, sr_in, sr_out)
    return out.squeeze(0).numpy()

# ============================================================
# 🔥 ANTI CPU FALLBACK UTILITIES
# ============================================================
def verify_cuda_available():
    """Kiểm tra CUDA còn available không"""
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available!")
    return True

def verify_model_on_gpu(model):
    """Kiểm tra model có còn trên GPU không"""
    try:
        device = str(next(model.parameters()).device)
        if "cuda" not in device:
            raise RuntimeError(f"❌ Model moved to CPU! Device: {device}")
        return True
    except Exception as e:
        raise RuntimeError(f"❌ Cannot verify model device: {e}")

def force_model_to_gpu(model, device="cuda"):
    """Force model về GPU nếu bị move sang CPU"""
    try:
        model = model.to(device)
        # Verify
        verify_model_on_gpu(model)
        logger.info(f"✅ Model forced back to {device}")
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Cannot move model to GPU: {e}")

def ensure_tensor_on_device(tensor, device="cuda"):
    """Đảm bảo tensor ở đúng device"""
    if tensor is None:
        return None
    if hasattr(tensor, "to"):
        return tensor.to(device)
    return tensor

# ============================================================
# MODEL LOADING - ANTI CPU FALLBACK VERSION
# ============================================================
logger.info(f"[INIT] Loading {MODEL_ID} on {DEVICE}")
init_start = time.time()

# ✅ Set CUDA default device trước
torch.cuda.set_device(0)
logger.info(f"[INIT] Default CUDA device set to: {torch.cuda.current_device()}")

# ✅ Load processor
processor = AutoProcessor.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    token=HF_TOKEN
)

# ✅ Load model - KHÔNG DÙNG device_map, torch_dtype
model = AutoModel.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    token=HF_TOKEN
)

# ✅ FORCE model lên GPU và freeze
model = model.to(DEVICE)
model.eval()
model.requires_grad_(False)  # Freeze để tránh autograd tự move sang CPU

# ✅ Verify model trên GPU
verify_model_on_gpu(model)
logger.info(f"[INIT] ✅ Model confirmed on device: {next(model.parameters()).device}")

processor.model = model
logger.info("[INIT] ✓ Processor linked to model")

# ✅ Pin model parameters để tránh bị swap
for param in model.parameters():
    param.data = param.data.pin_memory()
logger.info("[INIT] ✅ Model parameters pinned to memory")

PROMPT_PATH = find_prompt_audio()
if PROMPT_PATH:
    logger.info(f"[INIT] ✓ Prompt audio ready: {PROMPT_PATH}")
else:
    logger.warning("[INIT] ⚠️ Running without prompt audio")

init_time = time.time() - init_start
logger.info(f"[INIT] Model loaded in {init_time:.2f}s")

# ✅ WARM-UP
logger.info("[INIT] Warming up model...")
try:
    warmup_text = ". xin chào"
    warmup_inputs = processor(text=warmup_text, return_tensors="pt")
    warmup_inputs = {k: ensure_tensor_on_device(v, DEVICE) for k, v in warmup_inputs.items()}
    _ = warmup_inputs.pop("global_token_ids_prompt", None)
    
    with torch.no_grad():
        _ = model.generate(**warmup_inputs, max_new_tokens=50, do_sample=False)
    
    verify_model_on_gpu(model)
    torch.cuda.empty_cache()
    logger.info("[INIT] ✅ Warmup complete, model still on GPU")
except Exception as e:
    logger.warning(f"[INIT] Warmup failed: {e}")

# ============================================================
# HANDLER - ANTI CPU FALLBACK VERSION
# ============================================================
def handler(job):
    """
    Handler with aggressive CPU fallback prevention
    """
    job_start = time.time()
    
    # ✅ Pre-check GPU
    try:
        verify_cuda_available()
        verify_model_on_gpu(model)
    except Exception as e:
        logger.error(f"[HANDLER] GPU check failed at start: {e}")
        return {"error": str(e)}
    
    inp = job["input"] or {}
    text = (inp.get("text") or "").strip()
    
    if not text:
        return {"error": "Missing 'text'."}

    if len(text) > 500000:
        return {"error": "Text too long (max 500,000 characters)"}

    # Parameters
    gap_sec  = max(0.1, min(1.0, float(inp.get("gap_sec", 0.2))))
    fade_sec = max(0.05, min(0.5, float(inp.get("fade_sec", 0.1))))
    ret_mode = inp.get("return", "path")
    outfile  = inp.get("outfile")
    
    # Prompt configuration
    custom_prompt_path = inp.get("prompt_path")
    prompt_transcript = inp.get("prompt_transcript", "Tôi là chủ sở hữu giọng nói này, và tôi đồng ý cho Google sử dụng giọng nói này để tạo mô hình giọng nói tổng hợp.")
    
    active_prompt_path = None
    if custom_prompt_path and os.path.exists(custom_prompt_path):
        active_prompt_path = custom_prompt_path
    elif PROMPT_PATH:
        active_prompt_path = PROMPT_PATH

    # Preprocess text
    preprocess_start = time.time()
    try:
        chunks = preprocess_text(text)
        preprocess_time = time.time() - preprocess_start
        logger.info(f"[HANDLER] Preprocessed into {len(chunks)} chunks ({preprocess_time:.2f}s)")
    except Exception as e:
        logger.error(f"[HANDLER] Preprocessing failed: {e}")
        return {"error": f"Text preprocessing failed: {str(e)}"}

    full_audio = None
    sr = TARGET_SR
    global_tokens = None
    total_in = total_out = 0
    successful_chunks = 0
    
    generation_start = time.time()
    
    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            logger.warning(f"[HANDLER] Skipping empty chunk {idx+1}")
            continue
        
        # ✅ CHECK GPU TRƯỚC MỖI CHUNK
        try:
            verify_cuda_available()
            verify_model_on_gpu(model)
        except Exception as e:
            logger.error(f"[HANDLER] GPU check failed at chunk {idx+1}: {e}")
            # Try to recover
            try:
                global model
                model = force_model_to_gpu(model, DEVICE)
                logger.warning(f"[HANDLER] ⚠️ Model recovered to GPU for chunk {idx+1}")
            except:
                return {
                    "error": f"GPU lost at chunk {idx+1}/{len(chunks)}",
                    "successful_chunks": successful_chunks,
                    "total_chunks": len(chunks)
                }
        
        chunk_start = time.time()
        logger.info(f"[HANDLER] Processing Chunk {idx+1}/{len(chunks)}: {len(chunk)} chars")
        
        chunk_audio = None
        
        try:
            # ✅ AGGRESSIVE CACHE CLEAR
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Đợi tất cả operations hoàn thành
            
            # Prepare inputs
            proc_args = {
                "text": chunk,
                "return_tensors": "pt"
            }
            
            if active_prompt_path:
                proc_args["prompt_speech_path"] = active_prompt_path
                proc_args["prompt_text"] = prompt_transcript
            
            inputs = processor(**proc_args)
            
            # ✅ FORCE TẤT CẢ INPUTS LÊN GPU
            inputs = {k: ensure_tensor_on_device(v, DEVICE) for k, v in inputs.items()}
            
            in_tok = inputs["input_ids"].shape[-1]
            total_in += in_tok

            # Global tokens handling
            if idx == 0:
                global_tokens = inputs.pop("global_token_ids_prompt", None)
                if global_tokens is not None:
                    global_tokens = ensure_tensor_on_device(global_tokens, DEVICE)
                    logger.info(f"[HANDLER] Global tokens: {global_tokens.shape} on {global_tokens.device}")
            else:
                _ = inputs.pop("global_token_ids_prompt", None)

            max_new = calc_max_new(chunk, in_tok)

            # ✅ VERIFY TRƯỚC KHI GENERATE
            verify_model_on_gpu(model)
            
            # Generate với context manager để đảm bảo GPU
            with torch.cuda.device(0):
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
            
            logger.info(f"[HANDLER] Generated {out_tok} tokens (input: {in_tok})")

            # ✅ ENSURE output_ids trên GPU trước decode
            output_ids = ensure_tensor_on_device(output_ids, DEVICE)
            
            # Decode
            audio_dict = processor.decode(
                generated_ids=output_ids,
                global_token_ids_prompt=global_tokens,
                input_ids_len=in_tok
            )
            
            audio = np.asarray(audio_dict["audio"], dtype=np.float32)
            sr_in = int(audio_dict.get("sampling_rate", TARGET_SR))
            
            if sr_in != TARGET_SR:
                audio = resample_np(audio, sr_in, TARGET_SR)

            if validate_audio_output(audio, TARGET_SR, chunk):
                chunk_audio = audio
                successful_chunks += 1
                chunk_time = time.time() - chunk_start
                logger.info(f"[HANDLER] Chunk {idx+1} SUCCESS: {len(audio)/TARGET_SR:.2f}s audio ({chunk_time:.2f}s)")
                
        except Exception as e:
            logger.error(f"[HANDLER] Chunk {idx+1} error: {e}")
            # Try to recover model
            try:
                global model
                model = force_model_to_gpu(model, DEVICE)
                torch.cuda.empty_cache()
            except:
                pass

        # Combine audio
        if chunk_audio is not None:
            if full_audio is None:
                full_audio = chunk_audio
            else:
                full_audio = join_with_pause(full_audio, chunk_audio, TARGET_SR, gap_sec, fade_sec)
                logger.info(f"[HANDLER] Combined audio length: {len(full_audio)/TARGET_SR:.2f}s")
        else:
            logger.error(f"[HANDLER] Failed chunk {idx+1}")

        # Memory cleanup mỗi 5 chunks
        if (idx + 1) % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

    generation_time = time.time() - generation_start

    if full_audio is None or successful_chunks == 0:
        return {
            "error": f"TTS failed. Success: {successful_chunks}/{len(chunks)}",
            "total_chunks": len(chunks),
            "successful_chunks": successful_chunks
        }

    # Normalize and save
    full_audio = normalize_peak(full_audio, peak=0.95)
    final_duration = len(full_audio) / TARGET_SR
    
    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())
    name = outfile or f"{job_id}.wav"
    out_path = os.path.join(OUT_DIR, name)
    
    save_start = time.time()
    sf.write(out_path, full_audio, TARGET_SR)
    save_time = time.time() - save_start
    
    total_time = time.time() - job_start
    
    logger.info(f"[HANDLER] ✅ Saved: {out_path}, duration: {final_duration:.2f}s")
    logger.info(f"[TIMING] Total: {total_time:.2f}s | Preprocess: {preprocess_time:.2f}s | Generation: {generation_time:.2f}s | Save: {save_time:.2f}s")

    result = {
        "job_id": job_id,
        "sample_rate": TARGET_SR,
        "duration": round(final_duration, 2),
        "total_chunks": len(chunks),
        "successful_chunks": successful_chunks,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "used_prompt_voice": active_prompt_path is not None,
        "prompt_path": active_prompt_path,
        "network_volume_detected": NV_ROOT is not None,
        "processing_time": round(total_time, 2),
        "generation_time": round(generation_time, 2),
        "preprocess_time": round(preprocess_time, 2)
    }

    if ret_mode == "base64":
        with io.BytesIO() as buf:
            sf.write(buf, full_audio, TARGET_SR, format="WAV")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result["audio_b64"] = b64
    else:
        result["path"] = out_path

    return result

# Start serverless worker
runpod.serverless.start({"handler": handler})
