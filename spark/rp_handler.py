# rp_handler.py — RunPod Serverless Queue-based (SparkTTS)
# - GPU only (assert CUDA)
# - prompt_speech_path + prompt_text giống rendervideo.py
# - Không truyền global_token_ids_prompt vào generate() (chỉ decode)
# - Tự tìm prompt ở /runpod-volume/... hoặc workspace nếu có
# - Gộp preprocess vào cùng file (fallback nếu thiếu num2words)

import os, io, uuid, base64, gc, logging, re
import numpy as np
import soundfile as sf
import torch, torchaudio
import runpod
from typing import List, Optional
from transformers import AutoProcessor, AutoModel

# --------- optional dependency (fallback an toàn) ----------
try:
    from num2words import num2words
except Exception:
    def num2words(n, lang="vi"):  # fallback nếu thiếu lib
        return str(n)

# ----------------------- logging ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rp_handler")

# ================== EMBEDDED PREPROCESS ====================
MIN_CHARS_PER_CHUNK = 50
MAX_CHARS_PER_CHUNK = 130
OPTIMAL_CHUNK_SIZE = 80
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
        except Exception:
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
    cur, cur_len = [], 0
    for w in words:
        wl = len(w) + 1
        if cur_len + wl > MAX_CHARS_PER_CHUNK and cur:
            ck = " ".join(cur)
            if len(ck) < MIN_CHARS_PER_CHUNK and chunks:
                prev = chunks[-1]
                if len(prev) + len(ck) + 1 <= MAX_CHARS_PER_CHUNK:
                    chunks[-1] = prev + " " + ck
                    cur, cur_len = [w], wl
                    continue
            chunks.append(ck)
            cur, cur_len = [w], wl
        else:
            cur.append(w); cur_len += wl
    if cur:
        ck = " ".join(cur)
        if len(ck) < MIN_CHARS_PER_CHUNK and chunks:
            chunks[-1] += " " + ck
        else:
            chunks.append(ck)
    return chunks

def preprocess_text(text: str) -> List[str]:
    clean = normalize_text_vn(text)
    sents = split_into_sentences(clean)
    if not sents:
        s = ensure_punctuation(clean)
        return [ensure_leading_dot(s)]

    chunks, buf = [], ""
    for i, sent in enumerate(sents, 1):
        sent = ensure_punctuation(sent)
        sent = re.sub(r'^([^\w]*\w[^,]{0,10}),\s*', r'\1 ', sent)
        if i <= 5:
            if len(sent) > MAX_CHARS_PER_CHUNK:
                chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(sent)])
            else:
                chunks.append(ensure_leading_dot(sent))
        else:
            if len(sent) > MAX_CHARS_PER_CHUNK:
                if buf:
                    chunks.append(ensure_leading_dot(buf)); buf = ""
                chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(sent)])
            else:
                if buf and len(buf) + len(sent) + 1 > OPTIMAL_CHUNK_SIZE:
                    chunks.append(ensure_leading_dot(buf)); buf = sent
                elif buf:
                    buf += " " + sent
                else:
                    buf = sent
    if buf:
        if len(buf) > MAX_CHARS_PER_CHUNK:
            chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(buf)])
        else:
            chunks.append(ensure_leading_dot(ensure_punctuation(buf)))
    return chunks
# ================== END PREPROCESS ==========================

# --------------------- Config (GPU only) --------------------
MODEL_ID = os.getenv("MODEL_ID", "DragonLineageAI/Vi-SparkTTS-0.5B")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
assert HF_TOKEN, "Missing HF_TOKEN env."
assert torch.cuda.is_available(), "CUDA/GPU not available."
DEVICE = "cuda"
TARGET_SR = int(os.getenv("TARGET_SR", "24000"))

# ----------------- Storage / Prompt paths -------------------
def detect_nv_root() -> Optional[str]:
    candidates = [
        os.getenv("NV_ROOT"),
        "/runpod-volume",  # Network Volume mount (Runpod)
        "/workspace"      # fallback nếu bạn bake file sẵn trong image
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            logger.info(f"[VOLUME] Using NV root: {p}")
            return p
    logger.warning("[VOLUME] Not found, will use /tmp for outputs")
    return None

NV_ROOT = detect_nv_root()
OUT_DIR = os.path.join(NV_ROOT, "jobs") if NV_ROOT else "/tmp/jobs"

def resolve_prompt_path() -> Optional[str]:
    # Ưu tiên biến môi trường cho chuẩn hoá
    env_prompt = os.getenv("PROMPT_LOCAL")
    if env_prompt and os.path.exists(env_prompt):
        return env_prompt
    # Các đường dẫn phổ biến (tuỳ bạn đang lưu ở đâu)
    candidates = []
    if NV_ROOT:
        candidates += [
            os.path.join(NV_ROOT, "fixed/consent_audio.wav"),
            os.path.join(NV_ROOT, "consent_audio.wav"),
            os.path.join(NV_ROOT, "workspace/consent_audio.wav"),
        ]
    # baked-in (nếu bạn copy sẵn vào image như bản cũ)
    candidates += [
        "/workspace/minh-tran/tts-lates/fixed/consent_audio.wav"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

PROMPT_PATH = resolve_prompt_path()
if PROMPT_PATH:
    try:
        a, sr = sf.read(PROMPT_PATH)
        logger.info(f"[PROMPT] Found: {PROMPT_PATH} ({len(a)/sr:.2f}s @ {sr}Hz)")
    except Exception as e:
        logger.warning(f"[PROMPT] Read error {PROMPT_PATH}: {e}")
        PROMPT_PATH = None
else:
    logger.warning("[PROMPT] Not found → run without prompt voice")

# ------------------- Audio helpers -------------------------
def resample_np(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out: return x
    wav = torch.from_numpy(x).unsqueeze(0)
    out = torchaudio.functional.resample(wav, sr_in, sr_out)
    return out.squeeze(0).numpy()

def join_with_pause(a: np.ndarray, b: np.ndarray, sr: int,
                    gap_sec: float = 0.2, fade_sec: float = 0.1):
    gap_n  = max(int(sr * 0.1), int(sr * gap_sec))
    fade_n = max(0, int(sr * fade_sec))
    if a.ndim == 2:
        silence = np.zeros((gap_n, a.shape[1]), dtype=np.float32)
    else:
        silence = np.zeros(gap_n, dtype=np.float32)
    if fade_n <= 0 or len(a) < fade_n or len(b) < fade_n:
        return np.concatenate([a, silence, b], axis=0)
    fo = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    fi = fo[::-1]
    if a.ndim == 2:
        fo, fi = fo[:, None], fi[:, None]
    return np.concatenate([a[:-fade_n], a[-fade_n:] * fo, silence, b[:fade_n] * fi, b[fade_n:]], axis=0)

def normalize_peak(x: np.ndarray, peak=0.95):
    if x is None or x.size == 0: return x
    m = float(np.max(np.abs(x)))
    return (x / m * peak).astype(np.float32) if m > 0 else x

def calc_max_new(text: str, in_tok: int, base_ratio: float = 2.5, cap: int = 1800) -> int:
    L = len(text.strip())
    if L < 50: ratio = base_ratio * 1.5
    elif L < 200: ratio = base_ratio * 1.1
    elif L < 500: ratio = base_ratio
    else: ratio = base_ratio * 0.8
    est_text  = int(L * 0.3 * ratio)
    est_input = int(in_tok * ratio)
    max_new = int(max(est_text, est_input) * 1.2)
    return min(cap, max(max_new, 600))

# ------------------ Load model once (GPU) -------------------
logger.info(f"[INIT] Loading {MODEL_ID} on {DEVICE}")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
model     = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN).to(DEVICE).eval()

# Một số repo SparkTTS hỗ trợ link_model, nếu không có thì bỏ qua
try:
    processor.link_model(model)  # optional
    logger.info("[INIT] Processor linked to model")
except Exception:
    pass

# =================== RunPod handler =========================
def handler(job):
    """
    Queue-based handler
    input:
      text: str (bắt buộc)
      prompt_path: optional override path cho prompt_speech_path
      prompt_transcript: optional text đồng ý (mặc định tiếng Việt)
      gap_sec, fade_sec, return("path"|"base64"), outfile
    """
    inp = job.get("input") or {}
    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "Missing 'text'."}

    gap_sec  = max(0.05, min(1.0, float(inp.get("gap_sec", 0.2))))
    fade_sec = max(0.05, min(0.5,  float(inp.get("fade_sec", 0.1))))
    ret_mode = inp.get("return", "path")
    outfile  = inp.get("outfile")
    prompt_transcript = inp.get("prompt_transcript", "Tôi là chủ sở hữu giọng nói này, và tôi đồng ý cho hệ thống sử dụng giọng nói này để tạo mô hình giọng nói tổng hợp.")

    # Chọn prompt thực tế
    active_prompt = None
    custom_prompt = inp.get("prompt_path")
    if custom_prompt and os.path.exists(custom_prompt):
        active_prompt = custom_prompt
        logger.info(f"[PROMPT] Using custom prompt: {custom_prompt}")
    elif PROMPT_PATH:
        active_prompt = PROMPT_PATH
        logger.info(f"[PROMPT] Using default prompt: {PROMPT_PATH}")
    else:
        logger.info("[PROMPT] No prompt_speech_path → default voice")

    # Tiền xử lý
    try:
        chunks = preprocess_text(text)
    except Exception as e:
        return {"error": f"Preprocess failed: {e}"}
    logger.info(f"[HANDLER] {len(chunks)} chunks")

    os.makedirs(OUT_DIR, exist_ok=True)
    full, sr = None, TARGET_SR
    total_in = total_out = 0

    for i, ck in enumerate(chunks, 1):
        ck = ck.strip()
        if not ck:
            continue

        # Chuẩn bị input cho processor
        proc_args = {"text": ck, "return_tensors": "pt"}
        if active_prompt:
            proc_args["prompt_speech_path"] = active_prompt
            proc_args["prompt_text"] = prompt_transcript

        # Tạo inputs và đưa lên GPU
        inputs = processor(**proc_args)
        inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # LẤY global tokens rồi POP RA KHỎI inputs (QUAN TRỌNG)
        global_tokens = inputs.pop("global_token_ids_prompt", None)

        in_tok = int(inputs["input_ids"].shape[-1])
        total_in += in_tok
        max_new = calc_max_new(ck, in_tok)

        gen_kwargs = dict(
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

        # KHÔNG truyền global_token_ids_prompt vào generate()

        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)

        out_tok = int(output_ids.shape[-1])
        total_out += out_tok

        # Decode — TRUYỀN global_token_ids_prompt Ở ĐÂY (đúng chỗ)
        decode_kwargs = {
            "generated_ids": output_ids,
            "input_ids_len": in_tok,
            "return_type": "np"
        }
        if global_tokens is not None:
            decode_kwargs["global_token_ids_prompt"] = global_tokens

        audio_dict = processor.decode(**decode_kwargs)
        audio = np.asarray(audio_dict["audio"], dtype=np.float32)
        sr_in = int(audio_dict.get("sampling_rate", TARGET_SR))
        if sr_in != TARGET_SR:
            audio = resample_np(audio, sr_in, TARGET_SR)

        # Ghép
        if full is None:
            full = audio
        else:
            full = join_with_pause(full, audio, TARGET_SR, gap_sec, fade_sec)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if full is None or full.size == 0:
        return {"error": "TTS failed or empty output"}

    full = normalize_peak(full)
    job_id = str(uuid.uuid4())
    name = outfile or f"{job_id}.wav"
    out_path = os.path.join(OUT_DIR, name)
    sf.write(out_path, full, TARGET_SR)
    dur = round(len(full) / TARGET_SR, 2)

    result = {
        "job_id": job_id,
        "sample_rate": TARGET_SR,
        "duration": dur,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "used_prompt_voice": bool(active_prompt),
        "prompt_path": active_prompt,
    }

    if ret_mode == "base64":
        with io.BytesIO() as buf:
            sf.write(buf, full, TARGET_SR, format="WAV")
            result["audio_b64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        result["path"] = out_path

    return result

# Queue-based start
runpod.serverless.start({"handler": handler})
