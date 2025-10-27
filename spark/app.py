import os, io, uuid, base64, re
from typing import List

import numpy as np
import soundfile as sf
import torch, torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoProcessor, AutoModel
from num2words import num2words

# =========================
# (GỘP) NỘI DUNG preprocess.py
# =========================

# Tối ưu chunk size cho tốc độ xử lý
MIN_CHARS_PER_CHUNK = 50  # Giảm từ 150 để xử lý nhanh hơn
MAX_CHARS_PER_CHUNK = 130  # Giảm từ 250 để tránh token limit
OPTIMAL_CHUNK_SIZE = 80   # Sweet spot cho model
PUNCS = r".?!…"

# Cache compiled regex patterns để tăng tốc
_number_pattern = re.compile(r"(\d{1,3}(?:\.\d{3})*)(?:\s*(%|[^\W\d_]+))?", re.UNICODE)
_whitespace_pattern = re.compile(r"\s+")
_comma_pattern = re.compile(r"\s*,\s*")
_punct_spacing_pattern = re.compile(r"\s+([,;:])")
_repeated_punct_pattern = re.compile(rf"[{PUNCS}]{{2,}}")
_punct_no_space_pattern = re.compile(rf"([{PUNCS}])(?=\S)")

def normalize_text_vn(text: str) -> str:
    """Tối ưu normalize với cached regex patterns"""
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
    """Chia câu với regex tối ưu"""
    parts = re.split(rf"(?<=[{PUNCS}])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if re.fullmatch(rf"[{PUNCS}]+", p):
            continue
        out.append(p)
    return out

def ensure_punctuation(s: str) -> str:
    """Đảm bảo câu có dấu câu"""
    s = s.strip()
    if not s.endswith(tuple(PUNCS)):
        s += "."
    return s

def ensure_leading_dot(s: str) -> str:
    """Đảm bảo câu bắt đầu bằng dấu chấm nếu cần"""
    s = s.lstrip()
    if s and s[0] not in PUNCS:
        return ". " + s
    return s

def smart_chunk_split(text: str) -> List[str]:
    """Chia chunk thông minh dựa trên độ dài optimal"""
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word) + 1  # +1 for space
        
        if current_length + word_len > MAX_CHARS_PER_CHUNK and current_chunk:
            # Kiểm tra nếu chunk quá ngắn, thêm thêm từ
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) < MIN_CHARS_PER_CHUNK and len(chunks) > 0:
                # Merge với chunk trước nếu có thể
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

def split_long_text(text: str, max_len=MAX_CHARS_PER_CHUNK) -> List[str]:
    """Chia văn bản dài thành các phần nhỏ hơn"""
    parts = []
    t = text.strip()
    while len(t) > max_len:
        split_pos = t.rfind(" ", max_len - 50, max_len)
        if split_pos == -1:
            split_pos = max_len
        parts.append(t[:split_pos].strip())
        t = t[split_pos:].strip()
    if t:
        parts.append(t)
    return parts

def preprocess_text(text: str) -> List[str]:
    """Preprocessing tối ưu với chunk size thông minh - MAIN FUNCTION"""
    clean = normalize_text_vn(text)
    sentences = split_into_sentences(clean)
    
    if not sentences:
        s = ensure_punctuation(clean)
        return [ensure_leading_dot(s)]
    
    # Ưu tiên ghép câu để đạt optimal chunk size
    chunks = []
    buffer = ""
    
    for i, sent in enumerate(sentences, 1):
        sent = ensure_punctuation(sent)
        sent = re.sub(r'^([^\w]*\w[^,]{0,10}),\s*', r'\1 ', sent)
        
        # Xử lý 5 câu đầu riêng biệt như code gốc
        if i <= 5:
            if len(sent) > MAX_CHARS_PER_CHUNK:
                chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(sent)])
            else:
                chunks.append(ensure_leading_dot(sent))
        else:
            # Nếu câu quá dài, chia nhỏ
            if len(sent) > MAX_CHARS_PER_CHUNK:
                if buffer:
                    chunks.append(ensure_leading_dot(buffer))
                    buffer = ""
                chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(sent)])
            else:
                # Thêm vào buffer nếu không vượt quá optimal size
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

def get_chunk_stats(chunks: List[str]) -> dict:
    """Thống kê chunks để debug"""
    lengths = [len(c) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "optimal_ratio": sum(1 for l in lengths if MIN_CHARS_PER_CHUNK <= l <= OPTIMAL_CHUNK_SIZE) / len(lengths) if lengths else 0
    }

# Test nhanh (giữ nguyên nội dung)
if __name__ == "__main__" and False:
    sample = "Audio 360, nếu thấy hay thì like nhé! 10.000 tệ là phần thưởng. Đây là một đoạn văn bản dài hơn để test chunking. Chúng ta sẽ xem liệu nó có được chia đúng cách không."
    print("=== TEST PREPROCESS ===")
    chunks = preprocess_text(sample)
    stats = get_chunk_stats(chunks)
    print(f"Input length: {len(sample)} chars")
    print(f"Chunks: {stats['total_chunks']}")
    print(f"Length range: {stats['min_length']}-{stats['max_length']} (avg: {stats['avg_length']:.1f})")
    print(f"Optimal ratio: {stats['optimal_ratio']:.1%}")
    print()
    for idx, c in enumerate(chunks, 1):
        print(f"{idx}. ({len(c)} chars) {repr(c)}")

# =========================
# (GỐC) NỘI DUNG app.py
# =========================

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
