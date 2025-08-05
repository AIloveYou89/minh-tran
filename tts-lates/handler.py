"""
handler.py -- Runpod Serverless (Fixed Version)
──────────────────────────────
• Network-volume mount tại /workspace/minh-tran/tts-lates
• Nếu thiếu consent_audio.wav ➜ tải từ S3 (dùng ENV)
• Added comprehensive error handling
"""

import uuid, os, sys, traceback
from pathlib import Path
from typing import Dict
import numpy as np
import soundfile as sf
import torch
import runpod
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────── ĐƯỜNG DẪN CỐ ĐỊNH ───────────
BASE_DIR = Path("/workspace/minh-tran/tts-lates")
FIXED_DIR = BASE_DIR / "fixed"
PROMPT_WAV = FIXED_DIR / "consent_audio.wav"
JOBS_DIR = BASE_DIR / "jobs"
MODEL_ID = "DragonLineageAI/Vi-SparkTTS-0.5B"

# Safer device detection
try:
    DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    if DEVICE == "cuda":
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
except Exception as e:
    logger.warning(f"Error checking CUDA: {e}")
    DEVICE = "cpu"

logger.info(f"Using device: {DEVICE}")

PROMPT_TRANSCRIPT = (
    "Tôi là chủ sở hữu giọng nói này và tôi đồng ý cho Google sử dụng "
    "giọng nói này để tạo mô hình giọng nói tổng hợp."
)

# ────────── TẢI PROMPT TỪ S3 (nếu cần) ──────────
def _download_prompt_once() -> None:
    """Download prompt audio from S3 if not exists locally"""
    try:
        if PROMPT_WAV.exists():
            logger.info(f"Prompt file already exists: {PROMPT_WAV}")
            return

        # Check all required S3 env vars
        s3_ep = os.getenv("S3_ENDPOINT_URL")
        bucket = os.getenv("S3_BUCKET_NAME") 
        key = os.getenv("S3_PROMPT_KEY")  # vd: minh-tran/tts-lates/fixed/consent_audio.wav
        akid = os.getenv("AWS_ACCESS_KEY_ID")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        missing_vars = []
        if not s3_ep: missing_vars.append("S3_ENDPOINT_URL")
        if not bucket: missing_vars.append("S3_BUCKET_NAME")
        if not key: missing_vars.append("S3_PROMPT_KEY")
        if not akid: missing_vars.append("AWS_ACCESS_KEY_ID")
        if not secret: missing_vars.append("AWS_SECRET_ACCESS_KEY")
        
        if missing_vars:
            raise EnvironmentError(
                f"Missing S3 environment variables: {', '.join(missing_vars)}. "
                f"Prompt file {PROMPT_WAV} does not exist."
            )

        logger.info(f"Downloading s3://{bucket}/{key} ...")
        
        import boto3
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError
        
        s3 = boto3.client(
            "s3",
            endpoint_url=s3_ep,
            aws_access_key_id=akid,
            aws_secret_access_key=secret,
            config=Config(signature_version="s3v4", retries={'max_attempts': 3}),
        )
        
        # Ensure directory exists
        PROMPT_WAV.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with retry logic
        s3.download_file(bucket, key, str(PROMPT_WAV))
        
        if not PROMPT_WAV.exists():
            raise FileNotFoundError(f"Download completed but file not found: {PROMPT_WAV}")
            
        logger.info(f"Download completed ✓ Size: {PROMPT_WAV.stat().st_size} bytes")
        
    except Exception as e:
        logger.error(f"Failed to download prompt file: {e}")
        raise

# ────────── LOAD MODEL MỘT LẦN ──────────
def initialize_model():
    """Initialize the TTS model with error handling"""
    try:
        logger.info("Loading SparkTTS model...")
        
        from transformers import AutoProcessor, AutoModel
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            cache_dir="/tmp/hf_cache"  # Use tmp for caching
        )
        
        # Load model with memory management
        model = AutoModel.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            cache_dir="/tmp/hf_cache",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(DEVICE).eval()
        
        processor.model = model
        
        # Clear cache after loading
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            
        logger.info("Model loaded successfully ✓")
        return processor, model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise

# ────────── HÀM ÂM THANH ──────────
def _cross_fade(a, b, sr, sec=0.12):
    """Cross-fade two audio arrays"""
    try:
        n = int(sr * sec)
        if n == 0 or len(a) < n or len(b) < n:
            return np.concatenate([a, b])
        
        w = np.linspace(1.0, 0.0, n, dtype=np.float32)
        return np.concatenate([a[:-n], a[-n:]*w + b[:n]*w[::-1], b[n:]])
    except Exception as e:
        logger.error(f"Cross-fade error: {e}")
        return np.concatenate([a, b])  # Fallback to simple concat

# ────────── CLONE TTS ──────────
def clone_tts(text: str, processor, model) -> str:
    """Generate TTS audio with comprehensive error handling"""
    try:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
            
        from preprocess import preprocess_text
        
        parts = preprocess_text(text.strip())
        if not parts:
            raise ValueError("Text preprocessing returned empty result")
            
        logger.info(f"Processing {len(parts)} text segments")
        
        sr, merged = None, None
        
        for i, seg in enumerate(parts):
            try:
                logger.debug(f"Processing segment {i+1}/{len(parts)}: {seg[:50]}...")
                
                # Encode input
                enc = processor(
                    text=seg,
                    prompt_speech_path=str(PROMPT_WAV),
                    prompt_text=PROMPT_TRANSCRIPT,
                    return_tensors="pt",
                ).to(DEVICE)
                
                # Generate with memory management
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        max_new_tokens=1000,
                        do_sample=True,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.95,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
                
                # Decode audio
                dec = processor.decode(
                    generated_ids=gen,
                    global_token_ids_prompt=None,
                    input_ids_len=enc["input_ids"].shape[-1],
                )
                
                audio = np.asarray(dec["audio"], dtype=np.float32)
                current_sr = dec["sampling_rate"]
                
                if sr is None:
                    sr = current_sr
                elif sr != current_sr:
                    logger.warning(f"Sample rate mismatch: {sr} vs {current_sr}")
                
                # Merge audio segments
                if merged is None:
                    merged = audio
                else:
                    # Add silence between segments
                    silence = np.zeros(int(sr * 0.25), np.float32)
                    merged = _cross_fade(
                        np.concatenate([merged, silence]),
                        audio,
                        sr,
                    )
                
                # Clear GPU memory after each segment
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing segment {i+1}: {e}")
                raise
        
        if merged is None:
            raise RuntimeError("No audio generated")
        
        # Apply fade in/out
        try:
            edge = np.zeros(int(sr * 0.12), np.float32)
            merged = _cross_fade(edge, merged, sr, 0.04)
            merged = _cross_fade(merged, edge, sr, 0.04)
        except Exception as e:
            logger.warning(f"Edge fade failed: {e}")
        
        # Save output
        JOBS_DIR.mkdir(exist_ok=True)
        out_path = JOBS_DIR / f"{uuid.uuid4()}.wav"
        
        # Normalize audio to prevent clipping
        max_val = np.abs(merged).max()
        if max_val > 0.95:
            merged = merged * (0.95 / max_val)
            logger.info("Audio normalized to prevent clipping")
        
        sf.write(out_path, merged, sr)
        
        file_size = out_path.stat().st_size
        duration = len(merged) / sr
        logger.info(f"Audio generated: {duration:.2f}s, {file_size/1024:.1f}KB")
        
        return str(out_path)
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        logger.error(traceback.format_exc())
        raise

# ────────── INITIALIZATION ──────────
try:
    # Download prompt if needed
    _download_prompt_once()
    
    # Initialize model
    processor, model = initialize_model()
    
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    sys.exit(1)

# ────────── RUNPOD HANDLER ──────────
def handler(event: Dict):
    """Main RunPod handler with comprehensive error handling"""
    try:
        # Validate input
        if not event or "input" not in event:
            return {"error": "Missing 'input' in request"}
        
        input_data = event["input"]
        if not isinstance(input_data, dict) or "text" not in input_data:
            return {"error": "Missing 'text' in input data"}
        
        text = input_data["text"]
        if not isinstance(text, str):
            return {"error": "Text must be a string"}
        
        logger.info(f"Processing request: {len(text)} characters")
        
        # Generate TTS
        wav_path = clone_tts(text, processor, model)
        
        return {
            "wav_path": wav_path,
            "status": "success",
            "text_length": len(text)
        }
        
    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"Handler error: {error_msg}")
        logger.error(traceback.format_exc())
        
        return {
            "error": error_msg,
            "status": "failed",
            "traceback": traceback.format_exc() if os.getenv("DEBUG") else None
        }

# Start RunPod serverless
if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
