# jobs_app_on_demand.py

import os, uuid, threading, numpy as np, torch, soundfile as sf
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoProcessor, AutoModel
from preprocess import preprocess_text
from typing import Dict, Any

app = FastAPI()

device = "cuda"   # hoặc "cpu"
MODEL_ID = "DragonLineageAI/Vi-SparkTTS-0.5B"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()
processor.model = model

jobs: Dict[str, Dict[str, Any]] = {}
DEFAULT_PROMPT_PATH = "/workspace/tts-demo/consent_audio.wav"
DEFAULT_PROMPT_TRANSCRIPT = (
    "Tôi là chủ sở hữu giọng nói này và tôi đồng ý cho Google sử dụng giọng nói này để tạo mô hình giọng nói tổng hợp."
)

def cross_fade(a: np.ndarray, b: np.ndarray, sr: int, fade_duration: float = 0.15):
    fade_samples = int(sr * fade_duration)
    if fade_samples <= 0 or len(a) < fade_samples or len(b) < fade_samples:
        return np.concatenate([a, b], axis=0)
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in  = fade_out[::-1]
    return np.concatenate(
        [a[:-fade_samples], a[-fade_samples:]*fade_out + b[:fade_samples]*fade_in, b[fade_samples:]],
        axis=0
    )

def process_job(job_id: str, text: str, prompt_path: str, prompt_transcript: str):
    jobs[job_id]['status'] = 'running'
    try:
        chunks = preprocess_text(text)
        full_audio = None
        sr = None
        global_tokens = None
        total_in = total_out = 0

        for idx, chunk in enumerate(chunks):
            proc_args = {'text': chunk, 'return_tensors': 'pt'}
            if prompt_path and os.path.exists(prompt_path):
                proc_args['prompt_speech_path'] = prompt_path
                proc_args['prompt_text'] = prompt_transcript
            inputs = processor(**proc_args).to(device)
            total_in += inputs['input_ids'].shape[-1]

            if idx == 0:
                global_tokens = inputs.pop('global_token_ids_prompt', None)
            else:
                inputs.pop('global_token_ids_prompt', None)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    do_sample=True, temperature=0.8, top_k=50, top_p=0.95,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
            total_out += output_ids.shape[-1]

            audio_dict = processor.decode(
                generated_ids=output_ids,
                global_token_ids_prompt=global_tokens,
                input_ids_len=inputs['input_ids'].shape[-1]
            )
            audio = np.array(audio_dict['audio'], dtype=np.float32)
            rate  = audio_dict['sampling_rate']
            if sr is None: sr = rate
            elif sr != rate: raise RuntimeError("Sampling rate mismatch")

            if full_audio is None:
                full_audio = audio
            else:
                breath = np.zeros(int(0.3 * sr), dtype=np.float32)
                full_audio = cross_fade(np.concatenate([full_audio, breath]), audio, sr, 0.15)

        silence = np.zeros(int(sr * 0.15), dtype=np.float32)
        full_audio = cross_fade(silence, full_audio, sr, 0.01)
        full_audio = cross_fade(full_audio, silence, sr, 0.01)

        os.makedirs("jobs", exist_ok=True)
        out_path = f"jobs/{job_id}.wav"
        sf.write(out_path, full_audio, sr)
        jobs[job_id].update({
            "status":"done",
            "path": out_path,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out
        })
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)

@app.post("/clone_tts")
async def clone_tts(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    text = data.get("text","").strip()
    if not text:
        return JSONResponse({"error":"Missing text"}, status_code=400)
    prompt_path = data.get("prompt_path") or DEFAULT_PROMPT_PATH
    prompt_transcript = data.get("prompt_transcript") or DEFAULT_PROMPT_TRANSCRIPT
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status":"pending"}
    background_tasks.add_task(process_job, job_id, text, prompt_path, prompt_transcript)
    return JSONResponse({"job_id": job_id}, status_code=202)

@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job: raise HTTPException(404,"Job not found")
    return job | {"job_id": job_id}

@app.get("/result/{job_id}")
async def result(job_id: str):
    job = jobs.get(job_id)
    if not job: raise HTTPException(404,"Job not found")
    if job["status"] != "done":
        return JSONResponse({"error":"Result not ready","status":job["status"]}, status_code=202)
    return FileResponse(job["path"], media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("jobs_app_on_demand:app", host="0.0.0.0", port=5300)

