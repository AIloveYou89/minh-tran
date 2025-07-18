# jobs_app1.py (Async version)
import os
import uuid
import threading
import numpy as np
import torch
import soundfile as sf
from flask import Flask, request, jsonify, send_file, abort
from transformers import AutoProcessor, AutoModel
from preprocess import preprocess_text
from typing import TypeAlias, Dict

app = Flask(__name__)

# Initialize model & processor once
device = "cuda"
MODEL_ID = "DragonLineageAI/Vi-SparkTTS-0.5B"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()
processor.model = model

# Cross-fade helper function matching sync version
def cross_fade(a: np.ndarray, b: np.ndarray, sr: int, fade_duration: float = 0.15) -> np.ndarray:
    fade_samples = int(sr * fade_duration)
    if fade_samples <= 0 or len(a) < fade_samples or len(b) < fade_samples:
        return np.concatenate([a, b], axis=0)
    out_fade = np.linspace(1.0, 0.0, fade_samples)
    in_fade = out_fade[::-1]
    a_tail = a[-fade_samples:] * out_fade
    b_head = b[:fade_samples] * in_fade
    return np.concatenate([a[:-fade_samples], a_tail + b_head, b[fade_samples:]], axis=0)
# Global job store
Job = Dict[str, Any]      # định nghĩa kiểu cho rõ ràng
jobs: Dict[str, Job] = {} # jobs[job_id] = {'status': ..., 'path': ..., …}
# jobs[job_id] = { status, path?, error?, total_in?, total_out? }

# Worker function to process TTS asynchronously
def process_job(job_id: str, text: str, prompt_path: str, prompt_transcript: str):
    jobs[job_id]['status'] = 'running'
    try:
        chunks = preprocess_text(text)
        full_audio = None
        sr = None
        global_tokens = None
        total_in, total_out = 0, 0
        for idx, chunk in enumerate(chunks):
            proc_args = {'text': chunk, 'return_tensors': 'pt'}
            if prompt_path and os.path.exists(prompt_path):
                proc_args['prompt_speech_path'] = prompt_path
                proc_args['prompt_text'] = prompt_transcript
            inputs = processor(**proc_args).to(device)
            in_toks = inputs['input_ids'].shape[-1]
            total_in += in_toks
            if idx == 0:
                global_tokens = inputs.pop('global_token_ids_prompt', None)
            else:
                inputs.pop('global_token_ids_prompt', None)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=1000,
                    do_sample=True, temperature=0.8, top_k=50, top_p=0.95,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
            out_toks = output_ids.shape[-1]
            total_out += out_toks
            audio_dict = processor.decode(
                generated_ids=output_ids,
                global_token_ids_prompt=global_tokens,
                input_ids_len=in_toks
            )
            audio = np.array(audio_dict['audio'], dtype=np.float32)
            rate = audio_dict['sampling_rate']
            if sr is None:
                sr = rate
            elif sr != rate:
                raise RuntimeError('Sampling rate mismatch')
            if full_audio is None:
                full_audio = audio
            else:
                breath = np.zeros(int(0.3 * sr), dtype=np.float32)
                temp = np.concatenate([full_audio, breath], axis=0)
                full_audio = cross_fade(temp, audio, sr, fade_duration=0.15)
        silence = np.zeros(int(sr * 0.15), dtype=np.float32)
        full_audio = cross_fade(silence, full_audio, sr, fade_duration=0.01)
        full_audio = cross_fade(full_audio, silence, sr, fade_duration=0.01)
        out_dir = 'jobs'
        os.makedirs(out_dir, exist_ok=True)
        wav_path = os.path.join(out_dir, f'{job_id}.wav')
        sf.write(wav_path, full_audio, sr)
        jobs[job_id].update({ 'status': 'done', 'path': wav_path,
                              'total_input_tokens': total_in,
                              'total_output_tokens': total_out })
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)

@app.route('/clone_tts', methods=['POST'])
def submit_tts():
    data = request.json or {}
    text = data.get('text','').strip()
    if not text:
        return jsonify({'error':'Missing text'}),400
    prompt_path = data.get('prompt_path','')
    prompt_transcript = data.get('prompt_transcript',
        'Tôi là chủ sở hữu giọng nói này và tôi đồng ý cho Google sử dụng giọng nói này để tạo mô hình giọng nói tổng hợp.')
    job_id = str(uuid.uuid4())
    jobs[job_id] = { 'status':'pending' }
    threading.Thread(target=process_job,
                     args=(job_id,text,prompt_path,prompt_transcript),
                     daemon=True).start()
    return jsonify({'job_id':job_id}),202

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = jobs.get(job_id)
    if not job:
        abort(404)
    resp = {'job_id':job_id,'status':job['status']}
    if job.get('error'): resp['error']=job['error']
    return jsonify(resp)

@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    job = jobs.get(job_id)
    if not job:
        abort(404)
    if job['status'] != 'done':
        return jsonify({'error':'Result not ready','status':job['status']}),202
    return send_file(job['path'],mimetype='audio/wav')

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5003)
