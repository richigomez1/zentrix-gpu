#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Zentrix GPU — RunPod Start Script v12
# Wan2.2 native (async) + LTX-2 diffusers
# ═══════════════════════════════════════════════════════════════

VOLUME="/runpod-volume"
APP_DIR="$VOLUME/zentrix-app"
WAN_DIR="$VOLUME/Wan2.2"
WAN_WEIGHTS="$VOLUME/Wan2.2-T2V-A14B"
INSTALLED_FLAG="$VOLUME/.zentrix-v12-installed"

export HF_HOME="$VOLUME/huggingface"
export HF_HUB_CACHE="$VOLUME/huggingface/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "============================================"
echo "🚀 Zentrix GPU Server v12"
echo "============================================"

# ─── 1. Upgrade PyTorch to 2.5.1 for CUDA 12.4 ──────────────
echo "📦 Upgrading PyTorch for CUDA 12.4..."
pip install --no-cache-dir --root-user-action=ignore \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" 2>&1 | tail -3

# ─── 2. Install packages ─────────────────────────────────────
echo "📦 Installing packages..."
pip install --upgrade --no-cache-dir --root-user-action=ignore \
    "diffusers>=0.38.0" \
    "transformers>=4.52.0" \
    "accelerate>=1.0.0" \
    "peft>=0.17.0" \
    "av>=14.0.0" \
    "soundfile>=0.12.0" \
    "sentencepiece>=0.1.99" \
    "protobuf>=3.20.0" \
    "imageio[ffmpeg]>=2.30.0" \
    "Pillow>=10.0.0" \
    "scipy>=1.10.0" \
    "huggingface-hub>=0.25.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.29.0" \
    "easydict" \
    "ftfy" \
    "einops" \
    "opencv-python-headless" \
    "librosa" \
    "decord" \
    "dashscope" \
    "flash_attn" 2>&1 | tail -5
echo "✅ System packages ready"

python -c "import torch; print(f'🔥 PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"

# ─── 3. Clone Wan2.2 repo (once) ─────────────────────────────
if [ ! -d "$WAN_DIR" ]; then
    echo "📦 Cloning Wan2.2 repo..."
    cd "$VOLUME"
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd "$WAN_DIR"
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt 2>&1 | tail -3
    echo "✅ Wan2.2 repo cloned"
else
    echo "⚡ Wan2.2 repo exists"
fi

# ─── 4. Download Wan2.2 model weights (once, ~50GB) ──────────
weight_count=$(find "$WAN_WEIGHTS" -name "*.safetensors" 2>/dev/null | wc -l)
if [ ! -d "$WAN_WEIGHTS" ] || [ "$weight_count" -lt 5 ]; then
    rm -rf "$WAN_WEIGHTS"
    echo "📦 Downloading Wan2.2-T2V-A14B weights (~50GB, may take 10-15 min)..."
    huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir "$WAN_WEIGHTS"
    echo "✅ Wan2.2 weights downloaded"
else
    echo "⚡ Wan2.2 weights exist ($weight_count safetensors)"
fi

# ─── 5. Create/update app code ───────────────────────────────
if [ ! -f "$INSTALLED_FLAG" ]; then
    echo "📦 Creating app code v12..."
    mkdir -p "$APP_DIR"

    # ── handler.py ────────────────────────────────────────────
    cat > "$APP_DIR/handler.py" << 'HANDLER_EOF'
"""
Zentrix Handler v12 — Wan2.2 native (generate.py) + LTX-2 diffusers
"""
import torch
import base64
import io
import tempfile
import os
import gc
import subprocess
import traceback
import glob
import time
import numpy as np
from typing import Dict, Any
from PIL import Image

WAN_DIR = "/runpod-volume/Wan2.2"
WAN_WEIGHTS = "/runpod-volume/Wan2.2-T2V-A14B"

class EndpointHandler:
    def __init__(self, path=""):
        self.ltx2_pipe = None
        models = []
        if os.path.isdir(WAN_WEIGHTS):
            models.append("wan2.2-t2v")
        models.append("ltx-2")
        print(f"✅ Zentrix Endpoint v12 ready | Models: {', '.join(models)}")

    def __call__(self, data):
        try:
            inputs = data.get("inputs", data)
            params = data.get("parameters", {})
            model_name = inputs.get("model", "wan2.2-t2v")
            prompt = inputs.get("prompt", "")
            print(f"📥 model={model_name}, prompt={prompt[:80]}...")
            if model_name == "wan2.2-t2v":
                return self._generate_wan22(inputs, params)
            elif model_name == "ltx-2":
                return self._generate_ltx2(inputs, params)
            else:
                return {"error": f"Unknown model: {model_name}. Options: wan2.2-t2v, ltx-2"}
        except Exception as e:
            tb = traceback.format_exc()
            print(f"❌ {e}\n{tb}")
            return {"error": str(e), "traceback": tb}

    # ── Wan 2.2 T2V (native generate.py) ──────────────────────
    def _generate_wan22(self, inputs, params):
        prompt = inputs.get("prompt", "A cinematic scene")
        width = params.get("width", 1280)
        height = params.get("height", 720)
        size = f"{width}*{height}"

        out_dir = tempfile.mkdtemp(prefix="wan_")
        print(f"  🎬 Wan2.2: {size}, prompt={prompt[:60]}...")

        cmd = [
            "python", f"{WAN_DIR}/generate.py",
            "--task", "t2v-A14B",
            "--size", size,
            "--ckpt_dir", WAN_WEIGHTS,
            "--offload_model", "True",
            "--convert_model_dtype",
            "--prompt", prompt,
            "--src_root_path", out_dir,
        ]

        t0 = time.time()
        print(f"  ⏳ Running Wan2.2 generate.py...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=WAN_DIR, env={**os.environ, "PYTHONPATH": WAN_DIR}
        )

        elapsed = time.time() - t0
        print(f"  ⏱️ generate.py finished in {elapsed:.0f}s (exit={result.returncode})")

        if result.returncode != 0:
            print(f"  STDOUT: {result.stdout[-300:]}")
            print(f"  STDERR: {result.stderr[-500:]}")

        # Find output video
        video_files = glob.glob(os.path.join(out_dir, "**/*.mp4"), recursive=True)
        if not video_files:
            video_files = glob.glob(os.path.join(out_dir, "*.mp4"))
        if not video_files:
            # Try WAN_DIR as fallback
            video_files = glob.glob(os.path.join(WAN_DIR, "*.mp4"))
        if not video_files:
            stderr_tail = result.stderr[-500:] if result.stderr else "no stderr"
            stdout_tail = result.stdout[-500:] if result.stdout else "no stdout"
            raise Exception(f"Wan2.2 no generó video. Exit={result.returncode}. stderr={stderr_tail} stdout={stdout_tail}")

        video_path = video_files[0]
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        # Cleanup
        for vf in video_files:
            try: os.unlink(vf)
            except: pass
        try: os.rmdir(out_dir)
        except: pass

        mb = len(video_bytes) / (1024*1024)
        print(f"  📦 Video: {mb:.1f}MB — SUCCESS")
        return {
            "type": "video",
            "data": base64.b64encode(video_bytes).decode(),
            "content_type": "video/mp4",
            "model": "wan2.2-t2v",
            "width": width, "height": height,
            "size_mb": round(mb, 2),
            "elapsed_seconds": round(elapsed, 1),
        }

    # ── LTX-2 (diffusers) ────────────────────────────────────
    def _generate_ltx2(self, inputs, params):
        from diffusers.pipelines.ltx2 import LTX2Pipeline
        from diffusers.pipelines.ltx2.export_utils import encode_video

        if self.ltx2_pipe is None:
            print("  ⏳ Loading LTX-2 pipeline...")
            self.ltx2_pipe = LTX2Pipeline.from_pretrained(
                "Lightricks/LTX-2", torch_dtype=torch.bfloat16
            )
            self.ltx2_pipe.enable_sequential_cpu_offload(device="cuda:0")
            print("  ✅ LTX-2 loaded")

        pipe = self.ltx2_pipe
        prompt = inputs.get("prompt", "A cinematic scene")
        negative = params.get("negative_prompt", "shaky, glitchy, low quality, worst quality")
        w = params.get("width", 768)
        h = params.get("height", 512)
        nf = params.get("num_frames", 25)
        fps = params.get("frame_rate", 24.0)
        steps = params.get("num_inference_steps", 40)
        gs = params.get("guidance_scale", 4.0)

        w, h = (w//32)*32, (h//32)*32
        nf = ((nf-1)//8)*8+1
        print(f"  🎬 LTX-2: {w}x{h}, {nf}f, {steps}steps")

        pipe.vae.enable_tiling()

        t0 = time.time()
        video, audio = pipe(
            prompt=prompt, negative_prompt=negative,
            width=w, height=h, num_frames=nf, frame_rate=fps,
            num_inference_steps=steps, guidance_scale=gs,
            output_type="np", return_dict=False,
        )
        elapsed = time.time() - t0
        print(f"  ⏱️ LTX-2 inference: {elapsed:.0f}s")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        audio_sr = pipe.vocoder.config.output_sampling_rate if hasattr(pipe, 'vocoder') and pipe.vocoder else 24000
        export_kwargs = {"fps": fps, "output_path": tmp_path}
        has_audio = False
        if audio is not None:
            try:
                a = audio[0].float().cpu() if hasattr(audio, '__getitem__') else audio.float().cpu()
                export_kwargs["audio"] = a
                export_kwargs["audio_sample_rate"] = audio_sr
                has_audio = True
            except Exception as e:
                print(f"  ⚠️ Audio: {e}")

        vd = video[0] if isinstance(video, (list, np.ndarray)) and len(video) > 0 else video
        encode_video(vd, **export_kwargs)

        with open(tmp_path, "rb") as f:
            vb = f.read()
        os.unlink(tmp_path)

        mb = len(vb)/(1024*1024)
        print(f"  📦 Video: {mb:.1f}MB, audio={has_audio} — SUCCESS")
        return {
            "type": "video",
            "data": base64.b64encode(vb).decode(),
            "content_type": "video/mp4",
            "model": "ltx-2",
            "has_audio": has_audio,
            "num_frames": nf, "width": w, "height": h,
            "size_mb": round(mb, 2),
            "elapsed_seconds": round(elapsed, 1),
        }
HANDLER_EOF

    # ── server.py (async job handling) ────────────────────────
    cat > "$APP_DIR/server.py" << 'SERVER_EOF'
"""
Zentrix GPU Server v12 — Async job handling
POST /generate → returns job_id immediately
GET /status/{job_id} → returns status + result when done
"""
import os, time, traceback, torch, uuid, threading
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from handler import EndpointHandler

print("=" * 50)
print("🚀 Zentrix GPU Server v12 (async)")
print(f"  torch=={torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
print("=" * 50)

jobs = {}
handler = None

@asynccontextmanager
async def lifespan(app):
    global handler
    handler = EndpointHandler()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
@app.get("/")
def health():
    active = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
    return {"status": "ok", "gpu": torch.cuda.is_available(), "version": "v12-async", "active_jobs": active}

def _run_job(job_id, data):
    jobs[job_id]["status"] = "running"
    t0 = time.time()
    try:
        result = handler(data)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
        print(f"✅ Job {job_id} done in {time.time()-t0:.0f}s")
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["result"] = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"❌ Job {job_id} error: {e}")

@app.post("/generate")
async def generate_async(request: Request):
    if handler is None:
        return JSONResponse(status_code=503, content={"error": "Not ready"})
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "result": None, "created": time.time()}
    t = threading.Thread(target=_run_job, args=(job_id, data), daemon=True)
    t.start()
    model = data.get("inputs", {}).get("model", "?")
    print(f"📥 Job {job_id} queued (model={model})")
    return JSONResponse(content={"job_id": job_id, "status": "queued"})

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    resp = {"job_id": job_id, "status": job["status"]}
    if job["status"] in ("done", "error"):
        resp["result"] = job["result"]
    return JSONResponse(content=resp)

# Cleanup old jobs every 5 min
def _cleanup():
    while True:
        time.sleep(300)
        now = time.time()
        stale = [k for k, v in jobs.items() if now - v["created"] > 1800]
        for k in stale:
            jobs.pop(k, None)
threading.Thread(target=_cleanup, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🌐 Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=600)
SERVER_EOF

    touch "$INSTALLED_FLAG"
    echo "✅ App code v12 created!"
else
    echo "⚡ App code v12 exists"
fi

# ─── Start server ────────────────────────────────────────────
echo "============================================"
echo "🌐 Starting Zentrix GPU Server..."
echo "============================================"
cd "$APP_DIR"
export PYTHONPATH="/runpod-volume/Wan2.2:$APP_DIR:$PYTHONPATH"
python server.py
