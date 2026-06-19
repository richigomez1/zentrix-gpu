#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Zentrix GPU — RunPod Start Script v13
# LTX-2 (video+audio) + FLUX.2-dev (images)
# ═══════════════════════════════════════════════════════════════

VOLUME="/runpod-volume"
APP_DIR="$VOLUME/zentrix-app"
INSTALLED_FLAG="$VOLUME/.zentrix-v13c-installed"

export HF_HOME="$VOLUME/huggingface"
export HF_HUB_CACHE="$VOLUME/huggingface/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "============================================"
echo "🚀 Zentrix GPU Server v13"
echo "   Models: LTX-2 (video+audio), FLUX.2 (images)"
echo "============================================"

# ─── 1. Upgrade PyTorch for CUDA 12.4 ────────────────────────
echo "📦 Upgrading PyTorch..."
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
    "ftfy" \
    "einops" 2>&1 | tail -5
echo "✅ Packages ready"

python -c "import torch; print(f'🔥 PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"

# ─── 3. Create/update app code ───────────────────────────────
if [ ! -f "$INSTALLED_FLAG" ]; then
    echo "📦 Creating app code v13..."
    mkdir -p "$APP_DIR"

    # ── handler.py ────────────────────────────────────────────
    cat > "$APP_DIR/handler.py" << 'HANDLER_EOF'
"""
Zentrix Handler v13 — LTX-2 (video+audio) + FLUX.2-dev (images)
Uses ModelManager to load/unload one model at a time.
"""
import torch
import base64
import io
import tempfile
import os
import gc
import traceback
import numpy as np
from typing import Dict, Any
from PIL import Image


class ModelManager:
    """Loads/unloads models one at a time to manage VRAM."""

    def __init__(self):
        self.current_model = None
        self.pipe = None
        self.extra = {}
        self.hf_token = os.environ.get("HF_TOKEN", None)
        cache = os.environ.get("HF_HOME", "/runpod-volume/huggingface")
        print(f"🎬 ModelManager v13 | cache={cache} | token={'✅' if self.hf_token else '❌'}")

    def unload(self):
        if self.pipe is not None:
            name = self.current_model
            del self.pipe
            self.pipe = None
            self.current_model = None
            self.extra = {}
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  ✅ {name} unloaded, VRAM freed")

    def load(self, model_name: str):
        if self.current_model == model_name:
            return self.pipe
        self.unload()
        print(f"  ⏳ Loading {model_name}...")

        if model_name == "ltx-2":
            from diffusers.pipelines.ltx2 import LTX2Pipeline
            self.pipe = LTX2Pipeline.from_pretrained(
                "Lightricks/LTX-2",
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
            )
            self.pipe.enable_sequential_cpu_offload(device="cuda:0")
            if hasattr(self.pipe, 'vocoder') and self.pipe.vocoder and hasattr(self.pipe.vocoder, 'config'):
                self.extra["audio_sample_rate"] = getattr(
                    self.pipe.vocoder.config, "output_sample_rate",
                    getattr(self.pipe.vocoder.config, "sampling_rate", 24000)
                )
            else:
                self.extra["audio_sample_rate"] = 24000

        elif model_name == "ltx-2-i2v":
            from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline
            self.pipe = LTX2ImageToVideoPipeline.from_pretrained(
                "Lightricks/LTX-2",
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
            )
            self.pipe.enable_sequential_cpu_offload(device="cuda:0")
            if hasattr(self.pipe, 'vocoder') and self.pipe.vocoder and hasattr(self.pipe.vocoder, 'config'):
                self.extra["audio_sample_rate"] = getattr(
                    self.pipe.vocoder.config, "output_sample_rate",
                    getattr(self.pipe.vocoder.config, "sampling_rate", 24000)
                )
            else:
                self.extra["audio_sample_rate"] = 24000

        elif model_name == "flux2":
            from diffusers import FluxPipeline
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
            )
            self.pipe.enable_model_cpu_offload()

        else:
            raise ValueError(f"Unknown model: {model_name}. Options: ltx-2, flux2")

        self.current_model = model_name
        print(f"  ✅ {model_name} loaded and ready")
        return self.pipe


manager = ModelManager()


class EndpointHandler:
    def __init__(self, path=""):
        print("✅ Zentrix Endpoint v13 ready")
        print("   Models: ltx-2 (video+audio), flux2 (images)")

    def __call__(self, data: Dict[str, Any]) -> Any:
        try:
            inputs = data.get("inputs", data)
            params = data.get("parameters", {})
            model_name = inputs.get("model", "ltx-2")
            prompt = inputs.get("prompt", "")
            print(f"📥 model={model_name}, prompt={prompt[:80]}...")

            if model_name == "ltx-2":
                return self._generate_ltx2(inputs, params)
            elif model_name == "flux2":
                return self._generate_flux2(inputs, params)
            else:
                return {"error": f"Unknown model: {model_name}. Options: ltx-2, flux2"}
        except Exception as e:
            tb = traceback.format_exc()
            print(f"❌ Error: {e}\n{tb}")
            return {"error": str(e), "traceback": tb}

    def _decode_image(self, inputs):
        image_data = inputs.get("image", "")
        if not image_data:
            return None
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size
        w, h = (w // 32) * 32, (h // 32) * 32
        if w != image.size[0] or h != image.size[1]:
            image = image.resize((w, h), Image.LANCZOS)
        return image

    # ── LTX-2: video + audio ─────────────────────────────────

    def _generate_ltx2(self, inputs, params):
        import time
        from diffusers.pipelines.ltx2.export_utils import encode_video

        prompt = inputs.get("prompt", "A beautiful cinematic scene")
        negative = params.get("negative_prompt", "shaky, glitchy, low quality, worst quality")
        image = self._decode_image(inputs)

        # Use I2V pipeline when image is present, T2V otherwise
        if image is not None:
            pipe = manager.load("ltx-2-i2v")
            print(f"  📸 Using I2V pipeline (image provided)")
        else:
            pipe = manager.load("ltx-2")
            print(f"  📝 Using T2V pipeline (text only)")

        w = params.get("width", 768)
        h = params.get("height", 512)
        nf = params.get("num_frames", 25)
        fps = params.get("frame_rate", 24.0)
        steps = params.get("num_inference_steps", 40)
        gs = params.get("guidance_scale", 4.0)

        w, h = (w // 32) * 32, (h // 32) * 32
        nf = ((nf - 1) // 8) * 8 + 1

        print(f"  🎬 LTX-2: {w}x{h}, {nf} frames, {steps} steps")

        kwargs = {
            "prompt": prompt, "negative_prompt": negative,
            "width": w, "height": h, "num_frames": nf,
            "frame_rate": fps, "num_inference_steps": steps,
            "guidance_scale": gs, "output_type": "np", "return_dict": False,
        }
        if image is not None:
            kwargs["image"] = image

        t0 = time.time()
        result = pipe(**kwargs)
        elapsed = time.time() - t0
        print(f"  ⏱️ Inference: {elapsed:.0f}s")

        if isinstance(result, tuple) and len(result) == 2:
            video_np, audio_tensor = result
        else:
            video_np = result
            audio_tensor = None

        audio_sr = manager.extra.get("audio_sample_rate", 24000)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        export_kwargs = {"fps": fps, "output_path": tmp_path}
        has_audio = False
        if audio_tensor is not None:
            try:
                a = audio_tensor[0].float().cpu() if hasattr(audio_tensor, '__getitem__') else audio_tensor.float().cpu()
                export_kwargs["audio"] = a
                export_kwargs["audio_sample_rate"] = audio_sr
                has_audio = True
            except Exception as e:
                print(f"  ⚠️ Audio export: {e}")

        vd = video_np[0] if isinstance(video_np, (list, np.ndarray)) and len(video_np) > 0 else video_np
        encode_video(vd, **export_kwargs)

        del video_np, audio_tensor
        gc.collect()

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        os.unlink(tmp_path)

        mb = len(video_bytes) / (1024 * 1024)
        print(f"  📦 Video: {mb:.1f}MB, audio={has_audio} — SUCCESS ({elapsed:.0f}s)")

        return {
            "type": "video",
            "data": base64.b64encode(video_bytes).decode("utf-8"),
            "content_type": "video/mp4",
            "model": "ltx-2",
            "has_audio": has_audio,
            "num_frames": nf, "width": w, "height": h,
            "size_mb": round(mb, 2),
            "elapsed_seconds": round(elapsed, 1),
        }

    # ── FLUX.2-dev: images ───────────────────────────────────

    def _generate_flux2(self, inputs, params):
        import time
        pipe = manager.load("flux2")
        prompt = inputs.get("prompt", "")

        w = params.get("width", 1024)
        h = params.get("height", 1024)
        steps = params.get("num_inference_steps", 28)
        gs = params.get("guidance_scale", 3.5)

        print(f"  🖼️ FLUX.2: {w}x{h}, {steps} steps")

        t0 = time.time()
        result = pipe(
            prompt=prompt, guidance_scale=gs,
            num_inference_steps=steps, width=w, height=h,
        )
        elapsed = time.time() - t0
        image = result.images[0]
        del result

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        mb = len(img_bytes) / (1024 * 1024)
        print(f"  📦 Image: {mb:.1f}MB — SUCCESS ({elapsed:.0f}s)")

        return {
            "type": "image",
            "data": base64.b64encode(img_bytes).decode("utf-8"),
            "content_type": "image/png",
            "model": "flux2",
            "width": w, "height": h,
            "size_mb": round(mb, 2),
            "elapsed_seconds": round(elapsed, 1),
        }
HANDLER_EOF

    # ── server.py (async job handling) ────────────────────────
    cat > "$APP_DIR/server.py" << 'SERVER_EOF'
"""
Zentrix GPU Server v13 — Async job handling
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
print("🚀 Zentrix GPU Server v13 (async)")
print(f"  torch=={torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
print(f"  Models: LTX-2 (video+audio), FLUX.2 (images)")
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
    return {
        "status": "ok", "gpu": torch.cuda.is_available(),
        "version": "v13", "models": ["ltx-2", "flux2"],
        "active_jobs": active,
    }

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
    echo "✅ App code v13 created!"
else
    echo "⚡ App code v13 exists"
fi

# ─── Start server ────────────────────────────────────────────
echo "============================================"
echo "🌐 Starting Zentrix GPU Server..."
echo "============================================"
cd "$APP_DIR"
export PYTHONPATH="$APP_DIR:$PYTHONPATH"
python server.py
