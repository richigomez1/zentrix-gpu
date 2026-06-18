#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Zentrix GPU — RunPod Smart Start Script
# First run: installs everything to /runpod-volume (persists)
# Next runs: skips install, starts server immediately
# ═══════════════════════════════════════════════════════════════

VOLUME="/runpod-volume"
APP_DIR="$VOLUME/zentrix-app"
VENV_DIR="$VOLUME/zentrix-venv"
INSTALLED_FLAG="$VOLUME/.zentrix-installed"

export HF_HOME="$VOLUME/huggingface"
export HF_HUB_CACHE="$VOLUME/huggingface/hub"

echo "============================================"
echo "🚀 Zentrix GPU Server"
echo "============================================"

# ─── First-time install (only runs once) ─────────────────────
if [ ! -f "$INSTALLED_FLAG" ]; then
    echo "📦 First run — installing dependencies to Network Volume..."
    echo "   (This takes ~5 min. Next time it starts instantly.)"
    
    pip install --target="$VENV_DIR" --no-cache-dir \
        "peft>=0.17.0" \
        "diffusers>=0.34.0" \
        "transformers>=4.48.0" \
        "accelerate>=0.30.0" \
        "av>=14.0.0" \
        "Pillow>=10.0.0" \
        "imageio[ffmpeg]>=2.30.0" \
        "scipy>=1.10.0" \
        "sentencepiece>=0.1.99" \
        "protobuf>=3.20.0" \
        "soundfile>=0.12.0" \
        "huggingface-hub>=0.25.0" \
        "fastapi>=0.115.0" \
        "uvicorn>=0.29.0"
    
    echo "✅ Dependencies installed"
    
    # Download app code
    mkdir -p "$APP_DIR"
    
    # Create handler.py
    cat > "$APP_DIR/handler.py" << 'HANDLER_EOF'
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
    def __init__(self):
        self.current_model = None
        self.pipe = None
        self.extra = {}
        self.hf_token = os.environ.get("HF_TOKEN", None)
        cache = os.environ.get("HF_HOME", "/runpod-volume/huggingface")
        print(f"🎬 ModelManager v5 | cache={cache} | token={'yes' if self.hf_token else 'no'}")

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
            print(f"  ✅ {name} unloaded")

    def load(self, model_name):
        if self.current_model == model_name:
            return self.pipe
        self.unload()
        device = "cuda:0"
        print(f"  ⏳ Loading {model_name}...")

        if model_name == "ltx-2":
            from diffusers.pipelines.ltx2 import LTX2Pipeline
            self.pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16, token=self.hf_token)
            self.pipe.enable_sequential_cpu_offload(device=device)
            if hasattr(self.pipe, 'vocoder') and hasattr(self.pipe.vocoder, 'config'):
                self.extra["audio_sample_rate"] = getattr(self.pipe.vocoder.config, "output_sample_rate", getattr(self.pipe.vocoder.config, "sampling_rate", 24000))
            else:
                self.extra["audio_sample_rate"] = 24000

        elif model_name == "wan2.2-t2v":
            from diffusers import WanPipeline
            for repo in ["Wan-AI/Wan2.2-T2V-A14B-Diffusers", "Wan-AI/Wan2.2-T2V-A14B"]:
                try:
                    self.pipe = WanPipeline.from_pretrained(repo, torch_dtype=torch.float16, token=self.hf_token)
                    print(f"  ✅ Loaded from {repo}")
                    break
                except Exception as e:
                    print(f"  ⚠️ {repo} failed: {e}")
            if self.pipe is None:
                raise RuntimeError("Could not load Wan2.2-T2V")
            self.pipe.enable_model_cpu_offload()

        elif model_name == "flux2":
            from diffusers import FluxPipeline
            self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16, token=self.hf_token)
            self.pipe.enable_model_cpu_offload()

        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.current_model = model_name
        print(f"  ✅ {model_name} ready")
        return self.pipe

manager = ModelManager()

class EndpointHandler:
    def __init__(self, path=""):
        print("✅ Zentrix Endpoint v5 ready | Models: ltx-2, wan2.2-t2v, flux2")

    def __call__(self, data):
        try:
            inputs = data.get("inputs", data)
            params = data.get("parameters", {})
            model_name = inputs.get("model", "ltx-2")
            prompt = inputs.get("prompt", "")
            print(f"📥 model={model_name}, prompt={prompt[:80]}...")
            if model_name == "ltx-2":
                return self._generate_ltx2(inputs, params)
            elif model_name == "wan2.2-t2v":
                return self._generate_wan_t2v(inputs, params)
            elif model_name == "flux2":
                return self._generate_image(inputs, params)
            else:
                return {"error": f"Unknown model: {model_name}"}
        except Exception as e:
            tb = traceback.format_exc()
            print(f"❌ {e}\n{tb}")
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

    def _generate_ltx2(self, inputs, params):
        from diffusers.pipelines.ltx2.export_utils import encode_video
        pipe = manager.load("ltx-2")
        prompt = inputs.get("prompt", "A cinematic scene")
        negative = params.get("negative_prompt", "shaky, glitchy, low quality")
        image = self._decode_image(inputs)
        w, h = params.get("width", 768), params.get("height", 512)
        nf = params.get("num_frames", 25)
        fps = params.get("frame_rate", 24.0)
        steps = params.get("num_inference_steps", 40)
        gs = params.get("guidance_scale", 4.0)
        w, h = (w//32)*32, (h//32)*32
        nf = ((nf-1)//8)*8+1
        print(f"  🎬 LTX-2: {w}x{h}, {nf}f, {steps}steps")
        kwargs = {"prompt": prompt, "negative_prompt": negative, "width": w, "height": h, "num_frames": nf, "frame_rate": fps, "num_inference_steps": steps, "guidance_scale": gs, "output_type": "np", "return_dict": False}
        if image: kwargs["image"] = image
        result = pipe(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            video_np, audio_tensor = result
        else:
            video_np, audio_tensor = result, None
        audio_sr = manager.extra.get("audio_sample_rate", 24000)
        manager.unload()
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
                print(f"  ⚠️ Audio: {e}")
        vd = video_np[0] if isinstance(video_np, (list, np.ndarray)) and len(video_np) > 0 else video_np
        encode_video(vd, **export_kwargs)
        del video_np, audio_tensor; gc.collect()
        with open(tmp_path, "rb") as f: vb = f.read()
        os.unlink(tmp_path)
        mb = len(vb)/(1024*1024)
        print(f"  📦 Video: {mb:.1f}MB")
        return {"type": "video", "data": base64.b64encode(vb).decode(), "content_type": "video/mp4", "model": "ltx-2", "has_audio": has_audio, "num_frames": nf, "width": w, "height": h, "size_mb": round(mb,2)}

    def _generate_wan_t2v(self, inputs, params):
        from diffusers.utils import export_to_video
        pipe = manager.load("wan2.2-t2v")
        prompt = inputs.get("prompt", "Slow cinematic camera")
        w, h = params.get("width", 1280), params.get("height", 720)
        nf = params.get("num_frames", 49)
        gs = params.get("guidance_scale", 5.0)
        steps = params.get("num_inference_steps", 30)
        w, h = (w//16)*16, (h//16)*16
        print(f"  🎬 Wan2.2: {w}x{h}, {nf}f, {steps}steps")
        output = pipe(prompt=prompt, num_frames=nf, guidance_scale=gs, num_inference_steps=steps, width=w, height=h)
        frames = list(output.frames[0]); del output; manager.unload()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp: tmp_path = tmp.name
        export_to_video(frames, tmp_path, fps=16); del frames; gc.collect()
        with open(tmp_path, "rb") as f: vb = f.read()
        os.unlink(tmp_path)
        mb = len(vb)/(1024*1024)
        print(f"  📦 Video: {mb:.1f}MB")
        return {"type": "video", "data": base64.b64encode(vb).decode(), "content_type": "video/mp4", "model": "wan2.2-t2v", "num_frames": nf, "width": w, "height": h, "size_mb": round(mb,2)}

    def _generate_image(self, inputs, params):
        pipe = manager.load("flux2")
        prompt = inputs.get("prompt", "")
        w, h = params.get("width", 1024), params.get("height", 1024)
        steps = params.get("num_inference_steps", 28)
        gs = params.get("guidance_scale", 3.5)
        print(f"  🖼️ FLUX.2: {w}x{h}, {steps}steps")
        result = pipe(prompt=prompt, guidance_scale=gs, num_inference_steps=steps, width=w, height=h)
        image = result.images[0]; del result
        buf = io.BytesIO(); image.save(buf, format="PNG"); ib = buf.getvalue()
        mb = len(ib)/(1024*1024)
        print(f"  📦 Image: {mb:.1f}MB")
        return {"type": "image", "data": base64.b64encode(ib).decode(), "content_type": "image/png", "model": "flux2", "width": w, "height": h}
HANDLER_EOF

    # Create server.py
    cat > "$APP_DIR/server.py" << 'SERVER_EOF'
import os, time, traceback, torch
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from handler import EndpointHandler

handler = None

@asynccontextmanager
async def lifespan(app):
    global handler
    handler = EndpointHandler()
    print("✅ Server ready")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
@app.get("/")
def health():
    return {"status": "ok", "gpu": torch.cuda.is_available()}

@app.post("/")
async def predict(request: Request):
    if handler is None:
        return JSONResponse(status_code=503, content={"error": "Not ready"})
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
    t0 = time.time()
    try:
        result = handler(data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})
    print(f"⏱️ Done in {time.time()-t0:.1f}s")
    return JSONResponse(content=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🌐 Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=600)
SERVER_EOF

    touch "$INSTALLED_FLAG"
    echo "✅ First-time setup complete!"
else
    echo "⚡ Dependencies already installed — starting instantly"
fi

# ─── Add installed packages to Python path ───────────────────
export PYTHONPATH="$VENV_DIR:$APP_DIR:$PYTHONPATH"

# ─── Start server ────────────────────────────────────────────
echo "============================================"
echo "🌐 Starting Zentrix GPU Server..."
echo "============================================"
cd "$APP_DIR"
python server.py
