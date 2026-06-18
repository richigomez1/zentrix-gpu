#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Zentrix GPU — RunPod Start Script v3
# System-wide pip install (no --target conflicts).
# App code on Network Volume. Models cached on Network Volume.
# ═══════════════════════════════════════════════════════════════

VOLUME="/runpod-volume"
APP_DIR="$VOLUME/zentrix-app"
INSTALLED_FLAG="$VOLUME/.zentrix-v6-installed"

export HF_HOME="$VOLUME/huggingface"
export HF_HUB_CACHE="$VOLUME/huggingface/hub"

echo "============================================"
echo "🚀 Zentrix GPU Server v3"
echo "============================================"

# ─── Always upgrade system packages (fast if up-to-date) ─────
echo "📦 Upgrading system packages..."
pip install --upgrade --no-cache-dir --root-user-action=ignore \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch>=2.7.0" \
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
    "uvicorn>=0.29.0" 2>&1 | tail -3
echo "✅ System packages ready"

# Verify torch version (MUST be 2.7+)
python -c "import torch; v=torch.__version__; print(f'🔥 PyTorch {v} | CUDA: {torch.cuda.is_available()}')"

# Verify critical imports
python -c "
from diffusers.pipelines.ltx2 import LTX2Pipeline
from diffusers import WanPipeline
print('✅ LTX2Pipeline + WanPipeline importable')
" 2>&1 || echo "⚠️ Import check had issues — continuing anyway"

# ─── Create app code (only once per version) ─────────────────
if [ ! -f "$INSTALLED_FLAG" ]; then
    echo "📦 Creating app code v3 on Network Volume..."
    mkdir -p "$APP_DIR"

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
        print(f"🎬 ModelManager v7 | token={'yes' if self.hf_token else 'no'}")

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

    def load(self, model_name):
        if self.current_model == model_name:
            return self.pipe
        self.unload()
        device = "cuda:0"
        print(f"  ⏳ Loading {model_name}...")

        if model_name == "ltx-2":
            # LTX-2: audio+video foundation model — uses LTX2Pipeline
            from diffusers.pipelines.ltx2 import LTX2Pipeline
            self.pipe = LTX2Pipeline.from_pretrained(
                "Lightricks/LTX-2",
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
            )
            self.pipe.enable_sequential_cpu_offload(device=device)
            # Get audio sample rate from vocoder
            if hasattr(self.pipe, 'vocoder') and hasattr(self.pipe.vocoder, 'config'):
                self.extra["audio_sample_rate"] = getattr(
                    self.pipe.vocoder.config, "output_sample_rate",
                    getattr(self.pipe.vocoder.config, "sampling_rate", 24000)
                )
            else:
                self.extra["audio_sample_rate"] = 24000
            print(f"  ✅ LTX-2 loaded (audio SR: {self.extra['audio_sample_rate']})")

        elif model_name == "wan2.2-t2v":
            from diffusers import WanPipeline
            self.pipe = WanPipeline.from_pretrained(
                "Wan-AI/Wan2.2-T2V-A14B",
                torch_dtype=torch.float16,
                token=self.hf_token,
            )
            self.pipe.enable_model_cpu_offload()

        elif model_name == "flux2":
            from diffusers import FluxPipeline
            # FLUX.1-dev (stable, gated — needs HF token with access)
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
            )
            self.pipe.enable_model_cpu_offload()

        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.current_model = model_name
        print(f"  ✅ {model_name} ready")
        return self.pipe

manager = ModelManager()

class EndpointHandler:
    def __init__(self, path=""):
        print("✅ Zentrix Endpoint v7 ready | Models: ltx-2, wan2.2-t2v, flux2")

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

    # ── LTX-2: video + audio ──────────────────────────────────
    def _generate_ltx2(self, inputs, params):
        from diffusers.pipelines.ltx2.export_utils import encode_video
        pipe = manager.load("ltx-2")
        prompt = inputs.get("prompt", "A cinematic scene")
        negative = params.get("negative_prompt", "shaky, glitchy, low quality")
        image = self._decode_image(inputs)
        w = params.get("width", 768)
        h = params.get("height", 512)
        nf = params.get("num_frames", 25)
        fps = params.get("frame_rate", 24.0)
        steps = params.get("num_inference_steps", 40)
        gs = params.get("guidance_scale", 4.0)
        w, h = (w//32)*32, (h//32)*32
        nf = ((nf-1)//8)*8+1
        print(f"  🎬 LTX-2: {w}x{h}, {nf}f, {steps}steps")

        kwargs = {
            "prompt": prompt, "negative_prompt": negative,
            "width": w, "height": h, "num_frames": nf,
            "frame_rate": fps, "num_inference_steps": steps,
            "guidance_scale": gs, "output_type": "np", "return_dict": False,
        }
        if image is not None:
            kwargs["image"] = image

        result = pipe(**kwargs)
        # LTX2Pipeline returns (video, audio) tuple
        if isinstance(result, tuple) and len(result) == 2:
            video_np, audio_tensor = result
        else:
            video_np, audio_tensor = result, None

        audio_sr = manager.extra.get("audio_sample_rate", 24000)
        manager.unload()  # Free VRAM before export

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
        del video_np, audio_tensor; gc.collect()

        with open(tmp_path, "rb") as f:
            vb = f.read()
        os.unlink(tmp_path)
        mb = len(vb)/(1024*1024)
        print(f"  📦 Video: {mb:.1f}MB, audio={has_audio}")
        return {
            "type": "video", "data": base64.b64encode(vb).decode(),
            "content_type": "video/mp4", "model": "ltx-2",
            "has_audio": has_audio, "num_frames": nf,
            "width": w, "height": h, "size_mb": round(mb,2),
        }

    # ── Wan 2.2 Text-to-Video ─────────────────────────────────
    def _generate_wan_t2v(self, inputs, params):
        from diffusers.utils import export_to_video
        pipe = manager.load("wan2.2-t2v")
        prompt = inputs.get("prompt", "Slow cinematic camera")
        w = params.get("width", 832)
        h = params.get("height", 480)
        nf = params.get("num_frames", 49)
        gs = params.get("guidance_scale", 5.0)
        steps = params.get("num_inference_steps", 30)
        w, h = (w//16)*16, (h//16)*16
        print(f"  🎬 Wan2.2: {w}x{h}, {nf}f, {steps}steps")
        output = pipe(prompt=prompt, num_frames=nf, guidance_scale=gs, num_inference_steps=steps, width=w, height=h)
        frames = list(output.frames[0]); del output; manager.unload()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        export_to_video(frames, tmp_path, fps=16); del frames; gc.collect()
        with open(tmp_path, "rb") as f:
            vb = f.read()
        os.unlink(tmp_path)
        mb = len(vb)/(1024*1024)
        print(f"  📦 Video: {mb:.1f}MB")
        return {
            "type": "video", "data": base64.b64encode(vb).decode(),
            "content_type": "video/mp4", "model": "wan2.2-t2v",
            "num_frames": nf, "width": w, "height": h, "size_mb": round(mb,2),
        }

    # ── FLUX.1-dev images ─────────────────────────────────────
    def _generate_image(self, inputs, params):
        pipe = manager.load("flux2")
        prompt = inputs.get("prompt", "")
        w = params.get("width", 1024)
        h = params.get("height", 1024)
        steps = params.get("num_inference_steps", 28)
        gs = params.get("guidance_scale", 3.5)
        print(f"  🖼️ FLUX: {w}x{h}, {steps}steps")
        result = pipe(prompt=prompt, guidance_scale=gs, num_inference_steps=steps, width=w, height=h)
        image = result.images[0]; del result
        buf = io.BytesIO(); image.save(buf, format="PNG"); ib = buf.getvalue()
        mb = len(ib)/(1024*1024)
        print(f"  📦 Image: {mb:.1f}MB")
        return {
            "type": "image", "data": base64.b64encode(ib).decode(),
            "content_type": "image/png", "model": "flux2",
            "width": w, "height": h,
        }
HANDLER_EOF

    cat > "$APP_DIR/server.py" << 'SERVER_EOF'
import os, time, traceback, torch
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from handler import EndpointHandler

print("=" * 50)
print("🚀 Zentrix GPU Server v3")
for pkg in ["torch", "diffusers", "transformers", "accelerate", "av"]:
    try:
        mod = __import__(pkg)
        print(f"  {pkg}=={getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  {pkg} ❌")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
print("=" * 50)

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
    return {"status": "ok", "gpu": torch.cuda.is_available(), "version": "v3"}

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
    echo "✅ App code v3 created!"
else
    echo "⚡ App code exists — starting server"
fi

echo "============================================"
echo "🌐 Starting Zentrix GPU Server..."
echo "============================================"
cd "$APP_DIR"
python server.py
