"""
Zentrix GPU Server v1.0 — RunPod Edition
Starts instantly because all deps are pre-installed in Docker image.
Models cached on /runpod-volume (Network Volume).
"""
import os
import time
import traceback
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ─── Startup info ────────────────────────────────────────────────────────────
print("=" * 60)
print("🚀 Zentrix GPU Server v1.0 (RunPod)")
print("=" * 60)
for pkg in ["torch", "diffusers", "transformers", "peft", "accelerate", "av"]:
    try:
        mod = __import__(pkg)
        print(f"  {pkg}=={getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  {pkg} ❌ NOT INSTALLED")

if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠️ CUDA not available — CPU only")

hf_home = os.environ.get("HF_HOME", "default")
print(f"  HF_HOME: {hf_home}")
print(f"  HF_TOKEN: {'✅' if os.environ.get('HF_TOKEN') else '❌'}")
print("=" * 60)

# ─── Handler ─────────────────────────────────────────────────────────────────
from handler import EndpointHandler

handler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global handler
    print("📦 Initializing EndpointHandler...")
    handler = EndpointHandler()
    print("✅ Server ready to receive requests")
    yield
    print("👋 Shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
@app.get("/")
def health():
    return {"status": "ok", "gpu": torch.cuda.is_available()}


@app.post("/")
async def predict(request: Request):
    if handler is None:
        return JSONResponse(status_code=503, content={"error": "Not ready yet"})
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    t0 = time.time()
    try:
        result = handler(data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

    elapsed = time.time() - t0
    rtype = result.get("type", "?") if isinstance(result, dict) else "?"
    print(f"⏱️ Done in {elapsed:.1f}s — type={rtype}")
    return JSONResponse(content=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🌐 Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=600)
