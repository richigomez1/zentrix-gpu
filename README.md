# Zentrix GPU — RunPod Multi-Model Endpoint

Docker image for running LTX-2, Wan2.2-T2V, and FLUX.2-dev on RunPod.

## What's Inside
- **LTX-2** (19B) — Video + Audio generation
- **Wan2.2-T2V-A14B** — Text-to-Video 
- **FLUX.2-dev** — Image generation
- All dependencies pre-installed (zero pip install at runtime)
- Models cached on Network Volume (download once, persist forever)

## Docker Image
Auto-built on push: `ghcr.io/richigomez1/zentrix-gpu:latest`

## RunPod Setup
1. Create Network Volume (50GB minimum)
2. Create Pod: GPU A100 80GB, Docker image above, mount volume at `/runpod-volume`
3. Set env vars: `HF_TOKEN` (your HuggingFace token)
4. Expose port 8000
5. Server starts automatically

## API
```
POST /
{
  "inputs": {"model": "ltx-2", "prompt": "..."},
  "parameters": {"width": 768, "height": 512, ...}
}
```
