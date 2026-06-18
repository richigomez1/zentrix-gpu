FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# ─── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# ─── Python dependencies (ALL pre-installed, nothing to install at runtime) ──
RUN pip install --no-cache-dir \
    "peft>=0.17.0" \
    "diffusers>=0.34.0" \
    "transformers>=4.48.0" \
    "accelerate>=0.30.0" \
    "av>=14.0.0" \
    "Pillow>=10.0.0" \
    "imageio[ffmpeg]>=2.30.0" \
    "scipy>=1.10.0" \
    "numpy>=1.24.0" \
    "sentencepiece>=0.1.99" \
    "protobuf>=3.20.0" \
    "soundfile>=0.12.0" \
    "huggingface-hub>=0.25.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.29.0"

# ─── App code ────────────────────────────────────────────────────────────────
WORKDIR /app
COPY handler.py server.py ./

# ─── Model cache → Network Volume (persists between restarts) ────────────────
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV HF_HUB_CACHE=/runpod-volume/huggingface/hub

EXPOSE 8000

CMD ["python", "server.py"]
