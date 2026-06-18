"""
Zentrix Multi-Model Handler v5 (RunPod + Network Volume)
Models: LTX-2 (video+audio), Wan2.2-T2V-A14B (video), FLUX.2-dev (images)
All model weights cached on /runpod-volume — downloaded once, persist forever.
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
    """Loads/unloads models one at a time to fit in A100 80GB VRAM."""

    def __init__(self):
        self.current_model = None
        self.pipe = None
        self.extra = {}
        self.hf_token = os.environ.get("HF_TOKEN", None)
        cache = os.environ.get("HF_HOME", "/runpod-volume/huggingface")
        print(f"🎬 ModelManager v5 | cache={cache} | token={'✅' if self.hf_token else '❌'}")

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

    def load(self, model_name: str):
        if self.current_model == model_name:
            return self.pipe

        self.unload()
        device = "cuda:0"
        print(f"  ⏳ Loading {model_name}...")

        if model_name == "ltx-2":
            from diffusers.pipelines.ltx2 import LTX2Pipeline
            self.pipe = LTX2Pipeline.from_pretrained(
                "Lightricks/LTX-2",
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
            )
            self.pipe.enable_sequential_cpu_offload(device=device)
            if hasattr(self.pipe, 'vocoder') and hasattr(self.pipe.vocoder, 'config'):
                self.extra["audio_sample_rate"] = getattr(
                    self.pipe.vocoder.config, "output_sample_rate",
                    getattr(self.pipe.vocoder.config, "sampling_rate", 24000)
                )
            else:
                self.extra["audio_sample_rate"] = 24000

        elif model_name == "wan2.2-t2v":
            from diffusers import WanPipeline
            # Try Diffusers-format repo first, fall back to base repo
            for repo in ["Wan-AI/Wan2.2-T2V-A14B-Diffusers", "Wan-AI/Wan2.2-T2V-A14B"]:
                try:
                    self.pipe = WanPipeline.from_pretrained(
                        repo, torch_dtype=torch.float16, token=self.hf_token,
                    )
                    print(f"  ✅ Loaded from {repo}")
                    break
                except Exception as e:
                    print(f"  ⚠️ {repo} failed: {e}")
                    continue
            if self.pipe is None:
                raise RuntimeError("Could not load Wan2.2-T2V from any known repo")
            self.pipe.enable_model_cpu_offload()

        elif model_name == "flux2":
            from diffusers import FluxPipeline
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
            )
            self.pipe.enable_model_cpu_offload()

        else:
            raise ValueError(f"Unknown model: {model_name}. Options: ltx-2, wan2.2-t2v, flux2")

        self.current_model = model_name
        print(f"  ✅ {model_name} ready")
        return self.pipe


manager = ModelManager()


class EndpointHandler:
    def __init__(self, path=""):
        print("✅ Zentrix Multi-Model Endpoint v5 ready")
        print("   Models: ltx-2, wan2.2-t2v, flux2")

    def __call__(self, data: Dict[str, Any]) -> Any:
        try:
            inputs = data.get("inputs", data)
            params = data.get("parameters", {})
            model_name = inputs.get("model", "ltx-2")
            prompt = inputs.get("prompt", "")
            print(f"📥 Request: model={model_name}, prompt={prompt[:80]}...")

            if model_name == "ltx-2":
                return self._generate_ltx2(inputs, params)
            elif model_name == "wan2.2-t2v":
                return self._generate_wan_t2v(inputs, params)
            elif model_name == "flux2":
                return self._generate_image(inputs, params)
            else:
                return {"error": f"Unknown model: {model_name}. Options: ltx-2, wan2.2-t2v, flux2"}
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
        w = (w // 32) * 32
        h = (h // 32) * 32
        if w != image.size[0] or h != image.size[1]:
            image = image.resize((w, h), Image.LANCZOS)
        return image

    # ── LTX-2: video + audio ─────────────────────────────────────────────────

    def _generate_ltx2(self, inputs, params):
        from diffusers.pipelines.ltx2.export_utils import encode_video

        pipe = manager.load("ltx-2")
        prompt = inputs.get("prompt", "A beautiful cinematic scene")
        negative_prompt = params.get("negative_prompt", "shaky, glitchy, low quality, worst quality")
        image = self._decode_image(inputs)

        width = params.get("width", 768)
        height = params.get("height", 512)
        num_frames = params.get("num_frames", 25)
        frame_rate = params.get("frame_rate", 24.0)
        steps = params.get("num_inference_steps", 40)
        guidance_scale = params.get("guidance_scale", 4.0)

        width = (width // 32) * 32
        height = (height // 32) * 32
        num_frames = ((num_frames - 1) // 8) * 8 + 1

        print(f"  🎬 LTX-2: {width}x{height}, {num_frames}f, {steps} steps")

        kwargs = {
            "prompt": prompt, "negative_prompt": negative_prompt,
            "width": width, "height": height, "num_frames": num_frames,
            "frame_rate": frame_rate, "num_inference_steps": steps,
            "guidance_scale": guidance_scale, "output_type": "np", "return_dict": False,
        }
        if image is not None:
            kwargs["image"] = image

        result = pipe(**kwargs)

        if isinstance(result, tuple) and len(result) == 2:
            video_np, audio_tensor = result
        else:
            video_np = result
            audio_tensor = None

        audio_sr = manager.extra.get("audio_sample_rate", 24000)
        manager.unload()  # Free VRAM before export

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        export_kwargs = {"fps": frame_rate, "output_path": tmp_path}
        has_audio = False
        if audio_tensor is not None:
            try:
                audio_cpu = audio_tensor[0].float().cpu() if hasattr(audio_tensor, '__getitem__') else audio_tensor.float().cpu()
                export_kwargs["audio"] = audio_cpu
                export_kwargs["audio_sample_rate"] = audio_sr
                has_audio = True
            except Exception as e:
                print(f"  ⚠️ Audio export failed: {e}")

        video_data = video_np[0] if isinstance(video_np, (list, np.ndarray)) and len(video_np) > 0 else video_np
        encode_video(video_data, **export_kwargs)

        del video_np, audio_tensor
        gc.collect()

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        os.unlink(tmp_path)

        size_mb = len(video_bytes) / (1024 * 1024)
        print(f"  📦 Video: {size_mb:.1f} MB — SUCCESS")

        return {
            "type": "video", "data": base64.b64encode(video_bytes).decode("utf-8"),
            "content_type": "video/mp4", "model": "ltx-2", "has_audio": has_audio,
            "num_frames": num_frames, "width": width, "height": height, "size_mb": round(size_mb, 2),
        }

    # ── Wan 2.2 Text-to-Video ────────────────────────────────────────────────

    def _generate_wan_t2v(self, inputs, params):
        from diffusers.utils import export_to_video

        pipe = manager.load("wan2.2-t2v")
        prompt = inputs.get("prompt", "Slow cinematic camera movement")

        width = params.get("width", 1280)
        height = params.get("height", 720)
        num_frames = params.get("num_frames", 49)
        guidance_scale = params.get("guidance_scale", 5.0)
        steps = params.get("num_inference_steps", 30)

        width = (width // 16) * 16
        height = (height // 16) * 16

        print(f"  🎬 Wan2.2-T2V: {width}x{height}, {num_frames}f, {steps} steps")

        output = pipe(
            prompt=prompt, num_frames=num_frames, guidance_scale=guidance_scale,
            num_inference_steps=steps, width=width, height=height,
        )
        frames = list(output.frames[0])
        del output
        manager.unload()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        export_to_video(frames, tmp_path, fps=16)
        del frames
        gc.collect()

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        os.unlink(tmp_path)

        size_mb = len(video_bytes) / (1024 * 1024)
        print(f"  📦 Video: {size_mb:.1f} MB — SUCCESS")

        return {
            "type": "video", "data": base64.b64encode(video_bytes).decode("utf-8"),
            "content_type": "video/mp4", "model": "wan2.2-t2v",
            "num_frames": num_frames, "width": width, "height": height, "size_mb": round(size_mb, 2),
        }

    # ── FLUX.2-dev images ────────────────────────────────────────────────────

    def _generate_image(self, inputs, params):
        pipe = manager.load("flux2")
        prompt = inputs.get("prompt", "")

        width = params.get("width", 1024)
        height = params.get("height", 1024)
        steps = params.get("num_inference_steps", 28)
        guidance_scale = params.get("guidance_scale", 3.5)

        print(f"  🖼️ FLUX.2: {width}x{height}, {steps} steps")

        result = pipe(
            prompt=prompt, guidance_scale=guidance_scale,
            num_inference_steps=steps, width=width, height=height,
        )
        image = result.images[0]
        del result

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        size_mb = len(img_bytes) / (1024 * 1024)
        print(f"  📦 Image: {size_mb:.1f} MB — SUCCESS")

        return {
            "type": "image", "data": base64.b64encode(img_bytes).decode("utf-8"),
            "content_type": "image/png", "model": "flux2",
            "width": width, "height": height,
        }
