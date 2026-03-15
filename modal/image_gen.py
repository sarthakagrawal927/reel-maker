"""
Unified Modal endpoint: image gen (FLUX schnell/dev) + video gen (Wan T2V/I2V) + TTS + LatentSync.

Routes:
  POST /generate              image_b64          model="schnell"|"dev", steps, seed
  POST /generate-video-t2v    video_b64          prompt-only
  POST /generate-video-i2v    video_b64          image_b64 + prompt
  POST /tts                   audio+timestamps
  POST /lipsync               video_b64          image_b64 + audio_b64
  GET  /health

Setup:
  modal secret create huggingface HF_TOKEN=hf_xxx
  modal deploy modal/image_gen.py

.env:
  MODAL_IMAGE_GEN_URL=https://<you>--reel-maker-serve.modal.run/generate
  MODAL_TTS_URL=https://<you>--reel-maker-serve.modal.run/tts
  MODAL_T2V_URL=https://<you>--reel-maker-serve.modal.run/generate-video-t2v
  MODAL_I2V_URL=https://<you>--reel-maker-serve.modal.run/generate-video-i2v
  MODAL_LIPSYNC_URL=https://<you>--reel-maker-serve.modal.run/lipsync
"""

import io
import os
import base64
import subprocess
import tempfile
from typing import Optional
from pydantic import BaseModel
import modal

app = modal.App("reel-maker")

# ─── Model IDs ────────────────────────────────────────────────────────────────
FLUX_SCHNELL   = "black-forest-labs/FLUX.1-schnell"
FLUX_DEV       = "black-forest-labs/FLUX.1-dev"
WAN_T2V_MODEL  = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_I2V_MODEL  = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

FLUX_CACHE     = "/cache/flux"
WAN_CACHE      = "/cache/wan"

# ─── Secrets / Volumes ────────────────────────────────────────────────────────
hf_secret    = modal.Secret.from_name("huggingface")
flux_volume  = modal.Volume.from_name("reel-maker-flux-cache",   create_if_missing=True)
wan_volume   = modal.Volume.from_name("reel-maker-wan-cache",    create_if_missing=True)
tts_volume   = modal.Volume.from_name("reel-maker-tts-cache",    create_if_missing=True)
lsync_volume = modal.Volume.from_name("reel-maker-lsync-cache",  create_if_missing=True)

LSYNC_REPO_DIR  = "/latentsync"
LSYNC_CACHE_DIR = "/cache/latentsync"
LSYNC_CKPT_DIR  = f"{LSYNC_REPO_DIR}/checkpoints"


# ─── Shared pip deps ──────────────────────────────────────────────────────────
DIFFUSERS_DEPS = [
    "torch>=2.5.0", "diffusers>=0.33.0", "transformers>=4.44.0",
    "accelerate>=0.33.0", "sentencepiece", "Pillow", "pydantic",
]


# ════════════════════════════════════════════════════════════════════════════════
# FLUX IMAGE GENERATION
# ════════════════════════════════════════════════════════════════════════════════

def download_flux_models():
    import os, torch
    from diffusers import FluxPipeline
    token = os.environ["HF_TOKEN"]
    for model_id in [FLUX_SCHNELL, FLUX_DEV]:
        print(f"Downloading {model_id}…")
        FluxPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            cache_dir=FLUX_CACHE, token=token,
        )


flux_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*DIFFUSERS_DEPS)
    .run_function(
        download_flux_models,
        secrets=[hf_secret],
        volumes={FLUX_CACHE: flux_volume},
    )
)


class GenerateRequest(BaseModel):
    prompt: str
    width:  int   = 1024
    height: int   = 1792
    steps:  int   = 4       # 4 for schnell, 28 for dev
    seed:   Optional[int] = None
    model:  str   = "schnell"   # "schnell" | "dev"


@app.cls(
    image=flux_image,
    gpu="A10G",
    volumes={FLUX_CACHE: flux_volume},
    timeout=600,
)
class FluxGenerator:
    @modal.enter()
    def load_models(self):
        from diffusers import FluxPipeline
        import torch
        self._pipes = {}
        for name, model_id in [("schnell", FLUX_SCHNELL), ("dev", FLUX_DEV)]:
            pipe = FluxPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, cache_dir=FLUX_CACHE,
            )
            pipe.enable_sequential_cpu_offload()
            self._pipes[name] = pipe

    @modal.method()
    def generate(
        self, prompt: str, width: int, height: int,
        steps: int, seed: Optional[int], model: str,
    ) -> str:
        import torch
        pipe = self._pipes.get(model, self._pipes["schnell"])
        guidance = 3.5 if model == "dev" else 0.0

        gen = torch.Generator(device="cuda")
        if seed is not None:
            gen.manual_seed(seed)

        result = pipe(
            prompt=prompt, width=width, height=height,
            num_inference_steps=steps, guidance_scale=guidance,
            generator=gen,
        )
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


# ════════════════════════════════════════════════════════════════════════════════
# WAN TEXT-TO-VIDEO
# ════════════════════════════════════════════════════════════════════════════════

def download_wan_t2v():
    import os, torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from transformers import AutoTokenizer, UMT5EncoderModel
    print(f"Downloading {WAN_T2V_MODEL}…")
    WanPipeline.from_pretrained(
        WAN_T2V_MODEL, torch_dtype=torch.bfloat16, cache_dir=WAN_CACHE,
    )


wan_t2v_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*DIFFUSERS_DEPS, "imageio", "imageio-ffmpeg")
    .run_function(
        download_wan_t2v,
        volumes={WAN_CACHE: wan_volume},
    )
)


class VideoRequest(BaseModel):
    prompt: str
    width:  int   = 480
    height: int   = 832    # portrait 480p
    num_frames: int = 49   # ~3 s at 16 fps (must satisfy 4k+1)
    guidance_scale: float = 5.0
    seed:   Optional[int] = None


@app.cls(
    image=wan_t2v_image,
    gpu="A10G",
    volumes={WAN_CACHE: wan_volume},
    timeout=600,
)
class WanT2VGenerator:
    @modal.enter()
    def load_model(self):
        from diffusers import WanPipeline
        import torch
        self.pipe = WanPipeline.from_pretrained(
            WAN_T2V_MODEL, torch_dtype=torch.bfloat16, cache_dir=WAN_CACHE,
        )
        self.pipe.enable_model_cpu_offload()

    @modal.method()
    def generate(
        self, prompt: str, width: int, height: int,
        num_frames: int, guidance_scale: float, seed: Optional[int],
    ) -> str:
        import torch, imageio, numpy as np, tempfile, os
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cuda").manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            width=width, height=height,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=gen,
        )
        frames = [np.array(f) for f in output.frames[0]]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            imageio.mimwrite(tmp_path, frames, fps=16, quality=8)
            with open(tmp_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        finally:
            os.unlink(tmp_path)


# ════════════════════════════════════════════════════════════════════════════════
# WAN IMAGE-TO-VIDEO
# ════════════════════════════════════════════════════════════════════════════════

def download_wan_i2v():
    import torch
    from diffusers import WanImageToVideoPipeline
    print(f"Downloading {WAN_I2V_MODEL}…")
    WanImageToVideoPipeline.from_pretrained(
        WAN_I2V_MODEL, torch_dtype=torch.bfloat16, cache_dir=WAN_CACHE,
    )


wan_i2v_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*DIFFUSERS_DEPS, "imageio", "imageio-ffmpeg")
    .run_function(
        download_wan_i2v,
        volumes={WAN_CACHE: wan_volume},
    )
)


class I2VRequest(VideoRequest):
    image_b64: str


@app.cls(
    image=wan_i2v_image,
    gpu="A100-40GB",   # 14B model; A10G works but is very slow
    volumes={WAN_CACHE: wan_volume},
    timeout=600,
)
class WanI2VGenerator:
    @modal.enter()
    def load_model(self):
        from diffusers import WanImageToVideoPipeline
        import torch
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_I2V_MODEL, torch_dtype=torch.bfloat16, cache_dir=WAN_CACHE,
        )
        self.pipe.enable_model_cpu_offload()

    @modal.method()
    def generate(
        self, image_b64: str, prompt: str, width: int, height: int,
        num_frames: int, guidance_scale: float, seed: Optional[int],
    ) -> str:
        import torch, imageio, numpy as np, tempfile, os
        from PIL import Image as PILImage

        img_bytes = base64.b64decode(image_b64)
        image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize((width, height))

        gen = None
        if seed is not None:
            gen = torch.Generator(device="cuda").manual_seed(seed)

        output = self.pipe(
            image=image,
            prompt=prompt,
            width=width, height=height,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=gen,
        )
        frames = [np.array(f) for f in output.frames[0]]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            imageio.mimwrite(tmp_path, frames, fps=16, quality=8)
            with open(tmp_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        finally:
            os.unlink(tmp_path)


# ════════════════════════════════════════════════════════════════════════════════
# TTS  (unchanged)
# ════════════════════════════════════════════════════════════════════════════════

def download_tts_model():
    import os
    os.environ["HF_HUB_CACHE"] = "/cache/hf"
    from huggingface_hub import snapshot_download
    snapshot_download("hexgrad/Kokoro-82M")
    from faster_whisper import WhisperModel
    WhisperModel("base", device="cpu", compute_type="int8",
                 download_root="/cache/tts/whisper")


tts_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("kokoro>=0.9.4", "soundfile", "numpy",
                 "faster-whisper", "huggingface_hub", "scipy")
    .run_function(download_tts_model, volumes={"/cache": tts_volume})
)


class TTSRequest(BaseModel):
    text:  str
    voice: str = "af_heart"


@app.cls(image=tts_image, volumes={"/cache": tts_volume}, timeout=300)
class TTSGenerator:
    @modal.enter()
    def load_model(self):
        import os
        os.environ["HF_HUB_CACHE"] = "/cache/hf"
        from kokoro import KPipeline
        self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel("base", device="cpu", compute_type="int8",
                                    download_root="/cache/tts/whisper")

    @modal.method()
    def synthesize(self, text: str, voice: str) -> dict:
        import numpy as np, soundfile as sf, io as _io, tempfile, os

        audio_chunks = []
        for _, _, audio in self.pipeline(text, voice=voice):
            audio_chunks.append(audio)
        if not audio_chunks:
            raise ValueError("Kokoro produced no audio")
        audio_np = np.concatenate(audio_chunks)
        sample_rate = 24000

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio_np, sample_rate)

        try:
            segments, _ = self.whisper.transcribe(
                tmp_path, word_timestamps=True, language="en",
            )
            characters, char_starts, char_ends = [], [], []
            words = list(text.split())
            word_idx = 0
            for seg in segments:
                for winfo in (seg.words or []):
                    word_text = winfo.word.strip()
                    if not word_text:
                        continue
                    ws, we = winfo.start, winfo.end
                    dur = (we - ws) / max(len(word_text), 1)
                    for i, ch in enumerate(word_text):
                        characters.append(ch)
                        char_starts.append(float(ws + i * dur))
                        char_ends.append(float(ws + (i + 1) * dur))
                    word_idx += 1
                    if word_idx < len(words):
                        characters.append(" ")
                        char_starts.append(float(we))
                        char_ends.append(float(we))
        finally:
            os.unlink(tmp_path)

        buf = _io.BytesIO()
        sf.write(buf, audio_np, sample_rate, format="WAV")
        return {
            "audio_b64": base64.b64encode(buf.getvalue()).decode(),
            "characters": characters,
            "characterStartTimesSeconds": char_starts,
            "characterEndTimesSeconds": char_ends,
        }


# ════════════════════════════════════════════════════════════════════════════════
# LATENTSYNC TALKING HEAD
# ════════════════════════════════════════════════════════════════════════════════

def download_latentsync():
    import os
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    os.makedirs(LSYNC_CKPT_DIR, exist_ok=True)
    hf_hub_download(
        repo_id="ByteDance/LatentSync-1.5",
        filename="latentsync_unet.pt",
        local_dir=LSYNC_CKPT_DIR,
        token=token,
    )
    print("LatentSync checkpoint downloaded.")


latentsync_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        f"git clone https://github.com/bytedance/LatentSync.git {LSYNC_REPO_DIR}",
        f"pip install -r {LSYNC_REPO_DIR}/requirements.txt",
    )
    .pip_install("huggingface_hub")
    .run_function(
        download_latentsync,
        secrets=[hf_secret],
        volumes={LSYNC_CACHE_DIR: lsync_volume},
    )
)


class LipSyncRequest(BaseModel):
    image_b64: str
    audio_b64: str


@app.cls(
    image=latentsync_image,
    gpu="A10G",
    volumes={LSYNC_CACHE_DIR: lsync_volume},
    timeout=300,
)
class LatentSyncGenerator:

    @modal.method()
    def generate(self, image_b64: str, audio_b64: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path   = os.path.join(tmpdir, "face.png")
            audio_path = os.path.join(tmpdir, "audio.wav")
            video_path = os.path.join(tmpdir, "face.mp4")
            out_path   = os.path.join(tmpdir, "out.mp4")

            with open(img_path, "wb") as f:
                f.write(base64.b64decode(image_b64))
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_b64))

            # Loop static portrait to audio duration
            subprocess.run([
                "ffmpeg", "-y",
                "-loop", "1", "-i", img_path,
                "-i", audio_path,
                "-c:v", "libx264", "-tune", "stillimage",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest", video_path,
            ], check=True, capture_output=True)

            result = subprocess.run([
                "python", "-m", "scripts.inference",
                "--unet_config_path",    f"{LSYNC_REPO_DIR}/configs/unet/second_stage.yaml",
                "--inference_ckpt_path", f"{LSYNC_CKPT_DIR}/latentsync_unet.pt",
                "--guidance_scale",      "2.0",
                "--video_path",          video_path,
                "--audio_path",          audio_path,
                "--video_out_path",      out_path,
            ], capture_output=True, cwd=LSYNC_REPO_DIR)

            if result.returncode != 0:
                raise RuntimeError(
                    f"LatentSync failed:\n{result.stderr.decode()}"
                )

            with open(out_path, "rb") as f:
                return base64.b64encode(f.read()).decode()


# ════════════════════════════════════════════════════════════════════════════════
# UNIFIED ASGI ENDPOINT
# ════════════════════════════════════════════════════════════════════════════════

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pydantic", "fastapi[standard]", "numpy")
)


@app.function(image=serve_image, timeout=600)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    web   = FastAPI()
    flux  = FluxGenerator()
    t2v   = WanT2VGenerator()
    i2v   = WanI2VGenerator()
    tts   = TTSGenerator()
    lsync = LatentSyncGenerator()

    @web.post("/generate")
    async def generate_image(req: GenerateRequest):
        try:
            image_b64 = await flux.generate.remote.aio(
                req.prompt, req.width, req.height, req.steps, req.seed, req.model,
            )
            return JSONResponse({"image_b64": image_b64})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web.post("/generate-video-t2v")
    async def generate_video_t2v(req: VideoRequest):
        try:
            video_b64 = await t2v.generate.remote.aio(
                req.prompt, req.width, req.height,
                req.num_frames, req.guidance_scale, req.seed,
            )
            return JSONResponse({"video_b64": video_b64})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web.post("/generate-video-i2v")
    async def generate_video_i2v(req: I2VRequest):
        try:
            video_b64 = await i2v.generate.remote.aio(
                req.image_b64, req.prompt, req.width, req.height,
                req.num_frames, req.guidance_scale, req.seed,
            )
            return JSONResponse({"video_b64": video_b64})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web.post("/tts")
    async def text_to_speech(req: TTSRequest):
        try:
            result = await tts.synthesize.remote.aio(req.text, req.voice)
            return JSONResponse(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web.post("/lipsync")
    async def lipsync(req: LipSyncRequest):
        try:
            video_b64 = await lsync.generate.remote.aio(req.image_b64, req.audio_b64)
            return JSONResponse({"video_b64": video_b64})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web.get("/health")
    async def health():
        return {"status": "ok"}

    return web
