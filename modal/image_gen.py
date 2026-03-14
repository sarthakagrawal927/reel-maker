"""
Unified Modal endpoint: image generation (FLUX.1-schnell) + TTS (Kokoro-82M + Whisper).

Setup (one-time):
  1. Accept the FLUX.1-schnell license at:
       https://huggingface.co/black-forest-labs/FLUX.1-schnell
  2. Create a HuggingFace read token at:
       https://huggingface.co/settings/tokens
  3. Create a Modal secret:
       modal secret create huggingface HF_TOKEN=hf_xxx
  4. Deploy:
       modal deploy modal/image_gen.py

This prints URLs like:
  https://<you>--reel-maker-serve.modal.run/generate
  https://<you>--reel-maker-serve.modal.run/tts

Add to .env:
  MODAL_IMAGE_GEN_URL=https://<you>--reel-maker-serve.modal.run/generate
  MODAL_TTS_URL=https://<you>--reel-maker-serve.modal.run/tts

Cost: ~$0.002/image (A10G), TTS is CPU-only (~$0.0001/request)
"""

import io
import base64
from typing import Optional
from pydantic import BaseModel
import modal

app = modal.App("reel-maker")

FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"
CACHE_DIR = "/cache"

hf_secret = modal.Secret.from_name("huggingface")
model_volume = modal.Volume.from_name("reel-maker-flux-cache", create_if_missing=True)
tts_volume = modal.Volume.from_name("reel-maker-tts-cache", create_if_missing=True)


# ─── Image generation image ───────────────────────────────────────────────────

def download_flux_model():
    """Download FLUX weights into volume at build time (runs once)."""
    import os
    from diffusers import FluxPipeline
    import torch

    FluxPipeline.from_pretrained(
        FLUX_MODEL,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        token=os.environ["HF_TOKEN"],
    )


flux_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.5.0",
        "diffusers>=0.30.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "sentencepiece",
        "Pillow",
        "pydantic",
        "fastapi[standard]",
    )
    .run_function(
        download_flux_model,
        secrets=[hf_secret],
        volumes={CACHE_DIR: model_volume},
    )
)


# ─── TTS image ────────────────────────────────────────────────────────────────

def download_tts_model():
    """Download Kokoro-82M + Whisper base weights at build time (runs once)."""
    import os
    os.environ["HF_HUB_CACHE"] = "/cache/hf"
    from huggingface_hub import snapshot_download
    snapshot_download("hexgrad/Kokoro-82M")
    # Pre-download Whisper base model so container cold start is fast
    from faster_whisper import WhisperModel
    WhisperModel("base", device="cpu", compute_type="int8",
                 download_root="/cache/whisper")


serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pydantic", "fastapi[standard]", "numpy")
)

tts_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "kokoro>=0.9.4",
        "soundfile",
        "numpy",
        "faster-whisper",
        "huggingface_hub",
        "scipy",
    )
    .run_function(
        download_tts_model,
        volumes={"/cache": tts_volume},
    )
)


# ─── Request/response schemas ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1792
    steps: int = 4
    seed: Optional[int] = None


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"  # Kokoro voice ID


# ─── Image generation class ───────────────────────────────────────────────────

@app.cls(
    image=flux_image,
    gpu="A10G",
    volumes={CACHE_DIR: model_volume},
    timeout=300,
)
class FluxGenerator:
    @modal.enter()
    def load_model(self):
        from diffusers import FluxPipeline
        import torch

        self.pipe = FluxPipeline.from_pretrained(
            FLUX_MODEL,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        self.pipe.enable_sequential_cpu_offload()  # ~2-3 GB VRAM peak

    @modal.method()
    def generate(self, prompt: str, width: int, height: int, steps: int, seed: Optional[int]) -> str:
        import torch

        generator = torch.Generator(device="cuda")
        if seed is not None:
            generator.manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        )

        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── TTS class ────────────────────────────────────────────────────────────────

@app.cls(
    image=tts_image,
    volumes={"/cache": tts_volume},
    timeout=300,
)
class TTSGenerator:
    @modal.enter()
    def load_model(self):
        import os
        os.environ["HF_HUB_CACHE"] = "/cache/hf"
        from kokoro import KPipeline
        self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

        from faster_whisper import WhisperModel
        self.whisper = WhisperModel("base", device="cpu", compute_type="int8",
                                    download_root="/cache/whisper")

    @modal.method()
    def synthesize(self, text: str, voice: str) -> dict:
        import numpy as np
        import soundfile as sf
        import io as _io
        import tempfile, os

        # Generate audio with Kokoro
        audio_chunks = []
        sample_rate = 24000
        for _, _, audio in self.pipeline(text, voice=voice):
            audio_chunks.append(audio)

        if not audio_chunks:
            raise ValueError("Kokoro produced no audio")

        audio_np = np.concatenate(audio_chunks)

        # Write to temp WAV for Whisper alignment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio_np, sample_rate)

        try:
            # Transcribe with word-level timestamps
            segments, _ = self.whisper.transcribe(
                tmp_path,
                word_timestamps=True,
                language="en",
            )

            characters = []
            char_starts = []
            char_ends = []

            words = list(text.split())
            word_idx = 0

            for seg in segments:
                for winfo in (seg.words or []):
                    word_text = winfo.word.strip()
                    if not word_text:
                        continue
                    word_start = winfo.start
                    word_end = winfo.end
                    char_dur = (word_end - word_start) / max(len(word_text), 1)

                    for i, ch in enumerate(word_text):
                        characters.append(ch)
                        char_starts.append(float(word_start + i * char_dur))
                        char_ends.append(float(word_start + (i + 1) * char_dur))

                    # Add space after word (except last)
                    word_idx += 1
                    if word_idx < len(words):
                        characters.append(" ")
                        char_starts.append(float(word_end))
                        char_ends.append(float(word_end))
        finally:
            os.unlink(tmp_path)

        # Encode audio as MP3-compatible base64 (WAV)
        buf = _io.BytesIO()
        sf.write(buf, audio_np, sample_rate, format="WAV")
        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "audio_b64": audio_b64,
            "characters": characters,
            "characterStartTimesSeconds": char_starts,
            "characterEndTimesSeconds": char_ends,
        }


# ─── Unified ASGI endpoint ────────────────────────────────────────────────────

@app.function(image=serve_image, timeout=600)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    web = FastAPI()
    flux = FluxGenerator()
    tts = TTSGenerator()

    @web.post("/generate")
    async def generate_image(req: GenerateRequest):
        try:
            image_b64 = await flux.generate.remote.aio(
                req.prompt, req.width, req.height, req.steps, req.seed
            )
            return JSONResponse({"image_b64": image_b64})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web.post("/tts")
    async def text_to_speech(req: TTSRequest):
        try:
            result = await tts.synthesize.remote.aio(req.text, req.voice)
            return JSONResponse(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web.get("/health")
    async def health():
        return {"status": "ok"}

    return web
