"""
LatentSync talking-head endpoint.
Clones ByteDance/LatentSync at image-build time, downloads checkpoint to volume.

Routes:
  POST /lipsync   { image_b64: str, audio_b64: str } → { video_b64: str }
  GET  /health

Setup:
  modal secret create huggingface HF_TOKEN=hf_xxx   # (already exists)
  modal deploy modal/latentsync.py

.env:
  MODAL_LIPSYNC_URL=https://<you>--latentsync-serve.modal.run/lipsync
"""

import base64
import os
import subprocess
import tempfile

from pydantic import BaseModel
import modal

app = modal.App("latentsync")

REPO_DIR  = "/latentsync"
CACHE_DIR = "/cache/latentsync"
CKPT_DIR  = f"{REPO_DIR}/checkpoints"

hf_secret    = modal.Secret.from_name("huggingface")
cache_volume = modal.Volume.from_name("latentsync-cache", create_if_missing=True)


# ─── Download checkpoint (runs once at image build time) ─────────────────────

def download_model():
    import os
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Main UNet checkpoint
    hf_hub_download(
        repo_id="ByteDance/LatentSync-1.5",
        filename="latentsync_unet.pt",
        local_dir=CKPT_DIR,
        token=token,
    )
    print("LatentSync checkpoint downloaded.")


# ─── Container image ──────────────────────────────────────────────────────────

latentsync_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        f"git clone https://github.com/bytedance/LatentSync.git {REPO_DIR}",
        f"pip install -r {REPO_DIR}/requirements.txt",
    )
    .pip_install("huggingface_hub")
    .run_function(
        download_model,
        secrets=[hf_secret],
        volumes={CACHE_DIR: cache_volume},
    )
)


# ─── Request schema ───────────────────────────────────────────────────────────

class LipSyncRequest(BaseModel):
    image_b64: str   # face portrait PNG/JPG from FLUX
    audio_b64: str   # WAV audio from TTS


# ─── Inference class ──────────────────────────────────────────────────────────

@app.cls(
    image=latentsync_image,
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume},
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

            # Write inputs
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(image_b64))
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_b64))

            # Convert static portrait → video looped to audio duration
            subprocess.run([
                "ffmpeg", "-y",
                "-loop", "1", "-i", img_path,
                "-i", audio_path,
                "-c:v", "libx264", "-tune", "stillimage",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest", video_path,
            ], check=True, capture_output=True)

            # Run LatentSync inference
            result = subprocess.run([
                "python", "-m", "scripts.inference",
                "--unet_config_path",    f"{REPO_DIR}/configs/unet/second_stage.yaml",
                "--inference_ckpt_path", f"{CKPT_DIR}/latentsync_unet.pt",
                "--guidance_scale",      "2.0",
                "--video_path",          video_path,
                "--audio_path",          audio_path,
                "--video_out_path",      out_path,
            ], capture_output=True, cwd=REPO_DIR)

            if result.returncode != 0:
                raise RuntimeError(
                    f"LatentSync failed:\n{result.stderr.decode()}"
                )

            with open(out_path, "rb") as f:
                return base64.b64encode(f.read()).decode()


# ─── ASGI endpoint ────────────────────────────────────────────────────────────

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pydantic", "fastapi[standard]")
)


@app.function(image=serve_image, timeout=300)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    web = FastAPI()
    gen = LatentSyncGenerator()

    @web.post("/lipsync")
    async def lipsync(req: LipSyncRequest):
        try:
            video_b64 = await gen.generate.remote.aio(req.image_b64, req.audio_b64)
            return JSONResponse({"video_b64": video_b64})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web.get("/health")
    async def health():
        return {"status": "ok"}

    return web
