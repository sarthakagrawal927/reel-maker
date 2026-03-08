"""
Modal image generation endpoint using FLUX.1-schnell.

Setup (one-time):
  pip install modal
  modal setup           # authenticate
  modal deploy modal/image_gen.py

This prints a URL like:
  https://<you>--reel-maker-fluxgenerator-generate.modal.run

Add it to .env:
  MODAL_IMAGE_GEN_URL=https://...

Cost: ~$0.002/image on A10G vs $0.03/image on fal.ai
Model: FLUX.1-schnell (Apache 2.0, no API key needed)
"""

import io
import base64
import modal

app = modal.App("reel-maker")

FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"
CACHE_DIR = "/cache"

container_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.4.0",
        "diffusers>=0.31.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "sentencepiece",
        "Pillow",
    )
)

model_volume = modal.Volume.from_name("reel-maker-flux-cache", create_if_missing=True)


@app.cls(
    image=container_image,
    gpu="A10G",
    volumes={CACHE_DIR: model_volume},
    timeout=300,
)
class FluxGenerator:
    @modal.build()
    def download_model(self):
        """Download model weights into the volume at build time (runs once)."""
        from diffusers import FluxPipeline
        import torch

        FluxPipeline.from_pretrained(
            FLUX_MODEL,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        model_volume.commit()

    @modal.enter()
    def load_model(self):
        """Load model into GPU memory when container starts."""
        from diffusers import FluxPipeline
        import torch

        self.pipe = FluxPipeline.from_pretrained(
            FLUX_MODEL,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        self.pipe.to("cuda")

    @modal.web_endpoint(method="POST", docs=True)
    def generate(self, payload: dict) -> dict:
        """
        Generate an image from a text prompt.

        Request body:
          prompt  (str)  Image description
          width   (int)  Width in pixels  (default: 1024)
          height  (int)  Height in pixels (default: 1792)
          steps   (int)  Inference steps  (default: 4)
          seed    (int)  Optional random seed for reproducibility

        Returns:
          image_b64 (str) Base64-encoded PNG
        """
        import torch

        prompt = payload["prompt"]
        width = int(payload.get("width", 1024))
        height = int(payload.get("height", 1792))
        steps = int(payload.get("steps", 4))
        seed = payload.get("seed")

        generator = torch.Generator(device="cuda")
        if seed is not None:
            generator.manual_seed(int(seed))

        result = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0,  # schnell doesn't use classifier-free guidance
            generator=generator,
        )

        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return {"image_b64": base64.b64encode(buf.getvalue()).decode("utf-8")}
