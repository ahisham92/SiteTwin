"""SiteTwin Real: our own AI pass that turns a rendered film into real-looking footage.

It runs open models on our own Modal GPUs (no per-second fees to any video service):

1. **Depth**: Depth Anything V2 reads the depth of every frame of the rendered film, so the
   AI keeps exactly our geometry: the building, crane, plant and streets stay where the 3D put
   them, and the construction sequence stays ours.
2. **Repaint**: Wan2.1 VACE (Apache 2.0) generates real-looking video that follows that depth,
   in runs of 81 frames, guided by a description of a real construction site.

Try it on a film:

    modal run modal_real.py --src film.mp4 --out real.mp4 --seconds 5

Everything here is downloaded once into the ``sitetwin-models`` volume.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

app = modal.App("sitetwin-real")
models = modal.Volume.from_name("sitetwin-models", create_if_missing=True)
GPU = os.environ.get("SITETWIN_REAL_GPU", "H100")
MODEL = os.environ.get("SITETWIN_REAL_MODEL", "Wan-AI/Wan2.1-VACE-14B-diffusers")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("torch==2.7.1", "diffusers==0.35.1", "transformers==4.53.3", "accelerate==1.9.0",
                 "ftfy", "sentencepiece", "imageio[ffmpeg]", "numpy<2.3", "pillow")
    .env({"HF_HOME": "/models/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

PROMPT = ("Real aerial drone footage of a construction site in a Mediterranean city: a reinforced "
          "concrete frame building under construction with columns, slabs and formwork, a yellow "
          "tower crane, workers in orange hi-vis and white helmets, concrete mixer trucks, site "
          "hoarding, surrounded by real apartment buildings, streets, parked cars and trees. "
          "Natural daylight, sharp detail, photorealistic, shot on a DJI drone, 4K documentary.")
NEGATIVE = ("cartoon, CG, 3D render, video game, illustration, painting, plastic, toy, low poly, "
            "blurry, smeared, distorted, warped buildings, flicker, text, watermark, subtitles, "
            "overexposed, oversaturated, worst quality, low quality, JPEG artifacts")


def _frames(path: Path, fps: int, width: int, height: int, seconds: float):
    """The film's frames at ``fps``, cropped and scaled to width x height (PIL images)."""
    import subprocess

    from PIL import Image
    out = path.parent / "frames"
    out.mkdir(exist_ok=True)
    vf = f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-t", str(seconds),
                    "-vf", vf, str(out / "f%05d.png")], check=True)
    return [Image.open(p).convert("RGB") for p in sorted(out.glob("f*.png"))]


@app.function(image=image, gpu=GPU, volumes={"/models": models}, timeout=3 * 3600)
def realify(film: bytes, seconds: float = 5.0, prompt: str = PROMPT, steps: int = 30,
            width: int = 832, height: int = 480, scale: float = 1.0, seed: int = 7) -> dict:
    import tempfile
    import time

    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    from PIL import Image
    from transformers import pipeline

    t0 = time.time()
    work = Path(tempfile.mkdtemp())
    src = work / "film.mp4"
    src.write_bytes(film)
    fps = 16
    frames = _frames(src, fps, width, height, seconds)
    n = len(frames)
    runs = max(1, (n - 1) // 80)
    frames = frames[:runs * 80 + 1]

    depth_model = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",
                           device="cuda")
    depth = []
    for f in frames:
        d = np.array(depth_model(f)["depth"], dtype=np.float32)
        d = (d - d.min()) / max(1e-6, d.max() - d.min()) * 255
        depth.append(Image.fromarray(d.astype(np.uint8)).convert("RGB").resize((width, height)))
    del depth_model
    torch.cuda.empty_cache()
    t_depth = time.time() - t0

    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.to("cuda")
    models.commit()

    out_frames = []
    for r in range(runs):
        part = depth[r * 80:r * 80 + 81]
        got = pipe(video=part, prompt=prompt, negative_prompt=NEGATIVE, height=height, width=width,
                   num_frames=len(part), num_inference_steps=steps, guidance_scale=5.0,
                   conditioning_scale=scale, generator=torch.Generator().manual_seed(seed)).frames[0]
        out_frames.extend(got if r == 0 else got[1:])
    real = work / "real.mp4"
    export_to_video(out_frames, str(real), fps=fps)
    # the same shot side by side: our render on the left, the AI's on the right
    side = work / "side.mp4"
    import subprocess
    plain = work / "plain.mp4"
    export_to_video([np.asarray(f, dtype=np.float32) / 255 for f in frames[:len(out_frames)]], str(plain), fps=fps)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(plain), "-i", str(real),
                    "-filter_complex", "hstack", "-c:v", "libx264", "-crf", "18", str(side)], check=True)
    return {"real": real.read_bytes(), "side": side.read_bytes(), "frames": len(out_frames),
            "seconds": time.time() - t0, "depth_seconds": t_depth, "gpu": GPU, "model": MODEL}


@app.local_entrypoint()
def main(src: str = "film.mp4", out: str = "real.mp4", seconds: float = 5.0, steps: int = 30,
         scale: float = 1.0):
    got = realify.remote(Path(src).read_bytes(), seconds=seconds, steps=steps, scale=scale)
    Path(out).write_bytes(got.pop("real"))
    Path(out).with_name(Path(out).stem + "-side.mp4").write_bytes(got.pop("side"))
    print(f"Written {out}: {got}")
