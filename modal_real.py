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

PROMPT = ("Real aerial drone footage, slowly orbiting a construction site in the middle of Beirut: "
          "a reinforced concrete frame building under construction with grey concrete columns, "
          "slabs and timber formwork, a yellow tower crane, workers in orange hi-vis vests and white "
          "helmets, concrete mixer trucks, the site ground dark brown earth and grey gravel with tyre tracks, low dark "
          "green metal hoarding panels along the plot edge. Around it a dense, dry "
          "Mediterranean city: beige and white apartment blocks with balconies, asphalt streets and "
          "car parks full of white, silver, grey and black cars, a few palm and pine trees, dusty bare lots, a busy highway. Clear sunny daylight, soft shadows, sharp "
          "detail, natural muted colours, photorealistic, shot on a DJI Mavic drone, 4K documentary footage.")
NEGATIVE = ("cartoon, CG, 3D render, video game, illustration, painting, plastic, toy, low poly, "
            "blurry, smeared, distorted, warped buildings, flicker, text, watermark, subtitles, "
            "overexposed, oversaturated, worst quality, low quality, JPEG artifacts, mosaic, "
            "tiles pattern, cracked texture, black ground, abstract pattern, neon colours, white sand, "
            "snow, bleached ground, desert dunes, lush green fields, farmland, grass lawns, rows of blue cars, "
            "garbled text, captions")


def regrade(ai, src, strength: float = 0.7, radius: float = 24.0):
    """Our render's real colours (Google's photos, our materials) laid back under the AI's
    detail: each pixel of the AI frame is scaled by how far its blurred colour is from the
    render's blurred colour there. The AI keeps its texture and light; the brown site and the
    dry city keep their own colours instead of the AI's bright sand and green fields."""
    import numpy as np
    from PIL import Image, ImageFilter

    a = np.asarray(ai, np.float32)
    a = a / 255.0 if a.max() > 1.5 else a
    s = np.asarray(src.resize((a.shape[1], a.shape[0])), np.float32) / 255.0

    def soft(x):
        img = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
        return (np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255.0) ** 2.2

    ratio = np.clip((soft(s) + 1e-3) / (soft(a) + 1e-3), 0.08, 3.0) ** strength
    out = np.clip((a ** 2.2) * ratio, 0, 1) ** (1 / 2.2)
    return out.astype(np.float32)


def _frames(path: Path, fps: int, width: int, height: int, seconds: float):
    """The film's frames at ``fps``, cropped and scaled to width x height (PIL images)."""
    import subprocess

    from PIL import Image
    out = path.parent / f"frames_{width}x{height}"
    out.mkdir(exist_ok=True)
    vf = f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-t", str(seconds),
                    "-vf", vf, str(out / "f%05d.png")], check=True)
    return [Image.open(p).convert("RGB") for p in sorted(out.glob("f*.png"))]


BAND = 132 / 1080      # the caption band at the foot of our frames (render.caption), share of height


def _finish(src: Path, ai_frames: list, fps: int, width: int, height: int, seconds: float,
            out: Path) -> None:
    """The AI's frames brought up to the film's own height, with our caption band (step, week,
    progress bar, credits) laid back on, sharp, from the original film."""
    import subprocess

    import numpy as np
    from PIL import Image
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=height", "-of", "csv=p=0", str(src)], capture_output=True, text=True)
    H = min(1080, int(probe.stdout.strip() or height))
    H -= H % 2
    W = int(round(H * width / height / 2)) * 2
    hi = _frames(src, fps, W, H, seconds)
    band = int(round(H * BAND)) + 2
    out_dir = src.parent / "final"
    out_dir.mkdir(exist_ok=True)
    for i, f in enumerate(ai_frames):
        a = np.asarray(f)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(a).resize((W, H), Image.LANCZOS)
        if i < len(hi):
            im.paste(hi[i].crop((0, H - band, W, H)), (0, H - band))
        im.save(out_dir / f"f{i + 1:05d}.png")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i",
                    str(out_dir / "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(out)], check=True)


@app.function(image=image, gpu=GPU, volumes={"/models": models}, timeout=3 * 3600)
def realify(film: bytes, seconds: float = 5.0, prompt: str = PROMPT, steps: int = 30,
            width: int = 832, height: int = 480, scale: float = 1.0, seed: int = 7,
            control: str = "depth", near: float = 0.6, grade: float = 0.7) -> dict:
    """``control``: "depth" guides by depth alone; "mix" guides the near part (the nearest
    ``near`` share of each frame) by depth and the rest by our render in grey, so the AI keeps
    the real streets and blocks round the site (it colours them) instead of inventing new ones."""
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
    # Each run makes 81 frames; its last few wash out, so only the first 73 are kept. The next run
    # starts 9 frames back and is handed those 9 finished frames as they are (masked out), so it
    # carries straight on from them: no jump in the look where two runs meet.
    keep, overlap = 73, 9
    stride = keep - overlap
    runs = max(1, 1 + (n - 81) // stride) if n >= 81 else 1
    frames = frames[:stride * (runs - 1) + 81]

    depth_model = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",
                           device="cuda")
    depth = []
    for f in frames:
        d = np.array(depth_model(f)["depth"], dtype=np.float32)
        # spread the values evenly (histogram equalisation), so the far streets and buildings,
        # which a plain scale leaves near black, still give the AI their shapes
        ranks = np.argsort(np.argsort(d.ravel()))
        d = (ranks / max(1, ranks.size - 1) * 255).reshape(d.shape)
        if control == "mix":
            grey = np.asarray(f.convert("L").resize((d.shape[1], d.shape[0])), dtype=np.float32)
            far = d < (1.0 - near) * 255                  # depth-anything: bigger is nearer
            d = np.where(far, grey, d)
        depth.append(Image.fromarray(d.astype(np.uint8)).convert("RGB").resize((width, height)))
    del depth_model
    torch.cuda.empty_cache()
    t_depth = time.time() - t0

    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.to("cuda")
    models.commit()

    def to_pil(a):
        a = np.asarray(a)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(a)

    white = Image.new("L", (width, height), 255)
    black = Image.new("L", (width, height), 0)
    out_frames = []
    for r in range(runs):
        part = list(depth[r * stride:r * stride + 81])
        mask = [white] * len(part)
        if r:
            held = [to_pil(f) for f in out_frames[-overlap:]]
            part[:overlap] = held
            mask[:overlap] = [black] * overlap
        got = pipe(video=part, mask=mask, prompt=prompt, negative_prompt=NEGATIVE, height=height,
                   width=width, num_frames=len(part), num_inference_steps=steps, guidance_scale=5.0,
                   conditioning_scale=scale, generator=torch.Generator().manual_seed(seed)).frames[0]
        got = list(got)[:keep]
        out_frames.extend(got if r == 0 else got[overlap:])
    if grade > 0:
        out_frames = [regrade(f, frames[i], grade) for i, f in enumerate(out_frames)]
    import subprocess
    low = work / "low.mp4"
    export_to_video(out_frames, str(low), fps=fps)
    real = work / "real.mp4"
    _finish(src, out_frames, fps, width, height, seconds, real)
    # the same shot side by side: our render on the left, the AI's on the right
    side = work / "side.mp4"
    plain = work / "plain.mp4"
    export_to_video([np.asarray(f, dtype=np.float32) / 255 for f in frames[:len(out_frames)]], str(plain), fps=fps)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(plain), "-i", str(low),
                    "-filter_complex", "hstack", "-c:v", "libx264", "-crf", "18", str(side)], check=True)
    return {"real": real.read_bytes(), "side": side.read_bytes(), "frames": len(out_frames),
            "seconds": time.time() - t0, "depth_seconds": t_depth, "gpu": GPU, "model": MODEL,
            "control": control}


@app.local_entrypoint()
def main(src: str = "film.mp4", out: str = "real.mp4", seconds: float = 5.0, steps: int = 30,
         scale: float = 1.3, control: str = "mix", grade: float = 0.7):
    got = realify.remote(Path(src).read_bytes(), seconds=seconds, steps=steps, scale=scale,
                         control=control, grade=grade)
    Path(out).write_bytes(got.pop("real"))
    Path(out).with_name(Path(out).stem + "-side.mp4").write_bytes(got.pop("side"))
    print(f"Written {out}: {got}")
