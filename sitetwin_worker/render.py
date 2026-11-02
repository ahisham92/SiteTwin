"""From an IFC file to an MP4: the whole pipeline, and the command line for it.

    python -m sitetwin_worker render model.ifc --out film/ [--seconds 90] [--quality draft|final]

1. read the model (IfcOpenShell) and its location (the IFC's site, or --lat/--lon);
2. fetch the surroundings (terrain, imagery, OpenStreetMap) into film/environment;
3. plan the build timeline;
4. build the Blender scene and render every frame (Cycles, GPU when there is one);
5. caption each frame with the step being built and the data credits;
6. join the frames into film/<name>.mp4 with ffmpeg.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

QUALITY = {
    "preview": dict(width=640, height=360, samples=12, fps=24),
    "draft": dict(width=1280, height=720, samples=32, fps=30),
    "final": dict(width=1920, height=1080, samples=128, fps=30),
    "4k": dict(width=3840, height=2160, samples=192, fps=30),
}


def _font(size: int, bold: bool = False):
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                 "/usr/share/fonts/truetype/dejavu/" + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                 "Arial Bold.ttf" if bold else "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def caption(path: Path, frame: int, tl, title: str, credits: list[str], start: dt.date | None = None) -> None:
    """The step being built, a progress bar across the build, and the credits, burnt into the frame."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    u = H / 1080
    d = ImageDraw.Draw(im, "RGBA")
    step = None
    for st in tl.steps:
        if st.start <= frame < st.end:
            step = st
    fly_end = getattr(tl, "fly_end", 0)
    clear_start, clear_end = getattr(tl, "clear_start", 0), getattr(tl, "clear_end", 0)
    if frame < fly_end:
        line = "Flying in to the site"
    elif clear_end and frame < clear_start:
        line = "The site as it is today"
    elif clear_end and frame < clear_end:
        line = "Clearing the site"
    elif frame < tl.build_start:
        line = "Setting up: hoarding, cabins and the tower crane" if clear_end else "The site, cleared and set out"
    elif frame >= tl.build_end:
        line = "Complete"
    elif step is not None:
        line = f"Step {step.n} of {len(tl.steps)} · {step.storey} · {step.phase}"
    else:
        line = ""
    pad = int(36 * u)
    box_h = int(112 * u)
    d.rectangle([0, H - box_h - pad // 2, W, H], fill=(10, 16, 28, 150))
    d.text((pad, H - box_h), title, font=_font(int(30 * u), True), fill=(255, 255, 255, 255))
    # the job's own clock: weeks since the site was handed over, and the month it is
    from .sequence import calendar, day_at
    day = day_at(calendar(tl), frame)
    when = (start or dt.date.today()) + dt.timedelta(days=int(day))
    clock = f"Week {int(day // 7) + 1}  ·  {when.strftime('%B %Y')}"
    tw = d.textlength(title, font=_font(int(30 * u), True))
    d.text((pad + tw + int(24 * u), H - box_h + int(6 * u)), clock, font=_font(int(24 * u), True),
           fill=(245, 165, 36, 255))
    d.text((pad, H - box_h + int(42 * u)), line, font=_font(int(26 * u)), fill=(235, 238, 245, 255))
    span = max(1, tl.build_end - tl.build_start)
    prog = min(1.0, max(0.0, (frame - tl.build_start) / span))
    bx0, bx1, by = pad, W - pad, H - int(18 * u)
    d.rectangle([bx0, by - int(4 * u), bx1, by], fill=(255, 255, 255, 60))
    d.rectangle([bx0, by - int(4 * u), bx0 + (bx1 - bx0) * prog, by], fill=(245, 165, 36, 255))
    if credits:
        # right-aligned lines no wider than 60% of the frame, so nothing runs off the edge
        f = _font(int(15 * u))
        pieces = [p.strip() for c in credits for p in c.replace("; ", " · ").split(" · ") if p.strip()]
        lines, cur = [], ""
        for p in pieces:
            trial = f"{cur}  ·  {p}" if cur else p
            if cur and d.textlength(trial, font=f) > W * 0.6:
                lines.append(cur)
                cur = p
            else:
                cur = trial
        if cur:
            lines.append(cur)
        step_h = int(19 * u)
        lines = lines[: max(1, (box_h - int(30 * u)) // step_h)]
        for i, txt in enumerate(lines):
            tw = d.textlength(txt, font=f)
            d.text((W - pad - tw, H - box_h + int(4 * u) + i * step_h), txt, font=f, fill=(220, 225, 235, 200))
    im.save(path)


def run(ifc: Path, out: Path, lat: float | None = None, lon: float | None = None, radius: float = 350,
        seconds: float = 90, quality: str = "draft", title: str | None = None,
        when: dt.datetime | None = None, frames: list[int] | None = None, log=print,
        chunk: tuple[int, int] | None = None, prepare_only: bool = False,
        on_stage=None, on_frame=None, fly_in: bool = True) -> dict[str, Any]:
    """Render the film. ``chunk=(first, last)`` renders only those frames and encodes them as
    one segment (for splitting a film across several GPUs); ``prepare_only`` fetches the
    surroundings and plans the timeline, and stops there."""
    from . import environment, model, sequence
    from sitetwin_app import ifc as ifc_text

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    summary = ifc_text.summarise_file(ifc)
    lat = lat if lat is not None else summary.latitude
    lon = lon if lon is not None else summary.longitude
    if lat is None or lon is None:
        raise SystemExit("The model does not say where it is: pass --lat and --lon.")
    title = title or summary.project or ifc.stem
    q = dict(QUALITY[quality])

    stage = on_stage or (lambda name: None)
    stage("starting")
    log(f"Reading {ifc.name}…")
    elements = model.load(ifc)
    if not elements:
        raise SystemExit("No geometry could be read from the model.")
    log(f"  {len(elements)} elements")

    stage("surroundings")
    log(f"Fetching the surroundings of {lat:.5f}, {lon:.5f} (radius {radius:.0f} m)…")
    manifest = environment.gather(lat, lon, radius, out / "environment", intro=fly_in)
    for k, why in manifest.get("missing", {}).items():
        log(f"  no {k}: {why}")
    if manifest.get("google"):                     # Google's mesh replaces the terrain, imagery and OSM
        manifest = dict(manifest, credits=[c for c in manifest.get("credits", [])
                                           if not c.startswith(("Terrain", "Imagery", "Buildings"))])
    from . import assets, equipment
    cache = Path(os.environ.get("SITETWIN_ASSET_CACHE", str(out / "models")))
    models, model_credits = assets.prepare(cache, log=log)
    equipment.MODELS.clear()
    equipment.MODELS.update(models)
    from . import materials
    materials.TEXTURES.clear()
    materials.TEXTURES.update(assets.fetch_textures(cache, log=log))
    if materials.TEXTURES:
        model_credits.append("textures: Poly Haven (CC0)")
    if model_credits:
        manifest = dict(manifest, credits=list(manifest.get("credits", [])) + ["Models (Sketchfab): " + "; ".join(model_credits)])
        log(f"  real models: {', '.join(models)}")

    stage("scene")
    tl = sequence.plan(elements, seconds=seconds, fps=q["fps"], fly_in=fly_in)
    (out / "timeline.json").write_text(json.dumps(tl.as_dict(), indent=1))
    log(f"  {len(tl.steps)} steps, {tl.frames} frames")
    if prepare_only:
        return {"frames": tl.frames, "fps": q["fps"], "manifest": manifest, "title": title}

    from . import scene
    frames_dir = out / ("frames" if chunk is None else f"frames_{chunk[0]:05d}")
    if frames_dir.exists() and frames is None:
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(exist_ok=True)
    opts = scene.Options(width=q["width"], height=q["height"], samples=q["samples"], fps=q["fps"], when=when)
    info = scene.build(elements, tl, out / "environment", manifest, opts, frames_dir)
    log(f"  scene built: {info}")

    import bpy
    sc = bpy.context.scene
    if chunk is not None:
        todo = list(range(max(1, chunk[0]), min(tl.frames, chunk[1]) + 1))
    else:
        todo = frames or list(range(1, tl.frames + 1))
    t_render = time.time()
    stage("frames")
    for i, f in enumerate(todo):
        sc.frame_set(f)
        path = frames_dir / f"frame_{f:05d}.png"
        sc.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        caption(path, f, tl, title, manifest.get("credits", []), start=(when or dt.datetime.now()).date())
        if on_frame:
            on_frame(i + 1, len(todo), path)
        if i == 0 or (i + 1) % 25 == 0:
            per = (time.time() - t_render) / (i + 1)
            log(f"  frame {f} ({i + 1}/{len(todo)}), {per:.1f} s a frame, ~{per * (len(todo) - i - 1) / 60:.0f} min to go")

    result: dict[str, Any] = {"frames": len(todo), "seconds": time.time() - t0, "scene": info,
                              "manifest": manifest}
    stage("joining")
    if chunk is not None:
        segment = out / f"segment_{todo[0]:05d}.mp4"
        encode(frames_dir, q["fps"], segment, start=todo[0])
        shutil.rmtree(frames_dir, ignore_errors=True)
        result["segment"] = str(segment)
    elif frames is None:
        video = out / (ifc.stem + ".mp4")
        encode(frames_dir, q["fps"], video)
        result["video"] = str(video)
        log(f"Written {video}")
    return result


def encode(frames_dir: Path, fps: int, video: Path, start: int = 1) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-start_number", str(start),
                    "-i", str(frames_dir / "frame_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-preset", "slow", "-movflags", "+faststart", str(video)], check=True)


def join(segments: list[Path], video: Path) -> None:
    """Segments encoded alike are joined without re-encoding."""
    listing = video.with_suffix(".txt")
    listing.write_text("".join(f"file '{Path(s).resolve()}'\n" for s in segments))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", "-movflags", "+faststart", str(video)], check=True)
    listing.unlink()


def chunks(frames: int, fps: int, gpus: int, min_seconds: float = 15.0) -> list[tuple[int, int]]:
    """Split frames 1..frames into at most ``gpus`` runs of at least ``min_seconds`` of film."""
    n = max(1, min(gpus, int(frames // (fps * min_seconds)) or 1))
    size = -(-frames // n)
    return [(a, min(frames, a + size - 1)) for a in range(1, frames + 1, size)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sitetwin_worker")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render the construction sequence of an IFC model")
    r.add_argument("ifc", type=Path)
    r.add_argument("--out", type=Path, default=Path("film"))
    r.add_argument("--lat", type=float)
    r.add_argument("--lon", type=float)
    r.add_argument("--radius", type=float, default=350, help="metres of surroundings around the site")
    r.add_argument("--seconds", type=float, default=90)
    r.add_argument("--quality", choices=list(QUALITY), default="draft")
    r.add_argument("--title")
    r.add_argument("--frames", help="only these frames, e.g. 1,120,400 (stills for checking)")
    v = sub.add_parser("serve", help="take render jobs from a SiteTwin site, forever")
    v.add_argument("--site", required=True, help="SiteTwin's address, e.g. https://example.com/sitetwin")
    v.add_argument("--every", type=float, default=20, help="seconds between asking for work")
    v.add_argument("--once", action="store_true", help="ask once and stop")
    a = p.parse_args(argv)
    if a.cmd == "serve":
        from .serve import serve
        serve(a.site, every=a.every, once=a.once)
        return 0
    if a.cmd == "render":
        frames = [int(x) for x in a.frames.split(",")] if a.frames else None
        res = run(a.ifc, a.out, a.lat, a.lon, a.radius, a.seconds, a.quality, a.title, frames=frames)
        print(json.dumps({k: v for k, v in res.items() if k != "manifest"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
