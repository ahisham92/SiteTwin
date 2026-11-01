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


def caption(path: Path, frame: int, tl, title: str, credits: list[str]) -> None:
    """The step being built, a progress bar across the build, and the credits, burnt into the frame."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    u = H / 1080
    d = ImageDraw.Draw(im, "RGBA")
    step = None
    for st in tl.steps:
        if st.start <= frame < st.end:
            step = st
    if frame < tl.build_start:
        line = "The site, cleared and set out"
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
    d.text((pad, H - box_h + int(42 * u)), line, font=_font(int(26 * u)), fill=(235, 238, 245, 255))
    span = max(1, tl.build_end - tl.build_start)
    prog = min(1.0, max(0.0, (frame - tl.build_start) / span))
    bx0, bx1, by = pad, W - pad, H - int(18 * u)
    d.rectangle([bx0, by - int(4 * u), bx1, by], fill=(255, 255, 255, 60))
    d.rectangle([bx0, by - int(4 * u), bx0 + (bx1 - bx0) * prog, by], fill=(245, 165, 36, 255))
    if credits:
        f = _font(int(15 * u))
        txt = "  ·  ".join(credits)
        tw = d.textlength(txt, font=f)
        d.text((W - pad - tw, H - box_h + int(8 * u)), txt, font=f, fill=(220, 225, 235, 200))
    im.save(path)


def run(ifc: Path, out: Path, lat: float | None = None, lon: float | None = None, radius: float = 350,
        seconds: float = 90, quality: str = "draft", title: str | None = None,
        when: dt.datetime | None = None, frames: list[int] | None = None, log=print) -> dict[str, Any]:
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

    log(f"Reading {ifc.name}…")
    elements = model.load(ifc)
    if not elements:
        raise SystemExit("No geometry could be read from the model.")
    log(f"  {len(elements)} elements")

    log(f"Fetching the surroundings of {lat:.5f}, {lon:.5f} (radius {radius:.0f} m)…")
    manifest = environment.gather(lat, lon, radius, out / "environment")
    for k, why in manifest.get("missing", {}).items():
        log(f"  no {k}: {why}")

    tl = sequence.plan(elements, seconds=seconds, fps=q["fps"])
    (out / "timeline.json").write_text(json.dumps(tl.as_dict(), indent=1))
    log(f"  {len(tl.steps)} steps, {tl.frames} frames")

    from . import scene
    frames_dir = out / "frames"
    if frames_dir.exists() and frames is None:
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(exist_ok=True)
    opts = scene.Options(width=q["width"], height=q["height"], samples=q["samples"], fps=q["fps"], when=when)
    info = scene.build(elements, tl, out / "environment", manifest, opts, frames_dir)
    log(f"  scene built: {info}")

    import bpy
    sc = bpy.context.scene
    todo = frames or list(range(1, tl.frames + 1))
    t_render = time.time()
    for i, f in enumerate(todo):
        sc.frame_set(f)
        path = frames_dir / f"frame_{f:05d}.png"
        sc.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        caption(path, f, tl, title, manifest.get("credits", []))
        if i == 0 or (i + 1) % 25 == 0:
            per = (time.time() - t_render) / (i + 1)
            log(f"  frame {f} ({i + 1}/{len(todo)}), {per:.1f} s a frame, ~{per * (len(todo) - i - 1) / 60:.0f} min to go")

    result: dict[str, Any] = {"frames": len(todo), "seconds": time.time() - t0, "scene": info,
                              "manifest": manifest}
    if frames is None:
        video = out / (ifc.stem + ".mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(q["fps"]),
                        "-i", str(frames_dir / "frame_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", "-preset", "slow", "-movflags", "+faststart", str(video)], check=True)
        result["video"] = str(video)
        log(f"Written {video}")
    return result


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
