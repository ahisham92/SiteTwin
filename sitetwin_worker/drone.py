"""From a drone video (or a folder of drone photos) to a survey SiteTwin can stand a film in.

This runs where OpenDroneMap is installed: the ``opendronemap/odm`` image on Modal (see
``process_survey`` in modal_app.py), or any machine with ODM's ``/code/run.py``. It only uses
the standard library, so it can be imported anywhere.

1. The inputs go into an ODM project's ``images`` folder. A video's GPS track is its DJI ``.SRT``
   file: given beside it, it is renamed to the video's own name, which is how ODM finds it; or,
   when the drone wrote the track into the MP4 itself as subtitles, it is pulled out with ffmpeg.
2. ODM picks sharp frames out of the video, matches them, and builds the textured mesh and the
   orthophoto. With a GPS track, everything is in UTM; without, in its own frame (then the
   survey is placed by hand on the project page).
3. The result is packed as ``survey.zip``: the textured OBJ with its textures, ODM's
   ``coords.txt``, our ``survey.json`` (zone and offset), and a small ``preview.jpg`` of the
   orthophoto for the project page.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from . import survey as S

VIDEO = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".ts")
PHOTO = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".dng")
ODM = Path(os.environ.get("SITETWIN_ODM", "/code/run.py"))
GDAL_BIN = Path("/code/SuperBuild/install/bin")


def odm_python() -> str:
    for p in ("/code/venv/bin/python3", "/usr/bin/python3"):
        if Path(p).exists():
            return p
    return "python3"


def gather(inputs: list[Path], project: Path, log=print) -> dict[str, Any]:
    """Put the inputs where ODM wants them; returns what there is (videos, photos, tracks)."""
    images = project / "images"
    images.mkdir(parents=True, exist_ok=True)
    videos = [p for p in inputs if p.suffix.lower() in VIDEO]
    photos = [p for p in inputs if p.suffix.lower() in PHOTO]
    tracks = [p for p in inputs if p.suffix.lower() == ".srt"]
    for i, v in enumerate(videos):
        target = images / f"flight{i + 1}{v.suffix.lower()}"
        shutil.copy(v, target)
        srt = next((t for t in tracks if t.stem.lower() == v.stem.lower()), None)
        if srt is None and len(videos) == 1 and len(tracks) == 1:
            srt = tracks[0]                         # one video, one track: they go together
        if srt is not None:
            shutil.copy(srt, target.with_suffix(".srt"))
            log(f"  {v.name}: GPS track {srt.name}")
        elif embedded_track(target, target.with_suffix(".srt")):
            log(f"  {v.name}: GPS track read from inside the video")
        else:
            log(f"  {v.name}: no GPS track; the survey will be placed by hand")
    for p in photos:
        shutil.copy(p, images / p.name)
    have_track = any(p.suffix == ".srt" for p in images.iterdir())
    return {"videos": len(videos), "photos": len(photos), "gps": have_track or bool(photos)}


def embedded_track(video: Path, out: Path) -> bool:
    """DJI drones write the GPS track into the MP4 as a subtitle stream: pull it out as an .srt."""
    if not shutil.which("ffmpeg"):
        return False
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:s:0", str(out)],
                       check=True, capture_output=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        out.unlink(missing_ok=True)
        return False
    text = out.read_text(errors="ignore") if out.exists() else ""
    if "latitude" in text.lower() or "gps" in text.lower():
        return True
    out.unlink(missing_ok=True)
    return False


def odm_command(project: Path, info: dict[str, Any], frames: int = 450, quality: str = "medium") -> list[str]:
    """ODM's command line for a site survey: a textured mesh good enough to stand a film in,
    without the outputs SiteTwin does not use."""
    args = [odm_python(), str(ODM), "--project-path", str(project.parent), project.name,
            "--video-limit", str(frames), "--video-resolution", "2400",
            "--feature-quality", "high", "--pc-quality", quality,
            "--mesh-size", "400000", "--mesh-octree-depth", "11",
            "--auto-boundary", "--use-3dmesh",
            "--orthophoto-resolution", "5", "--skip-report", "--optimize-disk-space"]
    if not info.get("gps"):
        # no GPS: video frames are matched with their neighbours in time
        args += ["--matcher-order", "12"]
    return args


#: ODM's stages, in words for the project page.
STAGE_WORDS = {"dataset": "Picking sharp frames out of the video", "split": "Picking sharp frames out of the video",
               "opensfm": "Matching the frames and working out where the drone was",
               "openmvs": "Building a dense cloud of points", "odm_filterpoints": "Cleaning the points",
               "odm_meshing": "Building the 3D mesh", "mvs_texturing": "Painting the mesh from the video",
               "odm_georeferencing": "Placing it on Earth", "odm_dem": "Making the plan picture",
               "odm_orthophoto": "Making the plan picture"}


def run_odm(project: Path, info: dict[str, Any], log=print, frames: int = 450, quality: str = "medium",
            on_stage=None) -> None:
    cmd = odm_command(project, info, frames, quality)
    log("  " + " ".join(cmd[1:]))
    env = dict(os.environ)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                          cwd="/code" if Path("/code").exists() else None) as proc:
        tail: list[str] = []
        for line in proc.stdout:                    # type: ignore[union-attr]
            line = line.rstrip()
            tail = (tail + [re.sub(r"\x1b\[[0-9;]*m", "", line)])[-40:]
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
            if "[INFO]" in clean and ("Running" in clean or "Finished" in clean) or "[ERROR]" in clean:
                log("  odm: " + clean[-160:])
            m = re.search(r"Running (\w+) stage", clean)
            if m and on_stage and m.group(1) in STAGE_WORDS:
                on_stage(STAGE_WORDS[m.group(1)])
        if proc.wait() != 0:
            raise RuntimeError("OpenDroneMap stopped: " + " | ".join(t for t in tail[-6:] if t)[-700:])


def _turn(v: list[float]) -> list[list[float]]:
    """The rotation matrix of an axis-angle vector (Rodrigues)."""
    import math
    th = math.sqrt(sum(x * x for x in v))
    if th < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    k = [x / th for x in v]
    K = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]]
    KK = [[sum(K[i][m] * K[m][j] for m in range(3)) for j in range(3)] for i in range(3)]
    s, c = math.sin(th), 1 - math.cos(th)
    return [[(1.0 if i == j else 0.0) + s * K[i][j] + c * KK[i][j] for j in range(3)] for i in range(3)]


def camera_centres(project: Path) -> list[list[float]]:
    """Where the drone was for each frame used, in the model's own frame (from OpenSfM): with no
    GPS, they say which way is up (the drone flew above the ground)."""
    rec = project / "opensfm" / "reconstruction.json"
    if not rec.exists():
        return []
    parts = json.loads(rec.read_text())
    if not parts:
        return []
    shots = max(parts, key=lambda p: len(p.get("shots", {}))).get("shots", {})
    out = []
    for shot in shots.values():
        R, t = _turn(shot["rotation"]), shot["translation"]
        out.append([round(-sum(R[j][i] * t[j] for j in range(3)), 4) for i in range(3)])
    return out


def preview(project: Path, out: Path) -> bool:
    """A small JPEG of the orthophoto, for the project page."""
    ortho = project / "odm_orthophoto" / "odm_orthophoto.tif"
    tool = GDAL_BIN / "gdal_translate"
    if not ortho.exists():
        return False
    exe = str(tool) if tool.exists() else (shutil.which("gdal_translate") or "")
    if not exe:
        return False
    try:
        subprocess.run([exe, "-q", "-of", "JPEG", "-outsize", "1400", "0", "-b", "1", "-b", "2", "-b", "3",
                        "-co", "QUALITY=82", str(ortho), str(out)], check=True, capture_output=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return out.exists()


def package(project: Path, out: Path) -> dict[str, Any]:
    """The finished survey as one zip, and what it holds."""
    tex = project / "odm_texturing"
    meshes = sorted(tex.glob("*_geo.obj")) or sorted(tex.glob("*.obj"))
    if not meshes:
        raise RuntimeError("OpenDroneMap made no textured model (too few sharp, overlapping frames?).")
    mesh = meshes[0]
    geo, _ = S.read_georef(project / "odm_georeferencing") if (project / "odm_georeferencing").exists() \
        else (None, [])
    if not mesh.name.endswith("_geo.obj"):
        geo = None                                  # the plain model is in ODM's own frame
    info: dict[str, Any] = {"mesh": mesh.name, "made_by": "OpenDroneMap"}
    if geo is not None:
        info.update(utm_zone=geo.zone, north=geo.north, offset=list(geo.offset))
    else:
        info["cameras"] = camera_centres(project)
    has_preview = preview(project, project / "preview.jpg")
    mtl_names = set()
    for line in mesh.read_text(errors="ignore").splitlines()[:50]:
        if line.startswith("mtllib "):
            mtl_names.add(line.split(" ", 1)[1].strip())
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        z.write(mesh, f"model/{mesh.name}")
        for name in mtl_names:
            mtl = tex / name
            if mtl.exists():
                z.write(mtl, f"model/{name}")
                for line in mtl.read_text(errors="ignore").splitlines():
                    if line.strip().startswith("map_"):
                        img = tex / line.split()[-1]
                        if img.exists():
                            z.write(img, f"model/{img.name}", compress_type=zipfile.ZIP_STORED)
        coords = project / "odm_georeferencing" / "coords.txt"
        if coords.exists():
            z.write(coords, "odm_georeferencing/coords.txt")
        if has_preview:
            z.write(project / "preview.jpg", "preview.jpg", compress_type=zipfile.ZIP_STORED)
        z.writestr("survey.json", json.dumps(info, indent=1))
    info["preview"] = has_preview
    info["bytes"] = out.stat().st_size
    return info


def needs_odm(inputs: list[Path]) -> bool:
    """A video or photos must go through OpenDroneMap; a finished model need not."""
    return any(p.suffix.lower() in VIDEO + PHOTO for p in inputs)


def process(inputs: list[Path], work: Path, log=print, frames: int = 450, quality: str = "medium",
            on_stage=None) -> tuple[Path, dict[str, Any]]:
    """Inputs (videos, their .SRT tracks, or photos) to ``work/survey.zip``."""
    project = work / "survey"
    if project.exists():
        shutil.rmtree(project)
    info = gather(inputs, project, log)
    if not info["videos"] and info["photos"] < 5:
        raise ValueError("Send a drone video (with its .SRT if there is one) or at least five photos.")
    run_odm(project, info, log, frames, quality, on_stage)
    out = work / "survey.zip"
    made = package(project, out)
    made.update(info)
    return out, made
