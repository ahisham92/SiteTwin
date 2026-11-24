"""SiteTwin's drone step on Modal: a drone video (or photos) to a textured 3D model of the site,
with OpenDroneMap, on a big CPU machine that runs only while a survey is being made.

It is its own Modal app, so the film renderer (modal_app.py) deploys and runs whether or not
this one does:

    modal deploy modal_survey.py

modal_app.py's ``handle_survey`` puts what the person sent in the shared ``sitetwin-work`` volume
and calls ``odm`` here by name; ``odm`` leaves ``survey.zip`` beside it (see
sitetwin_worker/drone.py for what it holds). OpenDroneMap is open source (AGPL) and runs as its
own program; nothing is paid per use beyond the machine's time (about a dollar for a few minutes
of video).
"""

from __future__ import annotations

from pathlib import Path

import modal

ODM_IMAGE = "opendronemap/odm:3.6.2"

app = modal.App("sitetwin-survey")
work = modal.Volume.from_name("sitetwin-work", create_if_missing=True)
#: The stage each running survey is at, "<job>" -> words, for the project page.
stages = modal.Dict.from_name("sitetwin-survey-stage", create_if_missing=True)

odm_image = (
    # ODM's own image, with a separate Python for Modal's client so ODM's own stays as it is
    modal.Image.from_registry(ODM_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("ffmpeg")
    .add_local_python_source("sitetwin_worker")
)


@app.function(image=odm_image, volumes={"/work": work}, cpu=16.0, memory=32768, timeout=8 * 3600)
def odm(job_dir: str, key: str = "", frames: int = 450, quality: str = "medium") -> dict:
    """``job_dir/in`` (the video and its track, or photos) to ``job_dir/odm/survey.zip``."""
    import shutil
    import tempfile

    from sitetwin_worker import drone

    work.reload()
    src = Path(job_dir) / "in"
    # ODM works on fast local disk; only the result goes back to the volume
    tmp = Path(tempfile.mkdtemp())
    inputs = []
    for p in sorted(src.iterdir()):
        shutil.copy(p, tmp / p.name)
        inputs.append(tmp / p.name)

    def stage(words: str) -> None:
        if key:
            stages[key] = words

    made, info = drone.process(inputs, tmp / "run", frames=frames, quality=quality, on_stage=stage)
    out = Path(job_dir) / "odm"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(made, out / "survey.zip")
    work.commit()
    shutil.rmtree(tmp, ignore_errors=True)
    return {k: v for k, v in info.items() if k != "cameras"}


@app.local_entrypoint()
def try_video(video: str, srt: str = "", frames: int = 200, quality: str = "low", out: str = "survey.zip"):
    """Run the drone step on a video from this computer: ``modal run modal_survey.py --video x.mp4``."""
    import uuid
    name = f"/survey_try_{uuid.uuid4().hex[:8]}"
    with work.batch_upload() as up:
        up.put_file(video, f"{name}/in/{Path(video).name}")
        if srt:
            up.put_file(srt, f"{name}/in/{Path(srt).name}")
    info = odm.remote("/work" + name, "", frames, quality)
    data = b"".join(work.read_file(f"{name}/odm/survey.zip"))
    Path(out).write_bytes(data)
    print(f"Written {out} ({len(data) / 1e6:.1f} MB): {info}")
    work.remove_file(name, recursive=True)
