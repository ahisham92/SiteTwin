"""SiteTwin's renders on Modal: a cloud GPU that runs only while a film is being made.

Deploy once (see README, "Rendering in the cloud with Modal"):

    modal secret create sitetwin SITETWIN_SITE_URL=https://your.site/sitetwin SITETWIN_WORKER_TOKEN=...
    modal deploy modal_app.py

Then:

* every 2 minutes ``dispatch`` asks the site for waiting jobs (a few seconds of CPU);
  the site can also call ``wake`` the moment a job is queued, so nobody waits;
* each job is handled by ``handle_job`` (CPU): it downloads the IFC, fetches the
  surroundings and plans the timeline once, then splits the film into runs of frames;
* each run is rendered by ``render_chunk`` on its own GPU, in parallel, and encoded as
  a segment; the segments are joined without re-encoding and the MP4 goes back to
  the site with what it cost in GPU time.

Set ``SITETWIN_GPU`` (L4, A10G, L40S, A100...) and ``SITETWIN_MAX_GPUS`` when deploying
to trade speed against cost: more GPUs finish sooner for the same total GPU time.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

GPU = os.environ.get("SITETWIN_GPU", "L4")
MAX_GPUS = int(os.environ.get("SITETWIN_MAX_GPUS", "10"))
#: What an hour of each GPU costs on Modal, in US dollars (to report each film's cost).
PRICE_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100": 2.50, "H100": 3.95}

app = modal.App("sitetwin")
work = modal.Volume.from_name("sitetwin-work", create_if_missing=True)
secret = modal.Secret.from_name("sitetwin")

render_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg", "fonts-dejavu-core", "libxrender1", "libxxf86vm1", "libxfixes3", "libxi6",
                 "libxkbcommon0", "libsm6", "libgl1", "libegl1", "libgomp1")
    .pip_install("bpy==5.2.3", "ifcopenshell>=0.8", "numpy>=1.26", "pillow>=10", "flask>=3")
    .env({"SITETWIN_GPU": GPU, "SITETWIN_MAX_GPUS": str(MAX_GPUS)})
    .add_local_python_source("sitetwin_app", "sitetwin_worker")
)
light_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("fastapi[standard]")
    .add_local_python_source("sitetwin_app", "sitetwin_worker")
)


def _site():
    from sitetwin_worker.serve import Site
    return Site(os.environ["SITETWIN_SITE_URL"], os.environ["SITETWIN_WORKER_TOKEN"], f"modal-{GPU}")


@app.function(image=render_image, gpu=GPU, volumes={"/work": work}, timeout=12 * 3600, secrets=[secret])
def render_chunk(job_dir: str, args: dict, first: int, last: int) -> dict:
    from sitetwin_worker.render import run

    work.reload()
    t = time.time()
    folder = Path(job_dir)
    res = run(folder / "model.ifc", folder, args["lat"], args["lon"], seconds=args["seconds"],
              quality=args["quality"], title=args["title"], chunk=(first, last))
    work.commit()
    return {"segment": res["segment"], "seconds": time.time() - t, "first": first}


@app.function(image=render_image, volumes={"/work": work}, timeout=24 * 3600, secrets=[secret])
def handle_job(got: dict) -> None:
    import shutil
    import traceback

    from sitetwin_worker.render import chunks, join, run
    from sitetwin_worker.serve import mount_prefix

    site = _site()
    job, project = got["job"], got["project"]
    jid = job["id"]
    folder = Path(f"/work/job_{jid}")
    folder.mkdir(parents=True, exist_ok=True)

    def progress(pct: float, msg: str) -> None:
        try:
            site.json("POST", f"/api/worker/jobs/{jid}/progress", {"percent": pct, "message": msg})
        except Exception:                                  # noqa: BLE001 - progress is best effort
            pass

    try:
        progress(2, "Fetching the model and the surroundings")
        site.download(mount_prefix(site.base) + got["model_url"], folder / "model.ifc")
        args = {"lat": project["latitude"], "lon": project["longitude"], "seconds": job["seconds"],
                "quality": job["quality"], "title": project["name"]}
        prep = run(folder / "model.ifc", folder, args["lat"], args["lon"], seconds=args["seconds"],
                   quality=args["quality"], title=args["title"], prepare_only=True)
        work.commit()
        runs = chunks(prep["frames"], prep["fps"], MAX_GPUS)
        progress(8, f"Rendering {prep['frames']} frames on {len(runs)} {GPU} GPU{'s' if len(runs) > 1 else ''}")
        done, gpu_seconds, segments = 0, 0.0, {}
        for r in render_chunk.starmap([(str(folder), args, a, b) for a, b in runs], order_outputs=False):
            done += 1
            gpu_seconds += r["seconds"]
            segments[r["first"]] = r["segment"]
            progress(8 + 88 * done / len(runs), f"{done} of {len(runs)} parts rendered")
        work.reload()
        video = folder / "film.mp4"
        join([Path(segments[k]) for k in sorted(segments)], video)
        cost = gpu_seconds / 3600 * PRICE_PER_HOUR.get(GPU, 1.0)
        missing = prep["manifest"].get("missing", {})
        note = f"{gpu_seconds / 3600:.1f} GPU-hours on {GPU}, about ${cost:.2f}."
        if missing:
            note += " Without " + ", ".join(missing) + " (could not be fetched)."
        site.upload(f"/api/worker/jobs/{jid}/video", "video", video, {"message": note})
    except Exception as exc:                               # noqa: BLE001 - reported to the site
        traceback.print_exc()
        try:
            site.json("POST", f"/api/worker/jobs/{jid}/failed", {"message": f"{type(exc).__name__}: {exc}"[:900]})
        except Exception:                                  # noqa: BLE001
            pass
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        work.commit()


@app.function(image=light_image, secrets=[secret], schedule=modal.Period(minutes=2), timeout=300)
def dispatch() -> int:
    """Hand every waiting job to its own handler."""
    from sitetwin_worker.serve import claim

    site = _site()
    n = 0
    while (got := claim(site)) is not None:
        handle_job.spawn(got)
        n += 1
    return n


@app.function(image=light_image, secrets=[secret])
@modal.fastapi_endpoint(method="POST")
def wake(body: dict):
    """Called by the site when a job is queued (``{"token": ...}``), so the render starts at once."""
    import hmac

    if not hmac.compare_digest(str(body.get("token", "")), os.environ["SITETWIN_WORKER_TOKEN"]):
        return {"ok": False}
    dispatch.spawn()
    return {"ok": True}
