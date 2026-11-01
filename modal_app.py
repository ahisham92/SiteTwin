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
import modal.exception

GPU = os.environ.get("SITETWIN_GPU", "L4")
MAX_GPUS = int(os.environ.get("SITETWIN_MAX_GPUS", "10"))
#: What an hour of each GPU costs on Modal, in US dollars (to report each film's cost).
PRICE_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100": 2.50, "H100": 3.95}

app = modal.App("sitetwin")
work = modal.Volume.from_name("sitetwin-work", create_if_missing=True)
secret = modal.Secret.from_name("sitetwin")
#: How many frames each run of a job has rendered, "<job>:<first frame>" -> count, so the job's
#: handler can tell the site how far the whole film is while the GPUs work.
frames_done = modal.Dict.from_name("sitetwin-frames", create_if_missing=True)

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
def render_chunk(job_dir: str, args: dict, first: int, last: int, jid: int = 0) -> dict:
    from sitetwin_worker.render import run
    from sitetwin_worker.serve import Reporter

    work.reload()
    t = time.time()
    folder = Path(job_dir)
    rep = Reporter(_site(), jid, preview_every=45.0)
    key = f"{jid}:{first}"

    def frame(done: int, total: int, path: Path) -> None:
        if done == 1 or done % 5 == 0 or done == total:
            frames_done[key] = done
        rep.preview(path)

    res = run(folder / "model.ifc", folder, args["lat"], args["lon"], seconds=args["seconds"],
              quality=args["quality"], title=args["title"], chunk=(first, last), on_frame=frame)
    work.commit()
    return {"segment": res["segment"], "seconds": time.time() - t, "first": first}


@app.function(image=render_image, volumes={"/work": work}, timeout=24 * 3600, secrets=[secret])
def handle_job(got: dict) -> None:
    import shutil
    import traceback

    from sitetwin_worker.render import chunks, join, run
    from sitetwin_worker.serve import Reporter, Stopped, mount_prefix

    site = _site()
    job, project = got["job"], got["project"]
    jid = job["id"]
    folder = Path(f"/work/job_{jid}")
    folder.mkdir(parents=True, exist_ok=True)
    rep = Reporter(site, jid)
    calls: list = []

    try:
        rep.stage("starting")
        site.download(mount_prefix(site.base) + got["model_url"], folder / "model.ifc")
        args = {"lat": project["latitude"], "lon": project["longitude"], "seconds": job["seconds"],
                "quality": job["quality"], "title": project["name"]}
        prep = run(folder / "model.ifc", folder, args["lat"], args["lon"], seconds=args["seconds"],
                   quality=args["quality"], title=args["title"], prepare_only=True, on_stage=rep.stage)
        work.commit()
        runs = chunks(prep["frames"], prep["fps"], MAX_GPUS)
        total = prep["frames"]
        rep.frames(0, total, len(runs), 0.0, force=True)
        started = time.time()
        calls = [(a, render_chunk.spawn(str(folder), args, a, b, jid)) for a, b in runs]
        results: dict[int, dict] = {}
        while len(results) < len(calls):
            for first, call in calls:
                if first in results:
                    continue
                try:
                    results[first] = call.get(timeout=0)
                except (TimeoutError, modal.exception.TimeoutError):
                    pass
            done = sum(b - a + 1 for a, b in runs if a in results) + \
                sum(int(frames_done.get(f"{jid}:{a}", 0)) for a, b in runs if a not in results)
            gpu_seconds = sum(r["seconds"] for r in results.values()) + \
                (time.time() - started) * (len(calls) - len(results))
            rep.frames(min(done, total), total, len(runs), gpu_seconds / 3600 * PRICE_PER_HOUR.get(GPU, 1.0))
            if len(results) < len(calls):
                time.sleep(5)
        gpu_seconds = sum(r["seconds"] for r in results.values())
        rep.stage("joining")
        work.reload()
        video = folder / "film.mp4"
        join([Path(results[k]["segment"]) for k in sorted(results)], video)
        cost = gpu_seconds / 3600 * PRICE_PER_HOUR.get(GPU, 1.0)
        missing = prep["manifest"].get("missing", {})
        note = f"{gpu_seconds / 3600:.1f} GPU-hours on {GPU}, about ${cost:.2f}."
        if prep["manifest"].get("google"):
            note += " Surroundings: Google photo-real 3D."
        if missing:
            note += " Without " + ", ".join(missing) + " (could not be fetched)."
        rep.send(99, "Sending the film to the site", stage="uploading", cost=cost)
        site.upload(f"/api/worker/jobs/{jid}/video", "video", video, {"message": note})
    except Stopped:
        for _, call in calls:
            call.cancel()
    except Exception as exc:                               # noqa: BLE001 - reported to the site
        traceback.print_exc()
        for _, call in calls:
            call.cancel()
        try:
            site.json("POST", f"/api/worker/jobs/{jid}/failed", {"message": f"{type(exc).__name__}: {exc}"[:900]})
        except Exception:                                  # noqa: BLE001
            pass
    finally:
        for a, _ in calls:
            try:
                frames_done.pop(f"{jid}:{a}")
            except Exception:                              # noqa: BLE001 - never written
                pass
        shutil.rmtree(folder, ignore_errors=True)
        work.commit()


@app.function(image=render_image, volumes={"/work": work}, timeout=6 * 3600, secrets=[secret])
def sample_film(ifc: bytes, lat: float, lon: float, seconds: float, quality: str, title: str) -> dict:
    """A film of any model at any place, without the site: for trying changes out."""
    import shutil
    import uuid

    from sitetwin_worker.render import chunks, join, run

    folder = Path(f"/work/sample_{uuid.uuid4().hex[:8]}")
    folder.mkdir(parents=True)
    (folder / "model.ifc").write_bytes(ifc)
    args = {"lat": lat, "lon": lon, "seconds": seconds, "quality": quality, "title": title}
    try:
        prep = run(folder / "model.ifc", folder, lat, lon, seconds=seconds, quality=quality, title=title,
                   prepare_only=True)
        work.commit()
        runs = chunks(prep["frames"], prep["fps"], MAX_GPUS)
        results = list(render_chunk.starmap([(str(folder), args, a, b, 0) for a, b in runs]))
        work.reload()
        video = folder / "film.mp4"
        join([Path(r["segment"]) for r in sorted(results, key=lambda r: r["first"])], video)
        gpu_seconds = sum(r["seconds"] for r in results)
        manifest = prep["manifest"]
        return {"video": video.read_bytes(), "gpus": len(runs), "gpu_hours": gpu_seconds / 3600,
                "cost": gpu_seconds / 3600 * PRICE_PER_HOUR.get(GPU, 1.0),
                "google": bool(manifest.get("google")), "skipped": manifest.get("google_skipped"),
                "missing": manifest.get("missing", {})}
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        work.commit()


@app.local_entrypoint()
def sample(ifc: str = "samples/sample-building.ifc", lat: float = 33.66, lon: float = 35.42,
           seconds: float = 60, quality: str = "draft", title: str = "SiteTwin sample", out: str = "film.mp4"):
    got = sample_film.remote(Path(ifc).read_bytes(), lat, lon, seconds, quality, title)
    Path(out).write_bytes(got.pop("video"))
    print(f"Written {out}: {got}")


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
