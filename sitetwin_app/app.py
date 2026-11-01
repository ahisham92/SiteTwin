"""SiteTwin's web pages and API.

Who is signed in is decided by the site SiteTwin is mounted in (Project Control,
in front of ``/sitetwin``), which puts the account into the request under
``SITE_KEY``. SiteTwin has no sign-in of its own. Run alone for development
with ``SITETWIN_DEV_ACCOUNT=me python -m sitetwin_app``.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import time
from typing import Any

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

from . import ifc
from .store import Store, StoreError

#: Where a mounting site says who is asking: {"id", "name", "login"}.
SITE_KEY = "sitetwin.site"

MAX_MODEL_MB = 500

#: Answered without the site's sign-in: the stylesheet, and the render worker's API,
#: where every call carries the worker's own token (SITETWIN_WORKER_TOKEN).
def is_public(path: str) -> bool:
    return path.startswith("/static/") or path.startswith("/api/worker/")


#: The pipeline every project goes through, in order: (key, name, what it is).
STAGES = [
    ("model", "Model", "The IFC model exported from Revit."),
    ("location", "Location", "Where it stands: the real surroundings are rebuilt from it."),
    ("sequence", "Sequence", "The order it goes up in, storey by storey."),
    ("video", "Video", "The build as a film, rendered on cloud GPUs."),
]

#: Each quality as offered on the page, with a rough render time per frame on one L4 GPU
#: (seconds) for the estimate shown before a film is made. Frame sizes match the worker's.
QUALITY_CHOICES = [
    ("preview", "Preview", "640 × 360", "To check the sequence. Quick and cheap.", 24, 2.0),
    ("draft", "Draft", "1280 × 720 HD", "Good enough to share in a meeting.", 30, 6.0),
    ("final", "Final", "1920 × 1080 Full HD", "For a presentation to the client.", 30, 20.0),
    ("4k", "4K", "3840 × 2160", "For a big screen. Slow.", 30, 70.0),
]
GPU_PRICE_PER_HOUR = 0.80          # L4 on Modal, US dollars
MAX_GPUS = int(os.environ.get("SITETWIN_MAX_GPUS", "10"))
SETUP_SECONDS = 240                # starting a GPU and building the scene, once per GPU


def estimate(quality: str, seconds: float) -> dict[str, Any]:
    """About how long a film takes and what it costs, as the render machine will split it."""
    q = next((c for c in QUALITY_CHOICES if c[0] == quality), QUALITY_CHOICES[1])
    fps, per_frame = q[4], q[5]
    frames = int(round(seconds * fps))
    gpus = max(1, min(MAX_GPUS, int(frames // (fps * 15)) or 1))
    gpu_seconds = frames * per_frame + gpus * SETUP_SECONDS
    wall = SETUP_SECONDS + frames * per_frame / gpus + 60
    return {"frames": frames, "gpus": gpus, "minutes": max(1, round(wall / 60)),
            "cost": round(gpu_seconds / 3600 * GPU_PRICE_PER_HOUR, 2)}


def stage_states(project: dict[str, Any], job: dict[str, Any] | None = None,
                 done: dict[str, Any] | None = None) -> dict[str, str]:
    """Each stage as done, active (being worked on), next (the one to do now) or later."""
    have_model = bool(project.get("model_name"))
    have_place = project.get("latitude") is not None
    have_steps = bool((project.get("model_summary") or {}).get("sequence"))
    status = (job or {}).get("status")
    if status in ("queued", "running"):
        video = "active"
    elif done:
        video = "done"
    elif status == "failed":
        video = "failed"
    else:
        video = "next" if have_model and have_place else "later"
    return {
        "model": "done" if have_model else "next",
        "location": "done" if have_place else ("next" if have_model else "later"),
        "sequence": "done" if have_steps else "later",
        "video": video,
    }


def tile_url(lat: float | None, lon: float | None, zoom: int = 17) -> str | None:
    """The satellite picture of a place, one map tile (for a project's card)."""
    if lat is None or lon is None:
        return None
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"


def create_app(store: Store | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_MODEL_MB * 1024 * 1024
    app.store = store or Store()
    app.jinja_env.globals.update(tile_url=tile_url, QUALITY_CHOICES=QUALITY_CHOICES)

    def wants_json() -> bool:
        return request.path.startswith("/api/") or request.accept_mimetypes.best == "application/json" \
            or request.headers.get("X-Requested-With") == "fetch"

    @app.errorhandler(StoreError)
    def _store_error(exc: StoreError):
        if wants_json():
            return jsonify(error=exc.message), exc.status
        return render_template("message.html", message=exc.message, who=None), exc.status

    @app.errorhandler(ifc.IfcError)
    def _ifc_error(exc: ifc.IfcError):
        return _store_error(StoreError(400, str(exc)))

    @app.errorhandler(413)
    def _too_big(exc):
        return _store_error(StoreError(413, f"The model is bigger than {MAX_MODEL_MB} MB."))

    def account() -> dict[str, Any]:
        site = request.environ.get(SITE_KEY)
        if site is None and os.environ.get("SITETWIN_DEV_ACCOUNT"):
            site = {"id": os.environ["SITETWIN_DEV_ACCOUNT"], "name": "Developer"}
        if not site:
            raise StoreError(401, "Sign in to use SiteTwin.")
        return site

    def site_root() -> str:
        """The address of the site SiteTwin is mounted in: its home page and sign-out."""
        root = request.script_root or ""
        return root[: -len("/sitetwin")] if root.endswith("/sitetwin") else ""

    def done_or_back(pid: int, payload: dict[str, Any] | None = None):
        if wants_json():
            return jsonify(state(pid, payload))
        return redirect(url_for("project_page", pid=pid))

    # ---- what a page shows, also as JSON for the page to stay live ------------------

    def machine() -> dict[str, Any]:
        seen = app.store.last_seen("worker")
        ago = time.time() - seen["at"] if seen else None
        return {"configured": bool(worker_token()), "seen_ago": ago,
                "seen_detail": (seen or {}).get("detail"),
                "awake": ago is not None and ago < 6 * 60}

    def job_view(pid: int, job: dict[str, Any] | None) -> dict[str, Any] | None:
        if not job:
            return None
        out = dict(job)
        now = time.time()
        out["elapsed"] = (job.get("finished") or now) - job["started"] if job.get("started") else None
        out["waiting"] = now - job["created"] if job["status"] == "queued" else None
        out["quiet"] = now - (job.get("updated") or job.get("started") or now) if job["status"] == "running" else None
        left = None
        if job["status"] == "running" and job.get("frames_total") and job.get("frames_done") and out["elapsed"]:
            rate = job["frames_done"] / max(1.0, out["elapsed"])
            left = (job["frames_total"] - job["frames_done"]) / rate if rate > 0 else None
        out["left"] = left
        if job["status"] == "done" and job.get("video"):
            out["video_url"] = url_for("video", pid=pid, jid=job["id"])
        if job.get("preview"):
            out["preview_url"] = url_for("preview", pid=pid, jid=job["id"], v=int(job.get("updated") or 0))
        out["estimate"] = estimate(job["quality"], job["seconds"])
        return out

    def state(pid: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        who = account()
        project = app.store.project(who["id"], pid)
        jobs = app.store.jobs(who["id"], pid)
        job = jobs[0] if jobs else None
        done = next((j for j in jobs if j["status"] == "done"), None)
        out = {"project": project, "stages": stage_states(project, job, done),
               "job": job_view(pid, job), "done": job_view(pid, done),
               "history": [job_view(pid, j) for j in jobs[:10]], "machine": machine()}
        if extra:
            out.update(extra)
        return out

    # ---- pages -------------------------------------------------------------

    @app.get("/")
    def home():
        who = account()
        projects = app.store.projects(who["id"])
        films = 0
        for p in projects:
            jobs = app.store.jobs(who["id"], p["id"])
            done = next((j for j in jobs if j["status"] == "done"), None)
            p["job"] = jobs[0] if jobs else None
            p["done"] = done
            p["states"] = stage_states(p, p["job"], done)
            films += sum(1 for j in jobs if j["status"] == "done")
        return render_template("index.html", who=who, projects=projects, films=films, machine=machine(),
                               site_root=site_root())

    @app.post("/projects")
    def new_project():
        who = account()
        project = app.store.create(who["id"], request.form.get("name", ""))
        return redirect(url_for("project_page", pid=project["id"]))

    @app.get("/projects/<int:pid>")
    def project_page(pid: int):
        who = account()
        data = state(pid)
        return render_template("project.html", who=who, s=data, p=data["project"], stages=STAGES,
                               states=data["stages"], site_root=site_root(),
                               estimates={c[0]: estimate(c[0], 90) for c in QUALITY_CHOICES},
                               view_state={"pid": pid, "max_gpus": MAX_GPUS})

    @app.get("/projects/<int:pid>/state.json")
    def project_state(pid: int):
        return jsonify(state(pid))

    @app.get("/projects/<int:pid>/estimate.json")
    def project_estimate(pid: int):
        account()
        try:
            seconds = float(request.args.get("seconds", 90))
        except ValueError:
            seconds = 90.0
        return jsonify(estimate(request.args.get("quality", "draft"), max(10.0, min(600.0, seconds))))

    @app.post("/projects/<int:pid>/rename")
    def rename_project(pid: int):
        who = account()
        app.store.rename(who["id"], pid, request.form.get("name", ""))
        return done_or_back(pid)

    @app.post("/projects/<int:pid>/render")
    def render_video(pid: int):
        who = account()
        app.store.queue_render(who["id"], pid, request.form.get("quality", "draft"),
                               request.form.get("seconds", 90))
        wake_renderer()
        return done_or_back(pid)

    @app.post("/projects/<int:pid>/jobs/<int:jid>/cancel")
    def cancel_job(pid: int, jid: int):
        who = account()
        app.store.project(who["id"], pid)
        app.store.cancel(who["id"], pid, jid)
        return done_or_back(pid)

    def wake_renderer() -> None:
        """Tell a cloud renderer there is work now (SITETWIN_WAKE_URL), rather than at its next look."""
        url = os.environ.get("SITETWIN_WAKE_URL")
        if not url:
            return
        import json as _json
        import urllib.request
        try:
            req = urllib.request.Request(url, data=_json.dumps({"token": worker_token()}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=4).close()
        except Exception:                                  # noqa: BLE001 - it will look by itself soon
            pass

    @app.get("/projects/<int:pid>/video/<int:jid>.mp4")
    def video(pid: int, jid: int):
        who = account()
        app.store.project(who["id"], pid)
        job = app.store.job(jid)
        if job["project_id"] != pid or job["status"] != "done" or not job["video"]:
            abort(404)
        return send_file(app.store.project_folder(pid) / job["video"], mimetype="video/mp4", conditional=True)

    @app.get("/projects/<int:pid>/preview/<int:jid>.jpg")
    def preview(pid: int, jid: int):
        who = account()
        app.store.project(who["id"], pid)
        job = app.store.job(jid)
        if job["project_id"] != pid or not job.get("preview"):
            abort(404)
        answer = send_file(app.store.project_folder(pid) / job["preview"], mimetype="image/jpeg")
        answer.headers["Cache-Control"] = "no-cache"
        return answer

    @app.get("/projects/<int:pid>/model.ifc")
    def model_file(pid: int):
        who = account()
        project = app.store.project(who["id"], pid)
        path = app.store.project_folder(pid) / "model.ifc"
        if not project["model_name"] or not path.exists():
            abort(404)
        return send_file(path, mimetype="application/x-step", download_name=project["model_name"],
                         conditional=True)

    @app.get("/projects/<int:pid>/steps.json")
    def steps(pid: int):
        """The draft sequence with each step's elements by STEP id, for the 3D preview."""
        who = account()
        project = app.store.project(who["id"], pid)
        path = app.store.project_folder(pid) / "steps.json"
        if not project["model_name"]:
            abort(404)
        if not path.exists():                            # a model uploaded before steps were kept
            summary = ifc.summarise_file(app.store.project_folder(pid) / "model.ifc")
            path.write_text(json.dumps({str(k): v for k, v in summary.step_ids.items()}))
        ids = json.loads(path.read_text())
        return jsonify(steps=(project["model_summary"] or {}).get("sequence", []), ids=ids)

    @app.post("/projects/<int:pid>/model")
    def upload_model(pid: int):
        who = account()
        app.store.project(who["id"], pid)
        upload = request.files.get("model")
        if upload is None or not upload.filename:
            raise StoreError(400, "Choose an .ifc file exported from Revit.")
        if not upload.filename.lower().endswith(".ifc"):
            raise StoreError(400, "SiteTwin reads .ifc files. In Revit: File > Export > IFC.")
        folder = app.store.project_folder(pid)
        path = folder / "model.ifc"
        upload.save(path)
        summary = ifc.summarise_file(path)
        (folder / "steps.json").write_text(json.dumps({str(k): v for k, v in summary.step_ids.items()}))
        app.store.set_model(who["id"], pid, upload.filename, summary.as_dict())
        return done_or_back(pid)

    @app.post("/projects/<int:pid>/location")
    def set_location(pid: int):
        who = account()
        form = request.get_json(silent=True) or request.form
        try:
            lat, lon = float(form["latitude"]), float(form["longitude"])
            elev = form.get("elevation")
            elev = float(elev) if elev not in (None, "") else None
        except (KeyError, TypeError, ValueError):
            raise StoreError(400, "Latitude and longitude must be numbers.")
        app.store.set_location(who["id"], pid, lat, lon, elev)
        return done_or_back(pid)

    @app.post("/projects/<int:pid>/delete")
    def delete_project(pid: int):
        who = account()
        app.store.delete(who["id"], pid)
        return redirect(url_for("home"))

    # ---- the render worker ---------------------------------------------------
    # A machine with Blender polls for work, renders, and posts the film back.
    # Each call carries the worker token; nothing here uses the site's sign-in.

    def worker_token() -> str:
        return os.environ.get("SITETWIN_WORKER_TOKEN", "")

    def worker() -> str:
        expected = worker_token()
        auth = request.headers.get("Authorization", "")
        given = auth[7:] if auth.lower().startswith("bearer ") else ""
        if not expected or not hmac.compare_digest(given, expected):
            raise StoreError(401, "Not a SiteTwin worker.")
        return request.headers.get("X-Worker-Name", "worker")

    def running(jid: int) -> dict[str, Any]:
        job = app.store.job(jid)
        if job["status"] != "running":
            raise StoreError(409, "That job is not running.")
        return job

    @app.post("/api/worker/next")
    def worker_next():
        name = worker()
        app.store.saw("worker", name)
        job = app.store.claim(name)
        if job is None:
            return jsonify(job=None)
        p = app.store.project(job["owner"], job["project_id"])
        return jsonify(job=job, project={"id": p["id"], "name": p["name"], "latitude": p["latitude"],
                                         "longitude": p["longitude"], "model_name": p["model_name"]},
                       model_url=url_for("worker_model", jid=job["id"]))

    @app.get("/api/worker/jobs/<int:jid>/model.ifc")
    def worker_model(jid: int):
        worker()
        job = running(jid)
        return send_file(app.store.project_folder(job["project_id"]) / "model.ifc",
                         mimetype="application/x-step", download_name="model.ifc")

    @app.post("/api/worker/jobs/<int:jid>/progress")
    def worker_progress(jid: int):
        worker()
        running(jid)
        data = request.get_json(silent=True) or {}
        app.store.progress(jid, data.get("percent", 0), data.get("message", ""),
                           **{k: data.get(k) for k in ("stage", "frames_done", "frames_total", "gpus", "cost")})
        return jsonify(ok=True)

    @app.post("/api/worker/jobs/<int:jid>/preview")
    def worker_preview(jid: int):
        """The latest frame rendered, as a small JPEG, so the page shows the film as it is made."""
        worker()
        job = running(jid)
        upload = request.files.get("frame")
        if upload is None:
            raise StoreError(400, "Send the frame.")
        name = f"preview_{jid}.jpg"
        upload.save(app.store.project_folder(job["project_id"]) / name)
        app.store.set_preview(jid, name)
        return jsonify(ok=True)

    @app.post("/api/worker/jobs/<int:jid>/video")
    def worker_video(jid: int):
        worker()
        job = running(jid)
        upload = request.files.get("video")
        if upload is None:
            raise StoreError(400, "Send the video.")
        name = f"video_{jid}.mp4"
        upload.save(app.store.project_folder(job["project_id"]) / name)
        return jsonify(job=app.store.finish(jid, name, request.form.get("message", "")))

    @app.post("/api/worker/jobs/<int:jid>/failed")
    def worker_failed(jid: int):
        worker()
        running(jid)
        data = request.get_json(silent=True) or {}
        return jsonify(job=app.store.finish(jid, None, data.get("message", "The render failed.")))

    # ---- API ---------------------------------------------------------------

    @app.get("/api/projects")
    def api_projects():
        return jsonify(projects=app.store.projects(account()["id"]))

    @app.get("/api/projects/<int:pid>")
    def api_project(pid: int):
        who = account()
        project = app.store.project(who["id"], pid)
        return jsonify(state(project["id"]))

    return app
