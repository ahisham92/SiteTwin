"""SiteTwin's web pages and API.

Who is signed in is decided by the site SiteTwin is mounted in (Project Control,
in front of ``/sitetwin``), which puts the account into the request under
``SITE_KEY``. SiteTwin has no sign-in of its own. Run alone for development
with ``SITETWIN_DEV_ACCOUNT=me python -m sitetwin_app``.
"""

from __future__ import annotations

import hmac
import os
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


#: The pipeline every project goes through, in order.
STAGES = [
    ("model", "Model", "The IFC model from Revit, read for its elements, storeys and location."),
    ("location", "Location", "Where on Earth it stands, from the model's site or typed in."),
    ("environment", "Environment", "The real surroundings: terrain, buildings, roads, water, photo-real tiles."),
    ("sequence", "Sequence", "The order it is built in, with the plant and crews each step needs."),
    ("video", "Video", "The build rendered as a film a few minutes long."),
]


def stage_states(project: dict[str, Any], job: dict[str, Any] | None = None) -> dict[str, str]:
    have_model = bool(project.get("model_name"))
    have_place = project.get("latitude") is not None
    status = (job or {}).get("status")
    video = {"done": "done", "running": "running", "queued": "queued", "failed": "failed"}.get(
        status, "next" if have_model and have_place else "later")
    return {
        "model": "done" if have_model else "todo",
        "location": "done" if have_place else "todo",
        "environment": "done" if status == "done" else ("next" if have_model and have_place else "later"),
        "sequence": "done" if status == "done" else (
            "draft" if have_model and (project.get("model_summary") or {}).get("sequence") else "later"),
        "video": video,
    }


def create_app(store: Store | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_MODEL_MB * 1024 * 1024
    app.store = store or Store()

    @app.errorhandler(StoreError)
    def _store_error(exc: StoreError):
        if request.path.startswith("/api/"):
            return jsonify(error=exc.message), exc.status
        return render_template("message.html", message=exc.message), exc.status

    @app.errorhandler(ifc.IfcError)
    def _ifc_error(exc: ifc.IfcError):
        return _store_error(StoreError(400, str(exc)))

    def account() -> dict[str, Any]:
        site = request.environ.get(SITE_KEY)
        if site is None and os.environ.get("SITETWIN_DEV_ACCOUNT"):
            site = {"id": os.environ["SITETWIN_DEV_ACCOUNT"], "name": "Developer"}
        if not site:
            raise StoreError(401, "Sign in to use SiteTwin.")
        return site

    # ---- pages -------------------------------------------------------------

    def latest_job(owner: str, pid: int) -> dict[str, Any] | None:
        jobs = app.store.jobs(owner, pid)
        return jobs[0] if jobs else None

    @app.get("/")
    def home():
        who = account()
        projects = app.store.projects(who["id"])
        for p in projects:
            p["states"] = stage_states(p, latest_job(who["id"], p["id"]))
        return render_template("index.html", who=who, projects=projects)

    @app.post("/projects")
    def new_project():
        who = account()
        project = app.store.create(who["id"], request.form.get("name", ""))
        return redirect(url_for("project_page", pid=project["id"]))

    @app.get("/projects/<int:pid>")
    def project_page(pid: int):
        who = account()
        project = app.store.project(who["id"], pid)
        jobs = app.store.jobs(who["id"], pid)
        job = jobs[0] if jobs else None
        done = next((j for j in jobs if j["status"] == "done"), None)
        return render_template("project.html", who=who, p=project, stages=STAGES, job=job, done=done,
                               states=stage_states(project, job), worker_set=bool(worker_token()))

    @app.post("/projects/<int:pid>/render")
    def render_video(pid: int):
        who = account()
        app.store.queue_render(who["id"], pid, request.form.get("quality", "draft"),
                               request.form.get("seconds", 90))
        wake_renderer()
        return redirect(url_for("project_page", pid=pid))

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

    @app.post("/projects/<int:pid>/model")
    def upload_model(pid: int):
        who = account()
        app.store.project(who["id"], pid)
        upload = request.files.get("model")
        if upload is None or not upload.filename:
            raise StoreError(400, "Choose an .ifc file exported from Revit.")
        if not upload.filename.lower().endswith(".ifc"):
            raise StoreError(400, "SiteTwin reads .ifc files. Export the model from Revit as IFC.")
        path = app.store.project_folder(pid) / "model.ifc"
        upload.save(path)
        summary = ifc.summarise_file(path)
        app.store.set_model(who["id"], pid, upload.filename, summary.as_dict())
        return redirect(url_for("project_page", pid=pid))

    @app.post("/projects/<int:pid>/location")
    def set_location(pid: int):
        who = account()
        try:
            lat, lon = float(request.form["latitude"]), float(request.form["longitude"])
            elev = request.form.get("elevation")
            elev = float(elev) if elev not in (None, "") else None
        except (KeyError, ValueError):
            raise StoreError(400, "Latitude and longitude must be numbers.")
        app.store.set_location(who["id"], pid, lat, lon, elev)
        return redirect(url_for("project_page", pid=pid))

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
        job = app.store.claim(worker())
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
        app.store.progress(jid, data.get("percent", 0), data.get("message", ""))
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
        job = latest_job(who["id"], pid)
        return jsonify(project=project, job=job, stages=stage_states(project, job))

    return app
