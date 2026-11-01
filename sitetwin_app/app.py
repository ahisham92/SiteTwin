"""SiteTwin's web pages and API.

Who is signed in is decided by the site SiteTwin is mounted in (Project Control,
in front of ``/sitetwin``), which puts the account into the request under
``SITE_KEY``. SiteTwin has no sign-in of its own. Run alone for development
with ``SITETWIN_DEV_ACCOUNT=me python -m sitetwin_app``.
"""

from __future__ import annotations

import os
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, url_for

from . import ifc
from .store import Store, StoreError

#: Where a mounting site says who is asking: {"id", "name", "login"}.
SITE_KEY = "sitetwin.site"

MAX_MODEL_MB = 500

#: Answered without the site's sign-in.
def is_public(path: str) -> bool:
    return path.startswith("/static/")


#: The pipeline every project goes through, in order.
STAGES = [
    ("model", "Model", "The IFC model from Revit, read for its elements, storeys and location."),
    ("location", "Location", "Where on Earth it stands, from the model's site or typed in."),
    ("environment", "Environment", "The real surroundings: terrain, buildings, roads, water, photo-real tiles."),
    ("sequence", "Sequence", "The order it is built in, with the plant and crews each step needs."),
    ("video", "Video", "The build rendered as a film a few minutes long."),
]


def stage_states(project: dict[str, Any]) -> dict[str, str]:
    have_model = bool(project.get("model_name"))
    have_place = project.get("latitude") is not None
    return {
        "model": "done" if have_model else "todo",
        "location": "done" if have_place else "todo",
        "environment": "next" if have_model and have_place else "later",
        "sequence": "draft" if have_model and (project.get("model_summary") or {}).get("sequence") else "later",
        "video": "later",
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

    @app.get("/")
    def home():
        who = account()
        return render_template("index.html", who=who, projects=app.store.projects(who["id"]),
                               stage_states=stage_states)

    @app.post("/projects")
    def new_project():
        who = account()
        project = app.store.create(who["id"], request.form.get("name", ""))
        return redirect(url_for("project_page", pid=project["id"]))

    @app.get("/projects/<int:pid>")
    def project_page(pid: int):
        who = account()
        project = app.store.project(who["id"], pid)
        return render_template("project.html", who=who, p=project, stages=STAGES,
                               states=stage_states(project))

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

    # ---- API ---------------------------------------------------------------

    @app.get("/api/projects")
    def api_projects():
        return jsonify(projects=app.store.projects(account()["id"]))

    @app.get("/api/projects/<int:pid>")
    def api_project(pid: int):
        project = app.store.project(account()["id"], pid)
        return jsonify(project=project, stages=stage_states(project))

    return app
