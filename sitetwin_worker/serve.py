"""The render machine's loop: ask the site for work, render it, send the film back.

    SITETWIN_WORKER_TOKEN=... python -m sitetwin_worker serve --site https://your.site/sitetwin

It only ever calls out to the site, so it runs behind an office firewall, on a
PC under a desk or on a cloud GPU, unchanged. The token is the one set as
``SITETWIN_WORKER_TOKEN`` on the site.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class Site:
    def __init__(self, base: str, token: str, name: str | None = None):
        self.base = base.rstrip("/")
        self.token = token
        self.name = name or socket.gethostname()

    def _req(self, method: str, path: str, body: bytes | None = None, ctype: str | None = None):
        url = path if path.startswith("http") else self.base + (path if path.startswith("/") else "/" + path)
        req = urllib.request.Request(url, data=body, method=method, headers={
            "Authorization": f"Bearer {self.token}", "X-Worker-Name": self.name,
            **({"Content-Type": ctype} if ctype else {})})
        return urllib.request.urlopen(req, timeout=300)

    def json(self, method: str, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(data or {}).encode()
        with self._req(method, path, body, "application/json") as r:
            return json.loads(r.read() or b"{}")

    def download(self, path: str, to: Path) -> None:
        with self._req("GET", path) as r, open(to, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)

    def upload(self, path: str, field: str, file: Path, extra: dict[str, str] | None = None,
               ctype: str = "video/mp4") -> dict[str, Any]:
        boundary = uuid.uuid4().hex
        parts = []
        for k, v in (extra or {}).items():
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{file.name}\"\r\n"
                     f"Content-Type: {ctype}\r\n\r\n".encode())
        body = b"".join(parts) + file.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        with self._req("POST", path, body, f"multipart/form-data; boundary={boundary}") as r:
            return json.loads(r.read())


def mount_prefix(base: str) -> str:
    """The URL the API's own paths are relative to: for a site mounted at /sitetwin the
    model_url it returns already includes /sitetwin, so join it to the host only."""
    from urllib.parse import urlsplit
    s = urlsplit(base)
    return f"{s.scheme}://{s.netloc}"


class Stopped(Exception):
    """The site says the job is no longer running: the person stopped it."""


class Reporter:
    """Tells the site how a job is going: its stage, its frames, and now and then the latest
    frame as a small JPEG, so the page follows the film as it is made. Best effort, except that
    a job the site has stopped raises Stopped, so the render does not go on for nothing."""

    STAGE_PERCENT = {"starting": 2, "surroundings": 4, "scene": 7, "frames": 8, "joining": 97, "uploading": 99}
    STAGE_WORDS = {"starting": "Reading the model", "surroundings": "Fetching the real surroundings",
                   "scene": "Building the scene and planning the sequence", "frames": "Rendering the frames",
                   "joining": "Joining the film", "uploading": "Sending the film to the site"}

    def __init__(self, site: Site, jid: int, preview_every: float = 30.0):
        self.site, self.jid, self.preview_every = site, jid, preview_every
        self.last_post = 0.0
        self.last_preview = 0.0

    def send(self, percent: float, message: str, **more: Any) -> None:
        self.last_post = time.time()
        try:
            self.site.json("POST", f"/api/worker/jobs/{self.jid}/progress",
                           {"percent": percent, "message": message[:200], **more})
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                raise Stopped() from exc
        except urllib.error.URLError:
            pass

    def stage(self, name: str) -> None:
        self.send(self.STAGE_PERCENT.get(name, 5), self.STAGE_WORDS.get(name, name), stage=name)

    def frames(self, done: int, total: int, gpus: int = 1, cost: float | None = None, force: bool = False) -> None:
        if not force and time.time() - self.last_post < 10:
            return
        self.send(8 + 89 * done / max(1, total), f"Rendering frame {done:,} of {total:,} on {gpus} GPU{'s' if gpus > 1 else ''}",
                  stage="frames", frames_done=done, frames_total=total, gpus=gpus, cost=cost)

    def preview(self, frame: Path) -> None:
        if time.time() - self.last_preview < self.preview_every:
            return
        self.last_preview = time.time()
        try:
            from PIL import Image
            small = frame.with_name("preview.jpg")
            with Image.open(frame) as im:
                im = im.convert("RGB")
                im.thumbnail((960, 540))
                im.save(small, quality=82)
            self.site.upload(f"/api/worker/jobs/{self.jid}/preview", "frame", small, ctype="image/jpeg")
        except Exception:                                  # noqa: BLE001 - a preview is a nicety
            pass


def claim(site: Site) -> dict[str, Any] | None:
    """The next job and its project, now this worker's; None when nothing is waiting."""
    got = site.json("POST", "/api/worker/next")
    return got if got.get("job") else None


def work_once(site: Site, workdir: Path, log=print) -> bool:
    """Take one job if there is one. True if a job was handled."""
    got = claim(site)
    if got is None:
        return False
    handle(site, got, workdir, log)
    return True


def handle(site: Site, got: dict[str, Any], workdir: Path, log=print) -> None:
    """Render one claimed job here, start to finish, and deliver it (or say why not)."""
    from .render import run

    job = got["job"]
    jid = job["id"]
    project = got["project"]
    log(f"Job {jid}: {project['name']} ({job['quality']}, {job['seconds']:.0f} s)")
    folder = workdir / f"project_{project['id']}"
    folder.mkdir(parents=True, exist_ok=True)
    ifc = folder / "model.ifc"
    rep = Reporter(site, jid)

    def frame(done: int, total: int, path: Path) -> None:
        rep.frames(done, total)
        rep.preview(path)

    try:
        site.download(mount_prefix(site.base) + got["model_url"], ifc)
        result = run(ifc, folder / f"job_{jid}", project["latitude"], project["longitude"],
                     seconds=job["seconds"], quality=job["quality"], title=project["name"], log=log,
                     on_stage=rep.stage, on_frame=frame)
        rep.stage("uploading")
        missing = result["manifest"].get("missing", {})
        note = ("Without " + ", ".join(missing) + " (could not be fetched).") if missing else ""
        site.upload(f"/api/worker/jobs/{jid}/video", "video", Path(result["video"]), {"message": note})
        log(f"Job {jid} delivered")
    except Stopped:
        log(f"Job {jid} was stopped on the site")
    except Exception as exc:                              # noqa: BLE001 - reported to the site
        traceback.print_exc()
        try:
            site.json("POST", f"/api/worker/jobs/{jid}/failed", {"message": f"{type(exc).__name__}: {exc}"[:900]})
        except urllib.error.URLError:
            pass


def serve(base: str, token: str | None = None, workdir: Path | None = None, every: float = 20.0,
          once: bool = False) -> None:
    token = token or os.environ.get("SITETWIN_WORKER_TOKEN", "")
    if not token:
        raise SystemExit("Set SITETWIN_WORKER_TOKEN to the token set on the site.")
    site = Site(base, token)
    workdir = workdir or Path(os.environ.get("SITETWIN_WORK_DIR") or Path(tempfile.gettempdir()) / "sitetwin")
    print(f"SiteTwin worker '{site.name}' asking {site.base} for work every {every:.0f} s")
    while True:
        try:
            handled = work_once(site, workdir)
        except urllib.error.URLError as exc:
            print(f"Cannot reach the site: {exc}")
            handled = False
        if once:
            return
        if not handled:
            time.sleep(every)
