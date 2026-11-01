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

    def upload(self, path: str, field: str, file: Path, extra: dict[str, str] | None = None) -> dict[str, Any]:
        boundary = uuid.uuid4().hex
        parts = []
        for k, v in (extra or {}).items():
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{file.name}\"\r\n"
                     f"Content-Type: video/mp4\r\n\r\n".encode())
        body = b"".join(parts) + file.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        with self._req("POST", path, body, f"multipart/form-data; boundary={boundary}") as r:
            return json.loads(r.read())


def mount_prefix(base: str) -> str:
    """The URL the API's own paths are relative to: for a site mounted at /sitetwin the
    model_url it returns already includes /sitetwin, so join it to the host only."""
    from urllib.parse import urlsplit
    s = urlsplit(base)
    return f"{s.scheme}://{s.netloc}"


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
    last = [0.0]

    def report(msg: str) -> None:
        log(msg)
        pct = None
        if msg.strip().startswith("frame ") and "/" in msg:
            done, total = msg.split("(")[1].split(")")[0].split("/")
            pct = 10 + 88 * int(done) / int(total)
        if time.time() - last[0] > 20 or pct is None:
            last[0] = time.time()
            try:
                site.json("POST", f"/api/worker/jobs/{jid}/progress",
                          {"percent": pct if pct is not None else 5, "message": msg.strip()[:200]})
            except urllib.error.URLError:
                pass

    try:
        site.download(mount_prefix(site.base) + got["model_url"], ifc)
        result = run(ifc, folder / f"job_{jid}", project["latitude"], project["longitude"],
                     seconds=job["seconds"], quality=job["quality"], title=project["name"], log=report)
        missing = result["manifest"].get("missing", {})
        note = ("Without " + ", ".join(missing) + " (could not be fetched).") if missing else ""
        site.upload(f"/api/worker/jobs/{jid}/video", "video", Path(result["video"]), {"message": note})
        log(f"Job {jid} delivered")
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
