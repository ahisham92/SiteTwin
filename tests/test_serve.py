"""The worker loop against a real (local) site: take a job, render (faked here), deliver."""

import io
import threading
from pathlib import Path

from werkzeug.serving import make_server

from sitetwin_app.app import SITE_KEY, create_app
from sitetwin_app.store import Store
from sitetwin_worker import serve as serve_mod

SMALL = (Path(__file__).parent / "fixtures" / "small.ifc").read_bytes()


def test_the_worker_takes_a_job_and_delivers_the_film(tmp_path, monkeypatch):
    monkeypatch.setenv("SITETWIN_WORKER_TOKEN", "tok")
    app = create_app(Store(tmp_path / "site"))
    c = app.test_client()
    c.environ_base[SITE_KEY] = {"id": "7", "name": "A", "login": "a"}
    pid = int(c.post("/projects", data={"name": "Berth 3"}).headers["Location"].split("/")[-1])
    c.post(f"/projects/{pid}/model", data={"model": (io.BytesIO(SMALL), "b.ifc")},
           content_type="multipart/form-data")
    c.post(f"/projects/{pid}/render", data={"quality": "preview", "seconds": "20"})

    server = make_server("127.0.0.1", 0, app)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    seen = {}

    def fake_run(ifc, out, lat, lon, seconds, quality, title, log):
        seen.update(ifc=Path(ifc).read_bytes()[:12], lat=lat, quality=quality, title=title)
        log("  frame 10 (1/2), 1.0 s a frame, ~0 min to go")
        out.mkdir(parents=True, exist_ok=True)
        video = out / "film.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        return {"video": str(video), "manifest": {"missing": {"osm": "blocked"}}}

    monkeypatch.setattr("sitetwin_worker.render.run", fake_run)
    try:
        site = serve_mod.Site(base, "tok", "test-pc")
        assert serve_mod.work_once(site, tmp_path / "work", log=lambda *_: None) is True
        assert serve_mod.work_once(site, tmp_path / "work", log=lambda *_: None) is False
    finally:
        server.shutdown()
    assert seen["ifc"] == b"ISO-10303-21" and seen["quality"] == "preview" and seen["title"] == "Berth 3"
    job = c.get(f"/api/projects/{pid}").json["job"]
    assert job["status"] == "done" and job["worker"] == "test-pc" and "osm" in job["message"]
