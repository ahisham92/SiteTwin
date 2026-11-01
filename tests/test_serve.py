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

    def fake_run(ifc, out, lat, lon, seconds, quality, title, log, on_stage, on_frame):
        seen.update(ifc=Path(ifc).read_bytes()[:12], lat=lat, quality=quality, title=title)
        on_stage("frames")
        out.mkdir(parents=True, exist_ok=True)
        frame = out / "frame_00001.png"
        from PIL import Image
        Image.new("RGB", (64, 36), "grey").save(frame)
        on_frame(1, 2, frame)
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


def test_the_page_sees_the_stage_and_the_latest_frame(tmp_path, monkeypatch):
    import pytest
    from PIL import Image
    monkeypatch.setenv("SITETWIN_WORKER_TOKEN", "tok")
    app = create_app(Store(tmp_path / "site"))
    c = app.test_client()
    c.environ_base[SITE_KEY] = {"id": "7", "name": "A", "login": "a"}
    pid = int(c.post("/projects", data={"name": "P"}).headers["Location"].split("/")[-1])
    c.post(f"/projects/{pid}/model", data={"model": (io.BytesIO(SMALL), "b.ifc")},
           content_type="multipart/form-data")
    c.post(f"/projects/{pid}/render", data={"quality": "preview", "seconds": "20"})
    server = make_server("127.0.0.1", 0, app)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        site = serve_mod.Site(f"http://127.0.0.1:{server.server_port}", "tok", "pc")
        jid = serve_mod.claim(site)["job"]["id"]
        rep = serve_mod.Reporter(site, jid, preview_every=0)
        rep.stage("surroundings")
        assert c.get(f"/api/projects/{pid}").json["job"]["stage"] == "surroundings"
        Image.new("RGB", (64, 36), "grey").save(tmp_path / "f.png")
        rep.preview(tmp_path / "f.png")
        rep.frames(10, 100, 2, 0.05, force=True)
        job = c.get(f"/api/projects/{pid}").json["job"]
        assert job["frames_done"] == 10 and job["gpus"] == 2 and job["preview_url"]
        assert c.get(job["preview_url"]).status_code == 200
        c.post(f"/projects/{pid}/jobs/{jid}/cancel")
        with pytest.raises(serve_mod.Stopped):
            rep.frames(11, 100, force=True)
    finally:
        server.shutdown()
