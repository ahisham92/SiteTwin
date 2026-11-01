import io
from pathlib import Path

import pytest

from sitetwin_app.app import SITE_KEY, create_app, is_public
from sitetwin_app.store import Store

SMALL = (Path(__file__).parent / "fixtures" / "small.ifc").read_bytes()


@pytest.fixture()
def app(tmp_path):
    return create_app(Store(tmp_path))


def client(app, who="7"):
    c = app.test_client()
    c.environ_base[SITE_KEY] = {"id": who, "name": "Ahmed", "login": "ahmed"}
    return c


def test_nobody_signed_in_is_refused(app):
    assert app.test_client().get("/").status_code == 401
    assert app.test_client().get("/api/projects").status_code == 401
    assert is_public("/static/sitetwin.css") and not is_public("/")


def test_a_project_takes_a_model_and_its_location(app):
    c = client(app)
    made = c.post("/projects", data={"name": "Berth 3"})
    assert made.status_code == 302
    pid = int(made.headers["Location"].rstrip("/").split("/")[-1])

    up = c.post(f"/projects/{pid}/model", data={"model": (io.BytesIO(SMALL), "berth3.ifc")},
                content_type="multipart/form-data")
    assert up.status_code == 302
    p = c.get(f"/api/projects/{pid}").json
    assert p["project"]["location_from"] == "ifc"
    assert p["project"]["latitude"] == pytest.approx(33.6979167, abs=1e-6)
    assert p["stages"]["model"] == "done" and p["stages"]["sequence"] == "draft"
    page = c.get(f"/projects/{pid}").get_data(as_text=True)
    assert "Draft sequence" in page and "Foundations" in page


def test_a_typed_location_is_not_overwritten_by_the_next_model(app):
    c = client(app)
    pid = int(c.post("/projects", data={"name": "X"}).headers["Location"].split("/")[-1])
    c.post(f"/projects/{pid}/location", data={"latitude": "25.2", "longitude": "55.27"})
    c.post(f"/projects/{pid}/model", data={"model": (io.BytesIO(SMALL), "x.ifc")},
           content_type="multipart/form-data")
    assert c.get(f"/api/projects/{pid}").json["project"]["latitude"] == 25.2


def test_only_ifc_files_are_taken(app):
    c = client(app)
    pid = int(c.post("/projects", data={"name": "X"}).headers["Location"].split("/")[-1])
    bad = c.post(f"/projects/{pid}/model", data={"model": (io.BytesIO(b"x"), "x.rvt")},
                 content_type="multipart/form-data")
    assert bad.status_code == 400


def test_one_account_never_sees_anothers_projects(app):
    mine = client(app, "7")
    pid = int(mine.post("/projects", data={"name": "Mine"}).headers["Location"].split("/")[-1])
    theirs = client(app, "8")
    assert theirs.get(f"/projects/{pid}").status_code == 404
    assert theirs.get("/api/projects").json["projects"] == []


# --- the render worker ------------------------------------------------------------------

def project_with_model(c):
    pid = int(c.post("/projects", data={"name": "Berth 3"}).headers["Location"].split("/")[-1])
    c.post(f"/projects/{pid}/model", data={"model": (io.BytesIO(SMALL), "berth3.ifc")},
           content_type="multipart/form-data")
    return pid


def test_a_video_is_queued_picked_up_and_delivered(app, monkeypatch):
    monkeypatch.setenv("SITETWIN_WORKER_TOKEN", "s3cret")
    c = client(app)
    pid = project_with_model(c)
    assert c.post(f"/projects/{pid}/render", data={"quality": "preview", "seconds": "30"}).status_code == 302
    assert c.get(f"/api/projects/{pid}").json["stages"]["video"] == "queued"
    # a second request while one is waiting is refused
    assert c.post(f"/projects/{pid}/render", data={"quality": "preview"}).status_code == 409

    w = app.test_client()
    auth = {"Authorization": "Bearer s3cret", "X-Worker-Name": "PC-01"}
    got = w.post("/api/worker/next", headers=auth).json
    jid = got["job"]["id"]
    assert got["project"]["latitude"] and got["job"]["quality"] == "preview"
    assert w.post("/api/worker/next", headers=auth).json["job"] is None      # nothing else waiting
    ifc = w.get(got["model_url"], headers=auth)
    assert ifc.status_code == 200 and ifc.data.startswith(b"ISO-10303-21")
    w.post(f"/api/worker/jobs/{jid}/progress", headers=auth, json={"percent": 40, "message": "frame 300"})
    assert c.get(f"/api/projects/{pid}").json["job"]["percent"] == 40
    done = w.post(f"/api/worker/jobs/{jid}/video", headers=auth,
                  data={"video": (io.BytesIO(b"\x00\x00\x00\x18ftypmp42"), "film.mp4")},
                  content_type="multipart/form-data")
    assert done.status_code == 200 and done.json["job"]["status"] == "done"
    assert c.get(f"/api/projects/{pid}").json["stages"]["video"] == "done"
    page = c.get(f"/projects/{pid}").get_data(as_text=True)
    assert "<video" in page
    assert c.get(f"/projects/{pid}/video/{jid}.mp4").status_code == 200
    # another account cannot fetch it
    assert client(app, "8").get(f"/projects/{pid}/video/{jid}.mp4").status_code == 404


def test_the_worker_api_wants_the_token(app, monkeypatch):
    w = app.test_client()
    assert w.post("/api/worker/next").status_code == 401                    # no token set on the site
    monkeypatch.setenv("SITETWIN_WORKER_TOKEN", "s3cret")
    assert w.post("/api/worker/next", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert is_public("/api/worker/next")


def test_a_failed_render_says_why_and_can_be_tried_again(app, monkeypatch):
    monkeypatch.setenv("SITETWIN_WORKER_TOKEN", "t")
    c = client(app)
    pid = project_with_model(c)
    c.post(f"/projects/{pid}/render", data={"quality": "draft"})
    w = app.test_client()
    auth = {"Authorization": "Bearer t"}
    jid = w.post("/api/worker/next", headers=auth).json["job"]["id"]
    w.post(f"/api/worker/jobs/{jid}/failed", headers=auth, json={"message": "out of GPU memory"})
    assert "out of GPU memory" in c.get(f"/projects/{pid}").get_data(as_text=True)
    assert c.post(f"/projects/{pid}/render", data={"quality": "draft"}).status_code == 302


def test_no_video_without_a_model(app):
    c = client(app)
    pid = int(c.post("/projects", data={"name": "X"}).headers["Location"].split("/")[-1])
    assert c.post(f"/projects/{pid}/render", data={"quality": "draft"}).status_code == 400
