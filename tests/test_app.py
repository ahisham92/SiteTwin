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
