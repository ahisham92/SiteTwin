"""The drone survey: the site's side (upload, the worker's turn, placement) and the worker's pure
helpers (UTM, what a package says about itself, the placement maths)."""

import io
import json
import math
import zipfile

import pytest

from sitetwin_app.app import SITE_KEY, create_app
from sitetwin_app.store import Store
from sitetwin_worker import drone
from sitetwin_worker import survey as S


@pytest.fixture()
def app(tmp_path):
    return create_app(Store(tmp_path))


def client(app):
    c = app.test_client()
    c.environ_base[SITE_KEY] = {"id": "7", "name": "Ahmed", "login": "ahmed"}
    return c


def _project(c):
    return int(c.post("/projects", data={"name": "Sibline"}).headers["Location"].split("/")[-1])


def _ready_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("survey.glb", b"glTF")
        z.writestr("plan.png", b"\x89PNG")
        z.writestr("survey.json", json.dumps({"mesh": "survey.glb", "plan": "plan.png", "georef": None,
                                              "metric": False, "extent": [1.6, 1.5], "faces": 10}))
    return buf.getvalue()


def test_a_drone_video_goes_to_the_worker_and_comes_back_ready_to_place(app, monkeypatch):
    monkeypatch.setenv("SITETWIN_WORKER_TOKEN", "s3cret")
    auth = {"Authorization": "Bearer s3cret"}
    c, w = client(app), app.test_client()
    pid = _project(c)
    up = c.post(f"/projects/{pid}/survey", data={"survey": [(io.BytesIO(b"video"), "DJI_0001.MP4")]},
                content_type="multipart/form-data")
    assert up.status_code == 302
    s = c.get(f"/projects/{pid}/state.json").json
    assert s["survey"]["state"] == "queued" and s["job"] is None       # not a film: the Video tab is untouched

    got = w.post("/api/worker/next", headers=auth).json
    assert got["job"]["kind"] == "survey" and len(got["sources"]) == 1
    jid = got["job"]["id"]
    assert w.get(got["sources"][0], headers=auth).data == b"video"
    w.post(f"/api/worker/jobs/{jid}/progress", headers=auth, json={"percent": 30, "message": "Matching frames"})
    assert c.get(f"/projects/{pid}/state.json").json["survey"]["message"] == "Matching frames"

    done = w.post(f"/api/worker/jobs/{jid}/survey", headers=auth,
                  data={"survey": (io.BytesIO(_ready_zip()), "survey.zip")}, content_type="multipart/form-data")
    assert done.status_code == 200
    sv = c.get(f"/projects/{pid}/state.json").json["survey"]
    assert sv["state"] == "ready" and sv["extent"] == [1.6, 1.5] and not sv["metric"]
    assert c.get(sv["plan_url"]).status_code == 200
    assert not (app.store.project_folder(pid) / "survey" / "source").exists()     # the video is let go

    placed = c.post(f"/projects/{pid}/survey/place", json={"east": "12", "turn": "30", "width": "250"},
                    headers={"X-Requested-With": "fetch"})
    assert placed.json["survey"]["place"] == {"east": 12.0, "turn": 30.0, "width": 250.0}
    assert c.post(f"/projects/{pid}/survey/place", json={"turn": "x"}, headers={"X-Requested-With": "fetch"}).status_code == 400

    # the film's job carries the survey and its placement
    c.post(f"/projects/{pid}/model", data={"model": (io.BytesIO(open("tests/fixtures/small.ifc", "rb").read()), "m.ifc")},
           content_type="multipart/form-data")
    c.post(f"/projects/{pid}/render", data={"quality": "preview", "seconds": "20"})
    film = w.post("/api/worker/next", headers=auth).json
    assert film["job"]["kind"] == "film" and film["survey_place"]["width"] == 250.0
    assert w.get(film["survey_url"], headers=auth).data == _ready_zip()


def test_a_film_waits_while_the_survey_is_made_and_a_failed_survey_says_why(app, monkeypatch):
    monkeypatch.setenv("SITETWIN_WORKER_TOKEN", "s3cret")
    auth = {"Authorization": "Bearer s3cret"}
    c, w = client(app), app.test_client()
    pid = _project(c)
    c.post(f"/projects/{pid}/model", data={"model": (io.BytesIO(open("tests/fixtures/small.ifc", "rb").read()), "m.ifc")},
           content_type="multipart/form-data")
    c.post(f"/projects/{pid}/survey", data={"survey": [(io.BytesIO(b"v"), "a.mov"), (io.BytesIO(b"t"), "a.SRT")]},
           content_type="multipart/form-data")
    r = c.post(f"/projects/{pid}/render", data={"quality": "preview", "seconds": "20"}, headers={"X-Requested-With": "fetch"})
    assert r.status_code == 409
    jid = w.post("/api/worker/next", headers=auth).json["job"]["id"]
    w.post(f"/api/worker/jobs/{jid}/failed", headers=auth, json={"message": "OpenDroneMap stopped: too few frames"})
    sv = c.get(f"/projects/{pid}/state.json").json["survey"]
    assert sv["state"] == "failed" and "too few" in sv["message"]
    assert c.post(f"/projects/{pid}/render", data={"quality": "preview", "seconds": "20"}).status_code == 302


def test_only_survey_files_are_taken(app):
    c = client(app)
    pid = _project(c)
    for name in ("plans.dwg", "track.srt"):
        r = c.post(f"/projects/{pid}/survey", data={"survey": [(io.BytesIO(b"x"), name)]},
                   content_type="multipart/form-data", headers={"X-Requested-With": "fetch"})
        assert r.status_code == 400


# ---- the worker's helpers ------------------------------------------------------------------

def test_utm_both_ways():
    # Sibline: 33.651153 N, 35.405074 E is in UTM zone 36N
    e, n, zone, north = S.lonlat_to_utm(35.405074, 33.651153)
    assert zone == 36 and north
    lon, lat = S.utm_to_lonlat(e, n, zone, north)
    assert lon == pytest.approx(35.405074, abs=1e-8) and lat == pytest.approx(33.651153, abs=1e-8)
    # a known point: the central meridian at the equator
    assert S.utm_to_lonlat(500000, 0, 31, True) == pytest.approx((3.0, 0.0), abs=1e-9)


def test_crs_names_are_read():
    assert S.parse_crs("WGS84 UTM 36N") == (36, True)
    assert S.parse_crs("EPSG:32736") == (36, False)
    assert S.parse_crs('PROJCS["WGS 84 / UTM zone 36N",GEOGCS[...]]') == (36, True)
    assert S.parse_crs("Lambert") is None


def test_an_opendronemap_package_says_where_it_is(tmp_path):
    (tmp_path / "odm_georeferencing").mkdir()
    (tmp_path / "odm_georeferencing" / "coords.txt").write_text("WGS84 UTM 36N\n741200 3727400\n")
    (tmp_path / "odm_texturing").mkdir()
    (tmp_path / "odm_texturing" / "odm_textured_model_geo.obj").write_text("v 0 0 0\n")
    (tmp_path / "odm_texturing" / "other.obj").write_text("v 0 0 0\n" * 50)
    assert S.pick_mesh(tmp_path).name == "odm_textured_model_geo.obj"
    geo, _ = S.read_georef(tmp_path)
    assert geo == S.Georef(36, True, (741200.0, 3727400.0))


def test_a_georeferenced_survey_lands_where_its_utm_says():
    lat, lon = 33.651153, 35.405074
    e, n, zone, north = S.lonlat_to_utm(lon, lat)
    sv = S.Survey(mesh=None, georef=S.Georef(zone, north, (e - 50.0, n + 20.0)))
    m = S.frame(sv, lat, lon)
    # the survey's (50, -20) is the site's point
    x = m[0][0] * 50 + m[0][1] * -20 + m[0][3]
    y = m[1][0] * 50 + m[1][1] * -20 + m[1][3]
    assert math.hypot(x, y) < 0.05
    # grid north is turned from true north by the convergence, atan(tan(lon - 33) sin(lat)) in zone 36
    turn = math.degrees(math.atan2(m[1][0], m[0][0]))
    gamma = math.degrees(math.atan(math.tan(math.radians(lon - 33)) * math.sin(math.radians(lat))))
    assert abs(turn) == pytest.approx(gamma, abs=0.01)


def test_a_survey_with_no_gps_is_sized_and_turned_by_hand():
    sv = S.Survey(mesh=None, metric=False, extent=(2.0, 1.0))
    m = S.frame(sv, 33.65, 35.4, {"width": 300, "turn": 90, "east": 10})
    # its +x (east) unit, 150 m once sized, turned 90 degrees clockwise points south
    x = m[0][0] * 1 + m[0][3]
    y = m[1][0] * 1 + m[1][3]
    assert x == pytest.approx(10.0, abs=1e-6) and y == pytest.approx(-150.0, abs=1e-6)
    assert m[2][2] == pytest.approx(150.0)                             # heights scale with it


def test_hull_and_inside():
    hull = S.convex_hull([(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])
    assert len(hull) == 4 and S.inside(hull, 5, 5) and not S.inside(hull, 11, 5)
    assert S.inside(S.shrink(hull, 2), 5, 5) and not S.inside(S.shrink(hull, 2), 0.5, 0.5)


def test_odm_is_told_to_match_video_frames_in_order_when_there_is_no_gps(tmp_path):
    cmd = drone.odm_command(tmp_path / "survey", {"gps": False})
    assert "--matcher-order" in cmd and "survey" in cmd
    assert "--matcher-order" not in drone.odm_command(tmp_path / "survey", {"gps": True})


def test_a_video_and_its_track_go_in_together(tmp_path):
    v, t = tmp_path / "DJI_0042.MP4", tmp_path / "whatever.SRT"
    v.write_bytes(b"v")
    t.write_text("1\n00:00:00,000 --> 00:00:00,033\n[latitude: 33.65] [longitude: 35.40]\n")
    info = drone.gather([v, t], tmp_path / "proj", log=lambda *_: None)
    assert info == {"videos": 1, "photos": 0, "gps": True}
    assert sorted(p.name for p in (tmp_path / "proj" / "images").iterdir()) == ["flight1.mp4", "flight1.srt"]


def test_camera_centres_from_opensfm(tmp_path):
    (tmp_path / "opensfm").mkdir()
    # a camera turned 180 degrees about x (looking down), 30 units up: t = -R c
    shot = {"rotation": [math.pi, 0, 0], "translation": [0, 0, 30]}
    (tmp_path / "opensfm" / "reconstruction.json").write_text(json.dumps([{"shots": {"a.jpg": shot}}]))
    (c,) = drone.camera_centres(tmp_path)
    assert c == pytest.approx([0, 0, 30], abs=1e-6)
