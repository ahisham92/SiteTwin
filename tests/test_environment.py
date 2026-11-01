import io
import json

import numpy as np
from PIL import Image

from sitetwin_worker import environment as env
from sitetwin_worker.geo import Local, sun_position


def terrarium_png(metres: float) -> bytes:
    v = metres + 32768
    r, g = int(v // 256), int(v % 256)
    b = int(round((v - int(v)) * 256))
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (r, g, b)).save(buf, "PNG")
    return buf.getvalue()


OSM = {"elements": [
    {"type": "way", "id": 1, "tags": {"building": "yes", "building:levels": "4"},
     "geometry": [{"lat": 33.6605, "lon": 35.4205}, {"lat": 33.6605, "lon": 35.4207},
                  {"lat": 33.6607, "lon": 35.4207}, {"lat": 33.6605, "lon": 35.4205}]},
    {"type": "way", "id": 2, "tags": {"highway": "primary"},
     "geometry": [{"lat": 33.659, "lon": 35.419}, {"lat": 33.661, "lon": 35.421}]},
    {"type": "way", "id": 3, "tags": {"amenity": "bench"},
     "geometry": [{"lat": 33.66, "lon": 35.42}, {"lat": 33.6601, "lon": 35.4201}]},
]}


def fake_fetch(url, data=None):
    if "terrarium" in url:
        return terrarium_png(15.0)
    if data is not None:
        return json.dumps(OSM).encode()
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (90, 120, 60)).save(buf, "JPEG")
    return buf.getvalue()


def test_everything_is_gathered_into_one_folder(tmp_path):
    m = env.gather(33.66, 35.42, 200, tmp_path, fetch=fake_fetch)
    assert m["missing"] == {}
    h = np.load(tmp_path / "terrain.npy")
    assert abs(float(h.mean()) - 15.0) < 0.05
    assert Image.open(tmp_path / "imagery.jpg").size == (2048, 2048)
    feats = json.loads((tmp_path / "osm.json").read_text())
    assert [f["kind"] for f in feats] == ["building", "road"]
    assert feats[0]["height"] == 4 * env.LEVEL_HEIGHT + 1.0
    assert len(m["credits"]) == 3


def test_what_cannot_be_fetched_is_noted_and_the_rest_goes_on(tmp_path):
    def broken(url, data=None):
        if "terrarium" in url:
            return terrarium_png(3.0)
        raise OSError("blocked")
    m = env.gather(33.66, 35.42, 200, tmp_path, fetch=broken)
    assert "terrain" in m and set(m["missing"]) == {"imagery", "osm"}


def test_a_second_run_reuses_what_it_has(tmp_path):
    env.gather(33.66, 35.42, 200, tmp_path, fetch=fake_fetch)
    def nothing(url, data=None):
        raise AssertionError("fetched again")
    env.gather(33.66, 35.42, 200, tmp_path, fetch=nothing)


def test_unmapped_buildings_get_believable_stable_heights():
    a = env._height({"building": "apartments"}, 42)
    assert a == env._height({"building": "apartments"}, 42)
    assert 13 <= a <= 23
    assert env._height({"height": "31 m"}, 1) == 31.0


def test_local_metres_round_trip():
    loc = Local(33.66, 35.42)
    x, y = loc.to_xy(35.43, 33.67)
    assert 900 < x < 940 and 1100 < y < 1120
    lon, lat = loc.to_lonlat(x, y)
    assert abs(lon - 35.43) < 1e-9 and abs(lat - 33.67) < 1e-9


def test_the_sun_is_high_and_south_at_midday_in_june_in_beirut():
    elev, az = sun_position(33.9, 35.5, 2026, 6, 21, 12 - 35.5 / 15)
    assert 75 < elev < 81 and 150 < az < 210
