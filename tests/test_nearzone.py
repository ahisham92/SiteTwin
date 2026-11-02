import math

import numpy as np

from sitetwin_worker import nearzone as N


def _city(seed=0):
    """A flat street at z=0 with a 12 m block (8 x 8 m at x 40..48) and a parked car."""
    rng = np.random.default_rng(seed)
    pts = []
    for x in np.arange(-180, 180, 1.5):
        for y in np.arange(-180, 180, 1.5):
            z = 0.0
            if 40 <= x <= 48 and 40 <= y <= 48:
                z = 12.0
            elif 60 <= x <= 64 and 10 <= y <= 12:
                z = 1.5                                   # a car
            pts.append((x + rng.uniform(-0.2, 0.2), y + rng.uniform(-0.2, 0.2), z))
    xyz = np.array(pts)
    colour = np.tile([0.5, 0.45, 0.4], (len(xyz), 1))
    steep = np.zeros(len(xyz), bool)
    return xyz, colour, steep


FEATS = [{"kind": "building", "id": 1, "points": [[40, 40], [48, 40], [48, 48], [40, 48], [40, 40]]},
         {"kind": "building", "id": 2, "points": [[100, -60], [110, -60], [110, -50], [100, -50], [100, -60]]},
         {"kind": "road", "class": "residential", "width": 6.5, "points": [[-100, 20], [100, 20]]},
         {"kind": "road", "class": "footway", "width": 2, "points": [[-100, 30], [100, 30]]}]


def test_ground_grid_ignores_buildings_and_cars():
    xyz, _, _ = _city()
    g, o, c = N.ground_grid(xyz, 190)
    h = N.sample(g, o, c, np.array([44.0, 62.0, 0.0]), np.array([44.0, 11.0, 0.0]))
    assert np.all(np.abs(h) < 0.6), h


def test_plan_builds_mapped_block_with_real_height():
    xyz, colour, steep = _city()
    out = N.plan(xyz, colour, steep, FEATS, (-20, -15, 20, 15))
    assert len(out["buildings"]) == 1                      # the second has nothing standing in Google
    b = out["buildings"][0]
    assert abs(b["base"]) < 0.6 and 10.5 < b["top"] < 12.5 and b["floors"] in (3, 4)
    inside = N.inside(xyz[:, :2], FEATS[0]["points"])
    assert np.all(out["z"][inside] < b["base"])            # Google's block pressed under ours


def test_plan_flattens_cars_and_keeps_roads():
    xyz, colour, steep = _city()
    out = N.plan(xyz, colour, steep, FEATS, (-20, -15, 20, 15))
    car = (xyz[:, 0] >= 60) & (xyz[:, 0] <= 64) & (xyz[:, 1] >= 10) & (xyz[:, 1] <= 12)
    assert np.all(out["z"][car] < 0.5)
    assert [r["class"] for r in out["roads"]] == ["residential"]
    v, f, lv, lf = N.road_quads(out["roads"][0], out["grid"], N.RADIUS, (-20, -15, 20, 15))
    assert f and lf
    cross = {"points": [[-100, 0], [-60, 0], [-30, 0], [0, 0], [30, 0], [60, 0], [100, 0]], "width": 6.0}
    v, f, _, _ = N.road_quads(cross, out["grid"], N.RADIUS, (-20, -15, 20, 15))
    assert f and all(abs(x) > 20 for x, _, _ in v)       # no road over the site


def test_inside_buffer():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    pts = np.array([[5, 5], [11, 5], [12, 5], [-0.5, -0.5]])
    assert N.inside(pts, sq).tolist() == [True, False, False, False]
    assert N.inside(pts, sq, 1.5).tolist() == [True, True, False, True]


def test_osm_is_retried_on_a_busy_server():
    from sitetwin_worker import environment as E
    calls = []

    def fetch(url, body=None):
        calls.append(url)
        if len(calls) < 3:
            raise OSError("HTTP Error 502: Bad Gateway")
        return b'{"elements": []}'

    local = E.Local(33.89, 35.50) if hasattr(E, "Local") else None
    assert E.fetch_osm(local, 100, fetch, E.OVERPASS_MIRRORS[0], wait=lambda s: None) == []
    assert len(calls) == 3 and calls[2] != calls[0]


def test_trees_found_pressed_and_kept_off_roofs():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-60, 60, (6000, 2))
    z = np.zeros(len(xy))
    colour = np.full((len(xy), 3), 0.4)
    blob = np.hypot(xy[:, 0] - 30, xy[:, 1] - 30) < 4.5
    z[blob] = 7.0
    colour[blob] = (0.12, 0.2, 0.1)
    roof = (np.abs(xy[:, 0] + 30) < 5) & (np.abs(xy[:, 1] + 30) < 5)
    z[roof] = 9.0
    colour[roof] = (0.12, 0.2, 0.1)                      # a green roof is not a tree
    xyz = np.column_stack([xy, z])
    feats = [{"kind": "building", "points": [[-35, -35], [-25, -35], [-25, -25], [-35, -25], [-35, -35]]}]
    out = N.plan(xyz, colour, np.zeros(len(xy), bool), feats, (-5, -5, 5, 5), radius=100)
    assert out["trees"]
    assert all(math.hypot(t["x"] - 30, t["y"] - 30) < 6 for t in out["trees"])
    assert out["z"][blob].max() < 1.0
    assert "_idx" not in out["trees"][0]


def test_cars_park_at_kerbs_clear_of_site():
    g = (np.zeros((60, 60)), -120.0, 4.0)
    roads = [{"points": [[-100, 20], [100, 20]], "width": 8.0, "class": "residential"},
             {"points": [[-100, -40], [100, -40]], "width": 10.0, "class": "secondary"}]
    life = N.street_life(roads, [], g, (-10, -10, 10, 10), clear=15.0, radius=150)
    assert len(life["parked"]) > 20
    for x, y, z, h in life["parked"]:
        assert min(abs(y - 20), abs(y + 40)) > 1.5      # at the kerb, not in the middle
        assert math.hypot(max(-10 - x, x - 10, 0), max(-10 - y, y - 10, 0)) > 15
    assert life["moving"]
    assert life["street_trees"]


def test_parks_get_trees_but_not_the_site():
    g = (np.zeros((60, 60)), -120.0, 4.0)
    park = [[30, 30], [90, 30], [90, 90], [30, 90], [30, 30]]
    life = N.street_life([], [], g, (-10, -10, 10, 10), clear=15.0, radius=150, greens=[park])
    assert len(life["street_trees"]) > 20
    assert all(30 <= x <= 90 and 30 <= y <= 90 for x, y, z, h in life["street_trees"])
