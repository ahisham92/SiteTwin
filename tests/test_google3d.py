"""Google 3D tiles: walking the tileset, and putting a tile into the site's frame exactly."""

import json
import struct

import numpy as np
import pytest

from sitetwin_worker import google3d as g

LAT, LON = 33.66, 35.42


def gltf_point(p_ecef):
    """An Earth-centred point as glTF (y-up) coordinates."""
    return g.Y_UP_INV @ np.append(p_ecef, 1)


def tile_glb(node):
    doc = {"asset": {"version": "2.0", "copyright": "Data SIO;Airbus"}, "scene": 0,
           "scenes": [{"nodes": [0]}], "nodes": [node]}
    return g.write_glb(doc, b"")


def test_a_tile_lands_where_it_is_on_earth():
    # a node 111 m north of the site, 10 m up, given as a big Earth-centred translation
    target = g.ecef(LAT + 0.001, LON, 10.0)
    t = gltf_point(target)[:3]
    data = tile_glb({"translation": list(t)})
    out, credit = g.to_local(data, np.eye(4), g.enu_matrix(LAT, LON))
    doc, _ = g.read_glb(out)
    m = np.array(doc["nodes"][0]["matrix"]).reshape(4, 4).T
    # what Blender does with it: its own y-up to z-up turn
    where = (g.Y_UP @ m @ np.array([0, 0, 0, 1.0]))[:3]
    assert where[0] == pytest.approx(0, abs=0.01)
    assert where[1] == pytest.approx(110.9, abs=0.3)
    assert where[2] == pytest.approx(10, abs=0.05)
    assert abs(m[:3, 3]).max() < 1000            # small numbers only: no float32 trouble
    assert credit == "Data SIO;Airbus"


def test_the_walk_goes_finer_near_the_site_and_skips_far_tiles():
    site = g.ecef(LAT, LON)
    far = g.ecef(LAT + 0.1, LON)

    def box(c, half):
        return {"box": list(c) + [half, 0, 0, 0, half, 0, 0, 0, half]}

    tileset = {"root": {"boundingVolume": box(site, 5000), "geometricError": 500, "children": [
        {"boundingVolume": box(far, 100), "geometricError": 1, "content": {"uri": "far.glb"}},
        {"boundingVolume": box(site, 1000), "geometricError": 50, "content": {"uri": "coarse.glb"},
         "children": [{"boundingVolume": box(site, 200), "geometricError": 0.5,
                       "content": {"uri": "/v1/3dtiles/fine.glb?session=abc"}}]},
    ]}}
    asked = []

    def fetch(url, data):
        asked.append(url)
        return json.dumps(tileset).encode()

    chosen, best = g.select(g.ROOT, "KEY", LAT, LON, 350, fetch)
    urls = [u for u, _ in chosen]
    assert len(urls) == 1 and "fine.glb" in urls[0] and "key=KEY" in urls[0] and "session=abc" in urls[0]
    assert best == 0.5


def test_no_key_means_no_google():
    with pytest.raises(LookupError):
        g.gather(LAT, LON, 350, "/tmp/x", lambda u, d: b"", key="")
