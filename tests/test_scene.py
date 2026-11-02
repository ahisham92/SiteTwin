"""The Blender scene builds from a real IFC (skipped where Blender's bpy is not installed)."""

import json
from pathlib import Path

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")
pytest.importorskip("ifcopenshell")


def test_the_scene_has_the_building_the_site_and_the_plant(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
    from make_sample_ifc import build as make
    from sitetwin_worker import model, scene, sequence

    ifc = tmp_path / "b.ifc"
    make(str(ifc), 33.66, 35.42, storeys=2, nx=3, ny=3)
    env = tmp_path / "env"
    env.mkdir()
    np.save(env / "terrain.npy", np.full((65, 65), 12.0))
    manifest = {"lat": 33.66, "lon": 35.42, "radius": 200, "missing": {}, "credits": [],
                "terrain": {"file": "terrain.npy", "size": 65, "origin_height": 12.0},
                "osm": {"file": "osm.json"}}
    (env / "osm.json").write_text(json.dumps([
        {"kind": "building", "id": 1, "height": 12.0,
         "points": [[60, 60], [80, 60], [80, 75], [60, 75], [60, 60]]},
        {"kind": "road", "id": 2, "width": 8.0, "class": "primary", "points": [[-150, -60], [150, -60]]}]))
    els = model.load(ifc)
    tl = sequence.plan(els, seconds=10, fps=10)
    info = scene.build(els, tl, env, manifest, scene.Options(width=64, height=36, samples=1, fps=10), tmp_path)
    names = {o.name.split(".")[0] for o in bpy.data.objects}
    assert {"terrain", "site_pad", "excavation", "hoarding", "neighbours", "osm_asphalt", "camera",
            "tower_crane_mast", "piling_rig", "mixer_truck", "worker"} <= names
    assert sum(1 for o in bpy.data.objects if o.name.startswith("el_")) == len(els)
    assert info["pit_depth"] == 1.5
    # before the build nothing of the building is seen; at the end all of it is
    first = next(o for o in bpy.data.objects if o.name.startswith("el_"))
    bpy.context.scene.frame_set(1)
    assert first.hide_render
    bpy.context.scene.frame_set(tl.frames)
    assert not any(o.hide_render for o in bpy.data.objects if o.name.startswith("el_"))


def _glb_mesh(points_ecef, faces):
    """A plain (uncompressed) glb whose one node carries the Earth-centred offset, like Google's."""
    import struct
    from sitetwin_worker import google3d as g
    origin = points_ecef[0]
    local = np.array([(g.Y_UP_INV @ np.append(p - origin, 0))[:3] for p in points_ecef], dtype=np.float32)
    idx = np.array(faces, dtype=np.uint32).reshape(-1)
    binary = local.tobytes() + idx.tobytes()
    doc = {"asset": {"version": "2.0", "copyright": "Test data"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"mesh": 0, "translation": list((g.Y_UP_INV @ np.append(origin, 1))[:3])}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
           "buffers": [{"byteLength": len(binary)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": local.nbytes},
                           {"buffer": 0, "byteOffset": local.nbytes, "byteLength": idx.nbytes}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(local), "type": "VEC3",
                          "min": local.min(0).tolist(), "max": local.max(0).tolist()},
                         {"bufferView": 1, "componentType": 5125, "count": len(idx), "type": "SCALAR"}]}
    return g.write_glb(doc, binary)


def test_google_tiles_are_placed_cut_round_the_site_and_set_on_its_ground(tmp_path):
    from sitetwin_worker import google3d as g
    from sitetwin_worker import scene
    lat, lon = 33.66, 35.42
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # a 400 m square of ground at 25 m above the ellipsoid, cut into a 20 x 20 grid
    n, pts, faces = 21, [], []
    for i in range(n):
        for j in range(n):
            e, nn = -200 + 20 * j, -200 + 20 * i
            pts.append(g.ecef(lat + nn / 110900, lon + e / (111320 * np.cos(np.radians(lat))), 25.0))
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [(a, a + 1, a + n + 1), (a, a + n + 1, a + n)]
    env = tmp_path / "env"
    (env / "google").mkdir(parents=True)
    data, credit = g.to_local(_glb_mesh(np.array(pts), faces), np.eye(4), g.enu_matrix(lat, lon))
    (env / "google" / "tile_0000.glb").write_bytes(data)
    (env / "google" / "far_0000.glb").write_bytes(data)          # the horizon ring, same ground
    manifest = {"lat": lat, "lon": lon, "radius": 180,
                "google": {"folder": "google", "files": ["tile_0000.glb"], "far_files": ["far_0000.glb"],
                           "radius": 120, "horizon": 180}}
    gr = scene.Ground(manifest, env, (-15, -15, 15, 15))
    level, lowest = scene.build_google(gr, manifest, env, (-20, -20, 20, 20))
    assert level == pytest.approx(25.0, abs=0.3)
    assert lowest == pytest.approx(0.0, abs=0.4)
    google = [o for o in bpy.data.objects if o.name.startswith("google_")]
    assert google
    zs = [v.co.z for o in google for v in o.data.vertices]
    assert max(abs(z) for z in zs) < 0.4                        # its ground now meets the site's
    for o in google:
        for p in o.data.polygons:
            c = p.center
            assert not (-20 <= c.x <= 20 and -20 <= c.y <= 20)    # the site is cut out
            assert np.hypot(c.x, c.y) <= 181                      # and nothing beyond the horizon
            if o.name.startswith("google_far") or "far" in o.name:
                assert np.hypot(c.x, c.y) > 120                   # the ring starts where the circle ends


def test_a_real_model_is_stood_on_the_ground_scaled_and_shared_between_copies(tmp_path):
    from sitetwin_worker import equipment as E
    bpy.ops.wm.read_factory_settings(use_empty=True)
    E._library.clear()
    # a 2 x 1 x 1 box floating off-centre, in glTF's y-up axes
    pts = np.array([(x, y, z) for x in (5, 7) for y in (3, 4) for z in (-1, 0)], float)
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    path = tmp_path / "mixer.glb"
    path.write_bytes(_glb_mesh(pts, faces))
    E.MODELS.clear()
    E.MODELS["mixer_truck"] = str(path)
    try:
        a, b = E.mixer_truck(), E.mixer_truck()
        parts = [c for c in a.children if c.type == "MESH"]
        assert parts and parts[0].data is [c for c in b.children if c.type == "MESH"][0].data
        co = np.array([v.co[:] for v in parts[0].data.vertices])
        lo, hi = co.min(0), co.max(0)
        assert max(hi[0] - lo[0], hi[1] - lo[1]) == pytest.approx(9.0, abs=0.01)   # a real mixer's length
        assert lo[2] == pytest.approx(0.0, abs=1e-4)                              # on the ground
        assert abs(lo[0] + hi[0]) < 1e-3 and abs(lo[1] + hi[1]) < 1e-3            # centred
    finally:
        E.MODELS.clear()
        E._library.clear()


def test_props_stay_inside_the_hoarding_off_the_building_and_out_of_the_lane():
    from sitetwin_worker import logistics as L
    from sitetwin_worker import scene
    site = (-40, -35, 40, 35)
    fp = (-20, -15, 20, 15)
    pts = {"gate": (0.0, -35.0), "out": (0.0, -1.0), "along": (1.0, 0.0), "laydown": (-14.0, -30.0)}
    lane = L.Lane(far=(0.0, -90.0), bay=(0.0, -29.0))
    spots = scene.prop_spots(site, fp, pts, lane)
    kinds = {k for k, *_ in spots}
    assert {"prop_barrier", "prop_toilet", "prop_skip", "prop_bricks", "prop_light"} <= kinds
    for kind, x, y, _ in spots:
        assert not (fp[0] - 2 < x < fp[2] + 2 and fp[1] - 2 < y < fp[3] + 2)
        if kind == "prop_barrier":
            assert y < site[1] and abs(x) > 5                 # lining the road, clear of the trucks
        else:
            assert site[0] < x < site[2] and site[1] < y < site[3]
            assert lane.keeps_clear((x, y), margin=1.0)
