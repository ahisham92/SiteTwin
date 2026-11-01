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
