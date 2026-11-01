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
