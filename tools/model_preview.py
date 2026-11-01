"""Downloads the catalogue's models and renders each one as the scene would stand it up, beside
a red arrow pointing +X (the way its front must face) and a 1.75 m figure, so yaw and size can be
checked. Pictures go to the folder given."""
import math, os, sys, tempfile
from pathlib import Path

from sitetwin_worker import assets
from sitetwin_worker import equipment as E

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
files, credits = assets.prepare(Path(tempfile.mkdtemp()))
print("models", {k: Path(v).name for k, v in files.items()})

import bpy, mathutils  # noqa: E402
from sitetwin_worker import materials as M  # noqa: E402

for kind in files:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    M.reset()
    E._library.clear()
    E.MODELS.clear()
    E.MODELS.update(files)
    obj = getattr(E, kind)() if kind != "worker" else E.worker()
    parts = [c for c in obj.children if c.type == "MESH"]
    if not parts:
        print(kind, "fell back to the built-in shape")
        continue
    faces = sum(len(p.data.polygons) for p in parts)
    E.MODELS.clear()
    man = E.worker()                                   # the built-in 1.75 m figure for scale
    man.location = (0, -4, 0)
    arrow = E._build("arrow", [("box", "drum_orange", (3, 4, 0.1), (6, 0.3, 0.2), 0)])
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 16
    sc.render.resolution_x, sc.render.resolution_y = 1200, 700
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.8, 0.85, 0.9, 1)
    sc.world = w
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3
    sun.rotation_euler = (0.6, 0.2, 0.8)
    sc.collection.objects.link(sun)
    co = [p.matrix_world @ mathutils.Vector(v) for p in parts for v in p.bound_box]
    size = max(max(c[i] for c in co) - min(c[i] for c in co) for i in range(3))
    cam = bpy.data.objects.new("c", bpy.data.cameras.new("c"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    d = max(8.0, size * 1.6)
    cam.location = (d * 0.55, -d * 0.9, d * 0.45)
    target = mathutils.Vector((0, 0, size * 0.3))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = str(out / f"model-{kind}.png")
    bpy.ops.render.render(write_still=True)
    print(kind, "faces", faces, "size", round(size, 1))
