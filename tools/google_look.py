"""Downloads Google's tiles round a place, imports them into Blender the way a film does, and
renders two pictures of just them (from above, and from the film's usual angle), so gaps and
blur can be seen. Prints what failed and how detailed the textures are. No key is printed."""
import collections, math, os, sys, tempfile, traceback
from pathlib import Path

import numpy as np

from sitetwin_worker import google3d as g
from sitetwin_worker.environment import http_get

key = os.environ["SITETWIN_GOOGLE_KEY"].strip()
lat, lon, radius = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
out = Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)


def clean(s):
    return str(s).replace(key, "…")


tiles, best = g.select(g.ROOT, key, lat, lon, radius, http_get)
print("tiles chosen", len(tiles), "finest", best)
errs = collections.Counter()
folder = Path(tempfile.mkdtemp())
local = g.enu_matrix(lat, lon)
files = []
for n, (url, T) in enumerate(tiles):
    try:
        data = http_get(url)
    except Exception as exc:
        errs["download " + type(exc).__name__] += 1
        continue
    try:
        glb, _ = g.to_local(data, T, local)
    except Exception as exc:
        errs["to_local " + type(exc).__name__ + ": " + clean(exc)[:80]] += 1
        continue
    p = folder / f"tile_{n:04d}.glb"
    p.write_bytes(glb)
    files.append(p)
print("written", len(files), "errors", dict(errs))

import bpy  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
fails = collections.Counter()
per_tile = []
for p in files:
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=str(p))
    except Exception as exc:
        fails[clean(exc).strip().splitlines()[-1][:120]] += 1
        continue
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        if o.type == "MESH":
            mw = o.matrix_world.copy()
            o.parent = None
            o.data.transform(mw)
            o.matrix_world.identity()
    for o in new:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    ms = [o for o in new if o.name in bpy.data.objects and o.type == "MESH"]
    if not ms:
        fails["imported but no mesh"] += 1
        continue
    co = np.concatenate([np.array([v.co[:] for v in o.data.vertices]) for o in ms])
    per_tile.append((p.name, co.min(0), co.max(0), sum(len(o.data.polygons) for o in ms)))
print("import failures", dict(fails))
imgs = collections.Counter(f"{im.size[0]}x{im.size[1]}" for im in bpy.data.images)
print("texture sizes", dict(imgs.most_common(8)))
far = [t for t in per_tile if math.hypot(*(t[1][:2] + t[2][:2]) / 2) > radius * 1.5]
print("tiles placed far away", len(far), [(t[0], t[1].round(0).tolist()) for t in far[:5]])
zs = np.array([[t[1][2], t[2][2]] for t in per_tile])
if len(zs):
    print("z range of tiles: min", zs[:, 0].min().round(1), "median low", np.median(zs[:, 0]).round(1),
          "max", zs[:, 1].max().round(1))
    span = np.array([t[2][:2] - t[1][:2] for t in per_tile])
    print("tile footprint m: median", np.median(span, 0).round(0).tolist(), "faces median",
          int(np.median([t[3] for t in per_tile])))
    # where the tiles cover, on a 10 m grid inside the radius
    cells = int(radius * 2 / 10)
    cover = np.zeros((cells, cells), bool)
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        for poly in o.data.polygons:
            c = poly.center
            i, j = int((c.x + radius) / 10), int((c.y + radius) / 10)
            if 0 <= i < cells and 0 <= j < cells:
                cover[j, i] = True
    yy, xx = np.mgrid[0:cells, 0:cells]
    inside = np.hypot((xx + 0.5) * 10 - radius, (yy + 0.5) * 10 - radius) < radius - 10
    print(f"coverage inside the circle: {cover[inside].mean() * 100:.1f}% of 10 m cells")
    for row in cover[::-2]:
        print("".join("#" if c else "." for c in row[::2]))
    level = float(np.median(zs[:, 0]))
else:
    level = 0.0

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 8
scene.render.resolution_x, scene.render.resolution_y = 1600, 1600
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (1, 0, 1, 1)   # gaps show magenta
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
scene.world = world
for o in bpy.data.objects:
    for slot in o.material_slots:
        m = slot.material
        if m and m.use_nodes:
            nodes = m.node_tree.nodes
            bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf and bsdf.inputs["Base Color"].links:
                em = nodes.new("ShaderNodeEmission")
                m.node_tree.links.new(bsdf.inputs["Base Color"].links[0].from_socket, em.inputs[0])
                outn = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
                m.node_tree.links.new(em.outputs[0], outn.inputs[0])

cam_data = bpy.data.cameras.new("c")
cam = bpy.data.objects.new("c", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
cam_data.type = "ORTHO"
cam_data.ortho_scale = radius * 2
cam_data.clip_end = 5000
cam.location = (0, 0, level + 1000)
cam.rotation_euler = (0, 0, 0)
scene.render.filepath = str(out / "google-top.png")
bpy.ops.render.render(write_still=True)

cam_data.type = "PERSP"
cam_data.lens = 28
scene.render.resolution_x, scene.render.resolution_y = 1600, 900
d = 110
cam.location = (d * 0.7, -d * 0.7, level + 70)
direction = np.array([0, 0, level]) - np.array(cam.location)
import mathutils  # noqa: E402
cam.rotation_euler = mathutils.Vector(direction.tolist()).to_track_quat("-Z", "Y").to_euler()
scene.render.filepath = str(out / "google-oblique.png")
bpy.ops.render.render(write_still=True)
print("rendered", sorted(p.name for p in out.iterdir()))
