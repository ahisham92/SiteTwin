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


import time
t0 = time.time()
folder = Path(tempfile.mkdtemp())
info = g.gather(lat, lon, radius, folder / "google", http_get, key=key)
print(f"near tiles {len(info['files'])}, far tiles {len(info['far_files'])}, finest {info['finest']:.2f}, "
      f"horizon {info['horizon']}, {time.time() - t0:.0f} s, "
      f"{sum(f.stat().st_size for f in (folder / 'google').iterdir()) / 1e6:.1f} MB")

import bpy  # noqa: E402
from sitetwin_worker import scene as S  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
S.M.reset()
manifest = {"lat": lat, "lon": lon, "radius": radius, "google": info}
gr = S.Ground(manifest, folder, (-25, -20, 25, 20))
t0 = time.time()
got = S.build_google(gr, manifest, folder, (-27, -22, 27, 22), ((0.0, -20.0), (0.0, -1.0)))
print("level", got, f"{time.time() - t0:.0f} s to build")
S.build_base(gr, min(-3.0, got[1] - 3.0), info["horizon"])
imgs = collections.Counter(f"{im.size[0]}x{im.size[1]}" for im in bpy.data.images)
print("texture sizes", dict(imgs.most_common(8)))
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
cam.location = (0, 0, 1000)
cam.rotation_euler = (0, 0, 0)
scene.render.filepath = str(out / "google-top.png")
bpy.ops.render.render(write_still=True)

cam_data.type = "PERSP"
cam_data.lens = 28
scene.render.resolution_x, scene.render.resolution_y = 1600, 900
d = 120
cam.location = (d * 0.7, -d * 0.7, 55)
direction = np.array([0, 0, level]) - np.array(cam.location)
import mathutils  # noqa: E402
cam.rotation_euler = mathutils.Vector(direction.tolist()).to_track_quat("-Z", "Y").to_euler()
scene.render.filepath = str(out / "google-oblique.png")
bpy.ops.render.render(write_still=True)
cam.location = (260, -260, 60)
direction = np.array([0, 0, 0]) - np.array(cam.location)
cam.rotation_euler = mathutils.Vector(direction.tolist()).to_track_quat("-Z", "Y").to_euler()
scene.render.filepath = str(out / "google-horizon.png")
bpy.ops.render.render(write_still=True)
print("rendered", sorted(p.name for p in out.iterdir()))
