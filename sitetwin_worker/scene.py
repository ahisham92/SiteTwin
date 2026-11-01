"""The whole film as one Blender scene: the real ground, the neighbours, the site,
the building going up element by element, and the plant and crews doing it.

Everything here runs inside Blender (``bpy``). It reads only what ``environment``,
``model`` and ``sequence`` produced, so the same scene builds on a laptop, a
workstation or a cloud GPU.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import equipment as E
from . import logistics as L
from . import materials as M
from .geo import sun_position
from .model import Element, footprint
from .sequence import Timeline, phase_of, plant_windows


@dataclass
class Options:
    width: int = 1920
    height: int = 1080
    samples: int = 64
    fps: int = 30
    when: dt.datetime | None = None       # local solar time of the shoot; default today 10:30
    site_margin: float = 12.0
    camera_turn: float = 110.0            # degrees the camera travels round the site


# ---- small helpers ----------------------------------------------------------------------

def _mesh(name: str, verts, faces, material=None, uvs=None):
    import bpy
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(map(int, f)) for f in faces])
    if uvs is not None:
        uv = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li in poly.loop_indices:
                vi = me.loops[li].vertex_index
                uv.data[li].uv = uvs[vi]
    me.update()
    if material is not None:
        me.materials.append(material)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _visible(obj, start: int | None, end: int | None, total: int) -> None:
    """Keyframe hide_render/hide_viewport so obj is seen only in [start, end)."""
    def key(frame: int, hidden: bool):
        obj.hide_render = hidden
        obj.hide_viewport = hidden
        obj.keyframe_insert("hide_render", frame=frame)
        obj.keyframe_insert("hide_viewport", frame=frame)
    key(0, start is not None and start > 0)
    if start is not None and start > 0:
        key(start, False)
    if end is not None and end < total:
        key(end, True)
    for child in obj.children:
        _visible(child, start, end, total)


def _ease(obj, data_path: str) -> None:
    ad = obj.animation_data
    if not ad or not ad.action:
        return
    try:
        curves = ad.action.fcurves
    except AttributeError:                   # Blender 5: layered actions
        curves = [fc for layer in ad.action.layers for strip in layer.strips
                  for bag in strip.channelbags for fc in bag.fcurves]
    for fc in curves:
        if fc.data_path == data_path:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"
                kp.easing = "AUTO"


def _curves(obj):
    ad = obj.animation_data
    if not ad or not ad.action:
        return []
    try:
        return list(ad.action.fcurves)
    except AttributeError:                   # Blender 5: layered actions
        return [fc for layer in ad.action.layers for strip in layer.strips
                for bag in strip.channelbags for fc in bag.fcurves]


def _linear(obj) -> None:
    """Densely keyed motion (worked out frame by frame) plays straight between its keys."""
    for fc in _curves(obj):
        if not fc.data_path.startswith("hide"):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


def _constant(obj) -> None:
    ad = obj.animation_data
    if not ad or not ad.action:
        return
    try:
        curves = ad.action.fcurves
    except AttributeError:
        curves = [fc for layer in ad.action.layers for strip in layer.strips
                  for bag in strip.channelbags for fc in bag.fcurves]
    for fc in curves:
        if fc.data_path.startswith("hide"):
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


def _all_constant_hides() -> None:
    import bpy
    for o in bpy.data.objects:
        _constant(o)


# ---- ground -----------------------------------------------------------------------------

class Ground:
    """The terrain heights, flattened to a graded pad under the site."""

    def __init__(self, manifest: dict[str, Any], folder: Path, site: tuple[float, float, float, float]):
        r = float(manifest["radius"])
        self.radius = r
        if "terrain" in manifest:
            raw = np.load(folder / manifest["terrain"]["file"])
            origin = manifest["terrain"]["origin_height"]
        else:
            raw = np.zeros((65, 65))
            origin = 0.0
        # The tiles put the sea at or below 0. The model's zero is its ground floor,
        # so a site at the water's edge still stands a couple of metres above it.
        self.has_sea = bool((raw < 0.25).any())
        self.sea_level = min(0.0 - origin, -2.0)
        h = raw - origin
        if self.has_sea:
            h = np.where(raw < 0.25, np.minimum(h, self.sea_level - 1.5), h)
            h = np.maximum(h, self.sea_level - 8.0)
        n = h.shape[0]
        self.n = n
        self.xs = np.linspace(-r, r, n)
        self.ys = np.linspace(r, -r, n)
        gx, gy = np.meshgrid(self.xs, self.ys)
        x0, y0, x1, y1 = site
        dx = np.maximum(np.maximum(x0 - gx, gx - x1), 0)
        dy = np.maximum(np.maximum(y0 - gy, gy - y1), 0)
        d = np.hypot(dx, dy)
        w = np.clip(1 - d / 30.0, 0, 1)
        w = w * w * (3 - 2 * w)
        self.h = h * (1 - w)
        self.site = site

    def at(self, x: float, y: float) -> float:
        fx = (x + self.radius) / (2 * self.radius) * (self.n - 1)
        fy = (self.radius - y) / (2 * self.radius) * (self.n - 1)
        i0, j0 = int(np.clip(math.floor(fy), 0, self.n - 2)), int(np.clip(math.floor(fx), 0, self.n - 2))
        ty, tx = fy - i0, fx - j0
        h = self.h
        return float(h[i0, j0] * (1 - tx) * (1 - ty) + h[i0, j0 + 1] * tx * (1 - ty)
                     + h[i0 + 1, j0] * (1 - tx) * ty + h[i0 + 1, j0 + 1] * tx * ty)


def build_terrain(g: Ground, manifest: dict[str, Any], folder: Path, sink: float = 0.0):
    """The ground from the terrain heights, draped with the imagery. ``sink`` lowers it away from
    the site, under Google's photo-real ground where that is drawn."""
    n, r = g.n, g.radius
    verts, uvs = [], []
    sx0, sy0, sx1, sy1 = g.site
    for i in range(n):
        for j in range(n):
            x, y = g.xs[j], g.ys[i]
            far = math.hypot(max(sx0 - x, x - sx1, 0.0), max(sy0 - y, y - sy1, 0.0)) > 30.0
            verts.append((x, y, g.h[i, j] - (sink if far and math.hypot(x, y) < r * 0.97 else 0.0)))
            uvs.append(((x + r) / (2 * r), (y + r) / (2 * r)))
    x0, y0, x1, y1 = g.site
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j
            xs = (g.xs[j], g.xs[j + 1])
            ys = (g.ys[i], g.ys[i + 1])
            inside = all(x0 <= x <= x1 for x in xs) and all(y0 <= y <= y1 for y in ys)
            if not inside:
                faces.append((a, d, c, b))
    mat = (M.imagery(str(folder / manifest["imagery"]["file"])) if "imagery" in manifest
           else M.get("ground"))
    obj = _mesh("terrain", verts, faces, mat, uvs)
    for p in obj.data.polygons:
        p.use_smooth = True
    if g.has_sea:
        # the sea everywhere except over the site, where the excavation must stay dry
        s = r * 6
        z = g.sea_level
        v = [(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z), (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        _mesh("sea", v, [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)], M.get("water"))
    # beyond the fetched square: a skirt of plain ground (or sea) out to the horizon
    s = r * 6
    z = -0.6 if not g.has_sea else g.sea_level - 6.5
    _mesh("skirt", [(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)], [(0, 1, 2, 3)], M.get("ground"))
    return obj


def build_site(g: Ground, pit: tuple[float, float, float, float], depth: float, gate_side: str,
               graded_track: bool = False):
    """Gravel pad over the site with an excavation for the foundations, hoarding with a gate,
    cabins and a laydown area. Returns useful points (gate, laydown, parking)."""
    x0, y0, x1, y1 = g.site
    e = 2 * g.radius / (g.n - 1)                        # one terrain cell
    if graded_track:
        e += 4.0                                        # and under Google's cut edge
    X0, Y0, X1, Y1 = x0 - e, y0 - e, x1 + e, y1 + e     # the pad tucks under the terrain edge
    px0, py0, px1, py1 = pit
    slope = max(depth * 0.6, 0.01)
    ox0, oy0, ox1, oy1 = px0 - slope, py0 - slope, px1 + slope, py1 + slope
    z = 0.02
    verts = [(X0, Y0, z), (X1, Y0, z), (X1, Y1, z), (X0, Y1, z),
             (ox0, oy0, z), (ox1, oy0, z), (ox1, oy1, z), (ox0, oy1, z)]
    faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    _mesh("site_pad", verts, faces, M.get("gravel"))
    if depth > 0.05:
        zb = -depth
        v = [(ox0, oy0, z), (ox1, oy0, z), (ox1, oy1, z), (ox0, oy1, z),
             (px0, py0, zb), (px1, py0, zb), (px1, py1, zb), (px0, py1, zb)]
        f = [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        _mesh("excavation", v, f, M.get("soil"))

    # hoarding: 2.4 m panels with posts, a 10 m gate on the access side
    parts = []
    gate_w = 10.0
    sides = {"south": ((x0, y0), (x1, y0)), "east": ((x1, y0), (x1, y1)),
             "north": ((x1, y1), (x0, y1)), "west": ((x0, y1), (x0, y0))}
    gate = None
    for name, ((ax, ay), (bx, by)) in sides.items():
        length = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / length, (by - ay) / length
        ang = math.atan2(uy, ux)
        mid = length / 2
        t = 0.0
        while t < length - 0.01:
            l = min(3.5, length - t)
            c = t + l / 2
            if name == gate_side and abs(c - mid) < gate_w / 2:
                t += l
                continue
            cx, cy = ax + ux * c, ay + uy * c
            parts.append(("box", "hoarding", (cx, cy, 1.2 + 0.02), (l - 0.02, 0.05, 2.4), ang))
            parts.append(("box", "steel", (ax + ux * t, ay + uy * t, 1.25), (0.08, 0.08, 2.5), ang))
            t += l
        if name == gate_side:
            gate = (ax + ux * mid, ay + uy * mid, ang)
    E._build("hoarding", parts)

    # outward normal of the gate side
    normals = {"south": (0, -1), "east": (1, 0), "north": (0, 1), "west": (-1, 0)}
    nx, ny = normals[gate_side]
    gx, gy, gang = gate
    # cabins and laydown just inside, to one side of the gate
    tx, ty = -ny, nx                                       # along the fence
    cab = E.site_cabin()
    cab.location = (gx + tx * 14 - nx * 3, gy + ty * 14 - ny * 3, 0.02)
    cab.rotation_euler = (0, 0, gang)
    cab2 = E.site_cabin()
    cab2.location = (gx + tx * 14 - nx * 3, gy + ty * 14 - ny * 3, 2.72)
    cab2.rotation_euler = (0, 0, gang)
    lay = (gx - tx * 14 - nx * 5, gy - ty * 14 - ny * 5)
    stacks = []
    for k in range(4):
        sx, sy = lay[0] + tx * (k * 3.2 - 4.8), lay[1] + ty * (k * 3.2 - 4.8)
        mat = "formwork" if k % 2 == 0 else "rebar"
        stacks.append(("box", mat, (sx, sy, 0.45 if k % 2 == 0 else 0.25), (2.4, 6.0, 0.9 if k % 2 == 0 else 0.5), gang))
    E._build("laydown", stacks)
    # an access track from the gate out to the edge of the scene
    # (on Google's ground the road out is graded flat for 60 m, and the real roads go on)
    track_len = 62.0 if graded_track else g.radius * 0.9
    tv = []
    tf = []
    steps = 24
    for k in range(steps + 1):
        d = k / steps * track_len
        cx, cy = gx + nx * d, gy + ny * d
        zz = (g.at(cx, cy) if d > 2 and not graded_track else 0.0) + 0.06
        tv += [(cx + tx * 3.5, cy + ty * 3.5, zz), (cx - tx * 3.5, cy - ty * 3.5, zz)]
        if k:
            a = 2 * (k - 1)
            tf.append((a, a + 1, a + 3, a + 2))
    _mesh("access_track", tv, tf, M.get("asphalt"))
    return {"gate": (gx, gy), "out": (nx, ny), "along": (tx, ty), "laydown": lay}


def build_google(g: Ground, manifest: dict[str, Any], folder: Path,
                 keep_out: tuple[float, float, float, float],
                 access: tuple[tuple[float, float], tuple[float, float]] | None = None) -> tuple[float, float] | None:
    """Google's photo-real surroundings: the detailed tiles round the site and the coarse ring out
    to the horizon, the site itself cut out (it is rebuilt by the sequence), and everything
    lowered so the ground round the site meets the site's level. The ground just outside the
    hoarding, and the access road out of the gate (``access``: gate point, outward direction),
    are graded to the site's level, as a real site's are. Returns (how far it was lowered, the
    lowest ground left), or None if there are none."""
    info = manifest.get("google")
    if not info:
        return None
    import bmesh
    import bpy
    sub = folder / info["folder"]

    def load(names: list[str]) -> list:
        before = set(bpy.data.objects)
        for name in names:
            try:
                bpy.ops.import_scene.gltf(filepath=str(sub / name))
            except Exception:                               # noqa: BLE001 - one bad tile is a small gap
                continue
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH"]
        for o in meshes:                                    # bake each tile's placement into it
            mw = o.matrix_world.copy()
            o.parent = None
            o.data.transform(mw)
            o.matrix_world.identity()
        for o in new:
            if o.type != "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
        return meshes

    near = load(info["files"])
    if not near:
        return None
    far = load(info.get("far_files", []))

    kx0, ky0, kx1, ky1 = keep_out

    def from_site(x: float, y: float) -> float:
        return math.hypot(max(kx0 - x, x - kx1, 0.0), max(ky0 - y, y - ky1, 0.0))

    # The ground level around the site: low points in a ring 8 to 45 m outside it.
    ring = [v.co.z for o in near for v in o.data.vertices if 8.0 <= from_site(v.co.x, v.co.y) <= 45.0]
    level = float(np.percentile(ring, 12)) if ring else 0.0

    def graded(x: float, y: float, z: float) -> float:
        """z (already lowered) with the grading round the site and along the access road."""
        w = 0.0
        if abs(z) < 4.0:                                    # ground, not roofs
            d = from_site(x, y)
            w = 1.0 - smooth01(d / 12.0)
        if access is not None:
            (gx, gy), (nx, ny) = access
            along = (x - gx) * nx + (y - gy) * ny
            across = abs(-(x - gx) * ny + (y - gy) * nx)
            if -2.0 <= along <= 70.0:
                wa = 1.0 - smooth01((across - 5.0) / 7.0) if across > 5.0 else 1.0
                wa *= 1.0 - smooth01((along - 60.0) / 10.0) if along > 60.0 else 1.0
                w = max(w, wa)
        return z * (1.0 - w)

    inner = float(info.get("radius", g.radius))
    outer = float(info.get("horizon", inner))
    lowest = math.inf
    for group, keep in ((near, lambda c: math.hypot(c.x, c.y) <= inner),
                        (far, lambda c: inner < math.hypot(c.x, c.y) <= outer)):
        for o in group:
            bm = bmesh.new()
            bm.from_mesh(o.data)
            gone = [f for f in bm.faces
                    if not keep(c := f.calc_center_median()) or (kx0 <= c.x <= kx1 and ky0 <= c.y <= ky1)]
            bmesh.ops.delete(bm, geom=gone, context="FACES")
            for v in bm.verts:
                x, y, z = v.co
                v.co.z = graded(x, y, z - level - 0.05)
                lowest = min(lowest, v.co.z)
            bm.to_mesh(o.data)
            bm.free()
            o.name = "google_" + o.name
            # photogrammetry carries its own daylight: no shine, a little of its colour as light
            for slot in o.material_slots:
                mat = slot.material
                if mat and mat.use_nodes:
                    for node in mat.node_tree.nodes:
                        if node.type == "BSDF_PRINCIPLED":
                            node.inputs["Roughness"].default_value = 1.0
                            if "Specular IOR Level" in node.inputs:
                                node.inputs["Specular IOR Level"].default_value = 0.1
                            base = node.inputs["Base Color"]
                            if base.links and "Emission Color" in node.inputs:
                                mat.node_tree.links.new(base.links[0].from_socket, node.inputs["Emission Color"])
                                node.inputs["Emission Strength"].default_value = 0.12
    return level, (lowest if lowest < math.inf else 0.0)


def smooth01(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def build_base(g: Ground, z: float, horizon: float):
    """Under Google's mesh: plain ground far below it (so it never shows through where the
    mesh dips), and soil walls round the site pad down to it, so no gap under the cut shows."""
    s = max(horizon, g.radius) * 6
    if g.has_sea:                                         # beyond Google's ring the sea goes on
        zs = g.sea_level - 1.0
        _mesh("skirt", [(-s, -s, zs), (s, -s, zs), (s, s, zs), (-s, s, zs)], [(0, 1, 2, 3)], M.get("water"))
        z = min(z, zs - 0.5)
    else:
        _mesh("skirt", [(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)], [(0, 1, 2, 3)], M.get("ground"))
    x0, y0, x1, y1 = g.site
    e = 2 * g.radius / (g.n - 1) + 4.0
    X0, Y0, X1, Y1 = x0 - e, y0 - e, x1 + e, y1 + e
    top = 0.0
    v = [(X0, Y0, top), (X1, Y0, top), (X1, Y1, top), (X0, Y1, top),
         (X0, Y0, z), (X1, Y0, z), (X1, Y1, z), (X0, Y1, z)]
    _mesh("site_walls", v, [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)], M.get("soil"))


def build_neighbours(g: Ground, manifest: dict[str, Any], folder: Path, keep_out: tuple[float, float, float, float]):
    if "osm" not in manifest:
        return
    import bmesh
    import bpy
    feats = json.loads((folder / manifest["osm"]["file"]).read_text())
    kx0, ky0, kx1, ky1 = keep_out
    r = g.radius

    def inside_scene(pts):
        return any(abs(x) < r and abs(y) < r for x, y in pts)

    bm_b = bmesh.new()
    bm_flat: dict[str, Any] = {}
    for f in feats:
        pts = f["points"]
        if not inside_scene(pts):
            continue
        if f["kind"] == "building":
            ring = pts[:-1] if pts[0] == pts[-1] else pts
            cx = sum(p[0] for p in ring) / len(ring)
            cy = sum(p[1] for p in ring) / len(ring)
            if kx0 <= cx <= kx1 and ky0 <= cy <= ky1:
                continue                     # the site is cleared
            base = min(g.at(x, y) for x, y in ring) - 0.3
            vs = [bm_b.verts.new((x, y, base)) for x, y in ring]
            try:
                face = bm_b.faces.new(vs)
            except ValueError:
                continue
            if face.normal.z < 0:
                face.normal_flip()
            ext = bmesh.ops.extrude_face_region(bm_b, geom=[face])
            top = [v for v in ext["geom"] if isinstance(v, bmesh.types.BMVert)]
            bmesh.ops.translate(bm_b, verts=top, vec=(0, 0, f["height"] + 0.3))
            # darker glazed towers, rendered plaster below
            for fc in bm_b.faces:
                pass
        elif f["kind"] in ("road", "rail", "river", "pier_line"):
            mat = {"road": "asphalt", "rail": "rail", "river": "water", "pier_line": "concrete"}[f["kind"]]
            bm = bm_flat.setdefault(mat, bmesh.new())
            w = f.get("width", 6.0) / 2
            lift = {"asphalt": 0.08, "rail": 0.1, "water": 0.03, "concrete": 0.4}[mat]
            prev = None
            for a, b in zip(pts[:-1], pts[1:]):
                dx, dy = b[0] - a[0], b[1] - a[1]
                L = math.hypot(dx, dy) or 1.0
                ox, oy = -dy / L * w, dx / L * w
                if kx0 <= (a[0] + b[0]) / 2 <= kx1 and ky0 <= (a[1] + b[1]) / 2 <= ky1:
                    continue
                quad = [(a[0] + ox, a[1] + oy), (b[0] + ox, b[1] + oy), (b[0] - ox, b[1] - oy), (a[0] - ox, a[1] - oy)]
                vs = [bm.verts.new((x, y, g.at(x, y) + lift)) for x, y in quad]
                try:
                    bm.faces.new(vs)
                except ValueError:
                    pass
        elif f["kind"] in ("water", "green", "sand", "pier"):
            mat = {"water": "water", "green": "grass", "sand": "sand", "pier": "concrete"}[f["kind"]]
            bm = bm_flat.setdefault(mat, bmesh.new())
            ring = pts[:-1]
            lift = {"water": 0.02, "grass": 0.05, "sand": 0.04, "concrete": 0.6}[mat]
            z = (g.sea_level + 0.01) if mat == "water" else None
            vs = [bm.verts.new((x, y, (z if z is not None else g.at(x, y) + lift))) for x, y in ring]
            try:
                face = bm.faces.new(vs)
                if face.normal.z < 0:
                    face.normal_flip()
            except ValueError:
                pass
    me = bpy.data.meshes.new("neighbours")
    bm_b.to_mesh(me)
    bm_b.free()
    me.materials.append(M.get("facade"))
    ob = bpy.data.objects.new("neighbours", me)
    bpy.context.scene.collection.objects.link(ob)
    for mat, bm in bm_flat.items():
        me = bpy.data.meshes.new("osm_" + mat)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(M.get(mat))
        o = bpy.data.objects.new("osm_" + mat, me)
        bpy.context.scene.collection.objects.link(o)


# ---- the building ------------------------------------------------------------------------

def build_model(elements: list[Element], tl: Timeline) -> dict[str, Any]:
    objs = {}
    for e in elements:
        a, b, how = tl.appear.get(e.guid, (tl.build_start, tl.build_start + 4, "fit"))
        lo, hi = e.lo, e.hi
        base = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]], dtype=np.float32)
        obj = _mesh("el_" + e.guid, e.verts - base, e.faces, M.for_element(e.kind))
        obj.location = tuple(map(float, base))
        if how == "grow":
            obj.scale = (1, 1, 0.02)
            obj.keyframe_insert("scale", frame=a)
            obj.scale = (1, 1, 1)
            obj.keyframe_insert("scale", frame=b)
        elif how in ("lift", "crane"):
            pass                                  # carried in by the crane (build_plant keys it)
        else:
            obj.scale = (0.9, 0.9, 0.9)
            obj.keyframe_insert("scale", frame=a)
            obj.scale = (1, 1, 1)
            obj.keyframe_insert("scale", frame=b)
        _visible(obj, a, None, tl.frames)
        objs[e.guid] = obj
    return objs


# ---- plant and people --------------------------------------------------------------------

def _unwrap(prev: float, ang: float) -> float:
    while ang - prev > math.pi:
        ang -= 2 * math.pi
    while ang - prev < -math.pi:
        ang += 2 * math.pi
    return ang


def plan_site(elements: list[Element], tl: Timeline, fp, top: float, pts: dict[str, Any],
              margin: float) -> dict[str, Any]:
    """Where the crane stands and what it lifts when, the truck lane and its visits (logistics.py).
    Changes tl.appear: pieces the crane lifts become "crane", the rest of the lifted kinds "fit"."""
    windows = plant_windows(tl)
    x0, y0, x1, y1 = fp
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    nx, ny = pts["out"]
    tx, ty = pts["along"]
    gx, gy = pts["gate"]
    out: dict[str, Any] = {"windows": windows, "crane": None, "lifts": []}
    if "tower_crane" in windows:
        s, e = windows["tower_crane"][0]
        half_w, half_d = (x1 - x0) / 2, (y1 - y0) / 2
        off = (abs(nx) * half_w + abs(ny) * half_d) + 6.0
        side = 0.25 * (abs(ny) * half_w + abs(nx) * half_d)
        crx, cry = cx - nx * off + tx * side, cy - ny * off + ty * side
        lay = pts["laydown"]
        reach = max(math.hypot(px - crx, py - cry) for px in (x0, x1) for py in (y0, y1))
        reach = max(reach, math.hypot(lay[0] - crx, lay[1] - cry)) + 4
        crane = L.Crane(crx, cry, max(top + 12.0, 28.0), min(75.0, max(30.0, reach)))
        places = {}
        for el in elements:
            places[el.guid] = (float((el.lo[0] + el.hi[0]) / 2), float((el.lo[1] + el.hi[1]) / 2),
                               float(el.lo[2]), float(el.hi[2] - el.lo[2]))
        crane.lifts = L.plan_lifts(tl.steps, tl.appear, places, (lay[0], lay[1], 0.9), tl.fps, (s, e))
        out["crane"] = crane
        out["crane_window"] = (s, e)
    else:
        for g, (a, b, how) in list(tl.appear.items()):
            if how == "lift":
                tl.appear[g] = (a, b, "fit")
    bay_in = max(4.0, margin - 6.0)
    lane = L.Lane(far=(gx + nx * 55, gy + ny * 55), bay=(gx - nx * bay_in, gy - ny * bay_in))
    out["lane"] = lane
    out["visits"] = L.plan_visits(windows, lane, tl.fps)
    out["pump_at"] = (lane.bay[0] + tx * 5.5, lane.bay[1] + ty * 5.5)
    return out


def build_plant(elements: list[Element], tl: Timeline, fp, top: float, pts: dict[str, Any],
                pit_depth: float, rng: random.Random, plan: dict[str, Any] | None = None) -> None:
    by_guid = {e.guid: e for e in elements}
    plan = plan or {}
    windows = plan.get("windows") or plant_windows(tl)
    x0, y0, x1, y1 = fp
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    nx, ny = pts["out"]
    tx, ty = pts["along"]
    gx, gy = pts["gate"]

    # --- tower crane: one lift at a time, the load on the hook from the laydown to its place
    crane = plan.get("crane")
    if crane is not None:
        s, e = plan["crane_window"]
        mast, slew, trolley, hook = E.tower_crane(crane.height, crane.jib)
        mast.location = (crane.x, crane.y, 0.0)
        _visible(mast, s, e, tl.frames)
        if trolley is not None:
            lines, block = hook
            slings = E.slings()
            slings.parent = block
            slings.location = (0, 0, -1.2)
            objs = {o.name[3:]: o for o in __import__("bpy").data.objects if o.name.startswith("el_")}
            tops = sorted((tl.appear[g][1], float(by_guid[g].hi[2])) for g in tl.appear if g in by_guid)
            prev_a = None
            carrying = None
            first = crane.lifts[0].start if crane.lifts else s
            last = crane.lifts[-1].end + 2 if crane.lifts else s + 1
            built_top, k = 0.0, 0
            for f in range(max(s, first - 4), min(e, last) + 1, 2):
                while k < len(tops) and tops[k][0] <= f:
                    built_top = max(built_top, tops[k][1])
                    k += 1
                ang, rad, tip, lift, base_z = crane.pose(f, built_top)
                ang = ang if prev_a is None else L.unwrap(prev_a, ang)
                prev_a = ang
                slew.rotation_euler = (0, 0, ang)
                slew.keyframe_insert("rotation_euler", frame=f)
                trolley.location = (rad, 0, 0)
                trolley.keyframe_insert("location", frame=f)
                drop = max(1.2, crane.height + 0.5 - tip)
                lines.scale = (1, 1, drop - 1.0)
                lines.keyframe_insert("scale", frame=f)
                block.location = (0, 0, 0.7 - drop + 1.0)
                block.keyframe_insert("location", frame=f)
                held = lift.guid if lift is not None else None
                if held != carrying:
                    slings.hide_render = held is None
                    slings.keyframe_insert("hide_render", frame=f)
                    carrying = held
                if lift is not None and lift.guid in objs:
                    hx, hy = crane.hook_xy(ang, rad)
                    ob = objs[lift.guid]
                    ob.location = (hx, hy, base_z)
                    ob.keyframe_insert("location", frame=f)
            for lift in crane.lifts:                       # set down exactly in its place
                if lift.guid in objs:
                    objs[lift.guid].location = lift.place
                    objs[lift.guid].keyframe_insert("location", frame=lift.end + 1)
            for o in (slew, trolley, lines, block, *objs.values()):
                _linear(o)

    # --- piling rig and excavator in the pit during the foundations
    pile_guids = [g for st in tl.steps if st.phase == "Foundations" for g in st.guids
                  if by_guid[g].kind == "IFCPILE"]
    if "piling_rig" in windows:
        s, e = windows["piling_rig"][0]
        rig = E.piling_rig()
        _visible(rig, s, e, tl.frames)
        targets = pile_guids or [g for st in tl.steps if st.phase == "Foundations" for g in st.guids]
        stride = max(1, len(targets) // 30)
        for g in targets[::stride]:
            a, b, _ = tl.appear[g]
            c = by_guid[g].centre
            rig.location = (float(c[0]) - 2.6, float(c[1]), -pit_depth)
            rig.keyframe_insert("location", frame=a)
        rig.location = (cx, cy, -pit_depth) if not targets else rig.location
    if "excavator" in windows:
        s, e = windows["excavator"][0]
        ex = E.excavator()
        _visible(ex, s, e, tl.frames)
        n = 6
        for k in range(n + 1):
            f = int(s + (e - s) * k / n)
            ex.location = (x0 + 3 + (x1 - x0 - 6) * rng.random(), y0 + 3 + (y1 - y0 - 6) * rng.random(), -pit_depth)
            ex.rotation_euler = (0, 0, rng.uniform(-math.pi, math.pi))
            ex.keyframe_insert("location", frame=f)
            ex.keyframe_insert("rotation_euler", frame=f)

    # --- trucks: in along the lane to the bay, stand, reverse out; one in the lane at a time
    lane = plan.get("lane")
    if lane is not None:
        builders = {"mixer_truck": E.mixer_truck, "flatbed_truck": E.flatbed_truck}
        fleet: dict[str, list] = {}
        for v in plan.get("visits", []):
            # a truck of this kind that is free again, or a new one
            pool = fleet.setdefault(v.kind, [])
            truck = next((t for t, free in pool if free < v.arrive), None)
            if truck is None:
                truck = builders[v.kind]()
                truck.location = (0, 0, -500)
                truck.keyframe_insert("location", frame=0)
                pool.append([truck, -1])
            for item in pool:
                if item[0] is truck:
                    item[1] = v.gone
            truck.rotation_euler = (0, 0, lane.heading_in)
            truck.keyframe_insert("rotation_euler", frame=v.arrive)
            truck.location = (0, 0, -500)
            truck.keyframe_insert("location", frame=v.arrive - 1)
            for f in range(v.arrive, v.gone + 1, 3):
                (x, y), _ = L.truck_at(v, f, lane)
                truck.location = (x, y, 0.02)
                truck.keyframe_insert("location", frame=f)
            truck.location = (0, 0, -500)
            truck.keyframe_insert("location", frame=v.gone + 1)
        for pool in fleet.values():
            for truck, _ in pool:
                _linear(truck)

    # --- concrete pump: parked beside the bay, out of the lane, its boom towards the pour
    for (s, e) in windows.get("concrete_pump", []):
        pump = E.concrete_pump()
        _visible(pump, s, e, tl.frames)
        px, py = plan.get("pump_at", (cx + nx * 10, cy + ny * 10))
        pump.location = (px, py, 0.02)
        pump.rotation_euler = (0, 0, math.atan2(cy - py, cx - px))

    # --- MEWP round the facade
    for (s, e) in windows.get("mewp", []):
        lift = E.mewp()
        _visible(lift, s, e, tl.frames)
        perim = [(x0 - 2.5, y0 - 2.5), (x1 + 2.5, y0 - 2.5), (x1 + 2.5, y1 + 2.5), (x0 - 2.5, y1 + 2.5)]
        n = 8
        for k in range(n + 1):
            a, b = perim[k % 4], perim[(k + 1) % 4]
            t = rng.random()
            lift.location = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0.02)
            lift.rotation_euler = (0, 0, math.atan2(b[1] - a[1], b[0] - a[0]))
            f = int(s + (e - s) * k / n)
            lift.keyframe_insert("location", frame=f)
            lift.keyframe_insert("rotation_euler", frame=f)


def build_crews(tl: Timeline, fp, pts: dict[str, Any], pit_depth: float, rng: random.Random,
                plan: dict[str, Any] | None = None) -> None:
    """Each step's crew on the floor it is building, out of the truck lane and never where the
    crane is setting a load down; riggers at the laydown; a banksman at the gate, beside the lane."""
    plan = plan or {}
    lane = plan.get("lane")
    crane = plan.get("crane")
    x0, y0, x1, y1 = fp
    pool = max(st.crew for st in tl.steps) if tl.steps else 0
    people = [E.worker("hivis" if k % 3 else "hivis_yellow", seed=k) for k in range(pool)]
    for w in people:
        w.location = (0, 0, -500)                       # parked out of sight
        w.keyframe_insert("location", frame=0)
    for st in tl.steps:
        z = _level_for(st)
        loads = [l.place[:2] for l in (crane.lifts if crane else []) if st.start <= l.end and l.start <= st.end]
        spots = L.crew_spots(st.crew, (x0, y0, x1, y1), z, rng, lane, loads)
        ends = L.crew_spots(st.crew, (x0, y0, x1, y1), z, rng, lane, loads)
        for k, w in enumerate(people):
            if k >= len(spots):
                w.location = (0, 0, -500)
                w.keyframe_insert("location", frame=st.start)
                continue
            a = spots[k]
            # a short walk towards another safe spot, never more than 4 m
            b = ends[k] if k < len(ends) else a
            d = math.dist(a[:2], b[:2])
            if d > 4.0:
                b = (a[0] + (b[0] - a[0]) * 4.0 / d, a[1] + (b[1] - a[1]) * 4.0 / d, z)
            w.location = a
            w.rotation_euler = (0, 0, math.atan2(b[1] - a[1], b[0] - a[0]) if d > 0.1 else rng.uniform(-math.pi, math.pi))
            w.keyframe_insert("location", frame=st.start)
            w.keyframe_insert("rotation_euler", frame=st.start)
            w.location = b
            w.keyframe_insert("location", frame=max(st.start + 1, st.end - 1))
    for w in people:
        w.location = (0, 0, -500)
        w.keyframe_insert("location", frame=tl.build_end)
        _constant_jumps(w)
    gx, gy = pts["gate"]
    nx, ny = pts["out"]
    tx, ty = pts["along"]
    lay = pts["laydown"]
    # two riggers by the laydown, slinging the loads; a banksman by the gate, beside the lane
    for k, (px, py) in enumerate([(lay[0] + tx * 4.5 - nx * 1.5, lay[1] + ty * 4.5 - ny * 1.5),
                                  (lay[0] + tx * 4.5 + nx * 1.5, lay[1] + ty * 4.5 + ny * 1.5),
                                  (gx - nx * 2 + tx * 6.5, gy - ny * 2 + ty * 6.5)]):
        if lane is not None and not lane.keeps_clear((px, py)):
            px, py = px + tx * 3, py + ty * 3
        w = E.worker("hivis_yellow", seed=100 + k)
        w.location = (px, py, 0.02)
        w.rotation_euler = (0, 0, math.atan2(gy - py, gx - px) if k == 2 else math.atan2(lay[1] - py, lay[0] - px))


def _constant_jumps(obj) -> None:
    """A person sent away (z -500) or brought back goes in one frame, not by sliding."""
    for fc in _curves(obj):
        if fc.data_path == "location":
            pts = fc.keyframe_points
            for i, kp in enumerate(pts):
                nxt = pts[i + 1] if i + 1 < len(pts) else None
                if kp.co[1] < -400 or (nxt is not None and nxt.co[1] < -400):
                    kp.interpolation = "CONSTANT"


def _level_for(st) -> float:
    """Where a crew stands for a step: on the floor of its storey (its slab top)."""
    return float(st.floor) if hasattr(st, "floor") else 0.02


# ---- light, camera, render ----------------------------------------------------------------

def build_light(lat: float, lon: float, when: dt.datetime | None) -> tuple[float, float]:
    import bpy
    when = when or dt.datetime.now().replace(hour=10, minute=30)
    hour_utc = when.hour + when.minute / 60 - lon / 15.0
    elev, az = sun_position(lat, lon, when.year, when.month, when.day, hour_utc)
    if elev < 12:                                   # never a dark film: fall back to mid-morning sun
        elev, az = 35.0, 130.0
    world = bpy.data.worlds.new("sky")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    sky = nt.nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "MULTIPLE_SCATTERING"
    except TypeError:
        sky.sky_type = "NISHITA"
    sky.sun_disc = False
    sky.sun_elevation = math.radians(elev)
    sky.sun_rotation = math.radians((180 - az) % 360) + math.pi / 2
    sky.air_density = 1.2
    sky.aerosol_density = 1.6                      # a little haze: distance reads as distance
    # Below the horizon the sky model is black; reflections of it make dark holes in water
    # and glass. Fade to a haze colour there instead.
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(coord.outputs["Generated"], sep.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = -0.05
    ramp.inputs["From Max"].default_value = 0.02
    nt.links.new(sep.outputs["Z"], ramp.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.55, 0.62, 0.70, 1.0)
    nt.links.new(ramp.outputs["Result"], mix.inputs["Factor"])
    nt.links.new(sky.outputs["Color"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.13
    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 3.8
    sun_data.angle = math.radians(0.6)
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(90 - elev), 0, math.radians((180 - az) % 360))
    return elev, az


def build_camera(fp, top: float, tl: Timeline, opts: Options, gate_dir: tuple[float, float]):
    import bpy
    x0, y0, x1, y1 = fp
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = max(x1 - x0, y1 - y0, top)
    target = bpy.data.objects.new("cam_target", None)
    bpy.context.scene.collection.objects.link(target)
    target.location = (cx, cy, top * 0.35)
    rig = bpy.data.objects.new("cam_rig", None)
    bpy.context.scene.collection.objects.link(rig)
    rig.location = (cx, cy, 0)
    cam_data = bpy.data.cameras.new("camera")
    cam_data.lens = 30
    cam_data.clip_end = 20000
    cam = bpy.data.objects.new("camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.parent = rig
    radius = max(70.0, size * 2.3)
    cam.location = (radius, 0, radius * 0.55)
    c = cam.constraints.new("TRACK_TO")
    c.target = target
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    bpy.context.scene.camera = cam
    # start looking in through the gate side, quarter on, and travel round
    start = math.atan2(gate_dir[1], gate_dir[0]) - math.radians(35)
    turn = math.radians(opts.camera_turn)
    rig.rotation_euler = (0, 0, start)
    rig.keyframe_insert("rotation_euler", frame=0)
    rig.rotation_euler = (0, 0, start + turn)
    rig.keyframe_insert("rotation_euler", frame=tl.frames)
    # closer in for the build, out again for the finished building
    cam.location = (radius * 1.15, 0, radius * 0.7)
    cam.keyframe_insert("location", frame=0)
    cam.location = (radius, 0, radius * 0.5)
    cam.keyframe_insert("location", frame=tl.build_start + (tl.build_end - tl.build_start) // 3)
    cam.location = (radius * 0.95, 0, radius * 0.42)
    cam.keyframe_insert("location", frame=tl.build_end)
    cam.location = (radius * 1.25, 0, radius * 0.38)
    cam.keyframe_insert("location", frame=tl.frames)
    target.keyframe_insert("location", frame=0)
    target.location = (cx, cy, top * 0.5)
    target.keyframe_insert("location", frame=tl.frames)
    for fc_obj in (rig,):
        ad = fc_obj.animation_data
        try:
            curves = ad.action.fcurves
        except AttributeError:
            curves = [fc for layer in ad.action.layers for strip in layer.strips
                      for bag in strip.channelbags for fc in bag.fcurves]
        for fc in curves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
    return cam


def configure_render(opts: Options, frames: int, out_dir: Path) -> None:
    import bpy
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    device = "CPU"
    for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = kind
            prefs.refresh_devices()
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if gpus:
                for d in prefs.devices:
                    d.use = True
                device = "GPU"
                break
        except (TypeError, ValueError):
            continue
    sc.cycles.device = device
    sc.cycles.samples = opts.samples
    sc.cycles.use_denoising = True
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces = 6
    sc.cycles.transparent_max_bounces = 6
    sc.cycles.glossy_bounces = 3
    sc.cycles.transmission_bounces = 4
    sc.render.resolution_x, sc.render.resolution_y = opts.width, opts.height
    sc.render.resolution_percentage = 100
    sc.render.fps = opts.fps
    sc.frame_start, sc.frame_end = 1, frames
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = str(out_dir / "frame_")
    sc.render.use_persistent_data = True


def build(elements: list[Element], tl: Timeline, env_folder: Path, manifest: dict[str, Any],
          opts: Options, out_dir: Path) -> dict[str, Any]:
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    M.reset()
    E._library.clear()
    rng = random.Random(7)

    fp = footprint(elements)
    top = float(max(e.hi[2] for e in elements))
    m = opts.site_margin
    site = (fp[0] - m, fp[1] - m, fp[2] + m, fp[3] + m)
    found_lo = [float(e.lo[2]) for e in elements if phase_of(e) == "Foundations" and e.kind != "IFCPILE"]
    pit_depth = min(3.0, max(0.0, -min(found_lo))) if found_lo else 0.0
    pit = (fp[0] - 1.5, fp[1] - 1.5, fp[2] + 1.5, fp[3] + 1.5)

    g = Ground(manifest, env_folder, site)
    # the gate goes on the side with the highest ground outside it: land, not sea
    cx, cy = (fp[0] + fp[2]) / 2, (fp[1] + fp[3]) / 2
    probe = {"south": (cx, site[1] - 40), "north": (cx, site[3] + 40),
             "west": (site[0] - 40, cy), "east": (site[2] + 40, cy)}
    gate_side = max(probe, key=lambda k: (g.at(*probe[k]) if not g.has_sea else g.at(*probe[k]) - g.sea_level))

    has_google = bool(manifest.get("google"))
    pts = build_site(g, pit, pit_depth, gate_side, graded_track=has_google)
    keep_out = (site[0] - 2, site[1] - 2, site[2] + 2, site[3] + 2)
    got = build_google(g, manifest, env_folder, keep_out, (pts["gate"], pts["out"])) if has_google else None
    google_level = got[0] if got else None
    if got:
        # Google's mesh is the ground: no terrain under it to poke through where it dips
        build_base(g, min(-3.0, got[1] - 3.0), float(manifest["google"].get("horizon", g.radius)))
    else:
        build_terrain(g, manifest, env_folder)
        build_neighbours(g, manifest, env_folder, keep_out)
    # the floor each step's crew stands on
    for st in tl.steps:
        st.floor = -pit_depth + 0.02 if st.phase == "Foundations" else _floor_of(st, elements)
    plan = plan_site(elements, tl, fp, top, pts, m)
    build_model(elements, tl)
    build_plant(elements, tl, fp, top, pts, pit_depth, rng, plan)
    build_crews(tl, fp, pts, pit_depth, rng, plan)
    elev, az = build_light(manifest["lat"], manifest["lon"], opts.when)
    build_camera(fp, top, tl, opts, pts["out"])
    _all_constant_hides()
    configure_render(opts, tl.frames, out_dir)
    return {"gate_side": gate_side, "pit_depth": pit_depth, "google": google_level is not None, "sun": [round(elev, 1), round(az, 1)],
            "top": top, "footprint": fp}


def _floor_of(st, elements: list[Element]) -> float:
    """The storey's own floor level, where its crew stands (never below the graded site)."""
    wanted = set(st.guids)
    els = [e for e in elements if e.guid in wanted]
    if not els:
        return 0.02
    return max(0.02, float(els[0].storey_elevation) + 0.02)
