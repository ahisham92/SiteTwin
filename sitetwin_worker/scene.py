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
    _mesh("access_track", tv, tf, M.get("asphalt_road"))
    return {"gate": (gx, gy), "out": (nx, ny), "along": (tx, ty), "laydown": lay}


def build_google(g: Ground, manifest: dict[str, Any], folder: Path,
                 keep_out: tuple[float, float, float, float],
                 access: tuple[tuple[float, float], tuple[float, float]] | None = None,
                 existing: list | None = None, flat: float = 0.0) -> tuple[float, float] | None:
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
    if manifest.get("osm"):
        try:
            feats = json.loads((folder / manifest["osm"]["file"]).read_text())
            manifest["_street"] = rebuild_near(near, level, keep_out, feats, clear=flat + 12.0)
        except Exception as exc:                            # noqa: BLE001 - Google's own look stands
            print(f"  near streets not rebuilt: {type(exc).__name__}: {exc}")

    def graded(x: float, y: float, z: float) -> float:
        """z (already lowered) with the grading round the site and along the access road."""
        w = 0.0
        if abs(z) < 4.0:                                    # ground, not roofs
            d = from_site(x, y)
            w = 1.0 - smooth01(max(0.0, d - flat) / 12.0)
        if access is not None:
            (gx, gy), (nx, ny) = access
            along = (x - gx) * nx + (y - gy) * ny
            across = abs(-(x - gx) * ny + (y - gy) * nx)
            if -2.0 <= along <= 70.0:
                wa = 1.0 - smooth01((across - 5.0) / 7.0) if across > 5.0 else 1.0
                wa *= 1.0 - smooth01((along - 60.0) / 10.0) if along > 60.0 else 1.0
                w = max(w, wa)
        return z * (1.0 - w) - 0.1 * w                    # graded ground stays just under the pad

    inner = float(info.get("radius", g.radius))
    outer = float(info.get("horizon", inner))
    lowest = math.inf
    graded_objs = manifest.setdefault("_graded", [])
    def inside(f) -> bool:
        c = f.calc_center_median()
        return (kx0 <= c.x <= kx1 and ky0 <= c.y <= ky1) or any(
            kx0 <= v.co.x <= kx1 and ky0 <= v.co.y <= ky1 for v in f.verts)

    for group, keep in ((near, lambda c: math.hypot(c.x, c.y) <= inner),
                        (far, lambda c: inner < math.hypot(c.x, c.y) <= outer)):
        for o in group:
            if group is near and existing is not None:
                # what stands on the site today: kept apart in 4 m square pieces,
                # to be broken out one by one at the start of the film
                bm = bmesh.new()
                bm.from_mesh(o.data)
                bmesh.ops.delete(bm, geom=[f for f in bm.faces if not inside(f)], context="FACES")
                if not bm.faces:
                    bm.free()
                    continue_existing = False
                else:
                    continue_existing = True
                    # cut on a square grid, so the site is broken out in square patches, not slivers
                    step = 4.0
                    for axis, lo, hi in (((1, 0, 0), kx0 - 30, kx1 + 30), ((0, 1, 0), ky0 - 30, ky1 + 30)):
                        t = math.floor(lo / step) * step
                        while t <= hi:
                            geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
                            bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-4,
                                                   plane_co=tuple(t * a for a in axis), plane_no=axis)
                            t += step
                    bm.faces.index_update()
                cells: dict[tuple[int, int], list] = {}
                for f in (bm.faces if continue_existing else []):
                    c = f.calc_center_median()
                    cells.setdefault((int(c.x // 4), int(c.y // 4)), []).append(f.index)
                for cell, idx in cells.items():
                    piece = bm.copy()
                    piece.faces.index_update()
                    keep_idx = set(idx)
                    bmesh.ops.delete(piece, geom=[f for f in piece.faces if f.index not in keep_idx],
                                     context="FACES")
                    for v in piece.verts:          # its ground sits a hair over the new pad, never under
                        v.co.z = max(v.co.z - level - 0.05, 0.1)
                    data = bpy.data.meshes.new("existing")
                    piece.to_mesh(data)
                    piece.free()
                    for mat in o.data.materials:
                        data.materials.append(mat)
                    ob = bpy.data.objects.new("existing_site", data)
                    bpy.context.scene.collection.objects.link(ob)
                    _photo_material(ob)
                    existing.append(ob)
                if continue_existing:
                    bm.free()
            bm = bmesh.new()
            bm.from_mesh(o.data)
            gone = [f for f in bm.faces if not keep(f.calc_center_median()) or inside(f)]
            bmesh.ops.delete(bm, geom=gone, context="FACES")
            staged = group is near and existing is not None
            target = []
            for v in bm.verts:
                x, y, z = v.co
                z = z - level - 0.05
                gz = graded(x, y, z)
                if staged:                          # the ground as it is; graded during the clearing
                    v.co.z = z
                    target.append(gz)
                else:
                    v.co.z = gz
                lowest = min(lowest, gz)
            bm.to_mesh(o.data)
            bm.free()
            if staged and any(abs(t - v.co.z) > 1e-3 for t, v in zip(target, o.data.vertices)):
                o.shape_key_add(name="as_is")
                key = o.shape_key_add(name="graded", from_mix=False)
                for t, kv in zip(target, key.data):
                    kv.co.z = t
                graded_objs.append(o)
            o.name = "google_" + o.name
            _photo_material(o)
    # the opening flight's Earth, region and city: each kept only where the finer one inside it
    # has nothing (judged on a grid a little coarser than the finer one's faces)
    def covered(objs: list, cell: float) -> set:
        out = set()
        for o in objs:
            for poly in o.data.polygons:
                c = poly.center
                out.add((int(c.x // cell), int(c.y // cell)))
        return out

    finer = near + far
    for name, cell, drop in (("city", 60.0, 1.5), ("region", 600.0, 8.0), ("globe", 6000.0, 20.0)):
        objs = load(info.get("intro", {}).get(name, []))
        have = covered(finer, cell)
        kept = []
        for o in objs:
            bm = bmesh.new()
            bm.from_mesh(o.data)
            def hidden(f) -> bool:                      # wholly under the finer layer
                pts = [f.calc_center_median()] + [v.co for v in f.verts]
                return all((int(p.x // cell), int(p.y // cell)) in have for p in pts)
            gone = [f for f in bm.faces if hidden(f) or math.hypot(*f.calc_center_median()[:2]) < inner]
            bmesh.ops.delete(bm, geom=gone, context="FACES")
            # a little below the finer layer, so where both remain the finer one shows
            bmesh.ops.translate(bm, verts=bm.verts, vec=(0.0, 0.0, -level - 0.05 - drop))
            bm.to_mesh(o.data)
            bm.free()
            o.name = f"google_{name}"
            _photo_material(o)
            kept.append(o)
        finer = finer + kept
    return level, (lowest if lowest < math.inf else 0.0)


def rebuild_near(tiles: list, level: float, keep_out, feats: list, clear: float = 0.0) -> dict:
    """The streets round the site rebuilt clean (see nearzone.py): Google's surface is reshaped
    in place, and the buildings and roads are added. Works in the tiles' own (raw) heights."""
    import bmesh
    import bpy
    from . import nearzone as NZ
    shift = level + 0.05
    parts, cols, steeps = [], [], []
    for o in tiles:
        me = o.data
        n = len(me.vertices)
        co = np.empty(n * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(n, 3)
        nor = np.empty(n * 3)
        me.vertices.foreach_get("normal", nor)
        nor = nor.reshape(n, 3)
        parts.append(co)
        steeps.append(np.abs(nor[:, 2]) < 0.5)
        cols.append(_vertex_photo_colours(o, n))
    xyz = np.concatenate(parts)
    xyz[:, 2] -= shift
    out = NZ.plan(xyz, np.concatenate(cols), np.concatenate(steeps), feats, keep_out)
    k = 0
    for o, co in zip(tiles, parts):
        n = len(co)
        co[:, 2] = out["z"][k:k + n] + shift
        o.data.vertices.foreach_set("co", co.ravel())
        o.data.update()
        k += n
    # the buildings: one mesh, each face carrying its building's colours and ground level
    bm = bmesh.new()
    wall_l = bm.faces.layers.float_color.new("wall")
    roof_l = bm.faces.layers.float_color.new("roof")
    base_l = bm.faces.layers.float.new("base")
    top_l = bm.faces.layers.float.new("top")
    for b in out["buildings"]:
        ring = b["points"]
        area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]))
        if area < 0:
            ring = ring[::-1]
        lo, hi = b["base"] - 0.4, b["top"]
        bot = [bm.verts.new((x, y, lo)) for x, y in ring]
        rim = [bm.verts.new((x, y, hi + 0.9)) for x, y in ring]      # a parapet round the roof
        top = [bm.verts.new((x, y, hi)) for x, y in ring]
        faces = []
        m = len(ring)
        for i in range(m):
            try:
                faces.append(bm.faces.new((bot[i], bot[(i + 1) % m], rim[(i + 1) % m], rim[i])))
            except ValueError:
                pass
        try:
            faces.append(bm.faces.new(top))
        except ValueError:
            pass
        for f in faces:
            f[wall_l] = (*b["wall"], 1.0)
            f[roof_l] = (*b["roof"], 1.0)
            f[base_l] = b["base"]
            f[top_l] = hi
    if out["buildings"]:
        me = bpy.data.meshes.new("near_buildings")
        bm.to_mesh(me)
        me.materials.append(M.get("facade_real"))
        ob = bpy.data.objects.new("near_buildings", me)
        bpy.context.scene.collection.objects.link(ob)
    bm.free()
    # the roads, laid on the smoothed ground (in scene heights; the tiles are lowered later)
    # (OSM roads are not drawn over Google's: laid as strips they floated and looked false)
    life = NZ.street_life(out["roads"], out["buildings"], out["grid"], keep_out, clear)
    return {"buildings": len(out["buildings"]), "roads": len(out["roads"]),
            "trees": [(t["x"], t["y"], t["z"], t["height"]) for t in out["trees"]
                      if math.hypot(max(keep_out[0] - t["x"], t["x"] - keep_out[2], 0.0),
                                    max(keep_out[1] - t["y"], t["y"] - keep_out[3], 0.0)) > clear],
            **life}


def _vertex_photo_colours(o, n: int) -> np.ndarray:
    """Each vertex's colour in the tile's photo (linear RGB), grey where there is none."""
    out = np.full((n, 3), 0.4)
    me = o.data
    if not me.uv_layers or not me.materials:
        return out
    img = None
    for mat in me.materials:
        if mat and mat.use_nodes:
            img = next((nd.image for nd in mat.node_tree.nodes if nd.type == "TEX_IMAGE" and nd.image), None)
            if img:
                break
    if img is None or not img.size[0]:
        return out
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)[:, :, :3]
    nl = len(me.loops)
    vi = np.empty(nl, np.int64)
    me.loops.foreach_get("vertex_index", vi)
    uv = np.empty(nl * 2)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(nl, 2)
    per = np.zeros((n, 2))
    per[vi] = uv
    x = np.clip((per[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
    y = np.clip((per[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(int)
    c = px[y, x]
    if img.colorspace_settings.name == "sRGB" and not img.is_float:
        c = np.power(np.clip(c, 0, 1), 2.2)
    return c.astype(float)


def _photo_material(o) -> None:
    """Photogrammetry carries its own daylight: no shine, and a little of its colour as light."""
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


def smooth01(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


APRON = 9.0     # metres of clean paving round the site pad, where Google's mesh is worst


def build_apron(site, pad_e: float) -> list:
    """A clean paved ring round the site pad, with a kerb at its outer edge: the street the
    hoarding fronts, in place of Google's smeared ground right round the site."""
    x0, y0, x1, y1 = site
    a = (x0 - pad_e, y0 - pad_e, x1 + pad_e, y1 + pad_e)
    b = (a[0] - APRON, a[1] - APRON, a[2] + APRON, a[3] + APRON)
    z = 0.0
    v = [(a[0], a[1], z), (a[2], a[1], z), (a[2], a[3], z), (a[0], a[3], z),
         (b[0], b[1], z), (b[2], b[1], z), (b[2], b[3], z), (b[0], b[3], z)]
    f = [(4, 5, 1, 0), (5, 6, 2, 1), (6, 7, 3, 2), (7, 4, 0, 3)]
    apron = _mesh("apron", v, f, M.get("asphalt_road"))
    k, h = 0.3, 0.15
    c = (b[0] + k, b[1] + k, b[2] - k, b[3] - k)
    kv = []
    for (X0, Y0, X1, Y1), zz in ((b, h), (c, h), (b, -0.2), (c, -0.2)):
        kv += [(X0, Y0, zz), (X1, Y0, zz), (X1, Y1, zz), (X0, Y1, zz)]
    kf = []
    for i in range(4):
        j = (i + 1) % 4
        kf += [(i, j, 4 + j, 4 + i), (8 + i, 8 + j, j, i), (4 + i, 4 + j, 12 + j, 12 + i)]
    kerb = _mesh("apron_kerb", kv, kf, M.get("concrete"))
    return [apron, kerb]


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
            a = _cast(e, obj, a, b, tl)
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


def _box(name: str, lo, hi, material, cage: float = 0.0):
    """A box from lo to hi; with ``cage`` > 0, a wire cage of bars that far apart."""
    import bmesh
    import bpy
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    bm = bmesh.new()
    if cage <= 0:
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co.x = x0 + (v.co.x + 0.5) * (x1 - x0)
            v.co.y = y0 + (v.co.y + 0.5) * (y1 - y0)
            v.co.z = z0 + (v.co.z + 0.5) * (z1 - z0)
    else:                                                 # bars: verticals round the edge, links up it
        r = 0.012
        def bar(p, q):
            ax = [i for i in range(3) if abs(q[i] - p[i]) > 1e-6][0]
            lo_ = [min(p[i], q[i]) - (0 if i == ax else r) for i in range(3)]
            hi_ = [max(p[i], q[i]) + (0 if i == ax else r) for i in range(3)]
            res = bmesh.ops.create_cube(bm, size=1.0)
            for v in res["verts"]:
                for i in range(3):
                    v.co[i] = lo_[i] + (v.co[i] + 0.5) * (hi_[i] - lo_[i])
        nx = max(1, int((x1 - x0) / cage))
        ny = max(1, int((y1 - y0) / cage))
        for i in range(nx + 1):
            for j in (0, ny):
                x, y = x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny
                bar((x, y, z0), (x, y, z1))
        for j in range(1, ny):
            for i in (0, nx):
                x, y = x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny
                bar((x, y, z0), (x, y, z1))
        nz = max(1, int((z1 - z0) / 0.25))
        for k in range(nz + 1):
            z = z0 + (z1 - z0) * k / nz
            bar((x0, y0, z), (x1, y0, z)); bar((x0, y1, z), (x1, y1, z))
            bar((x0, y0, z), (x0, y1, z)); bar((x1, y0, z), (x1, y1, z))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _cast(e: Element, obj, a: int, b: int, tl: Timeline) -> int:
    """Cast in place as a site does it, between frames a and b. Columns and walls: the rebar
    cage stands, the forms close round it, the concrete is poured, and the forms are struck.
    Slabs and beams: props and deck go up, the rebar mat is laid, and the concrete is poured
    across from one side; the props stay a while after. Returns the frame the concrete shows."""
    lo = [float(v) for v in e.lo]
    hi = [float(v) for v in e.hi]
    dx, dy, dz = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    span = max(4, b - a)
    t = lambda f: int(round(a + span * f))
    flat = dz < 0.6 * max(dx, dy) and dz < 1.2
    if not flat:
        cage = _box("rebar_" + e.guid, (lo[0] + 0.04, lo[1] + 0.04, lo[2]), (hi[0] - 0.04, hi[1] - 0.04, hi[2] + 0.6),
                    M.get("rebar"), cage=0.15) if max(dx, dy) < 2.5 else None
        form = _box("form_" + e.guid, (lo[0] - 0.03, lo[1] - 0.03, lo[2]), (hi[0] + 0.03, hi[1] + 0.03, hi[2]),
                    M.get("formwork"))
        if cage is not None:
            _visible(cage, a, t(0.45), tl.frames)
        _visible(form, t(0.3), t(0.9), tl.frames)
        obj.scale = (1, 1, 0.02)                         # poured inside the forms (unseen), struck at 0.9
        obj.keyframe_insert("scale", frame=t(0.45))
        obj.scale = (1, 1, 1)
        obj.keyframe_insert("scale", frame=t(0.75))
        return t(0.45)
    # slabs and beams: props down to the floor below, the deck, the mat, the pour across
    under = max(lo[2] - 3.4, -2.0)
    props = None
    if dx * dy > 1.0 and lo[2] - under > 0.5:
        import bmesh
        import bpy
        bm = bmesh.new()
        step = 1.5
        for i in range(int(dx / step) + 1):
            for j in range(int(dy / step) + 1):
                x, y = lo[0] + 0.3 + i * step, lo[1] + 0.3 + j * step
                if x > hi[0] - 0.2 or y > hi[1] - 0.2:
                    continue
                res = bmesh.ops.create_cube(bm, size=1.0)
                for v in res["verts"]:
                    v.co.x = x + v.co.x * 0.06
                    v.co.y = y + v.co.y * 0.06
                    v.co.z = under + (v.co.z + 0.5) * (lo[2] - 0.05 - under)
        me = bpy.data.meshes.new("props_" + e.guid)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(M.get("steel"))
        props = bpy.data.objects.new("props_" + e.guid, me)
        bpy.context.scene.collection.objects.link(props)
    deck = _box("deck_" + e.guid, (lo[0], lo[1], lo[2] - 0.05), (hi[0], hi[1], lo[2]), M.get("formwork"))
    keep = min(tl.frames, b + int(tl.fps * 2.5))
    if props is not None:
        _visible(props, a, keep, tl.frames)
    _visible(deck, t(0.12), keep, tl.frames)
    import bmesh
    import bpy
    bm = bmesh.new()                                     # the rebar mat: a 0.2 m grid of bars
    bmesh.ops.create_grid(bm, x_segments=max(2, int(dx / 0.2)), y_segments=max(2, int(dy / 0.2)), size=0.5)
    for v in bm.verts:
        v.co.x = (lo[0] + hi[0]) / 2 + v.co.x * (dx - 0.1)
        v.co.y = (lo[1] + hi[1]) / 2 + v.co.y * (dy - 0.1)
        v.co.z = lo[2] + 0.06
    me = bpy.data.meshes.new("mat_" + e.guid)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M.get("rebar"))
    mat = bpy.data.objects.new("mat_" + e.guid, me)
    bpy.context.scene.collection.objects.link(mat)
    mat.modifiers.new("bars", "WIREFRAME").thickness = 0.016
    _visible(mat, t(0.3), t(0.95), tl.frames)
    # poured across from the low-x side: origin moved to that edge, then stretched out
    shift = dx / 2
    obj.data.transform(__import__("mathutils").Matrix.Translation((shift, 0, 0)))
    obj.location = (obj.location[0] - shift, obj.location[1], obj.location[2])
    obj.scale = (0.02, 1, 1)
    obj.keyframe_insert("scale", frame=t(0.45))
    obj.scale = (1, 1, 1)
    obj.keyframe_insert("scale", frame=t(0.95))
    return t(0.45)


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
    machines: list[L.Machine] = plan.setdefault("machines", []) if isinstance(plan, dict) else []
    if "piling_rig" in windows:
        s, e = windows["piling_rig"][0]
        rig = E.piling_rig()
        _visible(rig, s, e, tl.frames)
        targets = pile_guids or [g for st in tl.steps if st.phase == "Foundations" for g in st.guids]
        stride = max(1, len(targets) // 30)
        stops, times = [], []
        for g in targets[::stride]:
            a, _, _ = tl.appear[g]
            c = by_guid[g].centre
            t = max(s, a - 2)
            if times and t <= times[-1] + 2:
                continue
            stops.append((float(c[0]) - 2.6, float(c[1])))
            times.append(t)
        if not stops:
            stops, times = [(cx, cy)], [s]
        keys = L.drive(stops, times, -pit_depth)
        keys = [(s, *keys[0][1:])] + [k for k in keys if k[0] > s] + [(e, *keys[-1][1:])]
        _apply_keys(rig, keys)
        machines.append(L.Machine("piling_rig", _half_length(rig) + 0.5, keys))
    for s, e in windows.get("excavator", []):
        # clearing the site (on the ground, before the setup) or digging the pit
        clearing = getattr(tl, "clear_end", 0) > 0 and s < tl.clear_end
        z = 0.02 if clearing else -pit_depth
        ex = E.excavator()
        _visible(ex, s, e, tl.frames)
        n = 5
        pts_ = [(x0 + 4 + (x1 - x0 - 8) * rng.random(), y0 + 4 + (y1 - y0 - 8) * rng.random()) for _ in range(n)]
        stops = L.nearest_order(pts_, (x0 + 4, y0 + 4))
        times = [int(s + (e - s) * k / max(1, n - 1)) for k in range(n)]
        keys = L.drive(stops, times, z)
        _apply_keys(ex, keys)
        machines.append(L.Machine("excavator", _half_length(ex) + 0.5, keys))

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
            half = _half_length(truck)
            keys = []
            for f in list(range(v.arrive, v.gone + 1, 3)) + [v.gone]:
                (x, y), _ = L.truck_at(v, f, lane, back=half)
                truck.location = (x, y, 0.02)
                truck.keyframe_insert("location", frame=f)
                keys.append((f, x, y, lane.heading_in, 0.02))
            machines.append(L.Machine(v.kind, half + 0.5, keys))
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
        h = math.atan2(cy - py, cx - px)
        machines.append(L.Machine("concrete_pump", _half_length(pump) + 0.5, [(s, px, py, h, 0.02), (e, px, py, h, 0.02)]))

    # --- MEWP round the facade: drives along the perimeter, turning at the corners
    for (s, e) in windows.get("mewp", []):
        lift = E.mewp()
        _visible(lift, s, e, tl.frames)
        perim = [(x0 - 2.5, y0 - 2.5), (x1 + 2.5, y0 - 2.5), (x1 + 2.5, y1 + 2.5), (x0 - 2.5, y1 + 2.5)]
        stops = []
        for k in range(4):                                  # once round: a stop mid-side, then the corner
            a, b = perim[k], perim[(k + 1) % 4]
            stops += [a, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)]
        stops.append(perim[0])
        times = [int(s + (e - s) * k / (len(stops) - 1)) for k in range(len(stops))]
        keys = L.drive(stops, times, 0.02)
        _apply_keys(lift, keys)
        machines.append(L.Machine("mewp", _half_length(lift) + 0.5, keys))


def _half_length(obj) -> float:
    """Half the longest side of an object (and its parts) in plan, in metres."""
    import mathutils
    xs, ys = [], []
    for o in [obj, *obj.children_recursive]:
        if o.type != "MESH":
            continue
        for v in o.bound_box:
            p = mathutils.Vector(v)
            q = o
            while q is not obj and q is not None:            # up the chain in local terms
                p = q.matrix_local @ p
                q = q.parent
            xs.append(p.x)
            ys.append(p.y)
    if not xs:
        return 2.0
    return max(max(xs) - min(xs), max(ys) - min(ys)) / 2


def _apply_keys(obj, keys) -> None:
    for f, x, y, h, z in keys:
        obj.location = (x, y, z)
        obj.rotation_euler = (0, 0, h)
        obj.keyframe_insert("location", frame=f)
        obj.keyframe_insert("rotation_euler", frame=f)
    _linear(obj)


def build_crews(tl: Timeline, fp, pts: dict[str, Any], pit_depth: float, rng: random.Random,
                plan: dict[str, Any] | None = None) -> None:
    """Each step's crew on the floor it is building, out of the truck lane and never where the
    crane is setting a load down; riggers at the laydown; a banksman at the gate, beside the lane."""
    plan = plan or {}
    lane = plan.get("lane")
    crane = plan.get("crane")
    machines: list[L.Machine] = plan.get("machines", [])
    x0, y0, x1, y1 = fp
    pool = max(st.crew for st in tl.steps) if tl.steps else 0
    people = [E.worker("hivis" if k % 3 else "hivis_yellow", seed=k) for k in range(pool)]
    for w in people:
        w.location = (0, 0, -500)                       # parked out of sight
        w.keyframe_insert("location", frame=0)
    for st in tl.steps:
        z = _level_for(st)
        loads = [l.place[:2] for l in (crane.lifts if crane else []) if st.start <= l.end and l.start <= st.end]
        win = (st.start, st.end)
        spots = L.crew_spots(st.crew, (x0, y0, x1, y1), z, rng, lane, loads, machines=machines, frames=win)
        ends = L.crew_spots(st.crew, (x0, y0, x1, y1), z, rng, lane, loads, machines=machines, frames=win)
        for k, w in enumerate(people):
            if k >= len(spots):
                w.location = (0, 0, -500)
                w.keyframe_insert("location", frame=st.start)
                continue
            a = spots[k]
            # a short walk towards another safe spot, never more than 4 m, and only if the way is clear
            b = ends[k] if k < len(ends) else a
            d = math.dist(a[:2], b[:2])
            if d > 4.0:
                b = (a[0] + (b[0] - a[0]) * 4.0 / d, a[1] + (b[1] - a[1]) * 4.0 / d, z)
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z)
            if not (L.clear_spot(b, *win, machines) and L.clear_spot(mid, *win, machines)):
                b = a
            d = math.dist(a[:2], b[:2])
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
    # two riggers by the laydown, slinging the loads; a banksman by the gate, beside the lane.
    # Each stands where no machine ever comes; if the first place is not, the next one along.
    fixed = []
    for k, (px, py) in enumerate([(lay[0] + tx * 4.5 - nx * 1.5, lay[1] + ty * 4.5 - ny * 1.5),
                                  (lay[0] + tx * 4.5 + nx * 1.5, lay[1] + ty * 4.5 + ny * 1.5),
                                  (gx - nx * 2 + tx * 6.5, gy - ny * 2 + ty * 6.5)]):
        for step in range(12):
            q = (px + tx * 1.5 * step, py + ty * 1.5 * step, 0.02)
            if (lane is None or lane.keeps_clear(q[:2])) and L.clear_spot(q, 0, tl.frames, machines):
                break
        else:
            continue                                     # nowhere safe: this one is left out
        w = E.worker("hivis_yellow", seed=100 + k)
        w.location = q
        w.rotation_euler = (0, 0, math.atan2(gy - q[1], gx - q[0]) if k == 2 else math.atan2(lay[1] - q[1], lay[0] - q[0]))
        fixed.append(w)
    plan["people"] = people + fixed
    plan["fixed_people"] = fixed


def check_clearance(people: list, machines: list, frames: int, step: int = 2, fix: bool = True) -> int:
    """Every person against every machine, every ``step`` frames, from their actual keyframes.
    Returns how many (person, frame) clashes there were; with ``fix``, a person who would be
    within a machine's reach is taken out of the shot for those frames (a last guard: the
    planning above should leave none)."""
    clashes = 0
    for w in people:
        curves = {fc.array_index: fc for fc in _curves(w) if fc.data_path == "location"}
        bad: list[int] = []
        for f in range(0, frames + 1, step):
            p = [curves[i].evaluate(f) if i in curves else w.location[i] for i in range(3)]
            if p[2] < -400:
                continue
            for m in machines:
                at = m.at(f)
                if at is not None and abs(at[3] - p[2]) < 2.5 and math.dist(at[:2], p[:2]) < m.radius + 0.5:
                    bad.append(f)
                    break
        clashes += len(bad)
        if fix and bad:
            for o in [w, *w.children_recursive]:
                for f in bad:
                    for frame, hidden in ((max(0, f - step), True), (f + step + 1, False)):
                        o.hide_render = hidden
                        o.hide_viewport = hidden
                        o.keyframe_insert("hide_render", frame=frame)
                        o.keyframe_insert("hide_viewport", frame=frame)
                o.hide_render = o.hide_viewport = False
    return clashes


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

def build_light(lat: float, lon: float, when: dt.datetime | None, frames: int = 0, fps: int = 24,
                day_seconds: float = 5.0) -> tuple[float, float]:
    """Sky and sun for the site's date. With ``frames``, the film is a time-lapse: every
    ``day_seconds`` of film the sun sweeps from morning to evening (shadows wheel round the
    building) and the next day starts; nights are skipped, as a site camera's would be."""
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
    if frames > 0:
        _sweep_sun(sun, sky, lat, lon, when, frames, max(1, int(day_seconds * fps)))
    return elev, az


def _sweep_sun(sun, sky, lat: float, lon: float, when: dt.datetime, frames: int, cycle: int) -> None:
    """Key the sun (and the sky's sun) from 8:00 to 17:00 local solar time over each cycle."""
    keys = []
    f = 0
    while f <= frames + cycle:
        for share, hour in ((0.0, 8.0), (0.25, 10.25), (0.5, 12.5), (0.75, 14.75), (0.98, 17.0)):
            el, az = sun_position(lat, lon, when.year, when.month, when.day, hour - lon / 15.0)
            el = max(10.0, el)
            keys.append((f + int(share * cycle), el, az))
        f += cycle
    prev_rot = None
    for frame, el, az in keys:
        rot_z = math.radians((180 - az) % 360)
        if prev_rot is not None:
            while rot_z - prev_rot > math.pi:
                rot_z -= 2 * math.pi
            while rot_z - prev_rot < -math.pi:
                rot_z += 2 * math.pi
        sun.rotation_euler = (math.radians(90 - el), 0, rot_z)
        sun.keyframe_insert("rotation_euler", frame=frame)
        warm = min(1.0, max(0.0, (el - 10) / 30))
        sun.data.color = (1.0, 0.82 + 0.18 * warm, 0.66 + 0.34 * warm)
        sun.data.keyframe_insert("color", frame=frame)
        sky.sun_elevation = math.radians(el)
        sky.sun_rotation = rot_z + math.pi / 2
        sky.keyframe_insert("sun_elevation", frame=frame)
        sky.keyframe_insert("sun_rotation", frame=frame)
        prev_rot = rot_z
    # the evening key jumps straight to the next morning: a new day, not the sun running back
    for owner in (sun, sun.data, sky.id_data):
        for fc in _curves(owner):
            pts = fc.keyframe_points
            for i, kp in enumerate(pts):
                if i + 1 < len(pts) and int(round(pts[i + 1].co[0])) % cycle == 0:
                    kp.interpolation = "CONSTANT"
                else:
                    kp.interpolation = "LINEAR"


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
    # a real drone orbits slowly: never more than about 4 degrees a second (a short film turns
    # less), so the site and the crane do not seem to spin
    turn = math.radians(min(opts.camera_turn, 4.0 * tl.frames / tl.fps))
    rig.rotation_euler = (0, 0, start)
    rig.keyframe_insert("rotation_euler", frame=0)
    rig.rotation_euler = (0, 0, start + turn)
    rig.keyframe_insert("rotation_euler", frame=tl.frames)
    # closer in for the build, out again for the finished building
    cam.location = (radius * 1.15, 0, radius * 0.7)
    cam.keyframe_insert("location", frame=tl.fly_end)
    if tl.fly_end > 0:
        _fly_down(cam, cam_data, (radius * 1.15, 0, radius * 0.7), tl.fly_end)
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


def _fly_down(cam, cam_data, end: tuple[float, float, float], frames: int,
              start_height: float = 1.8e7) -> None:
    """From space to the site: the camera falls along a steepening-to-oblique line, its height
    shrinking evenly in log terms (as in a globe viewer), looking at the site all the way.
    Near and far clipping follow the height so the Earth is never cut."""
    ex, _, ez = end
    end_dist = math.hypot(ex, ez)
    end_el = math.atan2(ez, ex)
    low = 0
    for f in list(range(0, frames, 2)) + [frames]:
        u = f / frames
        k = u * u * (3 - 2 * u)                              # ease in and out
        dist = math.exp(math.log(start_height) + (math.log(end_dist) - math.log(start_height)) * k)
        el = math.radians(88.0) + (end_el - math.radians(88.0)) * min(1.0, k * 1.15)
        cam.location = (dist * math.cos(el), 0.0, dist * math.sin(el))
        cam.keyframe_insert("location", frame=f)
        cam_data.clip_start = max(0.1, dist * 2e-4)
        cam_data.clip_end = max(20000.0, dist * 3.0)
        cam_data.keyframe_insert("clip_start", frame=f)
        cam_data.keyframe_insert("clip_end", frame=f)
        if dist > 3000.0:
            low = f + 2
    for fc in _curves(cam) + _curves(cam_data):
        for kp in fc.keyframe_points:
            if kp.co[0] <= frames:
                kp.interpolation = "LINEAR"
    import bpy
    for o in bpy.data.objects:                           # the flat skirt would show over the curved Earth
        if o.name.startswith("skirt"):
            _visible(o, min(low, frames), None, bpy.context.scene.frame_end + 1)


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
    before = set(bpy.data.objects)
    pts = build_site(g, pit, pit_depth, gate_side, graded_track=has_google)
    site_objs = [o for o in bpy.data.objects if o not in before]
    keep_out = (site[0] - 2, site[1] - 2, site[2] + 2, site[3] + 2)
    existing: list = []
    pad_e = 2 * g.radius / (g.n - 1) + 4.0
    got = build_google(g, manifest, env_folder, keep_out, (pts["gate"], pts["out"]),
                       existing if tl.clear_end > 0 else None, flat=pad_e - 2.0 + APRON) if has_google else None
    if got:
        site_objs += build_apron(g.site, pad_e)
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
    before = set(bpy.data.objects)
    build_props(g, fp, pts, plan, rng)
    street = manifest.pop("_street", None)
    if street:
        print("  street life:", build_street(street, tl, rng))
    site_objs += [o for o in bpy.data.objects if o not in before and o.parent is None]
    site_objs += plan.get("fixed_people", [])
    clashes = check_clearance(plan.get("people", []), plan.get("machines", []), tl.frames)
    graded = manifest.pop("_graded", [])
    if tl.clear_end > 0:
        stage_opening(tl, site_objs, existing, pts["gate"], graded)
    elev, az = build_light(manifest["lat"], manifest["lon"], opts.when, tl.frames, opts.fps)
    build_camera(fp, top, tl, opts, pts["out"])
    if tl.fly_end > 0:
        fade_to_space(tl)
    _all_constant_hides()
    configure_render(opts, tl.frames, out_dir)
    return {"clashes_removed": clashes, "gate_side": gate_side, "pit_depth": pit_depth, "google": google_level is not None, "sun": [round(elev, 1), round(az, 1)],
            "top": top, "footprint": fp}


def stage_opening(tl: Timeline, site_objs: list, existing: list, gate: tuple[float, float],
                  graded: list | None = None) -> None:
    """The site as it is today, cleared piece by piece (nearest the gate first) while the
    excavator works, then the hoarding, cabins and laydown going up."""
    import bpy
    ground = {"site_pad", "excavation", "site_walls", "apron", "apron_kerb"}
    for o in site_objs:
        base = o.name.split(".")[0]
        if existing and base in ground:
            _visible(o, tl.clear_start, None, tl.frames)
        elif base not in ground:
            _visible(o, tl.clear_end, None, tl.frames)
    if not existing:
        return
    span = max(2, tl.clear_end - tl.clear_start)
    for o in graded or []:                              # the ground round the site levelled as it is cleared
        kb = o.data.shape_keys.key_blocks["graded"]
        kb.value = 0.0
        kb.keyframe_insert("value", frame=tl.clear_start)
        kb.value = 1.0
        kb.keyframe_insert("value", frame=tl.clear_start + span // 2)

    def centre(o):
        xs = [v.co.x for v in o.data.vertices] or [0.0]
        ys = [v.co.y for v in o.data.vertices] or [0.0]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    order = sorted(existing, key=lambda o: math.dist(centre(o), gate))
    n = len(order)
    for i, o in enumerate(order):
        s0 = tl.clear_start + int(span * 0.9 * i / max(1, n))
        s1 = s0 + max(2, int(span * 0.06))
        o.scale = (1, 1, 1)
        o.keyframe_insert("scale", frame=s0)
        o.scale = (1, 1, 0.02)                              # broken down to the ground and carted off
        o.keyframe_insert("scale", frame=s1)
        _visible(o, None, s1 + 1, tl.frames)


def fade_to_space(tl: Timeline) -> None:
    """Above the atmosphere the sky is black: the world fades from the sky to space as the
    camera climbs (keyed from the camera's height through the opening flight)."""
    import bpy
    world = bpy.context.scene.world
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    link = bg.inputs["Color"].links[0]
    src = link.from_socket
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["B"].default_value = (0.0, 0.0, 0.0, 1.0)
    nt.links.new(src, mix.inputs["A"])
    nt.links.new(mix.outputs["Result"], bg.inputs["Color"])
    cam = bpy.context.scene.camera
    fac = mix.inputs["Factor"]
    for f in range(0, tl.fly_end + 1, 2):
        bpy.context.scene.frame_set(f)
        h = cam.matrix_world.translation.z
        fac.default_value = min(1.0, max(0.0, (math.log10(max(h, 1.0)) - 4.5) / 1.0))   # 30 km .. 300 km
        fac.keyframe_insert("default_value", frame=f)
    bpy.context.scene.frame_set(0)


def prop_spots(site: tuple[float, float, float, float], fp, pts: dict[str, Any],
               lane: "L.Lane | None") -> list[tuple[str, float, float, float]]:
    """Where a real site keeps its things: (prop kind, x, y, heading). Barriers line the road
    outside the gate, cones mark the gate, toilets and water stand by the cabins, the
    skip near the gate, pallets, rebar and water by the laydown, and light towers in the far
    corners. Nothing is put in the truck lane, on the building or outside the hoarding (except
    the road barriers)."""
    gx, gy = pts["gate"]
    nx, ny = pts["out"]
    tx, ty = pts["along"]
    lx, ly = pts["laydown"]
    road = math.atan2(ny, nx)
    fence = math.atan2(ty, tx)

    def at(t: float, n: float) -> tuple[float, float]:
        return gx + tx * t + nx * n, gy + ty * t + ny * n

    want: list[tuple[str, float, float, float, bool]] = []      # kind, x, y, heading, inside
    for n in (6.0, 10.0, 14.0):
        for t in (-6.5, 6.5):
            want.append(("prop_barrier", *at(t, n), road, False))
    for t in (-5.5, 5.5):
        want.append(("prop_cones", *at(t, -1.5), 0.0, True))
    want += [("prop_toilet", *at(22.0, -2.0), fence + math.pi, True),      # a row of four
             ("prop_ibc", *at(18.5, -7.0), fence, True),
             ("prop_skip", *at(-8.0, -4.0), road, True)]
    for k in range(3):
        want.append(("prop_bricks", lx - nx * 6 + tx * (k * 1.8 - 1.8), ly - ny * 6 + ty * (k * 1.8 - 1.8), fence, True))
    want.append(("prop_paving", lx - nx * 6 + tx * 3.8, ly - ny * 6 + ty * 3.8, fence, True))
    want.append(("prop_pallet", lx - nx * 8 + tx * 0.5, ly - ny * 8 + ty * 0.5, fence + 0.3, True))
    want.append(("prop_rebar", lx - tx * 9.5 - nx * 1.0, ly - ty * 9.5 - ny * 1.0, fence, True))
    # light towers in the two corners furthest from the gate
    x0, y0, x1, y1 = site
    corners = sorted([(x0 + 2.5, y0 + 2.5), (x1 - 2.5, y0 + 2.5), (x1 - 2.5, y1 - 2.5), (x0 + 2.5, y1 - 2.5)],
                     key=lambda c: -math.hypot(c[0] - gx, c[1] - gy))
    cx, cy = (fp[0] + fp[2]) / 2, (fp[1] + fp[3]) / 2
    for x, y in corners[:2]:
        want.append(("prop_light", x, y, math.atan2(cy - y, cx - x), True))

    out = []
    for kind, x, y, h, inside in want:
        if inside and not (x0 + 0.8 < x < x1 - 0.8 and y0 + 0.8 < y < y1 - 0.8):
            continue
        if fp[0] - 2 < x < fp[2] + 2 and fp[1] - 2 < y < fp[3] + 2:
            continue                                         # never on the building
        if lane is not None and inside and not lane.keeps_clear((x, y), margin=1.0):
            continue
        out.append((kind, x, y, h))
    return out


def build_props(g: Ground, fp, pts: dict[str, Any], plan: dict[str, Any], rng: random.Random) -> int:
    """The site's clutter, from real models where there are any. Returns how many were placed."""
    n = 0
    for kind, x, y, h in prop_spots(g.site, fp, pts, plan.get("lane")):
        obj = E.prop(kind)
        if obj is None:
            continue
        obj.location = (x, y, 0.02)
        obj.rotation_euler = (0, 0, h + rng.uniform(-0.06, 0.06))
        n += 1
    return n


CAR_SPEED = 9.0         # m/s of film time: town traffic


def build_street(street: dict[str, Any], tl: Timeline, rng: random.Random) -> dict[str, int]:
    """Real trees where Google had tree blobs, cars parked at the kerbs and a few driving."""
    n = {"trees": 0, "parked": 0, "moving": 0}
    for i, (x, y, z, h) in enumerate(street.get("trees", [])):
        palm = rng.random() < 0.35
        o = E.prop("prop_palm") if palm else E.variant("prop_tree", rng.randrange(1000))
        if o is None:
            continue
        base = 9.0 if palm else 8.0
        k = max(0.5, min(1.8, h / base)) * rng.uniform(0.9, 1.1)
        o.scale = (k, k, k)
        o.location = (x, y, z - 0.05)
        o.rotation_euler = (0, 0, rng.uniform(0, 2 * math.pi))
        n["trees"] += 1
    for i, (x, y, z, head) in enumerate(street.get("parked", [])):
        o = E.variant("prop_cars", i)
        if o is None:
            break
        o.location = (x, y, z + 0.02)
        o.rotation_euler = (0, 0, head + rng.uniform(-0.03, 0.03))
        n["parked"] += 1
    for j, path in enumerate(street.get("moving", [])):
        o = E.variant("prop_cars", 50 + j * 3)
        if o is None:
            break
        _drive(o, path, tl, rng)
        n["moving"] += 1
    return n


def _drive(o, path: list, tl: Timeline, rng: random.Random) -> None:
    """A car along a path at a steady speed, from a random start; at the end it is hidden and
    starts again from the beginning (as if the next car came round the corner)."""
    seg = [math.dist(a[:2], b[:2]) for a, b in zip(path[:-1], path[1:])]
    total = sum(seg)
    step = CAR_SPEED / tl.fps
    pos = rng.uniform(0, total)
    shown = None
    prev = None
    for f in range(0, tl.frames + 1, 2):
        d = pos % total
        k = 0
        while k < len(seg) - 1 and d > seg[k]:
            d -= seg[k]
            k += 1
        a, b = path[k], path[k + 1]
        t = d / seg[k] if seg[k] else 0.0
        o.location = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t + 0.02)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        if prev is not None:
            ang = prev + (ang - prev + math.pi) % (2 * math.pi) - math.pi
        prev = ang
        o.rotation_euler = (0, 0, ang)
        o.keyframe_insert("location", frame=f)
        o.keyframe_insert("rotation_euler", frame=f)
        near_end = (pos % total) > total - 2 * step * 2 or (pos % total) < 2 * step * 2
        if near_end != shown:
            o.hide_render = near_end
            o.keyframe_insert("hide_render", frame=f)
            shown = near_end
        pos += step * 2
    ad = o.animation_data
    try:
        curves = ad.action.fcurves
    except AttributeError:
        curves = [fc for layer in ad.action.layers for strip in layer.strips
                  for bag in strip.channelbags for fc in bag.fcurves]
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR" if fc.data_path != "hide_render" else "CONSTANT"


def _floor_of(st, elements: list[Element]) -> float:
    """The storey's own floor level, where its crew stands (never below the graded site)."""
    wanted = set(st.guids)
    els = [e for e in elements if e.guid in wanted]
    if not els:
        return 0.02
    return max(0.02, float(els[0].storey_elevation) + 0.02)
