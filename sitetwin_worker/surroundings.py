"""Surroundings modelled the way the structure is, instead of a photograph or Google's mesh.

Draped aerial photographs and Google's photo-real mesh look soft and smeared next to a crisp
model, so here the place is rebuilt from data, in the same materials and light as the works:

* **Ground** from the terrain heights, in plain land, and every shore (breakwaters, armour, the
  coast) a clean rock slope of 1 in 1.5 in two tones of stone, as TritoMoor draws its mounds. The aerial photograph is never shown: it only says where the sea is, which
  coarse coastal heights cannot.
* **Sea** everywhere the photograph shows water, out to the horizon.
* **Buildings** from OpenStreetMap, extruded to their heights (or a believable spread), plastered
  walls and flat concrete roofs; greenhouses low and white; round tanks as tanks.
* **Roads** as asphalt ribbons on the ground.
* Beyond the fetched square the ground runs on flat to the horizon (land stays land, sea stays sea).

Everything is in metres with the sea at ``sea``; the area of the works (``keep``) is left empty
for the model's own ground.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from . import materials as M


def sea_share(image: Path, n: int) -> np.ndarray:
    """How much of each of the n×n terrain points the photograph shows as water, 0..1 (rows
    north→south). Sea in aerial photographs is blue or teal, well clear of red; land, roads,
    roofs and quays are not."""
    rgb = np.asarray(Image.open(image).convert("RGB"), dtype=np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    wet = ((b > r + 18) & (g > r + 12)).astype(np.float32)
    img = Image.fromarray((wet * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
    return np.asarray(img.convert("F").resize((n, n), Image.BILINEAR)) / 255.0


SLOPE = 1.5            # armour slopes, 1 in 1.5, as TritoMoor draws a rock mound
BED = 7.0              # the sea bed this far under the sea
CREST = 3.0            # the least height of land at the coast (a breakwater's crest)
BREAKWATER_CREST = float(__import__("os").environ.get("SITETWIN_BREAKWATER_CREST", "5.6"))  # above the sea
NEAR_WORKS = 60.0      # the photograph's breakwater strip this close to the works is redrawn
THIN = 35.0            # land no further than this from the sea all across is a breakwater
THIN_MASK = None       # set by heights(): the breakwaters' points
LAND_CLASS = None      # set by heights(): soil / green / paved at each point, from the photograph
UP = None              # set by heights(): metres in from the waterline at each point


def _distance(mask: np.ndarray, h: float, cells: int) -> np.ndarray:
    """Distance in metres from each point of ``mask`` to the nearest point outside it, up to
    ``cells`` points away (further is given as that far); 0 outside the mask."""
    n0, n1 = mask.shape
    out = np.full(mask.shape, cells * h, dtype=np.float32)
    pad = np.pad(~mask, cells, mode="edge")
    for di in range(-cells, cells + 1):
        for dj in range(-cells, cells + 1):
            d = math.hypot(di, dj)
            if d == 0 or d > cells:
                continue
            other = pad[cells + di:cells + di + n0, cells + dj:cells + dj + n1]
            np.minimum(out, np.where(other, d * h, out), out=out)
    return np.where(mask, out, 0.0)


def heights(manifest: dict[str, Any], folder: Path, sea: float,
            works: tuple[float, float, float, float] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """The ground in the scene's levels (z), the sea mask, and the radius.

    The coast is drawn the way TritoMoor draws it: land flat at its own level, and wherever it
    meets the sea a clean armour slope of 1 in 1.5 from the top edge down to the sea bed. The
    waterline comes from the photograph (coastal heights are far too coarse for it), at about
    3 m spacing, so breakwaters and the shore read as built rock mounds, not soft ramps."""
    r = float(manifest["radius"])
    raw = np.load(folder / manifest["terrain"]["file"]) if "terrain" in manifest else np.zeros((129, 129))
    n = 2 * (raw.shape[0] - 1) + 1 if raw.shape[0] < 900 else raw.shape[0]
    h = 2 * r / (n - 1)
    if "imagery" in manifest:
        rgb = np.asarray(Image.open(folder / manifest["imagery"]["file"]).convert("RGB"), dtype=np.int16)
        rr, gg, bb = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        wetp = ((bb > rr + 18) & (gg > rr + 12)).astype(np.uint8) * 255
        img = Image.fromarray(wetp).filter(ImageFilter.MedianFilter(7)).filter(ImageFilter.GaussianBlur(2.5))
        share = np.asarray(img.convert("F").resize((n, n), Image.BILINEAR)) / 255.0
        _land_cover(rgb, n)
    else:
        share = np.asarray(Image.fromarray((raw < 0.5).astype(np.float32)).resize((n, n), Image.BILINEAR))
    level = np.asarray(Image.fromarray(np.clip(raw, -20.0, 3000.0).astype(np.float32)).resize((n, n), Image.BILINEAR))
    dry = share < 0.5
    cells = max(int(math.ceil(SLOPE * (BED + BREAKWATER_CREST + 2.0) / h)), int(math.ceil(THIN / h)) + 1)
    up = _distance(dry, h, cells) - h / 2                      # metres in from the waterline
    down = _distance(~dry, h, cells) - h / 2                   # metres out from it
    # a breakwater: land in a strip no wider than twice THIN, all of it a rock mound with a
    # stone crest (its middle is never further than THIN from the sea)
    # (the works themselves count as water here: the photograph shows the old deck as land, and
    # the breakwater beside it would otherwise not read as a strip)
    up_t = up
    if works is not None:
        k0, k1, k2, k3 = works
        xs = np.linspace(-r, r, n)
        ys = np.linspace(r, -r, n)
        inw = ((ys[:, None] > k2) & (ys[:, None] < k3)) & ((xs[None, :] > k0) & (xs[None, :] < k1))
        up_t = _distance(dry & ~inw, h, cells) - h / 2
    far = np.asarray(Image.fromarray(up_t.astype(np.float32), "F").filter(ImageFilter.MaxFilter(2 * int(THIN / h) + 1)))
    global THIN_MASK
    THIN_MASK = dry & (far < THIN - h) & (up_t > 0)
    if works is not None:
        # the breakwater beside the works is drawn as a rock mound of its own (rock_mound), as
        # TritoMoor draws it, not traced from the photograph: clear the photograph's strip there
        gx = np.abs(xs[None, :] - np.clip(xs[None, :], k0, k1))
        gy = np.abs(ys[:, None] - np.clip(ys[:, None], k2, k3))
        clear = THIN_MASK & (np.hypot(gx, gy) < NEAR_WORKS)
        clear = np.asarray(Image.fromarray(clear.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0
        clear &= ~inw
        if clear.any():
            dry = dry & ~clear
            up = _distance(dry, h, cells) - h / 2
            down = _distance(~dry, h, cells) - h / 2
            up_t = np.where(clear, 0.0, up_t)
            far = np.where(clear, THIN, far)
            THIN_MASK &= ~clear
    # the land's own level, and at least +3 where it meets the sea, so the armour reads as a
    # slope from crest to sea bed as in TritoMoor; a breakwater stands higher, its crest above
    # the deck it shelters (rising to it over 10 m where it leaves the shore)
    # further in than the distances were counted is simply inland: the land at its own level
    # (counting stops at ``cells``, which must not cap the hills behind the coast)
    up = np.where(up >= cells * h - h, 1e6, up)
    global UP
    UP = up
    rise = np.clip((THIN - far) / 10.0, 0.0, 1.0) * (dry & (up_t > 0))
    crest = CREST + (BREAKWATER_CREST - CREST) * rise
    top = np.maximum(level, crest) + sea
    z = np.where(dry, np.minimum(top, sea + np.maximum(up, 0.0) / SLOPE),
                 np.maximum(sea - BED, sea - np.maximum(down, 0.0) / SLOPE))
    return z.astype(np.float32), z <= sea - BED + 1e-3, r


def _land_cover(rgb: np.ndarray, n: int) -> None:
    """What covers the ground at each point, read from the photograph: 1 bare soil and dry
    fields, 2 trees and green fields, 3 paved (asphalt, concrete yards, roofs). The photograph
    here has a green cast (grey asphalt reads 50/54/41), so green means well clear of it, or
    dark and green (tree crowns). Cleaned of specks, so each patch reads as a field or a yard."""
    global LAND_CLASS
    r, g, b = (rgb[..., k].astype(np.int16) for k in range(3))
    wet = (b > r + 18) & (g > r + 12)
    lum = (r + g + b) / 3.0
    gm = g - (r + b) / 2.0
    veg = (gm >= 15) & ((lum < 75) | (gm >= 26)) & ~wet
    paved = ((r - b) < 18) & ~wet & ~veg
    cls = np.where(veg, 2, np.where(paved, 3, 1)).astype(np.uint8)
    im = Image.fromarray(cls).filter(ImageFilter.ModeFilter(5)).resize((n, n), Image.NEAREST)
    LAND_CLASS = np.asarray(im.filter(ImageFilter.ModeFilter(3)))


class Surface:
    """Bilinear heights over the square, for standing things on the ground."""

    def __init__(self, z: np.ndarray, r: float):
        self.z, self.r, self.n = z, r, z.shape[0]

    def at(self, x: float, y: float) -> float:
        n, r = self.n, self.r
        fx = (x + r) / (2 * r) * (n - 1)
        fy = (r - y) / (2 * r) * (n - 1)
        j0 = int(np.clip(math.floor(fx), 0, n - 2))
        i0 = int(np.clip(math.floor(fy), 0, n - 2))
        tx, ty = min(max(fx - j0, 0.0), 1.0), min(max(fy - i0, 0.0), 1.0)
        z = self.z
        return float(z[i0, j0] * (1 - tx) * (1 - ty) + z[i0, j0 + 1] * tx * (1 - ty)
                     + z[i0 + 1, j0] * (1 - tx) * ty + z[i0 + 1, j0 + 1] * tx * ty)


def _mesh(name: str, verts, faces, mats: list[str], mat_index=None):
    import bpy
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    for m in mats:
        me.materials.append(M.get(m))
    if mat_index is not None:
        me.polygons.foreach_set("material_index", np.asarray(mat_index, dtype=np.int32))
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def h_of(r: float, n: int) -> float:
    return 2 * r / (n - 1)


def build_ground(z: np.ndarray, wet: np.ndarray, r: float, keep: tuple[float, float, float, float],
                 sea: float, horizon: float = 25000.0):
    """The ground mesh: land, with rock wherever it meets the sea; nothing where it is all sea or
    inside ``keep``. A flat run-out beyond the edges carries land and sea on to the horizon."""
    n = z.shape[0]
    xs = np.linspace(-r, r, n)
    ys = np.linspace(r, -r, n)
    gx, gy = np.meshgrid(xs, ys)
    verts = np.stack([gx, gy, z], axis=-1).reshape(-1, 3)
    # shore: dry points with water within two points of them
    near = wet.copy()
    for _ in range(2):
        p = np.pad(near, 1, mode="edge")
        near = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    shore = near & ~wet
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a = i * n + j
    quads = np.stack([a, a + n, a + n + 1, a + 1], axis=-1).reshape(-1, 4)
    w4 = np.stack([wet[:-1, :-1], wet[1:, :-1], wet[1:, 1:], wet[:-1, 1:]], axis=-1).reshape(-1, 4)
    s4 = np.stack([shore[:-1, :-1], shore[1:, :-1], shore[1:, 1:], shore[:-1, 1:]], axis=-1).reshape(-1, 4)
    cx = ((gx[:-1, :-1] + gx[1:, 1:]) / 2).reshape(-1)
    cy = ((gy[:-1, :-1] + gy[1:, 1:]) / 2).reshape(-1)
    k0, k1, k2, k3 = keep
    inside = (cx > k0) & (cx < k1) & (cy > k2) & (cy < k3)
    keep_face = ~w4.all(axis=1) & ~inside
    z4 = np.stack([z[:-1, :-1], z[1:, :-1], z[1:, 1:], z[:-1, 1:]], axis=-1).reshape(-1, 4)
    # the armour slope: any face that is not level (land tops are flat at their own level, the
    # coarse heights inland vary slowly); in blocks of about 9 m of two tones, as TritoMoor
    u4 = None
    if UP is not None and UP.shape == z.shape:
        u4 = np.stack([UP[:-1, :-1], UP[1:, :-1], UP[1:, 1:], UP[:-1, 1:]], axis=-1).reshape(-1, 4).min(axis=1)
    coast = u4 < SLOPE * (BREAKWATER_CREST + 1.0) + 2 * h_of(r, n) if u4 is not None else np.ones(len(cx), bool)
    slope = ((z4.max(axis=1) - z4.min(axis=1)) > 0.35 * h_of(r, n)) & (z4.min(axis=1) < sea + 8.0) & coast
    slope |= (s4.any(axis=1) | w4.any(axis=1)) & (z4.min(axis=1) < sea + 0.5)
    tone = ((np.floor(cx / 9.0) + np.floor(cy / 9.0)) % 2).astype(np.int32)
    mat = np.where(slope, 1 + tone, 0)
    if THIN_MASK is not None and THIN_MASK.shape == z.shape:
        t4 = np.stack([THIN_MASK[:-1, :-1], THIN_MASK[1:, :-1], THIN_MASK[1:, 1:], THIN_MASK[:-1, 1:]], axis=-1).reshape(-1, 4)
        mat = np.where((mat == 0) & t4.any(axis=1), 3, mat)          # a breakwater's crest
    if LAND_CLASS is not None and LAND_CLASS.shape == z.shape:
        cover = LAND_CLASS[:-1, :-1].reshape(-1)
        mat = np.where(mat == 0, np.select([cover == 2, cover == 3], [4, 5], 0), mat)
    mat = mat[keep_face]
    faces = quads[keep_face]
    _mesh("land", verts, faces, ["land", "armour_a", "armour_b", "armour_top", "land_green", "land_paved"],
          mat.astype(np.int32))
    # the run-out: each edge point carried straight out to the horizon at its own level
    ring = ([(0, j) for j in range(n)] + [(i, n - 1) for i in range(1, n)]
            + [(n - 1, j) for j in range(n - 2, -1, -1)] + [(i, 0) for i in range(n - 2, 0, -1)])
    rv, rf, rm = [], [], []
    for k, (ii, jj) in enumerate(ring):
        x, y = xs[jj], ys[ii]
        s = horizon / max(abs(x), abs(y), 1.0)
        land = not wet[ii, jj]
        zz = float(z[ii, jj]) if land else sea - 6.0
        rv += [(x, y, zz), (x * s, y * s, zz if land else sea - 6.0)]
    m = len(ring)
    for k in range(m):
        a0, b0 = 2 * k, 2 * ((k + 1) % m)
        if wet[ring[k]] and wet[ring[(k + 1) % m]]:
            continue
        rf.append((a0, b0, b0 + 1, a0 + 1))
        rm.append(0)
    if rf:
        _mesh("land_runout", rv, rf, ["land"], rm)
    return Surface(z, r)


def _ring(pts):
    return pts[:-1] if pts and pts[0] == pts[-1] else pts


def build_osm(feats: list[dict[str, Any]], ground: Surface, wet: np.ndarray, keep, sea: float) -> dict[str, int]:
    """Buildings, greenhouses, tanks and roads from OpenStreetMap, on the modelled ground."""
    import bmesh
    import bpy
    n, r = wet.shape[0], ground.r
    k0, k1, k2, k3 = keep

    def is_wet(x, y):
        j = int(round((x + r) / (2 * r) * (n - 1)))
        i = int(round((r - y) / (2 * r) * (n - 1)))
        if not (0 <= i < n and 0 <= j < n):
            return True
        return bool(wet[i, j])

    def in_keep(x, y):
        return k0 - 5 < x < k1 + 5 and k2 - 5 < y < k3 + 5

    meshes: dict[str, Any] = {}

    def bm_for(mat):
        if mat not in meshes:
            meshes[mat] = bmesh.new()
        return meshes[mat]

    counts = {"buildings": 0, "greenhouses": 0, "tanks": 0, "roads": 0}
    for f in feats:
        pts = f["points"]
        if not pts or not any(abs(x) < r and abs(y) < r for x, y in pts):
            continue
        kind = f["kind"]
        if kind in ("building", "tank"):
            ring = _ring(pts)
            if len(ring) < 3:
                continue
            cx = sum(p[0] for p in ring) / len(ring)
            cy = sum(p[1] for p in ring) / len(ring)
            if in_keep(cx, cy) or is_wet(cx, cy):
                continue
            base = min(ground.at(x, y) for x, y in ring) - 0.3
            if kind == "tank":
                rad = sum(math.hypot(x - cx, y - cy) for x, y in ring) / len(ring)
                h = f.get("height") or min(22.0, max(6.0, rad * 1.1))
                bm = bm_for("tank")
                bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=rad, radius2=rad, depth=h + 0.3,
                                      matrix=_translate(cx, cy, base + (h + 0.3) / 2))
                counts["tanks"] += 1
                continue
            green = f.get("use") == "greenhouse"
            h = 3.4 if green else float(f.get("height") or 9.0)
            walls, roof = ("greenhouse", "greenhouse") if green else ("facade", "roof")
            bw, br = bm_for(walls), bm_for(roof)
            top = base + 0.3 + h
            ring2 = ring if _area(ring) > 0 else ring[::-1]             # anticlockwise
            for (x0, y0), (x1, y1) in zip(ring2, ring2[1:] + ring2[:1]):
                vs = [bw.verts.new(v) for v in ((x0, y0, base), (x1, y1, base), (x1, y1, top), (x0, y0, top))]
                try:
                    bw.faces.new(vs)
                except ValueError:
                    pass
            try:
                br.faces.new([br.verts.new((x, y, top)) for x, y in ring2])
            except ValueError:
                pass
            counts["greenhouses" if green else "buildings"] += 1
        elif kind == "road":
            if f.get("raised"):
                lift = 0.15
            else:
                lift = 0.12
            w = float(f.get("width", 6.0)) / 2
            bm = bm_for("road_worn")
            for a, b in zip(pts[:-1], pts[1:]):
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                if in_keep(mx, my) or is_wet(mx, my):
                    continue
                dx, dy = b[0] - a[0], b[1] - a[1]
                L = math.hypot(dx, dy) or 1.0
                ox, oy = -dy / L * w, dx / L * w
                # long segments follow the ground in steps
                steps = max(1, int(L / 8))
                for s in range(steps):
                    t0, t1 = s / steps, (s + 1) / steps
                    p0 = (a[0] + dx * t0, a[1] + dy * t0)
                    p1 = (a[0] + dx * t1, a[1] + dy * t1)
                    quad = [(p0[0] - ox, p0[1] - oy), (p1[0] - ox, p1[1] - oy), (p1[0] + ox, p1[1] + oy), (p0[0] + ox, p0[1] + oy)]
                    try:
                        bm.faces.new([bm.verts.new((x, y, ground.at(x, y) + lift)) for x, y in quad])
                    except ValueError:
                        pass
            counts["roads"] += 1
        elif kind in ("green", "sand"):
            ring = _ring(pts)
            cx = sum(p[0] for p in ring) / len(ring)
            cy = sum(p[1] for p in ring) / len(ring)
            if len(ring) < 3 or in_keep(cx, cy):
                continue
            bm = bm_for("grass" if kind == "green" else "sand")
            try:
                fc = bm.faces.new([bm.verts.new((x, y, ground.at(x, y) + 0.06)) for x, y in ring])
                if fc.normal.z < 0:
                    fc.normal_flip()
            except ValueError:
                pass
    for mat, bm in meshes.items():
        me = bpy.data.meshes.new("around_" + mat)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(M.get(mat))
        ob = bpy.data.objects.new("around_" + mat, me)
        bpy.context.scene.collection.objects.link(ob)
    return counts


def boulders(objs, index: int, density: float, seed: int = 3, near=None, size=(0.9, 2.1), name: str = "armour") -> int:
    """Armour stone: single boulders laid over the faces of ``objs`` that carry material slot
    ``index`` (the rock slopes), ``density`` a square metre, half sunk into the slope, each one
    its own size, shape and turn. ``near(x, y)`` limits them to where they will be seen."""
    import random
    import bmesh
    import bpy
    from mathutils import Euler, Matrix, Vector
    rng = random.Random(seed)
    bm = bmesh.new()
    count = 0
    for ob in objs:
        mw = ob.matrix_world
        me = ob.data
        for poly in me.polygons:
            if (index is not None and poly.material_index != index) or poly.area <= 0:
                continue
            c = mw @ poly.center
            top = max((mw @ me.vertices[i].co).z for i in poly.vertices)
            if near is not None and not near(c.x, c.y, c.z, top):
                continue
            n_here = poly.area * density
            k = int(n_here) + (1 if rng.random() < n_here - int(n_here) else 0)
            if not k:
                continue
            vs = [mw @ me.vertices[i].co for i in poly.vertices]
            nrm = (mw.to_3x3() @ poly.normal).normalized()
            for _ in range(k):
                # a point in the face (fan triangles, by chance)
                t = rng.randrange(1, len(vs) - 1)
                a, b = rng.random(), rng.random()
                if a + b > 1:
                    a, b = 1 - a, 1 - b
                p = vs[0] + (vs[t] - vs[0]) * a + (vs[t + 1] - vs[0]) * b
                sz = rng.uniform(*size)
                m = (Matrix.Translation(p - nrm * sz * 0.25)
                     @ Euler((rng.uniform(0, 6.3), rng.uniform(0, 6.3), rng.uniform(0, 6.3))).to_matrix().to_4x4()
                     @ Matrix.Diagonal(Vector((sz * rng.uniform(0.85, 1.15) / 2, sz * rng.uniform(0.7, 1.0) / 2,
                                               sz * rng.uniform(0.5, 0.75) / 2, 1.0))))
                bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0, matrix=m)
                count += 1
    if count:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        me.materials.append(M.get("armour"))
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
    bm.free()
    return count


def _area(ring) -> float:
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]))


def _translate(x, y, z):
    from mathutils import Matrix
    return Matrix.Translation((x, y, z))


def build(manifest: dict[str, Any], folder: Path, sea: float, keep: tuple[float, float, float, float]) -> dict[str, Any]:
    """The modelled surroundings. ``keep`` is (x0, x1, y0, y1) of the works, left empty."""
    z, wet, r = heights(manifest, folder, sea, keep)
    ground = build_ground(z, wet, r, keep, sea)
    import bpy
    k0, k1, k2, k3 = keep
    cx, cy = (k0 + k1) / 2, (k2 + k3) / 2
    reach = max(k1 - k0, k3 - k2) * 2.5 + 150
    stones = 0
    feats = json.loads((folder / manifest["osm"]["file"]).read_text()) if "osm" in manifest else []
    counts = build_osm(feats, ground, wet, keep, sea)
    return {"radius": r, "sea_share": round(float(wet.mean()), 2), **counts, "shore_stones": stones, "ground": ground}


def rock_mound(pts: list[tuple[float, float]], width: float, top: float, bed: float,
               free: tuple[bool, bool] = (False, True), name: str = "breakwater") -> int:
    """A rock breakwater as TritoMoor draws one: along the line ``pts`` a crest ``width`` wide
    at ``top``, armour slopes of 1 in 1.5 each side down to ``bed``, in blocks of about 9 m of
    two tones of stone under a lighter crest, and a round head at each free end (``free`` for
    the first and last point); bends are rounded too. Returns the number of blocks."""
    import bmesh
    import bpy
    w, run = width / 2, SLOPE * (top - bed)
    top0 = top
    bm = bmesh.new()
    faces: list[tuple[list, int]] = []

    def quad(a, b, c, d, m):
        faces.append(([a, b, c, d], m))

    def head(e, ux, uy, sd, half: bool = True):
        nx, ny = -uy, ux
        # a bend's round lies over both arms' ends: lift it clear of their crests, or Cycles
        # shades the two coplanar faces black where they overlap
        top = top0 if half else top0 + 0.03
        steps = 12 if half else 24
        span = math.pi if half else 2 * math.pi
        for q in range(steps):
            a0 = -math.pi / 2 + q * span / steps
            a1 = a0 + span / steps
            d0 = (sd * ux * math.cos(a0) + nx * math.sin(a0), sd * uy * math.cos(a0) + ny * math.sin(a0))
            d1 = (sd * ux * math.cos(a1) + nx * math.sin(a1), sd * uy * math.cos(a1) + ny * math.sin(a1))
            t0 = (e[0] + d0[0] * w, e[1] + d0[1] * w, top)
            t1 = (e[0] + d1[0] * w, e[1] + d1[1] * w, top)
            g0 = (e[0] + d0[0] * (w + run), e[1] + d0[1] * (w + run), bed)
            g1 = (e[0] + d1[0] * (w + run), e[1] + d1[1] * (w + run), bed)
            faces.append(([(e[0], e[1], top), t0, t1], 2))
            quad(t0, g0, g1, t1, q % 2)

    blocks = 0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 0.01:
            continue
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        n = max(1, round(L / 9.0))
        for k in range(n):
            t0, t1 = k / n, (k + 1) / n
            p0 = (a[0] + dx * t0, a[1] + dy * t0)
            p1 = (a[0] + dx * t1, a[1] + dy * t1)

            def c(p, sd, z, o):
                return (p[0] + nx * sd * o, p[1] + ny * sd * o, z)
            quad(c(p0, -1, top, w), c(p1, -1, top, w), c(p1, 1, top, w), c(p0, 1, top, w), 2)
            quad(c(p0, 1, top, w), c(p1, 1, top, w), c(p1, 1, bed, w + run), c(p0, 1, bed, w + run), k % 2)
            quad(c(p1, -1, top, w), c(p0, -1, top, w), c(p0, -1, bed, w + run), c(p1, -1, bed, w + run), (k + 1) % 2)
            blocks += 1
        if i == 0 and free[0]:
            head(a, ux, uy, -1)
        if i == len(pts) - 2 and free[1]:
            head(b, ux, uy, 1)
        if i < len(pts) - 2:
            head(b, ux, uy, 1, half=False)            # a bend: round, so the two arms meet
    for vs, m in faces:
        try:
            f = bm.faces.new([bm.verts.new(v) for v in vs])
            f.material_index = m
        except ValueError:
            pass
    # every face of a mound looks up (its crest) or out and up (its slopes)
    for f in bm.faces:
        f.normal_update()
        if f.normal.z < 0:
            f.normal_flip()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in ("armour_a", "armour_b", "armour_top"):
        me.materials.append(M.get(m))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return blocks
