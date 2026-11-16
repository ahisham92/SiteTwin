"""Surroundings modelled the way the structure is, instead of a photograph or Google's mesh.

Draped aerial photographs and Google's photo-real mesh look soft and smeared next to a crisp
model, so here the place is rebuilt from data, in the same materials and light as the works:

* **Ground** from the terrain heights, in plain land, with rock along every shore (breakwaters,
  armour, the coast). The aerial photograph is never shown: it only says where the sea is, which
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


def heights(manifest: dict[str, Any], folder: Path, sea: float) -> tuple[np.ndarray, np.ndarray, float]:
    """The ground in the scene's levels (z), the sea mask, and the radius."""
    r = float(manifest["radius"])
    raw = np.load(folder / manifest["terrain"]["file"]) if "terrain" in manifest else np.zeros((129, 129))
    n = raw.shape[0]
    share = sea_share(folder / manifest["imagery"]["file"], n) if "imagery" in manifest else (raw < 0.5).astype(float)
    # Dry ground stays at least a metre up (coastal heights are coarse) and the sea bed drops
    # away. Between them the ground slopes through the waterline where the photograph's share of
    # water passes a half, so the coast is a smooth line where the sea meets a sloping shore, not
    # the steps of the height grid.
    t = np.clip((0.5 - share) / 0.2, -1.0, 1.0)               # 1 land, 0 the waterline, -1 sea
    land = np.maximum(raw, 1.0) + sea
    z = np.where(t >= 0, sea - 0.4 + t * (land - sea + 0.4), sea - 0.4 + t * 5.6)
    return z, t < -0.999, r


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
    rock = (s4.any(axis=1) | w4.any(axis=1) | (z4.min(axis=1) < sea + 0.9))[keep_face]
    faces = quads[keep_face]
    _mesh("land", verts, faces, ["land", "rock"], rock.astype(np.int32))
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


def _area(ring) -> float:
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]))


def _translate(x, y, z):
    from mathutils import Matrix
    return Matrix.Translation((x, y, z))


def build(manifest: dict[str, Any], folder: Path, sea: float, keep: tuple[float, float, float, float]) -> dict[str, Any]:
    """The modelled surroundings. ``keep`` is (x0, x1, y0, y1) of the works, left empty."""
    z, wet, r = heights(manifest, folder, sea)
    ground = build_ground(z, wet, r, keep, sea)
    feats = json.loads((folder / manifest["osm"]["file"]).read_text()) if "osm" in manifest else []
    counts = build_osm(feats, ground, wet, keep, sea)
    return {"radius": r, "sea_share": round(float(wet.mean()), 2), **counts, "ground": ground}
