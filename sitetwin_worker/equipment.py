"""Construction plant and crews, modelled in code so a scene never waits for assets.

Each builder returns one Blender object (parts joined, each keeping its material)
standing on its origin, facing +X. Proportions follow real machines: a Liebherr-
class flat-top tower crane, a mixer truck on a four-axle chassis, a piling rig with
a 20 m leader, a 20 t excavator, a truck-mounted boom pump, a scissor lift, and
people at 1.75 m in hi-vis and hard hats.

**Real models drop in.** Put a glTF/GLB named after the builder (``tower_crane.glb``,
``mixer_truck.glb``, ``worker.glb``...) in the folder ``SITETWIN_ASSETS`` points at,
facing +X with its base at the origin, and it is used instead. Photogrammetry or
manufacturer models are what take the film from convincing to real.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

from . import materials as M

Part = tuple[str, str, tuple, tuple, float]   # shape, material, centre, size, rotation about z


def _build(name: str, parts: Iterable[tuple], parent=None):
    """parts: (shape, material, (x,y,z) centre, (sx,sy,sz) size, rot_z [, rot_y])."""
    import bmesh
    import bpy
    from mathutils import Matrix

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    mats: list[str] = []
    for part in parts:
        shape, mat, c, s, rz = part[:5]
        ry = part[5] if len(part) > 5 else 0.0
        rx = part[6] if len(part) > 6 else 0.0
        if mat not in mats:
            mats.append(mat)
        idx = mats.index(mat)
        before = set(bm.faces)
        if shape == "box":
            geom = bmesh.ops.create_cube(bm, size=1.0)
        elif shape == "ball":
            geom = bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=0.5)
        else:  # cylinder along z
            geom = bmesh.ops.create_cone(bm, cap_ends=True, segments=16 if shape == "cyl" else 8,
                                         radius1=0.5, radius2=0.5, depth=1.0)
        verts = geom["verts"]
        m = (Matrix.Translation(c) @ Matrix.Rotation(rz, 4, "Z") @ Matrix.Rotation(ry, 4, "Y")
             @ Matrix.Rotation(rx, 4, "X") @ Matrix.Diagonal((*s, 1.0)))
        bmesh.ops.transform(bm, matrix=m, verts=verts)
        for f in set(bm.faces) - before:
            f.material_index = idx
    bm.to_mesh(mesh)
    bm.free()
    for mname in mats:
        mesh.materials.append(M.get(mname))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


#: Real models to use in place of the built-in shapes: kind -> glTF file (see assets.py).
MODELS: dict[str, str] = {}
#: Each model's meshes once imported and stood up: kind -> [mesh data].
_library: dict[str, list] = {}


def _model_file(kind: str) -> Path | None:
    if kind in MODELS and Path(MODELS[kind]).exists():
        return Path(MODELS[kind])
    folder = os.environ.get("SITETWIN_ASSETS")
    if folder and (Path(folder) / f"{kind}.glb").exists():
        return Path(folder) / f"{kind}.glb"
    return None


def _load(kind: str, path: Path, height: float, yaw: float, length: float | None = None) -> list:
    """Import the model once and stand it up: base on the ground, centred, front to +X,
    ``height`` tall (or ``length`` long, the longer side in plan, when given). Returns its
    meshes (shared by every copy in the scene)."""
    import bpy
    import mathutils
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = []
    for o in new:
        if o.type == "MESH":
            for m in list(o.modifiers):                       # a skinned model keeps its rest pose
                o.modifiers.remove(m)
            data = o.data.copy()
            data.transform(o.matrix_world)
            meshes.append(data)
    for o in new:
        bpy.data.objects.remove(o, do_unlink=True)
    if not meshes:
        return []
    turn = mathutils.Matrix.Rotation(math.radians(yaw), 4, "Z")
    for d in meshes:
        d.transform(turn)
    lo = [min(v.co[i] for d in meshes for v in d.vertices) for i in range(3)]
    hi = [max(v.co[i] for d in meshes for v in d.vertices) for i in range(3)]
    if length:
        k = length / max(1e-6, hi[0] - lo[0], hi[1] - lo[1])
    else:
        k = height / max(1e-6, hi[2] - lo[2]) if height else 1.0
    fit = mathutils.Matrix.Scale(k, 4) @ mathutils.Matrix.Translation(
        (-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    for d in meshes:
        d.transform(fit)
        d.name = f"model_{kind}"
    return meshes


def _split(meshes: list, cells: int = 300, trunks: bool = False) -> list[list]:
    """Cut a pack (several models side by side, often in one mesh) into its separate models.
    Every face is drawn on a plan grid ``cells`` across; touching patches of the grid are one
    model (a car's wheels lie under its body, a tree's leaves over its trunk). A face spanning
    much of the pack (a ground plane under them all) is dropped. With ``trunks`` (trees whose
    crowns overlap), only the lowest fifth is drawn on the grid, one patch per trunk, and every
    other face goes to the trunk nearest in plan."""
    import bmesh
    import numpy as np
    per = []
    for d in meshes:
        n = len(d.vertices)
        co = np.empty(n * 3)
        d.vertices.foreach_get("co", co)
        per.append(co.reshape(n, 3)[:, :2])
    allxy = np.concatenate(per) if per else np.zeros((1, 2))
    lo, hi = allxy.min(axis=0), allxy.max(axis=0)
    ext = float(max(hi - lo)) or 1.0
    cell = ext / cells
    nx, ny = (int((hi - lo)[0] / cell) + 3, int((hi - lo)[1] / cell) + 3)
    occ = np.zeros((nx, ny), bool)
    faces = []                                        # (mesh index, face index, cell) per face
    zs = [np.array([v.co.z for v in d.vertices]) for d in meshes]
    zall = np.concatenate(zs) if zs else np.zeros(1)
    low = float(zall.min() + 0.2 * (zall.max() - zall.min()))
    upper = []                                        # faces left for the nearest trunk
    for mi, (d, xy) in enumerate(zip(meshes, per)):
        for poly in d.polygons:
            vs = xy[list(poly.vertices)]
            span = vs.max(axis=0) - vs.min(axis=0)
            if max(span) > 0.4 * ext:
                continue
            if trunks and float(zs[mi][list(poly.vertices)].mean()) > low:
                upper.append((mi, poly.index, vs.mean(axis=0)))
                continue
            ij = ((vs - lo) / cell).astype(int) + 1
            i0, j0 = ij.min(axis=0)
            i1, j1 = ij.max(axis=0)
            occ[i0:i1 + 1, j0:j1 + 1] = True
            c = ((vs.mean(axis=0) - lo) / cell).astype(int) + 1
            faces.append((mi, poly.index, (int(c[0]), int(c[1]))))
    label = np.full(occ.shape, -1, int)
    n_lab = 0
    for i, j in zip(*np.nonzero(occ)):
        if label[i, j] >= 0:
            continue
        stack = [(i, j)]
        label[i, j] = n_lab
        while stack:
            a, b = stack.pop()
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    p, q = a + da, b + db
                    if 0 <= p < nx and 0 <= q < ny and occ[p, q] and label[p, q] < 0:
                        label[p, q] = n_lab
                        stack.append((p, q))
        n_lab += 1
    keep: dict[int, dict[int, set]] = {}
    for mi, fi, (i, j) in faces:
        keep.setdefault(int(label[i, j]), {}).setdefault(mi, set()).add(fi)
    if upper and n_lab:
        centre = np.array([np.argwhere(label == k).mean(axis=0) for k in range(n_lab)]) * cell + lo - cell
        for mi, fi, c in upper:
            k = int(np.argmin(((centre - c) ** 2).sum(axis=1)))
            keep.setdefault(k, {}).setdefault(mi, set()).add(fi)
    sizes = {k: sum(len(v) for v in m.values()) for k, m in keep.items()}
    big = max(sizes.values()) if sizes else 0
    out = []
    for k, by_mesh in keep.items():
        if k < 0 or sizes[k] < 0.02 * big:                # crumbs, not a model
            continue
        group = []
        for mi, fis in by_mesh.items():
            bm = bmesh.new()
            bm.from_mesh(meshes[mi])
            bm.faces.ensure_lookup_table()
            bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in fis], context="FACES")
            d = meshes[mi].copy()
            bm.to_mesh(d)
            bm.free()
            group.append(d)
        out.append(group)
    return out


def _load_pack(kind: str, path: Path, height: float, yaw: float, length: float | None) -> list[list]:
    """A pack of several models side by side in one file (cars, trees): each one stood up and
    sized on its own, as ``_load`` does for a single model. Returns a list of mesh lists."""
    import mathutils
    meshes = _load(kind, path, 0.0, yaw, None)                # as modelled, rotated
    out = []
    for group in _split(meshes, trunks=not length):
        lo = [min(v.co[i] for d in group for v in d.vertices) for i in range(3)]
        hi = [max(v.co[i] for d in group for v in d.vertices) for i in range(3)]
        size = max(hi[0] - lo[0], hi[1] - lo[1])
        if size < 1e-3 or (hi[2] - lo[2]) < 1e-3:
            continue
        k = (length / size) if length else (height / (hi[2] - lo[2]))
        fit = mathutils.Matrix.Scale(k, 4) @ mathutils.Matrix.Translation(
            (-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
        if length and (hi[1] - lo[1]) > (hi[0] - lo[0]):     # a car lies along X, as the scene expects
            fit = mathutils.Matrix.Rotation(-math.pi / 2, 4, "Z") @ fit
        copies = []
        for d in group:
            c = d.copy()
            c.transform(fit)
            copies.append(c)
        out.append(copies)
    return out


def variant(kind: str, n: int):
    """The n-th model of a pack (wrapping round), as a new object; None when there is none."""
    path = _model_file(kind)
    if path is None:
        return None
    import bpy
    key = kind + ":pack"
    if key not in _library:
        try:
            from .assets import CATALOG
            entry = CATALOG.get(kind, {})
            _library[key] = _load_pack(kind, path, float(entry.get("height", 1.0)),
                                       float(entry.get("yaw", 0.0)), entry.get("length"))
        except Exception:                                     # noqa: BLE001 - nothing stands in
            _library[key] = []
    packs = _library[key]
    if not packs:
        return None
    root = bpy.data.objects.new(f"{kind}_{n % len(packs)}", None)
    bpy.context.scene.collection.objects.link(root)
    for d in packs[n % len(packs)]:
        o = bpy.data.objects.new(f"{kind}_part", d)
        bpy.context.scene.collection.objects.link(o)
        o.parent = root
    return root


def _asset(kind: str, height: float):
    """A copy of the real model of ``kind`` (sharing its meshes with the other copies), or None
    when there is none and the built-in shape is used."""
    path = _model_file(kind)
    if path is None:
        return None
    import bpy
    if kind not in _library:
        try:
            from .assets import CATALOG
            entry = CATALOG.get(kind, {})
            _library[kind] = _load(kind, path, float(entry.get("height", height)),
                                   float(entry.get("yaw", 0.0)), entry.get("length"))
        except Exception:                                     # noqa: BLE001 - the built-in shape stands in
            _library[kind] = []
    if not _library[kind]:
        return None
    root = bpy.data.objects.new(kind, None)
    bpy.context.scene.collection.objects.link(root)
    for d in _library[kind]:
        o = bpy.data.objects.new(f"{kind}_part", d)
        bpy.context.scene.collection.objects.link(o)
        o.parent = root
    return root


def _lattice(parts: list, mat: str, x0, y0, z0, w, d, h, member=0.12, panel=None, along="z"):
    """A square lattice mast: four chords, horizontals and diagonals."""
    panel = panel or w
    n = max(1, int(round((h if along == "z" else w) / panel)))
    if along == "z":
        for dx, dy in ((0, 0), (w, 0), (0, d), (w, d)):
            parts.append(("box", mat, (x0 + dx, y0 + dy, z0 + h / 2), (member, member, h), 0))
        for i in range(n + 1):
            z = z0 + i * h / n
            parts.append(("box", mat, (x0 + w / 2, y0, z), (w, member * 0.7, member * 0.7), 0))
            parts.append(("box", mat, (x0 + w / 2, y0 + d, z), (w, member * 0.7, member * 0.7), 0))
            parts.append(("box", mat, (x0, y0 + d / 2, z), (member * 0.7, d, member * 0.7), 0))
            parts.append(("box", mat, (x0 + w, y0 + d / 2, z), (member * 0.7, d, member * 0.7), 0))
            if i < n:
                diag = math.hypot(w, h / n)
                ang = math.atan2(h / n, w)
                zc = z + h / n / 2
                for yy in (y0, y0 + d):
                    parts.append(("box", mat, (x0 + w / 2, yy, zc), (diag, member * 0.6, member * 0.6), 0,
                                  -ang if i % 2 == 0 else ang))
                for xx in (x0, x0 + w):
                    parts.append(("box", mat, (xx, y0 + d / 2, zc), (member * 0.6, diag, member * 0.6), 0, 0,
                                  ang if i % 2 == 0 else -ang))


def tower_crane(height: float, jib: float):
    """Returns (mast object, slewing object, trolley object, hook object). The slewing
    part turns about z at the mast top; the trolley runs along +X of it; the hook
    hangs from the trolley (scale its z to pay out line)."""
    import bpy
    asset = _asset("tower_crane", height)
    if asset is not None:
        return asset, asset, None, None
    w = 2.0
    mast_parts: list = [("box", "concrete", (0, 0, 0.6), (6.5, 6.5, 1.2), 0)]
    _lattice(mast_parts, "crane_yellow", -w / 2, -w / 2, 1.2, w, w, height - 1.2, member=0.16, panel=2.5)
    mast = _build("tower_crane_mast", mast_parts)

    slew_parts: list = [
        ("box", "crane_yellow", (0, 0, 0.6), (3.0, 3.0, 1.2), 0),                     # slewing ring/turntable
        ("box", "truck_white", (1.4, -1.8, 1.6), (2.2, 1.6, 2.0), 0),                  # cab
        ("box", "cab_glass", (2.52, -1.8, 1.8), (0.05, 1.4, 1.2), 0),
    ]
    # flat-top jib: triangular-ish lattice, 1.6 m deep, along +X
    depth, width = 1.6, 1.4
    seg = 2.5
    n = int(jib / seg)
    for yy in (-width / 2, width / 2):
        slew_parts.append(("box", "crane_yellow", (jib / 2, yy, 1.2), (jib, 0.18, 0.18), 0))
    slew_parts.append(("box", "crane_yellow", (jib / 2, 0, 1.2 + depth), (jib, 0.16, 0.16), 0))
    for i in range(n + 1):
        x = i * seg
        slew_parts.append(("box", "crane_yellow", (x, 0, 1.2), (0.1, width, 0.1), 0))
        for yy in (-width / 2, width / 2):
            l = math.hypot(depth, width / 2)
            slew_parts.append(("box", "crane_yellow", (x, yy / 2, 1.2 + depth / 2), (0.08, 0.08, l), 0, 0,
                               math.atan2(width / 2, depth) * (1 if yy < 0 else -1)))
        if i < n:
            l = math.hypot(seg, depth)
            slew_parts.append(("box", "crane_yellow", (x + seg / 2, 0, 1.2 + depth / 2), (l, 0.08, 0.08), 0,
                               -math.atan2(depth, seg) * (1 if i % 2 else -1)))
    # counter-jib with ballast
    cj = max(8.0, jib * 0.3)
    slew_parts += [
        ("box", "crane_yellow", (-cj / 2, -0.7, 1.2), (cj, 0.25, 0.4), 0),
        ("box", "crane_yellow", (-cj / 2, 0.7, 1.2), (cj, 0.25, 0.4), 0),
        ("box", "steel", (-cj / 2, 0, 1.25), (cj, 1.4, 0.08), 0),
        ("box", "concrete", (-cj + 1.4, 0, 2.2), (2.2, 2.2, 2.0), 0),
        ("box", "concrete", (-cj + 3.7, 0, 2.2), (2.2, 2.2, 2.0), 0),
        ("box", "dark_metal", (-cj + 6.0, 0, 1.9), (1.8, 1.2, 1.2), 0),              # hoist winch
    ]
    slew = _build("tower_crane_slew", slew_parts, parent=mast)
    slew.location = (0, 0, height)

    trolley = _build("tower_crane_trolley", [("box", "crane_yellow", (0, 0, 0.9), (1.4, 1.6, 0.5), 0)], parent=slew)
    trolley.location = (jib * 0.5, 0, 0)
    hook = _build("tower_crane_hook", [
        ("cyl", "dark_metal", (-0.12, 0, -0.5), (0.03, 0.03, 1.0), 0),
        ("cyl", "dark_metal", (0.12, 0, -0.5), (0.03, 0.03, 1.0), 0),
    ], parent=trolley)
    hook.location = (0, 0, 0.7)
    block = _build("tower_crane_block", [("box", "drum_orange", (0, 0, -0.4), (0.7, 0.4, 0.8), 0),
                                         ("cyl", "dark_metal", (0, 0, -1.0), (0.18, 0.18, 0.4), 0)], parent=trolley)
    return mast, slew, trolley, (hook, block)


def _wheels(parts: list, xs: Iterable[float], track: float, r: float = 0.52, w: float = 0.4):
    for x in xs:
        for y in (-track / 2, track / 2):
            parts.append(("cyl", "rubber", (x, y, r), (2 * r, 2 * r, w), 0, 0, math.pi / 2))


def _cab(parts: list, x: float, colour: str = "truck_white", h: float = 3.0):
    parts += [
        ("box", colour, (x, 0, 1.0 + h / 2 - 0.3), (1.9, 2.45, h - 0.6), 0),
        ("box", "cab_glass", (x + 0.96, 0, 1.0 + h - 1.1), (0.04, 2.2, 0.9), 0),
        ("box", "cab_glass", (x, 1.23, 1.0 + h - 1.1), (1.4, 0.03, 0.8), 0),
        ("box", "cab_glass", (x, -1.23, 1.0 + h - 1.1), (1.4, 0.03, 0.8), 0),
        ("box", "dark_metal", (x + 0.98, 0, 1.0), (0.1, 2.4, 0.4), 0),
    ]


def mixer_truck():
    a = _asset("mixer_truck", 3.9)
    if a is not None:
        return a
    p: list = [("box", "dark_metal", (-1.5, 0, 0.95), (8.0, 1.0, 0.35), 0)]
    _cab(p, 2.0)
    _wheels(p, (2.0, -1.4, -2.8, -4.2), 2.0)
    p += [("cyl", "drum_orange", (-2.1, 0, 2.55), (2.3, 2.3, 4.6), 0, math.pi / 2 - 0.18),
          ("cyl", "drum_orange", (-4.4, 0, 2.9), (1.2, 1.2, 0.8), 0, math.pi / 2 - 0.18),
          ("box", "dark_metal", (-5.2, 0, 2.6), (0.8, 0.6, 0.5), 0)]
    return _build("mixer_truck", p)


def flatbed_truck():
    a = _asset("flatbed_truck", 3.2)
    if a is not None:
        return a
    p: list = [("box", "dark_metal", (-2.5, 0, 0.95), (10.0, 1.0, 0.35), 0),
               ("box", "wood", (-3.6, 0, 1.3), (7.6, 2.5, 0.25), 0),
               ("box", "concrete", (-3.6, 0, 1.75), (6.5, 2.2, 0.6), 0)]   # a load of precast
    _cab(p, 2.0, colour="steel_red")
    _wheels(p, (2.0, 0.6, -5.0, -6.3), 2.0)
    return _build("flatbed_truck", p)


def concrete_pump():
    a = _asset("concrete_pump", 4.0)
    if a is not None:
        return a
    p: list = [("box", "dark_metal", (-1.5, 0, 0.95), (9.0, 1.0, 0.35), 0),
               ("box", "truck_white", (-2.5, 0, 1.6), (6.0, 2.3, 1.0), 0)]
    _cab(p, 2.2)
    _wheels(p, (2.2, 0.8, -2.8, -4.2), 2.0)
    # outriggers and the boom unfolded up and over
    for x in (-0.2, -4.8):
        for y in (-1, 1):
            p.append(("box", "plant_yellow", (x, y * 2.6, 0.4), (0.3, 3.0, 0.3), 0))
    p += [("cyl", "steel", (-1.2, 0, 2.6), (1.4, 1.4, 1.0), 0),
          ("box", "truck_white", (-1.2 + 5.5, 0, 2.6 + 6.0), (13.0, 0.5, 0.6), 0, -0.85),
          ("box", "truck_white", (-1.2 + 15.5, 0, 2.6 + 11.0), (11.0, 0.45, 0.5), 0, 0.15),
          ("box", "truck_white", (-1.2 + 21.0, 0, 2.6 + 6.0), (9.0, 0.4, 0.45), 0, 1.25),
          ("cyl", "rubber", (-1.2 + 22.0, 0, 2.6 + 0.5), (0.25, 0.25, 3.0), 0)]
    return _build("concrete_pump", p)


def piling_rig():
    a = _asset("piling_rig", 22.0)
    if a is not None:
        return a
    p: list = [("box", "dark_metal", (0, -1.6, 0.55), (6.0, 0.9, 1.1), 0),
               ("box", "dark_metal", (0, 1.6, 0.55), (6.0, 0.9, 1.1), 0),
               ("box", "plant_yellow", (-0.6, 0, 1.9), (4.4, 3.0, 1.6), 0),
               ("box", "plant_yellow", (0.8, 1.0, 3.2), (1.6, 1.0, 1.2), 0),
               ("box", "cab_glass", (1.62, 1.0, 3.3), (0.04, 0.9, 0.8), 0),
               ("box", "plant_yellow", (2.6, 0, 11.5), (0.8, 0.8, 21.0), 0),        # leader
               ("box", "dark_metal", (2.6, 0, 7.0), (1.2, 1.2, 2.0), 0),            # rotary head
               ("cyl", "steel", (2.6, 0, 3.0), (0.6, 0.6, 6.0), 0),                 # auger
               ("box", "plant_yellow", (1.5, 0, 6.0), (0.3, 0.3, 7.0), 0, 0.35)]    # backstay
    return _build("piling_rig", p)


def excavator():
    a = _asset("excavator", 3.2)
    if a is not None:
        return a
    p: list = [("box", "dark_metal", (0, -1.2, 0.45), (4.4, 0.7, 0.9), 0),
               ("box", "dark_metal", (0, 1.2, 0.45), (4.4, 0.7, 0.9), 0),
               ("box", "plant_yellow", (-0.4, 0, 1.6), (3.6, 2.6, 1.2), 0),
               ("box", "plant_yellow", (0.6, 0.8, 2.7), (1.2, 1.0, 1.6), 0),
               ("box", "cab_glass", (1.22, 0.8, 2.8), (0.04, 0.9, 1.0), 0),
               ("box", "plant_yellow", (2.6, -0.2, 3.0), (4.2, 0.5, 0.7), 0, -0.55),
               ("box", "plant_yellow", (5.0, -0.2, 2.4), (2.8, 0.4, 0.5), 0, 1.0),
               ("box", "dark_metal", (5.6, -0.2, 0.9), (1.0, 1.0, 0.8), 0)]
    return _build("excavator", p)


def mewp():
    a = _asset("mewp", 9.0)
    if a is not None:
        return a
    p: list = [("box", "plant_yellow", (0, 0, 0.6), (2.5, 1.2, 0.7), 0)]
    _wheels(p, (0.8, -0.8), 1.1, r=0.3, w=0.25)
    for i in range(6):
        z = 1.2 + i * 1.1
        ang = 0.45 if i % 2 == 0 else -0.45
        p.append(("box", "steel", (0, 0, z + 0.55), (2.4, 1.0, 0.12), 0, ang))
    p += [("box", "plant_yellow", (0, 0, 8.0), (2.6, 1.2, 0.15), 0),
          ("box", "plant_yellow", (0, 0, 8.6), (2.6, 1.2, 0.05), 0)]
    return _build("mewp", p)


def slings():
    """Four chain legs from the hook down to the corners of a load, 1.5 m long."""
    parts: list = [("ball", "dark_metal", (0, 0, 0), (0.18, 0.18, 0.18), 0)]
    spread = 0.55
    tilt = math.atan2(spread, 1.5)
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        parts.append(("cyl8", "dark_metal", (math.cos(a) * spread / 2, math.sin(a) * spread / 2, -0.75),
                      (0.03, 0.03, 1.6), a, -tilt))
    return _build("slings", parts)


def worker(vest: str = "hivis", seed: int = 0):
    a = _asset("worker_b" if seed % 2 and _model_file("worker_b") else "worker", 1.75)
    if a is not None:
        # a scanned person, dressed for site: hi-vis vest over the torso and a hard hat
        h = max((v.co.z for c in a.children if c.type == "MESH" for v in c.data.vertices), default=1.75)
        gear = _build("worker_gear", [
            ("cyl", vest, (0, 0, h * 0.66), (0.40, 0.50, h * 0.21), 0),
            ("ball", "helmet", (0, 0, h * 0.975), (0.27, 0.25, 0.17), 0),
            ("cyl", "helmet", (0.03, 0, h * 0.955), (0.32, 0.29, 0.02), 0),     # brim
        ])
        gear.parent = a
        return a
    lean = ((seed * 37) % 11 - 5) * 0.02
    p: list = [
        ("box", "workwear", (0, -0.1, 0.43), (0.16, 0.15, 0.86), 0),   # legs
        ("box", "workwear", (0, 0.1, 0.43), (0.16, 0.15, 0.86), 0),
        ("box", "dark_metal", (0.04, -0.1, 0.05), (0.28, 0.13, 0.1), 0),   # boots
        ("box", "dark_metal", (0.04, 0.1, 0.05), (0.28, 0.13, 0.1), 0),
        ("box", vest, (0, 0, 1.18), (0.26, 0.46, 0.62), 0, lean),     # torso in vest
        ("box", vest, (0, -0.29, 1.12), (0.12, 0.11, 0.6), 0),        # arms
        ("box", vest, (0, 0.29, 1.12), (0.12, 0.11, 0.6), 0),
        ("ball", "skin", (0, 0, 1.6), (0.21, 0.19, 0.24), 0),         # head
        ("cyl", "helmet", (0, 0, 1.72), (0.28, 0.26, 0.12), 0),
    ]
    return _build("worker", p)


def prop(kind: str):
    """A real model of a site prop (pallet, skip, barrier...), or None when there is none."""
    return _asset(kind, 1.0)


def site_cabin():
    p: list = [("box", "cabin", (0, 0, 1.35), (6.0, 2.4, 2.6), 0),
               ("box", "cab_glass", (0, -1.21, 1.6), (1.2, 0.03, 1.0), 0),
               ("box", "cab_glass", (2.0, -1.21, 1.6), (1.2, 0.03, 1.0), 0),
               ("box", "dark_metal", (-2.0, -1.21, 1.1), (0.9, 0.03, 2.0), 0)]
    return _build("site_cabin", p)
