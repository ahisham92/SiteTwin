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


def _asset(kind: str, height: float):
    folder = os.environ.get("SITETWIN_ASSETS")
    if not folder:
        return None
    path = Path(folder) / f"{kind}.glb"
    if not path.exists():
        return None
    import bpy
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new(kind, None)
    bpy.context.scene.collection.objects.link(root)
    top = 0.0
    for o in new:
        if o.parent is None:
            o.parent = root
        for v in getattr(o, "bound_box", []):
            top = max(top, (o.matrix_world @ __import__("mathutils").Vector(v)).z)
    if top > 0 and height:
        root.scale = (height / top,) * 3
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


def worker(vest: str = "hivis", seed: int = 0):
    a = _asset("worker", 1.75)
    if a is not None:
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


def site_cabin():
    p: list = [("box", "cabin", (0, 0, 1.35), (6.0, 2.4, 2.6), 0),
               ("box", "cab_glass", (0, -1.21, 1.6), (1.2, 0.03, 1.0), 0),
               ("box", "cab_glass", (2.0, -1.21, 1.6), (1.2, 0.03, 1.0), 0),
               ("box", "dark_metal", (-2.0, -1.21, 1.1), (0.9, 0.03, 2.0), 0)]
    return _build("site_cabin", p)
