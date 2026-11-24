"""A drone survey of the site as its surroundings: the real place, as it was flown.

A survey is a textured 3D mesh made from drone photos or a drone video by photogrammetry
(OpenDroneMap, Pix4D, DJI Terra, Metashape, RealityCapture...). It comes in as

* a ``.zip`` holding an ``.obj`` (with its ``.mtl`` and texture images), or a ``.glb``;
* a ``.glb`` on its own.

Where it stands on Earth is read from the package when it says so:

* OpenDroneMap: ``odm_georeferencing/coords.txt`` (``WGS84 UTM 36N`` and the easting/northing the
  ``*_geo.obj`` is relative to), or the ``survey.json`` our own drone step writes;
* Pix4D and others: an ``*offset.xyz`` file (easting northing height) with a ``.prj`` naming the
  UTM zone, or ``survey.json``.

Without that (a video flown with no GPS track) it is set on the site's own point, and the
person places it by hand: ``place`` = east/north/up offsets in metres, a turn in degrees and a
scale. Heights are matched automatically: the survey's ground just outside the works is set to
the level of the scene's own ground there, so a drone's height datum never matters.

In the scene the survey replaces the modelled or Google surroundings wherever it reaches: those
are carved away under it, and the works area (and, at sea, the water, which photogrammetry
cannot rebuild) is cut out of the survey, so the sequence builds on clean ground.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .geo import Local

MESH_TYPES = (".obj", ".glb", ".gltf", ".ply")


# ---- UTM, in pure Python (Krüger's series, to well under a millimetre) --------------------------

_A = 6378137.0
_F = 1 / 298.257223563
_K0 = 0.9996


def _series():
    n = _F / (2 - _F)
    a_ = _A / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    beta = (n / 2 - 2 * n ** 2 / 3 + 37 * n ** 3 / 96, n ** 2 / 48 + n ** 3 / 15, 17 * n ** 3 / 480)
    delta = (2 * n - 2 * n ** 2 / 3 - 2 * n ** 3, 7 * n ** 2 / 3 - 8 * n ** 3 / 5, 56 * n ** 3 / 15)
    alpha = (n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16, 13 * n ** 2 / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240)
    return n, a_, alpha, beta, delta


def utm_to_lonlat(easting: float, northing: float, zone: int, north: bool = True) -> tuple[float, float]:
    """(longitude, latitude) in degrees of a WGS84 UTM point."""
    n, a_, _, beta, delta = _series()
    xi = (northing - (0.0 if north else 10000000.0)) / (_K0 * a_)
    eta = (easting - 500000.0) / (_K0 * a_)
    xi_ = xi - sum(b * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j, b in enumerate(beta, 1))
    eta_ = eta - sum(b * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j, b in enumerate(beta, 1))
    chi = math.asin(math.sin(xi_) / math.cosh(eta_))
    lat = chi + sum(d * math.sin(2 * j * chi) for j, d in enumerate(delta, 1))
    lon0 = math.radians(zone * 6 - 183)
    lon = lon0 + math.atan2(math.sinh(eta_), math.cos(xi_))
    return math.degrees(lon), math.degrees(lat)


def lonlat_to_utm(lon: float, lat: float, zone: int | None = None) -> tuple[float, float, int, bool]:
    """(easting, northing, zone, north) of a point (for tests and for placing by hand)."""
    n, a_, alpha, _, _ = _series()
    zone = zone or int((lon + 180) // 6) + 1
    phi, lam = math.radians(lat), math.radians(lon - (zone * 6 - 183))
    e = math.sqrt(_F * (2 - _F))
    t = math.sinh(math.atanh(math.sin(phi)) - e * math.atanh(e * math.sin(phi)))
    xi_ = math.atan2(t, math.cos(lam))
    eta_ = math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    xi = xi_ + sum(al * math.sin(2 * j * xi_) * math.cosh(2 * j * eta_) for j, al in enumerate(alpha, 1))
    eta = eta_ + sum(al * math.cos(2 * j * xi_) * math.sinh(2 * j * eta_) for j, al in enumerate(alpha, 1))
    return 500000.0 + _K0 * a_ * eta, _K0 * a_ * xi + (0.0 if lat >= 0 else 10000000.0), zone, lat >= 0


# ---- what the package says about itself --------------------------------------------------------

@dataclass
class Georef:
    """The survey's mesh coordinates are UTM (zone, north) minus ``offset`` (easting, northing)."""
    zone: int
    north: bool
    offset: tuple[float, float]


@dataclass
class Survey:
    mesh: Path
    georef: Georef | None = None
    name: str = ""
    notes: list[str] = field(default_factory=list)
    #: are the mesh's units metres? (not for a model made with no GPS: its size is unknown)
    metric: bool = True
    #: its width and depth in its own units, about its centre (after ``normalise``)
    extent: tuple[float, float] = (0.0, 0.0)


def parse_crs(text: str) -> tuple[int, bool] | None:
    """The UTM zone named in ``WGS84 UTM 36N``, an EPSG code (326zz / 327zz) or a WKT/.prj."""
    t = text.upper()
    m = re.search(r"UTM[ _]*(?:ZONE[ _]*)?(\d{1,2})\s*([NS])", t)
    if m:
        return int(m.group(1)), m.group(2) == "N"
    m = re.search(r"\b32([67])(\d{2})\b", t)
    if m:
        return int(m.group(2)), m.group(1) == "6"
    m = re.search(r"UTM[ _]*(?:ZONE[ _]*)?(\d{1,2})\b.*?(NORTH|SOUTH)", t)
    if m:
        return int(m.group(1)), m.group(2) == "NORTH"
    return None


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)]


def read_georef(folder: Path) -> tuple[Georef | None, list[str]]:
    """Where the unpacked survey in ``folder`` stands, if it says."""
    notes: list[str] = []
    own = next(folder.rglob("survey.json"), None)
    if own:
        info = json.loads(own.read_text())
        if isinstance(info.get("georef"), dict):
            g = info["georef"]
            return Georef(int(g["zone"]), bool(g.get("north", True)),
                          (float(g["offset"][0]), float(g["offset"][1]))), ["placed by its GPS"]
        if info.get("utm_zone"):
            return Georef(int(info["utm_zone"]), bool(info.get("north", True)),
                          (float(info["offset"][0]), float(info["offset"][1]))), ["placed by its GPS"]
    coords = next(folder.rglob("coords.txt"), None)           # OpenDroneMap
    if coords:
        lines = [ln.strip() for ln in coords.read_text().splitlines() if ln.strip()]
        crs = parse_crs(lines[0]) if lines else None
        nums = _numbers(lines[1]) if len(lines) > 1 else []
        if crs and len(nums) >= 2:
            return Georef(crs[0], crs[1], (nums[0], nums[1])), ["placed by its GPS (OpenDroneMap)"]
        notes.append("coords.txt was not understood")
    offset = next((p for p in folder.rglob("*") if p.name.lower().endswith("offset.xyz")), None)
    prj = next(folder.rglob("*.prj"), None)
    if offset and prj:                                        # Pix4D and the like
        crs = parse_crs(prj.read_text(errors="ignore"))
        nums = _numbers(offset.read_text())
        if crs and len(nums) >= 2:
            return Georef(crs[0], crs[1], (nums[0], nums[1])), ["placed by its offset file"]
    return None, notes + ["no position in the package: set on the site's point, to be placed by hand"]


def pick_mesh(folder: Path) -> Path:
    """The textured model in an unpacked package: OpenDroneMap's georeferenced one first, then the
    biggest mesh file."""
    found = [p for p in folder.rglob("*") if p.suffix.lower() in MESH_TYPES and p.is_file()]
    if not found:
        raise ValueError("The survey has no 3D model in it (.obj, .glb, .gltf or .ply).")
    for want in ("odm_textured_model_geo.obj", "odm_textured_model_geo.glb", "odm_textured_model.obj"):
        for p in found:
            if p.name == want:
                return p
    textured = [p for p in found if p.suffix.lower() != ".ply"] or found
    return max(textured, key=lambda p: p.stat().st_size)


def unpack(path: Path, folder: Path) -> Survey:
    """Open a survey file (``.zip`` or a mesh) into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                target = (folder / info.filename).resolve()
                if not str(target).startswith(str(folder.resolve())):
                    raise ValueError("The survey zip has a file outside its own folder.")
            z.extractall(folder)
    else:
        (folder / path.name).write_bytes(path.read_bytes())
    mesh = pick_mesh(folder)
    geo, notes = read_georef(folder)
    return Survey(mesh=mesh, georef=geo, name=path.name, notes=notes)


# ---- from the survey's coordinates to the scene's -----------------------------------------------

def scale_of(survey: Survey, place: dict[str, Any] | None) -> float:
    """Metres per unit of the survey's mesh: 1 for a georeferenced or metric survey (times any
    ``scale`` given); for one made with no GPS, the ``width`` (m) the person gave over its width."""
    place = place or {}
    s = float(place.get("scale") or 1.0)
    if not survey.metric and survey.georef is None:
        width = float(place.get("width") or 0.0)
        if width > 0 and survey.extent[0] > 0:
            s *= width / survey.extent[0]
    return s


def frame(survey: Survey, lat: float, lon: float, place: dict[str, Any] | None = None) -> list[list[float]]:
    """The 4x4 matrix that carries the survey's mesh coordinates (z up) into the scene's metres
    east and north of (lat, lon), before the automatic height match. With a georeference it is
    fitted through three UTM points (so it carries the grid's turn from true north); without, the
    mesh's own origin (its middle, after ``normalise``) goes on the site's point. ``place`` adds
    the hand placement: ``east`` and ``north`` (m), ``turn`` (degrees clockwise, as on a map),
    ``width`` (m, for a survey with no GPS) or ``scale``; its ``up`` is added after the heights
    are matched (see build)."""
    place = place or {}
    if survey.georef is not None:
        g = survey.georef
        local = Local(lat, lon)
        pts = []
        for dx, dy in ((0.0, 0.0), (100.0, 0.0), (0.0, 100.0)):
            lo, la = utm_to_lonlat(g.offset[0] + dx, g.offset[1] + dy, g.zone, g.north)
            pts.append(local.to_xy(lo, la))
        (x0, y0), (x1, y1), (x2, y2) = pts
        a, c = (x1 - x0) / 100.0, (y1 - y0) / 100.0              # where mesh +x goes
        b, d = (x2 - x0) / 100.0, (y2 - y0) / 100.0              # where mesh +y goes
        base = [[a, b, 0.0, x0], [c, d, 0.0, y0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    else:
        base = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    s = scale_of(survey, place)
    t = -math.radians(float(place.get("turn") or 0.0))       # clockwise on the map
    ct, st = math.cos(t) * s, math.sin(t) * s
    hand = [[ct, -st, 0.0, float(place.get("east") or 0.0)],
            [st, ct, 0.0, float(place.get("north") or 0.0)],
            [0.0, 0.0, s, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
    return [[sum(hand[i][k] * base[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Anticlockwise hull of 2D points (Andrew's monotone chain)."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def shrink(hull: list[tuple[float, float]], by: float) -> list[tuple[float, float]]:
    """The hull pulled in towards its centre by ``by`` metres (roughly), so the ground it replaces
    overlaps the survey's ragged edge rather than leaving a gap."""
    if not hull:
        return hull
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    out = []
    for x, y in hull:
        d = math.hypot(x - cx, y - cy)
        k = max(0.0, d - by) / d if d else 0.0
        out.append((cx + (x - cx) * k, cy + (y - cy) * k))
    return out


def inside(poly: list[tuple[float, float]], x: float, y: float) -> bool:
    """Is (x, y) inside the polygon (even-odd rule)?"""
    hit = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def outside_box(box: tuple[float, float, float, float], x: float, y: float) -> float:
    """How far (x, y) is outside the box (x0, y0, x1, y1); 0 inside."""
    x0, y0, x1, y1 = box
    return math.hypot(max(x0 - x, x - x1, 0.0), max(y0 - y, y - y1, 0.0))


# ---- into Blender ------------------------------------------------------------------------------

def _import(mesh: Path) -> list:
    import bpy
    before = set(bpy.data.objects)
    suffix = mesh.suffix.lower()
    if suffix == ".obj":
        # Blender's OBJ reader turns Y-up files to Z-up by default; drone surveys are Z-up
        bpy.ops.wm.obj_import(filepath=str(mesh), forward_axis="Y", up_axis="Z")
    elif suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(mesh))          # glTF is Y-up: the importer turns it
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(mesh), forward_axis="Y", up_axis="Z")
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.data.transform(mw)
        o.matrix_world.identity()
    for o in new:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    return meshes


def _ray_down(objs: list, x: float, y: float) -> float | None:
    """The highest point of ``objs`` straight below (x, y), or None."""
    from mathutils import Vector
    best = None
    for o in objs:
        ok, loc, _, _ = o.ray_cast(Vector((x, y, 5000.0)), Vector((0.0, 0.0, -1.0)))
        if ok and (best is None or loc.z > best):
            best = loc.z
    return best


def _matte(objs: list) -> None:
    """Photogrammetry textures already carry the light of the day they were flown: keep them, but
    make every material matte, so the scene's own sun does not add a sheen on top."""
    for o in objs:
        for slot in o.material_slots:
            m = slot.material
            if m is None or not m.use_nodes:
                continue
            for node in m.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    for name, v in (("Roughness", 0.92), ("Specular IOR Level", 0.15), ("Metallic", 0.0)):
                        if name in node.inputs and not node.inputs[name].is_linked:
                            node.inputs[name].default_value = v
        for p in o.data.polygons:
            p.use_smooth = False


def carve(objs: list, cover: list, skip=("sea", "water", "skirt", "runout", "base")) -> int:
    """Delete the faces of ``objs`` (the modelled or Google surroundings) that have the survey
    (``cover``) straight above or below them: where the survey has a hole, the ground under it
    stays. Returns how many went."""
    import bmesh
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    trees = []
    for c in cover:
        bm = bmesh.new()
        bm.from_mesh(c.data)
        trees.append(BVHTree.FromBMesh(bm))
        bm.free()
    up, down = Vector((0.0, 0.0, 1.0)), Vector((0.0, 0.0, -1.0))

    def covered(p) -> bool:
        for t in trees:
            for d in (up, down):
                hit = t.ray_cast(p, d, 400.0)
                if hit[0] is not None:
                    return True
        return False

    gone = 0
    for o in objs:
        if o.type != "MESH" or any(k in o.name.lower() for k in skip) or o.name not in bpy.data.objects:
            continue
        if o.name.startswith("survey_"):
            continue
        mw = o.matrix_world
        bm = bmesh.new()
        bm.from_mesh(o.data)
        dead = [f for f in bm.faces if covered(mw @ f.calc_center_median())]
        if dead:
            bmesh.ops.delete(bm, geom=dead, context="FACES")
            bm.to_mesh(o.data)
            gone += len(dead)
        bm.free()
        if not o.data.polygons:
            bpy.data.objects.remove(o, do_unlink=True)
    return gone


def build(survey: Survey, lat: float, lon: float, keep_out: tuple[float, float, float, float],
          around: list, place: dict[str, Any] | None = None, sea: float | None = None,
          default_ground: float = 0.0) -> dict[str, Any]:
    """The survey in the scene. ``keep_out`` is the works (x0, y0, x1, y1), cut out of it;
    ``around`` the surroundings already built, carved away under it and used to match heights;
    ``sea`` the sea level at a marine site (the water in the survey is cut out). Returns what was
    done, with the hull it covers."""
    import bmesh
    from mathutils import Matrix

    objs = _import(survey.mesh)
    if not objs:
        raise ValueError(f"{survey.mesh.name} has no mesh in it.")
    place = place or {}
    kx0, ky0, kx1, ky1 = keep_out
    kcx, kcy = (kx0 + kx1) / 2, (ky0 + ky1) / 2
    m = Matrix(frame(survey, lat, lon, place))
    for o in objs:
        o.data.transform(m)
    verts = [v.co for o in objs for v in o.data.vertices]

    # heights: the survey's ground just outside the works onto the scene's ground there
    reach = max(kx1 - kx0, ky1 - ky0) / 2 + 60.0
    diffs = []
    for k in range(48):
        ang = 2 * math.pi * k / 48
        for out in (15.0, 30.0, 50.0):
            x = kcx + math.cos(ang) * ((kx1 - kx0) / 2 + out)
            y = kcy + math.sin(ang) * ((ky1 - ky0) / 2 + out)
            mine = _ray_down(objs, x, y)
            theirs = _ray_down(around, x, y) if around else None
            if mine is None:
                continue
            if theirs is None:
                theirs = default_ground
            if sea is not None and theirs < sea + 0.3:
                continue                                       # over water: the survey's sea is no level
            diffs.append(theirs - mine)
    if len(diffs) >= 8:
        diffs.sort()
        dz = diffs[len(diffs) // 2]
        how = f"heights matched on {len(diffs)} points round the works"
    else:
        zs = sorted(v.z for v in verts)
        dz = default_ground - zs[len(zs) // 10]
        how = "heights set from the survey's own low ground (too little ground round the works to match)"
    dz += float(place.get("up") or 0.0)
    lift = Matrix.Translation((0.0, 0.0, dz))
    for o in objs:
        o.data.transform(lift)

    # the works cut out, and at sea the water (photogrammetry makes a mess of moving water)
    cut = 0
    for o in objs:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        dead = []
        for f in bm.faces:
            c = f.calc_center_median()
            if kx0 <= c.x <= kx1 and ky0 <= c.y <= ky1:
                dead.append(f)
            elif sea is not None and max(v.co.z for v in f.verts) < sea + 0.25:
                dead.append(f)
        if dead:
            bmesh.ops.delete(bm, geom=dead, context="FACES")
            bm.to_mesh(o.data)
            cut += len(dead)
        bm.free()
    objs = [o for o in objs if o.data.polygons]
    for o in objs:
        o.name = "survey_" + o.name
    _matte(objs)
    pts = [(round(v.co.x, 1), round(v.co.y, 1)) for o in objs for v in list(o.data.vertices)[::7]]
    hull = shrink(convex_hull(pts), 4.0)
    carved = carve(around, objs) if around else 0
    faces = sum(len(o.data.polygons) for o in objs)
    return {"faces": faces, "cut": cut, "carved": carved, "dz": round(dz, 2), "how": how,
            "georeferenced": survey.georef is not None, "notes": survey.notes,
            "hull_area": round(_area(hull)), "reach": reach}


def _area(poly) -> float:
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly)))) / 2 if len(poly) > 2 else 0.0


# ---- once per survey: one clean file, level and centred, and a plan of it ----------------------

def _join(objs: list):
    import bpy
    if len(objs) == 1:
        return objs[0]
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def level(obj, cameras: list | None = None) -> list[list[float]]:
    """Turn a model made with no GPS (whose 'up' is wherever photogrammetry guessed it) so its
    ground is level: the direction the points spread least in is up, the side most of the surface
    faces is the top, or better, the side the drone flew on (``cameras``, its positions in the
    model's frame). Returns the 3x3 turn applied."""
    import numpy as np
    from mathutils import Matrix, Vector
    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    pick = co[:: max(1, len(co) // 200000)]
    _, vecs = np.linalg.eigh(np.cov((pick - pick.mean(axis=0)).T))
    up = Vector(vecs[:, 0].tolist()).normalized()
    if cameras:
        above = float(np.dot(np.mean(np.asarray(cameras), axis=0) - pick.mean(axis=0), vecs[:, 0]))
        facing = above
    else:
        facing = sum((p.normal.dot(up) * p.area for p in me.polygons), 0.0)
    if facing < 0:
        up = -up
    turn = up.rotation_difference(Vector((0.0, 0.0, 1.0))).to_matrix()
    me.transform(turn.to_4x4())
    me.update()
    return [list(r) for r in turn]


def outward(obj) -> None:
    """Faces turned to face up and out, and the file's own vertex normals dropped: photogrammetry
    meshes often come wound the wrong way, and a ground whose normals point down is lit from under
    the earth, i.e. black."""
    import bmesh
    import bpy
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    up = sum((f.normal.z * f.calc_area() for f in bm.faces), 0.0)
    if up < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    if me.has_custom_normals:
        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        except RuntimeError:
            pass
    me.update()


def _flat_colour(objs: list) -> None:
    """Each material shows its texture as it is, unlit (for the plan picture)."""
    for o in objs:
        for slot in o.material_slots:
            m = slot.material
            if m is None or not m.use_nodes:
                continue
            nt = m.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
            if bsdf is None or out is None:
                continue
            emit = nt.nodes.new("ShaderNodeEmission")
            src = bsdf.inputs["Base Color"]
            if src.links:
                nt.links.new(src.links[0].from_socket, emit.inputs["Color"])
            else:
                emit.inputs["Color"].default_value = src.default_value
            nt.links.new(emit.outputs[0], out.inputs["Surface"])


def plan_image(obj, out: Path, extent: tuple[float, float], px: int = 1400) -> None:
    """The survey seen from straight above, on a clear background, exactly its extent across:
    the picture the project page lays on the map to place it."""
    import bpy
    sc = bpy.context.scene
    w, h = extent
    cam = bpy.data.objects.new("plan_cam", bpy.data.cameras.new("plan_cam"))
    sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = max(w, h)
    zs = [v.co.z for v in obj.data.vertices]
    cam.location = (0.0, 0.0, max(zs) + 50.0)
    cam.data.clip_end = max(zs) - min(zs) + 200.0
    sc.camera = cam
    world = bpy.data.worlds.new("plan_world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    sc.world = world
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 8
    sc.cycles.use_denoising = False
    sc.render.film_transparent = True
    sc.render.resolution_x = px if w >= h else max(8, int(px * w / h))
    sc.render.resolution_y = px if h >= w else max(8, int(px * h / w))
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.filepath = str(out)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)


def normalise(src: Path, out: Path, max_faces: int = 450_000, log=print) -> dict[str, Any]:
    """A survey as it came (an ODM result, a zip of OBJ, a .glb) to ``out``: ``survey.glb``
    (compressed, JPEG textures), level and centred on its middle, ``survey.json`` (where it
    stands, if it says, and its size) and ``plan.png`` (seen from above)."""
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    out.mkdir(parents=True, exist_ok=True)
    raw = unpack(src, out / "unpacked")
    objs = _import(raw.mesh)
    if not objs:
        raise ValueError(f"{raw.mesh.name} has no mesh in it.")
    obj = _join(objs)
    obj.name = "survey"
    turn = None
    centre = None
    if raw.georef is None:
        own = next((out / "unpacked").rglob("survey.json"), None)
        cams = json.loads(own.read_text()).get("cameras") if own else None
        turn = level(obj, cams)
    vs = obj.data.vertices
    xs = [v.co.x for v in vs]
    ys = [v.co.y for v in vs]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    from mathutils import Matrix
    obj.data.transform(Matrix.Translation((-cx, -cy, 0.0)))
    extent = (max(xs) - min(xs), max(ys) - min(ys))
    faces = len(obj.data.polygons)
    if faces > max_faces:
        mod = obj.modifiers.new("fewer", "DECIMATE")
        mod.ratio = max_faces / faces
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)
        log(f"  {faces:,} faces brought down to {len(obj.data.polygons):,}")
    outward(obj)
    _matte([obj])
    geo = None
    if raw.georef is not None:
        g = raw.georef
        geo = {"zone": g.zone, "north": g.north, "offset": [g.offset[0] + cx, g.offset[1] + cy]}
        centre = list(utm_to_lonlat(geo["offset"][0], geo["offset"][1], g.zone, g.north))
    bpy.ops.export_scene.gltf(filepath=str(out / "survey.glb"), export_format="GLB",
                              export_image_format="JPEG", export_apply=True, export_yup=True)
    _flat_colour([obj])
    plan_image(obj, out / "plan.png", extent)
    info = {"version": 1, "mesh": "survey.glb", "plan": "plan.png", "georef": geo,
            "metric": raw.georef is not None, "extent": [round(extent[0], 4), round(extent[1], 4)],
            "faces": len(obj.data.polygons), "levelled": turn is not None, "source": raw.name,
            "centre": centre,
            "notes": raw.notes}
    (out / "survey.json").write_text(json.dumps(info, indent=1))
    import shutil
    shutil.rmtree(out / "unpacked", ignore_errors=True)
    return info


def load(folder: Path) -> Survey:
    """A survey made by ``normalise``."""
    info = json.loads((folder / "survey.json").read_text())
    g = info.get("georef")
    geo = Georef(int(g["zone"]), bool(g.get("north", True)), (float(g["offset"][0]), float(g["offset"][1]))) \
        if g else None
    return Survey(mesh=folder / info.get("mesh", "survey.glb"), georef=geo, name=info.get("source", ""),
                  notes=list(info.get("notes", [])), metric=bool(info.get("metric", geo is not None)),
                  extent=tuple(info.get("extent", (0.0, 0.0))))


def add_to_scene(manifest: dict[str, Any], keep_out: tuple[float, float, float, float], around: list,
                 sea: float | None = None, log=print) -> dict[str, Any] | None:
    """The project's drone survey, if it has one (``manifest["survey"]``: its folder and the hand
    placement), stood in the scene in place of the surroundings under it. A survey that cannot be
    used is noted in the manifest's ``missing`` and the film goes on without it."""
    spec = manifest.get("survey")
    if not spec:
        return None
    try:
        sv = load(Path(spec["folder"]))
        got = build(sv, float(manifest["lat"]), float(manifest["lon"]), keep_out, around,
                    spec.get("place") or {}, sea=sea)
    except Exception as exc:                                # noqa: BLE001 - the film goes on without it
        log(f"  drone survey not used: {type(exc).__name__}: {exc}")
        manifest.setdefault("missing", {})["drone survey"] = f"{type(exc).__name__}: {exc}"[:200]
        return None
    log(f"  drone survey: {got}")
    return got


def pack(folder: Path, out: Path) -> Path:
    """A survey made by ``normalise`` as one zip, as the site keeps it."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for name in ("survey.glb", "survey.json", "plan.png"):
            if (folder / name).exists():
                z.write(folder / name, name)
    return out


def make_ready(sources: list[Path], work: Path, log=print, odm=None, on_stage=None) -> Path:
    """What a person sent (a video and its track, photos, or a finished model) to the zip the site
    keeps. ``odm(sources, work) -> zip`` runs OpenDroneMap when it is needed (locally, or on Modal)."""
    from . import drone
    stage = on_stage or (lambda words: None)
    if drone.needs_odm(sources):
        if odm is None:
            if not drone.ODM.exists():
                raise RuntimeError("Turning a video into a 3D model needs OpenDroneMap, which the cloud "
                                   "worker has; this machine does not.")
            def odm(srcs, w):                               # noqa: E306 - the local one
                return drone.process(srcs, w, log=log, on_stage=stage)[0]
        made = odm(sources, work)
    else:
        made = next(p for p in sources if p.suffix.lower() in (".zip", ".glb", ".gltf", ".obj"))
    stage("Levelling the model and drawing its plan")
    normalise(Path(made), work / "ready", log=log)
    return pack(work / "ready", work / "survey-ready.zip")
