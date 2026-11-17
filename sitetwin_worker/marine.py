"""The scene for marine works: a jetty, pier or quay rebuilt over the water, stage by stage.

A building site sits on land behind hoarding, with a tower crane and a gate. A jetty does
not: the water is all round it, the old deck is there on its piles, and the work is done from
the land behind it (the fill or the rock mound) with crawler cranes, piling rigs, excavators
and pumps that come along the jetty from the shore. So:

* Google's real surroundings stay as they are (the sea, the rock mound, the rest of the jetty,
  the shore); only what stands above the water where the model is (the old deck that the
  model rebuilds) is taken out of them. Nothing is laid over the sea: no site pad, no hoarding
  in the water, no roads, cars or buildings rebuilt near the site (the streets are the shore's).
* The model's own ground (the fill and rock slope after the works, from its sections) is the
  ground under and behind the deck; what is under water is seen through it, as it is.
* The old deck is there at the start and is broken out piece by piece (stage "remove"); the
  piles that stay get jackets (stage "repair"); new piles are driven, caps and stitches are cast
  one after another, precast units are set by a crawler crane that travels along the land side
  to each lift, and the fenders and bollards are fitted last.
* Plant stands on the land side only, at its level; people stand only on what is built (or on
  the land), never on the water.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from . import equipment as E
from . import logistics as L
from . import materials as M
from . import scene as S
from .model import Element, footprint
from .sequence import Timeline, plant_windows


# ---- plant that a building site does not have --------------------------------------------

def crawler_crane(boom: float = 42.0):
    """A lattice-boom crawler crane (a 100 t class): returns (base, upper, boom, hook block).
    The upper works slew about z on the base; the boom luffs about y at its foot; the hook
    block hangs from the boom tip (move it in world terms)."""
    base = E._build("crawler_base", [
        ("box", "dark_metal", (0, -2.6, 0.6), (8.0, 1.1, 1.2), 0),
        ("box", "dark_metal", (0, 2.6, 0.6), (8.0, 1.1, 1.2), 0),
        ("box", "dark_metal", (0, 0, 1.1), (3.2, 4.2, 0.6), 0),
    ])
    upper = E._build("crawler_upper", [
        ("box", "crane_yellow", (-1.2, 0, 2.1), (6.4, 3.4, 1.6), 0),          # house
        ("box", "concrete", (-4.6, 0, 2.6), (1.6, 3.4, 2.2), 0),              # counterweight
        ("box", "crane_yellow", (1.6, 1.4, 3.1), (1.8, 1.1, 1.9), 0),          # cab
        ("box", "cab_glass", (2.52, 1.4, 3.2), (0.04, 0.9, 1.2), 0),
        ("box", "crane_yellow", (-2.4, 0, 4.6), (0.4, 0.4, 4.0), 0, 0.25),    # A-frame (gantry)
    ], parent=base)
    upper.location = (0, 0, 1.4)
    parts: list = []
    w = 1.6
    _boom_lattice(parts, boom, w)
    parts.append(("box", "crane_yellow", (boom + 0.4, 0, 0), (1.0, 1.2, 0.8), 0))     # boom head
    boom_obj = E._build("crawler_boom", parts, parent=upper)
    boom_obj.location = (1.8, 0, 2.4)
    block = E._build("crawler_block", [("box", "drum_orange", (0, 0, -0.4), (0.6, 0.4, 0.8), 0),
                                       ("cyl", "dark_metal", (0, 0, -1.0), (0.18, 0.18, 0.4), 0)])
    line = E._build("crawler_line", [("cyl8", "dark_metal", (0, 0, -0.5), (0.04, 0.04, 1.0), 0)])
    return base, upper, boom_obj, block, line


def _boom_lattice(parts: list, length: float, w: float, member: float = 0.14, panel: float = 2.0) -> None:
    """A square lattice boom along +x from its foot: four chords, cross members and diagonals,
    tapering a little to the head."""
    n = max(2, int(round(length / panel)))
    for dy in (-w / 2, w / 2):
        for dz in (-w / 2, w / 2):
            parts.append(("box", "crane_yellow", (length / 2, dy * 0.85, dz * 0.85), (length, member, member), 0))
    for i in range(n + 1):
        x = i * length / n
        parts.append(("box", "crane_yellow", (x, 0, w / 2 * 0.85), (member * 0.7, w * 0.85, member * 0.7), 0))
        parts.append(("box", "crane_yellow", (x, 0, -w / 2 * 0.85), (member * 0.7, w * 0.85, member * 0.7), 0))
        parts.append(("box", "crane_yellow", (x, w / 2 * 0.85, 0), (member * 0.7, member * 0.7, w * 0.85), 0))
        parts.append(("box", "crane_yellow", (x, -w / 2 * 0.85, 0), (member * 0.7, member * 0.7, w * 0.85), 0))
        if i < n:
            seg = length / n
            d = math.hypot(seg, w * 0.85)
            a = math.atan2(w * 0.85, seg) * (1 if i % 2 else -1)
            for dz in (-w / 2 * 0.85, w / 2 * 0.85):
                parts.append(("box", "crane_yellow", (x + seg / 2, 0, dz), (d, member * 0.5, member * 0.5), a))
            for dy in (-w / 2 * 0.85, w / 2 * 0.85):
                parts.append(("box", "crane_yellow", (x + seg / 2, dy, 0), (d, member * 0.5, member * 0.5), 0, -a))


def workboat():
    """A small steel work boat with a wheelhouse, at the waterline."""
    return E._build("workboat", [
        ("box", "steel_red", (0, 0, 0.1), (11.0, 3.8, 1.2), 0),
        ("box", "truck_white", (-1.5, 0, 1.2), (3.2, 3.0, 1.2), 0),
        ("box", "truck_white", (-1.5, 0, 2.4), (2.6, 2.6, 1.2), 0),
        ("box", "cab_glass", (0.1, 0, 2.5), (0.04, 2.2, 0.7), 0),
        ("box", "dark_metal", (3.4, 0, 0.75), (3.0, 3.2, 0.1), 0),
    ])


# ---- the water, and the model's ground -------------------------------------------------

def build_water(around: tuple[float, float, float, float], z: float, clear: bool = False) -> None:
    """The sea where Google's surface was cut away (it shows only through the cut). ``clear``:
    the sea itself, a little clear, over a sea bed, so piles are seen going down into it."""
    x0, y0, x1, y1 = around
    S._mesh("sea_under", [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], [(0, 1, 2, 3)],
            M.get("sea_clear" if clear else "water"))
    if clear:
        d = z - 7.0
        S._mesh("sea_bed", [(x0, y0, d), (x1, y0, d), (x1, y1, d), (x0, y1, d)], [(0, 1, 2, 3)], M.get("sea_bed"))


def build_ground(ground: list[Element], keep: tuple[float, float, float, float]) -> list:
    """The model's ground after the works, cut to ``keep`` (beyond it Google's is the ground):
    rock armour on the slopes, fill on the flat top."""
    import bmesh
    import bpy
    x0, y0, x1, y1 = keep
    out = []
    for e in ground:
        bm = bmesh.new()
        vs = [bm.verts.new(tuple(map(float, v))) for v in e.verts]
        for f in e.faces:
            try:
                bm.faces.new([vs[i] for i in f])
            except ValueError:
                pass
        for co, no in (((x0, 0, 0), (-1, 0, 0)), ((x1, 0, 0), (1, 0, 0)), ((0, y0, 0), (0, -1, 0)), ((0, y1, 0), (0, 1, 0))):
            geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
            res = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-4, plane_co=co, plane_no=no, clear_outer=True)
            del res
        if not bm.faces:
            bm.free()
            continue
        me = bpy.data.meshes.new("ground_" + e.guid)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(M.get("rock"))
        me.materials.append(M.get("gravel"))
        for p in me.polygons:
            p.material_index = 1 if (p.normal.z > 0.92 and p.center.z > 0.8) else 0
        o = bpy.data.objects.new("ground_model", me)
        bpy.context.scene.collection.objects.link(o)
        out.append(o)
    if out:
        # the breakwater's slopes as armour stone, boulder by boulder (slot 0 is the rock)
        from .surroundings import boulders
        boulders(out, 0, 0.35, seed=11, size=(1.0, 2.2), name="breakwater_armour")
    return out


def ground_level(ground: list[Element], x: float, y: float, default: float) -> float:
    """The top of the model's ground at a point (the highest of its faces over it), or ``default``."""
    best = None
    for e in ground:
        if not (e.lo[0] <= x <= e.hi[0] and e.lo[1] <= y <= e.hi[1]):
            continue
        tri = e.verts[e.faces]                                  # (m, 3, 3)
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        ok = np.abs(d) > 1e-9
        d = np.where(ok, d, 1.0)
        w1 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / d
        w2 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / d
        w3 = 1 - w1 - w2
        hit = ok & (w1 >= -1e-6) & (w2 >= -1e-6) & (w3 >= -1e-6)
        if hit.any():
            z = float(np.max((w1 * a[:, 2] + w2 * b[:, 2] + w3 * c[:, 2])[hit]))
            best = z if best is None else max(best, z)
    return default if best is None else best


# ---- the model, stage by stage ------------------------------------------------------------

def _remove(obj, a: int, b: int, tl: Timeline) -> None:
    """Broken out: there from the start, it sinks to nothing between a and b, then is gone."""
    obj.scale = (1, 1, 1)
    obj.keyframe_insert("scale", frame=a)
    obj.scale = (1, 1, 0.02)
    obj.keyframe_insert("scale", frame=b)
    S._visible(obj, None, b + 1, tl.frames)


def _drive(obj, e: Element, a: int, b: int, tl: Timeline) -> None:
    """A pile driven in: it stands its full length above its place, and goes down into it."""
    length = float(e.hi[2] - e.lo[2])
    rise = min(length, 18.0)
    z = obj.location[2]
    obj.location = (obj.location[0], obj.location[1], z + rise)
    obj.keyframe_insert("location", frame=a)
    obj.location = (obj.location[0], obj.location[1], z)
    obj.keyframe_insert("location", frame=b)
    S._visible(obj, a, None, tl.frames)


def _lower(obj, a: int, b: int, tl: Timeline, drop: float = 4.0) -> None:
    """Set by the crane: it comes down the last few metres into its place."""
    x, y, z = obj.location
    obj.location = (x, y, z + drop)
    obj.keyframe_insert("location", frame=a)
    obj.location = (x, y, z)
    obj.keyframe_insert("location", frame=b)
    S._visible(obj, a, None, tl.frames)


def _jacket(e: Element, a: int, b: int, tl: Timeline, sea: float) -> None:
    """A pile repair: a concrete jacket round the pile through the splash zone, cast in a form."""
    lo, hi = e.lo, e.hi
    top = float(hi[2]) - 0.15
    bot = max(float(lo[2]), sea - 1.5)
    if top - bot < 0.5:
        return
    pad = 0.12
    jacket = S._box("jacket_" + e.guid, (lo[0] - pad, lo[1] - pad, bot), (hi[0] + pad, hi[1] + pad, top),
                    M.get("concrete_fresh"))
    form = S._box("jacketform_" + e.guid, (lo[0] - pad - 0.05, lo[1] - pad - 0.05, bot),
                  (hi[0] + pad + 0.05, hi[1] + pad + 0.05, top), M.get("formwork"))
    mid = (a + b) // 2
    S._visible(form, a, mid + 2, tl.frames)
    S._visible(jacket, mid, None, tl.frames)


def build_model(elements: list[Element], tl: Timeline, sea: float) -> dict[str, Any]:
    objs = {}
    for e in elements:
        if e.how == "ground":
            continue
        lo, hi = e.lo, e.hi
        base = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]], dtype=np.float32)
        mat = M.get("concrete_old") if e.how in ("remove", "repair") else (
            M.get("rock") if e.how == "fill" and float(hi[2]) < sea + 0.3 else
            M.get("gravel") if e.how == "fill" else M.for_element(e.kind))
        if e.how == "fit" and "IFCBUILDINGELEMENTPROXY" in e.kind:
            mat = M.get("steel")
        obj = S._mesh("el_" + e.guid, e.verts - base, e.faces, mat)
        obj.location = tuple(map(float, base))
        objs[e.guid] = obj
        if e.guid in tl.removed:
            a, b = tl.removed[e.guid]
            _remove(obj, a, b, tl)
            continue
        a, b, how = tl.appear.get(e.guid, (0, 0, "exist"))
        if how in ("exist", "repair"):
            if ("repair:" + e.guid) in tl.removed:
                ra, rb = tl.removed["repair:" + e.guid]
                _jacket(e, ra, rb, tl, sea)
            continue
        if how == "grow":
            a = S._cast(e, obj, a, b, tl)
            S._visible(obj, a, None, tl.frames)
        elif how == "drive":
            _drive(obj, e, a, b, tl)
        elif how == "lift":
            _lower(obj, a, b, tl)
        else:
            obj.scale = (0.9, 0.9, 0.9)
            obj.keyframe_insert("scale", frame=a)
            obj.scale = (1, 1, 1)
            obj.keyframe_insert("scale", frame=b)
            S._visible(obj, a, None, tl.frames)
    return objs


# ---- plant ------------------------------------------------------------------------------

def land_line(built: list[Element], ground: list[Element], sea: float) -> tuple[float, float, int]:
    """Where plant works from: the land side of the structure (where the model's ground is
    above the sea), as (x of the working line, its level, which side: -1 west / +1 east).
    The structure's long axis is taken as y."""
    fp = footprint(built)
    x0, y0, x1, y1 = fp
    ym = (y0 + y1) / 2
    best = None
    for side, x in ((-1, x0 + 6.0), (1, x1 - 6.0)):
        z = ground_level(ground, x, ym, sea - 5.0)
        if best is None or z > best[1]:
            best = (side, z, x)
    side, z, x = best
    return x, max(z, sea + 0.8), side


def barge(name: str = "barge", length: float = 40.0, width: float = 14.0):
    """A flat-top spud barge, its origin on the middle of its deck, long axis along local x:
    a red hull with a dark deck, a kerb round the edge and a spud leg at each corner."""
    hx, hy = length / 2, width / 2
    p: list = [("box", "steel_red", (0, 0, -1.3), (length, width, 2.5), 0),
               ("box", "dark_metal", (0, 0, -0.04), (length - 0.3, width - 0.3, 0.1), 0)]
    for sx in (-1, 1):
        p.append(("box", "steel_red", (0, sx * (hy - 0.15), 0.2), (length, 0.3, 0.45), 0))
        p.append(("box", "steel_red", (sx * (hx - 0.15), 0, 0.2), (0.3, width, 0.45), 0))
        for sy in (-1, 1):
            p.append(("cyl", "dark_metal", (sx * (hx - 2.0), sy * (hy - 1.4), 3.0), (1.0, 1.0, 16.0), 0))
    return E._build(name, p)


def build_plant(elements: list[Element], tl: Timeline, line: tuple[float, float, int], objs: dict,
                rng: random.Random, sea: float) -> list[L.Machine]:
    """All plant works from the water, on barges along the open face of the berth, never on
    the breakwater: the crawler crane on a crane barge (it lifts, breaks out and drives the new
    piles with a hammer on its hook), the excavator on a work barge, the pump and a mixer on a
    concrete barge, the precast on a flatbed on the crane barge's deck, and a work boat at the
    pile repairs. Returns the machines' ways (for keeping people clear)."""
    by = {e.guid: e for e in elements}
    lx, lz, side = line
    out = -side                                            # the open water side: +1 east / -1 west
    machines: list[L.Machine] = []
    built = [e for e in elements if e.how != "ground"]
    fp = footprint(built)
    y_lo, y_hi = float(min(e.lo[1] for e in built)), float(max(e.hi[1] for e in built))
    edge = fp[2] if out > 0 else fp[0]                     # the berth face
    deck = sea + 1.5                                       # a barge's deck level
    bx = edge + out * 9.0                                  # a barge alongside the face

    def clamp_y(y: float) -> float:
        return min(y_hi - 4.0, max(y_lo + 22.0, y))        # clear of the shore at the root

    def front(step, f: int) -> tuple[float, float, float] | None:
        """The element being worked on in a step at frame f: (x, y, top)."""
        cur = None
        for g in step.guids:
            if g in tl.removed:
                a, b = tl.removed[g]
            elif g in tl.appear:
                a, b, _ = tl.appear[g]
            elif ("repair:" + g) in tl.removed:
                a, b = tl.removed["repair:" + g]
            else:
                continue
            if a <= f:
                cur = by[g]
        if cur is None:
            return None
        c = cur.centre
        return float(c[0]), float(c[1]), float(cur.hi[2])

    def follow(st, offset: float = 0.0, every: int = 6, hop: float = 10.0) -> list[tuple[int, float]]:
        """A barge's y through a step: it warps along the face in hops after the work."""
        keys, last = [], None
        for f in range(st.start, st.end + 1, every):
            tgt = front(st, f)
            y = clamp_y((tgt[1] if tgt else y_lo + 30.0) + offset)
            if last is not None and abs(y - last) < hop:
                y = last
            last = y
            keys.append((f, y))
        return keys

    def on_barge(obj, st, keys, x: float, dy: float = 0.0, hdg: float = math.pi / 2, kind: str = "", r: float = 4.0):
        mk = [(f, x, y + dy, hdg, deck) for f, y in keys]
        S._visible(obj, st.start, st.end, tl.frames)
        S._apply_keys(obj, mk)
        if kind:
            machines.append(L.Machine(kind, r, mk))
        return mk

    # --- the crane barge: the crawler crane slews and luffs to each lift; the barge warps along
    cr_steps = [st for st in tl.steps if "crawler_crane" in st.plant]
    crane_y: dict[int, float] = {}
    if cr_steps:
        s, e = cr_steps[0].start, cr_steps[-1].end
        cx = bx - out * 3.0                                # its crane on the berth side of the deck
        real = E._asset("crawler_crane", 40.0)
        hull = barge("crane_barge", 44.0, 16.0)
        hammer = E._build("pile_hammer", [("cyl", "dark_metal", (0, 0, 0), (1.1, 1.1, 5.0), 0),
                                           ("box", "drum_orange", (0, 0, 2.2), (1.4, 1.4, 0.6), 0)])
        hammer.location = (0, 0, -500)
        hammer.keyframe_insert("location", frame=0)
        piling = [st for st in cr_steps if st.how == "drive"]

        def target(f: int, last):
            st = next((t for t in cr_steps if t.start <= f < t.end), None)
            tgt = front(st, f) if st is not None else None
            return tgt if tgt is not None else (last or (edge - out * 6.0, clamp_y(y_lo + 30), deck + 3.0))

        if real is not None:
            tip = _tip(real)
            reach = max(8.0, tip[0])
            print(f"  crane barge: reach {reach:.1f} m from x {cx:.1f}")
            line_obj = E._build("crawler_line", [("cyl8", "dark_metal", (0, 0, -0.5), (0.04, 0.04, 1.0), 0)])
            block = E._build("crawler_block", [("box", "drum_orange", (0, 0, -0.4), (0.6, 0.4, 0.8), 0),
                                               ("cyl", "dark_metal", (0, 0, -1.0), (0.18, 0.18, 0.4), 0)])
            for o in (real, line_obj, block, hull):
                S._visible(o, s, e, tl.frames)
            keys, hk, prev_h, last, by_ = [], [], None, None, None
            for f in range(s, e + 1, 3):
                tgt = target(f, last)
                last = tgt
                dx = tgt[0] - cx
                dy = math.sqrt(max(0.0, reach * reach - dx * dx))
                want = clamp_y(tgt[1] - dy)
                if by_ is None or abs(want - by_) > 8.0:     # the barge warps along in hops
                    by_ = want
                crane_y[f] = by_
                h = math.atan2(tgt[1] - by_, dx)
                h = h if prev_h is None else L.unwrap(prev_h, h)
                prev_h = h
                keys.append((f, cx, by_, h, deck))
                hk.append((f, cx + out * 3.0, by_, math.pi / 2, deck))
                tx_, ty_ = cx + math.cos(h) * tip[0], by_ + math.sin(h) * tip[0]
                tip_z = deck + tip[1]
                hook_z = max(tgt[2] + 3.5, deck + 2.0)
                drive_now = any(p.start <= f < p.end for p in piling)
                if drive_now:
                    hook_z = max(tgt[2] + 6.5, deck + 2.0)
                block.location = (tx_, ty_, hook_z + 1.0)
                block.keyframe_insert("location", frame=f)
                hammer.location = (tx_, ty_, hook_z - 2.0 if drive_now else -500)
                hammer.keyframe_insert("location", frame=f)
                line_obj.location = (tx_, ty_, tip_z)
                line_obj.scale = (1, 1, max(1.0, tip_z - hook_z - 1.0))
                line_obj.keyframe_insert("location", frame=f)
                line_obj.keyframe_insert("scale", frame=f)
            S._apply_keys(real, keys)
            S._apply_keys(hull, hk)
            for o in (block, line_obj, hammer):
                S._linear(o)
            machines.append(L.Machine("crawler_crane", 5.0, keys))
        else:
            boom_len = 42.0
            base, upper, boom, block, line_obj = crawler_crane(boom_len)
            for o in (base, block, line_obj, hull):
                S._visible(o, s, e, tl.frames)
            keys, hk, prev_h, last, by_ = [], [], None, None, None
            for f in range(s, e + 1, 3):
                tgt = target(f, last)
                last = tgt
                want = clamp_y(tgt[1] - 6.0)
                if by_ is None or abs(want - by_) > 8.0:
                    by_ = want
                crane_y[f] = by_
                keys.append((f, cx, by_, 0.0, deck))
                hk.append((f, cx + out * 3.0, by_, math.pi / 2, deck))
                dx, dy = tgt[0] - cx, tgt[1] - by_
                h = math.atan2(dy, dx)
                h = h if prev_h is None else L.unwrap(prev_h, h)
                prev_h = h
                r = min(boom_len * 0.92, max(14.0, math.hypot(dx, dy)))
                foot = 1.4 + 2.4
                luff = math.acos(min(1.0, (r - 1.8) / boom_len))
                upper.rotation_euler = (0, 0, h)
                upper.keyframe_insert("rotation_euler", frame=f)
                boom.rotation_euler = (0, -luff, 0)
                boom.keyframe_insert("rotation_euler", frame=f)
                tip_z = deck + foot + math.sin(luff) * boom_len
                hx, hy = cx + math.cos(h) * r, by_ + math.sin(h) * r
                hook_z = max(tgt[2] + 3.5, deck + 2.0)
                drive_now = any(p.start <= f < p.end for p in piling)
                if drive_now:
                    hook_z = max(tgt[2] + 6.5, deck + 2.0)
                block.location = (hx, hy, hook_z + 1.0)
                block.keyframe_insert("location", frame=f)
                hammer.location = (hx, hy, hook_z - 2.0 if drive_now else -500)
                hammer.keyframe_insert("location", frame=f)
                line_obj.location = (hx, hy, tip_z)
                line_obj.scale = (1, 1, max(1.0, tip_z - hook_z - 1.0))
                line_obj.keyframe_insert("location", frame=f)
                line_obj.keyframe_insert("scale", frame=f)
            S._apply_keys(base, keys)
            S._apply_keys(hull, hk)
            for o in (upper, boom, block, line_obj, hammer):
                S._linear(o)
            machines.append(L.Machine("crawler_crane", 5.0, keys))
        S._constant_jumps(hammer)

    def near_crane(st) -> list[tuple[int, float]]:
        """The crane barge's y through a step (sampled from its keys), for what rides on it or
        lies alongside it."""
        ks = sorted(crane_y)
        out_ = []
        for f in range(st.start, st.end + 1, 6):
            k = min(ks, key=lambda q: abs(q - f)) if ks else None
            out_.append((f, crane_y[k] if k is not None else clamp_y(y_lo + 30.0)))
        return out_

    # --- the work barge with the excavator: breaking out beside the crane barge, placing fill
    for st in tl.steps:
        if "excavator" not in st.plant:
            continue
        with_crane = "crawler_crane" in st.plant and crane_y
        ys = [(f, clamp_y(y + 34.0) if clamp_y(y + 34.0) - y > 30.0 else clamp_y(y - 34.0))
              for f, y in near_crane(st)] if with_crane else follow(st)
        hull = barge("work_barge", 26.0, 12.0)
        on_barge(hull, st, ys, bx)
        on_barge(E.excavator(), st, ys, bx - out * 2.0, 0.0, math.pi if out < 0 else 0.0, "excavator", 3.5)
        if not with_crane and "flatbed_truck" in st.plant:
            on_barge(E.flatbed_truck(), st, ys, bx + out * 2.5, 2.0, math.pi / 2, "flatbed_truck", 7.0)

    # --- the concrete barge: the pump and a mixer on it, alongside each pour
    for st in tl.steps:
        if "concrete_pump" not in st.plant:
            continue
        ys = follow(st, -4.0, every=8, hop=12.0)
        hull = barge("concrete_barge", 30.0, 12.0)
        on_barge(hull, st, ys, bx)
        on_barge(E.concrete_pump(), st, ys, bx - out * 2.5, 2.0, math.pi / 2, "concrete_pump", 6.0)
        on_barge(E.mixer_truck(), st, ys, bx + out * 2.5, -6.0, math.pi / 2, "mixer_truck", 5.0)

    # --- the flatbed with the precast (and the broken-out pieces) on the crane barge's deck
    for st in tl.steps:
        if "flatbed_truck" not in st.plant or "crawler_crane" not in st.plant or not crane_y:
            continue
        on_barge(E.flatbed_truck(), st, near_crane(st), bx + out * 3.5, -12.0, math.pi / 2, "flatbed_truck", 7.0)

    # --- work boat at the pile repairs, on the open side of the deck
    for st in tl.steps:
        if "workboat" not in st.plant:
            continue
        boat = workboat()
        S._visible(boat, st.start, st.end, tl.frames)
        xs = [float(by[g].centre[0]) for g in st.guids]
        wx = (max(xs) + 6.0) if out > 0 else (min(xs) - 6.0)
        keys = []
        for f in range(st.start, st.end + 1, 8):
            tgt = front(st, f)
            y = clamp_y(tgt[1] if tgt else y_lo + 10)
            keys.append((f, wx, y, math.pi / 2, sea - 0.4))
        S._apply_keys(boat, keys)
    return machines


def _tip(root) -> tuple[float, float]:
    """A real crane model's boom tip, as (forward distance, height) from its origin: the
    furthest-forward point of its top fifth."""
    pts = []
    for c in root.children:
        if c.type == "MESH":
            for v in c.data.vertices:
                pts.append((v.co.x, v.co.z))
    if not pts:
        return 30.0, 35.0
    top = max(z for _, z in pts)
    high = [(x, z) for x, z in pts if z > top * 0.8]
    x, z = max(high)
    return float(x), float(z)


# ---- people -----------------------------------------------------------------------------

def build_crews(elements: list[Element], tl: Timeline, line: tuple[float, float, int],
                machines: list[L.Machine], rng: random.Random) -> list:
    """Each stage's crew where its work is: on what is already built round the element being
    worked on (never on the water, never on the breakwater), clear of the machines."""
    by = {e.guid: e for e in elements}
    lx, lz, side = line
    pool = max((st.crew for st in tl.steps), default=0)
    people = [E.worker("hivis" if k % 3 else "hivis_yellow", seed=k) for k in range(pool)]
    for w in people:
        w.location = (0, 0, -500)
        w.keyframe_insert("location", frame=0)

    def standing(g: str, f: int) -> bool:
        if g in tl.removed:
            return f < tl.removed[g][0]
        a, b, how = tl.appear.get(g, (0, 0, "exist"))
        return how in ("exist", "repair") or b <= f

    # a surface a person can be on: a slab, shell or beam of some size, not a lone pile cap out
    # in the water with no way to it
    flat = [e for e in elements if e.how not in ("ground", "fill") and (e.hi[0] - e.lo[0]) > 1.0
            and (e.hi[1] - e.lo[1]) > 1.0 and (e.hi[2] - e.lo[2]) < 2.6 and e.hi[2] > 1.0
            and (e.hi[0] - e.lo[0]) * (e.hi[1] - e.lo[1]) >= 6.0]

    for st in tl.steps:
        chunk = max(12, (st.end - st.start) // 4)
        for f0 in range(st.start, st.end, chunk):
            f1 = min(st.end, f0 + chunk)
            mid = (f0 + f1) // 2
            work = None
            for g in st.guids:
                a = tl.removed.get(g, tl.removed.get("repair:" + g, tl.appear.get(g, (0, 0))[:2]))[0]
                if a <= mid:
                    work = by[g]
            cy = float(work.centre[1]) if work is not None else float(np.mean([by[g].centre[1] for g in st.guids]))
            # surfaces to stand on near the work: built tops within 10 m of it, then the land
            ok = [e for e in flat if standing(e.guid, f0) and standing(e.guid, f1)
                  and not (work is not None and e.guid == work.guid)]
            tops = [e for e in ok if abs(float(e.centre[1]) - cy) < 10.0]
            if not tops:                                    # nothing near yet: the nearest built
                tops = sorted(ok, key=lambda e: abs(float(e.centre[1]) - cy))[:6]
            spots = []
            for k in range(len(people)):
                if k >= st.crew:
                    break
                for _ in range(20):
                    # never on the breakwater: only on what is built (the plant is on barges)
                    if not tops:
                        break
                    e = rng.choice(tops)
                    x = rng.uniform(float(e.lo[0]) + 0.4, float(e.hi[0]) - 0.4)
                    y = rng.uniform(float(e.lo[1]) + 0.4, float(e.hi[1]) - 0.4)
                    p = (x, y, float(e.hi[2]))
                    if L.clear_spot(p, f0, f1, machines, gap=2.0) and all(math.dist(p[:2], q[:2]) > 1.0 for q in spots):
                        spots.append(p)
                        break
            for k, w in enumerate(people):
                if k < len(spots):
                    w.location = spots[k]
                    w.rotation_euler = (0, 0, rng.uniform(-math.pi, math.pi))
                    w.keyframe_insert("location", frame=f0)
                    w.keyframe_insert("rotation_euler", frame=f0)
                    w.keyframe_insert("location", frame=max(f0 + 1, f1 - 1))
                else:
                    w.location = (0, 0, -500)
                    w.keyframe_insert("location", frame=f0)
    for w in people:
        w.location = (0, 0, -500)
        w.keyframe_insert("location", frame=tl.build_end)
        S._constant_jumps(w)
    return people


# ---- camera ---------------------------------------------------------------------------------

def build_camera(fp, top: float, tl: Timeline, side: int, sea: float):
    """A camera held still, as a time-lapse camera on a mast would be: out on the open water off
    the berth face, toward the head, looking back across all of it to its root and the shore
    behind, so the whole structure is in the picture and the land fills what is behind it."""
    import bpy
    x0, y0, x1, y1 = fp
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    length = max(x1 - x0, y1 - y0)
    target = bpy.data.objects.new("cam_target", None)
    bpy.context.scene.collection.objects.link(target)
    target.location = (cx, cy - length * 0.25, sea + 1.0)
    cam_data = bpy.data.cameras.new("camera")
    cam_data.lens = 30
    cam_data.clip_end = 20000
    cam = bpy.data.objects.new("camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    open_x = x1 if side < 0 else x0                         # the water side
    cam.location = (open_x - side * length * 0.7, y1 + length * 0.55, sea + length * 0.5)
    c = cam.constraints.new("TRACK_TO")
    c.target = target
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    bpy.context.scene.camera = cam
    return cam


# ---- the whole scene --------------------------------------------------------------------------

def build(elements: list[Element], tl: Timeline, env_folder: Path, manifest: dict[str, Any],
          opts: "S.Options", out_dir: Path, sea: float = 0.0) -> dict[str, Any]:
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    M.reset()
    E._library.clear()
    rng = random.Random(7)

    ground = [e for e in elements if e.how == "ground"]
    built = [e for e in elements if e.how != "ground"]
    fp = footprint(built)
    top = float(max(e.hi[2] for e in built))
    keep_out = (fp[0] - 1.5, fp[1] - 1.5, fp[2] + 1.5, fp[3] + 1.5)
    g = S.Ground(manifest, env_folder, keep_out)
    has_google = bool(manifest.get("google")) and not manifest.get("modelled")
    got = None
    around = None
    if manifest.get("modelled"):
        from . import surroundings
        # the model's own ground is the breakwater, crest and slopes as designed: keep all of it,
        # and leave the modelled land out of its whole area
        if ground:
            glo = np.min([e.lo for e in ground], axis=0)
            ghi = np.max([e.hi for e in ground], axis=0)
            keep_out = (min(keep_out[0], float(glo[0])), min(keep_out[1], float(glo[1])),
                        max(keep_out[2], float(ghi[0])), max(keep_out[3], float(ghi[1])))
            print("  works and breakwater area:", [round(v, 1) for v in keep_out])
        around = surroundings.build(manifest, env_folder, sea, (keep_out[0], keep_out[2], keep_out[1], keep_out[3]))
        around.pop("ground", None)
        print("  modelled surroundings:", around)
    elif has_google:
        # the shore's own streets stay as Google has them: nothing is rebuilt near a site at sea
        quiet = {k: v for k, v in manifest.items() if k != "osm"}
        got = S.build_google(g, quiet, env_folder, keep_out, None, None, flat=0.0, cut_above=0.6)
        manifest.update({k: v for k, v in quiet.items() if k.startswith("_")})
        manifest.pop("_street", None)
    if around is not None:
        pass
    elif got:
        S.build_base(g, min(-3.0, got[1] - 3.0), float(manifest["google"].get("horizon", g.radius)))
    else:
        S.build_terrain(g, manifest, env_folder)
    # under Google's sea everywhere: it shows where the old deck was cut out and through any
    # hole in Google's mesh, so the sea never shows white
    reach = (25000.0 if around is not None else
             float((manifest.get("google") or {}).get("horizon", g.radius)) if got else 60.0)
    build_water((-reach, -reach, reach, reach) if (got or around is not None) else
                (keep_out[0] - 40, keep_out[1] - 40, keep_out[2] + 40, keep_out[3] + 40), sea - 0.12,
                clear=around is not None)
    build_ground(ground, keep_out)
    objs = build_model(elements, tl, sea)
    line = land_line(built, ground, sea)
    machines = build_plant(elements, tl, line, objs, rng, sea)
    people = build_crews(elements, tl, line, machines, rng)
    clashes = S.check_clearance(people, machines, tl.frames)
    if around is not None:
        # one steady morning light: the camera looks back south to the shore, and the sun of
        # midday (south) would shine straight into it off the water; at nine it comes from the
        # east-south-east, from the side, and the light no longer swings through the film
        import datetime as _dt
        when = (opts.when or _dt.datetime.now()).replace(hour=9, minute=0)
        elev, az = S.build_light(manifest["lat"], manifest["lon"], when, 0, opts.fps)
    else:
        elev, az = S.build_light(manifest["lat"], manifest["lon"], opts.when, tl.frames, opts.fps)
    build_camera(fp, top, tl, line[2], sea)
    S._all_constant_hides()
    S.configure_render(opts, tl.frames, out_dir)
    return {"setting": "marine", "clashes_removed": clashes, "google": got is not None, "modelled": around, "land_line": line,
            "sun": [round(elev, 1), round(az, 1)], "top": top, "footprint": fp}
