"""The streets right round the site, rebuilt clean from what Google's photo mesh shows.

Google's 3D tiles are a single photo-draped mesh: from far off they read as a real city, but
close to the site the buildings melt into the trees and cars, and the walls smear. Inside a
radius of about 150 m this module turns that mesh into separate, clean things:

* **Buildings**: every OpenStreetMap footprint near the site becomes a crisp block. Its height
  is read off Google's mesh (the roof's real level, not a guess), its wall and roof colours are
  sampled from Google's photos of it, and it gets floors and windows. Google's own surface
  inside (and just round) the footprint is pressed down out of sight under it.
* **Ground**: cars, kiosks and other low clutter on the ground are pressed flat onto a smoothed
  ground surface, so streets and pavements read as flat, level surfaces.
* **Roads**: OpenStreetMap roads are laid over that ground as asphalt with edge lines.

Trees and anything taller that is not a mapped building stay as Google shows them.

Everything here is plain numpy, so it is tested without Blender; scene.py applies it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

RADIUS = 150.0          # metres round the site centre that are rebuilt
FADE = 25.0             # over the last metres the ground flattening fades out
CLUTTER = 2.6           # anything up to this far above the ground (cars, kiosks) is pressed flat
BUFFER = 1.5            # Google's surface this close round a footprint goes under the building


def ground_grid(xyz: np.ndarray, extent: float, cell: float = 4.0) -> tuple[np.ndarray, float, float]:
    """A smoothed bare-ground height grid from the photo mesh's points (x, y, z): the low points
    of each cell, with a 3x3 minimum (so a tree or car no wider than a cell drops out) and a 3x3
    mean (so it is smooth). Returns (grid, origin, cell)."""
    n = int(math.ceil(2 * extent / cell)) + 1
    origin = -extent
    ix = np.clip(((xyz[:, 0] - origin) / cell).astype(int), 0, n - 1)
    iy = np.clip(((xyz[:, 1] - origin) / cell).astype(int), 0, n - 1)
    grid = np.full((n, n), np.nan)
    order = np.lexsort((xyz[:, 2], iy, ix))
    keys = ix[order] * n + iy[order]
    first = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
    counts = np.diff(np.r_[first, len(keys)])
    lowish = first + (counts * 0.15).astype(int)                 # the 15th percentile in each cell
    grid.flat[keys[first]] = xyz[order[lowish], 2]
    grid = _fill(grid)
    grid = _filter(grid, np.min)
    return _filter(grid, np.mean), origin, cell


def _fill(grid: np.ndarray) -> np.ndarray:
    g = grid.copy()
    for _ in range(64):
        holes = np.isnan(g)
        if not holes.any():
            break
        p = np.pad(g, 1, constant_values=np.nan)
        stack = np.stack([p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        import warnings
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            near = np.nanmean(stack, axis=0)
        g[holes] = near[holes]
    return np.nan_to_num(g, nan=0.0)


def _filter(grid: np.ndarray, fn) -> np.ndarray:
    p = np.pad(grid, 1, mode="edge")
    stack = np.stack([p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    return fn(stack, axis=0)


def sample(grid: np.ndarray, origin: float, cell: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear height of the grid at points (cell centres at origin + (i + 0.5) * cell)."""
    n = grid.shape[0]
    fx = np.clip((np.asarray(x) - origin) / cell - 0.5, 0, n - 1.001)
    fy = np.clip((np.asarray(y) - origin) / cell - 0.5, 0, n - 1.001)
    i, j = fx.astype(int), fy.astype(int)
    tx, ty = fx - i, fy - j
    return (grid[i, j] * (1 - tx) * (1 - ty) + grid[i + 1, j] * tx * (1 - ty)
            + grid[i, j + 1] * (1 - tx) * ty + grid[i + 1, j + 1] * tx * ty)


def inside(xy: np.ndarray, ring: list, buffer: float = 0.0) -> np.ndarray:
    """Which points lie inside the polygon, or within ``buffer`` metres of its edge."""
    pts = np.asarray(ring, float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    x, y = xy[:, 0], xy[:, 1]
    res = np.zeros(len(xy), bool)
    a, b = pts, np.roll(pts, -1, axis=0)
    for (x0, y0), (x1, y1) in zip(a, b):
        cross = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        res ^= cross & (x < xi)
    if buffer > 0:
        for (x0, y0), (x1, y1) in zip(a, b):
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy or 1e-9
            t = np.clip(((x - x0) * dx + (y - y0) * dy) / L2, 0, 1)
            res |= np.hypot(x - (x0 + t * dx), y - (y0 + t * dy)) <= buffer
    return res


def plan(xyz: np.ndarray, colour: np.ndarray, steep: np.ndarray, feats: list[dict[str, Any]],
         keep_out: tuple[float, float, float, float], radius: float = RADIUS) -> dict[str, Any]:
    """From the photo mesh's points (lowered to the site's level), their sampled photo colours
    (n x 3, linear) and whether each sits on a steep (wall) face: the buildings to put up, and the
    new height of every point. Returns {"z": new z per point, "buildings": [...], "roads": [...],
    "trees": [...], "grid": (grid, origin, cell)}."""
    kx0, ky0, kx1, ky1 = keep_out
    z = xyz[:, 2].copy()
    xy = xyz[:, :2]
    grid, origin, cell = ground_grid(xyz[np.abs(xy).max(axis=1) < radius + 40], radius + 40)
    ground = sample(grid, origin, cell, xy[:, 0], xy[:, 1])
    r = np.hypot(xy[:, 0], xy[:, 1])

    def near_site(px, py):
        return kx0 - 1 <= px <= kx1 + 1 and ky0 - 1 <= py <= ky1 + 1

    buildings = []
    for f in feats:
        if f.get("kind") != "building":
            continue
        ring = f["points"][:-1] if f["points"][0] == f["points"][-1] else f["points"]
        if len(ring) < 3:
            continue
        cx = sum(p[0] for p in ring) / len(ring)
        cy = sum(p[1] for p in ring) / len(ring)
        if math.hypot(cx, cy) > radius or near_site(cx, cy):
            continue
        xs, ys = [p[0] for p in ring], [p[1] for p in ring]
        box = ((xy[:, 0] >= min(xs) - 4) & (xy[:, 0] <= max(xs) + 4)
               & (xy[:, 1] >= min(ys) - 4) & (xy[:, 1] <= max(ys) + 4))
        idx = np.flatnonzero(box)
        if len(idx) < 6:
            continue
        sub = xy[idx]
        core = inside(sub, ring)
        if core.sum() < 4:
            continue
        under = idx[inside(sub, ring, BUFFER)]
        base = float(np.min(sample(grid, origin, cell, np.array(xs), np.array(ys))))
        top = float(np.percentile(z[idx[core]], 85))
        height = top - base
        if height < 2.5:
            continue                                         # nothing standing there in Google
        roofs = idx[core][(~steep[idx[core]]) & (z[idx[core]] > top - 2.0)]
        walls = idx[core | inside(sub, ring, 1.0)][steep[idx[core | inside(sub, ring, 1.0)]]]
        roof = _median_colour(colour, roofs, (0.45, 0.43, 0.40))
        wall = _median_colour(colour, walls, (0.62, 0.58, 0.52))
        z[under] = np.minimum(z[under], base - 0.4)           # Google's version goes under the block
        buildings.append({"id": f.get("id"), "points": [list(map(float, p)) for p in ring],
                          "base": base, "top": top, "wall": wall, "roof": roof,
                          "floors": max(1, int(round(height / 3.2)))})

    # Google's trees are green blobs: each is found, pressed flat, and a real tree stands there
    w = np.clip((radius - r) / FADE, 0.0, 1.0)
    trees, pressed = find_trees(xy, z, ground, colour, buildings, keep_out, radius - FADE)
    z[pressed] = ground[pressed]

    # low clutter on the ground pressed flat (fading out at the edge of the rebuilt zone)
    over = z - ground
    clutter = (over > 0.0) & (over < CLUTTER) & (w > 0)
    z[clutter] = z[clutter] - over[clutter] * w[clutter]

    roads = []
    for f in feats:
        if f.get("kind") != "road" or f.get("class") in ("footway", "path", "cycleway", "track",
                                                          "motorway", "trunk") or f.get("raised"):
            continue
        pts = [p for p in f["points"]]
        if not any(_seg_dist(a, b) < radius for a, b in zip(pts[:-1], pts[1:])):
            continue
        roads.append({"points": pts, "width": float(f.get("width", 6.0)), "class": f.get("class")})
    return {"z": z, "buildings": buildings, "roads": roads, "trees": trees, "grid": (grid, origin, cell)}


def _seg_dist(a, b) -> float:
    """Distance from the site centre to the segment a-b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy or 1e-9
    t = min(1.0, max(0.0, -(a[0] * dx + a[1] * dy) / L2))
    return math.hypot(a[0] + t * dx, a[1] + t * dy)


def _median_colour(colour: np.ndarray, idx: np.ndarray, default) -> tuple[float, float, float]:
    if len(idx) < 3:
        return tuple(default)
    c = np.median(colour[idx], axis=0)
    # photos carry the day's shadows and haze: lift dark samples a little toward a plaster grey
    lum = float(c @ np.array([0.2126, 0.7152, 0.0722]))
    if lum < 0.18:
        c = c + (0.18 - lum)
    return tuple(float(v) for v in np.clip(c, 0.03, 0.9))


def road_quads(road: dict[str, Any], grid, radius: float, keep_out, lift: float = 0.06):
    """Asphalt quads (and two thin edge lines) along a road, draped on the ground grid, cut at the
    rebuilt zone's edge and at the site. Returns (verts, faces, line_verts, line_faces)."""
    g, origin, cell = grid
    kx0, ky0, kx1, ky1 = keep_out
    w = road["width"] / 2
    verts, faces, lv, lf = [], [], [], []
    pts = road["points"]
    for a, b in zip(pts[:-1], pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 0.2:
            continue
        nx, ny = -dy / L, dx / L
        steps = max(1, int(L / 4.0))
        for s in range(steps):
            t0, t1 = s / steps, (s + 1) / steps
            p0 = (a[0] + dx * t0, a[1] + dy * t0)
            p1 = (a[0] + dx * t1, a[1] + dy * t1)
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            if math.hypot(mx, my) > radius or (kx0 - w - 2 <= mx <= kx1 + w + 2 and ky0 - w - 2 <= my <= ky1 + w + 2):
                continue
            strips = [(-w, w, verts, faces), (w - 0.45, w - 0.3, lv, lf), (-w + 0.3, -w + 0.45, lv, lf)]
            if w >= 3.0 and s % 2 == 0:                      # a dashed centre line
                strips.append((-0.07, 0.07, lv, lf))
            for off0, off1, vs, fs in strips:
                quad = [(p0[0] + nx * off0, p0[1] + ny * off0), (p1[0] + nx * off0, p1[1] + ny * off0),
                        (p1[0] + nx * off1, p1[1] + ny * off1), (p0[0] + nx * off1, p0[1] + ny * off1)]
                xs = np.array([q[0] for q in quad])
                ys = np.array([q[1] for q in quad])
                zs = sample(g, origin, cell, xs, ys) + (lift if vs is verts else lift + 0.01)
                k = len(vs)
                vs.extend((float(x), float(y), float(zz)) for x, y, zz in zip(xs, ys, zs))
                fs.append((k, k + 1, k + 2, k + 3))
    return verts, faces, lv, lf


TREE_CELL = 5.0         # one tree per this many metres of green canopy
MAX_TREES = 160


def find_trees(xy: np.ndarray, z: np.ndarray, ground: np.ndarray, colour: np.ndarray,
               buildings: list, keep_out, radius: float) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Green points standing 2 to 20 m over the ground, off the buildings: Google's trees.
    Returns one tree per 5 m cell of them (as tall as the canopy), and the indices of all those
    points, to press flat."""
    kx0, ky0, kx1, ky1 = keep_out
    over = z - ground
    c = colour
    # photo greens are dark and dusty: greener than red and blue, by a little
    green = (c[:, 1] >= c[:, 0] * 0.98) & (c[:, 1] > c[:, 2] * 1.03) & (c[:, 1] < 0.5)
    tall = (over > 2.0) & (over < 20.0) & (np.hypot(xy[:, 0], xy[:, 1]) < radius)
    cand = green & tall
    print(f"  trees: {int(green.sum())} green points, {int(tall.sum())} standing, {int(cand.sum())} both")
    cand &= ~((xy[:, 0] > kx0 - 4) & (xy[:, 0] < kx1 + 4) & (xy[:, 1] > ky0 - 4) & (xy[:, 1] < ky1 + 4))
    idx = np.flatnonzero(cand)
    for b in buildings:                                   # never a tree on a roof
        if not len(idx):
            break
        idx = idx[~inside(xy[idx], b["points"], 1.0)]
    cells: dict[tuple[int, int], list[int]] = {}
    for i, (px, py) in zip(idx, (xy[idx] // TREE_CELL).astype(int)):
        cells.setdefault((int(px), int(py)), []).append(int(i))
    out = []
    for pts in cells.values():
        if len(pts) < 3:
            continue
        p = np.array(pts)
        out.append({"x": float(xy[p, 0].mean()), "y": float(xy[p, 1].mean()),
                    "z": float(np.median(ground[p])),
                    "height": float(np.clip(np.percentile(over[p], 90), 4.0, 15.0)), "_n": len(pts)})
    out.sort(key=lambda t: -t.pop("_n"))
    return out[:MAX_TREES], idx


PARKING = ("residential", "tertiary", "secondary", "primary", "unclassified", "living_street")
DRIVING = ("tertiary", "secondary", "primary", "residential", "unclassified")


def street_life(roads: list, buildings: list, grid, keep_out, clear: float = 0.0,
                radius: float = RADIUS, seed: int = 3, max_parked: int = 110,
                max_moving: int = 6) -> dict[str, list]:
    """Cars along the streets round the site: parked ones at the kerbs (with gaps, never on a
    building or by the site), and a few driving on the right. Returns {"parked": [(x, y, z,
    heading)], "moving": [[(x, y, z), ...] path, ...]}, in the ground grid's heights.
    ``clear``: metres round the site (its graded ring) kept free."""
    import random
    rng = random.Random(seed)
    g, origin, cell = grid
    kx0, ky0, kx1, ky1 = keep_out

    def from_site(x, y):
        return math.hypot(max(kx0 - x, x - kx1, 0.0), max(ky0 - y, y - ky1, 0.0))

    def free(x, y):
        if math.hypot(x, y) > radius - 12 or from_site(x, y) < clear + 3:
            return False
        p = np.array([[x, y]])
        return not any(inside(p, b["points"], 0.6)[0] for b in buildings
                       if abs(b["points"][0][0] - x) < 120 and abs(b["points"][0][1] - y) < 120)

    ends = [tuple(r["points"][i]) for r in roads for i in (0, -1)]
    parked = []
    for r in roads:
        if r.get("class") not in PARKING or r["width"] < 4.5:
            continue
        w = r["width"] / 2
        pts = r["points"]
        for a, b in zip(pts[:-1], pts[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy)
            if L < 8:
                continue
            ux, uy = dx / L, dy / L
            nx, ny = -uy, ux
            for side in (1, -1):
                t = rng.uniform(2.0, 6.0)
                while t < L - 3.0:
                    if rng.random() < 0.6:
                        x = a[0] + ux * t + nx * side * (w - 1.1)
                        y = a[1] + uy * t + ny * side * (w - 1.1)
                        corner = any(math.hypot(x - ex, y - ey) < 9 for ex, ey in ends)
                        if not corner and free(x, y):
                            zz = float(sample(g, origin, cell, np.array([x]), np.array([y]))[0])
                            head = math.atan2(uy, ux) + (math.pi if rng.random() < 0.15 else 0.0)
                            parked.append((x, y, zz, head))
                    t += rng.uniform(5.4, 6.6)
    rng.shuffle(parked)
    parked = parked[:max_parked]

    moving = []
    lanes = sorted((r for r in roads if r.get("class") in DRIVING),
                   key=lambda r: -sum(math.dist(a, b) for a, b in zip(r["points"][:-1], r["points"][1:])))
    for r in lanes[:max_moving]:
        pts = r["points"] if rng.random() < 0.5 else r["points"][::-1]
        off = r["width"] / 4
        path = []
        for a, b in zip(pts[:-1], pts[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy)
            if L < 0.5:
                continue
            ux, uy = dx / L, dy / L
            steps = max(1, int(L / 3.0))
            for s in range(steps + 1):
                x = a[0] + dx * s / steps + uy * off              # keep right
                y = a[1] + dy * s / steps - ux * off
                if math.hypot(x, y) > radius - 5 or from_site(x, y) < clear + 2:
                    if len(path) >= 2:
                        moving.append(path)
                    path = []
                    continue
                zz = float(sample(g, origin, cell, np.array([x]), np.array([y]))[0])
                if not path or math.dist(path[-1][:2], (x, y)) > 0.5:
                    path.append((x, y, zz))
        if len(path) >= 2:
            moving.append(path)
    moving = [p for p in moving if sum(math.dist(a[:2], b[:2]) for a, b in zip(p[:-1], p[1:])) > 30]
    return {"parked": parked, "moving": moving[:max_moving]}
