"""Google's Photorealistic 3D Tiles around a site: the real buildings, trees and ground as
textured meshes, where Google has flown them (most large cities in about 50 countries).

Needs ``SITETWIN_GOOGLE_KEY``, a Google Maps Platform key with the Map Tiles API turned on.

The tileset is walked from its root, keeping the tiles that touch a circle of ``radius`` around
the site and going down to finer tiles near the site than far from it. Each tile is a glTF
(.glb, Draco-compressed, in Earth-centred coordinates). Before it is saved its root nodes are
given, in double precision, the transform into SiteTwin's local frame (metres east, north, and
up from the ellipsoid at the site), so Blender's single-precision floats never see the
6,000 km offsets. The scene then lifts the mesh so its ground meets the site's (see scene.py).

If Google has no detailed 3D at the site (its finest tile there is coarse), nothing is kept and
the scene is built from terrain, imagery and OpenStreetMap as before.
"""

from __future__ import annotations

import json
import math
import os
import struct
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = "https://tile.googleapis.com/v1/3dtiles/root.json"
#: A tile at least this fine (geometric error, metres) near the site counts as real 3D.
DETAILED = 4.0
#: The most tiles one site downloads for its circle; each is 10 to 30 kB.
MAX_TILES = 700
#: How far the coarse ring of Google's 3D reaches beyond the circle, for a real horizon.
HORIZON = 2500.0
#: The most tiles of that ring.
MAX_FAR_TILES = 500

A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)

Fetch = Callable[[str, bytes | None], bytes]


# ---- the Earth's frames --------------------------------------------------------------------

def ecef(lat: float, lon: float, h: float = 0.0) -> np.ndarray:
    la, lo = math.radians(lat), math.radians(lon)
    n = A / math.sqrt(1 - E2 * math.sin(la) ** 2)
    return np.array([(n + h) * math.cos(la) * math.cos(lo), (n + h) * math.cos(la) * math.sin(lo),
                     (n * (1 - E2) + h) * math.sin(la)])


def enu_matrix(lat: float, lon: float) -> np.ndarray:
    """4×4 taking Earth-centred metres to metres east, north and up from (lat, lon, 0)."""
    la, lo = math.radians(lat), math.radians(lon)
    r = np.array([[-math.sin(lo), math.cos(lo), 0.0],
                  [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                  [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = -r @ ecef(lat, lon)
    return m


#: glTF is y-up; 3D Tiles (and Blender) are z-up: (x, y, z) -> (x, -z, y).
Y_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
Y_UP_INV = np.linalg.inv(Y_UP)


def _mat(values) -> np.ndarray:
    """A column-major 16-list (glTF, 3D Tiles) as a 4×4."""
    return np.array(values, dtype=float).reshape(4, 4).T


def _trs(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return _mat(node["matrix"])
    t = np.eye(4)
    t[:3, 3] = node.get("translation", [0, 0, 0])
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    r = np.eye(4)
    r[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    s = np.diag(list(node.get("scale", [1, 1, 1])) + [1])
    return t @ r @ s


# ---- bounding volumes --------------------------------------------------------------------------

def distance_to_volume(volume: dict[str, Any], point: np.ndarray, transform: np.ndarray) -> float:
    """How far (metres) an Earth-centred point is from a tile's bounding volume (0 inside)."""
    if "box" in volume:
        b = np.array(volume["box"], dtype=float)
        c = (transform @ np.append(b[:3], 1))[:3]
        axes = [transform[:3, :3] @ b[3 + 3 * k: 6 + 3 * k] for k in range(3)]
        d = point - c
        outside = 0.0
        for ax in axes:
            half = np.linalg.norm(ax)
            if half == 0:
                continue
            t = abs(d @ ax / half)
            outside += max(0.0, t - half) ** 2
        return math.sqrt(outside)
    if "sphere" in volume:
        s = volume["sphere"]
        c = (transform @ np.append(s[:3], 1))[:3]
        return max(0.0, float(np.linalg.norm(point - c)) - s[3])
    if "region" in volume:
        w, s_, e, n, lo, hi = volume["region"]
        lat = math.degrees(min(max(_lat_of(point), s_), n))
        lon = math.degrees(min(max(_lon_of(point), w), e))
        return float(np.linalg.norm(point - ecef(lat, lon, (lo + hi) / 2)))
    return 0.0


def _lat_of(p: np.ndarray) -> float:
    return math.atan2(p[2], math.hypot(p[0], p[1]) * (1 - E2))


def _lon_of(p: np.ndarray) -> float:
    return math.atan2(p[1], p[0])


# ---- the tileset -----------------------------------------------------------------------------

def _absolute(uri: str, base: str, key: str, session: str | None) -> str:
    url = urllib.parse.urljoin(base, uri)
    parts = urllib.parse.urlsplit(url)
    q = dict(urllib.parse.parse_qsl(parts.query))
    q.setdefault("key", key)
    if session and "session" not in q:
        q["session"] = session
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(q)))


def select(root_url: str, key: str, lat: float, lon: float, radius: float, fetch: Fetch,
           max_tiles: int = MAX_TILES, coarse: bool = False) -> tuple[list[tuple[str, np.ndarray]], float]:
    """The glTF tiles to draw the circle around the site with: [(url, tile transform)], and the
    finest geometric error found at the site itself. ``coarse`` asks for the far ring's level of
    detail (the horizon), a few metres of error close in and tens of metres far out."""
    site = ecef(lat, lon)
    chosen: list[tuple[str, np.ndarray]] = []
    best = math.inf
    session: list[str | None] = [None]

    def wanted_error(dist: float) -> float:
        if coarse:
            return max(8.0, dist * 0.03)
        # finest at the site, coarser with distance: about 1 cm of error per metre away
        return max(0.6, dist * 0.012)

    def walk(tile: dict[str, Any], base: str, parent: np.ndarray, depth: int) -> None:
        nonlocal best
        if len(chosen) >= max_tiles or depth > 40:
            return
        transform = parent @ _mat(tile["transform"]) if "transform" in tile else parent
        dist = distance_to_volume(tile.get("boundingVolume", {}), site, transform)
        if dist > radius:
            return
        error = float(tile.get("geometricError", 0))
        children = tile.get("children") or []
        content = tile.get("content") or {}
        uri = content.get("uri") or content.get("url")
        if uri and uri.split("?")[0].endswith(".json"):
            url = _absolute(uri, base, key, session[0])
            sub = json.loads(fetch(url, None))
            q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
            session[0] = session[0] or q.get("session")
            walk(sub["root"], url, transform, depth + 1)
            return
        if children and error > wanted_error(dist):
            for child in children:
                walk(child, base, transform, depth + 1)
            return
        if uri:
            chosen.append((_absolute(uri, base, key, session[0]), transform))
            # the site point is on the ellipsoid, and the ground can be tens of metres off it
            # (the geoid), so "at the site" is within 80 m of it
            if dist < 80:
                best = min(best, error)
        elif children:                                   # an empty tile: its children carry it
            for child in children:
                walk(child, base, transform, depth + 1)

    root_url = _absolute(root_url, root_url, key, None)
    root = json.loads(fetch(root_url, None))
    walk(root["root"], root_url, np.eye(4), 0)
    return chosen, best


# ---- one tile -------------------------------------------------------------------------------

def read_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError("not a glb")
    at, doc, binary = 12, None, b""
    while at < length:
        size, kind = struct.unpack_from("<I4s", data, at)
        chunk = data[at + 8: at + 8 + size]
        if kind == b"JSON":
            doc = json.loads(chunk)
        elif kind == b"BIN\x00":
            binary = chunk
        at += 8 + size
    if doc is None:
        raise ValueError("glb without JSON")
    return doc, binary


def write_glb(doc: dict[str, Any], binary: bytes) -> bytes:
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * ((4 - len(text) % 4) % 4)
    binary += b"\x00" * ((4 - len(binary) % 4) % 4)
    chunks = struct.pack("<I4s", len(text), b"JSON") + text
    if binary:
        chunks += struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def to_local(data: bytes, tile_transform: np.ndarray, local: np.ndarray) -> tuple[bytes, str]:
    """The tile with its root nodes moved into the local frame; and its copyright line."""
    doc, binary = read_glb(data)
    # Blender's importer turns glTF y-up into z-up by itself (Y_UP), so the node matrix it is
    # given must be the whole chain with that turn taken back off the front.
    rtc = np.eye(4)
    center = ((doc.get("extensions") or {}).get("CESIUM_RTC") or {}).get("center")
    if center:
        rtc[:3, 3] = center
        doc["extensions"].pop("CESIUM_RTC", None)
        if "extensionsUsed" in doc:
            doc["extensionsUsed"] = [e for e in doc["extensionsUsed"] if e != "CESIUM_RTC"]
    pre = Y_UP_INV @ local @ tile_transform @ rtc @ Y_UP
    scene = doc.get("scenes", [{}])[doc.get("scene", 0)] if doc.get("scenes") else {"nodes": []}
    for i in scene.get("nodes", []):
        node = doc["nodes"][i]
        m = pre @ _trs(node)
        for k in ("translation", "rotation", "scale"):
            node.pop(k, None)
        node["matrix"] = [float(v) for v in m.T.reshape(-1)]
    return write_glb(doc, binary), (doc.get("asset") or {}).get("copyright", "")


def gather(lat: float, lon: float, radius: float, folder: Path, fetch: Fetch,
           key: str | None = None, horizon: float = HORIZON) -> dict[str, Any]:
    """Download the tiles into ``folder``: the detailed circle, and a coarse ring out to
    ``horizon`` so the skyline is real. Returns the manifest's "google" entry, or raises
    LookupError when Google has no detailed 3D here (the caller falls back)."""
    key = key or os.environ.get("SITETWIN_GOOGLE_KEY", "")
    if not key:
        raise LookupError("no SITETWIN_GOOGLE_KEY")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        tiles, best = select(ROOT, key, lat, lon, radius, fetch)
        far: list[tuple[str, np.ndarray]] = []
        if horizon > radius and tiles and best <= DETAILED:
            try:
                far, _ = select(ROOT, key, lat, lon, horizon, fetch, max_tiles=MAX_FAR_TILES, coarse=True)
            except Exception:                            # noqa: BLE001 - the horizon is a bonus
                far = []
    except LookupError:
        raise
    except Exception as exc:                             # never let the key reach a message
        raise RuntimeError(str(exc).replace(key, "…")) from None
    if best > DETAILED or not tiles:
        raise LookupError(f"Google has no detailed 3D here (finest tile {best:.0f} m)")
    local = enu_matrix(lat, lon)
    files, far_files, credits = [], [], set()

    def one(job: tuple[str, tuple[str, np.ndarray]]) -> tuple[str, str] | None:
        name, (url, transform) = job
        try:
            glb, credit = to_local(fetch(url, None), transform, local)
        except Exception:                                # noqa: BLE001 - a missing tile is a small gap
            return None
        (folder / name).write_bytes(glb)
        return name, credit

    jobs = [(f"tile_{n:04d}.glb", t) for n, t in enumerate(tiles)] + \
           [(f"far_{n:04d}.glb", t) for n, t in enumerate(far)]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as pool:
        for got in pool.map(one, jobs):
            if got is None:
                continue
            (far_files if got[0].startswith("far_") else files).append(got[0])
            for c in got[1].split(";"):
                if c.strip() and c.strip() != "Google":
                    credits.add(c.strip())
    if not files:
        raise LookupError("none of Google's tiles could be downloaded")
    return {"folder": folder.name, "files": files, "far_files": far_files, "radius": radius,
            "horizon": horizon if far_files else radius, "finest": best,
            "credit": "Google" + (" · " + "; ".join(sorted(credits)) if credits else "")}
