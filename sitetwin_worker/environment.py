"""The real surroundings of a site, fetched from its latitude and longitude alone.

Three free, worldwide sources, each optional; whatever cannot be had is noted in
the manifest and the scene carries on without it:

* **Terrain**: AWS Terrain Tiles (Terrarium PNGs, from SRTM, Copernicus and
  national survey data), sampled into a height grid in metres.
* **Aerial imagery**: any XYZ imagery service (Esri World Imagery by default;
  set ``SITETWIN_IMAGERY_URL`` for Mapbox, MapTiler, a national orthophoto...),
  cut into one picture laid exactly over that grid.
* **Photo-real 3D** (optional): Google's Photorealistic 3D Tiles where Google has them
  and ``SITETWIN_GOOGLE_KEY`` is set (see google3d.py); the buildings then come from it.
* **What is built around it**: OpenStreetMap through the Overpass API:
  buildings with their heights or storeys, roads, railways, water and land use.

Everything is written into one folder with ``manifest.json`` beside it, and is
reused on the next run rather than fetched again.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .geo import Local, TILE, lonlat_to_pixel, metres_per_pixel

USER_AGENT = "SiteTwin/0.1 (construction sequence renderer)"


@dataclass
class Sources:
    terrain: str = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
    imagery: str = field(default_factory=lambda: os.environ.get(
        "SITETWIN_IMAGERY_URL",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"))
    imagery_credit: str = field(default_factory=lambda: os.environ.get(
        "SITETWIN_IMAGERY_CREDIT", "Imagery: Esri, Maxar, Earthstar Geographics"))
    overpass: str = field(default_factory=lambda: os.environ.get(
        "SITETWIN_OVERPASS_URL", "https://overpass-api.de/api/interpreter"))


Fetch = Callable[[str, bytes | None], bytes]


def http_get(url: str, data: bytes | None = None, tries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as exc:                       # noqa: BLE001 - retried, then raised
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


# ---- tiles -------------------------------------------------------------------------------

def _mosaic(local: Local, radius: float, zoom: int, url: str, fetch: Fetch,
            size: int, mode: str) -> np.ndarray:
    """Fetch the tiles covering the square and resample them onto a size×size grid
    whose rows run north→south and columns west→east, centred on the origin."""
    xs = np.linspace(-radius, radius, size)
    ys = np.linspace(radius, -radius, size)
    s, w, n, e = local.bbox(radius)
    px0, py0 = lonlat_to_pixel(w, n, zoom)
    px1, py1 = lonlat_to_pixel(e, s, zoom)
    tx0, ty0, tx1, ty1 = int(px0 // TILE), int(py0 // TILE), int(px1 // TILE), int(py1 // TILE)
    cols, rows = tx1 - tx0 + 1, ty1 - ty0 + 1
    if cols * rows > 400:
        raise ValueError("The area asked for needs too many tiles; make the radius smaller.")
    channels = 3
    canvas = np.zeros((rows * TILE, cols * TILE, channels), dtype=np.uint8)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            raw = fetch(url.format(z=zoom, x=tx, y=ty), None)
            tile = Image.open(io.BytesIO(raw)).convert("RGB").resize((TILE, TILE))
            canvas[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE,
                   (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = np.asarray(tile)
    # Where each output cell falls in the canvas.
    gx, gy = np.meshgrid(xs, ys)
    lon = local.lon + np.degrees(gx / (6378137.0 * math.cos(math.radians(local.lat))))
    lat = local.lat + np.degrees(gy / 6378137.0)
    world = TILE * 2 ** zoom
    px = (lon + 180) / 360 * world - tx0 * TILE
    lat_r = np.radians(lat)
    py = (1 - np.log(np.tan(lat_r) + 1 / np.cos(lat_r)) / math.pi) / 2 * world - ty0 * TILE
    if mode == "nearest":
        ix = np.clip(np.round(px).astype(int), 0, canvas.shape[1] - 1)
        iy = np.clip(np.round(py).astype(int), 0, canvas.shape[0] - 1)
        return canvas[iy, ix]
    # bilinear, for heights
    x0 = np.clip(np.floor(px).astype(int), 0, canvas.shape[1] - 2)
    y0 = np.clip(np.floor(py).astype(int), 0, canvas.shape[0] - 2)
    fx, fy = (px - x0)[..., None], (py - y0)[..., None]
    c = canvas.astype(np.float64)
    top = c[y0, x0] * (1 - fx) + c[y0, x0 + 1] * fx
    bottom = c[y0 + 1, x0] * (1 - fx) + c[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def terrarium_to_metres(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return r * 256.0 + g + b / 256.0 - 32768.0


def fetch_terrain(local: Local, radius: float, fetch: Fetch, url: str, size: int = 129) -> dict[str, Any]:
    zoom = 14 if radius > 900 else 15
    size = max(size, int(2 * radius / 5.5) | 1)        # about one sample every 5.5 m
    size = min(size, 513)
    # Heights are decoded per channel before interpolation would mix the bytes,
    # so sample nearest and smooth afterwards.
    rgb = _mosaic(local, radius, zoom, url, fetch, size, "nearest").astype(np.float64)
    heights = terrarium_to_metres(rgb)
    heights = _smooth(heights)
    return {"heights": heights, "zoom": zoom}


def _smooth(h: np.ndarray) -> np.ndarray:
    p = np.pad(h, 1, mode="edge")
    return (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:] + 2 * p[1:-1, :-2] + 4 * p[1:-1, 1:-1]
            + 2 * p[1:-1, 2:] + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) / 16.0


def fetch_imagery(local: Local, radius: float, fetch: Fetch, url: str, size: int = 2048) -> Image.Image:
    # About 1 m a pixel where the service has it; fewer tiles for a big area.
    zoom = 18 if radius <= 300 else 17 if radius <= 700 else 16
    while zoom > 12 and (2 * radius / (metres_per_pixel(local.lat, zoom) * TILE)) ** 2 > 300:
        zoom -= 1
    rgb = _mosaic(local, radius, zoom, url, fetch, size, "nearest")
    return Image.fromarray(rgb.astype(np.uint8))


# ---- OpenStreetMap ------------------------------------------------------------------------

LEVEL_HEIGHT = 3.2


def _height(tags: dict[str, str], key: int) -> float:
    for k in ("height", "building:height"):
        if k in tags:
            try:
                return max(2.5, float(tags[k].split()[0].replace(",", ".")))
            except ValueError:
                pass
    if "building:levels" in tags:
        try:
            return max(1.0, float(tags["building:levels"])) * LEVEL_HEIGHT + 1.0
        except ValueError:
            pass
    # Not mapped: a believable, stable spread rather than one flat height.
    kind = tags.get("building", "yes")
    base = {"house": 7, "residential": 12, "apartments": 18, "industrial": 9, "warehouse": 10,
            "commercial": 14, "office": 20, "garage": 3, "shed": 3, "retail": 6}.get(kind, 9)
    spread = int(hashlib.md5(str(key).encode()).hexdigest()[:4], 16) / 65535.0
    return round(base * (0.75 + 0.5 * spread), 1)


ROAD_WIDTH = {"motorway": 14, "trunk": 12, "primary": 11, "secondary": 9, "tertiary": 8,
              "residential": 6.5, "unclassified": 6, "service": 4.5, "living_street": 5,
              "footway": 2, "path": 1.5, "cycleway": 2, "pedestrian": 4, "track": 3}


def overpass_query(local: Local, radius: float) -> str:
    s, w, n, e = local.bbox(radius)
    bb = f"({s:.6f},{w:.6f},{n:.6f},{e:.6f})"
    return (f"[out:json][timeout:90];("
            f"way[building]{bb};relation[building]{bb};"
            f"way[highway]{bb};way[railway=rail]{bb};"
            f"way[natural=water]{bb};relation[natural=water]{bb};way[waterway]{bb};"
            f"way[landuse]{bb};way[leisure~\"park|pitch|garden\"]{bb};way[natural~\"wood|scrub|grassland|beach|sand\"]{bb};"
            f"way[man_made~\"pier|breakwater\"]{bb};"
            f");out geom;")


def features_from_osm(data: dict[str, Any], local: Local) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        rings: list[list[dict[str, float]]] = []
        if el.get("type") == "way" and el.get("geometry"):
            rings = [el["geometry"]]
        elif el.get("type") == "relation":
            rings = [m["geometry"] for m in el.get("members", [])
                     if m.get("role") == "outer" and m.get("geometry")]
        for ring in rings:
            pts = [list(local.to_xy(p["lon"], p["lat"])) for p in ring]
            if len(pts) < 2:
                continue
            closed = len(pts) > 3 and pts[0] == pts[-1]
            f: dict[str, Any] | None = None
            if "building" in tags and closed:
                f = {"kind": "building", "height": _height(tags, el["id"]),
                     "use": tags.get("building", "yes")}
            elif "highway" in tags and tags["highway"] in ROAD_WIDTH:
                f = {"kind": "road", "width": float(tags.get("width", ROAD_WIDTH[tags["highway"]]) or 6)
                     if str(tags.get("width", "")).replace(".", "").isdigit() else ROAD_WIDTH[tags["highway"]],
                     "class": tags["highway"],
                     "raised": tags.get("bridge", "no") != "no" or tags.get("tunnel", "no") != "no"
                     or str(tags.get("layer", "0")) not in ("0", "")}
            elif tags.get("railway") == "rail":
                f = {"kind": "rail", "width": 3.0}
            elif (tags.get("natural") == "water" or "waterway" in tags) and closed:
                f = {"kind": "water"}
            elif "waterway" in tags:
                f = {"kind": "river", "width": 8.0 if tags["waterway"] == "river" else 3.0}
            elif tags.get("man_made") in ("pier", "breakwater"):
                f = {"kind": "pier" if closed else "pier_line", "width": 8.0}
            elif closed and (tags.get("landuse") in ("grass", "meadow", "forest", "recreation_ground", "village_green")
                             or tags.get("leisure") in ("park", "pitch", "garden")
                             or tags.get("natural") in ("wood", "scrub", "grassland")):
                f = {"kind": "green", "use": tags.get("landuse") or tags.get("leisure") or tags.get("natural")}
            elif closed and tags.get("natural") in ("beach", "sand"):
                f = {"kind": "sand"}
            if f is not None:
                f["id"] = el["id"]
                f["points"] = [[round(x, 2), round(y, 2)] for x, y in pts]
                out.append(f)
    return out


#: Public Overpass servers tried in turn when one is busy (they often answer 502/504 under load).
OVERPASS_MIRRORS = ("https://overpass-api.de/api/interpreter",
                    "https://overpass.kumi.systems/api/interpreter",
                    "https://overpass.private.coffee/api/interpreter")


def fetch_osm(local: Local, radius: float, fetch: Fetch, url: str, wait=None) -> list[dict[str, Any]]:
    """OpenStreetMap features round the site; a busy server is retried, then the mirrors."""
    import time
    wait = wait or time.sleep
    body = urllib.parse.urlencode({"data": overpass_query(local, radius)}).encode()
    urls = [url] + [u for u in OVERPASS_MIRRORS if u != url]
    last: Exception | None = None
    for attempt, u in enumerate([urls[0], urls[0]] + urls[1:]):
        try:
            return features_from_osm(json.loads(fetch(u, body)), local)
        except Exception as exc:                          # noqa: BLE001 - try again elsewhere
            last = exc
            wait(min(20, 4 * (attempt + 1)))
    raise last if last else RuntimeError("no Overpass server answered")


# ---- all of it --------------------------------------------------------------------------

def gather(lat: float, lon: float, radius: float, folder: Path, sources: Sources | None = None,
           fetch: Fetch = http_get, refresh: bool = False, intro: bool = True) -> dict[str, Any]:
    """Fetch (or reuse) everything around (lat, lon) into ``folder``; returns the manifest."""
    sources = sources or Sources()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists() and not refresh:
        old = json.loads(manifest_path.read_text())
        if old.get("lat") == lat and old.get("lon") == lon and old.get("radius") == radius:
            return old

    local = Local(lat, lon)
    manifest: dict[str, Any] = {"lat": lat, "lon": lon, "radius": radius, "missing": {}, "credits": []}

    try:
        t = fetch_terrain(local, radius, fetch, sources.terrain)
        np.save(folder / "terrain.npy", t["heights"])
        manifest["terrain"] = {"file": "terrain.npy", "size": int(t["heights"].shape[0]),
                               "origin_height": float(t["heights"][t["heights"].shape[0] // 2,
                                                                   t["heights"].shape[1] // 2])}
        manifest["credits"].append("Terrain: AWS Terrain Tiles (SRTM, Copernicus, national surveys)")
    except Exception as exc:                           # noqa: BLE001 - the scene goes on flat
        manifest["missing"]["terrain"] = str(exc)

    try:
        fetch_imagery(local, radius, fetch, sources.imagery).save(folder / "imagery.jpg", quality=92)
        manifest["imagery"] = {"file": "imagery.jpg"}
        manifest["credits"].append(sources.imagery_credit)
    except Exception as exc:                           # noqa: BLE001 - the scene goes on untextured
        manifest["missing"]["imagery"] = str(exc)

    try:
        from . import google3d
        manifest["google"] = google3d.gather(lat, lon, radius, folder / "google", fetch, intro=intro)
        manifest["credits"].insert(0, "3D surroundings: " + manifest["google"]["credit"])
    except LookupError as exc:                         # no key, or Google has no 3D here
        manifest["google_skipped"] = str(exc)
    except Exception as exc:                           # noqa: BLE001 - the scene goes on without it
        manifest["missing"]["google 3D"] = str(exc)

    try:
        feats = fetch_osm(local, radius, fetch, sources.overpass)
        (folder / "osm.json").write_text(json.dumps(feats))
        manifest["osm"] = {"file": "osm.json", "counts": _count(feats)}
        manifest["credits"].append("Buildings, roads and water: © OpenStreetMap contributors")
    except Exception as exc:                           # noqa: BLE001 - the scene goes on empty
        manifest["missing"]["osm"] = str(exc)

    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest


def _count(feats: list[dict[str, Any]]) -> dict[str, int]:
    c: dict[str, int] = {}
    for f in feats:
        c[f["kind"]] = c.get(f["kind"], 0) + 1
    return c
