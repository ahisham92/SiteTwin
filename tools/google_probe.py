"""Follows Google's 3D tileset down the branch nearest a place, printing each level (for checks)."""
import json, os, sys
import numpy as np
from sitetwin_worker import google3d as g
from sitetwin_worker.environment import http_get

key = os.environ["SITETWIN_GOOGLE_KEY"].strip()
lat, lon = float(sys.argv[1]), float(sys.argv[2])
site = g.ecef(lat, lon)
url = g._absolute(g.ROOT, g.ROOT, key, None)
doc = json.loads(http_get(url))
tile, base, T, session = doc["root"], url, np.eye(4), None
for depth in range(40):
    T = T @ g._mat(tile["transform"]) if "transform" in tile else T
    c = tile.get("content") or {}
    uri = c.get("uri", "")
    kids = tile.get("children") or []
    d = g.distance_to_volume(tile["boundingVolume"], site, T)
    print(depth, "err", tile.get("geometricError"), "dist", round(d, 1), "kids", len(kids), "content", uri.split("?")[0][-30:])
    if uri.split("?")[0].endswith(".json"):
        url2 = g._absolute(uri, base, key, session)
        import urllib.parse
        session = session or dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url2).query)).get("session")
        sub = json.loads(http_get(url2))
        tile, base = sub["root"], url2
        continue
    if not kids:
        if uri:
            data = http_get(g._absolute(uri, base, key, session))
            print("leaf glb bytes", len(data))
        break
    tile = min(kids, key=lambda k: g.distance_to_volume(k["boundingVolume"], site, T))
