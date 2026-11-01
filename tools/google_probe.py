"""Walks Google's 3D tileset down every branch within 60 m of a place, level by level (for checks)."""
import json, os, sys, urllib.parse
import numpy as np
from sitetwin_worker import google3d as g
from sitetwin_worker.environment import http_get

key = os.environ["SITETWIN_GOOGLE_KEY"].strip()
lat, lon = float(sys.argv[1]), float(sys.argv[2])
site = g.ecef(lat, lon)
url = g._absolute(g.ROOT, g.ROOT, key, None)
level = [(json.loads(http_get(url))["root"], url, np.eye(4))]
session = None
for depth in range(45):
    nxt, errs, glbs = [], [], []
    for tile, base, T in level:
        T = T @ g._mat(tile["transform"]) if "transform" in tile else T
        if g.distance_to_volume(tile["boundingVolume"], site, T) > 60:
            continue
        errs.append(tile.get("geometricError"))
        uri = (tile.get("content") or {}).get("uri", "")
        if uri.split("?")[0].endswith(".json"):
            u = g._absolute(uri, base, key, session)
            session = session or dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(u).query)).get("session")
            nxt.append((json.loads(http_get(u))["root"], u, T))
            continue
        if uri:
            glbs.append((g._absolute(uri, base, key, session), bool(tile.get("children"))))
        for k in tile.get("children") or []:
            nxt.append((k, base, T))
    if not errs:
        break
    leaf = [u for u, kids in glbs if not kids]
    print(depth, "tiles near", len(errs), "error min", min(errs), "max", max(errs), "glb", len(glbs), "leaves", len(leaf))
    if leaf:
        print("   a leaf glb is", len(http_get(leaf[0])), "bytes")
    level = nxt
