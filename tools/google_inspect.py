"""Downloads Google's tiles round a place and reports what is in them (no tiles are kept)."""
import collections, json, os, sys, tempfile
from pathlib import Path
from sitetwin_worker import google3d as g
from sitetwin_worker.environment import http_get

key = os.environ["SITETWIN_GOOGLE_KEY"].strip()
lat, lon, radius = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
tiles, best = g.select(g.ROOT, key, lat, lon, radius, http_get)
print("tiles chosen", len(tiles), "finest", best)
errs = collections.Counter()
ext, mimes, sizes, fails = collections.Counter(), collections.Counter(), [], 0
for url, T in tiles:
    try:
        data = http_get(url)
    except Exception as exc:
        fails += 1
        print("download failed:", type(exc).__name__, str(exc).replace(key, "…")[:120])
        continue
    doc, binary = g.read_glb(data)
    sizes.append(len(data))
    for e in doc.get("extensionsUsed", []):
        ext[e] += 1
    for im in doc.get("images", []):
        mimes[im.get("mimeType", "?")] += 1
    if not doc.get("images"):
        mimes["no image"] += 1
print("failed downloads", fails)
print("extensions", dict(ext))
print("images", dict(mimes))
if sizes:
    print("tile size kB: min", min(sizes) // 1000, "median", sorted(sizes)[len(sizes) // 2] // 1000, "max", max(sizes) // 1000)
