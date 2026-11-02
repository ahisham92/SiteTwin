"""Lists CC0 Poly Haven ground textures that suit a building site (names to use in materials)."""
import json, urllib.request
req = urllib.request.Request("https://api.polyhaven.com/assets?t=textures", headers={"User-Agent": "SiteTwin"})
data = json.load(urllib.request.urlopen(req, timeout=60))
words = ("gravel", "dirt", "mud", "soil", "ground", "concrete", "asphalt", "sand", "rock")
for k, v in sorted(data.items(), key=lambda kv: -kv[1].get("download_count", 0)):
    cats = " ".join(v.get("categories", []))
    if any(w in k or w in cats for w in words):
        print(f"{k:<36} {v.get('download_count', 0):>7}  {cats[:60]}")
