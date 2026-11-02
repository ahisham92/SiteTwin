"""Lists free-licence, downloadable Sketchfab models for each kind of thing a site film needs,
best first, so they can be picked by hand. Needs no token (searching is public)."""
import json, sys, urllib.parse, urllib.request

KINDS = {
    "worker": ["construction worker", "worker scan", "workman", "builder man", "hard hat worker"],
    "flatbed_truck": ["flatbed lorry", "volvo truck", "scania truck", "mercedes actros", "man tgs truck", "dump truck", "semi truck trailer"],
    "mixer_truck": ["cement mixer truck", "transit mixer"],
}
if len(sys.argv) > 1 and sys.argv[1].strip():
    KINDS = json.loads(sys.argv[1])
for kind, queries in KINDS.items():
    seen, rows = set(), []
    for q in queries:
        url = "https://api.sketchfab.com/v3/search?" + urllib.parse.urlencode(
            {"type": "models", "q": q, "downloadable": "true", "count": 24, "sort_by": "-likeCount"})
        try:
            data = json.load(urllib.request.urlopen(url, timeout=30))
        except Exception as exc:
            print(kind, q, "failed", exc)
            continue
        for m in data.get("results", []):
            lic = (m.get("license") or {}).get("label", "?")
            if m["uid"] in seen or not any(k in lic for k in ("CC0", "Attribution")) or "NonCommercial" in lic \
                    or "NoDerivs" in lic:
                continue
            seen.add(m["uid"])
            rows.append((m.get("likeCount", 0), m["uid"], m["name"][:50], lic, m.get("faceCount"),
                         (m.get("user") or {}).get("username"), m.get("animationCount", 0)))
    rows.sort(reverse=True)
    print(f"\n== {kind}")
    for r in rows[:10]:
        print(f"  {r[0]:>5} likes  {r[1]}  {r[2]:<50} {r[3]:<22} faces {r[4]}  by {r[5]}  anims {r[6]}")
