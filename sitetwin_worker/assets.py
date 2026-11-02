"""Real 3D models of the plant and the crews, from Sketchfab, under free licences.

With ``SITETWIN_SKETCHFAB_TOKEN`` set (a Sketchfab account's API token, from Settings →
Password & API), each model in ``CATALOG`` is downloaded once into the cache folder and
then used by equipment.py in place of the built-in shapes. Every model here is Creative
Commons Attribution (or CC0) and downloadable; the film's credits name each one.

Without the token, or if a download fails, the built-in shapes are used, so a film is
never held up by a model.

Each entry says how to stand the model up the way the scene wants it: base on the ground,
centred, facing +X. ``yaw`` is the turn (degrees about the vertical) that makes the
model's front face +X once imported; ``height`` or ``length`` (metres, the longer of its two
plan sides) is the real size it is scaled to.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

API = "https://api.sketchfab.com/v3/models/{uid}/download"

#: kind -> the model used for it. Picked by hand from tools/sketchfab_search.py.
#: Workers are posed, textured full-body scans; equipment.worker() dresses them in a hi-vis
#: vest and a hard hat. The props dress the site round the cabins, the gate and the laydown.
CATALOG: dict[str, dict[str, Any]] = {
    "worker": {"uid": "e211ffd52a96416e89c0cc415a388d61", "name": "Dennis Posed 004",
               "author": "renderpeople", "licence": "CC BY 4.0", "yaw": 0.0, "height": 1.78},
    "worker_b": {"uid": "ae9559912f6240a8a87057911b69edd0", "name": "Full Body Scan - Clean Pose",
                 "author": "OverseerTheVirtual", "licence": "CC BY 4.0", "yaw": 0.0, "height": 1.74},
    "prop_bricks": {"uid": "aeab2b4fd7fb424cb7e68c692048e4f9", "name": "Pallet of Bricks",
                    "author": "thewalkdead", "licence": "CC BY 4.0", "yaw": 0.0, "height": 1.0},
    "prop_pallet": {"uid": "ee1dc525a33d499bbad75c4e110050cd", "name": "Construction Pallet - Photoscan",
                    "author": "GameDevNick", "licence": "CC BY 4.0", "yaw": 0.0, "length": 1.2},
    "prop_paving": {"uid": "0bfb0a34680f489ebcf1f605ed49cf69", "name": "Pile of paving stones on a wooden pallet",
                    "author": "orphanrtg", "licence": "CC BY 4.0", "yaw": 0.0, "length": 1.2},
    "prop_barrier": {"uid": "b52246c9611a42a99a03b425535a0237", "name": "Jersey Barrier",
                     "author": "emran.bayati", "licence": "CC BY 4.0", "yaw": 0.0, "length": 3.0},
    "prop_cones": {"uid": "e03b1d2dfdc0416bb8324d15573ea60c", "name": "Traffic Cones",
                   "author": "notanotherartist", "licence": "CC BY 4.0", "yaw": 0.0, "height": 0.75},
    "prop_skip": {"uid": "f3cf49ea7ad54573a6f5526de75a13d6", "name": "Skip Bin (Scaniverse)",
                  "author": "db4", "licence": "CC BY 4.0", "yaw": 0.0, "length": 3.8},
    "prop_rebar": {"uid": "e9cda4a295a34f25a563e583ffaa34fe", "name": "Construction Steel Rebar (Scaniverse)",
                   "author": "db4", "licence": "CC BY 4.0", "yaw": 0.0, "length": 6.0},
    "prop_light": {"uid": "9addea6a4129438b83a1fc3ed4a784ec", "name": "Light Tower",
                   "author": "krambulini", "licence": "CC BY 4.0", "yaw": 0.0, "length": 3.6},
    "prop_toilet": {"uid": "480c542d5b0240fca1d9ed303800d2bb", "name": "Port-a-john",
                    "author": "doty_aecom", "licence": "CC BY 4.0", "yaw": 0.0, "height": 2.3},
    "prop_ibc": {"uid": "41cb527dd5b340878799559d5ed3abd7", "name": "IBC Water Tank",
                 "author": "Mystyc", "licence": "CC BY 4.0", "yaw": 0.0, "height": 1.16},
    "prop_cars": {"uid": "20f9af9b8a404d5cb022ac6fe87f21f5", "name": "Generic passenger car pack",
                  "author": "comrade1280", "licence": "CC BY 4.0", "yaw": 0.0, "length": 4.5, "pack": True},
    "prop_tree": {"uid": "d841c3bcc5324daebee50f45619e05fc", "name": "oak trees",
                  "author": "bumstrum", "licence": "CC BY 4.0", "yaw": 0.0, "height": 8.0, "pack": True},
    "prop_palm": {"uid": "11acf710e6c149daa8d6fb8cdc5d087f", "name": "Date Palm",
                  "author": "evolveduk", "licence": "CC BY 4.0", "yaw": 0.0, "height": 9.0},
    "mixer_truck": {"uid": "8d1d88a5955f4e7da13562bb7bec146f", "name": "Concrete Mixer Truck",
                    "author": "doty_aecom", "licence": "CC BY 4.0", "yaw": 0.0, "length": 9.0},
    "concrete_pump": {"uid": "4328809da1c94ffb95a8eeb60bb6ce9b", "name": "Concrete Pump Truck (Boom Pump)",
                      "author": "doty_aecom", "licence": "CC BY 4.0", "yaw": 0.0, "length": 11.0},
    "piling_rig": {"uid": "997e8984809540e5b3e5ea693eb96983", "name": "Pile Driver",
                   "author": "doty_aecom", "licence": "CC BY 4.0", "yaw": 0.0, "height": 22.0},
    "excavator": {"uid": "3dde2c846cc54a4d9fabcc821c9b9caa", "name": "Hitachi excavator",
                  "author": "thesidekick", "licence": "CC BY 4.0", "yaw": 180.0, "length": 9.5},
    "flatbed_truck": {"uid": "8ee4c1329219413e879760a25208ee15", "name": "Mercedes Actros com Semi Reboque Prancha",
                      "author": "LuizZed36", "licence": "CC BY 4.0", "yaw": 180.0, "length": 16.5},
    "mewp": {"uid": "4823e73dd2f94cbd9291b5d55df7f226", "name": "Electric Scissor Lift 1",
             "author": "cloudhub", "licence": "CC BY 4.0", "yaw": 0.0, "length": 2.5},
}

Get = Callable[[str, dict[str, str]], bytes]


def _get(url: str, headers: dict[str, str]) -> bytes:
    """GET, waiting and trying again when the server says too many requests (Sketchfab does)."""
    import time
    import urllib.error
    for wait in (5, 15, 30, 60, None):
        req = urllib.request.Request(url, headers={"User-Agent": "SiteTwin", **headers})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or wait is None:
                raise
            time.sleep(int(exc.headers.get("Retry-After") or wait))
    raise RuntimeError("unreachable")


def fetch_one(kind: str, entry: dict[str, Any], token: str, cache: Path, get: Get = _get) -> Path:
    """The model's file in the cache (a .glb, or a .gltf with its files), downloading it once."""
    folder = cache / entry["uid"]
    done = folder / "model.json"
    if done.exists():
        return folder / json.loads(done.read_text())["file"]
    links = json.loads(get(API.format(uid=entry["uid"]), {"Authorization": f"Token {token}"}))
    tmp = cache / (entry["uid"] + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if links.get("glb", {}).get("url"):
        (tmp / "model.glb").write_bytes(get(links["glb"]["url"], {}))
        name = "model.glb"
    else:
        with zipfile.ZipFile(io.BytesIO(get(links["gltf"]["url"], {}))) as z:
            for info in z.infolist():                      # no paths out of the folder
                target = (tmp / info.filename).resolve()
                if not str(target).startswith(str(tmp.resolve()) + os.sep) and target != tmp.resolve():
                    raise ValueError("bad path in the model's zip")
            z.extractall(tmp)
        found = sorted(tmp.rglob("*.gltf")) or sorted(tmp.rglob("*.glb"))
        if not found:
            raise ValueError("no glTF in the download")
        name = str(found[0].relative_to(tmp))
    (tmp / "model.json").write_text(json.dumps({"file": name, "kind": kind}))
    shutil.rmtree(folder, ignore_errors=True)
    tmp.rename(folder)
    return folder / name


#: Ground and concrete textures (CC0, Poly Haven, no token needed): role -> asset name.
TEXTURES = {"gravel": "gravelly_sand", "soil": "brown_mud_dry",
            "asphalt": "asphalt_02", "formwork": "weathered_brown_planks"}
POLYHAVEN = "https://api.polyhaven.com/files/{name}"


def fetch_textures(cache: Path, get: Get = _get, log=print) -> dict[str, dict[str, str]]:
    """{role: {"diff": path, "nor": path, "rough": path}} for every texture that could be had."""
    out: dict[str, dict[str, str]] = {}
    for role, name in TEXTURES.items():
        folder = Path(cache) / "textures" / name
        try:
            maps = {"diff": "Diffuse", "nor": "nor_gl", "rough": "Rough"}
            have = {k: folder / f"{k}.jpg" for k in maps}
            if not all(p.exists() for p in have.values()):
                files = json.loads(get(POLYHAVEN.format(name=name), {}))
                folder.mkdir(parents=True, exist_ok=True)
                for k, key in maps.items():
                    url = files[key]["2k"]["jpg"]["url"]
                    have[k].write_bytes(get(url, {}))
            out[role] = {k: str(p) for k, p in have.items()}
        except Exception as exc:                          # noqa: BLE001 - the procedural material stands in
            log(f"  texture for {role} not used: {type(exc).__name__}: {str(exc)[:120]}")
    return out


def prepare(cache: Path, token: str | None = None, get: Get = _get, log=print) -> tuple[dict[str, str], list[str]]:
    """Every catalogue model that could be had: {kind: file path}, and their credit lines."""
    token = (token if token is not None else os.environ.get("SITETWIN_SKETCHFAB_TOKEN", "")).strip()
    if not token:
        return {}, []
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    files, credits = {}, []
    for kind, entry in CATALOG.items():
        try:
            files[kind] = str(fetch_one(kind, entry, token, cache, get))
            credits.append(f"{entry['name']} by {entry['author']} ({entry['licence']})")
        except Exception as exc:                          # noqa: BLE001 - the built-in shape stands in
            log(f"  model for {kind} not used: {type(exc).__name__}: {str(exc).replace(token, '…')[:160]}")
    return files, credits
