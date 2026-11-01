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
CATALOG: dict[str, dict[str, Any]] = {
    "worker": {"uid": "9075b0deb11b40bca7f5ce8a8e7b63cb", "name": "Construction Worker",
               "author": "katelaruine", "licence": "CC BY 4.0", "yaw": 0.0, "height": 1.75},
    "mixer_truck": {"uid": "a8cd2a2cc2e243b59c168c4bb28e489b", "name": "Blue Cement Mixer Truck",
                    "author": "sandisbergvalds21", "licence": "CC BY 4.0", "yaw": 0.0, "length": 9.0},
    "concrete_pump": {"uid": "4328809da1c94ffb95a8eeb60bb6ce9b", "name": "Concrete Pump Truck (Boom Pump)",
                      "author": "doty_aecom", "licence": "CC BY 4.0", "yaw": 0.0, "length": 11.0},
    "piling_rig": {"uid": "997e8984809540e5b3e5ea693eb96983", "name": "Pile Driver",
                   "author": "doty_aecom", "licence": "CC BY 4.0", "yaw": 0.0, "height": 22.0},
    "excavator": {"uid": "3dde2c846cc54a4d9fabcc821c9b9caa", "name": "Hitachi excavator",
                  "author": "thesidekick", "licence": "CC BY 4.0", "yaw": 180.0, "length": 9.5},
    "flatbed_truck": {"uid": "59889032d0ad457c81d7e058c79eedf8", "name": "Scania truck",
                      "author": "PusztaiAndras", "licence": "CC BY 4.0", "yaw": 0.0, "length": 9.5},
    "mewp": {"uid": "4823e73dd2f94cbd9291b5d55df7f226", "name": "Electric Scissor Lift 1",
             "author": "cloudhub", "licence": "CC BY 4.0", "yaw": 0.0, "length": 2.5},
}

Get = Callable[[str, dict[str, str]], bytes]


def _get(url: str, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "SiteTwin", **headers})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


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
