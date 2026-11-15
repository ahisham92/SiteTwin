"""Prepare a Revit IFC for SiteTwin with its construction stages, as the planner set them.

Revit exports every element with its element id as the IFC ``Tag``. A stages file says which
stage each element belongs to (by that id), and what each stage is:

    {"setting": "marine",                       # optional: "marine" for a pier, quay or jetty
     "location": {"lat": 33.651153, "lon": 35.405074, "north": 2.09, "sea_level": 0.4},
     "stages": [{"code": "1", "name": "Demolition, only the piles stay", "how": "remove"}, ...],
     "elements": {"671811": "1", "670688": "E", "796686": "X", ...}}

``how`` is how the stage's elements come and go in the film: ``exist`` (there from the start
and kept), ``remove`` (there today, broken out one by one), ``drive`` (piles driven in),
``fill`` (rock and fill placed), ``cast`` (cast in place, one after another), ``lift`` (precast,
set by the crane), ``fit`` (fitted), ``repair`` (there from the start, repaired in its stage). An element's
stage may name its own way, as ``"7/cast"`` (stage 7, cast in place). An element whose stage is ``X`` is left out of the film
(drawing aids: water, sections' ground, design options, vessels).

The output IFC carries a property set ``SiteTwin_Sequence`` on every element (Stage, Stage
name, How, Order), the stages list on the site, the site's true latitude/longitude, and the
site moved to the model's own origin, turned to true north (``north``: degrees the model's
grid north lies west of true north, e.g. the grid convergence of a national projection), so the
model sits at its true place and bearing; ``sea_level`` is mean sea level in the model's levels
(e.g. +0.4 for levels above chart datum), so the sea meets the model where it really does. Writes ``.ifc`` or ``.ifc.gz``.

    python tools/stage_ifc.py model.ifc stages.json out.ifc.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import tempfile
from pathlib import Path

import ifcopenshell
import ifcopenshell.api as api
import ifcopenshell.guid


def dms(value: float) -> list[int]:
    sign = -1 if value < 0 else 1
    v = abs(value)
    d = int(v)
    m = int((v - d) * 60)
    s = int(((v - d) * 60 - m) * 60)
    micro = int(round(((((v - d) * 60 - m) * 60) - s) * 1e6))
    return [sign * d, sign * m, sign * s, sign * micro]


def stage(src: Path, stages_file: Path, out: Path) -> dict:
    spec = json.loads(Path(stages_file).read_text())
    f = ifcopenshell.open(str(src))
    stages = {s["code"]: s for s in spec["stages"]}
    order = {s["code"]: i for i, s in enumerate(spec["stages"])}
    by_tag = spec["elements"]
    report = {"kept": 0, "left_out": 0, "no_stage": 0, "by_stage": {}}

    # what is left out goes from the file altogether
    gone = []
    for el in f.by_type("IfcElement"):
        code = by_tag.get(str(getattr(el, "Tag", "") or ""))
        if code == "X" or (code is None and spec.get("unlisted", "keep") == "drop"):
            gone.append(el)
    for el in gone:
        api.run("root.remove_product", f, product=el)
    report["left_out"] = len(gone)

    oh = f.by_type("IfcOwnerHistory")
    oh = oh[0] if oh else None

    def label(v: str):
        return f.create_entity("IfcLabel", v)

    groups: dict[str, list] = {}
    for el in f.by_type("IfcElement"):
        if el.is_a("IfcOpeningElement") or el.is_a("IfcVirtualElement"):
            continue
        value = by_tag.get(str(getattr(el, "Tag", "") or ""))
        code, _, how = (value or "").partition("/")      # "7/cast": stage 7, but cast in place
        if code not in stages:
            report["no_stage"] += 1
            continue
        groups.setdefault((code, how or stages[code]["how"]), []).append(el)
    for (code, how), els in groups.items():
        s = stages[code]
        props = [f.create_entity("IfcPropertySingleValue", "Stage", None, label(code), None),
                 f.create_entity("IfcPropertySingleValue", "Stage name", None, label(s["name"]), None),
                 f.create_entity("IfcPropertySingleValue", "How", None, label(how), None),
                 f.create_entity("IfcPropertySingleValue", "Order", None,
                                 f.create_entity("IfcInteger", order[code]), None)]
        pset = f.create_entity("IfcPropertySet", ifcopenshell.guid.new(), oh, "SiteTwin_Sequence", None, props)
        f.create_entity("IfcRelDefinesByProperties", ifcopenshell.guid.new(), oh, None, None, els, pset)
        report["by_stage"][code] = report["by_stage"].get(code, 0) + len(els)
        report["kept"] += len(els)

    site = f.by_type("IfcSite")[0]
    loc = spec.get("location") or {}
    if "lat" in loc and "lon" in loc:
        site.RefLatitude = dms(float(loc["lat"]))
        site.RefLongitude = dms(float(loc["lon"]))
    # the site at the model's origin, turned so its north is true north
    place = site.ObjectPlacement.RelativePlacement
    ref = place.RefDirection.DirectionRatios if place.RefDirection else (1.0, 0.0, 0.0)
    turn = math.atan2(ref[1], ref[0]) + math.radians(float(loc.get("north", 0.0)))
    place.Location = f.create_entity("IfcCartesianPoint", (0.0, 0.0, 0.0))
    place.RefDirection = f.create_entity("IfcDirection", (math.cos(turn), math.sin(turn), 0.0))
    if place.Axis is None:
        place.Axis = f.create_entity("IfcDirection", (0.0, 0.0, 1.0))
    site_props = [f.create_entity("IfcPropertySingleValue", "Setting", None, label(spec.get("setting", "land")), None),
                  f.create_entity("IfcPropertySingleValue", "Sea level", None,
                                  f.create_entity("IfcLengthMeasure", float(loc.get("sea_level", 0.0))), None),
                  f.create_entity("IfcPropertySingleValue", "Stages", None,
                                  f.create_entity("IfcText", json.dumps(spec["stages"])), None)]
    pset = f.create_entity("IfcPropertySet", ifcopenshell.guid.new(), oh, "SiteTwin_Site", None, site_props)
    f.create_entity("IfcRelDefinesByProperties", ifcopenshell.guid.new(), oh, None, None, [site], pset)

    out = Path(out)
    if out.suffix == ".gz":
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "model.ifc"
            f.write(str(plain))
            with open(plain, "rb") as a, gzip.open(out, "wb", compresslevel=9) as b:
                shutil.copyfileobj(a, b)
    else:
        f.write(str(out))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ifc")
    p.add_argument("stages")
    p.add_argument("out")
    a = p.parse_args()
    print(json.dumps(stage(Path(a.ifc), Path(a.stages), Path(a.out)), indent=1))
