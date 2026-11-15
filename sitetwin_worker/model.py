"""The IFC model as triangles, element by element, in the site's metres.

IfcOpenShell turns every product's representation into a mesh in world
coordinates (metres). Each element keeps its GUID, its IFC class, its storey
and its bounding box, which is what the sequence and the renderer need.
"""

from __future__ import annotations

import math
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SKIP = {"IFCSPACE", "IFCOPENINGELEMENT", "IFCSITE", "IFCBUILDING", "IFCBUILDINGSTOREY",
        "IFCANNOTATION", "IFCGRID", "IFCVIRTUALELEMENT", "IFCOPENINGSTANDARDCASE"}


@dataclass
class Element:
    guid: str
    kind: str              # upper-case IFC class, e.g. IFCCOLUMN
    name: str
    storey: str
    storey_elevation: float
    verts: np.ndarray      # (n, 3) float32
    faces: np.ndarray      # (m, 3) int32
    colour: tuple[float, float, float] | None = None
    #: the planner's construction stage, when the model carries one (property set
    #: SiteTwin_Sequence, see tools/stage_ifc.py): its code, name, place in the order and how
    #: its elements come and go (exist, remove, repair, drive, fill, cast, lift, fit)
    stage: str = ""
    stage_name: str = ""
    stage_order: int = -1
    how: str = ""
    tag: str = ""

    @property
    def lo(self) -> np.ndarray:
        return self.verts.min(axis=0)

    @property
    def hi(self) -> np.ndarray:
        return self.verts.max(axis=0)

    @property
    def centre(self) -> np.ndarray:
        return (self.lo + self.hi) / 2


def open_ifc(path: Path | str):
    """The IFC file, plain or gzipped (``.ifc.gz``)."""
    import ifcopenshell
    path = Path(path)
    with open(path, "rb") as fh:
        zipped = fh.read(2) == b"\x1f\x8b"            # gzip, whatever the file is called
    if zipped:
        import gzip
        return ifcopenshell.file.from_string(gzip.open(path, "rt", encoding="utf-8", errors="replace").read())
    return ifcopenshell.open(str(path))


def site_info(f) -> dict[str, Any]:
    """The SiteTwin_Site property set on the site (setting, stages), if there is one."""
    import ifcopenshell.util.element as util
    sites = f.by_type("IfcSite")
    if not sites:
        return {}
    ps = util.get_psets(sites[0]).get("SiteTwin_Site", {})
    out: dict[str, Any] = {"setting": ps.get("Setting", ""), "sea_level": float(ps.get("Sea level") or 0.0)}
    if ps.get("Stages"):
        import json
        try:
            out["stages"] = json.loads(ps["Stages"])
        except ValueError:
            pass
    return out


def _site_offset(f) -> np.ndarray:
    """A site placed kilometres from the origin (a Revit export in a national grid's shared
    coordinates) is brought back to it: the model is drawn round the site's own point."""
    try:
        import ifcopenshell.util.placement as up
        import ifcopenshell.util.unit as uu
        site = f.by_type("IfcSite")[0]
        m = up.get_local_placement(site.ObjectPlacement)
        t = np.array(m[:3, 3], dtype=float) * uu.calculate_unit_scale(f)
    except Exception:                                    # noqa: BLE001 - no site, no offset
        return np.zeros(3, dtype=np.float32)
    t[2] = 0.0
    return t.astype(np.float32) if math.hypot(t[0], t[1]) > 2000.0 else np.zeros(3, dtype=np.float32)


def load(path: Path | str) -> list[Element]:
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element as util

    f = open_ifc(path)
    offset = _site_offset(f)
    staged: dict[int, tuple[str, str, int, str]] = {}
    for rel in f.by_type("IfcRelDefinesByProperties"):
        ps = rel.RelatingPropertyDefinition
        if ps.is_a("IfcPropertySet") and ps.Name == "SiteTwin_Sequence":
            v = {p.Name: p.NominalValue.wrappedValue for p in ps.HasProperties if p.is_a("IfcPropertySingleValue") and p.NominalValue}
            for o in rel.RelatedObjects:
                staged[o.id()] = (str(v.get("Stage", "")), str(v.get("Stage name", "")),
                                  int(v.get("Order", -1)), str(v.get("How", "")))
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    storeys: dict[int, tuple[str, float]] = {}
    for s in f.by_type("IfcBuildingStorey"):
        storeys[s.id()] = (s.Name or f"#{s.id()}", float(s.Elevation or 0.0))

    out: list[Element] = []
    it = ifcopenshell.geom.iterator(settings, f, max(1, multiprocessing.cpu_count() - 1))
    if not it.initialize():
        return out
    while True:
        shape = it.get()
        product = f.by_id(shape.id)
        kind = product.is_a().upper()
        if kind not in SKIP:
            g = shape.geometry
            verts = np.asarray(g.verts, dtype=np.float32).reshape(-1, 3) - offset
            faces = np.asarray(g.faces, dtype=np.int32).reshape(-1, 3)
            if len(faces):
                container = util.get_container(product)
                # an element inside an aggregate (a stair flight in a stair) takes its parent's storey
                if container is None and util.get_aggregate(product) is not None:
                    container = util.get_container(util.get_aggregate(product))
                name, elev = storeys.get(container.id(), ("", 0.0)) if container is not None else ("", 0.0)
                colour = None
                mats = list(g.materials)
                if mats and mats[0].diffuse is not None:
                    d = mats[0].diffuse
                    try:
                        colour = (float(d.r()), float(d.g()), float(d.b()))
                    except (AttributeError, TypeError):
                        colour = tuple(float(c) for c in list(d)[:3])  # type: ignore[assignment]
                st = staged.get(product.id(), ("", "", -1, ""))
                out.append(Element(product.GlobalId, kind, product.Name or "", name, elev, verts, faces, colour,
                                   stage=st[0], stage_name=st[1], stage_order=st[2], how=st[3],
                                   tag=str(getattr(product, "Tag", "") or "")))
        if not it.next():
            break
    return out


def footprint(elements: list[Element], margin: float = 0.0) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) of everything in plan, plus a margin."""
    lo = np.min([e.lo for e in elements], axis=0)
    hi = np.max([e.hi for e in elements], axis=0)
    return float(lo[0] - margin), float(lo[1] - margin), float(hi[0] + margin), float(hi[1] + margin)


def summary(elements: list[Element]) -> dict[str, Any]:
    lo = np.min([e.lo for e in elements], axis=0)
    hi = np.max([e.hi for e in elements], axis=0)
    return {"elements": len(elements), "min": lo.round(2).tolist(), "max": hi.round(2).tolist()}
