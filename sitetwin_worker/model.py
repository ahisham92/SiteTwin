"""The IFC model as triangles, element by element, in the site's metres.

IfcOpenShell turns every product's representation into a mesh in world
coordinates (metres). Each element keeps its GUID, its IFC class, its storey
and its bounding box, which is what the sequence and the renderer need.
"""

from __future__ import annotations

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

    @property
    def lo(self) -> np.ndarray:
        return self.verts.min(axis=0)

    @property
    def hi(self) -> np.ndarray:
        return self.verts.max(axis=0)

    @property
    def centre(self) -> np.ndarray:
        return (self.lo + self.hi) / 2


def load(path: Path | str) -> list[Element]:
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element as util

    f = ifcopenshell.open(str(path))
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
            verts = np.asarray(g.verts, dtype=np.float32).reshape(-1, 3)
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
                out.append(Element(product.GlobalId, kind, product.Name or "", name, elev, verts, faces, colour))
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
