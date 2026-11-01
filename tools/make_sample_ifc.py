"""Write a sample IFC4 building with real geometry, for trying SiteTwin without a Revit export.

A four-storey concrete frame on pad footings and piles: a 5×4 grid of columns at
7.5 m, beams both ways, slabs, a stair core, glazed facades, a parapet and a
roof. Located at the latitude/longitude given.

    python tools/make_sample_ifc.py sample.ifc --lat 33.6979 --lon 35.4178
"""

from __future__ import annotations

import argparse

import ifcopenshell
import ifcopenshell.api as api
import numpy as np


def dms(value: float) -> list[int]:
    sign = -1 if value < 0 else 1
    v = abs(value)
    d = int(v); m = int((v - d) * 60); s = int(((v - d) * 60 - m) * 60)
    micro = int(round(((((v - d) * 60 - m) * 60) - s) * 1e6))
    return [sign * d, sign * m, sign * s, sign * micro]


def build(path: str, lat: float, lon: float, storeys: int = 4, nx: int = 5, ny: int = 4,
          bay: float = 7.5, height: float = 4.0) -> None:
    f = api.run("project.create_file", version="IFC4")
    project = api.run("root.create_entity", f, ifc_class="IfcProject", name="SiteTwin sample building")
    api.run("unit.assign_unit", f)
    ctx = api.run("context.add_context", f, context_type="Model")
    body = api.run("context.add_context", f, context_type="Model", context_identifier="Body",
                   target_view="MODEL_VIEW", parent=ctx)
    site = api.run("root.create_entity", f, ifc_class="IfcSite", name="Sample site")
    site.RefLatitude, site.RefLongitude, site.RefElevation = dms(lat), dms(lon), 0.0
    building = api.run("root.create_entity", f, ifc_class="IfcBuilding", name="Sample building")
    api.run("aggregate.assign_object", f, relating_object=project, products=[site])
    api.run("aggregate.assign_object", f, relating_object=site, products=[building])

    def place(product, x, y, z, rot=0.0):
        m = np.eye(4)
        c, s = np.cos(rot), np.sin(rot)
        m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        m[:3, 3] = [x, y, z]
        api.run("geometry.edit_object_placement", f, product=product, matrix=m)

    def box(cls, name, storey, x, y, z, lx, ly, lz, rot=0.0, predefined=None):
        el = api.run("root.create_entity", f, ifc_class=cls, name=name, predefined_type=predefined)
        rep = api.run("geometry.add_wall_representation", f, context=body, length=lx, height=lz, thickness=ly)
        api.run("geometry.assign_representation", f, product=el, representation=rep)
        place(el, x, y, z, rot)
        api.run("spatial.assign_container", f, relating_structure=storey, products=[el])
        return el

    W, D = (nx - 1) * bay, (ny - 1) * bay
    found = api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Foundation")
    found.Elevation = -1.5
    levels = [found]
    for i in range(storeys):
        s = api.run("root.create_entity", f, ifc_class="IfcBuildingStorey",
                    name="Ground Floor" if i == 0 else f"Level {i}")
        s.Elevation = i * height
        levels.append(s)
    roof_level = api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Roof")
    roof_level.Elevation = storeys * height
    levels.append(roof_level)
    api.run("aggregate.assign_object", f, relating_object=building, products=levels)

    for i in range(nx):
        for j in range(ny):
            x, y = i * bay, j * bay
            for k, (dx, dy) in enumerate(((-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6))):
                box("IfcPile", f"P{i}{j}{k}", found, x + dx - 0.3, y + dy - 0.3, -14.0, 0.6, 0.6, 12.5)
            box("IfcFooting", f"F{i}{j}", found, x - 1.2, y - 1.2, -1.5, 2.4, 2.4, 1.0, predefined="PAD_FOOTING")

    for lv in range(storeys):
        st = levels[lv + 1]
        z = lv * height
        for i in range(nx):
            for j in range(ny):
                box("IfcColumn", f"C{lv}{i}{j}", st, i * bay - 0.25, j * bay - 0.25, z - (0.5 if lv == 0 else 0),
                    0.5, 0.5, height - 0.25 + (0.5 if lv == 0 else 0))
        for j in range(ny):
            for i in range(nx - 1):
                box("IfcBeam", f"BX{lv}{i}{j}", st, i * bay, j * bay - 0.2, z + height - 0.85, bay, 0.4, 0.6)
        for i in range(nx):
            for j in range(ny - 1):
                box("IfcBeam", f"BY{lv}{i}{j}", st, i * bay + 0.2, j * bay, z + height - 0.85, bay, 0.4, 0.6,
                    rot=np.pi / 2)
        # floor slab of the level above, in bays
        for i in range(nx - 1):
            for j in range(ny - 1):
                box("IfcSlab", f"S{lv}{i}{j}", st, i * bay, j * bay, z + height - 0.25, bay, bay, 0.25,
                    predefined="FLOOR")
        # stair core walls
        cx, cy = bay * 1, bay * 1
        box("IfcWall", f"WC{lv}a", st, cx + 1.0, cy + 1.0, z, 5.5, 0.25, height - 0.25)
        box("IfcWall", f"WC{lv}b", st, cx + 1.0, cy + 6.25, z, 5.5, 0.25, height - 0.25)
        box("IfcStairFlight", f"ST{lv}", st, cx + 1.5, cy + 2.0, z, 4.0, 1.4, height - 0.4)
        # facade: glazing panels between columns, spandrel below
        for side in range(4):
            length, along = (W, (1, 0)) if side % 2 == 0 else (D, (0, 1))
            n = int(round(length / bay))
            for b in range(n):
                if side == 0:
                    x0, y0, rot = b * bay + 0.3, -0.45, 0.0
                elif side == 1:
                    x0, y0, rot = W + 0.45, b * bay + 0.3, np.pi / 2
                elif side == 2:
                    x0, y0, rot = W - b * bay - 0.3, D + 0.45, np.pi
                else:
                    x0, y0, rot = -0.45, D - b * bay - 0.3, -np.pi / 2
                box("IfcWall", f"SP{lv}{side}{b}", st, x0, y0, z, bay - 0.6, 0.2, 0.9, rot=rot)
                box("IfcWindow", f"GL{lv}{side}{b}", st, x0, y0, z + 0.9, bay - 0.6, 0.06, height - 1.2, rot=rot)
    rz = storeys * height
    for side, (x0, y0, length, rot) in enumerate(((0, -0.45, W, 0), (W + 0.45, 0, D, np.pi / 2),
                                                  (W, D + 0.45, W, np.pi), (-0.45, D, D, -np.pi / 2))):
        box("IfcRailing", f"PAR{side}", roof_level, x0, y0, rz, length, 0.2, 1.1, rot=rot)
    box("IfcRoof", "Roof plant deck", roof_level, bay, bay, rz, 2 * bay, 1.5 * bay, 2.5)
    f.write(path)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("out")
    a.add_argument("--lat", type=float, default=33.6979167)
    a.add_argument("--lon", type=float, default=35.4177778)
    a.add_argument("--storeys", type=int, default=4)
    args = a.parse_args()
    build(args.out, args.lat, args.lon, storeys=args.storeys)
