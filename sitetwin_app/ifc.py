"""Reading an IFC model without a compiled library.

An IFC file from Revit is STEP text: one entity per ``#id=TYPE(args);``. This
module reads just enough of it for SiteTwin's first steps, in plain Python:

* **where it is** — ``IfcSite``'s reference latitude, longitude and elevation;
* **what is in it** — how many of each kind of element;
* **how it stacks** — the storeys, by elevation, and which elements each holds;
* **a first draft of the sequence** — storey by storey, the elements in the
  order they are built (foundations, then the frame, then the floors, then the
  envelope and the roof).

The geometry itself is not read here. Turning the model into meshes for the
renderer is the job of IfcOpenShell in the rendering pipeline (see README).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class IfcError(ValueError):
    pass


@dataclass(frozen=True)
class Ref:
    id: int


@dataclass(frozen=True)
class Enum:
    value: str


@dataclass
class Entity:
    id: int
    type: str
    args: list[Any]


# The order things go up in, within a storey. Anything not named here is
# fitted out after the envelope.
PHASES: list[tuple[str, tuple[str, ...]]] = [
    ("Foundations", ("IFCPILE", "IFCFOOTING")),
    ("Vertical structure", ("IFCCOLUMN", "IFCWALL", "IFCWALLSTANDARDCASE")),
    ("Horizontal structure", ("IFCBEAM", "IFCMEMBER", "IFCPLATE")),
    ("Floors", ("IFCSLAB",)),
    ("Stairs and ramps", ("IFCSTAIR", "IFCSTAIRFLIGHT", "IFCRAMP", "IFCRAMPFLIGHT")),
    ("Roof", ("IFCROOF",)),
    ("Envelope", ("IFCCURTAINWALL", "IFCWINDOW", "IFCDOOR")),
    ("Railings and finishes", ("IFCRAILING", "IFCCOVERING")),
    ("Services", ("IFCFLOWSEGMENT", "IFCFLOWTERMINAL", "IFCFLOWFITTING", "IFCDUCTSEGMENT",
                  "IFCPIPESEGMENT", "IFCCABLECARRIERSEGMENT")),
]
PHASE_OF = {kind: name for name, kinds in PHASES for kind in kinds}
BUILT = set(PHASE_OF) | {"IFCBUILDINGELEMENTPROXY", "IFCFURNISHINGELEMENT"}


# ---- STEP text ---------------------------------------------------------------

_ENTITY = re.compile(r"#(\d+)\s*=\s*([A-Za-z0-9_]+)\s*\(", re.S)


def _statements(text: str):
    """Each ``#id=TYPE(...)`` in the DATA section, split on ';' outside strings."""
    start = text.upper().find("DATA;")
    if start < 0:
        raise IfcError("This is not an IFC file: it has no DATA section.")
    i, n, buf, quoted = start + 5, len(text), [], False
    while i < n:
        c = text[i]
        if quoted:
            buf.append(c)
            if c == "'":
                if i + 1 < n and text[i + 1] == "'":
                    buf.append("'")
                    i += 1
                else:
                    quoted = False
        elif c == "'":
            quoted = True
            buf.append(c)
        elif c == ";":
            statement = "".join(buf).strip()
            buf = []
            if statement.upper() == "ENDSEC":
                return
            if statement:
                yield statement
        else:
            buf.append(c)
        i += 1


def _decode(s: str) -> str:
    """IFC's \\X2\\...\\X0\\ escapes, which Revit uses for anything not ASCII."""
    def x2(m):
        hexes = m.group(1)
        return "".join(chr(int(hexes[k:k + 4], 16)) for k in range(0, len(hexes), 4))
    s = re.sub(r"\\X2\\([0-9A-Fa-f]+)\\X0\\", x2, s)
    s = re.sub(r"\\X\\([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), s)
    return s


def _args(s: str, i: int) -> tuple[list[Any], int]:
    """Parse a parenthesised list starting just after '('. Returns (values, index after ')')."""
    out: list[Any] = []
    n = len(s)
    while i < n:
        c = s[i]
        if c in " \t\r\n,":
            i += 1
        elif c == ")":
            return out, i + 1
        elif c == "(":
            inner, i = _args(s, i + 1)
            out.append(inner)
        elif c == "'":
            j, chars = i + 1, []
            while j < n:
                if s[j] == "'":
                    if j + 1 < n and s[j + 1] == "'":
                        chars.append("'")
                        j += 2
                        continue
                    break
                chars.append(s[j])
                j += 1
            out.append(_decode("".join(chars)))
            i = j + 1
        elif c == "#":
            j = i + 1
            while j < n and s[j].isdigit():
                j += 1
            out.append(Ref(int(s[i + 1:j])))
            i = j
        elif c == ".":
            j = s.index(".", i + 1)
            word = s[i + 1:j]
            out.append(True if word == "T" else False if word == "F" else Enum(word))
            i = j + 1
        elif c in "$*":
            out.append(None)
            i += 1
        elif c.isalpha():                       # a typed value: IFCLABEL('x')
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            inner, i = _args(s, s.index("(", j) + 1)
            out.append(inner[0] if len(inner) == 1 else inner)
        else:
            m = re.compile(r"[-+0-9.Ee]+").match(s, i)
            if not m:
                raise IfcError(f"Could not read the model near: {s[i:i + 40]!r}")
            word = m.group(0)
            out.append(float(word) if any(ch in word for ch in ".Ee") else int(word))
            i = m.end()
    raise IfcError("An entity in the model is not closed.")


def parse(text: str) -> dict[int, Entity]:
    entities: dict[int, Entity] = {}
    for statement in _statements(text):
        m = _ENTITY.match(statement)
        if not m:
            continue
        args, _ = _args(statement, m.end())
        entities[int(m.group(1))] = Entity(int(m.group(1)), m.group(2).upper(), args)
    if not entities:
        raise IfcError("The model has no entities in it.")
    return entities


# ---- what SiteTwin wants from it ---------------------------------------------

def _angle(compound: Any) -> float | None:
    """IfcCompoundPlaneAngleMeasure (deg, min, sec[, millionths]) as decimal degrees."""
    if not isinstance(compound, list) or not compound:
        return None
    parts = [float(p) for p in compound] + [0.0] * (4 - len(compound))
    deg, minutes, seconds, millionths = parts[:4]
    sign = -1.0 if any(p < 0 for p in parts) else 1.0
    value = abs(deg) + abs(minutes) / 60 + (abs(seconds) + abs(millionths) / 1e6) / 3600
    return round(sign * value, 7)


@dataclass
class Summary:
    schema: str = ""
    project: str = ""
    site: str = ""
    latitude: float | None = None
    longitude: float | None = None
    elevation: float | None = None
    counts: dict[str, int] = field(default_factory=dict)
    storeys: list[dict[str, Any]] = field(default_factory=list)
    sequence: list[dict[str, Any]] = field(default_factory=list)
    #: Each step's elements by their STEP id (#123), parts of an assembly with it. Kept out of
    #: as_dict: a big model has tens of thousands, and only the 3D preview wants them.
    step_ids: dict[int, list[int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = self.__dict__.copy()
        out.pop("step_ids", None)
        return out


def summarise(text: str) -> Summary:
    m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", text, re.I)
    entities = parse(text)
    out = Summary(schema=m.group(1) if m else "")

    by_type: dict[str, list[Entity]] = {}
    for e in entities.values():
        by_type.setdefault(e.type, []).append(e)

    if by_type.get("IFCPROJECT"):
        p = by_type["IFCPROJECT"][0].args
        out.project = (p[2] if len(p) > 2 and isinstance(p[2], str) else "") or ""
    if by_type.get("IFCSITE"):
        s = by_type["IFCSITE"][0].args
        out.site = (s[2] if len(s) > 2 and isinstance(s[2], str) else "") or ""
        if len(s) > 11:
            out.latitude, out.longitude = _angle(s[9]), _angle(s[10])
            out.elevation = float(s[11]) if isinstance(s[11], (int, float)) else None

    out.counts = dict(sorted(((t, len(v)) for t, v in by_type.items() if t in BUILT),
                             key=lambda kv: -kv[1]))

    storeys = by_type.get("IFCBUILDINGSTOREY", [])
    holds: dict[int, list[Entity]] = {s.id: [] for s in storeys}
    for rel in by_type.get("IFCRELCONTAINEDINSPATIALSTRUCTURE", []):
        where = rel.args[5] if len(rel.args) > 5 else None
        if isinstance(where, Ref) and where.id in holds and isinstance(rel.args[4], list):
            for r in rel.args[4]:
                e = entities.get(r.id) if isinstance(r, Ref) else None
                if e is not None and e.type in BUILT:
                    holds[where.id].append(e)

    def level(e: Entity) -> float:
        v = e.args[9] if len(e.args) > 9 else None
        return float(v) if isinstance(v, (int, float)) else 0.0

    # A stair, roof or curtain wall is an assembly: its flights, slabs and panels are its parts,
    # held by an IfcRelAggregates rather than by the storey. They go up with it.
    parts: dict[int, list[int]] = {}
    for rel in by_type.get("IFCRELAGGREGATES", []):
        whole = rel.args[4] if len(rel.args) > 4 else None
        if isinstance(whole, Ref) and isinstance(rel.args[5] if len(rel.args) > 5 else None, list):
            parts.setdefault(whole.id, []).extend(r.id for r in rel.args[5] if isinstance(r, Ref))

    def with_parts(eid: int) -> list[int]:
        out_ids, todo = [], [eid]
        while todo:
            i = todo.pop()
            out_ids.append(i)
            todo.extend(parts.get(i, []))
        return out_ids

    order = [name for name, _ in PHASES] + ["Other"]
    step = 0
    for storey in sorted(storeys, key=level):
        name = storey.args[2] if isinstance(storey.args[2], str) else f"#{storey.id}"
        inside = holds[storey.id]
        out.storeys.append({"name": name, "elevation": level(storey), "elements": len(inside)})
        phases: dict[str, list[Entity]] = {}
        for e in inside:
            phases.setdefault(PHASE_OF.get(e.type, "Other"), []).append(e)
        for phase in order:
            if phases.get(phase):
                step += 1
                out.sequence.append({"step": step, "storey": name, "phase": phase,
                                     "elements": len(phases[phase])})
                out.step_ids[step] = [i for e in phases[phase] for i in with_parts(e.id)]
    return out


def summarise_file(path: Path) -> Summary:
    return summarise(Path(path).read_text(encoding="utf-8", errors="replace"))
