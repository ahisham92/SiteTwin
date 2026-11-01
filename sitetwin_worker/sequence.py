"""The build as a timeline: when each element appears, and what plant and crews are working.

The order follows the draft sequence the web app shows (storey by storey from
the bottom; within a storey foundations, then columns and walls, then beams,
slabs, stairs, roof, envelope, finishes, services). Within a step, elements go
in the order a site would place them: lowest first, then across the plan in
bands, so the frame visibly spreads across each floor.

The film has an opening shot of the cleared site, the build, and a closing shot
of the finished building. Each step's share of the build time grows with its
element count, but less than proportionally, so a step of ten big slabs is not
over in a blink beside a step of five hundred windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sitetwin_app.ifc import PHASE_OF, PHASES

from .model import Element

ORDER = [name for name, _ in PHASES] + ["Other"]

#: How an element arrives. "grow": cast in place, rises from its base.
#: "lift": precast or steel, lowered in by the crane. "fit": appears in place.
ARRIVAL = {
    "Foundations": "grow", "Vertical structure": "grow", "Floors": "grow",
    "Horizontal structure": "lift", "Stairs and ramps": "lift", "Roof": "lift",
    "Envelope": "lift", "Railings and finishes": "fit", "Services": "fit", "Other": "fit",
}

#: The plant each kind of work brings to site.
PLANT = {
    "Foundations": ["piling_rig", "excavator", "mixer_truck"],
    "Vertical structure": ["tower_crane", "mixer_truck", "concrete_pump"],
    "Horizontal structure": ["tower_crane", "flatbed_truck"],
    "Floors": ["tower_crane", "concrete_pump", "mixer_truck"],
    "Stairs and ramps": ["tower_crane"],
    "Roof": ["tower_crane", "flatbed_truck"],
    "Envelope": ["tower_crane", "mewp", "flatbed_truck"],
    "Railings and finishes": ["mewp"],
    "Services": ["mewp", "flatbed_truck"],
    "Other": [],
}

CREW = {"Foundations": 6, "Vertical structure": 10, "Horizontal structure": 8, "Floors": 12,
        "Stairs and ramps": 4, "Roof": 6, "Envelope": 8, "Railings and finishes": 4,
        "Services": 6, "Other": 3}


@dataclass
class Step:
    n: int
    storey: str
    phase: str
    start: int
    end: int
    guids: list[str] = field(default_factory=list)
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    top: float = 0.0
    plant: list[str] = field(default_factory=list)
    crew: int = 0


@dataclass
class Timeline:
    fps: int
    frames: int
    build_start: int
    build_end: int
    steps: list[Step]
    appear: dict[str, tuple[int, int, str]]     # guid -> (start, end, arrival)

    def as_dict(self) -> dict[str, Any]:
        return {"fps": self.fps, "frames": self.frames, "build_start": self.build_start,
                "build_end": self.build_end,
                "steps": [s.__dict__ for s in self.steps]}


def phase_of(e: Element) -> str:
    kind = e.kind
    if kind == "IFCSTAIRFLIGHT":
        return "Stairs and ramps"
    return PHASE_OF.get(kind, "Other")


def plan(elements: list[Element], seconds: float = 90.0, fps: int = 30,
         intro: float = 0.08, outro: float = 0.12) -> Timeline:
    frames = int(round(seconds * fps))
    build_start = int(frames * intro)
    build_end = int(frames * (1 - outro))

    # Storeys by elevation; elements with no storey go by their own height.
    groups: dict[tuple[float, str, int], list[Element]] = {}
    for e in elements:
        elev = e.storey_elevation if e.storey else float(e.lo[2])
        key = (round(elev, 3), e.storey or "Site", ORDER.index(phase_of(e)))
        groups.setdefault(key, []).append(e)
    keys = sorted(groups)

    weights = [len(groups[k]) ** 0.6 for k in keys]
    total = sum(weights) or 1.0
    span = build_end - build_start
    steps: list[Step] = []
    appear: dict[str, tuple[int, int, str]] = {}
    t = float(build_start)
    for n, (k, w) in enumerate(zip(keys, weights), start=1):
        length = span * w / total
        s0, s1 = int(round(t)), int(round(t + length))
        t += length
        els = groups[k]
        phase = ORDER[k[2]]
        arrival = ARRIVAL[phase]
        # lowest first, then in bands across the plan (a serpentine, as a gang works a floor)
        band = 8.0
        def order_key(e: Element) -> tuple[float, int, float]:
            c = e.centre
            row = int(math.floor(c[1] / band))
            return (round(float(e.lo[2]), 1), row, float(c[0]) if row % 2 == 0 else -float(c[0]))
        els = sorted(els, key=order_key)
        each = max(4, min(int(fps * 0.8), (s1 - s0) // 2 or 4))
        if len(els) > 1:
            gap = max(0.0, (s1 - s0 - each) / (len(els) - 1))
        else:
            gap = 0.0
        for i, e in enumerate(els):
            a = int(round(s0 + i * gap))
            appear[e.guid] = (a, min(a + each, s1 if s1 > a else a + each), arrival)
        lo = np.min([e.lo for e in els], axis=0)
        hi = np.max([e.hi for e in els], axis=0)
        c = (lo + hi) / 2
        steps.append(Step(n=n, storey=k[1], phase=phase, start=s0, end=s1,
                          guids=[e.guid for e in els],
                          centre=(float(c[0]), float(c[1]), float(c[2])), top=float(hi[2]),
                          plant=list(PLANT[phase]), crew=CREW[phase]))
    return Timeline(fps=fps, frames=frames, build_start=build_start, build_end=build_end,
                    steps=steps, appear=appear)


def plant_windows(tl: Timeline) -> dict[str, list[tuple[int, int]]]:
    """For each kind of plant, the frame ranges it is on site (adjacent steps merged).
    A tower crane, once up, stays until the last step that needs it."""
    out: dict[str, list[tuple[int, int]]] = {}
    for s in tl.steps:
        for p in s.plant:
            spans = out.setdefault(p, [])
            if spans and s.start - spans[-1][1] <= tl.fps * 2:
                spans[-1] = (spans[-1][0], s.end)
            else:
                spans.append((s.start, s.end))
    if "tower_crane" in out:
        spans = out["tower_crane"]
        out["tower_crane"] = [(spans[0][0], spans[-1][1])]
    return out
