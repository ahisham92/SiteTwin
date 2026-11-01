"""How the site moves, worked out before anything is drawn: what the crane lifts and when, where
trucks drive, and where people may stand. Plain Python so it can be tested without Blender.

The rules a real site keeps, and the film keeps too:

* **One lift at a time.** The crane picks a load at the laydown area, hoists it, slews and
  trolleys it over its place, lowers it in and lets go, then goes back for the next. A load is
  only ever seen on the hook or in its place, never floating on its own. Where a step has more
  pieces than the crane has time to lift one by one in the film, the crane lifts as many as fit
  and the rest are fixed in place by the crew (they appear where they belong, never in the air).
* **Trucks keep to the lane.** They drive in along the access road, through the gate, to the
  delivery bay, stand there to unload or pour, and reverse out the way they came. One truck in
  the lane at a time. Wheels stay on the ground.
* **People keep clear.** Nobody stands in the truck lane, and nobody stands where a load is
  being lowered. Crews stand on the floor they are building, or on the ground by the laydown.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable

Point = tuple[float, float]

#: The shortest crane cycle the film shows, in seconds of film.
CYCLE = 2.0
#: Fractions of a cycle: go back to the laydown, lower to the load, sling it, hoist, slew, lower in, let go.
PHASES = (("return", 0.18), ("down", 0.08), ("sling", 0.08), ("hoist", 0.14), ("slew", 0.28),
          ("lower", 0.16), ("release", 0.08))


def smooth(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


@dataclass
class Lift:
    guid: str
    start: int                 # frame the crane leaves for the laydown
    end: int                   # frame it lets go
    pick: tuple[float, float, float]
    place: tuple[float, float, float]      # where the element's base goes
    height: float              # the element's own height (its top is place z + height)

    def times(self) -> dict[str, tuple[float, float]]:
        out, t = {}, float(self.start)
        span = self.end - self.start
        for name, share in PHASES:
            out[name] = (t, t + span * share)
            t += span * share
        return out

    @property
    def attached(self) -> int:
        """The frame the load leaves the ground (it is seen from here on)."""
        return int(math.floor(self.times()["hoist"][0]))


@dataclass
class Crane:
    x: float
    y: float
    height: float              # of the jib
    jib: float
    lifts: list[Lift] = field(default_factory=list)

    def polar(self, x: float, y: float) -> tuple[float, float]:
        r = min(self.jib - 1.5, max(4.0, math.hypot(x - self.x, y - self.y)))
        return math.atan2(y - self.y, x - self.x), r

    def travel(self, top: float) -> float:
        """The height loads travel at: well over everything built so far."""
        return min(self.height - 3.0, top + 6.0)

    def pose(self, frame: float, top: float) -> tuple[float, float, float, Lift | None, float]:
        """(slew angle, trolley radius, hook tip height, lift on the hook, load base height)
        at a frame. The hook tip is where the slings meet, 1.5 m over the load's top."""
        rest = self.lifts[0] if self.lifts else None
        before = None
        for lift in self.lifts:
            if frame < lift.start:
                rest = lift
                break
            before = lift
            rest = None
        if before is None and rest is not None:            # before the first lift: waiting at the laydown
            a, r = self.polar(rest.pick[0], rest.pick[1])
            return a, r, self.travel(top), None, 0.0
        if before is None:
            return 0.0, self.jib * 0.5, self.height - 4, None, 0.0
        lift = before
        if frame > lift.end:                                # after a lift, before the next: hold
            a, r = self.polar(lift.place[0], lift.place[1])
            return a, r, self.travel(top), None, 0.0
        t = lift.times()
        prev = self.lifts[self.lifts.index(lift) - 1] if self.lifts.index(lift) > 0 else None
        a0, r0 = self.polar(*(prev.place[:2] if prev else lift.pick[:2]))
        ap, rp = self.polar(lift.pick[0], lift.pick[1])
        at, rt = self.polar(lift.place[0], lift.place[1])
        ap = unwrap(a0, ap)
        at = unwrap(ap, at)
        high = self.travel(top)
        low_pick = lift.pick[2] + lift.height + 1.5
        low_place = lift.place[2] + lift.height + 1.5

        def k(name: str) -> float:
            s, e = t[name]
            return smooth((frame - s) / max(1e-6, e - s))

        if frame < t["down"][0]:
            u = k("return")
            return a0 + (ap - a0) * u, r0 + (rp - r0) * u, high, None, 0.0
        if frame < t["sling"][0]:
            u = k("down")
            return ap, rp, high + (low_pick - high) * u, None, 0.0
        if frame < t["hoist"][0]:
            return ap, rp, low_pick, None, 0.0
        if frame < t["slew"][0]:
            u = k("hoist")
            z = low_pick + (high - low_pick) * u
            return ap, rp, z, lift, z - lift.height - 1.5
        if frame < t["lower"][0]:
            u = k("slew")
            return ap + (at - ap) * u, rp + (rt - rp) * u, high, lift, high - lift.height - 1.5
        if frame < t["release"][0]:
            u = k("lower")
            z = high + (low_place - high) * u
            return at, rt, z, lift, z - lift.height - 1.5
        u = k("release")
        return at, rt, low_place + 2.0 * u, None, 0.0

    def hook_xy(self, angle: float, radius: float) -> Point:
        return self.x + radius * math.cos(angle), self.y + radius * math.sin(angle)


def unwrap(prev: float, ang: float) -> float:
    while ang - prev > math.pi:
        ang -= 2 * math.pi
    while ang - prev < -math.pi:
        ang += 2 * math.pi
    return ang


def plan_lifts(steps: Iterable, appear: dict[str, tuple[int, int, str]], places: dict[str, tuple],
               pick: tuple[float, float, float], fps: int, window: tuple[int, int]) -> list[Lift]:
    """One lift at a time through the steps whose pieces are lifted in. Each lifted piece's
    appear entry becomes (frame it leaves the ground, frame it is set down, "crane"); a piece the
    crane has no time for becomes "fit" (fixed in place by the crew). ``places`` maps a guid to
    (base x, base y, base z, height)."""
    lifts: list[Lift] = []
    cycle = int(round(CYCLE * fps))
    s_win, e_win = window
    free_from = s_win
    for st in steps:
        lifted = [g for g in st.guids if appear.get(g, (0, 0, ""))[2] == "lift"]
        if not lifted:
            continue
        start = max(st.start, free_from)
        room = max(0, min(st.end, e_win) - start)
        n = min(len(lifted), max(1 if room >= 0.7 * cycle else 0, room // cycle))
        if n <= 0:
            for g in lifted:
                a, b, _ = appear[g]
                appear[g] = (a, b, "fit")
            continue
        chosen = set(lifted[round(i * (len(lifted) - 1) / max(1, n - 1))] for i in range(n)) if n > 1 else {lifted[0]}
        span = room / len(chosen)
        f = float(start)
        for g in lifted:
            if g not in chosen:
                a, b, _ = appear[g]
                appear[g] = (a, b, "fit")
                continue
            x, y, z, h = places[g]
            lift = Lift(g, int(f), int(f + min(span, cycle * 1.6)) - 1, pick, (x, y, z), h)
            lifts.append(lift)
            appear[g] = (lift.attached, lift.end, "crane")
            f += span
        free_from = int(f)
    return lifts


# ---- trucks ----------------------------------------------------------------------------------

@dataclass
class Lane:
    """The trucks' way in: from the far end of the access road to the delivery bay, straight
    through the gate, ``width`` metres wide."""
    far: Point
    bay: Point
    width: float = 5.0

    @property
    def heading_in(self) -> float:
        return math.atan2(self.bay[1] - self.far[1], self.bay[0] - self.far[0])

    def at(self, s: float) -> Point:
        """The point s metres in from the far end."""
        length = math.dist(self.far, self.bay)
        u = min(1.0, max(0.0, s / length))
        return self.far[0] + (self.bay[0] - self.far[0]) * u, self.far[1] + (self.bay[1] - self.far[1]) * u

    @property
    def length(self) -> float:
        return math.dist(self.far, self.bay)

    def keeps_clear(self, p: Point, margin: float = 1.5, past_bay: float = 4.0) -> bool:
        """True if a person at p is out of the lane (and the space just beyond the bay)."""
        dx, dy = self.bay[0] - self.far[0], self.bay[1] - self.far[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        along = (p[0] - self.far[0]) * ux + (p[1] - self.far[1]) * uy
        across = abs(-(p[0] - self.far[0]) * uy + (p[1] - self.far[1]) * ux)
        return not (-margin <= along <= length + past_bay and across <= self.width / 2 + margin)


@dataclass
class Visit:
    kind: str
    arrive: int                # frame it enters at the far end
    parked: int                # frame it stops at the bay
    leave: int                 # frame it starts reversing out
    gone: int                  # frame it is back at the far end


def plan_visits(windows: dict[str, list[tuple[int, int]]], lane: Lane, fps: int,
                speed: float = 14.0, stand: dict[str, float] | None = None) -> list[Visit]:
    """Truck visits through the one lane, never two in it at once, taking turns between the
    kinds wanted at the time. ``windows`` gives when each kind is wanted (from the sequence)."""
    stand = stand or {"mixer_truck": 3.5, "flatbed_truck": 3.0}
    drive = int(round(lane.length / speed * fps))
    spans = [(s, e, kind) for kind, ws in windows.items() if kind in stand for s, e in ws]
    if not spans:
        return []
    out: list[Visit] = []
    last: dict[str, int] = {}
    f = min(s for s, _, _ in spans)
    end = max(e for _, e, _ in spans)
    while f < end:
        need = {k: 2 * drive + int(stand[k] * fps) + int(drive * 0.4) for k in stand}
        # a visit starts while its kind is wanted (a short step still gets its one truck)
        ready = [k for s, e, k in spans if s <= f < e]
        if not ready:
            later = [s for s, _, _ in spans if s > f]
            if not later:
                break
            f = min(later)
            continue
        kind = min(set(ready), key=lambda k: last.get(k, -1))
        park = f + drive
        leave = park + int(stand[kind] * fps)
        v = Visit(kind, f, park, leave, leave + drive + int(drive * 0.4))
        out.append(v)
        last[kind] = v.arrive
        f = v.gone + int(fps * 1.0)
    return out


def truck_at(v: Visit, frame: float, lane: Lane) -> tuple[Point, float] | None:
    """Where the truck of a visit is at a frame, and its heading; None when it is not there.
    It drives in forwards and reverses out, so it always faces the site."""
    if frame < v.arrive or frame > v.gone:
        return None
    if frame <= v.parked:
        u = smooth((frame - v.arrive) / max(1, v.parked - v.arrive))
    elif frame <= v.leave:
        u = 1.0
    else:
        u = 1.0 - smooth((frame - v.leave) / max(1, v.gone - v.leave))
    return lane.at(u * lane.length), lane.heading_in


# ---- people --------------------------------------------------------------------------------

def crew_spots(n: int, area: tuple[float, float, float, float], z: float, rng: random.Random,
               lane: Lane | None, avoid: list[Point], gap: float = 3.0, spacing: float = 1.2) -> list[tuple[float, float, float]]:
    """n places to stand in ``area`` (x0, y0, x1, y1) on level z, out of the truck lane, at least
    ``gap`` from each point in ``avoid`` (where loads come down) and ``spacing`` from each other."""
    x0, y0, x1, y1 = area
    spots: list[tuple[float, float, float]] = []
    for _ in range(n * 40):
        if len(spots) >= n:
            break
        p = (rng.uniform(x0 + 0.8, x1 - 0.8), rng.uniform(y0 + 0.8, y1 - 0.8))
        if lane is not None and not lane.keeps_clear(p):
            continue
        if any(math.dist(p, a) < gap for a in avoid):
            continue
        if any(math.dist(p, s[:2]) < spacing for s in spots):
            continue
        spots.append((p[0], p[1], z))
    return spots
