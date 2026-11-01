"""The site's rules: one lift at a time, loads only on the hook or in place, trucks in the lane
one at a time, nobody in the lane or under a load."""

import math
import random
from types import SimpleNamespace

from sitetwin_worker import logistics as L


def steps_and_places(n=20):
    guids = [f"b{i}" for i in range(n)]
    st = SimpleNamespace(guids=guids, start=100, end=400)
    appear = {g: (100 + i * 10, 110 + i * 10, "lift") for i, g in enumerate(guids)}
    places = {g: (i * 2.0, 5.0, 4.0, 0.6) for i, g in enumerate(guids)}
    return [st], appear, places


def test_the_crane_lifts_one_piece_at_a_time_and_the_rest_are_fixed_in_place():
    steps, appear, places = steps_and_places(20)
    lifts = L.plan_lifts(steps, appear, places, (-10, -10, 0.0), fps=30, window=(0, 1000))
    cycle = round(L.CYCLE * 30)
    assert len(lifts) == 300 // cycle
    for a, b in zip(lifts, lifts[1:]):
        assert a.end < b.start                       # never two at once
    lifted = {l.guid for l in lifts}
    assert all(appear[g][2] == "crane" for g in lifted)
    assert all(appear[g][2] == "fit" for g in appear if g not in lifted)


def test_a_load_is_on_the_hook_from_hoist_to_set_down_and_comes_down_where_it_goes():
    steps, appear, places = steps_and_places(4)
    lifts = L.plan_lifts(steps, appear, places, (-10, -10, 0.0), fps=30, window=(0, 1000))
    crane = L.Crane(0.0, -20.0, 40.0, 50.0, lifts)
    lift = lifts[1]
    t = lift.times()
    # before it leaves the ground nothing is carried
    assert crane.pose(t["sling"][0] + 1, 8.0)[3] is None
    # mid-slew the load hangs under the hook, high over the building
    a, r, tip, carried, base = crane.pose(sum(t["slew"]) / 2, 8.0)
    assert carried is lift and base > 8.0
    # at set-down the hook is over its place and the load at its level
    a, r, tip, carried, base = crane.pose(t["lower"][1] - 0.01, 8.0)
    hx, hy = crane.hook_xy(a, r)
    assert math.hypot(hx - lift.place[0], hy - lift.place[1]) < 0.5
    assert abs(base - lift.place[2]) < 0.3
    # after release it is no longer on the hook
    assert crane.pose(lift.end, 8.0)[3] is None


def test_trucks_use_the_lane_one_at_a_time_and_stay_on_it():
    lane = L.Lane(far=(0.0, -80.0), bay=(0.0, -10.0))
    visits = L.plan_visits({"mixer_truck": [(0, 3000)], "flatbed_truck": [(500, 2500)]}, lane, fps=30)
    assert len(visits) >= 3 and {v.kind for v in visits} == {"mixer_truck", "flatbed_truck"}
    for a, b in zip(visits, visits[1:]):
        assert a.gone < b.arrive
    for v in visits:
        for f in range(v.arrive, v.gone, 7):
            (x, y), heading = L.truck_at(v, f, lane)
            assert abs(x) < 0.01 and -80.01 <= y <= -9.99    # on the lane's centre line
        assert L.truck_at(v, v.parked + 1, lane)[0] == lane.bay


def test_people_keep_out_of_the_lane_and_from_under_loads():
    lane = L.Lane(far=(0.0, -200.0), bay=(0.0, -10.0))
    avoid = [(5.0, 5.0), (-5.0, -5.0)]
    spots = L.crew_spots(12, (-15, -15, 15, 15), 4.0, random.Random(1), lane, avoid)
    assert len(spots) == 12
    for x, y, z in spots:
        assert lane.keeps_clear((x, y)) and z == 4.0
        assert all(math.dist((x, y), a) >= 3.0 for a in avoid)
