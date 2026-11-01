import numpy as np

from sitetwin_worker.model import Element
from sitetwin_worker.sequence import plan, plant_windows


def el(guid, kind, storey, elev, x=0.0, z=None):
    z = elev if z is None else z
    v = np.array([[x, 0, z], [x + 1, 1, z + 1]], dtype=np.float32)
    return Element(guid, kind, guid, storey, elev, v, np.zeros((1, 3), dtype=np.int32))


def model():
    return [
        el("s1", "IFCSLAB", "Level 1", 4.0, z=7.7),
        el("c1", "IFCCOLUMN", "Level 1", 4.0),
        el("f1", "IFCFOOTING", "Foundation", -1.5),
        el("p1", "IFCPILE", "Foundation", -1.5, z=-14),
        el("w1", "IFCWINDOW", "Level 1", 4.0, z=5),
        el("c0", "IFCCOLUMN", "Ground", 0.0),
        el("c0b", "IFCCOLUMN", "Ground", 0.0, x=10),
    ]


def test_bottom_up_and_in_build_order():
    tl = plan(model(), seconds=30, fps=10)
    assert [(s.storey, s.phase) for s in tl.steps] == [
        ("Foundation", "Foundations"), ("Ground", "Vertical structure"),
        ("Level 1", "Vertical structure"), ("Level 1", "Floors"), ("Level 1", "Envelope")]
    # every element appears, inside the build, and later steps start later
    assert set(tl.appear) == {e.guid for e in model()}
    starts = [s.start for s in tl.steps]
    assert starts == sorted(starts) and starts[0] == tl.build_start
    assert tl.steps[-1].end == tl.build_end
    assert all(tl.build_start <= a <= b <= tl.frames for a, b, _ in tl.appear.values())


def test_the_pile_goes_in_before_the_footing_on_top_of_it():
    tl = plan(model(), seconds=30, fps=10)
    assert tl.appear["p1"][0] < tl.appear["f1"][0]


def test_cast_in_place_grows_and_beams_and_glazing_are_lifted():
    tl = plan(model(), seconds=30, fps=10)
    assert tl.appear["c1"][2] == "grow" and tl.appear["w1"][2] == "lift"


def test_the_tower_crane_stays_once_it_is_up():
    tl = plan(model(), seconds=30, fps=10)
    w = plant_windows(tl)
    assert len(w["tower_crane"]) == 1
    assert w["tower_crane"][0][0] == tl.steps[1].start
    assert "piling_rig" in w and w["piling_rig"][0][1] <= tl.steps[1].start + 1


def test_a_film_is_split_into_runs_for_several_gpus():
    from sitetwin_worker.render import chunks
    runs = chunks(9000, 30, 10)
    assert len(runs) == 10 and runs[0] == (1, 900) and runs[-1][1] == 9000
    assert all(b + 1 == c for (_, b), (c, _) in zip(runs, runs[1:]))
    assert chunks(300, 30, 10) == [(1, 300)]          # ten seconds: not worth splitting
