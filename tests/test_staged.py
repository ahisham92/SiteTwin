"""The planner's own stages, carried in the model: preparing them, reading them, and the plan."""

import json
from pathlib import Path

import numpy as np
import pytest

from sitetwin_app import ifc as ifc_text
from sitetwin_worker.model import Element
from sitetwin_worker.sequence import calendar, plan_staged

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.api as api  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _box(guid, stage, order, how, x, y, z0=0.0, z1=1.0, w=1.0):
    v = np.array([(x, y, z0), (x + w, y, z0), (x + w, y + w, z0), (x, y + w, z0),
                  (x, y, z1), (x + w, y, z1), (x + w, y + w, z1), (x, y + w, z1)], dtype=np.float32)
    f = np.array([(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)], dtype=np.int32)
    return Element(guid, "IFCBEAM", guid, "", 0.0, v, f, stage=stage, stage_name=f"stage {stage}",
                   stage_order=order, how=how)


def test_stages_run_in_order_along_the_structure_one_cast_after_another():
    els = [_box(f"old{i}", "1", 1, "remove", 0, i * 6.0) for i in range(5)]
    els += [_box(f"pile{i}", "2", 2, "repair", 0, i * 6.0, -10, 0.5, 0.6) for i in range(5)]
    els += [_box(f"cap{i}", "5A", 3, "cast", 0, y) for i, y in enumerate([24.0, 0.0, 12.0, 6.0, 18.0])]
    els += [_box("ground", "G", 0, "ground", -5, 0, -3, 0, 30)]
    tl = plan_staged(els, seconds=40, fps=24, stages=[{"code": "1", "days": 10}, {"code": "2", "days": 10},
                                                      {"code": "5A", "days": 20}])
    assert [s.storey for s in tl.steps] == ["Stage 1", "Stage 2", "Stage 5A"]
    # the old deck is there today and broken out during stage 1, from the root outwards
    starts = [tl.removed[f"old{i}"][0] for i in range(5)]
    assert starts == sorted(starts) and starts[0] >= tl.steps[0].start and tl.removed["old4"][1] <= tl.steps[0].end + 1
    # the piles stay and get their repair in stage 2
    assert tl.appear["pile0"][2] == "repair" and "repair:pile0" in tl.removed
    # caps: from the root outwards, each started after the one before, never more than two at once
    caps = sorted((tl.appear[f"cap{i}"][0], tl.appear[f"cap{i}"][1], els[10 + i].centre[1]) for i in range(5))
    assert [c[2] for c in caps] == sorted(c[2] for c in caps)
    for (a0, b0, _), (a1, b1, _), (a2, _, _) in zip(caps, caps[1:], caps[2:]):
        assert a0 < a1 < a2 and b0 <= a2
    # the planner's durations set the shares: stage 5A (20 days) twice stage 1 (10)
    d1 = tl.steps[0].end - tl.steps[0].start
    d3 = tl.steps[2].end - tl.steps[2].start
    assert abs(d3 - 2 * d1) <= 2
    assert tl.appear["ground"][2] == "exist"
    cal = calendar(tl)
    assert cal[-1][1] == pytest.approx(40 * 7 / 5)


def test_a_revit_model_staged_for_sitetwin(tmp_path):
    # a model as Revit exports it: element ids as Tags, the site far out in a national grid
    f = ifcopenshell.open(str(ROOT / "samples" / "sample-building.ifc"))
    for k, el in enumerate(f.by_type("IfcElement")):
        el.Tag = str(1000 + k)
    site = f.by_type("IfcSite")[0]
    site.ObjectPlacement = f.createIfcLocalPlacement(None, f.createIfcAxis2Placement3D(
        f.createIfcCartesianPoint((-347337781.0, -54505750.0, 0.0)), f.createIfcDirection((0.0, 0.0, 1.0)),
        f.createIfcDirection((0.0, 1.0, 0.0))))
    src = tmp_path / "revit.ifc"
    f.write(str(src))
    n = len(f.by_type("IfcElement"))
    spec = {"setting": "marine",
            "location": {"lat": 33.651153, "lon": 35.405074, "north": 2.09, "sea_level": 0.4},
            "stages": [{"code": "1", "name": "Demolition", "how": "remove"},
                       {"code": "5A", "name": "Caps", "how": "cast"}],
            "elements": {str(1000 + k): ("X" if k == n - 1 else "1" if k % 2 else "5A/cast") for k in range(n)}}
    stages = tmp_path / "stages.json"
    stages.write_text(json.dumps(spec))
    from tools.stage_ifc import stage
    out = tmp_path / "staged.ifc.gz"
    report = stage(src, stages, out)
    assert report["left_out"] == 1 and report["kept"] == n - 1

    s = ifc_text.summarise_file(out)
    assert s.setting == "marine"
    assert s.latitude == pytest.approx(33.651153, abs=1e-5) and s.longitude == pytest.approx(35.405074, abs=1e-5)
    assert [st["storey"] for st in s.sequence] == ["Stage 1", "Stage 5A"]
    assert sum(st["elements"] for st in s.sequence) == n - 1

    from sitetwin_worker import model
    els = model.load(out)
    assert {e.stage for e in els} == {"1", "5A"}
    assert all(e.how in ("remove", "cast") for e in els)
    info = model.site_info(model.open_ifc(out))
    assert info["setting"] == "marine" and info["sea_level"] == pytest.approx(0.4)
    assert [st["code"] for st in info["stages"]] == ["1", "5A"]
    # the site is at the model's origin: nothing is kilometres away
    assert max(float(np.abs(e.verts[:, :2]).max()) for e in els) < 1000
