from pathlib import Path

import pytest

from sitetwin_app import ifc

SMALL = Path(__file__).parent / "fixtures" / "small.ifc"


def test_it_reads_where_the_model_stands():
    s = ifc.summarise_file(SMALL)
    assert s.schema == "IFC4"
    assert s.project == "Berth 3"
    assert s.site == "Sibline é"
    assert s.latitude == pytest.approx(33 + 41 / 60 + 52.5 / 3600, abs=1e-7)
    assert s.longitude == pytest.approx(35 + 25 / 60 + 4 / 3600, abs=1e-7)
    assert s.elevation == 4.5


def test_southern_and_western_angles_are_negative():
    assert ifc._angle([-33, -51, -54, -360000]) == pytest.approx(-(33 + 51 / 60 + 54.36 / 3600), abs=1e-7)


def test_it_counts_what_is_built():
    s = ifc.summarise_file(SMALL)
    assert s.counts["IFCCOLUMN"] == 2 and s.counts["IFCSLAB"] == 2
    assert "IFCRELCONTAINEDINSPATIALSTRUCTURE" not in s.counts


def test_the_draft_sequence_goes_up_storey_by_storey_in_build_order():
    s = ifc.summarise_file(SMALL)
    assert [st["name"] for st in s.storeys] == ["Ground", "Level 1"]
    assert [(q["storey"], q["phase"], q["elements"]) for q in s.sequence] == [
        ("Ground", "Foundations", 1),
        ("Ground", "Vertical structure", 2),
        ("Ground", "Floors", 1),
        ("Level 1", "Horizontal structure", 1),
        ("Level 1", "Floors", 1),
        ("Level 1", "Envelope", 1),
    ]


def test_semicolons_and_quotes_inside_names_do_not_break_it():
    entities = ifc.parse(SMALL.read_text())
    assert entities[20].args[2] == "Pad; F1"
    assert entities[26].args[2] == "It's a window"


def test_something_that_is_not_ifc_is_refused():
    with pytest.raises(ifc.IfcError):
        ifc.summarise("hello")
