"""The AI pass's runs (modal_real.plan_runs): every frame covered, no wasted padding."""
import sys
import types

import pytest


@pytest.fixture(scope="module")
def real():
    fake = types.SimpleNamespace(App=lambda *a, **k: types.SimpleNamespace(function=lambda **k: (lambda f: f),
                                                                         local_entrypoint=lambda: (lambda f: f)),
                                 Volume=types.SimpleNamespace(from_name=lambda *a, **k: None),
                                 Image=types.SimpleNamespace(debian_slim=lambda **k: _Chain()))
    saved = sys.modules.get("modal")
    sys.modules["modal"] = fake
    try:
        import importlib
        import modal_real
        yield importlib.reload(modal_real)
    finally:
        if saved is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = saved


class _Chain:
    def __getattr__(self, name):
        return lambda *a, **k: self


@pytest.mark.parametrize("n", [20, 73, 74, 80, 137, 138, 160, 480, 961])
def test_runs_cover_every_frame(real, n):
    runs = real.plan_runs(n)
    kept = 0
    for i, (start, length) in enumerate(runs):
        assert (length - 1) % 4 == 0 and length <= real.FULL
        assert start == i * (real.KEEP - real.OVERLAP)
        kept = start + min(length, real.KEEP)
    assert kept >= n                                        # nothing left out
    start, length = runs[-1]
    assert start + length - n < real.FULL - real.KEEP + 4   # last run not padded beyond its tail


def test_ten_second_film_makes_fewer_frames(real):
    # 10 s at 16 fps: three runs of 81 (243 frames) before, now 81 + 81 + 41
    assert [length for _, length in real.plan_runs(160)] == [81, 81, 41]
    assert [length for _, length in real.plan_runs(80)] == [81, 25]
