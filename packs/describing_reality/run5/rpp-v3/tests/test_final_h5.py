import pytest

from v3 import evolve as E
from v3 import final as F


@pytest.fixture(autouse=True)
def run1_rules(monkeypatch):
    monkeypatch.setitem(E.EV, "primary_builder", "claude")     # these cases are run 1's Claude-primary rule


def test_slope_positive_when_gap_grows_with_k():
    diff, cl = {}, {}
    for k in (1, 2, 4, 8):
        for j in range(6):
            i = f"K{k}x{1000000 + 10 * k + j}"
            diff[i], cl[i] = 0.05 * (k.bit_length() - 1) + 0.01 * (j % 3), f"s{k}{j}"
    r = F.slope_ci(diff, cl, n_boot=500)
    assert abs(r["est"] - 0.05) < 1e-6 and r["lo"] > 0


def test_slope_needs_two_levels():
    assert F.slope_ci({"K2x1": 0.1, "K2x2": 0.2}, {"K2x1": "a", "K2x2": "b"})["est"] is None


def _g(core, proto="one_shot"):
    return {"genes": {"core": {"kind": core}, "build_protocol": {"kind": proto}}}


def test_select_keeps_a_structured_finalist():
    data = {"front": ["a", "b", "c", "d"], "controls": [],
            "genomes": {"a": _g("equations"), "b": _g("simulation"), "c": _g("rules"), "d": _g("ontology")},
            "evals": {x: {"full": {"claude": {"e2e": e, "build": 1}}} for x, e in
                      (("a", .9), ("b", .8), ("c", .7), ("d", .5))}}
    assert F.select(data) == ["a", "b", "d"]
