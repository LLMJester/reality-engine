"""Run 5 item sets and the run 5 seal (V3-042)."""
import pytest

from v3 import items as I
from v3 import run5_sets as R5


@pytest.fixture(autouse=True)
def tmp_opened(tmp_path, monkeypatch):
    monkeypatch.setattr(R5, "OPENED", tmp_path / "opened.json")


def test_set_sizes_and_shared_bases():
    S = R5.sets()
    stages = {f"s{n}_screen": 320 for n in (3, 4, 5)} | {f"s{n}_confirm": 480 for n in (3, 4, 5)}   # V3-058
    assert {n: len(v) for n, v in S.items()} == {"main": 480, "sealed": 720, "pilot": 60, "mini": 6, "mini_sealed": 4,
                                                 **stages, "sealed2": 720}
    for name in ("main", "sealed", "pilot", "mini_sealed", *stages, "sealed2"):
        bases = {i.split("x")[1] for i in S[name]}
        assert len(bases) * 4 == len(S[name])                 # every base at k = 1, 2, 4, 8
    allb = [{int(i.split("x")[1]) for i in v} for v in S.values()]
    for a in range(len(allb)):
        for b in range(a + 1, len(allb)):
            assert all(abs(x - y) >= R5.STRIDE for x in allb[a] for y in allb[b])


def test_sealed_until_opened_once():
    S = R5.sets()
    with pytest.raises(I.Sealed):
        I.load(S["mini_sealed"][0])
    with pytest.raises(I.Sealed):
        I.load(S["sealed"][-1])
    assert I.load(S["main"][0])["id"] == S["main"][0]
    first = R5.open_sealed("mini_sealed", "test")
    assert I.load(S["mini_sealed"][0])["id"] == S["mini_sealed"][0]
    with pytest.raises(I.Sealed):                             # opening one sealed set leaves the other sealed
        I.load(S["sealed"][0])
    assert R5.open_sealed("mini_sealed", "again") == first     # recorded once, never changed


def test_fresh_sealed_set_is_sealed_until_opened():
    S = R5.sets()                                             # V3-058: sealed2, opened only at the final check
    with pytest.raises(I.Sealed):
        I.load(S["sealed2"][0])
    assert I.load(S["s3_screen"][0])["id"] == S["s3_screen"][0]
    R5.open_sealed("sealed2", "test")
    assert I.load(S["sealed2"][0])["id"] == S["sealed2"][0]


def test_stage_sets_avoid_v2_seeds():
    allb = [int(i.split("x")[1]) for n in R5.stage_set_names() for i in R5.sets()[n]]
    assert all(not lo - R5.STRIDE < b < hi for b in allb for lo, hi in R5.EXCLUDED_RANGES)


def test_seal_covers_only_the_chosen_windows():
    assert I.load("K1x860000")["id"] == "K1x860000"            # an earlier run's item near the sealed bases
