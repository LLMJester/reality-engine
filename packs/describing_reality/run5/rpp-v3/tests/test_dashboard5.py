"""The run 5 dashboard feed (v3/dashboard5/feed5.py): the record scan and the trust report's before and after
(2026-10-30). Checked against hand-computed values on fake records."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from v3.dashboard5 import feed5 as F


def _rec(score, tok=1000, built=True, error=None):
    r = {"grade": {"status": "ok", "score": score}, "output_tokens": tok, "error": error}
    if built is not None:
        r["validity"] = {"usable": built, "executable": built}
    return r


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "RUNS", tmp_path)
    monkeypatch.setattr(F, "CACHE", F.ScoreCache())

    def put(phase, gid, iid, rec=None, zero=False):
        d = tmp_path / phase / "codex" / gid / iid
        d.mkdir(parents=True, exist_ok=True)
        if zero:
            (d / "zero.json").write_text("{}")
        else:
            (d / "record.json").write_text(json.dumps(rec))
    return put


def test_scan_reads_scores_and_trust_fields(runs):
    runs("p", "g", "K1x1", _rec(0.5, tok=1200, built=False))
    runs("p", "g", "K2x1", _rec(1.0, built=None))               # a core with no checker
    runs("p", "g", "K4x1", _rec(0.9, error="boom"))             # a failed run has no score
    runs("p", "g", "K8x1", zero=True)                            # a design-induced failure counts 0
    raw = F.CACHE.scan("p")
    assert raw["g"] == {"K1x1": 0.5, "K2x1": 1.0, "K4x1": None, "K8x1": 0.0}
    ext = F.CACHE.extra["p"]["g"]
    assert ext["K1x1"]["built"] is False and ext["K1x1"]["tok"] == 1200
    assert ext["K2x1"]["built"] is None
    assert ext["K4x1"]["built"] is None                          # validity of a failed run is ignored


def test_scan_rereads_records_cached_by_older_code(runs):
    runs("p", "g", "K1x1", _rec(0.5, tok=7))
    F.CACHE.scan("p")
    path = next(iter(F.CACHE.seen))
    mt, g, i, s, _ = F.CACHE.seen[path]
    F.CACHE.seen[path] = (mt, g, i, s, {"t": mt})                # an entry from before the trust fields
    F.CACHE.scan("p")
    assert F.CACHE.extra["p"]["g"]["K1x1"]["tok"] == 7


def test_old_cache_is_replaced(monkeypatch):
    monkeypatch.setattr(F, "CACHE", SimpleNamespace(seen={}))   # no .extra: made by older code
    assert isinstance(F.fresh_cache(), F.ScoreCache) and isinstance(F.CACHE, F.ScoreCache)


def test_metrics_by_hand():
    raw = {"g": {"K1x1": 1.0, "K1x2": 0.5, "K8x1": 0.0, "K8x2": None}}
    ext = {"g": {"K1x1": {"built": True, "tok": 100}, "K1x2": {"built": False, "tok": 300},
                 "K8x1": {"built": None, "tok": None}}}
    m = F._metrics(raw, ext, "g", ["K1x1", "K1x2", "K8x1", "K8x2"])
    assert m == {"n": 3, "mean": 0.5, "by_k": {"1": 0.75, "8": 0.0}, "built": 0.5, "tokens": 200}
    assert F._metrics(raw, ext, "missing", ["K1x1"]) is None


def _mode():
    return SimpleNamespace(phase="m", phase2="m_s2", stage2_items=["K1x1", "K1x2", "K8x1"], sealed=["K1x9", "K8x9"])


def test_before_after_uses_only_items_both_have_done(monkeypatch):
    monkeypatch.setattr(F, "CACHE", F.ScoreCache())
    F.CACHE.extra = {"m_s2": {"w": {i: {"built": True, "tok": 10} for i in ("K1x1", "K1x2", "K8x1")},
                              "t": {i: {"built": True, "tok": 20} for i in ("K1x1", "K1x2")}}, "m": {}}
    raw2 = {"w": {"K1x1": 0.2, "K1x2": 0.4, "K8x1": 0.9}, "t": {"K1x1": 0.6, "K1x2": 0.8}}
    labs = {"stage2": {"D01": {"id": "w"}, "D09": {"id": "t"}}}
    live = {"fit": {"final_design": "D09", "final_levels": {"A": -1, "B": -1, "C": -1, "D": 1, "E": -1}}}
    out = F.before_after(_mode(), {}, {}, {}, raw2, labs, live)
    assert out["target"] == "D09" and out["final"] is False and out["levels"]["D"] == 1
    assert out["stage2"]["before"]["n"] == 2 and out["stage2"]["before"]["mean"] == pytest.approx(0.3)   # K8x1 left out
    assert out["stage2"]["after"]["mean"] == pytest.approx(0.7) and out["stage2"]["after"]["tokens"] == 20
    assert "sealed" not in out


def test_before_after_uses_the_retention_verdict_and_never_the_retired_sealed_set(monkeypatch):
    # V3-058: the stage 2 verdict is the gated retention rule's (D01 here), not the fitted argmax (D17); the old
    # sealed set's records (K1x9, K8x9 in raw) are retired and never shown
    monkeypatch.setattr(F, "CACHE", F.ScoreCache())
    F.CACHE.extra = {"m_s2": {}, "m": {}}
    raw2 = {"w": {"K1x1": 0.5}, "t": {"K1x1": 0.5}, "u": {"K1x1": 0.5}}
    raw = {"w": {"K1x9": 0.4, "K8x9": 0.2}, "u": {"K1x9": 0.8, "K8x9": 0.6}}
    labs = {"stage2": {"D01": {"id": "w"}, "D09": {"id": "t"}, "D17": {"id": "u"}}}
    analysis = {"stage2": {"final_design": "D17", "final_levels": {"A": -1, "B": -1, "C": -1, "D": -1, "E": 1}},
                "stage2_retention": {"final_design": "D01", "candidate_levels": {f: "-" for f in "ABCDE"}}}
    out = F.before_after(_mode(), {"final_design": "D01"}, analysis, raw, raw2, labs, {"fit": {"final_design": "D09"}})
    assert out["target"] == "D01" and out["final"] is True                 # the official verdict, not the live guess
    assert "sealed" not in out and out["stages"] == []


def test_span_counts_scored_runs_and_their_times():
    raw = {"g": {"K1x1": 0.5, "K1x2": None, "K8x1": 0.0}}
    ext = {"g": {"K1x1": {"t": 1000.0}, "K8x1": {"t": 4600.0}}}
    done, first, last = F._span(raw, ext, [("g", "K1x1"), ("g", "K1x2"), ("g", "K8x1"), ("h", "K1x1")])
    assert done == 2 and first < last
    assert F._span({}, {}, [("g", "K1x1")]) == (0, None, None)


def test_stage_tiles_counts_and_final_estimate(monkeypatch):
    monkeypatch.setattr(F, "CACHE", F.ScoreCache())
    F.CACHE.extra = {"m": {}, "m_s2": {}}
    m = SimpleNamespace(phase="m", phase2="m_s2", gen=["K1x1", "K8x1"], hand=["H1"], stage2_items=["K1x1", "K8x1"],
                        sealed=["K1x9", "K8x9"], dir=Path("does-not-exist"), name="mini", stage_numbers=(),
                        final_sealed_name="s2", final_sealed=["K1x7", "K8x7"])
    founders = {"FACTS": {"id": "f"}, "A3": {"id": "a"}, "A0P": {"id": "c"}, "ONT_LOOSE": {"id": "o"}}
    s2 = {"D01": {"id": "f"}, "D09": {"id": "t"}}
    raw = {"f": {"K1x1": 1.0, "K8x1": 0.5, "H1": 0.2}, "a": {"K1x1": 0.4}}
    raw2 = {"f": {"K1x1": 0.6}, "t": {"K1x1": 0.7, "K8x1": 0.1}}
    st = {"winner": "FACTS", "runner_up": "A3", "hand_top": "ONT_LOOSE", "events": [
        {"t": "2026-10-29T22:50:36", "msg": "plus texts approved: stage 2 starts"}]}
    analysis = {"stage1": {"generated": {"designs": {"FACTS": {"mean": 0.657}}}}}
    t = F.stage_tiles(m, st, raw, raw2, {"founders": founders, "stage2": s2}, analysis, None)
    assert (t["stage1"]["done"], t["stage1"]["planned"]) == (4, 4 * 3)
    assert t["stage1"]["winner_mean"] == 0.657 and t["plus"]["approved"] == "2026-10-29T22:50:36"
    assert (t["stage2"]["done"], t["stage2"]["planned"]) == (3, 32 * 2)
    # V3-058: the final check is on the fresh sealed set: the four founders (FACTS is also the final design while
    # nothing has replaced it), no hand-written runs for FACTS itself; the retired set has no tile scores
    assert t["final2"]["planned"] == 4 * 2 and t["final2"]["done"] == 0 and "final" not in t
    assert t["order"] == ["stage1", "plus", "stage2", "final2"] and t["retired"] is None
    assert t["trust"] is None


def test_stage_tiles_after_the_stages_stop(monkeypatch):
    # run 5 ended after stage 3: stage 4's tiles are marked skipped (not done), and the final check counts only the
    # fresh sealed runs, not the final design's stage 1 hand-written runs when it is the stage 1 winner itself
    from v3 import run5 as R
    monkeypatch.setattr(R, "best_genome", lambda st: {"id": "f"})
    monkeypatch.setattr(F, "CACHE", F.ScoreCache())
    F.CACHE.extra = {"m": {}, "m_s2": {}}
    m = SimpleNamespace(phase="m", phase2="m_s2", gen=["K1x1"], hand=["H1", "H2"], stage2_items=["K1x1"], sealed=[],
                        dir=Path("does-not-exist"), name="mini", stage_numbers=(3, 4), final_sealed_name="s2",
                        final_sealed=["K1x7", "K8x7"], stage_phase=lambda n, p: f"m_s{n}{p[0]}",
                        stage_items=lambda n, p: ["K1x5"])
    founders = {"FACTS": {"id": "f"}, "A3": {"id": "a"}, "A0P": {"id": "c"}, "ONT_LOOSE": {"id": "o"}}
    raw = {g: {"K1x7": 0.5, "K8x7": 0.5} for g in "faco"}
    raw["f"].update({"H1": 0.3, "H2": 0.4})
    st = {"stage": "done", "n": 3, "winner": "FACTS", "runner_up": "A3", "hand_top": "ONT_LOOSE", "best": {"label": "x"},
          "stages": {"3": {"screen": {"advancing": {}}, "confirm": None}}, "events": []}
    t = F.stage_tiles(m, st, raw, {}, {"founders": founders, "stage2": {}}, {}, None)
    assert (t["final2"]["done"], t["final2"]["planned"]) == (8, 8)
    assert not t.get("s3_screen", {}).get("skipped")
    assert all(t[k]["skipped"] and t[k]["planned"] is None for k in ("s4_write", "s4_screen", "s4_confirm"))


def test_before_after_waits_for_stage2():
    assert F.before_after(_mode(), {}, {}, {}, {}, {"stage2": {}}, None) is None
