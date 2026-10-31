"""Engine tests on the toy pack (no model, no network; about ten seconds)."""
import json
import pathlib

import numpy as np
import pytest

from engine import plan as P
from engine.runner import Halted
from engine.stages import Engine
from engine.stats import Stats, factorial_levels

ROOT = pathlib.Path(__file__).resolve().parents[1]


def toy_plan(tmp_path, **over):
    raw = json.loads((ROOT / "plans" / "toy.json").read_text(encoding="utf-8"))
    raw.update(over, work=str(tmp_path / "work"))
    f = tmp_path / "plan.json"
    f.write_text(json.dumps(raw), encoding="utf-8")
    return P.load(f)


# ---------------------------------------------------------------------- statistics
def test_holm_steps_down():
    h = Stats().holm({"a": 0.001, "b": 0.02, "c": 0.04})
    assert h["a"]["reject"] and h["b"]["reject"] and h["c"]["reject"]      # 0.04 <= 0.05/1
    h = Stats().holm({"a": 0.001, "b": 0.03, "c": 0.04})
    assert h["a"]["reject"] and not h["b"]["reject"] and not h["c"]["reject"]  # 0.03 > 0.05/2 stops the rest


def test_retention_keeps_a_real_effect_and_nothing_else():
    rng = np.random.default_rng(1)
    items = [f"c{c}-{j}" for c in range(30) for j in range(2)]
    st = Stats(cluster_of=lambda i: i.split("-")[0])
    names = [f"D{n + 1:02d}" for n in range(4)]
    sc = {}
    for n, lv in enumerate(factorial_levels(2)):
        sc[names[n]] = {i: 0.5 + 0.1 * (lv[0] > 0) + rng.normal(0, 0.05) for i in items}   # only part a helps
    res = st.retention(sc, names, items, ["a", "b"])
    assert res["kept_mains"] == ["a"] and res["final_design"] == "D02" and res["replaces"]


def test_retention_keeps_the_current_best_when_nothing_helps():
    rng = np.random.default_rng(2)
    items = [f"c{c}-{j}" for c in range(30) for j in range(2)]
    st = Stats(cluster_of=lambda i: i.split("-")[0])
    names = [f"D{n + 1:02d}" for n in range(4)]
    sc = {d: {i: 0.5 + rng.normal(0, 0.05) for i in items} for d in names}
    res = st.retention(sc, names, items, ["a", "b"])
    assert res["final_design"] == "D01" and not res["replaces"]


def test_screen_bar_and_cap():
    items = [f"c{c}-1" for c in range(10)]
    st = Stats(cluster_of=lambda i: i.split("-")[0])
    sc = {"CUR": {i: 0.5 for i in items}, "x.1": {i: 0.52 for i in items}, "x.2": {i: 0.51 for i in items},
          "y.1": {i: 0.505 for i in items}}
    meta = {"x.1": {"slot": "x", "archetype": 1}, "x.2": {"slot": "x", "archetype": 2},
            "y.1": {"slot": "y", "archetype": 1}}
    res = st.screen(sc, "CUR", meta, items, gate=0.015)
    assert res["advancing"] == {"x": "x.1"}                       # y's best (+0.005) is under the bar


# ---------------------------------------------------------------------- the whole loop
def test_toy_run_end_to_end(tmp_path):
    plan = toy_plan(tmp_path)
    P.lock(plan, "test")
    eng = Engine(plan)
    assert eng.loop() == "done"
    an = eng.analysis()
    assert an["compare"]["winner"] == "FACTS"                     # the hidden truth: facts rules help most
    assert an["final"]["sequence"]["1_vs_control"]["counted"]
    rep = json.loads((eng.work / "report.json").read_text(encoding="utf-8"))
    assert rep["right"]["mean"] is not None and rep["effort"]["runs"] > 0
    assert (eng.work / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>")


def test_run_refuses_without_a_matching_lock(tmp_path):
    plan = toy_plan(tmp_path)
    with pytest.raises(SystemExit):
        Engine(plan)
    P.lock(plan, "test")
    pathlib.Path(plan["_path"]).write_text(pathlib.Path(plan["_path"]).read_text().replace('"auto"', '"human"'))
    with pytest.raises(SystemExit):
        Engine(P.load(plan["_path"]))                             # the plan changed after the lock


def test_human_approval_waits(tmp_path):
    plan = toy_plan(tmp_path, approval="human")
    P.lock(plan, "test")
    eng = Engine(plan)
    assert eng.loop() == "takeapart_write" and "a person" in eng.st["waiting_for"]
    (eng.work / "stages" / "takeapart" / "APPROVED").write_text("ok")
    assert Engine(plan).loop() == "round_write"                   # resumes, then waits again for round 1


def test_sealed_items_never_run_before_the_final_check(tmp_path):
    plan = toy_plan(tmp_path)
    plan["sets"]["takeapart"] = "sealed"
    eng = Engine(plan, check_lock=False)
    eng.loop()
    assert eng.st["step"] == "takeapart_run" and "sealed" in eng.st["halted"]


def test_pause_stops_and_resumes(tmp_path):
    plan = toy_plan(tmp_path)
    eng = Engine(plan, check_lock=False)
    eng.pause.write_text("")
    assert eng.loop() == "compare"
    eng.pause.unlink()
    assert Engine(plan, check_lock=False).loop() == "done"


def test_infra_failure_halts_and_design_failure_scores_zero(tmp_path):
    from packs.toy.pack import ToyPack

    class Flaky(ToyPack):
        def run(self, candidate, item, ctx):
            if item == "T3-1" and candidate["name"] == "TABLE":
                return {"error": "the model broke a rule", "failure": "design"}
            if item == "T5-2" and candidate["name"] == "PLAIN" and not self.ok:
                return {"error": "connection reset", "failure": "infra"}
            return super().run(candidate, item, ctx)
    plan = toy_plan(tmp_path)
    pk = Flaky()
    pk.ok = False
    eng = Engine(plan, pack=pk, check_lock=False)
    assert eng.loop() == "compare" and "infra" in eng.st["halted"]
    pk.ok = True
    eng = Engine(plan, pack=pk, check_lock=False)
    eng.loop()
    tid = eng.founders()["TABLE"]["id"]
    assert eng.store.score("main", tid, "T3-1") == 0.0           # scored 0, never dropped


def test_designer_brief_has_no_item_content(tmp_path):
    plan = toy_plan(tmp_path)
    Engine(plan, check_lock=False).loop()
    brief = json.loads((tmp_path / "work" / "stages" / "round1" / "brief.json").read_text(encoding="utf-8"))
    assert "T1-" not in brief["text"] and "T9" not in json.dumps(brief)


def test_halted_on_frozen_versions_changed(tmp_path):
    plan = toy_plan(tmp_path, approval="human")
    eng = Engine(plan, check_lock=False)
    eng.loop()
    f = eng.work / "stages" / "takeapart" / "candidates.json"
    doc = json.loads(f.read_text(encoding="utf-8"))
    doc["alts"]["rules"][0]["value"] = "facts facts"
    f.write_text(json.dumps(doc), encoding="utf-8")
    (f.parent / "APPROVED").write_text("ok")
    eng = Engine(plan, check_lock=False)
    with pytest.raises(Halted):
        eng.takeapart_write()
