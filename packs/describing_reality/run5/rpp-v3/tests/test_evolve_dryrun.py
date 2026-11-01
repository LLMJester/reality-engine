"""M7 acceptance: a dry run of the generation loop with stub builders, grader and designer (no spending)."""
import hashlib
import json

import pytest

from v3 import evolve as E
from v3.genome.seeds import build as seeds
from v3.state import Store


@pytest.fixture(autouse=True)
def run1_rules(monkeypatch):
    """These tests cover run 1's two-builder rules (Claude primary, V3-023); run 2's Luna-only mode (V3-030) has its
    own tests below, which switch the mode on."""
    # run 3 (V3-034) changed the config to directed breeding; these tests keep runs 1-2's settings
    monkeypatch.setitem(E.EV, "mode", "")
    monkeypatch.setitem(E.EV, "islands", 3)
    monkeypatch.setitem(E.EV, "migrants", 2)
    monkeypatch.setitem(E.EV, "migrate_every", 3)
    monkeypatch.setitem(E.EV, "operator_shares", {"point": 0.2, "swap": 0.15, "simplify": 0.1, "repair": 0.1,
                                                  "numeric": 0.05, "guidance": 0.1, "graft": 0.1, "big_jump": 0.05,
                                                  "param": 0.15})
    monkeypatch.setitem(E.EV, "micro", {"enabled": True, "bases": 3, "bases_small": 0, "tweaks_per_base": 6,
                                        "ops": ["param", "simplify", "numeric"], "batch": 5, "confirm_tries": 2})
    monkeypatch.setitem(E.EV, "primary_builder", "claude")
    monkeypatch.setitem(E.EV, "full_top_k", 3)
    monkeypatch.setitem(E.EV, "full_top_k_small", 1)
    monkeypatch.setitem(E.EV, "children_per_generation", 20)
    monkeypatch.setitem(E.EV, "micro", dict(E.EV["micro"], bases_small=1))


def _h(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def stub_runner(g, item, builder, stage):
    # quality depends on genes: a stepwise protocol and error hints help building; noise per item
    gn = g["genes"]
    base = 0.25 + 0.15 * (gn["build_protocol"]["kind"] == "stepwise_checked") + \
        0.05 * (gn["tool_interface"]["errors"] != "first_only") - 0.1 * (gn["core"]["kind"] == "prose")
    built = _h(g["id"], item, builder, "b") < 0.6 + base / 2
    score = max(0.0, min(1.0, base + 0.3 * (_h(g["id"], item, builder, "s") - 0.5))) if built else 0.0
    return {"item": item, "system": item.split("-")[0], "grade": {"status": "ok", "score": score},
            "validity": {"usable": built, "executable": built}, "rep": {"x": 1} if built else None,
            "output_tokens": 2000 + int(3000 * _h(g["id"], item)), "final_raw": "x"}


def stub_designer(text):
    # always proposes a stepwise protocol with fix hints, a plausible "repair"
    return json.dumps({"genes": {"build_protocol": {"kind": "stepwise_checked", "params": {}}}, "why": "stub"})


SETS = {"quick": [f"Q{i}" for i in range(5)], "screen": [f"S{i}-1" for i in range(30)],
        "full": [f"F{i}-1" for i in range(40)], "micro": [f"M{i}-1" for i in range(20)]}


def test_two_generation_dry_run(tmp_path, monkeypatch):
    monkeypatch.setitem(E.EV, "children_per_generation", 12)
    st = Store(tmp_path / "state.json")
    ev = E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    sd = seeds()
    ev.baseline(sd)
    d = st.data
    assert set(d["controls"]) == {sd["A0P"]["id"]} and d["front"]
    for gen in (1, 2):
        s = ev.generation(gen)
        assert s["children"] > 0 and s["quick_passed"] <= s["children"]
        assert len(s["full"]) <= E.EV["full_top_k"]
    # every non-seed genome declares 1-3 existing parents (D-065)
    for g in d["genomes"].values():
        if g["operator"] != "seed":
            assert 1 <= len(g["parents"]) <= 3
            assert all(p in d["genomes"] or p in {x.get("source") for x in d["library"].values()} for p in g["parents"])
    # the stub rewards stepwise protocols: evolution should find them and credit the operators that produce them
    best = max((gid for gid in d["front"]), key=lambda gid: ev.objectives(gid)["e2e"])
    assert d["genomes"][best]["genes"]["build_protocol"]["kind"] == "stepwise_checked"
    assert abs(sum(d["credit"]["shares"].values()) - 1) < 1e-9
    assert d["archive"] and all("|" in k for k in d["archive"])
    # improvement records cite parents and the best control
    imps = [d["evals"][g]["improvement"] for g in d["evals"] if "improvement" in d["evals"][g]]
    assert imps and all({c["kind"] for c in i["comparisons"]} >= {"best_control"} for i in imps)
    # determinism: same inputs, same lineage
    st2 = Store(tmp_path / "state2.json")
    ev2 = E.Evolution(st2, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    ev2.baseline(sd); ev2.generation(1); ev2.generation(2)
    assert [x["child"] for x in st2.data["lineage"]] == [x["child"] for x in d["lineage"]]


def test_crossover_and_graft_are_in_the_operator_mix():
    sh = E.EV["operator_shares"]
    assert {"swap", "graft"} <= set(sh) and abs(sum(sh.values()) - 1) < 1e-9
    assert set(sh) <= E.DETERMINISTIC | set(E.AI.ALLOWED)


def test_dry_run_produces_crossover_children(tmp_path, monkeypatch):
    monkeypatch.setitem(E.EV, "children_per_generation", 20)
    st = Store(tmp_path / "s.json")
    ev = E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    ev.baseline(seeds()); ev.generation(1)
    ops = {x["op"] for x in st.data["lineage"]}
    assert "swap" in ops and any(len(x["parents"]) == 2 for x in st.data["lineage"])


def test_final_selection_rule_and_sealing(tmp_path, monkeypatch):
    from v3 import final as F, items as I
    monkeypatch.setitem(E.EV, "children_per_generation", 12)
    st = Store(tmp_path / "s.json")
    ev = E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    ev.baseline(seeds()); ev.generation(1)
    fin = F.select(st.data)
    d = st.data
    assert 1 <= len(fin) <= 3 and not set(fin) & set(d["controls"]) and set(fin) <= set(d["front"])
    niches = [(d["genomes"][g]["genes"]["core"]["kind"], d["genomes"][g]["genes"]["build_protocol"]["kind"]) for g in fin]
    assert len(set(niches)) == len(niches)
    monkeypatch.setattr(I, "FINAL_LOCK", tmp_path / "FINAL_LOCK.json")
    import pytest
    with pytest.raises(I.Sealed):
        F.sealed_items()


def test_micro_level_feeds_winners_into_the_generation(tmp_path, monkeypatch):
    """V3-015: micro tweaks of the top designs, judged on the small builder against their parent on the quick items;
    winners join the generation's children with level "micro" and earn no operator credit."""
    monkeypatch.setitem(E.EV, "children_per_generation", 4)
    monkeypatch.setitem(E.EV, "micro", {"enabled": True, "bases": 2, "tweaks_per_base": 5,
                                        "ops": ["param", "simplify", "numeric"]})
    st = Store(tmp_path / "state.json")
    ev = E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    sd = seeds()
    ev.baseline({k: sd[k] for k in ("A0P", "ONT_BASIC", "ONT_LOOSE", "SEC_A1", "G1C3")})
    trials_before = sum((st.data.get("credit") or {}).get("trials", {}).values())
    s = ev.generation(1)
    m = s["micro"]
    assert m["tried"] > 0 and m["winners"] <= 2
    d = st.data
    micro_edges = [l for l in d["lineage"] if l.get("level") == "micro"]
    assert len(micro_edges) == m["winners"] and {l["child"] for l in micro_edges} == set(m["winner_ids"])
    for cid in m["winner_ids"]:
        g = d["genomes"][cid]
        assert g["micro"]["gain_a"] > 0 and g["micro"]["gain_b"] > 0
        assert not set(g["micro"]["items_a"]) & set(g["micro"]["items_b"])
        assert set(g["micro"]["items_a"]) <= set(SETS["micro"]) and not set(g["micro"]["items_a"]) & set(SETS["quick"])
        assert len(g["parents"]) == 1 and g["parents"][0] == g["micro"]["base"]
    macro = s["children"] - m["winners"]
    assert sum(d["credit"]["trials"].values()) - trials_before <= macro + len(d["credit"]["trials"]) * 2


class _Pause(Exception):
    pass


def _pausing_runner(stage_prefix, after):
    """stub_runner that raises (like state/PAUSE) on the `after`-th episode of a stage starting with stage_prefix."""
    n = {"k": 0}

    def run(g, item, builder, stage):
        if stage.startswith(stage_prefix):
            n["k"] += 1
            if n["k"] == after:
                raise _Pause(stage)
        return stub_runner(g, item, builder, stage)
    return run


def _resume_case(tmp_path, monkeypatch, stage_prefix, after):
    monkeypatch.setitem(E.EV, "children_per_generation", 6)
    monkeypatch.setitem(E.EV, "micro", {"enabled": True, "bases": 2, "tweaks_per_base": 4, "batch": 5,
                                        "confirm_tries": 2, "ops": ["param", "simplify", "numeric"]})
    sd = seeds()
    seed_set = {k: sd[k] for k in ("A0P", "ONT_BASIC", "ONT_LOOSE", "SEC_A1", "G1C3")}
    # reference: generation 1 without a pause
    ref = Store(tmp_path / "ref.json")
    ev = E.Evolution(ref, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    ev.baseline(seed_set); ev.generation(1)
    # paused once mid-generation, saved (as run.py does), then a fresh process resumes from the state file
    st = Store(tmp_path / "s.json")
    ev = E.Evolution(st, _pausing_runner(stage_prefix, after), SETS, designer=stub_designer, log=lambda *a: None)
    ev.baseline(seed_set)
    try:
        ev.generation(1)
        raise AssertionError("expected a pause")
    except _Pause:
        st.save()
    micro = lambda x: sorted(k for k, g in x["genomes"].items() if g.get("level") == "micro")
    before = micro(st.data)
    st2 = Store(tmp_path / "s.json")
    s = E.Evolution(st2, stub_runner, SETS, designer=stub_designer, log=lambda *a: None).generation(1)
    d, r = st2.data, ref.data
    assert set(before) <= set(micro(d)) and len(micro(d)) == len(micro(r))   # pre-pause tweaks kept, none extra
    kids = [l["child"] for l in d["lineage"]]
    assert len(kids) == len(set(kids)) == s["children"]                      # one set of children
    assert len(d["generations"]) == 1 and "pending" not in d
    return d, r


def test_resume_mid_micro_round_reuses_tweaks(tmp_path, monkeypatch):
    """V3-022: a pause during the micro round must not create new tweaks on resume."""
    _resume_case(tmp_path, monkeypatch, "micro1a", 12)


def test_resume_mid_screening_reuses_children(tmp_path, monkeypatch):
    """V3-022: a pause after breeding must not breed a second set of children on resume."""
    d, r = _resume_case(tmp_path, monkeypatch, "screen", 3)
    assert [l["child"] for l in d["lineage"]] == [l["child"] for l in r["lineage"]]   # the very same children


# ---------------------------------------------------------------- V3-023: two screening tracks
def _track_case(tmp_path, table):
    """Evolution whose evaluate() returns fixed screening scores per (genome, builder) from `table`."""
    st = Store(tmp_path / "s.json")
    ev = E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    base = seeds()["SEC_FLOW"]
    gs = []
    for gid in table:
        g = json.loads(json.dumps(base))
        g["id"] = gid
        st.data["genomes"][gid] = g
        gs.append(g)
    seen = []

    def fake(g, stage, items, builders):
        seen.append((g["id"], stage, tuple(builders)))
        return {b: {"e2e": table[g["id"]][b]} for b in builders}
    ev.evaluate = fake
    return ev, gs, seen


def test_screening_sends_the_small_builders_best_to_full_evaluation(tmp_path):
    # g_luna is weak on Claude (would be dropped by Claude-only screening) but best on the small builder
    table = {"g_a": {"claude": 0.95, "codex": 0.40}, "g_b": {"claude": 0.93, "codex": 0.42},
             "g_c": {"claude": 0.90, "codex": 0.41}, "g_luna": {"claude": 0.70, "codex": 0.60},
             "g_d": {"claude": 0.60, "codex": 0.30}}
    ev, gs, seen = _track_case(tmp_path, table)
    surv, hist, n1 = ev.screening(gs)
    assert surv == ["g_a", "g_b", "g_luna"] and len(surv) == E.EV["full_top_k"]
    assert set(hist) == {"claude", "codex"}
    # both builders screened every child in round 1
    assert {(g, b) for g, st_, bs in seen if st_ == f"screen{n1}" for b in bs} == \
        {(g, b) for g in table for b in ("claude", "codex")}


def test_screening_small_pick_skips_designs_claude_already_chose(tmp_path):
    table = {"g_a": {"claude": 0.95, "codex": 0.60}, "g_b": {"claude": 0.93, "codex": 0.30},
             "g_c": {"claude": 0.90, "codex": 0.50}, "g_d": {"claude": 0.60, "codex": 0.20}}
    ev, gs, _ = _track_case(tmp_path, table)
    surv, _, _ = ev.screening(gs)
    assert surv == ["g_a", "g_b", "g_c"]       # g_a tops both tracks, so the small pick is the next best, g_c


def test_screening_without_small_track_is_the_old_rule(tmp_path, monkeypatch):
    monkeypatch.setitem(E.EV, "full_top_k_small", 0)
    table = {"g_a": {"claude": 0.95, "codex": 0.1}, "g_b": {"claude": 0.93, "codex": 0.1},
             "g_c": {"claude": 0.90, "codex": 0.1}, "g_e": {"claude": 0.85, "codex": 0.1},
             "g_f": {"claude": 0.80, "codex": 0.1}, "g_luna": {"claude": 0.70, "codex": 0.9}}
    ev, gs, seen = _track_case(tmp_path, table)
    surv, _, _ = ev.screening(gs)
    assert surv == ["g_a", "g_b", "g_c"] and all(bs == ("claude",) for _, _, bs in seen)


def test_objectives_include_the_small_builder_and_improvement_is_per_builder(tmp_path, monkeypatch):
    monkeypatch.setitem(E.EV, "children_per_generation", 12)
    st = Store(tmp_path / "state.json")
    ev = E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    ev.baseline(seeds())
    ev.generation(1)
    d = st.data
    for gid in d["front"]:
        o = ev.objectives(gid)
        assert set(o) == set(E.KEYS) and o["e2e_small"] == (d["evals"][gid]["full"].get("codex") or {}).get("e2e")
    imps = [d["evals"][g]["improvement"] for g in d["evals"] if "improvement" in d["evals"][g]]
    assert imps
    for i in imps:
        assert set(i["by_builder"]) <= {"claude", "codex"}
        assert {c["builder"] for c in i["comparisons"]} == set(i["by_builder"])
        assert i["improved"] == (i["non_dominated"] and any(i["by_builder"].values()))
    # the screening history records both tracks
    assert set(d["generations"][-1]["halving"]) == {"claude", "codex"}


def test_final_selection_includes_the_small_builders_best(tmp_path):
    from v3 import final as F
    ev = {}
    def full(c, s):
        return {"full": {"claude": {"e2e": c, "build": 1.0}, "codex": {"e2e": s, "build": 1.0}}}
    G = {}
    for gid, core, proto, c, s in (("a", "prose", "free", 0.95, 0.40), ("b", "sectioned", "stepwise_checked", 0.93, 0.41),
                                   ("c", "equations", "free", 0.92, 0.42), ("l", "ontology", "free", 0.80, 0.60)):
        G[gid] = {"genes": {"core": {"kind": core}, "build_protocol": {"kind": proto}}}
        ev[gid] = full(c, s)
    fin = F.select({"evals": ev, "genomes": G, "front": list(G), "controls": []})
    assert len(fin) == 3 and "l" in fin and "a" in fin
    assert any(G[x]["genes"]["core"]["kind"] in F.STRUCTURED for x in fin)


def test_state_save_retries_a_briefly_locked_file(tmp_path, monkeypatch):
    """V3-024: a PermissionError from os.replace (file open elsewhere on Windows) is retried, not fatal."""
    import os as _os
    from v3 import state as S
    st = Store(tmp_path / "state.json")
    real, calls = _os.replace, []

    def flaky(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(5, "Access is denied")
        return real(a, b)
    monkeypatch.setattr(S.os, "replace", flaky)
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    st.data["x"] = 1
    st.save()
    assert len(calls) == 3 and json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["x"] == 1
    calls.clear()
    monkeypatch.setattr(S.os, "replace", lambda a, b: (_ for _ in ()).throw(PermissionError(5, "denied")))
    import pytest
    with pytest.raises(PermissionError):
        st.save()


def test_catch_up_runs_the_small_track_for_an_old_rule_generation(tmp_path, monkeypatch):
    """V3-023: generation 1 screened on Claude only gets its Luna track before generation 2 breeds, once."""
    monkeypatch.setitem(E.EV, "children_per_generation", 12)
    st = Store(tmp_path / "s.json")
    ev = E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    ev.baseline(seeds())
    monkeypatch.setitem(E.EV, "full_top_k_small", 0)
    s1 = ev.generation(1)                                   # the old rule: Claude-only screening
    d = st.data
    assert "codex" not in s1["halving"] and "catchup" not in d
    full_before = {g for g in d["evals"] if "full" in d["evals"][g]}
    monkeypatch.setitem(E.EV, "full_top_k_small", 1)
    ev.catch_up_small()
    cu = d["catchup"]["1"]
    kids = {x["child"] for x in d["lineage"] if x["gen"] == 1}
    assert len(cu["picks"]) <= 1 and set(cu["picks"]) <= kids and not set(cu["picks"]) & full_before
    for gid in cu["picks"]:
        assert set(d["evals"][gid]["full"]) == {"claude", "codex"} and "improvement" in d["evals"][gid]
    # the Luna ranking used only the small builder's screening scores
    assert all(set(d["evals"][g]["screen10"]) >= {"codex"} for g in cu["halving"][0]["scores"])
    n_events = len(d["events"])
    ev.catch_up_small()                                     # once only
    assert len(d["events"]) == n_events
    s2 = ev.generation(2)                                   # the new rule from here on
    assert "codex" in s2["halving"] and set(d["catchup"]) == {"1"}


def test_micro_bases_include_the_small_builders_best(tmp_path, monkeypatch):
    monkeypatch.setitem(E.EV, "micro", {"enabled": True, "bases": 3, "bases_small": 1, "tweaks_per_base": 1,
                                        "ops": ["param"]})
    st = Store(tmp_path / "s.json")
    ev = E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    ev.baseline(seeds())
    d = st.data
    objs = {g: ev.objectives(g) for g in d["genomes"] if ev.objectives(g) and E.small_ok(d["genomes"][g])}
    by_c = sorted(objs, key=lambda g: (-(objs[g]["e2e"] or 0), g))
    worst_c = by_c[-1]                                      # make the worst Claude design the best on Luna
    d["evals"][worst_c]["full"]["codex"]["e2e"] = 0.99
    bases = []
    real = ev.evaluate

    def spy(g, stage, items, builders):
        if stage.startswith("micro") and stage.endswith("a") and g["id"] in d["genomes"] and \
                g.get("level") != "micro" and g["id"] not in bases:
            bases.append(g["id"])
        return real(g, stage, items, builders)
    ev.evaluate = spy
    ev.micro(1, E.A.rng_for("micro", 1))
    assert bases == by_c[:2] + [worst_c]


def test_baseline_rerun_lists_each_seed_once_and_old_duplicates_are_removed(tmp_path, monkeypatch):
    """V3-026: rerunning generation 0's setup must not re-add seeds to the islands; a state that already has
    duplicates is cleaned when it is loaded."""
    st = Store(tmp_path / "state.json")
    ev = E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    ev.baseline(seeds())
    ev.baseline(seeds())
    flat = [x for isl in st.data["islands"] for x in isl]
    assert len(flat) == len(set(flat)) == len(seeds())
    st.data["islands"][0] = st.data["islands"][0] * 3
    E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    flat = [x for isl in st.data["islands"] for x in isl]
    assert len(flat) == len(set(flat)) and "duplicate island entries" in st.data["events"][-1]["msg"]


def test_tournament_counts_a_listed_twice_design_once_and_breaks_ties_by_score():
    """V3-026: duplicates do not raise a design's chance; equal rank and crowding go to the higher score, not the id."""
    import random
    from v3.selection import algorithms as A
    objs = {"a": {"e2e": 0.9, "build": 0.1}, "b": {"e2e": 0.1, "build": 0.9}}   # both on the front, both extremes
    wins = [A.tournament(["a", "b", "b", "b"], objs, E.KEYS[:2], random.Random(i)) for i in range(400)]
    assert all(w == "a" for w in wins)          # pool of two distinct designs: the tie always goes to the lower id
    wins = [A.tournament(["a", "b"], objs, E.KEYS[:2], random.Random(i), score=lambda i: objs[i]["build"])
            for i in range(50)]
    assert all(w == "b" for w in wins)


def test_reaching_full_evaluation_counts_as_an_operator_win(tmp_path, monkeypatch):
    """V3-027: every bred child that reaches full evaluation is a win for its operator, and a state credited under the
    old rule gets those wins once on load."""
    monkeypatch.setitem(E.EV, "children_per_generation", 12)
    st = Store(tmp_path / "state.json")
    ev = E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    ev.baseline(seeds())
    s = ev.generation(1)
    d = st.data
    bred = [gid for gid in s["full"] if d["genomes"][gid].get("level") != "micro"]
    assert sum(d["credit"]["wins"].values()) - len(d["credit"]["wins"]) >= len(bred)
    # old-rule state: no wins recorded; loading credits each fully evaluated bred child once
    d["credit"]["wins"] = {k: 1 for k in d["credit"]["wins"]}
    d.pop("credit_v3_027", None)
    E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    assert sum(d["credit"]["wins"].values()) - len(d["credit"]["wins"]) == len(bred)
    E.Evolution(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    assert sum(d["credit"]["wins"].values()) - len(d["credit"]["wins"]) == len(bred)


# ---------------------------------------------------------------- V3-030: run 2, GPT-6 Luna the only builder
def luna_only(monkeypatch):
    monkeypatch.setitem(E.EV, "primary_builder", "codex")
    monkeypatch.setitem(E.EV, "small_builder", "codex")
    monkeypatch.setitem(E.EV, "full_top_k", 6)
    monkeypatch.setitem(E.EV, "full_top_k_small", 0)
    monkeypatch.setitem(E.EV, "micro", dict(E.EV["micro"], bases_small=0))


def test_luna_only_never_calls_claude_and_selects_on_luna(tmp_path, monkeypatch):
    luna_only(monkeypatch)
    monkeypatch.setitem(E.EV, "children_per_generation", 14)
    calls = []

    def runner(g, item, builder, stage):
        calls.append(builder)
        return stub_runner(g, item, builder, stage)
    st = Store(tmp_path / "state.json")
    ev = E.Evolution(st, runner, SETS, designer=stub_designer, log=lambda *a: None)
    ev.baseline(seeds())
    for gen in (1, 2):
        s = ev.generation(gen)
        assert len(s["full"]) <= 6 and set(s["halving"]) == {"codex"}
    assert calls and set(calls) == {"codex"}
    d = st.data
    assert E.keys() == ["e2e", "build", "neg_tokens"]
    for gid in d["front"]:
        o = ev.objectives(gid)
        assert set(o) == set(E.keys()) and o["e2e"] == d["evals"][gid]["full"]["codex"]["e2e"]
        assert "claude" not in d["evals"][gid]["full"]
    imps = [d["evals"][g]["improvement"] for g in d["evals"] if "improvement" in d["evals"][g]]
    assert imps and all(set(i["by_builder"]) == {"codex"} for i in imps)
    assert "catchup" not in d or not d["catchup"]


def test_luna_only_screening_is_one_track_taking_the_top_six(tmp_path, monkeypatch):
    luna_only(monkeypatch)
    table = {f"g{i:02d}": {"claude": 0.99, "codex": i / 20} for i in range(14)}
    ev, gs, seen = _track_case(tmp_path, table)
    surv, hist, _ = ev.screening(gs)
    assert surv == ["g13", "g12", "g11", "g10", "g09", "g08"] and set(hist) == {"codex"}
    assert all(bs == ("codex",) for _, _, bs in seen)


def test_luna_only_drops_designs_luna_cannot_build(monkeypatch):
    luna_only(monkeypatch)
    g = json.loads(json.dumps(seeds()["SEC_FLOW"]))
    assert E.quick_builder(g) == "codex" and E.full_builders(g) == ["codex"]
    g["genes"]["core"]["kind"] = "simulation"
    assert E.quick_builder(g) is None and E.full_builders(g) == []


def test_luna_only_final_selection_ranks_on_luna(monkeypatch):
    from v3 import final as F
    luna_only(monkeypatch)
    g = lambda core: {"genes": {"core": {"kind": core}, "build_protocol": {"kind": "one_shot"}}}
    data = {"front": ["a", "b", "c"], "controls": [],
            "genomes": {"a": g("prose"), "b": g("sectioned"), "c": g("equations")},
            "evals": {"a": {"full": {"codex": {"e2e": .40, "build": 1}}},
                      "b": {"full": {"codex": {"e2e": .60, "build": 1}}},
                      "c": {"full": {"codex": {"e2e": .50, "build": 1}}}}}
    assert F.select(data) == ["b", "c", "a"]


# ---------------------------------------------------------------- V3-031: transient failures retried, voids kept
def test_transient_failures_are_retried_and_seed_voids_rerun(tmp_path, monkeypatch):
    luna_only(monkeypatch)
    monkeypatch.setitem(E.EV, "retry_wait_s", 0)
    fails, calls = {}, []

    def runner(g, item, builder, stage):
        calls.append((g["id"], item))
        k = (g["id"], item)
        if fails.get(k, 0) > 0:
            fails[k] -= 1
            return {"item": item, "system": item.split("-")[0], "error": "Reconnecting... 5/5 (unexpected status 403)",
                    "grade": {"status": "not_graded"}}
        return stub_runner(g, item, builder, stage)
    st = Store(tmp_path / "state.json")
    ev = E.Evolution(st, runner, SETS, designer=stub_designer, log=lambda *a: None)
    sd = {"A0P": seeds()["A0P"]}
    gid = sd["A0P"]["id"]
    fails.update({(gid, "F0-1"): 2, (gid, "F1-1"): 5})        # F0 recovers on the 2nd retry, F1 never does
    ev.baseline(sd)
    r = st.data["evals"][gid]["full"]["codex"]
    assert "F0-1" in r["scores"] and "F1-1" not in r["scores"] and E.voided(r) == 1
    assert calls.count((gid, "F1-1")) == 3                      # first try + 2 retries
    # a later baseline (the loop restarting) re-runs only the voided item; the rest come from their records
    fails[(gid, "F1-1")] = 0
    calls.clear()
    ev.baseline(sd)
    r = st.data["evals"][gid]["full"]["codex"]
    assert E.voided(r) == 0 and "F1-1" in r["scores"]


def test_integrity_and_allowlist_failures_are_not_retried():
    assert E.transient({"error": "rc=1: Reconnecting... 403 Forbidden"})
    assert not E.transient({"error": "INTEGRITY: unauthorized tool use"})
    assert not E.transient({"error": "design tools outside the approved Codex allowlist"})
    assert not E.transient({"error": "{'message': 'Invalid prompt: your prompt was flagged'}"})
    assert not E.transient({"error": None})


def test_a_failed_record_is_kept_when_the_episode_reruns(tmp_path, monkeypatch):
    from v3 import episode as EP
    monkeypatch.setattr(EP, "RUNS", tmp_path)
    monkeypatch.setattr(EP, "two_phase", lambda: False)
    g = seeds()["A0P"]
    item = {"id": "K1x1", "system": "K1x1", "question": "q", "context": "c"}
    d = EP.record_dir("p", "codex", g["id"], "K1x1")
    d.mkdir(parents=True)
    (d / "record.json").write_text(json.dumps({"error": "403", "grade": {"status": "not_graded"}}), encoding="utf-8")
    monkeypatch.setattr(EP, "_raw", lambda *a, **k: {"final_raw": "x", "representation": None})
    monkeypatch.setattr(EP, "grade", lambda *a, **k: {"status": "ok", "score": 1.0})
    monkeypatch.setattr("v3.genome.render.render", lambda g: None)
    r = EP.run(g, item, "codex", "p")
    assert r["grade"]["score"] == 1.0
    assert json.loads((d / "record.void0.json").read_text(encoding="utf-8"))["error"] == "403"
