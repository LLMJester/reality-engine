"""Run 4 (V3-037): honest screening, report cards, smart combining, the shelf, family spots. Dry runs with stub builders
and designer (no spending)."""
import collections
import hashlib
import json

import pytest

from v3 import directed4 as D4
from v3 import evolve as E
from v3 import run as R
from v3.genome import schema as S
from v3.genome.seeds import load_all
from v3.operators import ai as AI
from v3.state import Store

SEEDS = ("A0P", "SEC_A1", "ONT_LOOSE", "SEC_FLOW", "G1C3", "A5", "FACTS", "STATES", "SPEC")


@pytest.fixture(autouse=True)
def run4(monkeypatch):
    monkeypatch.setitem(E.EV, "primary_builder", "codex")
    monkeypatch.setitem(E.EV, "small_builder", "codex")
    monkeypatch.setitem(E.EV, "retry_wait_s", 0)
    monkeypatch.setitem(E.EV, "directed", dict(E.EV["directed"], confirm_bases=4, parallel_children=2))


def _h(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def stub_runner(g, item, builder, stage):
    """Quality from the genes: guidance with MAGIC is much better, BAD much worse; small per-item noise."""
    gn = g["genes"]
    text = gn["guidance"]["text"]
    base = 0.45 + 0.3 * ("MAGIC" in text) - 0.35 * ("BAD" in text) + 0.05 * (gn["core"]["kind"] == "sectioned")
    built = _h(g["id"], item, "b") < 0.95
    score = max(0.0, min(1.0, base + 0.1 * (_h(g["id"], item, "s") - 0.5))) if built else 0.0
    return {"item": item, "system": item, "grade": {"status": "ok", "score": score},
            "validity": {"usable": built, "executable": built, "holes": []},
            "representation": "THINGS\n- s1.p1: 3" if built else None, "output_tokens": 1000, "final_raw": "x"}


CALLS = []


def stub_designer(text):
    CALLS.append(text)
    if "\"pick\"" in text and "THREE parent" in text:
        return json.dumps({"pick": {g: "ABC"[i % 3] for i, g in enumerate(S.GENES)},
                           "why": {g: "stub strength" for g in S.GENES}, "summary": "stub three-way mix"})
    if "\"pick\"" in text:
        return json.dumps({"pick": {g: ("A" if i % 2 else "B") for i, g in enumerate(S.GENES)},
                           "why": {g: "stub strength" for g in S.GENES}, "summary": "stub mix"})
    if "USES its representation" in text:
        return json.dumps({"genes": {"answering": {"text": "MAGIC: find the subsystems a question touches first."}},
                           "why": "stub"})
    if "WRONG ANSWERS" in text or "Rewrite ONLY the guidance" in text:
        return json.dumps({"genes": {"guidance": {"text": "MAGIC: list the subsystems, then trace links.",
                                                  "examples": []}}, "why": "stub"})
    return "no json"


def k_ids(prefix_n, start=0):
    seg = []
    for i in range(prefix_n):
        seg.append(f"K{(1, 2, 4, 8)[i % 4]}x{start + i * 500}")
    return seg


SETS = {"main": k_ids(120), "practice": ["K4x300000", "K8x300000"],
        "ext1": k_ids(40, 1_000_000), "ext2": k_ids(40, 2_000_000), "ext3": k_ids(40, 3_000_000)}
SETS.update(quick=SETS["main"][:5], screen=SETS["main"][5:65], full=SETS["main"])


def make(tmp_path):
    st = Store(tmp_path / "s.json")
    ev = D4.Directed4(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    s = load_all()
    ev.baseline({k: s[k] for k in SEEDS})
    return st, ev


def test_config_is_run4():
    assert E.EV["mode"] == "directed4"
    assert E.EV["children_per_generation"] == 20 and E.EV["full_top_k"] == 6
    assert E.EV["stop_no_improve_generations"] == 8 and E.EV["max_generations"] == 20
    d = E.EV["directed"]
    assert d["mix"] == {"fix": 0.3, "combine": 0.35, "big": 0.35} and d["reserve"] == 0.2
    for kind in d["ops"].values():
        assert abs(sum(kind.values()) - 1) < 1e-9
    assert "smart_combine" in d["ops"]["combine"] and "answer_method" in d["ops"]["big"]
    known = set(AI.ALLOWED) | {"smart_combine", "transplant", "swap", "simplify", "param", "numeric", "graft"}
    assert {op for kind in d["ops"].values() for op in kind} <= known


def test_every_stage_of_the_main_set_is_spread_over_system_size():
    s = D4.run4_sets()
    assert len(s["main"]) == 120 and s["full"] == s["main"] and s["quick"] == s["main"][:5]
    for n in (5, 10, 35, 65):
        c = collections.Counter(D4.k_of(i) for i in s["main"][:n])
        assert set(c) == {1, 2, 4, 8}, (n, c)                  # run 3's first 10 held only k = 1 and 2
        assert max(c.values()) - min(c.values()) <= max(2, n // 10)
    flat = [i for k in ("main", "ext1", "ext2", "ext3", "practice") for i in s[k]]
    assert len(flat) == len(set(flat))
    assert not set(s["practice"]) & set(s["main"] + s["ext1"] + s["ext2"] + s["ext3"])


def test_quick_test_is_pass_fail_and_scores_add_up(tmp_path):
    st, ev = make(tmp_path)
    g = load_all()["SEC_A1"]
    g = S.make(dict(g["genes"], guidance={"text": "MAGIC child", "examples": []}), [g["id"]], "point", 1)
    st.data["genomes"][g["id"]] = g
    assert ev.quick_pass(g)
    ev.run_main(g["id"], "screen35", 35)
    ev.run_main(g["id"], "screen65", 65)
    ev_ = st.data["evals"][g["id"]]
    assert ev_["quick"]["codex"]["n"] == 5 and ev_["screen35"]["codex"]["n"] == 35 and ev_["screen65"]["codex"]["n"] == 65
    assert set(ev_["screen35"]["codex"]["scores"]) == set(SETS["main"][:35])     # the quick items count too


def test_screening_cuts_only_clear_losers_and_keeps_a_family_slot(tmp_path):
    st, ev = make(tmp_path)
    champ = st.data["genomes"][st.data["champion"]]
    kids = []
    for i, word in enumerate(["BAD", "BAD", "", "", "MAGIC", "MAGIC"] * 2):
        g = S.make(dict(champ["genes"], guidance={"text": f"{word} variant {i}", "examples": []}), [champ["id"]],
                   "point", 1)
        st.data["genomes"][g["id"]] = g
        kids.append(g)
    surv, hist, n1 = ev.screening(kids)
    cut = set(hist[0]["cut"]) | set(hist[1]["cut"])
    bad = {k["id"] for k in kids if "BAD" in k["genes"]["guidance"]["text"]}
    assert bad <= cut and not (cut - bad)                       # the clearly worse go, the close ones stay
    assert len(surv) <= E.EV["full_top_k"] and n1 == 35
    for gid in surv:
        assert st.data["evals"][gid]["full"]["codex"]["n"] == 120
    assert all(st.data["evals"][g]["screen65"]["codex"]["n"] == 65 for g in hist[1]["kept"])


def test_report_card_shows_practice_wrong_answers_not_scoring_items(tmp_path):
    st, ev = make(tmp_path)
    gid = st.data["champion"]
    rc = ev.report_card(gid)["report_card"]
    assert rc["builds_are_fine"] in (True, False) and rc["score_by_size"]
    assert rc["wrong_answers"] and all({"question", "builder_answered", "correct"} <= set(w) for w in rc["wrong_answers"])
    assert rc["its_representation_of_one_k8_practice_system"]
    assert set(rc["practice_accuracy_by_size"]) == {"k=4", "k=8"}
    assert gid in st.data["practice_reports"]                  # run once per design
    text = AI.prompt("point", [st.data["genomes"][gid]], ev.report_card(gid), [])
    assert "REPORT CARD" in text and "fails mostly at CONSTRUCTION" not in text and "WRONG ANSWERS" in text


def test_smart_combine_takes_whole_genes_with_a_reason_each(tmp_path):
    st, ev = make(tmp_path)
    G = st.data["genomes"]
    a, b = G[st.data["champion"]], next(G[g] for g in ev.leaders() if g != st.data["champion"]
                                        and G[g]["genes"]["core"]["kind"] == G[st.data["champion"]]["genes"]["core"]["kind"])
    c = AI.smart_combine(a, b, ev.strengths_card(a["id"]), ev.strengths_card(b["id"]), ev.gene_stats(a["id"], b["id"]),
                         [], call=stub_designer, generation=1)
    if c is not None:                                          # incompatible picks are refused, never patched
        for gname, side in c["op_notes"]["pick"].items():
            assert c["genes"][gname] == (a if side == "A" else b)["genes"][gname]
        assert all(c["op_notes"]["why"].values()) and c["parents"] == [a["id"], b["id"]]
    one_side = lambda t: json.dumps({"pick": {g: "A" for g in S.GENES}, "why": {}})
    assert AI.smart_combine(a, b, {}, {}, {}, [], call=one_side) is None


def test_answer_method_may_change_only_answering_genes():
    g = load_all()["FACTS"]
    bad = lambda t: json.dumps({"genes": {"core": {"kind": "prose", "params": {}}}, "why": "x"})
    assert AI.apply("answer_method", [g], {"report_card": {}}, [], call=bad) is None
    writing = lambda t: json.dumps({"genes": {"guidance": {"text": "When answering: list subsystems first.",
                                                           "examples": []}}, "why": "x"})
    assert AI.apply("answer_method", [g], {"report_card": {}}, [], call=writing) is None    # V3-038: not the notes
    good = lambda t: json.dumps({"genes": {"answering": {"text": "List the subsystems a question touches first."}},
                                 "why": "x"})
    c = AI.apply("answer_method", [g], {"report_card": {}}, [], call=good)
    assert c is not None and c["genes"]["guidance"] == g["genes"]["guidance"]


def test_guidance_split_keeps_ids_and_builder_prompts():
    """V3-038: guidance / answering / examples are separate genes; old designs keep their ids and render the same."""
    from v3.genome.render import render
    legacy = json.loads(json.dumps(load_all()["SEC_A1"]))
    legacy["genes"] = S.legacy(legacy["genes"])
    legacy["genes"]["guidance"]["examples"] = ["THINGS\n- pump P1: on"]
    assert set(legacy["genes"]) == set(S.LEGACY_GENES)
    up = S.make(legacy["genes"], [], "seed")
    assert up["genes"]["examples"]["items"] == ["THINGS\n- pump P1: on"] and up["genes"]["guidance"]["examples"] == []
    assert up["genes"]["answering"] == {"text": ""} and up["id"] == S.genome_id({"genes": legacy["genes"]})
    assert render(up).guide() == render(dict(legacy, id=up["id"])).guide()        # same prompt, same id
    with_ans = S.make(dict(up["genes"], answering={"text": "Trace links first."}), [up["id"]], "answer_method", 1)
    assert with_ans["id"] != up["id"] and "Trace links first." in render(with_ans).guide()
    for k, v in (("answering", {"text": 3}), ("examples", {"items": "x"})):
        with pytest.raises(S.GenomeError):
            S.make(dict(up["genes"], **{k: v}), [up["id"]], "point", 1)
    c = R.carryover_seeds(E.EV["directed"]["carryover_dropped"])
    assert c and all(g["id"] == g["carryover"]["run3_id"] for g in c.values())   # run 3 ids survive the split


def test_combines_need_a_gene_that_matters_from_every_parent(tmp_path):
    st, ev = make(tmp_path)
    G = st.data["genomes"]
    a, b = load_all()["SEC_A1"], load_all()["ONT_LOOSE"]
    only_schema = {g: ("B" if g == "schema" else "A") for g in S.GENES}     # B gives only a gene sectioned ignores
    call = lambda t: json.dumps({"pick": only_schema, "why": {}})
    assert AI.smart_combine(a, b, {}, {}, {}, [], call=call) is None
    split = {g: ("B" if g in ("answering", "examples") else "A") for g in S.GENES}
    b2 = S.make(dict(a["genes"], answering={"text": "Answer per subsystem."},
                     guidance={"text": "Other notes.", "examples": []}), [a["id"]], "answer_method", 1)
    c = AI.smart_combine(a, b2, {}, {}, {}, [], call=lambda t: json.dumps({"pick": split, "why": {}}))
    assert c is not None and c["genes"]["answering"]["text"] == "Answer per subsystem." \
        and c["genes"]["guidance"] == a["genes"]["guidance"]


def test_repair_becomes_point_when_builds_already_pass(tmp_path):
    st, ev = make(tmp_path)
    CALLS.clear()
    import random
    ev.breed_one("fix", False, "repair", random.Random(1), 1)
    assert CALLS and all("WRONG ANSWERS" in c for c in CALLS)


def test_a_round_runs_end_to_end_and_reports_families_and_shelf(tmp_path):
    st, ev = make(tmp_path)
    s = ev.generation(1)
    assert s["children"] >= 15 and "family_shares" in s and "shelf" in s
    kinds = collections.Counter(st.data["genomes"][g].get("kind") for g in
                                [x["child"] for x in st.data["lineage"] if x["gen"] == 1])
    assert kinds["fix"] and kinds["combine"] and kinds["big"]
    res = sum(1 for x in st.data["lineage"] if x["gen"] == 1 and x["reserve"])
    assert res >= 2
    ops = collections.Counter(st.data["genomes"][x["child"]]["operator"] for x in st.data["lineage"] if x["gen"] == 1)
    assert ops["smart_combine3"] <= E.EV["directed"]["three_parent"]          # a clashing trio falls back to 2 parents
    assert st.data["generations"][-1]["halving"][0]["stage"] == "screen35"


def test_carryover_seeds_keep_run3_ids_and_families(monkeypatch):
    assert R.carryover_seeds() == {}                       # V3-039: dropped from run 4; the file stays on record
    s = R.carryover_seeds(E.EV["directed"]["carryover_dropped"])
    assert {g["carryover"]["run3_name"] for g in s.values()} == {"Stormy Marten", "Lanky Finch", "Little Marten",
                                                                  "Lucky Beetle"}
    facts = load_all()["FACTS"]["id"]
    assert sum(g["family"] == facts for g in s.values()) == 3
    assert all(not g["parents"] for g in s.values())


def test_pilot_sets_are_small_spread_and_disjoint():
    s = D4.pilot4_sets()
    assert len(s["main"]) == 16 and len(s["practice"]) == 2
    assert {D4.k_of(i) for i in s["main"][:5]} == {1, 2, 4, 8}
    flat = [i for k in ("main", "ext1", "ext2", "ext3", "practice") for i in s[k]]
    assert len(flat) == len(set(flat))
    main4 = set(D4.run4_sets()["main"])
    assert not set(flat) & main4


def test_dashboard_shows_run4_steps_mid_round(tmp_path):
    from v3.dashboard import feed
    st, ev = make(tmp_path)
    ev.generation(1)
    data = st.data
    summ = data["generations"].pop()
    data["generation"] = 0
    data["pending"] = {"gen": 1, "children": [x["child"] for x in data["lineage"] if x["gen"] == 1],
                       "halving4": summ["halving"]}
    stages = feed.pipeline(data)
    stages = stages["stages"] if isinstance(stages, dict) else stages
    keys = [s["key"] for s in stages]
    assert keys[:5] == ["breed", "quick", "screen1", "screen2", "full"] and "proof" in keys
    s1 = next(s for s in stages if s["key"] == "screen1")
    assert "35 items" in s1["label"] and s1["review"]["rows"]
    assert all(r[-1] in ("kept", "cut (clearly behind)") for r in s1["review"]["rows"])
    breed = next(s for s in stages if s["key"] == "breed")
    assert "big ideas" in breed["detail"]


def test_three_parent_combine_uses_three_families_and_every_parent(tmp_path):
    st, ev = make(tmp_path)
    import random
    G = st.data["genomes"]
    ids = ev.trio(random.Random(3))
    assert len(ids) == 3 and len({ev.family(g) for g in ids}) == 3
    c = AI.smart_combine3([G[g] for g in ids], [ev.strengths_card(g) for g in ids], ev.gene_stats3(ids), [],
                          call=stub_designer, generation=1)
    if c is not None:                                          # incompatible picks are refused, never patched
        assert c["parents"] == ids and c["operator"] == "smart_combine3" and len(c["parents"]) == 3
        assert set(c["op_notes"]["pick"].values()) == {"A", "B", "C"}
        assert {c["op_notes"]["pick"][k] for k in S.live_genes(c["genes"])} == {"A", "B", "C"}
        for gname, side in c["op_notes"]["pick"].items():
            assert c["genes"][gname] == G[ids["ABC".index(side)]]["genes"][gname]
    two_sides = lambda t: json.dumps({"pick": {g: "AB"[i % 2] for i, g in enumerate(S.GENES)}, "why": {}})
    assert AI.smart_combine3([G[g] for g in ids], [{}] * 3, {}, [], call=two_sides) is None
    assert E.EV["directed"]["three_parent"] == 2


def test_champion_gets_a_fix_and_an_answer_method_child_from_round_two(tmp_path):
    st, ev = make(tmp_path)
    import random
    champ = st.data["champion"]
    kids = ev.breed(1, random.Random(5), ev.credit())
    kids2 = ev.breed(2, random.Random(5), ev.credit())
    from_champ = [c for c in kids2 if c["parents"] == [champ]]
    assert any(c["operator"] == "answer_method" for c in from_champ)                # V3-040, round 2 on
    assert any(c["operator"] in ("point", "guidance", "repair") for c in from_champ)
    assert E.EV["directed"]["champion_children_from_round"] == 2 and kids

