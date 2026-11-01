"""Run 3 (V3-034): directed breeding, the growing proof, fresh confirmation, the new seeds and the finals' pieces.
Dry runs with stub builders and designer (no spending)."""
import hashlib
import json

import pytest

from v3 import directed as DR
from v3 import evolve as E
from v3.genome import schema as S
from v3.genome.seeds import load_all
from v3.operators import deterministic as D
from v3.selection import algorithms as A
from v3.state import Store

SEEDS3 = ("A0P", "SEC_A1", "ONT_LOOSE", "SEC_FLOW", "G1C3", "A5", "FACTS", "STATES", "SPEC")
# run 3's settings as locked for run 3 (the config holds run 4's since V3-037)
RUN3_DIRECTED = {
    "mix": {"fix": 0.5, "combine": 0.3, "explore": 0.2},
    "ops": {"fix": {"point": 0.3, "repair": 0.25, "guidance": 0.3, "big_jump": 0.15},
            "combine": {"transplant": 0.8, "swap": 0.2},
            "explore": {"simplify": 0.3, "param": 0.3, "numeric": 0.2, "graft": 0.2}},
    "reserve": 0.1, "parent_pool": 6, "tries_per_child": 4, "proof_levels": ["ext1", "ext2", "ext3"],
    "proof_best_always": True, "ext_bases": 30, "confirm_bases": 30, "confirm_per_round": 1, "auto_final": True}


@pytest.fixture(autouse=True)
def run3(monkeypatch):
    monkeypatch.setitem(E.EV, "mode", "directed")
    monkeypatch.setitem(E.EV, "primary_builder", "codex")
    monkeypatch.setitem(E.EV, "small_builder", "codex")
    monkeypatch.setitem(E.EV, "children_per_generation", 20)
    monkeypatch.setitem(E.EV, "full_top_k", 3)
    monkeypatch.setitem(E.EV, "stop_no_improve_generations", 4)
    monkeypatch.setitem(E.EV, "retry_wait_s", 0)
    monkeypatch.setitem(E.EV, "directed", dict(RUN3_DIRECTED, confirm_bases=4))


def _h(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def stub_runner(g, item, builder, stage):
    """Quality from the genes: guidance mentioning MAGIC is much better (the stub designer's fix), small noise."""
    gn = g["genes"]
    base = 0.35 + 0.4 * ("MAGIC" in gn["guidance"]["text"]) + 0.05 * (gn["core"]["kind"] == "sectioned")
    built = _h(g["id"], item, "b") < 0.9
    score = max(0.0, min(1.0, base + 0.1 * (_h(g["id"], item, "s") - 0.5))) if built else 0.0
    return {"item": item, "system": item, "grade": {"status": "ok", "score": score},
            "validity": {"usable": built, "executable": built, "holes": [] if built else [{"problem": "undeclared term 'x'"}]},
            "rep": {"x": 1} if built else None, "output_tokens": 1000 + int(2000 * _h(g["id"], item)), "final_raw": "x"}


def stub_designer(text):
    if "Rewrite ONLY the guidance" in text or "exactly ONE gene" in text:
        return json.dumps({"genes": {"guidance": {"text": "MAGIC: write every part and rule.", "examples": []}},
                           "why": "stub"})
    return "no json"


SETS = {"quick": [f"K1x{i}" for i in range(5)], "screen": [f"K2x{i}" for i in range(30)],
        "full": [f"K4x{i}" for i in range(40)], "ext1": [f"K8x{i}" for i in range(40)],
        "ext2": [f"K8x{i}" for i in range(40, 80)], "ext3": [f"K8x{i}" for i in range(80, 120)]}


def seeds3():
    s = load_all()
    return {k: s[k] for k in SEEDS3}


def make(tmp_path, name="s.json"):
    st = Store(tmp_path / name)
    ev = DR.Directed(st, stub_runner, SETS, designer=stub_designer, log=lambda *a: None)
    ev.baseline(seeds3())
    return st, ev


def test_config_is_run3():
    assert E.EV["mode"] == "directed" and E.EV["islands"] == 1 and not E.EV["micro"]["enabled"]
    assert E.EV["children_per_generation"] == 20      # patched here; the config says 36
    assert set(E.EV["seeds"]) == set(SEEDS3) and "ONT_BASIC" not in E.EV["seeds"]
    mix = E.EV["directed"]["mix"]
    assert mix == {"fix": 0.5, "combine": 0.3, "explore": 0.2}
    ops = E.EV["directed"]["ops"]
    assert set(ops["fix"]) <= set(E.AI.ALLOWED) and "transplant" in ops["combine"] and "graft" in ops["explore"]
    for kind in ops.values():
        assert abs(sum(kind.values()) - 1) < 1e-9


def test_new_seeds_are_valid_sectioned_designs_luna_can_build():
    from v3.builders.codex_builder import usable_for
    from v3.genome.render import render
    s = load_all()
    for name in ("FACTS", "STATES", "SPEC", "A5"):
        g = s[name]
        assert not S.incompatibilities(g) and g["operator"] == "seed" and not g["parents"]
        assert usable_for(render(g)), name
    assert s["FACTS"]["genes"]["core"]["params"]["granularity"] == "fine"
    assert "TRANSITIONS" in s["STATES"]["genes"]["core"]["params"]["headers"]
    assert "REQUIREMENTS" in s["SPEC"]["genes"]["core"]["params"]["headers"]
    assert "SHALL" in s["SPEC"]["genes"]["guidance"]["text"]


def test_plan_follows_the_mix_with_a_reserve(tmp_path):
    st, ev = make(tmp_path)
    plan = ev.plan(36)
    kinds = [k for k, _ in plan]
    assert kinds.count("fix") == 18 and kinds.count("combine") == 11 and kinds.count("explore") == 7
    res = [k for k, r in plan if r]
    assert len(res) == 4 and set(res) <= {"fix", "explore"}


def test_transplant_carries_the_proven_change():
    s = load_all()
    a, parent = s["SEC_A1"], s["SEC_FLOW"]
    b = json.loads(json.dumps(parent))
    b["genes"]["guidance"] = {"text": "better guidance", "examples": []}
    b = S.make(b["genes"], [parent["id"]], "guidance", 1)
    import random
    genes, pids, op, notes = D.transplant(a, b, random.Random(0), b_parent=parent)
    assert notes["genes"] == ["guidance"] and notes["key_change"] and pids == [a["id"], b["id"]]
    assert genes["guidance"]["text"] == "better guidance" and genes["core"] == a["genes"]["core"]
    # without a proven change: one differing gene
    genes, _, _, notes = D.transplant(a, b, random.Random(0))
    assert len(notes["genes"]) == 1 and not notes["key_change"]


def test_round_breeds_the_mix_and_credits_failed_attempts(tmp_path):
    st, ev = make(tmp_path)
    d = st.data
    assert d["champion"] in d["genomes"] and d["champions"][0]["round"] == 0
    s = ev.generation(1)
    assert s["children"] >= 15 and s["kinds"]["fix"] >= 1 and s["kinds"]["combine"] >= 1
    kids = [x for x in d["lineage"] if x["gen"] == 1]
    assert all(1 <= len(x["parents"]) <= 3 for x in kids)
    assert any(x["op"] in ("transplant", "swap") and len(x["parents"]) == 2 for x in kids)
    # the designer fails for big_jump/repair in this stub: those failed attempts count as tries
    c = d["credit"]
    assert set(c["by_kind"]) == {"fix", "combine", "explore"} and abs(sum(c["shares"].values()) - 1) < 1e-9
    assert c["trials"]["big_jump"] > 2 or c["trials"]["repair"] > 2
    # graft has nothing to graft from before the first new champion: never applied
    assert not any(x["op"] == "graft" for x in kids if d["library"] == {} or x["gen"] == 1)


def test_a_clearly_better_child_is_proven_confirmed_and_crowned(tmp_path):
    st, ev = make(tmp_path)
    d = st.data
    old = d["champion"]
    s = ev.generation(1)
    assert s["improved"], "the MAGIC guidance child should be proven and confirmed"
    new = d["champion"]
    assert new != old and "MAGIC" in d["genomes"][new]["genes"]["guidance"]["text"]
    proofs = d["proofs"][new]
    assert proofs[0]["verdict"] == "proven" and proofs[-1]["kind"] == "confirmation" and proofs[-1]["passed"]
    assert d["confirm_sets"] and len(d["confirm_sets"][0]) == 16            # 4 bases x 4 k (patched)
    assert d["streak"] == 0 and d["library"]["guidance"]["source"] == new
    assert d["champions"][-1] == {"gid": new, "round": 1, "how": "proven and confirmed", "previous": old}
    # confirmation sets never repeat
    s2 = ev.generation(2)
    allc = [i for c in d["confirm_sets"] for i in c]
    assert len(allc) == len(set(allc))


def test_no_gain_rounds_count_towards_the_stop(tmp_path, monkeypatch):
    st, ev = make(tmp_path)
    monkeypatch.setattr(ev, "designer", lambda text: "no json")          # nothing better can be bred
    for gen in (1, 2):
        ev.generation(gen)
    trailing = 0
    for g in st.data["generations"]:
        trailing = 0 if g["improved"] else trailing + 1
    assert st.data["streak"] == trailing and not ev.should_stop()
    st.data["streak"] = 4
    assert ev.should_stop()


def test_resume_reuses_children_and_the_pending_confirmation(tmp_path):
    st, ev = make(tmp_path)
    d = st.data
    rng = A.rng_for("gen", 1)
    kids = ev.breed(1, rng, ev.credit())
    for c in kids:
        d["genomes"][c["id"]] = c
    d["pending"] = {"gen": 1, "children": [c["id"] for c in kids]}
    s = ev.generation(1)
    assert s["children"] == len(kids)
    # a confirmation interrupted mid-way reuses its set
    gid = d["champion"]
    d["pending"] = {"gen": 2, "confirm": {"gid": "gX", "index": 7, "items": ["K1x1"]}}
    d["genomes"]["gX"] = d["genomes"][gid]
    n = len(d["confirm_sets"])
    ev.confirm("gX")
    assert len(d["confirm_sets"]) == n and d["proofs"]["gX"][-1]["set"] == "confirm7"


def test_failure_profile_has_no_item_text(tmp_path):
    st, ev = make(tmp_path)
    p = ev.failure_profile(st.data["champion"])
    assert set(p) >= {"construction", "reasoning", "answered_ok", "score_by_k", "build_rate", "common_build_problems"}
    assert all("'x'" not in m for m, _ in p["common_build_problems"])


def test_run3_item_sets_are_disjoint_and_the_confirmation_sets_fresh():
    sets = DR.run3_sets()
    assert [len(sets[k]) for k in ("quick", "screen", "full", "ext1", "ext2", "ext3")] == [5, 30, 120, 120, 120, 120]
    comp = [i for v in sets.values() for i in v if i.startswith("K")]
    bases = {DR.base_of(i) for v in sets.values() for i in v if i.startswith("K")}
    per_set = {k: {DR.base_of(i) for i in v if i.startswith("K")} for k, v in sets.items()}
    for a in per_set:
        for b in per_set:
            assert a == b or not per_set[a] & per_set[b], (a, b)
    assert len(comp) == len(set(comp)) and len(bases) == 145 + 90          # quick 5 + screen 30 + full 110 composite, ext 3 x 30
    c0 = DR.confirm_set([])
    c1 = DR.confirm_set([c0])
    assert not {DR.base_of(i) for i in c0} & {DR.base_of(i) for i in c1}
    from discovery.world.splits import split_of
    assert all(split_of(int(DR.base_of(i)[1:])) == "train" for i in c0 + c1)
    used = [int(DR.base_of(i)[1:]) for i in comp]
    assert min(used) >= 1_200_000 and max(used) < 1_800_000 and not [u for u in used if 1_500_000 <= u < 1_617_000]


def test_growth_turns_add_only_new_subsystems():
    from v3 import items as I
    t = I.growth_turns(1_960_000)
    assert [x["k"] for x in t] == [1, 2, 4, 8]
    assert [len(x["item"]["question_ids"]) for x in t] == [3, 6, 12, 24]
    assert t[1]["new_text"].startswith("NEW INFORMATION: 1 more") and "4 more separate systems" in t[3]["new_text"]
    first = t[0]["item"]["context"].split("\n\n")[1]
    assert first not in t[1]["new_text"]                     # earlier text is never repeated


def test_final3_rep_text_and_transfer_builders():
    from v3 import final3 as F
    from v3.episode import openrouter_model
    s = load_all()
    assert "No representation" in F.rep_text(s["SEC_A1"], None)
    assert F.rep_text(s["A0P"], {"notes": "N"}) == "N"
    from v3.episode import claude_model, transfer_builder
    models = F.transfer_models()
    assert len(models) == 5 and sorted(m for m in models if openrouter_model(m)) == ["or_gemma", "or_gptoss",
                                                                                    "or_mistral", "or_qwen"]
    assert claude_model("cl_haiku") == "haiku" and openrouter_model("cl_haiku") is None
    assert openrouter_model("codex") is None and not transfer_builder("codex") and transfer_builder("cl_haiku")
    assert F.base_of("G3x2700500") == "B2700500" and F.base_of("K8x2500000") == "B2500000"


def test_contrast_ci_reads_the_direction():
    from v3 import final3 as F
    diff = {f"K{k}x{b}": (0.3 if k == 8 else 0.0) + 0.01 * (b % 3) for b in range(10) for k in (1, 8)}
    cl = {i: F.base_of(i) for i in diff}
    r = F.contrast_ci(diff, cl, lambda i: i.startswith("K8"), lambda i: i.startswith("K1"), n_boot=500)
    assert abs(r["est"] - 0.3) < 1e-9 and r["lo"] > 0


def test_final_c_waits_for_v3_035(monkeypatch):
    """V3-034 addendum: final C is skipped (no episodes) until final.transfer_ready is set by V3-035."""
    from v3 import final3
    events = []

    class S:
        data = {}

        def event(self, m):
            events.append(m)

        def save(self):
            pass
    monkeypatch.setitem(final3.FC, "transfer_ready", False)
    monkeypatch.setattr(final3, "pool_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran episodes")))
    final3.run_c(S(), {})
    assert events and "skipped" in events[0]


def test_lenient_answers_only_for_final_c():
    """V3-035: the evolution keeps the strict ANSWERS reader; final C also reads FINAL ANSWER {..} and bare pairs."""
    from v3 import items as I
    assert I.parse_answers('FINAL ANSWER\n{"s1.p2": 3}') == {}
    assert I.parse_answers('FINAL ANSWER\n{"s1.p2": 3}', lenient=True) == {"s1.p2": 3}
    assert I.parse_answers("FINAL ANSWER: s1.p2: 2, s2.p6: 0", lenient=True) == {"s1.p2": 2, "s2.p6": 0}
    assert I.parse_answers("s1.p2: 2 (no marker)", lenient=True) == {}
    assert I.parse_answers('ANSWERS: {"s1.p2": 1}') == I.parse_answers('ANSWERS: {"s1.p2": 1}', lenient=True)


def test_transfer_settings_reach_the_client():
    import harness.llm as H
    import v3.llm  # noqa: F401
    assert H.MODEL_CFG["qwen/qwen3.8-flash"]["reasoning"] == {"enabled": False}
    assert "reasoning" not in H.MODEL_CFG["google/gemma-4-26b-a4b-it"]
    assert H.MODEL_CFG["openai/gpt-oss-20b"]["reasoning"] == {"effort": "low"}


def test_unusable_transfer_model_left_out_of_h9(monkeypatch):
    """V3-035: a model with paired records on fewer than half the items is reported unusable, not counted."""
    from v3 import final3 as F
    items = [f"K{k}x{b}" for b in range(2500000, 2500015) for k in (1, 2, 4, 8)]
    monkeypatch.setattr(F, "composite_test_items", lambda: items)

    def fake_scores(phase, m, gid):
        its = items[:5] if m == "or_mistral" else items
        x = {i: (0.7 if gid == "C" else 0.5) + 0.01 * (hash(i) % 5) for i in its}
        return x, {i: F.base_of(i) for i in its}, {}
    monkeypatch.setattr(F, "scores", fake_scores)
    r = F.h9({}, "C", "K")
    assert list(r["unusable"]) == ["or_mistral"] and len(r["per_model"]) == 4
    assert r["n_models"] == 4 and r["models_ahead"] == 4 and r["passed"] is True
    monkeypatch.setattr(F, "scores", lambda *a: ({}, {}, {}))
    assert F.h9({}, "C", "K")["status"] == "not run"


def test_v3036_proof_judges_on_fresh_items_only(tmp_path):
    """V3-036: the growing proof never uses the full set; its steps are ext1, then ext1+ext2, then all three."""
    st, ev = make(tmp_path)
    champ = st.data["champion"]
    other = next(g for g in st.data["genomes"] if g != champ)
    ev.evaluate(st.data["genomes"][other], "full", SETS["full"], ["codex"])
    rec = ev.prove(other)
    assert rec["steps"] and all(s["stage"] in ("ext1", "ext2", "ext3") for s in rec["steps"])
    assert rec["steps"][0]["n_items"] <= len(SETS["ext1"])


def test_v3036_best_child_meets_the_champion_even_below_its_full_score(tmp_path, monkeypatch):
    st, ev = make(tmp_path)
    champ = st.data["champion"]
    kids = [g for g in st.data["genomes"] if g != champ][:3]
    scores = {champ: 0.9, kids[0]: 0.5, kids[1]: 0.7, kids[2]: 0.6}
    monkeypatch.setattr(ev, "e2e", lambda g: scores.get(g))
    proved = []
    monkeypatch.setattr(ev, "prove", lambda g: proved.append(g) or {"verdict": "no_lead"})
    ev.judge(kids, 1)
    assert proved == [kids[1]]                     # only the round's best child, though below the champion
    monkeypatch.setitem(DR.cfg(), "proof_best_always", False)
    proved.clear()
    ev.judge(kids, 1)
    assert proved == []


def test_v3036_ext3_is_added_to_a_running_store_without_changing_other_sets(tmp_path):
    from v3 import run as R
    st = Store(tmp_path / "s.json")
    fixed = DR.run3_sets()
    st.data["item_sets"] = {k: v for k, v in fixed.items() if k != "ext3"}
    R.evolution(st)
    assert st.data["item_sets"]["ext3"] == fixed["ext3"]
    assert all(st.data["item_sets"][k] == fixed[k] for k in fixed)
