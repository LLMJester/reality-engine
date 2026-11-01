"""Tests for the genome renderer (spec §3, M2). Offline: no model calls, no network; fake clients only."""
import copy
import json

import pytest

from v3.genome import schema as S
from v3.genome import seeds
from v3.genome.render import (BUILD_ONLY, CORE_SPECS, QUERY_MAP, RenderedArm, build_template, core_arm_name,
                              parse_statements, render)
from tournament.arms import get_arm            # importable once v3.genome.render has put rpp-core on sys.path
from tournament.arms.base import Arm

SEEDS = seeds.load_all()

# minimal G1C3 documents (syntax from rpp-core/tournament/prompts/g1c3_guide.md and tests/test_g1c3.py)
G1C3_VALID = "time steps=3 dt=1\ntargets: x\nx : real = 1"
G1C3_INVALID = "targets: x\nx : real := y +"          # dangling operator
G1C3_TWO_ERRORS = "targets: x\nx : real := y +\nz := = 3"
SHOP = """time steps=7 dt=1
targets: revenue, stock
price : real EUR = 10
base_demand : real units/day = 100
price_slope : real units/day/EUR = 5
demand : real units/day := max(0, base_demand - price_slope * price)
revenue : real EUR/day := price * demand
stock : real units = 500 := max(0, stock[t-1] - demand * dt)
whatif dearer: do price = 12
"""


def variant(seed, **changes):
    """A child of a seed with some genes changed (gene: value, or 'gene.key': value)."""
    genes = copy.deepcopy(SEEDS[seed]["genes"])
    for k, v in changes.items():
        if "." in k:
            g, key = k.split(".", 1)
            genes[g][key] = v
        else:
            genes[k] = v
    return S.make(genes, [SEEDS[seed]["id"]], "test", 1)


def names(tools):
    return [t["function"]["name"] for t in tools]


def stepwise_g1c3(errors="first_only", check_every=1):
    g = variant("G1C3", build_protocol={"kind": "stepwise_checked", "params": {}}, **{"tool_interface.errors": errors})
    g["genes"]["numeric"]["check_every"] = check_every
    return render(g)


# ---------------------------------------------------------------------------- seeds
@pytest.mark.parametrize("name", sorted(SEEDS))
def test_every_seed_renders(name):
    arm = render(SEEDS[name])
    assert isinstance(arm, Arm) and isinstance(arm, RenderedArm)
    if SEEDS[name]["genes"]["core"]["kind"] == "prose":
        assert arm.tools() == [] and arm.custom_episode
    else:
        assert arm.tools()
    assert arm.limits()["max_turns"] == 12 and arm.limits()["retries"] == 2
    assert SEEDS[name]["genes"]["guidance"]["text"] in arm.guide()


@pytest.mark.parametrize("name", sorted(n for n in SEEDS if SEEDS[n]["genes"]["core"]["kind"] not in ("sectioned", "json", "markdown",
                                                                                                    "ontology")))
def test_seed_one_shot_passes_core_tools_through(name):          # v2-backed seeds (v3 cores: tests/test_cores.py)
    arm = render(SEEDS[name])
    core = get_arm(core_arm_name(SEEDS[name]))
    core_tools = core.tools()
    mut = set(core.mutating_tools)
    assert [t for t in arm.tools() if t["function"]["name"] in mut] == [t for t in core_tools
                                                                         if t["function"]["name"] in mut]
    assert arm.tools() == core_tools          # seeds reproduce the v2 tool list exactly
    assert arm.custom_episode == core.custom_episode
    assert arm.guide().endswith(core.guide().strip())


def test_seed_handle_matches_v2_core():
    arm, core = render(SEEDS["G1C3"]), get_arm("G1C3")
    s1, s2 = arm.new_state(), core.new_state()
    assert arm.handle(s1, "submit_graph", {"source": SHOP}) == core.handle(s2, "submit_graph", {"source": SHOP})
    assert arm.handle(s1, "intervene", {"do": {"price": 12}, "query": "stock"}) == \
        core.handle(s2, "intervene", {"do": {"price": 12}, "query": "stock"})
    assert arm.export(s1) == core.export(s2)


def test_incompatible_genome_raises():
    with pytest.raises(S.GenomeError):
        render(variant("A0P", build_protocol={"kind": "stepwise_checked", "params": {}}))
    with pytest.raises(S.GenomeError):
        render(variant("G1C1", query_layer=["explain"]))       # probabilistic core cannot explain
    with pytest.raises(S.GenomeError):
        render(variant("A3", **{"tool_interface.submit": "line_statements"}))


def test_core_kind_default_mapping_without_v2_arm():
    g = variant("A3")
    g["genes"]["core"] = {"kind": "simulation", "params": {}}
    arm = render(S.make(g["genes"], [SEEDS["A3"]["id"]], "test"))
    assert arm.core_name == "A3" and names(arm.tools()) == names(get_arm("A3").tools())


# ---------------------------------------------------------------------------- stepwise_checked
def test_stepwise_g1c3_offers_parts_and_drives_real_checker():
    arm = stepwise_g1c3()
    t = names(arm.tools())
    assert {"add_part", "check", "finish"} <= set(t) and "submit_graph" not in t
    st = arm.new_state()
    out = arm.handle(st, "add_part", {"section": "model", "text": G1C3_INVALID})
    assert "PROBLEM" in out and "unexpected end of expression" in out
    st = arm.new_state()
    out = arm.handle(st, "add_part", {"section": "model", "text": G1C3_VALID})
    assert "no problems found" in out
    fin = arm.handle(st, "finish", {})
    assert "graph submitted" in fin and "TARGETS: x: computed" in fin
    assert arm.validity(st)["executable"] and arm.export(st)["source"] == G1C3_VALID


def test_stepwise_parts_accumulate_and_check_every():
    arm = stepwise_g1c3(check_every=2)
    st = arm.new_state()
    a = arm.handle(st, "add_part", {"section": "header", "text": "time steps=3 dt=1\ntargets: y"})
    assert "Next automatic check" in a and "CHECK" not in a
    b = arm.handle(st, "add_part", {"section": "vars", "text": "x : real = 2\ny : real := 3 * x"})
    assert "CHECK: no problems found" in b
    c = arm.handle(st, "add_part", {"section": "bad", "text": "z : real := w + 1"})
    assert "CHECK" not in c
    chk = arm.handle(st, "check", {})
    assert "undefined name 'w'" in chk and "part 3 'bad' line 1" in chk
    rep = arm.handle(st, "add_part", {"section": "bad", "text": "z : real := x + 1", "replace": True})
    assert "CHECK: no problems found" in rep          # the replacement is the 2nd part since the last check
    assert "no problems found" in arm.handle(st, "check", {})
    assert "3 part(s) submitted" in arm.handle(st, "finish", {})


def test_stepwise_fenced_block_becomes_a_part():
    arm = stepwise_g1c3()
    st = arm.new_state()
    st.pending_block = G1C3_VALID            # what the G1C3 episode loop sets for a ```graph block
    out = arm.handle(st, "submit_graph", {})
    assert out.startswith("part 1 added") and "no problems found" in out


def test_stepwise_episode_with_fake_client():
    from tournament.run import run_episode
    arm = stepwise_g1c3()
    script = [{"role": "assistant", "content": "",
               "tool_calls": [_tc("add_part", {"section": "m", "text": G1C3_VALID}, "c1"),
                              _tc("finish", {}, "c2")]},
              {"role": "assistant", "content": "FINAL ANSWER x stays 1"}]
    client = FakeClient(script)
    rec = run_episode(arm, ITEM, "A thing.", "fake", client)
    assert rec["final_raw"].startswith("FINAL ANSWER")
    assert [t["tool"] for t in rec["tool_trace"]] == ["add_part", "finish"]
    assert not any(t["error"] for t in rec["tool_trace"])
    assert "add_part" in names(client.calls[0]["tools"]) and "submit_graph" not in names(client.calls[0]["tools"])
    assert "add_part(section, text)" in client.calls[0]["messages"][0]["content"]
    assert rec["rep"]["source"] == G1C3_VALID


def test_seed_episode_matches_v2_core_episode():
    from tournament.run import run_episode
    script = [{"role": "assistant", "content": "```graph\n" + SHOP + "```", "tool_calls": [_tc("submit_graph", {})]},
              {"role": "assistant", "content": "FINAL ANSWER stock falls"}]
    r1 = run_episode(render(SEEDS["G1C3"]), ITEM, "A shop.", "fake", FakeClient(script))
    r2 = run_episode(get_arm("G1C3"), ITEM, "A shop.", "fake", FakeClient(script))
    assert [t["result"] for t in r1["tool_trace"]] == [t["result"] for t in r2["tool_trace"]]
    assert r1["rep"] == r2["rep"] and r1["final_raw"] == r2["final_raw"]


def test_prose_seed_keeps_a0plus_episode():
    from tournament.run import run_episode
    script = [{"role": "assistant", "content": "It rises."}, {"role": "assistant", "content": "NO CHANGE"}]
    client = FakeClient(script)
    rec = run_episode(render(SEEDS["A0P"]), ITEM, "A tank.", "fake", client)
    assert rec["final_raw"] == "It rises." and rec["n_calls"] == 2 and client.calls[0]["tools"] is None


# ---------------------------------------------------------------------------- other protocols
def test_template_first():
    arm = render(variant("G1C3", build_protocol={"kind": "template_first", "params": {}}))
    st = arm.new_state()
    assert names(arm.current_tools(st)) == ["get_template"]
    tpl = arm.handle(st, "get_template", {})
    assert "```graph" in tpl and "targets:" in tpl and "time steps=" not in tpl   # no time_scale field
    for f in SEEDS["G1C3"]["genes"]["schema"]["fields"]:
        assert f in tpl
    assert set(names(get_arm("G1C3").tools())) <= set(names(arm.current_tools(st)))
    assert "get_template()" in arm.guide()
    for core in ("G1C1", "A3", "A5", "A2"):
        assert "TEMPLATE" in build_template(core, CORE_SPECS[core], ["variables", "parameters", "assumptions"])


def test_sketch_refine_shows_outline_with_checks():
    arm = render(variant("G1C3", build_protocol={"kind": "sketch_refine", "params": {}}))
    st = arm.new_state()
    assert {"sketch", "check", "submit_graph"} <= set(names(arm.tools()))
    arm.handle(st, "sketch", {"outline": "demand falls with price; stock drains"})
    out = arm.handle(st, "submit_graph", {"source": SHOP})
    assert "graph submitted" in out and "YOUR SKETCH" in out and "stock drains" in out
    assert "stock drains" in arm.handle(st, "check", {})
    assert "stock drains" not in arm.handle(st, "evaluate", {"query": "stock"})


def test_decompose_merges_subsystems():
    arm = render(variant("G1C3", build_protocol={"kind": "decompose", "params": {}}))
    assert {"add_subsystem", "finish"} <= set(names(arm.tools())) and "submit_graph" not in names(arm.tools())
    st = arm.new_state()
    arm.handle(st, "add_subsystem", {"name": "demand", "text": "time steps=3 dt=1\ntargets: d\np : real = 2\n"
                                                                "d : real := 10 - p"})
    arm.handle(st, "add_subsystem", {"name": "stock", "text": "s : real = 50 := s[t-1] - d"})
    out = arm.handle(st, "finish", {})
    assert "2 subsystem(s) submitted" in out and "UNDEFINED NAMES: 0" in out
    assert "# subsystem: stock" in arm.export(st)["source"]


def test_stepwise_g1c1_uses_static_analysis():
    g = variant("G1C1", build_protocol={"kind": "stepwise_checked", "params": {}},
                **{"tool_interface.errors": "all_at_once"})
    arm = render(g)
    st = arm.new_state()
    out = arm.handle(st, "add_part", {"section": "params", "text": "PARAMS = {\"a\": 1,\nHORIZON = 5"})
    assert "SyntaxError" in out and "line" in out
    st = arm.new_state()
    out = arm.handle(st, "add_part", {"section": "params", "text": "PARAMS = {\"a\": 1}\nRULES = {}\nHORIZON = 5"})
    assert "no class Model" in out                 # a hole until the Model part arrives
    out = arm.handle(st, "add_part", {"section": "model", "text": (
        "class Model:\n    def init_state(self, scenario, rng):\n        return {'x': scenario['a']}\n"
        "    def step(self, state, t, rng):\n        return state\n"
        "    def observe(self, state):\n        return dict(state)")})
    assert "no problems found" in out


def test_stepwise_json_core_a2():
    g = variant("A2", build_protocol={"kind": "stepwise_checked", "params": {}},
                **{"tool_interface.errors": "all_at_once"})
    arm = render(g)
    st = arm.new_state()
    bad = arm.handle(st, "add_part", {"section": "variables", "text": "[{\"name\": "})
    assert bad.startswith("ERROR") and "bad JSON" in bad
    arm.handle(st, "add_part", {"section": "variables", "text": json.dumps(
        [{"name": "a", "type": "real", "exogenous": True, "value": 1},
         {"name": "b", "type": "real", "exogenous": False}])})
    out = arm.handle(st, "add_part", {"section": "equations", "text": json.dumps({"b": "2 * a + c"})})
    assert "PROBLEM" in out and "c" in out
    arm.handle(st, "add_part", {"section": "equations", "text": json.dumps({"b": "2 * a"}), "replace": True})
    fin = arm.handle(st, "finish", {})
    assert not fin.startswith("ERROR") and arm.export(st)["equations"] == {"b": "2 * a"}


def test_line_statements_for_rules_core():
    arm = render(variant("A5", **{"tool_interface.submit": "line_statements"}))
    sub = next(t for t in arm.tools() if t["function"]["name"] == "submit_model")
    assert list(sub["function"]["parameters"]["properties"]) == ["statements"]
    model, probs = parse_statements('fluents: {"name": "level", "args": [], "type": "int", "range": [0, 9]}\n'
                                    '# comment\nsystem = "tank"')
    assert not probs and model == {"fluents": [{"name": "level", "args": [], "type": "int", "range": [0, 9]}],
                                   "system": "tank"}
    st = arm.new_state()
    bad = arm.handle(st, "submit_model", {"statements": "fluents {oops"})
    assert bad.startswith("ERROR") and "line 1" in bad
    ok = arm.handle(st, "submit_model", {"statements": 'fluents: {"name": "level", "args": [], "type": "int", '
                                                       '"range": [0, 9]}'})
    assert ok.startswith("Model stored") and arm.export(st)["fluents"][0]["name"] == "level"
    # a text core keeps its own text submission
    arm2 = render(variant("G1C3", **{"tool_interface.submit": "line_statements"}))
    assert arm2.tools() == get_arm("G1C3").tools()


# ---------------------------------------------------------------------------- tool interface
def test_error_styles_change_output_format():
    outs = {}
    for style in S.ERROR_STYLES:
        arm = stepwise_g1c3(errors=style)
        st = arm.new_state()
        outs[style] = arm.handle(st, "add_part", {"section": "m", "text": G1C3_TWO_ERRORS})
    f, a, h = outs["first_only"], outs["all_at_once"], outs["all_with_fix_hints"]
    assert "unexpected end" in f and "single '='" not in f and "1 more problem" in f
    assert "2 PROBLEM(S)" in a and "single '='" in a and "line 2" in a and "line 3" in a and "hint:" not in a
    assert "2 PROBLEM(S)" in h and h.count("hint:") == 2
    assert len({f, a, h}) == 3


def test_error_style_appends_problems_to_core_results():
    src = "time steps=3\ntargets: x\nx : real := y * 2"
    seed_out = render(SEEDS["G1C3"]).handle(get_arm("G1C3").new_state(), "submit_graph", {"source": src})
    arm = render(variant("G1C3", **{"tool_interface.errors": "all_with_fix_hints"}))
    out = arm.handle(arm.new_state(), "submit_graph", {"source": src})
    assert out.startswith(seed_out) and "PROBLEMS: 1 PROBLEM(S)" in out and "undefined name 'y'" in out \
        and "hint:" in out
    assert "PROBLEMS:" not in seed_out


@pytest.mark.parametrize("seed", ["G1C3", "G1C1", "A3", "A5", "A2"])
def test_patch_false_removes_patch_tools(seed):
    arm = render(variant(seed, **{"tool_interface.patch": False}))
    patch = set(CORE_SPECS[seed]["patch"])
    assert patch and not patch & set(names(arm.tools()))
    assert CORE_SPECS[seed]["submit"] in names(arm.tools())
    st = arm.new_state()
    assert arm.handle(st, sorted(patch)[0], {"old": "a", "new": "b"}).startswith("ERROR")
    assert "not available" in arm.guide()


def test_query_layer_filtering():
    full = names(render(SEEDS["G1C3"]).tools())
    only_whatif = names(render(variant("G1C3", query_layer=["what_if"])).tools())
    only_traj = names(render(variant("G1C3", query_layer=["trajectory"])).tools())
    none = names(render(variant("G1C3", query_layer=[])).tools())
    assert {"evaluate", "intervene"} <= set(full)
    assert "intervene" in only_whatif and "evaluate" not in only_whatif
    assert "evaluate" in only_traj and "intervene" not in only_traj
    assert "evaluate" not in none and "intervene" not in none and {"submit_graph", "show"} <= set(none)
    a3 = names(render(variant("A3", query_layer=[])).tools())
    assert "run_scenario" not in a3 and "show_model" in a3
    arm = render(variant("G1C3", query_layer=["what_if"]))
    assert arm.handle(arm.new_state(), "evaluate", {"query": "x"}).startswith("ERROR")
    assert set().union(*QUERY_MAP.values()) <= set(S.QUERIES)


# ---------------------------------------------------------------------------- robustness and state
GARBAGE = [("no_such_tool", {}), ("", {}), (None, {}), ("submit_graph", "not json {"), ("submit_graph", [1, 2]),
           ("add_part", {"section": None, "text": 5}), ("finish", {}), ("check", {"x": object()}),
           ("intervene", {"do": "nonsense"}), ("evaluate", {"query": {"targets": 7}}), ("show", None)]


@pytest.mark.parametrize("seed", sorted(SEEDS) + ["stepwise"])
def test_handle_never_raises(seed):
    arm = stepwise_g1c3() if seed == "stepwise" else render(SEEDS[seed])
    st = arm.new_state()
    for name, args in GARBAGE:
        out = arm.handle(st, name, args)
        assert isinstance(out, str)
    assert arm.handle(st, "no_such_tool", {}).startswith("ERROR")
    assert arm.handle(st, "add_part", "{bad json").startswith("ERROR")
    first = next(iter(names(arm.tools())), "x")
    assert arm.handle(st, first, [1, 2, 3]).startswith("ERROR")


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_export_load_round_trip_seeds(seed):
    arm = render(SEEDS[seed])
    st = arm.new_state()
    if seed == "G1C3":
        arm.handle(st, "submit_graph", {"source": SHOP})
    elif seed == "A3":
        arm.handle(st, "write_model", {"code": "class Model:\n    def init_state(self, s):\n        return {}\n"})
    rep = arm.export(st)
    assert arm.export(arm.load(copy.deepcopy(rep))) == rep


def test_export_load_round_trip_with_draft():
    arm = stepwise_g1c3()
    st = arm.new_state()
    arm.handle(st, "add_part", {"section": "m", "text": G1C3_VALID})
    arm.handle(st, "finish", {})
    arm.handle(st, "add_part", {"section": "more", "text": "y : real := x + 1"})
    rep = arm.export(st)
    assert rep["_v3"]["parts"][1]["section"] == "more"
    st2 = arm.load(copy.deepcopy(rep))
    assert arm.export(st2) == rep
    assert "2 part(s) submitted" in arm.handle(st2, "finish", {})
    assert arm.export(st2)["source"].endswith("y : real := x + 1")
    assert "_v3" not in arm.strip_answers(rep)


def test_gold_mode_hides_build_tools():
    arm = stepwise_g1c3()
    st = arm.load({"source": G1C3_VALID}, given_gold=True)
    gt = names(arm.gold_tools(st))
    assert not set(gt) & set(BUILD_ONLY) and "evaluate" in gt
    assert arm.handle(st, "add_part", {"section": "m", "text": "x : real = 2"}).startswith("ERROR")


# ---------------------------------------------------------------------------- fakes
ITEM = {"id": "T1", "system_id": "T", "bank": "synthetic", "question": "What is x at the end?",
        "task_type": "prediction", "answer_key": {"key_points": []}}


def _tc(name, args, cid=None):
    return {"id": cid or f"call_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class FakeClient:
    """harness.llm.chat signature; scripted responses; no network."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def __call__(self, model, messages, *, purpose, seed=0, max_tokens=8000, tools=None, response_format=None, **kw):
        n = len(self.calls)
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        m = self.script[min(n, len(self.script) - 1)]
        return dict(m), {"prompt_tokens": 10, "completion_tokens": min(500, max_tokens), "cost": 0.0,
                         "wall_s": 0.0, "provider": "fake", "finish_reason": "stop"}
