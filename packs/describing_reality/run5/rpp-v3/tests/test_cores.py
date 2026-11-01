"""Tests for the v3 plain-language cores (v3.cores): structured prose ("sectioned") and plain-text ontology.
Offline: no model calls; fake clients and the in-process MCP session only."""
import copy
import io
import json
import random

import pytest

from v3.builders import mcp_server as M
from v3.cores import ontology as O
from v3.cores import sectioned as SEC
from v3.episode import buildable
from v3.genome import schema as S
from v3.genome.render import render
from v3.genome.seeds import build as build_seeds, load_all
from v3.operators import deterministic as D

SEEDS = build_seeds()
A1_HEADERS = ["ENTITIES", "STATE", "MECHANISMS", "CAUSAL LINKS", "TIMING", "UNCERTAINTY"]

SEC_VALID = """## ENTITIES
- E1 Rain tank, 500 litres [stated: "a 500-litre tank"]
- E2 Four garden beds [stated: "four beds"]

## STATE
- S1 Tank level 300 litres at day 0 [stated: "holds 300 litres"]

## MECHANISMS
- M1 Each dry day each bed uses 20 litres.
  Rainy days need no watering. [stated: "20 litres per bed"]

## CAUSAL LINKS
1. Rain increases the tank level [derived: from M1]

## TIMING
- W1 One step is one day [assumed: not given]

## UNCERTAINTY
- U1 Rain inflow 30-100 litres per rainy day [assumed: no figure]
"""

ONT_PARAMS = copy.deepcopy(S.ONTOLOGY_DEFAULT)
ONT_VALID = """## THINGS
pump: moves coolant
coolant flow
reactor
temperature
alarm
power supply
## HIERARCHY
centrifugal pump is a kind of pump.
pump is a kind of machine.
pump is part of cooling system.
cooling system is part of plant.
## RELATIONSHIPS
pump drives coolant flow.
coolant flow decreases temperature, after 2 minutes, when reactor is running.
machine requires power supply.
## RULES
If temperature has value above 90 then alarm is on.
## PROPERTIES
pump has flow rate 5 litres per second.
"""


def sec_check(text, **kw):
    p = dict(headers=A1_HEADERS, tags=True, granularity="coarse")
    p.update(kw)
    return SEC.check(text, **p)


def only(probs, fragment):
    hits = [p for p in probs if fragment in p["msg"]]
    assert hits, f"no problem containing {fragment!r} in {[p['msg'] for p in probs]}"
    return hits[0]


def variant(seed, **changes):
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


# ---------------------------------------------------------------------------- sectioned checker
def test_sectioned_accepts_valid_document():
    assert sec_check(SEC_VALID) == []


@pytest.mark.parametrize("edit, fragment, line", [
    (lambda t: t.replace("## TIMING", "## OUTCOME"), "unknown section header 'OUTCOME'", 15),
    (lambda t: "intro text\n" + t, "text before the first header", 1),
    (lambda t: "- stray [assumed: x]\n" + t, "element before the first header", 1),
    (lambda t: t.replace("- S1 Tank", "S1 Tank"), "line is not an element", 6),
    (lambda t: t.replace(' [stated: "four beds"]', ""), "no provenance tag", 3),
    (lambda t: t.replace("- W1 One step is one day", "- <element>"), "unfilled template placeholder", 16),
    (lambda t: t.replace("## UNCERTAINTY\n- U1 Rain inflow 30-100 litres per rainy day [assumed: no figure]\n",
                         "## UNCERTAINTY\n"), "section 'UNCERTAINTY' has no content", 18),
])
def test_sectioned_faults_reported_with_line_numbers(edit, fragment, line):
    p = only(sec_check(edit(SEC_VALID)), fragment)
    assert p["line"] == line and p["hint"]


def test_sectioned_missing_section_and_empty_document():
    probs = sec_check(SEC_VALID.replace("## TIMING\n- W1 One step is one day [assumed: not given]\n", ""))
    p = only(probs, "missing section 'TIMING'")
    assert p["line"] is None and "## TIMING" in p["hint"]
    assert sec_check("   ")[0]["msg"] == "the document is empty"


def test_sectioned_tags_off_and_fine_granularity():
    untagged = "\n".join(SEC.TAG_RE.sub("", ln) for ln in SEC_VALID.splitlines())
    assert sec_check(untagged, tags=False) == []
    assert only(sec_check(untagged), "no provenance tag")["line"] == 2
    fine = sec_check(SEC_VALID, granularity="fine")
    assert only(fine, "continuation line")["line"] == 10          # coarse allows it, fine does not
    two = SEC_VALID.replace("- E2 Four garden beds", "- E2 Four garden beds. They are raised")
    assert only(sec_check(two, granularity="fine"), "more than one fact")["line"] == 3
    assert not [p for p in sec_check(two) if "more than one fact" in p["msg"]]


def test_sectioned_custom_headers_and_upper_case_terms():
    doc = "## PURPOSE\n- keep plants watered\nPARTS:\n- tank\n- CPU controller\n## FLOWS\n- water to beds\n"
    assert SEC.check(doc, ["PURPOSE", "PARTS", "FLOWS"], tags=False) == []
    probs = SEC.check(doc + "CPU\n", ["PURPOSE", "PARTS", "FLOWS"], tags=False)
    assert only(probs, "not an element")["line"] == 8               # a bare upper-case term is not a header


# ---------------------------------------------------------------------------- ontology checker
def test_ontology_accepts_valid_document():
    assert O.check(ONT_VALID, ONT_PARAMS) == []


@pytest.mark.parametrize("edit, fragment, line", [
    (lambda t: t.replace("pump drives coolant flow.", "pump zaps coolant flow."), "relation 'zaps' is not in", 14),
    (lambda t: t.replace("pump drives coolant flow.", "Pumps are nice and useful things"), "not a statement", 14),
    (lambda t: t.replace("power supply\n", ""), "undeclared term 'power supply'", 15),
    (lambda t: t.replace("alarm\n", ""), "undeclared term 'alarm'", 17),
    (lambda t: t.replace("cooling system is part of plant.", "cooling system is part of pump."), "part-of cycle", 12),
    (lambda t: t.replace("pump is a kind of machine.", "pump is a kind of pump."), "is a kind of itself", 10),
    (lambda t: t.replace("pump drives coolant flow.", "pump drives coolant flow. It is fast."),
     "more than one statement", 14),
    (lambda t: t.replace("If temperature has value above 90 then alarm is on.",
                         "If temperature zooms then alarm is on."), "clause 'temperature zooms'", 18),
    (lambda t: t.replace("pump is a kind of machine.", "pump is a type of machine."),
     "hierarchy sentence not in the grammar", 10),
    (lambda t: "pump drives coolant flow.\n" + t, "statement before the first header", 1),
    (lambda t: t.replace("## PROPERTIES", "## NOTES"), "unknown section header 'NOTES'", 19),
])
def test_ontology_faults_reported_with_line_numbers(edit, fragment, line):
    p = only(O.check(edit(ONT_VALID), ONT_PARAMS), fragment)
    assert p["line"] == line and p["hint"]


def test_ontology_hint_suggests_close_term_and_vocabulary():
    p = only(O.check(ONT_VALID.replace("pump drives coolant flow.", "pump drives coolant flw."), ONT_PARAMS),
             "undeclared term 'coolant flw'")
    assert "coolant flow" in p["hint"]
    p = only(O.check(ONT_VALID.replace("pump drives", "pump zaps"), ONT_PARAMS), "relation 'zaps'")
    assert "depends on" in p["hint"]


def test_ontology_hierarchy_kinds_depth_and_missing_header():
    no_isa = dict(ONT_PARAMS, hierarchy=["part_of"])
    assert only(O.check(ONT_VALID, no_isa), "'is a kind of' sentences are not used")["line"] == 9
    shallow = dict(ONT_PARAMS, max_depth=1)
    p = only(O.check(ONT_VALID, shallow), "levels deep")
    assert p["line"] in (8, 9) and "max 1" in p["msg"]
    assert only(O.check(ONT_VALID, shallow), "part-of chain")["line"] in (10, 11)
    p = only(O.check(ONT_VALID.replace("## RULES\nIf temperature has value above 90 then alarm is on.\n", ""),
                     ONT_PARAMS), "missing section 'RULES'")
    assert p["line"] is None


def test_ontology_is_a_cycle_detected():
    doc = ONT_VALID.replace("pump is a kind of machine.", "pump is a kind of machine.\nmachine is a kind of "
                                                          "centrifugal pump.")
    p = only(O.check(doc, ONT_PARAMS), "is-a cycle")
    assert p["line"] == 11 and "machine" in p["msg"] and "centrifugal pump" in p["msg"]


def test_term_normalisation():
    assert O.norm_term("The Pump") == "pump" and O.norm_term("a") == "a"
    assert O.norm_term("supply and demand") is None and O.norm_term("<thing>") is None


# ---------------------------------------------------------------------------- ontology queries
def test_lookup_path_hierarchy():
    o = O.Ontology(ONT_VALID, ONT_PARAMS)
    lk = o.lookup("Centrifugal pump")
    assert "is a kind of (nearest first): pump (line 9), machine (line 10)" in lk
    assert "inherited from its categories:" in lk and "line 16: machine requires power supply." in lk
    assert "line 14: pump drives coolant flow." in lk
    lk = o.lookup("pump")
    assert "is part of (nearest first): cooling system (line 11), plant (line 12)" in lk
    assert "kinds of pump: centrifugal pump (line 9)" in lk and "declared: line 2" in lk
    pa = o.path("pump", "alarm")
    assert pa.splitlines()[0] == "PATH pump -> alarm (3 step(s)):"
    assert "pump --[drives]--> coolant flow   (line 14)" in pa
    assert "--[decreases, after 2 minutes, when reactor is running]--> temperature" in pa
    assert "--[rule: If temperature has value above 90 then alarm is on.]--> alarm   (line 18)" in pa
    back = o.path("alarm", "pump")
    assert "no directed chain" in back and "other direction" in back
    via = o.path("centrifugal pump", "power supply")
    assert "using hierarchy links" in via and "--[is a kind of]--> pump" in via and "--[requires]--> power supply" in via
    h = o.hierarchy("plant")
    assert h.splitlines()[:3] == ["HIERARCHY plant", "plant", "  - cooling system (part, line 12)"]
    assert "centrifugal pump (kind, line 9)" in h
    whole = o.hierarchy()
    assert "machine" in whole.splitlines() and "plant" in whole.splitlines()
    assert o.lookup("pumpp").startswith("ERROR") and "similar: pump" in o.lookup("pumpp")


def test_queries_are_deterministic():
    a, b = O.Ontology(ONT_VALID, ONT_PARAMS), O.Ontology(ONT_VALID, ONT_PARAMS)
    assert a.path("pump", "alarm") == b.path("pump", "alarm") and a.hierarchy() == b.hierarchy()


# ---------------------------------------------------------------------------- seeds and genome schema
def test_seeds_validate_and_ids_stable():
    disk = load_all()
    for name in ("SEC_A1", "SEC_FLOW", "ONT_BASIC"):
        g = SEEDS[name]
        S.validate(g)
        assert S.incompatibilities(g) == [] and g["id"] == S.genome_id(g) == disk[name]["id"]
        assert render(g).tools()
    assert SEEDS["SEC_A1"]["id"] == "g56e144cc15f"
    assert SEEDS["SEC_FLOW"]["id"] == "g71a939eeeb0"
    assert SEEDS["ONT_BASIC"]["id"] == "ge8df67f5f47"
    for name in ("A0P", "A2", "A3", "A5", "G1C1", "G1C3"):        # existing seeds unchanged
        assert SEEDS[name]["id"] == disk[name]["id"]


@pytest.mark.parametrize("mut", [
    lambda p: p.update(headers=["ENTITIES", "STATE"]),                  # too few
    lambda p: p.update(headers=["ENTITIES", "STATE", "lower case"]),
    lambda p: p.update(headers=["ENTITIES", "STATE", "STATE"]),
    lambda p: p.update(tags="yes"),
    lambda p: p.update(granularity="medium"),
    lambda p: p.update(extra=1),
])
def test_bad_sectioned_params_rejected(mut):
    g = copy.deepcopy(SEEDS["SEC_A1"]["genes"])
    mut(g["core"]["params"])
    with pytest.raises(S.GenomeError):
        S.make(g, ["p"], "point")


@pytest.mark.parametrize("mut", [
    lambda p: p.update(relations=["causes"]),
    lambda p: p.update(relations=["causes", "is caused by"]),         # grammar word
    lambda p: p.update(relations=["causes", "Drives"]),
    lambda p: p.update(hierarchy=["is_a", "kind_of"]),
    lambda p: p.update(max_depth=0),
    lambda p: p.update(max_depth=True),
    lambda p: p.update(hierarchy=[], headers=["RELATIONSHIPS", "RULES"]),   # nothing could declare a term
])
def test_bad_ontology_params_rejected(mut):
    g = copy.deepcopy(SEEDS["ONT_BASIC"]["genes"])
    mut(g["core"]["params"])
    with pytest.raises(S.GenomeError):
        S.make(g, ["p"], "point")


def test_ontology_query_layer_compatibility():
    assert S.incompatibilities(variant("ONT_BASIC", query_layer=["compare"])) == []
    assert S.incompatibilities(variant("ONT_BASIC", query_layer=["trajectory"]))
    assert S.incompatibilities(variant("SEC_A1", query_layer=["explain"]))
    assert S.incompatibilities(variant("SEC_A1", **{"tool_interface.submit": "line_statements"}))
    arm = render(variant("ONT_BASIC", query_layer=["compare"]))
    t = names(arm.tools())
    assert "hierarchy" in t and "lookup" not in t and "path" not in t


# ---------------------------------------------------------------------------- protocols through the rendered arm
ONT_PARTS = [("THINGS", "pump: moves coolant\ncoolant flow\nreactor\ntemperature\nalarm\npower supply"),
             ("HIERARCHY", "centrifugal pump is a kind of pump.\npump is a kind of machine.\npump is part of cooling "
                           "system.\ncooling system is part of plant."),
             ("RELATIONSHIPS", "pump drives coolant flow.\ncoolant flow decreases temperature, after 2 minutes, "
                               "when reactor is running.\nmachine requires power supply."),
             ("RULES", "If temperature has value above 90 then alarm is on."),
             ("PROPERTIES", "pump has flow rate 5 litres per second.")]


def sec_parts(text=SEC_VALID):
    out, cur = [], None
    for ln in text.splitlines():
        if ln.startswith("## "):
            cur = [ln[3:], []]
            out.append(cur)
        elif ln.strip():
            cur[1].append(ln)
    return [(h, "\n".join(b)) for h, b in out]


@pytest.mark.parametrize("seed, doc, parts", [("SEC_A1", SEC_VALID, sec_parts()),
                                              ("ONT_BASIC", ONT_VALID, ONT_PARTS)])
@pytest.mark.parametrize("proto", list(S.PROTOCOLS))
def test_every_protocol_builds_a_valid_document(seed, doc, parts, proto):
    arm = render(variant(seed, build_protocol={"kind": proto, "params": {}}))
    st = arm.new_state()
    t = names(arm.current_tools(st))
    if proto == "template_first":
        assert t == ["get_template"]
        tpl = arm.handle(st, "get_template", {})
        headers = arm.genes["core"]["params"]["headers"]
        assert tpl.startswith("TEMPLATE") and all(f"## {h}" in tpl for h in headers) and "<" in tpl
        t = names(arm.current_tools(st))
    if proto == "sketch_refine":
        assert "sketch" in t and "check" in t
        arm.handle(st, "sketch", {"outline": "a pump cools a reactor"})
    if proto == "stepwise_checked":
        assert "write_model" not in t and {"add_part", "check", "finish"} <= set(t)
        for sec, text in parts:
            out = arm.handle(st, "add_part", {"section": sec, "text": text})
            assert out.startswith("part ")
        assert "no problems found" in arm.handle(st, "check", {})
        fin = arm.handle(st, "finish", {})
        assert "submitted" in fin and "CHECK: no problems found" in fin
    elif proto == "decompose":
        assert "write_model" not in t and "add_subsystem" in t
        half = len(parts) // 2
        for nm, chunk in (("first", parts[:half]), ("second", parts[half:])):
            arm.handle(st, "add_subsystem", {"name": nm, "text": "\n".join(f"## {h}\n{b}" for h, b in chunk)})
        assert "CHECK: no problems found" in arm.handle(st, "finish", {})
    else:
        out = arm.handle(st, "write_model", {"text": doc})
        assert "saved" in out and "CHECK: no problems found" in out
    v = arm.validity(st)
    assert v["usable"] and v["executable"]
    assert buildable({"rep": arm.export(st), "validity": v})


def test_stepwise_problem_located_in_part_with_style():
    g = variant("ONT_BASIC", **{"tool_interface.errors": "all_with_fix_hints"})
    arm = render(g)
    st = arm.new_state()
    arm.handle(st, "add_part", {"section": "THINGS", "text": "pump\ncoolant flow"})
    out = arm.handle(st, "add_part", {"section": "RELATIONSHIPS", "text": "pump drives coolant flow.\npump zaps "
                                                                          "reactor."})
    assert "relation 'zaps' is not in the vocabulary" in out and "part 2 'RELATIONSHIPS' line 2" in out
    assert "hint: use one of: causes" in out and "missing section 'RULES'" in out
    first = render(SEEDS["ONT_BASIC"])
    st = first.new_state()
    out = first.handle(st, "add_part", {"section": "THINGS", "text": "pump"})
    assert "CHECK: PROBLEM:" in out and "more problem(s) not shown" in out
    fin = first.handle(st, "finish", {})
    assert "CHECK: PROBLEM" in fin and not first.validity(st)["usable"]
    assert not buildable({"rep": first.export(st), "validity": first.validity(st)})


def test_one_shot_failed_build_not_buildable_and_guide_mentions_params():
    g = variant("SEC_FLOW", build_protocol={"kind": "one_shot", "params": {}})
    arm = render(g)
    st = arm.new_state()
    out = arm.handle(st, "write_model", {"text": "## PURPOSE\n- water plants\n"})
    assert "CHECK: PROBLEM: missing section 'PARTS'" in out
    rec = {"rep": arm.export(st), "validity": arm.validity(st)}
    assert not buildable(rec)
    assert "`## TIMELINE`" in arm.guide() and "provenance tag" not in arm.guide()
    assert "lookup" in render(SEEDS["ONT_BASIC"]).guide() and "depends on" in render(SEEDS["ONT_BASIC"]).guide()


def test_given_mode_offers_queries_only():
    arm = render(SEEDS["ONT_BASIC"])
    st = arm.load({"text": ONT_VALID}, given_gold=True)
    t = names(arm.gold_tools(st))
    assert {"lookup", "path", "show_model"} <= set(t) and not {"add_part", "finish", "write_model"} & set(t)
    assert "pump --[drives]--> coolant flow" in arm.handle(st, "path", {"source": "pump", "target": "temperature"})


# ---------------------------------------------------------------------------- builders
def _rpc(session, msgs):
    out = io.StringIO()
    M.serve(session, io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n"), out)
    return [json.loads(ln) for ln in out.getvalue().splitlines()]


def _call(i, name, args):
    return {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}}


def test_mcp_session_builds_ontology_then_queries(tmp_path):
    out = tmp_path / "mcp_state.json"
    s = M.Session(render(SEEDS["ONT_BASIC"]), str(out), max_tool_calls=12)
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    msgs += [_call(10 + k, "add_part", {"section": sec, "text": text}) for k, (sec, text) in enumerate(ONT_PARTS)]
    msgs += [_call(20, "check", {}), _call(21, "finish", {}), _call(22, "lookup", {"term": "pump"}),
             _call(23, "path", {"source": "pump", "target": "alarm"})]
    r = _rpc(s, msgs)
    tools = {t["name"] for t in r[1]["result"]["tools"]}
    assert tools == {"add_part", "check", "finish", "show_model", "lookup", "path"}
    texts = [x["result"]["content"][0]["text"] for x in r[2:]]
    assert not any(x["result"]["isError"] for x in r[2:])
    assert "no problems found" in texts[-4] and "CHECK: no problems found" in texts[-3]
    assert texts[-2].startswith("LOOKUP pump") and "PATH pump -> alarm (3 step(s))" in texts[-1]
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["validity"]["usable"] and doc["validity"]["executable"] and doc["tool_calls"] == 9
    assert buildable({"rep": doc["rep"], "validity": doc["validity"]})
    assert O.check(doc["rep"]["text"], ONT_PARAMS) == []


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = script, []

    def __call__(self, model, messages, *, purpose, seed=0, max_tokens=8000, tools=None, response_format=None, **kw):
        n = len(self.calls)
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        m = self.script[min(n, len(self.script) - 1)]
        return dict(m), {"prompt_tokens": 10, "completion_tokens": min(500, max_tokens), "cost": 0.0,
                         "wall_s": 0.0, "provider": "fake", "finish_reason": "stop"}


def _tc(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


ITEM = {"id": "T1", "system_id": "T", "bank": "synthetic", "question": "What happens to temperature?",
        "task_type": "prediction", "answer_key": {"key_points": []}}


def test_qwen_path_template_first_sectioned():
    from tournament.run import run_episode
    flow = ("## PURPOSE\n- keep the reactor cool\n## PARTS\n- pump\n## FLOWS\n- coolant from pump to reactor\n"
            "## RULES\n- alarm above 90 degrees\n## TIMELINE\n- cooling takes 2 minutes\n## UNKNOWNS\n- pump age\n")
    script = [{"role": "assistant", "content": "", "tool_calls": [_tc("get_template", {}, "c1")]},
              {"role": "assistant", "content": "", "tool_calls": [_tc("write_model", {"text": flow}, "c2")]},
              {"role": "assistant", "content": "FINAL ANSWER temperature falls"}]
    client = FakeClient(script)
    rec = run_episode(render(SEEDS["SEC_FLOW"]), ITEM, "A reactor.", "fake", client)
    assert names(client.calls[0]["tools"]) == ["get_template"]
    assert "write_model" in names(client.calls[1]["tools"])
    assert [t["tool"] for t in rec["tool_trace"]] == ["get_template", "write_model"]
    assert "CHECK: no problems found" in rec["tool_trace"][1]["result"]
    assert rec["final_raw"].startswith("FINAL ANSWER") and buildable(rec)


def test_qwen_path_ontology_stepwise_with_queries():
    from tournament.run import run_episode
    calls = [_tc("add_part", {"section": s, "text": t}, f"p{k}") for k, (s, t) in enumerate(ONT_PARTS)]
    script = [{"role": "assistant", "content": "", "tool_calls": calls + [_tc("finish", {}, "f")]},
              {"role": "assistant", "content": "", "tool_calls": [_tc("path", {"source": "pump",
                                                                                "target": "temperature"}, "q")]},
              {"role": "assistant", "content": "FINAL ANSWER temperature falls"}]
    rec = run_episode(render(SEEDS["ONT_BASIC"]), ITEM, "A reactor.", "fake", FakeClient(script))
    assert not any(t["error"] for t in rec["tool_trace"])
    assert "coolant flow --[decreases" in rec["tool_trace"][-1]["result"]
    assert buildable(rec) and rec["validity"]["checked"]


def test_qwen_path_failed_build_is_not_buildable():
    from tournament.run import run_episode
    script = [{"role": "assistant", "content": "", "tool_calls": [_tc("add_part", {"section": "THINGS",
                                                                                    "text": "pump"}, "a"),
                                                                   _tc("finish", {}, "f")]},
              {"role": "assistant", "content": "FINAL ANSWER unsure"}]
    rec = run_episode(render(SEEDS["ONT_BASIC"]), ITEM, "A reactor.", "fake", FakeClient(script))
    assert rec["rep"]["text"] and not buildable(rec)


# ---------------------------------------------------------------------------- operators
def _ok(child):
    S.validate(child)
    assert S.incompatibilities(child) == []
    render(child)


def test_param_operator_reaches_every_param_and_stays_valid():
    rng = random.Random(3)
    seen = set()
    for name in ("SEC_A1", "SEC_FLOW", "ONT_BASIC"):
        parent = SEEDS[name]
        for _ in range(150):
            c = D.apply("param", [parent], rng)
            assert c is not None and c["parents"] == [parent["id"]] and c["operator"] == "param"
            _ok(c)
            seen.add(c["op_notes"]["param"])
            parent = c if rng.random() < 0.5 else parent         # random walks, too
    assert seen == {"add_header", "remove_header", "move_header", "tags", "granularity", "add_relation",
                    "remove_relation", "hierarchy", "depth", "strict"}
    assert D.apply("param", [SEEDS["G1C3"]], rng) is None        # v2-backed cores have no evolvable params


def test_deterministic_operators_keep_children_valid():
    rng = random.Random(11)
    names_ = list(SEEDS)
    made = {"swap": 0, "one_point": 0, "simplify": 0}
    for _ in range(300):
        a, b = rng.sample(names_, 2)
        for op in ("swap", "one_point"):
            c = D.apply(op, [SEEDS[a], SEEDS[b]], rng)
            if c:
                made[op] += 1
                _ok(c)
        c = D.apply("simplify", [SEEDS[rng.choice(["SEC_A1", "SEC_FLOW", "ONT_BASIC"])]], rng)
        if c:
            made["simplify"] += 1
            _ok(c)
    assert all(made.values())
    removed = {D.apply("simplify", [SEEDS["ONT_BASIC"]], random.Random(k))["op_notes"]["removed"][0]
               for k in range(40)}
    assert {"headers", "relations", "query"} <= removed


def test_ai_operator_can_mutate_headers_and_relations():
    from v3.operators import ai
    assert "granularity" in ai.GENE_HELP and "relations" in ai.GENE_HELP and "max_depth" in ai.GENE_HELP
    p = SEEDS["ONT_BASIC"]
    params = dict(p["genes"]["core"]["params"], relations=p["genes"]["core"]["params"]["relations"] + ["enables"])
    good = lambda t: json.dumps({"genes": {"core": {"kind": "ontology", "params": params}}, "why": "x"})
    c = ai.apply("point", [p], {"construction": 3}, [], call=good)
    assert c and "enables" in c["genes"]["core"]["params"]["relations"]
    bad = lambda t: json.dumps({"genes": {"core": {"kind": "sectioned", "params": {"headers": ["A"]}}}})
    assert ai.apply("point", [SEEDS["SEC_A1"]], {}, [], call=bad) is None


def test_loose_ontology_keeps_free_text_as_notes():
    p = dict(ONT_PARAMS, strict=False, headers=["THINGS", "RELATIONSHIPS"])
    text = ("## THINGS\npump\n## RELATIONSHIPS\npump drives coolant flow.\nThe pump is old and rattles a bit.\n"
            "coolant flow decreases temperature.\n")
    assert O.check(text, p) == []
    o = O.Ontology(text, p)
    assert o.notes and "rattles" in o.notes[0][1] and "temperature" in o.decl
    assert "PATH pump -> temperature (2 step(s))" in o.path("pump", "temperature")
    assert O.check(text, dict(p, strict=True))      # the same text fails in strict mode


def test_param_operator_reaches_strict_toggle():
    import random
    seen = set()
    for i in range(300):
        r = D.param_step(SEEDS["ONT_BASIC"], random.Random(i))
        if r and r[3]["param"] == "strict":
            seen.add(r[0]["core"]["params"]["strict"])
    assert seen == {False}
