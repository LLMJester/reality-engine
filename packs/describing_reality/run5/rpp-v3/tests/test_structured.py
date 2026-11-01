"""Run 5 founders: the JSON and Markdown checked cores, the OWN form classifier, and the founder set (V3-042)."""
import json

from v3.builders import codex_builder as CB
from v3.cores.structured import JsonCore, MarkdownCore, json_check, md_check
from v3.formclass import classify
from v3.genome import schema as S
from v3.genome.render import render
from v3.genome.seeds import SEED_DIR

KEYS = ["things", "quantities", "links", "events", "rules", "unknowns"]
GOOD_JSON = json.dumps({
    "things": [{"name": "pump P1", "kind": "pump", "does": "fills tank T1"}],
    "quantities": [{"thing": "T1", "name": "level", "value": 40, "unit": "%", "when": "start"}],
    "links": [{"from": "P1", "to": "T1", "relation": "fills", "condition": "when valve V2 is open"}],
    "events": [{"when": "t=5 min", "what": "V2 closes", "effect": "P1 stops filling T1"}],
    "rules": ["if T1 level > 90% then alarm A1 sounds"],
    "unknowns": ["the pump's maximum flow"]}, indent=1)
HEADERS = ["THINGS", "VALUES", "LINKS", "EVENTS", "RULES", "UNKNOWNS"]
TABLES = ["THINGS", "VALUES", "LINKS", "EVENTS"]
GOOD_MD = """## THINGS
| name | kind | what it does |
|---|---|---|
| P1 | pump | fills T1 |
| T1 | tank | stores water |

## VALUES
| thing | quantity | value | unit | when |
|---|---|---|---|---|
| T1 | level | 40 | % | start |

## LINKS
| from | relation | to | condition |
|---|---|---|---|
| P1 | fills | T1 | V2 open |

## EVENTS
| when | what happens | effect |
|---|---|---|
| t=5 min | V2 closes | P1 stops filling T1 |

## RULES
- if T1 level > 90% then alarm A1 sounds

## UNKNOWNS
- the pump's maximum flow
"""


def msgs(probs):
    return " | ".join(p["msg"] for p in probs)


# ------------------------------------------------------------------ JSON
def test_json_valid_and_fenced():
    assert json_check(GOOD_JSON, KEYS, 4) == []
    assert json_check("```json\n" + GOOD_JSON + "\n```", KEYS, 4) == []


def test_json_problems():
    assert "not valid JSON" in msgs(json_check(GOOD_JSON[:-3], KEYS, 4))
    assert json_check(GOOD_JSON[:-3], KEYS, 4)[0]["line"] is not None
    assert "one JSON object" in msgs(json_check("[1, 2]", KEYS, 4))
    o = json.loads(GOOD_JSON)
    del o["events"]
    o["extra"] = 1
    o["rules"] = []
    m = msgs(json_check(json.dumps(o), KEYS, 4))
    assert "missing key 'events'" in m and "unknown top-level key 'extra'" in m and "'rules' is empty" in m
    o = json.loads(GOOD_JSON)
    o["unknowns"] = [{"a": {"b": {"c": {"d": 1}}}}]
    assert "nesting is" in msgs(json_check(json.dumps(o), KEYS, 4))
    o = json.loads(GOOD_JSON)
    o["rules"] = ["if <condition> then x"]
    assert "placeholder" in msgs(json_check(json.dumps(o), KEYS, 4))
    assert "empty" in msgs(json_check("  ", KEYS, 4))


def test_json_tools():
    core = JsonCore({"keys": KEYS, "max_depth": 4})
    st = core.new_state()
    assert "0 check problem(s)" in core.handle(st, "write_model", {"text": GOOD_JSON})
    assert core.validity(st)["usable"]
    assert '"fills"' in core.handle(st, "lookup", {"term": "links[0]"})
    assert "quantities[0]" in core.handle(st, "lookup", {"term": "T1"}) or \
        "nothing named" in core.handle(st, "lookup", {"term": "T1"})
    assert "things[0]" in core.handle(st, "lookup", {"term": "pump P1"})
    assert "links[0].condition" in core.handle(st, "search", {"text": "valve"})
    assert "no entry" in core.handle(st, "search", {"text": "zebra"})
    st2 = core.new_state()
    core.handle(st2, "write_model", {"text": GOOD_JSON[:-3]})
    assert not core.validity(st2)["usable"]
    assert "2:" in core.handle(st2, "search", {"text": "things"})     # falls back to lines when not valid JSON


# ------------------------------------------------------------------ Markdown
def test_markdown_valid():
    assert md_check(GOOD_MD, HEADERS, TABLES) == []


def test_markdown_problems():
    bad = GOOD_MD.replace("|---|---|---|\n| P1", "| P1", 1)
    assert "separator row" in msgs(md_check(bad, HEADERS, TABLES))
    bad = GOOD_MD.replace("| P1 | pump | fills T1 |", "| P1 | pump |")
    assert "row has 2 cells" in msgs(md_check(bad, HEADERS, TABLES))
    bad = GOOD_MD.replace("- the pump's maximum flow", "The pump's maximum flow is unknown.")
    assert "not a table row or a bullet" in msgs(md_check(bad, HEADERS, TABLES))
    bad = GOOD_MD.replace("| t=5 min | V2 closes | P1 stops filling T1 |\n", "")
    assert "needs a table with at least one data row" in msgs(md_check(bad, HEADERS, TABLES))
    bad = GOOD_MD.replace("## UNKNOWNS\n- the pump's maximum flow\n", "")
    assert "missing section 'UNKNOWNS'" in msgs(md_check(bad, HEADERS, TABLES))
    assert "text before the first header" in msgs(md_check("hello\n" + GOOD_MD, HEADERS, TABLES))
    bad = GOOD_MD.replace("| T1 | tank | stores water |", "| <name> | tank | stores water |")
    assert "placeholder" in msgs(md_check(bad, HEADERS, TABLES))


def test_markdown_tools():
    core = MarkdownCore({"headers": HEADERS, "tables": TABLES})
    st = core.new_state()
    assert "0 check problem(s)" in core.handle(st, "write_model", {"text": GOOD_MD})
    assert core.validity(st)["usable"]
    out = core.handle(st, "lookup", {"term": "T1"})
    assert "## VALUES" in out and "| thing | quantity |" in out and "## RULES" in out
    assert "alarm" in core.handle(st, "search", {"text": "alarm A1"})
    assert "no line" in core.handle(st, "search", {"text": "zebra"})


# ------------------------------------------------------------------ schema and founders
def test_schema_validates_params():
    import pytest
    for bad in ({"keys": ["Things", "b", "c"], "max_depth": 4}, {"keys": ["a1", "b1"], "max_depth": 4},
                {"keys": KEYS, "max_depth": 9}, {"keys": KEYS}):
        with pytest.raises(S.GenomeError):
            S.validate_core_params("json", bad)
    with pytest.raises(S.GenomeError):
        S.validate_core_params("markdown", {"headers": HEADERS, "tables": ["OTHER"]})
    S.validate_core_params("json", {"keys": KEYS, "max_depth": 4})
    S.validate_core_params("markdown", {"headers": HEADERS, "tables": TABLES})


def test_fifteen_founders_render_for_luna():
    names = ["A0P", "SEC_A1", "ONT_LOOSE", "SEC_FLOW", "G1C3", "A5", "FACTS", "STATES", "SPEC", "A2", "A3", "G1C1",
             "JSON", "MARKDOWN", "OWN"]
    from v3 import sandbox
    for n in names:
        g = json.loads((SEED_DIR / f"{n}.json").read_text(encoding="utf-8"))
        arm = render(g)
        assert CB.usable_for(arm) or (g["genes"]["core"]["kind"] in CB.SANDBOXED_CORES and not sandbox.available()), n
    assert json.loads((SEED_DIR / "A0P.json").read_text(encoding="utf-8"))["id"] == "g8e2dd9d60dc"   # run 4 control


# ------------------------------------------------------------------ OWN form classifier
def test_classify_forms():
    assert classify(GOOD_JSON)["form"] == "json"
    assert classify("```json\n" + GOOD_JSON + "\n```")["form"] == "json"
    assert classify(GOOD_MD)["form"] == "tables"
    assert classify("# Parts\nthe pump\n# Rules\nif x then y")["form"] == "headings"
    assert classify("- a\n- b\n- c\nplain")["form"] == "bullets"
    assert classify("```python\nx = 1\n```")["form"] == "code"
    assert classify("import math\nlevel = 40\ndef step(s):\n  return s * 0.9")["form"] == "code"
    assert classify("T1_level = 40\nP1_flow = 15\nvalve_V2 = open\nThe pump fills the tank.")["form"] == "prose"
    # comparisons are not template placeholders (code review)
    from v3.cores.structured import PLACEHOLDER_RE
    assert not PLACEHOLDER_RE.search("- if T1 < 10% and P1 > 50 L/s then open V1")
    assert PLACEHOLDER_RE.search("- <one fact>") and PLACEHOLDER_RE.search("| <...> |")
    assert classify("The pump fills the tank. The tank drains slowly.")["form"] == "prose"
