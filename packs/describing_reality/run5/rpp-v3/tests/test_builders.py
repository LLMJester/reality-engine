"""M3 builder plumbing: MCP server protocol, budget enforcement, recording; stream-json parsing."""
import io
import pathlib
import json

from tests.toy_arm import ToyArm
from v3.builders import mcp_server as M
from v3.builders.claude_builder import parse_stream, command


def _rpc(session, msgs):
    out = io.StringIO()
    M.serve(session, io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n"), out)
    return [json.loads(l) for l in out.getvalue().splitlines()]


def test_mcp_protocol_tools_and_recording(tmp_path):
    s = M.Session(ToyArm(), str(tmp_path / "o.json"), max_tool_calls=3)
    r = _rpc(s, [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                 {"jsonrpc": "2.0", "method": "notifications/initialized"},
                 {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                 {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "set_value",
                                                                             "arguments": {"name": "a", "value": 2}}},
                 {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "total", "arguments": {}}},
                 {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "nope", "arguments": {}}},
                 {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "total", "arguments": {}}},
                 {"jsonrpc": "2.0", "id": 7, "method": "bogus"}])
    assert [x["id"] for x in r] == [1, 2, 3, 4, 5, 6, 7]                     # no reply to the notification
    assert {t["name"] for t in r[1]["result"]["tools"]} == {"set_value", "total"}
    assert r[3]["result"]["content"][0]["text"] == "2.0"
    assert r[4]["result"]["isError"]
    assert r[5]["result"]["content"][0]["text"] == M.BUDGET_MSG              # 4th call exceeds max 3
    assert "error" in r[6]
    doc = json.loads((tmp_path / "o.json").read_text())
    assert doc["rep"] == {"a": 2.0} and doc["tool_calls"] == 4 and len(doc["tool_trace"]) == 4


def test_mcp_stdin_is_read_as_utf8(monkeypatch):
    # V3-051: a piped stdin on Windows decodes with cp1252; raw UTF-8 from the client must still arrive intact
    class Rec:
        args = None

        def tools(self):
            return []

        def call(self, name, args):
            Rec.args = args
            return "ok"
    text = "Line 3 – 4 m³ at 90 °C ≈ ‘fine’"
    msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "w", "arguments": {"text": text}}}
    raw = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="cp1252", errors="surrogateescape"))
    out = io.StringIO()
    M.serve(Rec(), out=out)
    assert Rec.args == {"text": text}
    assert json.loads(out.getvalue())["result"]["content"][0]["text"] == "ok"


def test_given_mode_offers_only_query_tools(tmp_path):
    s = M.Session(ToyArm(), str(tmp_path / "o.json"), given_rep={"x": 1})
    assert [t["name"] for t in s.tools()] == ["total"]
    assert s.call("set_value", {"name": "y", "value": 1}).startswith("ERROR")


def test_parse_stream_and_command_isolation():
    lines = [json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "thinking"}]}}),
             json.dumps({"type": "result", "result": "FINAL: 5", "num_turns": 3, "usage": {"output_tokens": 120},
                         "total_cost_usd": 0.01})]
    r = parse_stream(lines)
    assert r["final_raw"] == "FINAL: 5" and r["turns"] == 3 and r["usage"]["output_tokens"] == 120
    c = command("SYS", "m.json", "s.json")
    assert c[c.index("--tools") + 1] == "" and "--strict-mcp-config" in c and "--bare" not in c
    assert c[c.index("--system-prompt") + 1] == "SYS"


def test_integrity_check_flags_non_arm_tools_and_walled_paths():
    from v3.qa.integrity import check_builder
    ok = [json.dumps({"type": "system", "subtype": "init", "tools": ["mcp__arm__set_value"],
                      "mcp_servers": [{"name": "arm", "status": "connected"}]}),
          json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "content":
                      "evidence quotes rpp-core/data/heldout/x.json"}]}})]          # quoted text: not a breach
    assert check_builder(ok) == []
    bad = [json.dumps({"type": "system", "subtype": "init", "tools": ["Read", "mcp__arm__x"], "mcp_servers": []}),
           json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "mcp__arm__x",
                       "input": {"path": "../rpp-core/data/items/S01-1.json"}}]}})]
    probs = check_builder(bad)
    assert any("non-arm tools" in p for p in probs) and any("walled path" in p for p in probs)
    smoke = pathlib.Path(__file__).resolve().parent.parent / "runs" / "smoke" / "claude_toy" / "transcript.jsonl"
    if smoke.exists():                                   # the real isolated run is clean
        assert check_builder(smoke.read_text(encoding="utf-8").splitlines()) == []


def test_template_first_lists_all_tools_but_enforces_order(tmp_path):
    # V3-019: MCP clients fetch tools/list once, so staged tools must be listed from the start
    from v3.genome.render import render
    from v3.genome.seeds import load_all
    s = M.Session(render(load_all()["SEC_FLOW"]), str(tmp_path / "o.json"))
    listed = {t["name"] for t in s.tools()}
    assert "get_template" in listed and len(listed) > 1
    other = sorted(listed - {"get_template"})[0]
    r = s.call(other, {})
    assert r.startswith("ERROR") and "get_template" in r
    s.call("get_template", {})
    assert other in {t["name"] for t in s.available()}


def test_claude_weekly_guard_pauses_at_threshold(monkeypatch):
    """V3-029: the Claude weekly-allowance guard reads usage on the first call and every N calls, and pauses at the
    configured percent or when the plan reports the limit reached; a failed probe does not stop the run."""
    import pathlib
    import tempfile
    import pytest
    from v3 import episode as E
    from v3.builders import claude_builder as CL
    from v3.run import Paused
    monkeypatch.setattr(E, "_GATE", type("G", (), {"wait": lambda self: None})())
    monkeypatch.setattr(E, "_CLAUDE_CALLS", [0])
    monkeypatch.setattr(E, "RUNS", pathlib.Path(tempfile.mkdtemp()))
    monkeypatch.setitem(E.CFG["budget"], "claude_weekly_pause_percent", 85)
    monkeypatch.setitem(E.CFG["budget"], "claude_usage_check_every", 40)
    monkeypatch.setattr(CL, "usage_snapshot", lambda: {"status": "allowed", "seven_day": 85.0})
    with pytest.raises(Paused, match="Claude weekly allowance guard"):
        E.claude_gate()
    monkeypatch.setattr(CL, "usage_snapshot", lambda: {"status": "rejected", "seven_day": 40.0})
    with pytest.raises(Paused):
        E.claude_gate()
    monkeypatch.setattr(CL, "usage_snapshot", lambda: None)
    E.claude_gate()
    monkeypatch.setattr(CL, "usage_snapshot", lambda: {"status": "allowed", "seven_day": 29.0})
    E.claude_gate()                         # call 1: no check until call 40
    assert E._CLAUDE_CALLS[0] == 2
    assert (E.RUNS / "claude_usage.jsonl").read_text(encoding="utf-8").count("\n") == 3
