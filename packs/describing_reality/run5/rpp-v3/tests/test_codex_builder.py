"""Codex builder lockdown (V3-010): allowlist, command flags, integrity check. No Codex calls."""
import io
import json

from v3.builders import codex_builder as CB
from v3.builders import mcp_server as M
from v3.genome.render import render
from v3.genome.seeds import load_all

SEEDS = load_all()


def ev(item):
    return json.dumps({"type": "item.completed", "item": item})


def test_integrity_allows_only_arm_allowlist():
    ok = [ev({"type": "agent_message", "text": "hi"}), ev({"type": "reasoning"}),
          ev({"type": "mcp_tool_call", "server": "arm", "tool": "write_model"}),
          ev({"type": "mcp_tool_call", "server": "codex", "tool": "list_mcp_resources"}),        # V3-020
          ev({"type": "mcp_tool_call", "server": "codex", "tool": "list_mcp_resource_templates"})]
    assert CB.check_codex(ok) == []
    for bad in ({"type": "command_execution"}, {"type": "file_change"}, {"type": "web_search"},
                {"type": "mcp_tool_call", "server": "codex", "tool": "read_mcp_resource"},
                {"type": "mcp_tool_call", "server": "other", "tool": "list_mcp_resources"},
                {"type": "mcp_tool_call", "server": "arm", "tool": "delete_everything"},
                {"type": "collab_tool_call", "tool": "spawn_agent"}):
        assert CB.check_codex(ok + [ev(bad)]), bad


def test_command_disables_everything_and_allowlists_four_tools():
    cmd = " ".join(CB.command(["-m", "x"]))
    for f in ("shell_tool", "unified_exec", "multi_agent", "computer_use", "browser_use", "apps", "plugins"):
        assert f"--disable {f}" in cmd
    assert "-s read-only" in cmd and "--ignore-user-config" in cmd and "--ephemeral" in cmd
    assert "enabled_tools=[" + ",".join(f'"{x}"' for x in CB.ALLOW) + "]" in cmd and "-m gpt-6-luna" in " ".join(CB.command([], CB.MODEL))
    assert 'default_tools_approval_mode="approve"' in cmd and 'web_search="disabled"' in cmd


def test_server_allowlist_hides_and_refuses(tmp_path):
    s = M.Session(render(SEEDS["ONT_BASIC"]), str(tmp_path / "o.json"), allow=CB.ALLOW)
    names = {t["name"] for t in s.tools()}
    assert names <= set(CB.ALLOW) and "add_part" in names
    s2 = M.Session(render(SEEDS["ONT_BASIC"]), str(tmp_path / "o2.json"), allow=["show_model"])
    assert {t["name"] for t in s2.tools()} <= {"show_model"}
    assert s2.call("add_part", {"section": "THINGS", "text": "pump"}).startswith("ERROR")


def test_usable_only_for_designs_inside_allowlist():
    assert CB.usable_for(render(SEEDS["ONT_LOOSE"]))
    for ok in ("ONT_BASIC", "SEC_A1", "SEC_FLOW", "G1C3", "A2", "A5", "A0P"):
        assert CB.usable_for(render(SEEDS[ok])), ok
    from v3 import sandbox
    for coded in ("G1C1", "A3"):                               # model-written Python: only in Docker (V3-043)
        assert CB.usable_for(render(SEEDS[coded])) == sandbox.available(), coded


def test_code_cores_refused_without_docker(monkeypatch):
    from v3 import sandbox
    monkeypatch.setattr(sandbox, "available", lambda: False)
    for coded in ("G1C1", "A3"):
        assert not CB.usable_for(render(SEEDS[coded])), coded


def test_weekly_guard_pauses_at_threshold(monkeypatch):
    from v3 import episode as E
    from v3.run import Paused
    import pytest
    monkeypatch.setattr(E, "_CODEX_GATE", type("G", (), {"wait": lambda self: None})())
    monkeypatch.setattr(E, "_CODEX_CALLS", [0])
    monkeypatch.setattr(E, "RUNS", __import__("pathlib").Path(__import__("tempfile").mkdtemp()))
    limit = __import__("v3.config", fromlist=["CFG"]).CFG["budget"].get("codex_weekly_pause_percent", 70)
    monkeypatch.setattr(CB, "usage_snapshot", lambda: {"weekly_used_percent": limit + 1.0})
    with pytest.raises(Paused):
        E.codex_gate()
    monkeypatch.setattr(CB, "usage_snapshot", lambda: {"weekly_used_percent": 10.0})
    E.codex_gate()
    assert E._CODEX_CALLS[0] == 1
