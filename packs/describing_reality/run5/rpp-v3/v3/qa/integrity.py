"""Integrity checks on agent transcripts (spec §10).

Builder transcripts must show: only the `arm` MCP server, only mcp__arm__* tools, and no sealed or keyed path in any
tool call. Listing detection is structural (v2 D-069): text quoted inside a tool result is not a listing; only an
actual tool call naming a forbidden path, or a tool other than the arm's, is a breach.
"""
import json

from v3 import walls


def check_builder(lines, role="builder"):
    problems = []
    pats = walls.forbidden_patterns(role)
    for ln in lines:
        try:
            ev = json.loads(ln)
        except (ValueError, TypeError):
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            tools = ev.get("tools") or []
            bad = [t for t in tools if not t.startswith("mcp__arm__")]
            if bad:
                problems.append(f"builder had non-arm tools: {bad[:5]}")
            servers = [s.get("name") for s in ev.get("mcp_servers") or []]
            if any(s != "arm" for s in servers):
                problems.append(f"unexpected MCP servers: {servers}")
        if ev.get("type") == "assistant":
            for c in (ev.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    name = c.get("name", "")
                    if not name.startswith("mcp__arm__"):
                        problems.append(f"builder called non-arm tool {name}")
                    blob = json.dumps(c.get("input") or {})
                    for p in pats:
                        if p.search(blob):
                            problems.append(f"tool call names a walled path: {blob[:160]}")
                            break
    return problems
