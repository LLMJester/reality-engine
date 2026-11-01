"""MCP stdio server exposing one rendered arm's tools to a Claude builder (spec §7).

The builder (headless `claude -p`) has no built-in tools; this server is its only interface. It holds the arm state,
enforces the effort budget (tool calls), and writes the representation and trace to --out after every call, so the
episode is recorded even if the builder is cut off.

Run: python -m v3.builders.mcp_server --genome G.json --out OUT.json [--given REP.json] [--max-tool-calls N]
Protocol: JSON-RPC 2.0, one message per line (MCP stdio transport): initialize, tools/list, tools/call.
"""
import argparse
import json
import sys
import time

PROTOCOL = "2025-06-18"
BUDGET_MSG = "ERROR: the tool budget is used up. Give your FINAL ANSWER now, in plain text."


class Session:
    def __init__(self, arm, out, given_rep=None, max_tool_calls=24, allow=None):
        self.arm, self.out, self.max = arm, out, max_tool_calls
        self.allow = set(allow) if allow else None   # server-side allowlist (Codex builder, V3-010)
        self.given = given_rep is not None
        self.state = arm.load(given_rep, given_gold=True) if self.given else arm.new_state()
        self.trace, self.calls = [], 0
        self.flush()

    def _mcp(self, offered):
        out = []
        for t in offered:
            f = t["function"]
            if self.allow is not None and f["name"] not in self.allow:
                continue
            out.append({"name": f["name"], "description": f.get("description", ""),
                        "inputSchema": f.get("parameters") or {"type": "object", "properties": {}}})
        return out

    def available(self):
        """Tools callable in the current state (staged by the build protocol)."""
        return self._mcp(self.arm.gold_tools(self.state) if self.given else self.arm.current_tools(self.state))

    def tools(self):
        """Tools listed to the client. MCP clients fetch tools/list once, so in build mode the whole set is listed from
        the start and protocol staging (e.g. template_first) is enforced at call time (V3-019)."""
        if self.given:
            return self.available()
        full = getattr(self.arm, "tools", None)
        out = self.available()
        seen = {t["name"] for t in out}
        out += [t for t in self._mcp(full() if full else []) if t["name"] not in seen]
        return out

    def call(self, name, args):
        self.calls += 1
        now = {t["name"] for t in self.available()}
        if self.calls > self.max:
            res = BUDGET_MSG
        elif name not in {t["name"] for t in self.tools()}:
            res = f"ERROR: tool {name!r} is not available now"
        elif name not in now:
            first = " Call get_template first." if now == {"get_template"} else ""
            res = (f"ERROR: tool {name!r} is not available yet at this step of the build.{first} "
                   f"Available now: {', '.join(sorted(now)) or 'none'}")
        else:
            try:
                res = self.arm.handle(self.state, name, args or {})
            except Exception as e:  # arms must not raise; record it as a tool error, never crash the session
                res = f"ERROR: internal {type(e).__name__}: {e}"
        res = res if isinstance(res, str) else json.dumps(res)
        self.trace.append({"t": time.time(), "tool": name, "args": args, "result": res[:4000],
                           "error": res.startswith("ERROR")})
        self.flush()
        return res

    def flush(self):
        try:
            rep, validity = self.arm.export(self.state), self.arm.validity(self.state)
        except Exception as e:
            rep, validity = None, {"usable": False, "executable": False, "errors": [f"export failed: {e}"]}
        doc = {"rep": rep, "validity": validity, "tool_trace": self.trace, "tool_calls": self.calls,
               "given": self.given}
        import sys as _sys
        sb = _sys.modules.get("v3.sandbox")
        if sb is not None and getattr(sb, "UNAVAILABLE", None):
            doc["infra_error"] = "SANDBOX UNAVAILABLE: " + sb.UNAVAILABLE[-1]     # code review: never graded
        import pathlib
        pathlib.Path(self.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, default=str)
        import os
        os.replace(tmp, self.out)


def serve(session, inp=None, out=None):
    out = out if out is not None else sys.stdout
    if inp is None:
        # V3-051: MCP messages are UTF-8, but a piped stdin on Windows decodes with the locale code page (cp1252), so a
        # client that sends raw UTF-8 (the Claude CLI) had "–" arrive as "â€“"; clients that escape non-ASCII (Codex)
        # were unaffected
        inp = sys.stdin
        if hasattr(inp, "reconfigure"):
            inp.reconfigure(encoding="utf-8", errors="replace")
    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:          # notification (e.g. notifications/initialized): no reply
            continue
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion", PROTOCOL),
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": "arm", "version": "3.0"}}
        elif method == "tools/list":
            result = {"tools": session.tools()}
        elif method == "tools/call":
            text = session.call(params.get("name"), params.get("arguments"))
            result = {"content": [{"type": "text", "text": text}], "isError": text.startswith("ERROR")}
        elif method == "ping":
            result = {}
        else:
            out.write(json.dumps({"jsonrpc": "2.0", "id": mid,
                                  "error": {"code": -32601, "message": f"unknown method {method}"}}) + "\n")
            out.flush()
            continue
        out.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}) + "\n")
        out.flush()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--given")
    ap.add_argument("--max-tool-calls", type=int, default=24)
    ap.add_argument("--allow-tools", help="comma-separated allowlist: other tools are hidden and refused")
    ap.add_argument("--arm-class", help="tests only: module:Class of an arm to serve instead of rendering the genome")
    a = ap.parse_args(argv)
    given = json.load(open(a.given, encoding="utf-8")) if a.given else None
    if a.arm_class:
        import importlib
        mod, cls = a.arm_class.split(":")
        arm = getattr(importlib.import_module(mod), cls)()
    else:
        from v3.genome.render import render
        arm = render(json.load(open(a.genome, encoding="utf-8")))
    allow = [t for t in (a.allow_tools or "").split(",") if t] or None
    serve(Session(arm, a.out, given, a.max_tool_calls, allow))


if __name__ == "__main__":
    main()
