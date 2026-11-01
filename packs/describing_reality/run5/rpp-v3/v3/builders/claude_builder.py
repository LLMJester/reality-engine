"""Primary builder: a headless Claude Code agent whose only tools are one arm's MCP tools (spec §7).

Isolation:
- `--tools ""` disables every built-in tool (no file system, shell or web);
- `--strict-mcp-config` with a single `arm` server;
- `--system-prompt` replaces Claude Code's default prompt, so the builder sees exactly the experiment prompt (the same
  text v2 gives every builder);
- the agent runs in an empty working directory, with setting sources limited to that directory (no user settings,
  plugins or memory);
- deny rules from v3.walls are applied anyway (defence in depth).

Runs on the user's Claude subscription (no --bare, which would require an API key). Effort is measured from the
stream-json result (output tokens, turns) and from the MCP server's tool-call count.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

from v3 import walls
from v3.config import CFG, ROOT

ARM_TOOLS = "mcp__arm"


def claude_exe():
    return shutil.which("claude") or str(pathlib.Path.home() / ".local" / "bin" / "claude.exe")


def usage_snapshot():
    """The plan's rate-limit windows ({"status", "five_hour", "seven_day", "*_resets"}, utilisation in %) from one
    minimal headless call (Haiku, no tools, no settings, no session), the same probe the dashboard uses (V3-029).
    None if the call fails or reports no windows."""
    work = tempfile.mkdtemp(prefix="v3probe_")
    info = None
    try:
        r = subprocess.run([claude_exe(), "-p", "--output-format", "stream-json", "--verbose", "--model", "haiku",
                            "--tools", "", "--strict-mcp-config", "--setting-sources", "",
                            "--no-session-persistence"], input="Reply with the single word OK.", cwd=work,
                           capture_output=True, text=True, encoding="utf-8", timeout=180)
        for ln in r.stdout.splitlines():
            if '"rate_limit_event"' in ln:
                try:
                    info = json.loads(ln).get("rate_limit_info") or info
                except ValueError:
                    pass
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)
    w = (info or {}).get("unifiedWindows") or {}
    if not w:
        return None
    snap = {"t": time.time(), "status": info.get("status"), "source": "guard"}
    for k in ("five_hour", "seven_day"):
        if (w.get(k) or {}).get("utilization") is not None:
            snap[k] = round(w[k]["utilization"] * 100, 1)
            snap[k + "_resets"] = w[k].get("resetsAt")
    return snap


def command(system_prompt, mcp_cfg, settings, model=None):
    return [claude_exe(), "-p", "--output-format", "stream-json", "--verbose",
            "--model", model or CFG["models"]["builder_claude"],
            "--tools", "", "--strict-mcp-config", "--mcp-config", str(mcp_cfg),
            "--allowedTools", ARM_TOOLS, "--permission-mode", "default",
            "--setting-sources", "project", "--settings", str(settings),
            "--no-session-persistence", "--system-prompt", system_prompt]


def parse_stream(lines):
    """Final answer text, usage and turn count from Claude stream-json output."""
    final, usage, turns, cost, texts, err = "", {}, 0, None, [], None
    for ln in lines:
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        if ev.get("type") == "assistant":
            for c in (ev.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "text":
                    texts.append(c.get("text", ""))
        elif ev.get("type") == "result":
            final = ev.get("result") or ""
            usage = ev.get("usage") or {}
            turns = ev.get("num_turns") or 0
            cost = ev.get("total_cost_usd")
            if ev.get("is_error"):
                err = ev.get("subtype") or "error"
    return {"final_raw": final or (texts[-1] if texts else ""), "usage": usage, "turns": turns,
            "reported_cost_usd": cost, "error": err}


def run(genome, system_prompt, user_prompt, out_dir, *, given_rep=None, max_tool_calls=24, timeout=900,
        model=None, arm_class=None):
    """One builder episode. Returns a record dict (no grade)."""
    out_dir = pathlib.Path(out_dir).resolve()   # the MCP server runs in the empty work dir: paths must be absolute
    out_dir.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(tempfile.mkdtemp(prefix="v3b_"))        # empty working directory for the agent
    try:
        gpath = work / "genome.json"
        gpath.write_text(json.dumps(genome), encoding="utf-8")
        state_out = out_dir / "mcp_state.json"
        args = [sys.executable, "-u", "-m", "v3.builders.mcp_server", "--genome", str(gpath), "--out",
                str(state_out), "--max-tool-calls", str(max_tool_calls)]
        if given_rep is not None:
            rp = work / "given.json"
            rp.write_text(json.dumps(given_rep), encoding="utf-8")
            args += ["--given", str(rp)]
        if arm_class:
            args += ["--arm-class", arm_class]
        mcp_cfg = work / "mcp.json"
        mcp_cfg.write_text(json.dumps({"mcpServers": {"arm": {
            "command": args[0], "args": args[1:], "env": {"PYTHONPATH": str(ROOT)}}}}), encoding="utf-8")
        settings = work / "settings.json"
        settings.write_text(json.dumps({"permissions": {"deny": walls.deny_rules("builder") + [
            "Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebFetch", "WebSearch", "Agent", "Task",
            "NotebookEdit"]}}), encoding="utf-8")
        t0 = time.time()
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)        # always the subscription, never an API key
        try:
            user_prompt = user_prompt.encode("utf-8", "replace").decode("utf-8")       # stray surrogates
            system_prompt = system_prompt.encode("utf-8", "replace").decode("utf-8")
            p = subprocess.run(command(system_prompt, mcp_cfg, settings, model), input=user_prompt, cwd=work,
                               capture_output=True, text=True, encoding="utf-8", timeout=timeout, env=env)
            lines, rc, stderr = p.stdout.splitlines(), p.returncode, p.stderr[-2000:]
        except subprocess.TimeoutExpired as e:
            lines, rc, stderr = (e.stdout or "").splitlines() if isinstance(e.stdout, str) else [], -9, "timeout"
        (out_dir / "transcript.jsonl").write_text("\n".join(lines), encoding="utf-8")
        res = parse_stream(lines)
        from v3.qa.integrity import check_builder
        integrity = check_builder(lines)
        st = json.loads(state_out.read_text(encoding="utf-8")) if state_out.exists() else {}
        return {"builder": "claude", "model": model or CFG["models"]["builder_claude"], "final_raw": res["final_raw"],
                "rep": st.get("rep"), "validity": st.get("validity"), "tool_trace": st.get("tool_trace", []),
                "tool_calls": st.get("tool_calls", 0), "turns": res["turns"], "usage": res["usage"],
                "output_tokens": (res["usage"] or {}).get("output_tokens"), "reported_cost_usd": res["reported_cost_usd"],
                "wall_s": time.time() - t0, "rc": rc, "error": res["error"] or (None if rc == 0 else f"rc={rc}: {stderr}"),
                "mode": "gold" if given_rep is not None else "e2e", "integrity": integrity,
                **({"error": "INTEGRITY: " + integrity[0]} if integrity else {})}
    finally:
        shutil.rmtree(work, ignore_errors=True)
