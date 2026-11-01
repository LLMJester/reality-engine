"""Optional third builder: headless OpenAI Codex (`codex exec`) on the human's ChatGPT subscription (V3-010).

Lockdown, approved by the human with conditions (2026-10-21):
- every built-in Codex capability that can be switched off is switched off (shell, unified exec, apps, plugins,
  browser, computer use, images, multi-agent, skills, goals, web search); read-only sandbox; an empty temporary
  working directory; the human's own Codex config and rules are ignored; no session files are kept;
- the only MCP server is v3's `arm` server, with an explicit tool allowlist enforced three times: the server hides and
  refuses other tools (--allow-tools), Codex enables only those tools (enabled_tools), and only that server's tools
  are pre-approved (default_tools_approval_mode="approve");
- the allowlisted tools take text arguments only and are pure Python: no file, process or network access;
- `check_codex` voids any episode with a call outside the allowlist (any other tool, command, file change, web search,
  agent spawn or unknown item type) and logs it to runs/integrity_log.jsonl.
Only designs whose tools all lie inside ALLOW can use this builder, and a core in SANDBOXED_CORES only while the
Docker sandbox is available (V3-043);
widening ALLOW needs a new human approval. The model is pinned to MODEL.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

from v3.config import ROOT, RUNS

# Approved by the human, 2026-10-21 (V3-010, widened in V3-011). Every tool takes data, never code.
ALLOW = ("get_template", "sketch", "add_part", "add_subsystem", "check", "finish",          # build steps
         "write_model", "revise_model", "show_model", "lookup", "path", "hierarchy",        # sectioned / ontology
         "submit_graph", "patch_graph", "show", "evaluate", "intervene",                    # equations (G1C3)
         "set_graph", "add_variable", "add_edge", "set_equation",                           # causal graph (A2)
         "submit_model", "patch_model", "run_scenario",                                     # rules (A5)
         "search")                                                         # JSON / Markdown notes (V3-042)
# Cores whose tools run model-written Python: probabilistic (G1C1) and simulation (A3). Refused to Codex from V3-011;
# from V3-043 (run 5) allowed only when that code runs in the Docker sandbox (v3/sandbox.py), never on the host.
SANDBOXED_CORES = ("probabilistic", "simulation")
# Codex's built-in resource listing (V3-020): it can only list resources of the configured MCP servers, and the arm
# server is the only one and has none, so it returns an empty list. Reading a resource stays disallowed.
CODEX_BUILTIN_OK = ("list_mcp_resources", "list_mcp_resource_templates")
MODEL = "gpt-6-luna"             # pinned small builder (V3-011)
DISABLE = ("shell_tool", "unified_exec", "apps", "plugins", "remote_plugin", "browser_use", "browser_use_external",
           "computer_use", "in_app_browser", "image_generation", "view_image", "multi_agent", "sleep_tool",
           "tool_suggest", "skill_search", "goals", "shell_snapshot", "skill_mcp_dependency_install")
SAFE_ITEMS = {"agent_message", "reasoning", "error", "todo_list"}
INTEGRITY_LOG = RUNS / "integrity_log.jsonl"


def codex_exe():
    for p in (shutil.which("codex"), pathlib.Path.home() / ".codex" / "plugins" / ".plugin-appserver" / "codex.exe"):
        if p and pathlib.Path(p).exists():
            return str(p)
    raise FileNotFoundError("codex CLI not found")


def usable_for(arm):
    """True if every tool the design can offer is inside the allowlist and, for a code-executing core, the Docker
    sandbox is available and installed in this process (V3-043)."""
    if arm.genes["core"]["kind"] in SANDBOXED_CORES:
        from v3 import sandbox
        if not (sandbox.available() and sandbox.installed()):
            return False
    names = {t["function"]["name"] for t in arm.tools()}
    return names <= set(ALLOW)


def _q(s):
    return json.dumps(str(s))          # a TOML basic string is JSON-compatible for these values


def command(mcp_args, model=None):
    cmd = [codex_exe(), "exec"]
    for f in DISABLE:
        cmd += ["--disable", f]
    cmd += ["--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check", "-s", "read-only",
            "--json", "-c", 'web_search="disabled"',
            "-c", f"mcp_servers.arm.command={_q(sys.executable)}",
            "-c", "mcp_servers.arm.args=[" + ",".join(_q(a) for a in mcp_args) + "]",
            "-c", "mcp_servers.arm.env={PYTHONPATH=" + _q(ROOT) + "}",
            "-c", "mcp_servers.arm.enabled_tools=[" + ",".join(_q(t) for t in ALLOW) + "]",
            "-c", 'mcp_servers.arm.default_tools_approval_mode="approve"']
    if model:
        cmd += ["-m", model]
    return cmd + ["-"]                 # prompt from stdin


def check_codex(lines, allow=ALLOW):
    """Integrity breaches in a Codex JSONL event stream: any item that is not a message/reasoning/error and is not a
    call to an allowlisted tool on the arm server."""
    bad = []
    for ln in lines:
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        it = e.get("item") or {}
        kind = it.get("type")
        if not kind or kind in SAFE_ITEMS:
            continue
        if kind == "mcp_tool_call" and it.get("server") == "arm" and it.get("tool") in allow:
            continue
        if kind == "mcp_tool_call" and it.get("server") == "codex" and it.get("tool") in CODEX_BUILTIN_OK:
            continue
        what = f"{kind}:{it.get('server', '')}/{it.get('tool', '')}" if kind == "mcp_tool_call" else kind
        if what not in bad:
            bad.append(what)
    return [f"unauthorized tool use: {b}" for b in bad]


class WeeklyCapReached(RuntimeError):
    pass


def usage_snapshot():
    """The account's Codex rate-limit snapshot ({"weekly_used_percent", "resets_at", "plan_type", ...}) from one
    minimal non-ephemeral call (all tools off, reply "OK"); the session file it leaves is deleted at once."""
    sessions = pathlib.Path.home() / ".codex" / "sessions"
    before = set(sessions.rglob("*.jsonl")) if sessions.exists() else set()
    work = pathlib.Path(tempfile.mkdtemp(prefix="v3u_"))
    try:
        cmd = [codex_exe(), "exec"] + [x for f in DISABLE for x in ("--disable", f)] + [
            "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "-s", "read-only", "-m", MODEL,
            "-c", 'web_search="disabled"', "-"]
        env = dict(os.environ)
        env.pop("OPENAI_API_KEY", None)
        subprocess.run(cmd, input="Reply with the single word OK. Do not call any tool.", cwd=work,
                       capture_output=True, text=True, encoding="utf-8", timeout=180, env=env)
    except (OSError, subprocess.SubprocessError):
        return None                                   # code review: a failed usage check returns no snapshot
    finally:
        shutil.rmtree(work, ignore_errors=True)
    new = sorted(set(sessions.rglob("*.jsonl")) - before, key=lambda q: q.stat().st_mtime) if sessions.exists() else []
    snap = None
    for f in new:
        try:
            for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if '"rate_limits"' in ln:
                    rl = (json.loads(ln).get("payload") or {}).get("rate_limits") or json.loads(ln).get("rate_limits")
                    if rl:
                        snap = rl
        except (OSError, ValueError):
            pass
        f.unlink(missing_ok=True)
    if not snap:
        return None
    windows = [w for w in (snap.get("primary"), snap.get("secondary")) if w]
    weekly = max(windows, key=lambda w: w.get("window_minutes") or 0) if windows else {}
    return {"weekly_used_percent": weekly.get("used_percent"), "resets_at": weekly.get("resets_at"),
            "window_minutes": weekly.get("window_minutes"), "plan_type": snap.get("plan_type"),
            "limit_reached": snap.get("rate_limit_reached_type"), "t": time.strftime("%Y-%m-%dT%H:%M:%S")}


def parse_stream(lines):
    final, usage, err, calls = "", {}, None, 0
    for ln in lines:
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        it = e.get("item") or {}
        if e.get("type") == "item.completed" and it.get("type") == "agent_message":
            final = it.get("text", "")
        if e.get("type") == "item.completed" and it.get("type") == "mcp_tool_call":
            calls += 1
        if e.get("type") == "turn.completed":
            usage = e.get("usage") or {}
            err = None                                   # a completed turn after a reconnect message is a success
        if e.get("type") in ("turn.failed", "error"):
            err = str(e.get("error") or e.get("message") or e)[:300]
    return {"final_raw": final, "usage": usage, "error": err, "codex_tool_events": calls}


def log_breach(rec_id, breaches):
    INTEGRITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(INTEGRITY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "builder": "codex", "record": str(rec_id),
                            "breaches": breaches}) + "\n")


def run(genome, system_prompt, user_prompt, out_dir, *, given_rep=None, max_tool_calls=24, timeout=900, model=MODEL):
    """One Codex builder episode. Returns a record dict (no grade); an integrity breach sets "error"."""
    out_dir = pathlib.Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(tempfile.mkdtemp(prefix="v3c_"))
    try:
        gpath = work / "genome.json"
        gpath.write_text(json.dumps(genome), encoding="utf-8")
        state_out = out_dir / "mcp_state.json"
        state_out.unlink(missing_ok=True)                # a retry never reads the previous attempt's state
        args = ["-u", "-m", "v3.builders.mcp_server", "--genome", str(gpath), "--out", str(state_out),
                "--max-tool-calls", str(max_tool_calls), "--allow-tools", ",".join(ALLOW)]
        if given_rep is not None:
            rp = work / "given.json"
            rp.write_text(json.dumps(given_rep), encoding="utf-8")
            args += ["--given", str(rp)]
        rule = ("Use only the mcp__arm__ tools; no other tool is permitted." if genome["genes"]["core"]["kind"] != "prose"
                else "This task needs no tools: do not call any tool, write your response directly.")
        prompt = system_prompt + "\n\n" + rule + "\n\n" + user_prompt
        prompt = prompt.encode("utf-8", "replace").decode("utf-8")   # a stray surrogate must not crash the episode
        env = dict(os.environ)
        env.pop("OPENAI_API_KEY", None)           # always the ChatGPT subscription, never an API key
        t0 = time.time()
        try:
            p = subprocess.run(command(args, model), input=prompt, cwd=work, capture_output=True, text=True,
                               encoding="utf-8", timeout=timeout, env=env)
            lines, rc, stderr = p.stdout.splitlines(), p.returncode, p.stderr[-2000:]
        except subprocess.TimeoutExpired as e:
            lines, rc, stderr = (e.stdout or "").splitlines() if isinstance(e.stdout, str) else [], -9, "timeout"
        (out_dir / "transcript.jsonl").write_text("\n".join(lines), encoding="utf-8")
        res = parse_stream(lines)
        integrity = check_codex(lines)
        if integrity:
            log_breach(out_dir, integrity)
        st = json.loads(state_out.read_text(encoding="utf-8")) if state_out.exists() else {}
        rec = {"builder": "codex", "model": model, "final_raw": res["final_raw"],
               "rep": st.get("rep"), "validity": st.get("validity"), "tool_trace": st.get("tool_trace", []),
               "tool_calls": st.get("tool_calls", 0), "turns": 1, "usage": res["usage"],
               "output_tokens": (res["usage"] or {}).get("output_tokens"), "wall_s": time.time() - t0, "rc": rc,
               "error": st.get("infra_error") or res["error"] or (None if rc == 0 else f"rc={rc}: {stderr}"),
               "mode": "gold" if given_rep is not None else "e2e", "integrity": integrity}
        if integrity:
            rec["error"] = "INTEGRITY: " + integrity[0]
        return rec
    finally:
        shutil.rmtree(work, ignore_errors=True)
