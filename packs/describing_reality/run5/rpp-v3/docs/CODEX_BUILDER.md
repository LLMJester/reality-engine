# Codex builder (GPT-6 Luna on the ChatGPT Pro plan)

How v3 runs OpenAI Codex as its second, smaller builder, why each setting is there, and how to check it.
The code is the source of truth: `v3/builders/codex_builder.py`, the gate in `v3/episode.py` (`codex_gate`), the
allowlist in `v3/builders/mcp_server.py` (`--allow-tools`) and `config/v3.json`. Decisions: V3-010 to V3-013 in
`DECISIONS.md`.

## Account and installation (as of 2026-10-21)

| Item | Value |
|---|---|
| Plan | ChatGPT **Pro 100** (Codex reports `plan_type: "prolite"`) |
| Sign-in | ChatGPT login, stored by Codex in `%USERPROFILE%\.codex\auth.json`. Never read, copied or printed by v3 |
| API key | None stored with Codex (`codex doctor`: "stored API key false"); v3 removes `OPENAI_API_KEY` from the environment |
| Credits | None (balance 0), so reaching a limit stops calls; nothing is billed |
| Codex CLI | `codex-cli 0.163.0-alpha.10.2`, at `%USERPROFILE%\.codex\plugins\.plugin-appserver\codex.exe` (not on PATH) |
| Limits | Pro: no five-hour limit; an unpublished weekly cap. `/status` in the interactive app shows "Weekly limit … left" |
| Models on the plan | gpt-6.1-sol, gpt-6-astra, gpt-6-sol, gpt-6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna (`codex debug models`) |

The interactive Codex default is GPT-6.1 Sol; the experiment is pinned to **GPT-6 Luna** (`MODEL`), which uses far
less allowance (OpenAI lists 350–3,000 Luna messages per five hours on Plus, against 15–160 for 6.1 Sol).

## The command v3 runs (one per build or answer phase)

```
codex.exe exec
  --disable shell_tool --disable unified_exec --disable apps --disable plugins --disable remote_plugin
  --disable browser_use --disable browser_use_external --disable computer_use --disable in_app_browser
  --disable image_generation --disable view_image --disable multi_agent --disable sleep_tool
  --disable tool_suggest --disable skill_search --disable goals --disable shell_snapshot
  --disable skill_mcp_dependency_install
  --ignore-user-config --ignore-rules --ephemeral --skip-git-repo-check -s read-only --json
  -c web_search="disabled"
  -c mcp_servers.arm.command="<python.exe>"
  -c mcp_servers.arm.args=["-u","-m","v3.builders.mcp_server","--genome","<tmp>/genome.json",
                           "--out","<record dir>/mcp_state.json","--max-tool-calls","<n>",
                           "--allow-tools","<ALLOW joined by commas>"]
  -c mcp_servers.arm.env={PYTHONPATH="<rpp-v3>"}
  -c mcp_servers.arm.enabled_tools=[<ALLOW>]
  -c mcp_servers.arm.default_tools_approval_mode="approve"
  -m gpt-6-luna
  -                      (prompt on stdin; working directory = a fresh empty temp folder)
```

The prompt is the v2 system prompt + one rule + the user prompt. Designs with tools get "Use only the mcp__arm__
tools; no other tool is permitted." Plain-language designs, which have no tools, get "This task needs no tools: do
not call any tool, write your response directly." Prompts are sanitised to valid UTF-8.

## Approved tool allowlist (the human approved each list in chat)

Enforced three times: the arm server hides and refuses anything else (`--allow-tools`), Codex enables only these
(`enabled_tools`), and only the `arm` server is pre-approved (`default_tools_approval_mode`).

| Group | Tools |
|---|---|
| Build steps | get_template, sketch, add_part, add_subsystem, check, finish |
| Sectioned / ontology | write_model, revise_model, show_model, lookup, path, hierarchy |
| Equations (G1C3) | submit_graph, patch_graph, show, evaluate, intervene |
| Causal graph (A2) | set_graph, add_variable, add_edge, set_equation |
| Rules (A5) | submit_model, patch_model, run_scenario (a fixed checker process, fed structured data) |

**Never given to Codex**, whatever the tool names: the probabilistic (G1C1) and simulation (A3) designs, whose tools
run model-written Python (`REFUSED_CORES`). They are Claude-only.

Every allowlisted tool takes data, not code, and has no file, process or network access. The server's only file
input/output uses paths v3 sets: the genome, the state output and the fixed v2 guide texts.

## Integrity check (fail closed)

`check_codex` reads Codex's JSON event stream. Any item other than a message, reasoning, error or to-do list that is
not an allowlisted call on the `arm` server voids the episode (`error: "INTEGRITY: …"`) and is appended to
`runs/integrity_log.jsonl`. Unknown item types count as breaches.

Codex still lists `exec`, `collaboration.*` (spawn_agent and similar), `request_user_input` and `wait`; this
version has no switch for them. `exec` is needed: Codex reaches MCP tools through it ("code mode"). In tests it had
no file access, and any other use voids the episode.

## Rate control

| Setting (`config/v3.json` → budget) | Value | Effect |
|---|---|---|
| `codex_runs_per_hour` | 60 | Process-wide throttle on Codex calls |
| `codex_usage_check_every` | 40 | Weekly-usage reading on the first call and every 40 calls (locked: one at a time) |
| `codex_weekly_pause_percent` | 70 | At or above 70% of the weekly allowance used, Codex work stops (`v3.run.Paused`) |

A usage reading is one minimal non-ephemeral call ("Reply with the single word OK", all tools off). It reads
`rate_limits` from the session file Codex writes, then deletes that file. Readings go to `runs/codex_usage.jsonl`.

## Lessons from setting it up

1. Disabling `code_mode_host` also hides MCP tools: Codex calls MCP tools from inside `exec`. Keep it on.
2. Headless `codex exec` cannot answer approval prompts. Without `default_tools_approval_mode="approve"` every arm
   call fails with "MCP tool call requires approval, but approval policy is never".
3. `include_apply_patch_tool` is not a valid setting in this version; it is ignored.
4. "Use only the arm tools" made Luna refuse to write notes for a design with no tools. Hence the per-design rule.
5. A stray surrogate character in a phase-1 representation crashed the phase-2 hand-off once. Hence the sanitising.
6. Usage checks in parallel threads deleted each other's session files and logged nothing. Hence the lock.
7. Codex's model catalogue is cached in `.codex/models_cache.json`; `codex debug models` prints it without a call.

## How to check it

- Unit tests (no Codex calls): `python -m pytest -q tests/test_codex_builder.py`
- Lockdown probe (a few calls): `bash scripts/codex_lockdown_test.sh` lists the tools Codex sees, tries to read a
  canary file and `C:/Windows/win.ini`, and reports any non-arm tool call.
- Current allowance: `/status` in the interactive Codex app, or `python -c "from v3.builders.codex_builder import
  usage_snapshot; print(usage_snapshot())"` from `rpp-v3` (one tiny call).
- Installation health: `codex.exe doctor` (no model call).

## Test results at approval (2026-10-21)

- Canary file and `win.ini` unreadable (the human's two manual runs plus two probes).
- Two-phase episodes on K1x1901000 with GPT-6 Luna:
  - SEC_A1 section by section: valid, 1.0;
  - G1C3: valid, 1.0, and it used `evaluate` and `intervene` when answering;
  - ONT_LOOSE: valid, 1.0;
  - A0P notes: 0.67.
- No integrity breaches. Weekly usage at the first guarded reading: 0%.
