"""One episode = one genome × one item × one builder, graded (spec §7, §8, §9).

Both builders get the same prompts: the v2 tournament system prompt (common instructions + the arm's guide, built
from the genome's guidance gene) and the v2 user prompt (description + question). Effort is recorded the same way for
both: output tokens, model turns and tool calls.

Two-phase mode (V3-007, config evolution.episode_mode = "two_phase"): a build phase sees the description but no
question and must write the representation (prose designs: plain-language notes); a fresh answer phase sees the
questions and only that representation (loaded in the query tools, or the notes), never the description. The score
then measures how well the representation captures the system: a builder cannot skip it and reason from the text.
The built representation is stored as "representation" ("rep" is the repeat index).
"""
import json
import pathlib
import time

from v3 import items as I
from v3.config import CFG, RUNS
from v3.grading.grade import grade

BUILDERS = ("claude", "qwen", "codex")      # plus run 3's transfer models (config final.transfer_models): "or_<name>"
                                            # on OpenRouter, "cl_<name>" on the Claude subscription (V3-034, V3-035)


def openrouter_model(builder):
    """The OpenRouter model id of a run 3 transfer builder ("or_<name>", V3-034), or None."""
    if not builder.startswith("or_"):
        return None
    return ((CFG.get("final") or {}).get("transfer_models") or {}).get(builder)


def claude_model(builder):
    """The Claude model of a run 3 transfer builder on the subscription ("cl_<name>": "claude:<model>", V3-035), or
    None."""
    if not builder.startswith("cl_"):
        return None
    v = ((CFG.get("final") or {}).get("transfer_models") or {}).get(builder) or ""
    return v.split(":", 1)[1] if v.startswith("claude:") else None


def transfer_builder(builder):
    """A final C (transfer) builder: its answers are read leniently (V3-035)."""
    return bool(openrouter_model(builder) or claude_model(builder))

BUILD_TASK = ("There is no question yet. Build your representation of this system now, as described above, so that "
              "questions about the system can be answered from it later. Those questions will be answered from your "
              "representation alone: this description will not be available then. Capture everything a question "
              "might need: the parts, their states and values, the rules, conditions and timing. When the "
              "representation is complete, end with FINAL ANSWER: DONE.")
BUILD_TASK_PROSE = ("There is no question yet. Write notes on this system in plain language so that questions about it "
                    "can be answered from your notes later. Those questions will be answered from your notes alone: "
                    "this description will not be available then. Capture everything a question might need: the "
                    "parts, their states and values, the rules, conditions and timing. Give the complete notes after "
                    "FINAL ANSWER.")
REVISION_TASK = ("Here are the notes you wrote. Check them against the description above: correct anything wrong, add "
                 "anything a question might need that is missing (parts, states and values, rules, conditions, "
                 "timing), and give the corrected, complete notes after FINAL ANSWER.\n\nYOUR NOTES:\n")
NO_DESCRIPTION = ("(The original description is not available. A representation of the system was built from it "
                  "earlier; answer only from that representation.)")
NOTES_HEAD = "NOTES WRITTEN EARLIER FROM THE DESCRIPTION (the description itself is not available):\n\n"


_GATE = None
_CLAUDE_CALLS = [0]
_CLAUDE_LOCK = __import__("threading").Lock()


def claude_gate():
    """One process-wide throttle for every Claude call (both phases, every command): CFG claude_runs_per_hour; plus
    the weekly-allowance guard (V3-029): on the first call and every claude_usage_check_every calls the plan's weekly
    used % is read (logged to runs/claude_usage.jsonl); at or above claude_weekly_pause_percent, or when the plan
    reports the limit reached, Claude work stops (v3.run.Paused)."""
    global _GATE
    from v3.run import Paused, Throttle
    if _GATE is None:
        _GATE = Throttle(CFG["budget"]["claude_runs_per_hour"])
    _GATE.wait()
    b = CFG["budget"]
    if "claude_weekly_pause_percent" not in b:
        return
    with _CLAUDE_LOCK:
        if _CLAUDE_CALLS[0] % int(b.get("claude_usage_check_every", 40)) == 0:
            from v3.builders.claude_builder import usage_snapshot
            snap = usage_snapshot()
            with open(RUNS / "claude_usage.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(snap or {"t": time.time(), "source": "guard", "error": "no snapshot"},
                                        calls_so_far=_CLAUDE_CALLS[0])) + "\n")
            used = (snap or {}).get("seven_day")
            if (snap or {}).get("status") == "rejected" or (used is not None
                                                            and used >= b["claude_weekly_pause_percent"]):
                raise Paused(f"Claude weekly allowance guard: {used}% used "
                             f"(pause at {b['claude_weekly_pause_percent']}%)")
        _CLAUDE_CALLS[0] += 1


_CODEX_GATE = None


_CODEX_CALLS = [0]
_CODEX_LOCK = __import__("threading").Lock()


def codex_gate_throttle():
    """The process-wide Codex throttle (created on first use), so a stage can change its hourly cap (V3-025)."""
    global _CODEX_GATE
    from v3.run import Throttle
    if _CODEX_GATE is None:
        _CODEX_GATE = Throttle(CFG["budget"].get("codex_runs_per_hour", 60))
    return _CODEX_GATE


def codex_gate():
    """Process-wide throttle for Codex calls (CFG budget codex_runs_per_hour) plus the weekly-allowance guard: on the
    first call and every codex_usage_check_every calls the account's weekly used % is read (logged to
    runs/codex_usage.jsonl); at or above codex_weekly_pause_percent Codex work stops (v3.run.Paused)."""
    global _CODEX_GATE
    from v3.run import Paused, Throttle
    if _CODEX_GATE is None:
        _CODEX_GATE = Throttle(CFG["budget"].get("codex_runs_per_hour", 60))
    _CODEX_GATE.wait()
    b = CFG["budget"]
    with _CODEX_LOCK:          # one usage check at a time; parallel checks would delete each other's session files
        n_call = _CODEX_CALLS[0]
        _CODEX_CALLS[0] += 1                          # counted first: a failed usage check cannot stall the queue
        if n_call % int(b.get("codex_usage_check_every", 40)) == 0:
            from v3.builders.codex_builder import usage_snapshot
            snap = usage_snapshot()
            with open(RUNS / "codex_usage.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(snap or {"error": "no snapshot"}, calls_so_far=_CODEX_CALLS[0])) + "\n")
            used = (snap or {}).get("weekly_used_percent")
            if (snap or {}).get("limit_reached") or (used is not None
                                                     and used >= b.get("codex_weekly_pause_percent", 70)):
                _CODEX_CALLS[0] = n_call                  # a paused call does not count: the next call checks again
                raise Paused(f"Codex weekly allowance guard: {used}% used "
                             f"(pause at {b.get('codex_weekly_pause_percent', 70)}%)")


def two_phase():
    return CFG["evolution"].get("episode_mode", "single") == "two_phase"


def prompts(arm, item, given_text=None):
    from tournament.run import system_prompt, user_prompt
    with_tools = (not getattr(arm, "custom_episode", False)) and bool(arm.tools())
    return system_prompt(arm, with_tools), user_prompt(item, item["context"], given_text)


def record_dir(phase, builder, genome_id, item_id, rep=0):
    return RUNS / phase / builder / genome_id / (f"{item_id}" + (f".r{rep}" if rep else ""))


def _raw(builder, genome, arm, item, d, given_rep=None, client=None, cap_check=None):
    """One builder call, not graded. The representation comes back as "representation"."""
    limits = arm.limits() if hasattr(arm, "limits") else {}
    max_tools = int(limits.get("max_turns", 12))
    if builder == "claude" or claude_model(builder):
        from v3.builders import claude_builder
        claude_gate()
        sysp, user = prompts(arm, item, arm.render(arm.load(given_rep, given_gold=True)) if given_rep else None)
        rec = claude_builder.run(genome, sysp, user, d, given_rep=given_rep, max_tool_calls=max_tools,
                                 model=claude_model(builder))
        rec["builder"] = builder
    elif builder == "codex":
        from v3.builders import codex_builder
        if arm.genes["core"]["kind"] in codex_builder.SANDBOXED_CORES:
            from v3 import sandbox                   # code review: Docker down is an outage, not a config error
            if not sandbox.available():
                return {"builder": "codex", "error": "SANDBOX UNAVAILABLE: Docker is not available",
                        "final_raw": None}
        if not codex_builder.usable_for(arm):
            return {"builder": "codex", "error": "design tools outside the approved Codex allowlist", "final_raw": None}
        codex_gate()
        sysp, user = prompts(arm, item, arm.render(arm.load(given_rep, given_gold=True)) if given_rep else None)
        rec = codex_builder.run(genome, sysp, user, d, given_rep=given_rep, max_tool_calls=max_tools)
    elif builder == "qwen" or openrouter_model(builder):
        from v3.llm import chat
        from tournament.run import run_episode
        model = openrouter_model(builder) or CFG["models"]["builder_openrouter"]
        for attempt in range(3):    # an empty provider response is retried (V3-008); other errors are recorded
            try:
                rec = run_episode(arm, item, item["context"], model, client or chat,
                                  given_rep=given_rep, cap_check=cap_check, purpose="v3_build")
                rec["builder"], rec["model"] = builder, model
                rec["output_tokens"] = rec.get("output_tokens") or rec.get("tokens_out")
                rec["attempts"] = attempt + 1
                break
            except Exception as e:  # a generation error is recorded and retried later, never graded as 0
                rec = {"builder": builder, "error": f"{type(e).__name__}: {e}", "final_raw": None,
                       "attempts": attempt + 1}
                if type(e).__name__ != "EmptyResponse":
                    break
    else:
        raise ValueError(builder)
    rec["representation"] = rec.pop("rep", None)
    return rec


def _answer_item(genome, arm, item, b):
    """The answer-phase item and the representation to load in the tools (None: shown as text or absent)."""
    rep, it = b.get("representation"), dict(item, context=NO_DESCRIPTION)
    if genome["genes"]["core"]["kind"] == "prose":
        it["context"] = NOTES_HEAD + (b.get("final_raw") or "(no notes were written)")
        return it, None
    if rep is None:
        it["context"] = NO_DESCRIPTION + "\n\n(No representation was built.)"
        return it, None
    try:
        arm.render(arm.load(rep, given_gold=True))
        return it, rep
    except Exception:           # an unloadable representation is shown as text, without tools
        it["context"] = NO_DESCRIPTION + "\n\nREPRESENTATION (could not be loaded into the tools):\n" + str(rep)
        return it, None


def _two_phase(builder, genome, arm, item, d, client, cap_check):
    """Build (description, no question), then answer (question, representation only). The answer record is returned
    with the build summarised under "build"; validity, representation and buildability come from the build."""
    prose = genome["genes"]["core"]["kind"] == "prose"
    b = _raw(builder, genome, arm, dict(item, question=BUILD_TASK_PROSE if prose else BUILD_TASK, task_type="build"),
             d / "build", None, client, cap_check)
    summary = {k: b.get(k) for k in ("validity", "tool_calls", "turns", "output_tokens", "wall_s", "rc", "integrity",
                                     "error")}
    if b.get("error"):
        return {"builder": builder, "error": "build: " + str(b["error"]), "final_raw": None, "build": summary}
    from v3.genome.render import PROSE_BASE_TURNS
    if prose and arm.limits()["max_turns"] > PROSE_BASE_TURNS:   # run 5 factor E+ for prose: one revision turn
        r = _raw(builder, genome, arm, dict(item, question=REVISION_TASK + (b.get("final_raw") or "(none)"),
                                            task_type="build"), d / "revise", None, client, cap_check)
        if r.get("error"):
            return {"builder": builder, "error": "revise: " + str(r["error"]), "final_raw": None, "build": summary}
        summary["revision_output_tokens"] = r.get("output_tokens")
        b = dict(b, final_raw=r.get("final_raw"), output_tokens=(b.get("output_tokens") or 0)
                 + (r.get("output_tokens") or 0))
    ans_item, given = _answer_item(genome, arm, item, b)
    a = _raw(builder, genome, arm, ans_item, d / "answer", given, client, cap_check)
    out = dict(a, build=summary, representation=b.get("representation"), validity=None if prose else b.get("validity"),
               output_tokens=(a.get("output_tokens") or 0) + (b.get("output_tokens") or 0),
               answer_output_tokens=a.get("output_tokens"), mode="two_phase")
    if prose:
        out["notes"] = b.get("final_raw")
    if b.get("integrity"):
        out["error"] = "INTEGRITY (build): " + b["integrity"][0]
    return out


def run(genome, item, builder, phase, *, rep=0, given_rep=None, client=None, grade_client=None, cap_check=None):
    """Run, grade and store one episode. Returns the record. Idempotent: an existing complete record is returned."""
    from v3.genome.render import render
    d = record_dir(phase, builder, genome["id"], item["id"], rep)
    path = d / "record.json"
    if path.exists():
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("grade", {}).get("status") == "ok" and not r.get("error"):
            return r
        # V3-031: a failed attempt is kept, never overwritten: record.json becomes record.void<N>.json
        path.rename(d / f"record.void{sum(1 for _ in d.glob('record.void*.json'))}.json")
    arm = render(genome)
    t0 = time.time()
    if two_phase() and given_rep is None:
        rec = _two_phase(builder, genome, arm, item, d, client, cap_check)
    else:
        rec = _raw(builder, genome, arm, item, d, given_rep, client, cap_check)
    rec.update({"genome": genome["id"], "item": item["id"], "system": item["system"], "phase": phase, "rep": rep,
                "bank": item.get("bank"), "wall_s_total": time.time() - t0})
    rec["grade"] = {"status": "not_graded"} if rec.get("error") else \
        grade(item, rec.get("final_raw"), client=grade_client, cap_check=cap_check, lenient=transfer_builder(builder))
    d.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return rec


def buildable(rec):
    """Buildability signal for one record: a usable, executable representation was produced (prose: answered, or in
    two-phase mode wrote notes)."""
    v = rec.get("validity") or {}
    if not v:
        return bool(rec.get("notes") or (rec.get("representation") is None and rec.get("final_raw")))
    return bool(v.get("usable") and v.get("executable", True))
