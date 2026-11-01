"""Run 5 stage 2 prep (V3-056): the stage 1 evidence Opus sees when it writes the five "+" parts.

Before V3-056 Opus wrote the parts from first principles. Now it also sees a short summary of how the stage 1 winner
did on the GENERATED items, drawn only from what is safe to show (an information firewall, agreed with the human and
a Gemini Argon review on 2026-10-30):

  shown       the winner's overall score on the generated items; its score by system size k; how many builds
              failed the notes check, by k and by the checker's fixed problem categories (quoted names replaced,
              never any notes text); build tool calls by k; output tokens; and (V3-057) how the build tools
              behave, read from the rendered design itself (whether a save replaces the whole notes; what the
              builder is shown of the checker's problems after a save).
  never       anything from the hand-written items (the final check reuses them); scores by problem family (they
              invite domain tricks); the score on passing builds only; any question, question kind, item, answer
              key, notes text or tool trace.

The summary reads only the winner's records on mode.gen and only the fields named in SAFE_FIELDS. Everything it
returns is numbers and strings from CATEGORIES, so nothing from an item can reach Opus through it.
"""
import json
import re
import statistics

# the record fields this module reads; tests check that nothing else is touched
SAFE_FIELDS = ("validity", "build", "output_tokens", "answer_output_tokens", "error")
# the checkers' problem messages, with any quoted name replaced (V3-056: a quoted header could be the model's words)
CATEGORIES = (
    "more than one fact in one element",
    "element longer than N characters",
    "element has no provenance tag",
    "continuation line (fine granularity allows one line per element)",
    "line is not an element (bullet or numbered line)",
    "text before the first header",
    "element before the first header",
    "unknown section header",
    "missing section",
    "section has no content",
    "unfilled template placeholder",
    "the document is empty",
    "no model written",
)
OTHER = "other checker problem"
K_RE = re.compile(r"^K(\d+)x\d+$")


def category(msg):
    """A checker message -> one of CATEGORIES (or OTHER). Quoted text and numbers are dropped first."""
    m = re.sub(r"'[^']*'", "", str(msg))
    m = re.sub(r"\d+", "N", m)
    m = " ".join(m.split())
    return m if m in CATEGORIES else OTHER


def _k(iid):
    m = K_RE.match(iid)
    return int(m.group(1)) if m else None


def _safe(rec):
    return {f: rec.get(f) for f in SAFE_FIELDS} if rec else None


def summary(mode, winner, record, score, items=None):
    """The evidence for the winner (a genome). record(mode, gid, iid) and score(mode, gid, iid) are v3.run5's
    readers. Uses mode.gen only (V3-058: or the given generated items, the previous stage's confirmation items)."""
    from v3.run5 import counts_as_built
    from v3.run5_sets import check_not_retired
    gid = winner["id"]
    items = sorted(mode.gen if items is None else items)
    if any(not K_RE.match(i) for i in items):
        raise ValueError("the evidence uses generated items only")
    check_not_retired(items)
    by_k, all_s = {}, []
    built, failed, cats = 0, 0, {}
    failed_k, checked_k, calls_k = {}, {}, {}
    build_tok, answer_tok = [], []
    for iid in items:
        s = score(mode, gid, iid)
        if s is not None:
            all_s.append(s)
            if _k(iid) is not None:
                by_k.setdefault(_k(iid), []).append(s)
        r = _safe(record(mode, gid, iid))
        if not r or r.get("error"):
            continue
        b = r.get("build") or {}
        ok = counts_as_built(r)
        checked = bool(r.get("validity"))                 # a core with no checker reports no build check
        built += checked and ok
        failed += checked and not ok
        k = _k(iid)
        if checked and k is not None:
            checked_k[k] = checked_k.get(k, 0) + 1
            failed_k[k] = failed_k.get(k, 0) + (not ok)
        if k is not None and isinstance(b.get("tool_calls"), int):
            calls_k.setdefault(k, []).append(b["tool_calls"])
        if checked and not ok:
            for c in sorted({category(h.get("problem")) for h in (r.get("validity") or {}).get("holes") or []}):
                cats[c] = cats.get(c, 0) + 1
        if isinstance(b.get("output_tokens"), int):
            build_tok.append(b["output_tokens"])
        if isinstance(r.get("answer_output_tokens"), int):
            answer_tok.append(r["answer_output_tokens"])

    def tok(xs):
        return {"mean": round(statistics.mean(xs)), "median": round(statistics.median(xs)),
                "p90": round(sorted(xs)[int(0.9 * (len(xs) - 1))])} if xs else None

    return {
        "winner": winner.get("name"),
        "items": len(all_s),
        "score_generated": round(statistics.mean(all_s), 3) if all_s else None,
        "score_by_k": {str(k): round(statistics.mean(v), 3) for k, v in sorted(by_k.items())},
        "builds": {"checked": built + failed, "failed_check": failed,
                   "failed_share": round(failed / (built + failed), 3) if built + failed else None,
                   "failed_builds_by_problem": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
                   "failed_by_k": {str(k): f"{failed_k[k]} of {checked_k[k]}" for k in sorted(checked_k)}},
        "build_tool_calls_by_k": {str(k): {"mean": round(statistics.mean(v), 1), "max": max(v)}
                                  for k, v in sorted(calls_k.items())},
        "output_tokens": {"build": tok(build_tok), "answer": tok(answer_tok)},
        "max_turns": winner["genes"]["numeric"]["max_turns"],
        "tools": tool_facts(winner),
    }


def tool_facts(winner):
    """V3-057: how the winner's build tools behave, read from the rendered design (never assumed). V3-056 told Opus
    the builder sees no checker problems; in fact v3.genome.render appends the check to every save. save_replaces:
    the save tool's own description says it replaces the whole notes. check_feedback: what a save shows of the
    checker's problems (tool_interface.errors: first_only, all, all_with_fix_hints), or None if it shows none."""
    from v3.genome.render import render
    arm = render(winner)
    if arm.prose:
        return {"save_tool": None, "save_replaces": None, "check_feedback": None}
    sub = arm.spec.get("submit")
    desc = next((t["function"]["description"] for t in arm.tools() if t["function"]["name"] == sub), "")
    shows = arm.v3_core or arm.ti["errors"] != "first_only"          # mirrors render._core_call
    return {"save_tool": sub if not arm.has_draft else None,
            "save_replaces": bool(re.search(r"\breplaces?\b", desc, re.I)) and not arm.has_draft,
            "check_feedback": arm.ti["errors"] if shows else None}


def render(ev):
    """The evidence as the text block added to Opus's brief."""
    from v3.cores.sectioned import MAX_FINE_CHARS
    b = ev["builds"]
    probs = "; ".join(f"{c.replace(' N ', f' {MAX_FINE_CHARS} ')}: {n}"
                      for c, n in b["failed_builds_by_problem"].items()) or "none"
    ks = ", ".join(f"k={k}: {v}" for k, v in ev["score_by_k"].items())
    t = ev["output_tokens"]
    tk = lambda d: f"mean {d['mean']}, median {d['median']}, 90th percentile {d['p90']}" if d else "not recorded"
    fk = ", ".join(f"k={k}: {v}" for k, v in b["failed_by_k"].items()) or "none"
    ck = ", ".join(f"k={k}: mean {v['mean']}, max {v['max']}" for k, v in ev["build_tool_calls_by_k"].items()) or "none"
    tf = ev.get("tools") or {}
    shown = {"first_only": "the first checker problem (with its line number) and the count of how many more problems "
                           "remain",
             "all": "every checker problem, with line numbers",
             "all_with_fix_hints": "every checker problem, with line numbers and a hint for fixing each"}
    tools = []
    if tf.get("save_replaces"):
        tools.append("Each save replaces the whole notes; there is no append. So a part that asks the builder to save "
                     "in several passes makes it rewrite everything written so far on every save.")
    if tf.get("check_feedback") in shown:
        tools.append(f"After each save, the builder is shown {shown[tf['check_feedback']]}."
                     + (" So fixing reported problems one save at a time uses up tool calls from the budget."
                        if tf["check_feedback"] == "first_only" else ""))
    elif tf.get("save_tool"):
        tools.append("After a save, the builder is shown no checker problems.")
    tool_line = ("- How the build tools work (from the design itself, not from stage 1 results): " + " ".join(tools)
                 + "\n") if tools else ""
    return (
        f"- Mean score (0 to 1) on the {ev['items']} generated items: {ev['score_generated']}.\n"
        f"- Mean score by system size k (k = 1 is one system; k = 2, 4, 8 combine that many in one description): "
        f"{ks}.\n"
        f"- Notes check: {b['failed_check']} of {b['checked']} builds ({100 * (b['failed_share'] or 0):.1f}%) ended "
        f"with notes that failed the layout checker. Number of failed builds with each checker problem (a build can "
        f"have several): {probs}. Failed builds by system size: {fk}.\n"
        f"{tool_line}"
        f"- Tool calls while building the notes, by system size (no call is an automatic retry): {ck}.\n"
        f"- Output tokens per item: building the notes {tk(t['build'])}; answering {tk(t['answer'])}.\n"
        f"- Current tool-call budget for building (max_turns): {ev['max_turns']}.")


def dumps(ev):
    return json.dumps(ev, indent=1)


# ---------------------------------------------------------------- V3-058: the evidence for stage n
EXAMPLE_LESSON = ("- Lesson from stage 2: its worked example was written in the old fact syntax but shown beside new "
                  "fact-writing rules in some designs, so the two contradicted each other. From stage 3 on, every "
                  "worked example is rewritten to match its design's own layout and fact-writing rules.")


def _clip(text, n=1500):
    t = json.dumps(text) if isinstance(text, dict) else str(text or "")
    return t if len(t) <= n else t[:n] + " [...]"


def render_history(hist):
    """Earlier stages' candidates with what was measured (design texts and numbers only). hist: list of {stage,
    part, archetype, description, text, effect, range, length, verdict, full}; full texts for advancing candidates
    and the worst candidate per part, a one-line row for the rest."""
    if not hist:
        return ""
    lines = ["WHAT EARLIER STAGES TRIED (effect = mean score with the part changed minus the design it was compared "
             "with; 95% range; on generated items only)"]
    for h in hist:
        eff = "not measured" if h.get("effect") is None else f"{h['effect']:+.3f} {h.get('range')}"
        head = (f"- stage {h['stage']}, {h['part']}" + (f", approach {h['archetype']}" if h.get("archetype") else "")
                + f": {eff}; {h.get('length', 0)} characters; {h.get('verdict', '')}".rstrip("; "))
        if h.get("full"):
            lines.append(head + "\n  text: " + _clip(h.get("text")).replace("\n", "\n  "))
        else:
            lines.append(head + (f" - {h['description']}" if h.get("description") else ""))
    return "\n".join(lines)


def stage_brief(stage, ev, hist):
    """The evidence block for the stage n writer: the current best's evidence on the previous stage's confirmation
    items, then the history of earlier candidates, then the worked-example lesson."""
    return (f"WHAT IS KNOWN ABOUT THE CURRENT BEST DESIGN (before stage {stage}; generated items only; the builder is a "
            f"small model)\n" + render(ev) + "\n\n" + render_history(hist) + "\n" + EXAMPLE_LESSON)
