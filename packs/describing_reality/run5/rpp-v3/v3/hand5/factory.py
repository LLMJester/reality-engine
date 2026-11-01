"""Build and check the run 5 hand-written-style items. Resumable: every step's result is a file, and a step whose
file exists is not repeated.

  python -m v3.hand5.factory write  [--only H001,H002] [--workers 8]   writer sessions (Claude Sonnet, Claude plan)
  python -m v3.hand5.factory check  [--workers 8]                      validator, judge, leak check, adjudicator
  python -m v3.hand5.factory rewrite [--workers 8]                    one rewrite of systems with a failed item
  python -m v3.hand5.factory select                                     the final 500 (+ the 6 pilot items)
  python -m v3.hand5.factory status

Checks per item (pre-specified, V3-042):
  validator    a fresh Sonnet session, given the description, answers the question (no tools)
  judge        DeepSeek v4.1 Flash grades that answer against the key (the run's own grader)
  leak         Claude Sonnet answers WITHOUT the description (code review: a model at least as capable as the
               writer, not the judge grading its own answer); the judge grades it; must pass leak_ok
  adjudicator  a fresh Sonnet session sees the item, description, simulator, computed keys, the validator's answer and
               its grade, and returns keep / fix_key (with a corrected key) / exclude (v2 adjudicator brief)
An item is accepted if the adjudicator keeps it (or fixes its key) and it passes the leak check. V3-044 (Argon review):
the leak check passes only if the no-document answer scores at most LEAK_MAX = 0.25 AND misses the item's
highest-weight key point (a "hit"; partial credit from general reasoning is allowed); the final 500 are chosen by quota per question type (first accepted in plan order within
each type), so a type that fails more often (diagnosis) is not under-represented.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from v3.config import ROOT
from v3.hand5 import plan as P
from v3.hand5.writer import brief

OUT = ROOT / "items" / "hand5"
WORK = OUT / "work"            # writer workspaces (sim.py, description, items, keys, trace)
CHECKS = OUT / "checks"        # one file per item: validator answer, grades, leak, adjudication
FINAL = OUT / "items.json"     # the 500
PILOT_FILE = OUT / "pilot_items.json"
LEAK_MAX = 0.25          # V3-044 (was 0.5)
MODEL = "sonnet"
CLAUDE_WEEKLY_PAUSE = 85.0
CLAUDE_FIVE_HOUR_PAUSE = 85.0
_lock = threading.Lock()

# The run's own answer instruction (rpp-core harness/conditions/prompts/system_common.txt): the grader reads only the
# text after the FINAL ANSWER line (harness/strip.py final_answer), so checkers must end the same way.
FINAL_RULE = ("End your response with a line containing only the words FINAL ANSWER, followed by your answer in plain "
              "prose. The final answer must be self-contained and specific: give the concrete values, causes, "
              "mechanisms, conflicts, assumptions or omissions that the question asks for.")
VALIDATOR = ("You are a careful analyst of systems and situations. Read the system description, build a mental model "
             "of how the system works and reason through the question step by step. If the description lacks "
             "something you need, say what is missing and how it would change the answer. " + FINAL_RULE)
LEAKER = "Answer the question as well as you can. " + FINAL_RULE
ADJUDICATOR = """You adjudicate one evaluation item for a study of AI reasoning about systems. You see the item
(question, answer key, construction notes), the description the model under test sees, the ground truth (the
simulator code and its computed output) and an independent validator's answer written from the description alone,
with its grade (0 to 1) against the key.

Decide one verdict:
- keep: the key is right and the item is fair: a careful reasoner can get it from the description (validator agreement
  is good evidence, but a validator failing for the right reason - the item needs modelling and it modelled
  imperfectly - is fine).
- fix_key: the key is wrong or mis-stated relative to the simulator output, or its tolerances are tighter than the
  description supports; give the corrected answer_key (same fields: summary, key_points with id/text/weight,
  wrong_answers). Keys must agree with the simulator output.
- exclude: the description has an unintended gap, ambiguity or contradiction that makes the key unreachable or the
  question unfair (deliberate omissions and conflicts recorded in the notes are the point of completeness and
  consistency items and are fine); or the item can be answered by finding one sentence or keyword matching without
  modelling; or the key cannot be determined.

Reply with ONE JSON object and nothing else:
{"verdict": "keep" | "fix_key" | "exclude", "validator_agrees": "yes" | "partly" | "no", "reason": "one or two
sentences", "answer_key": {...only for fix_key...}}"""


# ---------------------------------------------------------------- Claude (plan) and DeepSeek calls
def claude_exe():
    from v3.builders.claude_builder import claude_exe as ce
    return ce()


def _parse(lines):
    from v3.builders.claude_builder import parse_stream
    return parse_stream(lines)


def claude_text(system, user, timeout=900, model=None):
    """One headless call (Sonnet unless model is given) with no tools, no settings and no session (Claude plan)."""
    work = tempfile.mkdtemp(prefix="h5c_")
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    # stray surrogate characters in a document cannot be encoded for the call (as in claude_builder.run)
    system = system.encode("utf-8", "replace").decode("utf-8")
    user = user.encode("utf-8", "replace").decode("utf-8")
    try:
        p = subprocess.run([claude_exe(), "-p", "--output-format", "stream-json", "--verbose", "--model", model or MODEL,
                            "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
                            "--system-prompt", system], input=user, cwd=work, capture_output=True, text=True,
                           encoding="utf-8", timeout=timeout, env=env)
        r = _parse(p.stdout.splitlines())
        if r["error"] or not r["final_raw"].strip():
            raise RuntimeError(f"claude call failed: {r['error']} {p.stderr[-500:]}")
        return r
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def claude_guard():
    """Stop Claude-plan work at 85% of the weekly window or of the 5-hour window (the human's interactive sessions
    share the 5-hour window; 2026-10-29 it reached 95% during the item rewrites)."""
    from v3.builders.claude_builder import usage_snapshot
    s = usage_snapshot()
    if s and (s.get("seven_day") or 0) >= CLAUDE_WEEKLY_PAUSE:
        raise SystemExit(f"Claude weekly use {s['seven_day']}% >= {CLAUDE_WEEKLY_PAUSE}%: stopping (resume later)")
    if s and (s.get("five_hour") or 0) >= CLAUDE_FIVE_HOUR_PAUSE:
        raise SystemExit(f"Claude 5-hour use {s['five_hour']}% >= {CLAUDE_FIVE_HOUR_PAUSE}%: stopping (resume after "
                         f"the window resets)")
    return s


def deepseek(system, user):
    from v3.grading.grade import grader_model
    from v3.llm import chat
    msg, usage = chat(grader_model(), [{"role": "system", "content": system}, {"role": "user", "content": user}],
                      purpose="hand5_leak", seed=7, max_tokens=4000)
    return msg.get("content") or "", usage


def _leak_done(rec):
    """The leak check counts only if Sonnet answered it and the judge graded it."""
    lk = rec.get("leak") or {}
    return lk.get("by") == MODEL and (lk.get("grade") or {}).get("status") == "ok"


def usable_fixed_key(k, item=None):
    """The adjudicator's corrected key if usable (a summary, key points each with id and text), with the original
    wrong answers kept when it gives none (sixth code review); else None."""
    if not isinstance(k, dict):
        return None
    kps = k.get("key_points")
    if not (isinstance(k.get("summary"), str) and k["summary"].strip() and isinstance(kps, list) and kps
            and all(isinstance(p, dict) and p.get("id") and p.get("text") for p in kps)):
        return None
    k = dict(k)
    if not k.get("wrong_answers"):
        k["wrong_answers"] = ((item or {}).get("answer_key") or {}).get("wrong_answers") or []
    return k


def effective_item(rec, item):
    """The item with the key it will be graded against: the adjudicator's corrected key for a fix_key verdict."""
    adj = rec.get("adjudication") or {}
    if adj.get("verdict") == "fix_key":
        k = usable_fixed_key(adj.get("answer_key"), item)
        if k:
            return dict(item, answer_key=k)
    return item


def leak_grade(rec, item):
    """The leak answer's grade against the effective key (re-graded for a corrected key, code review)."""
    if effective_item(rec, item) is not item:
        return rec.get("leak_fixed") or {}
    return (rec.get("leak") or {}).get("grade") or {}


def leak_ok(rec, item):
    """The leak check (V3-044, code review): the no-document answer, graded against the effective key, scores
    <= LEAK_MAX and hits none of the key points tied for the highest weight (partial credit allowed)."""
    g = leak_grade(rec, item)
    if g.get("status") != "ok" or g.get("score") is None or g["score"] > LEAK_MAX:
        return False
    kps = (effective_item(rec, item).get("answer_key") or {}).get("key_points") or []
    if not kps:
        return True
    top_w = max(float(k.get("weight", 1.0)) for k in kps)
    top = {k["id"] for k in kps if float(k.get("weight", 1.0)) >= top_w - 1e-9}
    verdicts = {k.get("id"): k.get("verdict") for k in ((g.get("grade") or {}).get("key_points") or [])}
    return all(verdicts.get(t) != "hit" for t in top)


def is_accepted(rec, item):
    adj = rec.get("adjudication") or {}
    kps = (effective_item(rec, item).get("answer_key") or {}).get("key_points") or []
    weights_ok = all(isinstance(k.get("weight", 1.0), (int, float)) and float(k.get("weight", 1.0)) > 0 for k in kps)
    return adj.get("verdict") in ("keep", "fix_key") and _leak_done(rec) and leak_ok(rec, item) and weights_ok


def judge(item, context, answer):
    from v3.grading.grade import grade
    return grade(dict(item, context=context, grading="ai"), answer)


# ---------------------------------------------------------------- writing
def _write_one(a, timeout=3600, feedback=None):
    from v3 import walls
    out = WORK / f"{a['system_id']}.json"
    if feedback is None and out.exists() and json.loads(out.read_text(encoding="utf-8")).get("accepted"):
        return "already accepted"
    work = pathlib.Path(tempfile.mkdtemp(prefix="h5w_"))
    try:
        (work / "assignment.json").write_text(json.dumps(a), encoding="utf-8")
        args = [sys.executable, "-u", "-m", "v3.hand5.writer", "--out", str(out), "--assignment",
                str(work / "assignment.json")]
        if out.exists():
            args += ["--resume"]
        (work / "mcp.json").write_text(json.dumps({"mcpServers": {"writer": {
            "command": args[0], "args": args[1:], "env": {"PYTHONPATH": str(ROOT)}}}}), encoding="utf-8")
        (work / "settings.json").write_text(json.dumps({"permissions": {"deny": walls.deny_rules("builder") + [
            "Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebFetch", "WebSearch", "Agent", "Task",
            "NotebookEdit"]}}), encoding="utf-8")
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)
        cmd = [claude_exe(), "-p", "--output-format", "stream-json", "--verbose", "--model", MODEL,
               "--tools", "", "--strict-mcp-config", "--mcp-config", str(work / "mcp.json"),
               "--allowedTools", "mcp__writer", "--permission-mode", "default",
               "--setting-sources", "project", "--settings", str(work / "settings.json"),
               "--no-session-persistence", "--system-prompt", brief(a)]
        try:
            prompt = (feedback if feedback else
                      "Your files from an earlier session are saved: check them with show_file, call run_sim, fix "
                      "what is needed and submit." if out.exists() else
                      "Build the system and its three items now, then submit.")
            p = subprocess.run(cmd, input=prompt, cwd=work,
                               capture_output=True, text=True, encoding="utf-8", timeout=timeout, env=env)
            lines = p.stdout.splitlines()
        except subprocess.TimeoutExpired:
            lines = []
        (OUT / "transcripts").mkdir(parents=True, exist_ok=True)
        (OUT / "transcripts" / f"{a['system_id']}.jsonl").write_text("\n".join(lines), encoding="utf-8")
        doc = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        return "accepted" if doc.get("accepted") else "not accepted"
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def write(only=None, workers=8, tries=2):
    WORK.mkdir(parents=True, exist_ok=True)
    todo = [a for a in P.plan() if not only or a["system_id"] in only]
    claude_guard()
    done = [0]

    def one(a):
        res = "?"
        for t in range(tries):
            res = _write_one(a)
            if res != "not accepted":
                break
        with _lock:
            done[0] += 1
            print(f"[{done[0]}/{len(todo)}] {a['system_id']}: {res}", flush=True)
            if done[0] % 20 == 0:
                try:
                    claude_guard()
                except SystemExit as e:                    # stop starting new work; running jobs finish
                    print(f"stopping: {e}", flush=True)
                    stop.set()
        return res
    stop = threading.Event()
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(lambda a: "not started (Claude guard)" if stop.is_set() else one(a), todo))
    if stop.is_set():
        raise SystemExit("writing stopped by the Claude guard; run again after the window resets")
    print({r: res.count(r) for r in set(res)})


# ---------------------------------------------------------------- checking
def system_items(sid):
    doc = json.loads((WORK / f"{sid}.json").read_text(encoding="utf-8"))
    if not doc.get("accepted"):
        return None, None, None
    return doc, json.loads(doc["files"]["items.json"]), doc["files"]["description.md"]


def _check_item(doc, item, desc):
    f = CHECKS / f"{item['id']}.json"
    rec = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"id": item["id"]}

    def save():
        f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if "validator" not in rec:
        r = claude_text(VALIDATOR, f"SYSTEM DESCRIPTION\n\n{desc}\n\nQUESTION\n\n{item['question']}")
        rec["validator"] = r["final_raw"]
        save()
    if (rec.get("validator_grade") or {}).get("status") != "ok":
        g = judge(item, desc, rec["validator"])
        if g.get("status") != "ok":
            raise RuntimeError("the judge failed on the validator answer; retried on the next check")
        rec["validator_grade"] = g
        save()
    if not _leak_done(rec):
        rec.pop("leak_fixed", None)                       # a new leak answer needs a new corrected-key grade
        if "leak" in rec and "leak_deepseek" not in rec:
            rec["leak_deepseek"] = rec.pop("leak")            # the earlier DeepSeek leak check, kept on record
        ans = claude_text(LEAKER, item["question"])["final_raw"]
        g = judge(item, desc, ans)
        if g.get("status") != "ok":
            raise RuntimeError("the judge failed on the leak answer; retried on the next check")
        rec["leak"] = {"answer": ans, "grade": g, "by": MODEL}
        save()
    if "adjudication" not in rec:
        user = (f"ITEM\n{json.dumps(item, indent=1)}\n\nDESCRIPTION\n{desc}\n\nSIMULATOR (sim.py)\n"
                f"{doc['files']['sim.py']}\n\nSIMULATOR OUTPUT\n{json.dumps((doc.get('keys') or {}).get('output'), indent=1)[:12000]}"
                f"\n\nVALIDATOR ANSWER (grade {rec['validator_grade'].get('score')})\n{rec['validator']}")
        r = claude_text(ADJUDICATOR, user)
        txt = r["final_raw"]
        from v3.jsonx import objects
        verdicts = [o for o in objects(txt) if "verdict" in o]     # code review: robust to braces around the JSON
        if not verdicts:
            raise RuntimeError("the adjudicator reply had no verdict object; retried on the next check")
        rec["adjudication"] = verdicts[-1]
        save()
    adj = rec["adjudication"]
    if adj.get("verdict") == "fix_key":
        k = usable_fixed_key(adj.get("answer_key"), item)
        if not k:
            adj["verdict"], adj["reason"] = "exclude", "fix_key without a usable corrected key"
            save()
        else:
            adj["answer_key"] = k                         # stored with the original wrong answers kept
            fixed = dict(item, answer_key=k)
            if (rec.get("fixed_grade") or {}).get("status") != "ok":
                g = judge(fixed, desc, rec["validator"])
                if g.get("status") != "ok":
                    raise RuntimeError("the judge failed on the corrected key; retried on the next check")
                rec["fixed_grade"] = g
                save()
            if _leak_done(rec) and (rec.get("leak_fixed") or {}).get("status") != "ok":
                g = judge(fixed, desc, rec["leak"]["answer"])
                if g.get("status") != "ok":
                    raise RuntimeError("the judge failed on the leak answer for the corrected key; retried later")
                rec["leak_fixed"] = g
                save()
    rec["accepted"] = is_accepted(rec, item)
    save()
    return rec


def check(workers=8, only=None):
    CHECKS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for a in P.plan():
        if only and a["system_id"] not in only:
            continue
        if not (WORK / f"{a['system_id']}.json").exists():
            continue
        doc, items, desc = system_items(a["system_id"])
        if doc:
            jobs += [(doc, it, desc) for it in items]
    claude_guard()
    done = [0]

    def one(j):
        try:
            rec = _check_item(*j)
            res = "accepted" if rec["accepted"] else "rejected"
        except Exception as e:
            res = f"error {type(e).__name__}: {str(e)[:200]}"
        with _lock:
            done[0] += 1
            print(f"[{done[0]}/{len(jobs)}] {j[1]['id']}: {res}", flush=True)
            if done[0] % 40 == 0:
                try:
                    claude_guard()
                except SystemExit as e:                    # stop starting new work; running jobs finish
                    print(f"stopping: {e}", flush=True)
                    stop.set()
        return res
    stop = threading.Event()
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(lambda j: "not started: Claude guard" if stop.is_set() else one(j), jobs))
    if stop.is_set():
        print({r.split(':')[0]: sum(1 for x in res if x.split(':')[0] == r.split(':')[0]) for r in res})
        raise SystemExit("checking stopped by the Claude guard; run again after the window resets")
    print({r.split(':')[0]: sum(1 for x in res if x.split(':')[0] == r.split(':')[0]) for r in res})


# ---------------------------------------------------------------- rewriting failed items (once per system)
REWRITES = OUT / "rewrites.json"
REWRITE_INTRO = """Your system's files are saved (check them with show_file). Some of its items FAILED the quality checks:

{failures}

Rewrite each failed item so that it can only be answered by modelling THIS system from the description:
- A leak means a model that never saw the description answered the question well. Usually the question itself
  gives the answer away (for a diagnosis: symptoms that point to a textbook cause on their own). Make the decisive
  evidence live in the description, not in the question: the question should state only what is observed, and the
  cause, value or condition must follow from facts, numbers and rules found only in the description, with plausible
  decoys that only those facts rule out.
- An exclusion means the item was unfair or the key wrong; fix the cause the adjudicator named.
Keep the items that passed unchanged unless a description change requires updating them. Change sim.py and the
description only as much as needed, call run_sim, make every key agree with its output, then submit."""


def _item(sid, n):
    f = WORK / f"{sid}.json"
    if not f.exists():
        return None
    doc = json.loads(f.read_text(encoding="utf-8"))
    try:
        return next(it for it in json.loads(doc["files"]["items.json"]) if it["id"] == f"{sid}-{n}")
    except (ValueError, StopIteration, KeyError):
        return None


def _failures(sid):
    out = []
    for n in (1, 2, 3):
        f = CHECKS / f"{sid}-{n}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        item = _item(sid, n)
        if "adjudication" not in r or not _leak_done(r) or item is None or is_accepted(r, item):
            continue
        leak = leak_grade(r, item).get("score")
        leak_s = f"{leak:.2f}" if isinstance(leak, (int, float)) else "?"
        adj = r.get("adjudication") or {}
        if not leak_ok(r, item):
            out.append(f"- {sid}-{n}: LEAK (score {leak_s} without the description, or a main key point was hit). "
                       f"The answer it gave without the description began: \"{r['leak']['answer'][:500]}\"")
        if adj.get("verdict") == "exclude":
            out.append(f"- {sid}-{n}: EXCLUDED by the adjudicator: {adj.get('reason', '')}")
    return out


def rewrite(workers=8, only=None):
    """One rewrite session for every checked system with a failed item that has not been rewritten yet; its old
    check records move to checks_void_rewrite/ so `check` checks all three items again."""
    done = json.loads(REWRITES.read_text(encoding="utf-8")) if REWRITES.exists() else {}
    todo = []
    for a in P.plan():
        sid = a["system_id"]
        if a["pilot"] or sid in done or (only and sid not in only):
            continue
        if not all((CHECKS / f"{sid}-{n}.json").exists() for n in (1, 2, 3)):
            continue
        recs = [json.loads((CHECKS / f"{sid}-{n}.json").read_text(encoding="utf-8")) for n in (1, 2, 3)]
        if any("adjudication" not in r or not _leak_done(r) for r in recs):
            continue
        fails = _failures(sid)
        if fails:
            todo.append((a, fails))
    claude_guard()
    void = OUT / "checks_void_rewrite"
    void.mkdir(parents=True, exist_ok=True)

    def one(job):
        a, fails = job
        sid = a["system_id"]
        work, backup = WORK / f"{sid}.json", OUT / "work_backup" / f"{sid}.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        doc = json.loads(work.read_text(encoding="utf-8"))
        backup.write_text(json.dumps(doc), encoding="utf-8")
        items = json.loads(doc["files"]["items.json"])          # code review: keep the adjudicator's key fixes
        for it in items:
            f = CHECKS / f"{it['id']}.json"
            if f.exists():
                adj = json.loads(f.read_text(encoding="utf-8")).get("adjudication") or {}
                k = usable_fixed_key(adj.get("answer_key"), it) if adj.get("verdict") == "fix_key" else None
                if k:
                    it["answer_key"] = k
        doc["files"]["items.json"] = json.dumps(items, indent=1)
        doc["accepted"] = False                           # only a submitted rewrite counts as accepted (code review)
        work.write_text(json.dumps(doc), encoding="utf-8")
        res = _write_one(a, feedback=REWRITE_INTRO.format(failures="\n".join(fails)))
        if res != "accepted":                                  # a failed rewrite never costs the passing items
            work.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        with _lock:
            done[a["system_id"]] = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "failures": fails, "result": res}
            REWRITES.write_text(json.dumps(done, indent=1), encoding="utf-8")
            if res == "accepted":
                for n in (1, 2, 3):
                    f = CHECKS / f"{a['system_id']}-{n}.json"
                    if f.exists():
                        f.replace(void / f.name)
            print(f"rewrite {a['system_id']}: {res}", flush=True)
            n_done[0] += 1
            if n_done[0] % 10 == 0:                        # the Claude guard every 10 rewrites (5-hour window too)
                try:
                    claude_guard()
                except SystemExit as e:
                    print(f"stopping: {e}", flush=True)
                    stop.set()
        return res

    def guarded(job):
        return "not started (Claude guard)" if stop.is_set() else one(job)
    stop, n_done = threading.Event(), [0]
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(guarded, todo))
    print(f"rewrote {len(todo)} systems:", {r: res.count(r) for r in set(res)})
    if stop.is_set():
        raise SystemExit("rewrites stopped by the Claude guard; run again after the window resets")


def releak(workers=8):
    """Code review (2026-10-29): re-run the leak check with Claude Sonnet on every checked item whose leak check was
    done by DeepSeek. The DeepSeek result stays on record as leak_deepseek."""
    jobs = []
    for f in sorted(CHECKS.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if "adjudication" not in rec:
            continue
        sid, n = rec["id"].rsplit("-", 1)
        item = _item(sid, int(n))
        if item is None:
            continue
        need_fixed = (_leak_done(rec) and effective_item(rec, item) is not item
                      and (rec.get("leak_fixed") or {}).get("status") != "ok")
        if not _leak_done(rec) or need_fixed:            # code review: corrected keys checked before V3-046 too
            doc = json.loads((WORK / f"{sid}.json").read_text(encoding="utf-8"))
            jobs.append((f, item, doc["files"]["description.md"]))
    claude_guard()
    done = [0]

    def one(job):
        f, item, desc = job
        rec = json.loads(f.read_text(encoding="utf-8"))
        try:
            if "leak" in rec and "leak_deepseek" not in rec:
                rec["leak_deepseek"] = rec.pop("leak")
            ans = claude_text(LEAKER, item["question"])["final_raw"]
            g = judge(item, desc, ans)
            if g.get("status") != "ok":
                raise RuntimeError("judge failed")         # left for the next releak, never saved as a rejection
            eff = effective_item(rec, item)
            gf = judge(eff, desc, ans) if eff is not item else None
            if gf is not None and gf.get("status") != "ok":
                raise RuntimeError("judge failed on the corrected key")
            rec["leak"] = {"answer": ans, "grade": g, "by": MODEL}
            rec.pop("leak_fixed", None)
            if gf is not None:
                rec["leak_fixed"] = gf
            rec["accepted"] = is_accepted(rec, item)
            f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            res = "accepted" if rec["accepted"] else "rejected"
        except Exception as e:
            res = f"error {type(e).__name__}"
        with _lock:
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"[{done[0]}/{len(jobs)}] releak", flush=True)
        return res
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(one, jobs))
    print({r: res.count(r) for r in set(res)})


# ---------------------------------------------------------------- selecting the final sets
def final_item(doc, item, desc, rec):
    it = dict(item)
    k = usable_fixed_key(rec["adjudication"].get("answer_key"), item) \
        if rec["adjudication"].get("verdict") == "fix_key" else None
    if k:
        it["answer_key"] = k
        it["key_fixed_by_adjudicator"] = True
    sysj = json.loads(doc["files"]["system.json"])
    # kind, domain and messiness come from the system record, which the writer check holds to the plan; the
    # writer sometimes put the task type into an item's own "kind" label (9 of the first 500)
    it.update({"bank": "hand5", "grading": "ai", "context": desc, "system": item["system_id"],
               "system_title": sysj.get("title", ""),
               **{f: sysj[f] for f in ("kind", "domain", "messiness") if sysj.get(f)}})
    return unmangle(it)


def unmangle(x):
    """Undo cp1252 mojibake in every string of x. The writer's tool server read the Claude CLI's UTF-8 through a
    cp1252 stdin (surrogateescape), so "–" arrived as "â€“"; that decoding is lossless, so encoding back and decoding
    as UTF-8 restores the text exactly. A string that does not round-trip (clean text) is left as it is."""
    if isinstance(x, dict):
        return {k: unmangle(v) for k, v in x.items()}
    if isinstance(x, list):
        return [unmangle(v) for v in x]
    if isinstance(x, str) and any(ord(c) > 127 for c in x):
        try:
            return x.encode("cp1252", "surrogateescape").decode("utf-8")
        except UnicodeError:
            return x
    return x


QUOTA_ORDER = ("diagnosis", "prediction", "completeness", "intervention", "assumptions", "consistency")


def quotas(n=P.N_ITEMS):
    q = {t: n // len(QUOTA_ORDER) for t in QUOTA_ORDER}
    for t in QUOTA_ORDER[: n % len(QUOTA_ORDER)]:
        q[t] += 1
    return q


def select():
    main, pilot = [], []
    for a in P.plan():
        if not (WORK / f"{a['system_id']}.json").exists():
            continue
        doc, items, desc = system_items(a["system_id"])
        if not doc:
            continue
        for it in items:
            f = CHECKS / f"{it['id']}.json"
            if not f.exists():
                continue
            rec = json.loads(f.read_text(encoding="utf-8"))
            if "adjudication" in rec and is_accepted(rec, it):
                (pilot if a["pilot"] else main).append(final_item(doc, it, desc, rec))
    q, chosen, rest = quotas(), [], []
    for it in main:                                  # plan order within each question type, up to its quota
        if q.get(it["task_type"], 0) > 0:
            q[it["task_type"]] -= 1
            chosen.append(it)
        else:
            rest.append(it)
    short = {t: n for t, n in q.items() if n > 0}
    if short:
        print(f"quota shortfall {short}: filled from other types in plan order")
        chosen += rest[: sum(short.values())]
    if len(chosen) < P.N_ITEMS:
        print(f"only {len(chosen)} accepted items; {P.N_ITEMS} needed: build more systems")
    order = {it["id"]: n for n, it in enumerate(main)}
    main = sorted(chosen, key=lambda it: order[it["id"]])[:P.N_ITEMS]
    FINAL.write_text(json.dumps(main, indent=1), encoding="utf-8")
    PILOT_FILE.write_text(json.dumps(pilot, indent=1), encoding="utf-8")
    print(f"final set: {len(main)} items; pilot: {len(pilot)} items")
    print(balance(main))


def balance(items):
    out = {}
    for k in ("task_type", "kind", "messiness", "domain"):
        c = {}
        for it in items:
            c[it.get(k)] = c.get(it.get(k), 0) + 1
        out[k] = c
    return json.dumps(out, indent=1)


def status():
    plan = P.plan()
    w = [json.loads((WORK / f"{a['system_id']}.json").read_text(encoding="utf-8"))
         for a in plan if (WORK / f"{a['system_id']}.json").exists()]
    acc = sum(1 for d in w if d.get("accepted"))
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in CHECKS.glob("*.json")] if CHECKS.exists() else []
    print(f"systems: {len(plan)} planned, {len(w)} started, {acc} accepted by the writer check")
    acc = 0
    for r in recs:
        sid, n = r["id"].rsplit("-", 1)
        it = _item(sid, int(n))
        acc += bool(it and "adjudication" in r and is_accepted(r, it))
    print(f"items checked: {len(recs)}, accepted under the current rule {acc}")
    verdicts = {}
    for r in recs:
        v = (r.get("adjudication") or {}).get("verdict")
        verdicts[v] = verdicts.get(v, 0) + 1
    print("adjudicator verdicts:", verdicts)
    leaks = [r["leak"]["grade"].get("score") for r in recs if r.get("leak") and r["leak"].get("grade")]
    print(f"leak scores > {LEAK_MAX}:", sum(1 for s in leaks if s is not None and s > LEAK_MAX), "of", len(leaks))
    vg = [r["validator_grade"].get("score") for r in recs if r.get("validator_grade")]
    vg = [s for s in vg if s is not None]
    if vg:
        print(f"validator mean grade {sum(vg) / len(vg):.2f} over {len(vg)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("write", "check", "releak", "rewrite", "select", "status"))
    ap.add_argument("--only", help="comma-separated system ids")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    only = set(a.only.split(",")) if a.only else None
    if a.cmd == "write":
        write(only, a.workers)
    elif a.cmd == "check":
        check(a.workers, only)
    elif a.cmd == "releak":
        releak(a.workers)
    elif a.cmd == "rewrite":
        rewrite(a.workers, only)
    elif a.cmd == "select":
        select()
    else:
        status()


if __name__ == "__main__":
    main()
