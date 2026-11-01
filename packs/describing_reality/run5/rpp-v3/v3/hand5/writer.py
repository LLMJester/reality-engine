"""The item writer's tool server and brief (run 5 hand-written-style items).

A writer is a headless Claude Sonnet session with no file, shell or web access; these tools are its only interface:
  write_file(name, text)  save one of the system's files in its workspace (held by this server)
  show_file(name)         show a saved file with line numbers
  run_sim()               run sim.py in the Docker sandbox (v3.sandbox; no network, no host files); its stdout is
                          kept as the computed keys
  submit()                check everything; on success the system is accepted and the session should end
The workspace is written to --out after every call, so nothing is lost if the session is cut off.

Run: python -m v3.hand5.writer --out OUT.json --assignment ASSIGNMENT.json
"""
import argparse
import json
import os
import re
import time

from v3.builders.mcp_server import serve

FILES = ("sim.py", "system.json", "description.md", "items.json")
TASK_TYPES = ("prediction", "diagnosis", "intervention", "consistency", "assumptions", "completeness")
SIM_TIMEOUT = 120
MAX_FILE = 60_000
DRIVER = "import runpy\nrunpy.run_path('sim.py', run_name='__main__')\n"

BRIEF = """You build evaluation items for a study of how well AI models reason about systems and situations: their
state, rules, dynamics, actors and uncertainties. Each item is built from a hidden ground-truth simulator that you
write in ordinary Python. The simulator is the answer key: the models under test never see it; they see only a
natural-language description of the system and one question.

You work only through your tools: write_file, show_file, run_sim, submit. You have no other access.

## Your assignment
{assignment}

## The four files

1. sim.py - the ground-truth simulator.
   - Stateful Python: classes or objects, loops over time steps, numeric quantities, queues, scheduled events, simple
     agent policies. Not a table of if-then rules. It should look like code an engineer would write to simulate the
     thing.
   - Deterministic given a seed (random.Random(seed)); standard library only (plus math).
   - Defines simulate(scenario: dict, seed: int = 0) -> dict returning the trajectory or final values of named
     observables.
   - When run as a script it runs the scenarios behind your three questions and prints ONE JSON object to stdout:
     {{"<item id>": {{...the computed values the key needs...}}, ...}}. For a stochastic system run 200 seeds and
     report the outcome distribution and the modal outcome.
   - Non-trivial: at least about 6 interacting state variables or entities, and at least one feedback, delay,
     capacity limit, threshold, priority rule or interaction that makes naive reasoning go wrong.
   - run_sim runs it in a sealed sandbox (no network, no files outside its folder, 120 s). Call run_sim after every
     change to sim.py; the keys must come from its latest output.

2. system.json - {{"system_id", "title", "domain", "kind", "messiness", "inexpressible_dynamics": true|false,
   "inexpressible_note", "observables": [{{"name", "meaning", "unit"}}], "omissions": [{{"what", "matters_for"}}],
   "injected_conflicts": [{{"what", "where"}}]}}. inexpressible_dynamics is true if the true dynamics involve
   continuous feedback, probabilistic transitions or emergent multi-agent behaviour that a simple discrete rule
   language could not express exactly.

3. description.md - what the model under test sees: 500 to 2500 words. It describes the system as a person who knows
   it would: its parts, how things flow, rules, actors, current state, recent events. It must contain enough to answer
   the questions by building a mental model and reasoning through the dynamics, not by finding one sentence. Written
   at the assigned messiness:
   - realistic: partial, out of order, some irrelevant detail, some terms used inconsistently (a component called two
     names), information spread over sections (for example an operations wiki page, a chat excerpt and a status
     table). Everything needed is present or inferable.
   - adversarial: as realistic, plus at least one thing that matters for at least one question is deliberately
     missing. Record every omission in system.json and in the item's construction_notes.
   Never mention a simulator, a key or a test.

4. items.json - a JSON list of exactly three items, one per assigned task type, in the assigned order:
   {{"id": "<system_id>-<n>", "system_id", "bank": "synthetic", "domain", "kind", "messiness", "task_type",
     "question": "...",
     "answer_key": {{"summary": "one-paragraph correct answer",
                    "key_points": [{{"id": "k1", "text": "...", "weight": 1.0}}],
                    "wrong_answers": ["tempting but wrong answers a shallow reasoner would give, and why"]}},
     "construction_notes": "which scenario and seeds gave the key, which conflicts or omissions exist, why the item
     needs modelling rather than lookup"}}

## Task types: how to write the question and the key
- prediction: "Starting from the state described, if <events> happen / after <time>, what is the state of <named
  observables>?" Key points: each observable's true value from the simulator. Pick events whose consequences pass
  through at least two interactions.
- diagnosis: present observed symptoms from a simulator run with an injected fault the description does not name.
  "What is the most likely cause?" Key points: the injected cause (heavy weight) and the chain linking it to each
  symptom. Include at least one plausible decoy cause that the symptoms rule out.
- intervention: "If <change to rules, parameters or structure> is made, what follows and what breaks?" Key points,
  from re-running the simulator with the change: the direct effect, the knock-on effects, and at least one
  non-obvious thing that breaks or gets worse.
- consistency: the description contains 1 to 3 injected conflicts (statements that cannot all be true given the rest
  of the system). "Is this description of the system coherent? If not, identify exactly where and why." Key points:
  each injected conflict. The description is shared by the system's three questions: keep conflicts away from facts
  the other two depend on.
- assumptions: state a conclusion about the system's behaviour (for example "the team concludes the backlog will
  clear by Friday") that holds in the simulator only if some unstated condition holds. "What must be true for this
  conclusion to hold?" Key points: the necessary unstated conditions (check by running the simulator with each
  condition violated).
- completeness: "What does this description leave out that matters for answering <a concrete question>?" Key points:
  the deliberate omissions that change the answer (show with the simulator that two values of each missing parameter
  give different answers).

## Quality bar
- Keys come only from running sim.py. Never hand-compute a key you could compute by simulation.
- An item a careful reader could answer by keyword matching or by finding one sentence is a failed item.
- A capable reasoner who models the system carefully from the description should be able to get it right (except the
  recorded adversarial omissions, which are the point of completeness items).
- Do not make the description read like a rule list: prose, tables, logs, messages, the way real systems are
  documented.
- Answers must be gradable without excessive precision: state tolerances in key points (for example "about 340, +-15
  units") or ask for qualitative outcomes plus magnitudes.

When all four files are written and run_sim has produced the keys, call submit. Fix every problem it reports and
submit again. End the session once submit says ACCEPTED."""


def assignment_text(a):
    types = "\n".join(f"  - {a['system_id']}-{n}: {t}" for n, t in enumerate(a["task_types"], 1))
    return (f"- system_id: {a['system_id']}\n- domain: {a['domain']} (invent a specific, plausible system in this "
            f"domain; not a famous real incident)\n- kind: {a['kind']}\n- messiness: {a['messiness']}\n"
            f"- items, in this order:\n{types}")


def brief(a):
    return BRIEF.format(assignment=assignment_text(a))


def _tool(name, desc, props=None, required=()):
    return {"name": name, "description": desc,
            "inputSchema": {"type": "object", "properties": props or {}, "required": list(required)}}


def check_submission(a, files, keys):
    """Problems with a writer's files (empty list = accepted). Code only."""
    probs = []
    for f in FILES:
        if not files.get(f, "").strip():
            probs.append(f"{f} is missing or empty")
    if probs:
        return probs
    if keys is None:
        probs.append("run_sim has not produced output for the current sim.py: call run_sim")
    elif keys.get("error"):
        probs.append("the last run_sim failed: fix sim.py and call run_sim again")
    else:
        out = keys.get("output")
        need = [f"{a['system_id']}-{n}" for n in (1, 2, 3)]
        if not isinstance(out, dict) or any(k not in out for k in need):
            probs.append(f"sim.py must print one JSON object with an entry for each item: {', '.join(need)}")
    if "def simulate" not in files["sim.py"]:
        probs.append("sim.py must define simulate(scenario, seed=0)")
    words = len(re.findall(r"\S+", files["description.md"]))
    if not 500 <= words <= 2500:
        probs.append(f"description.md has {words} words; it must have 500 to 2500")
    if re.search(r"\b(simulator|answer key|sim\.py)\b", files["description.md"], re.I):
        probs.append("description.md must not mention a simulator, a key or sim.py")
    try:
        sysj = json.loads(files["system.json"])
        for k in ("system_id", "title", "domain", "kind", "messiness", "observables", "omissions",
                  "injected_conflicts"):
            if k not in sysj:
                probs.append(f"system.json lacks '{k}'")
        if sysj.get("kind") != a["kind"] or sysj.get("messiness") != a["messiness"]:
            probs.append("system.json kind and messiness must match the assignment")
        if a["messiness"] == "adversarial" and not sysj.get("omissions"):
            probs.append("an adversarial system must record its deliberate omissions in system.json")
        if "consistency" in a["task_types"] and not sysj.get("injected_conflicts"):
            probs.append("a consistency item needs injected conflicts recorded in system.json")
    except ValueError as e:
        probs.append(f"system.json is not valid JSON: {e}")
    try:
        items = json.loads(files["items.json"])
        if not isinstance(items, list) or len(items) != 3:
            probs.append("items.json must be a JSON list of exactly three items")
        else:
            for n, (it, tt) in enumerate(zip(items, a["task_types"]), 1):
                iid = f"{a['system_id']}-{n}"
                if it.get("id") != iid:
                    probs.append(f"item {n} must have id {iid}")
                if it.get("task_type") != tt:
                    probs.append(f"item {iid} must have task_type {tt}")
                if not str(it.get("question", "")).strip():
                    probs.append(f"item {iid} has no question")
                k = it.get("answer_key") or {}
                kps = k.get("key_points") or []
                if not str(k.get("summary", "")).strip() or not kps or not k.get("wrong_answers"):
                    probs.append(f"item {iid} needs answer_key summary, key_points and wrong_answers")
                if any(not isinstance(p, dict) or not p.get("id") or not p.get("text") for p in kps):
                    probs.append(f"item {iid}: every key point needs id and text")
                elif any(not isinstance(p.get("weight", 1.0), (int, float)) or float(p.get("weight", 1.0)) <= 0
                         for p in kps):
                    probs.append(f"item {iid}: every key point weight must be a positive number")
                if not str(it.get("construction_notes", "")).strip():
                    probs.append(f"item {iid} needs construction_notes")
    except ValueError as e:
        probs.append(f"items.json is not valid JSON: {e}")
    return probs


class WriterSession:
    def __init__(self, assignment, out, max_calls=80):
        self.a, self.out, self.max = assignment, out, max_calls
        self.files, self.keys, self.runs, self.accepted, self.calls, self.trace = {}, None, 0, False, 0, []
        self.flush()

    def tools(self):
        return [
            _tool("write_file", "Save one file of the system (replaces it): sim.py, system.json, description.md or "
                                "items.json.", {"name": {"type": "string", "enum": list(FILES)},
                                                 "text": {"type": "string"}}, ["name", "text"]),
            _tool("show_file", "Show a saved file with line numbers.",
                  {"name": {"type": "string", "enum": list(FILES)}}, ["name"]),
            _tool("run_sim", "Run sim.py as a script in the sealed sandbox (120 s). Its printed JSON becomes the "
                             "computed keys."),
            _tool("submit", "Check all four files and the latest run_sim output. ACCEPTED ends the job."),
        ]

    def call(self, name, args):
        self.calls += 1
        args = args or {}
        try:
            if self.calls > self.max:
                res = "ERROR: the tool budget is used up."
            elif name == "write_file":
                fn, text = args.get("name"), args.get("text")
                if fn not in FILES or not isinstance(text, str):
                    res = f"ERROR: name must be one of {FILES} and text a string"
                else:
                    self.files[fn] = text[:MAX_FILE]
                    if fn == "sim.py":
                        self.keys = None
                    res = f"saved {fn} ({len(text)} characters)" + \
                        ("; sim.py changed, so call run_sim again" if fn == "sim.py" else "")
            elif name == "show_file":
                t = self.files.get(args.get("name"), "")
                res = "\n".join(f"{i:4d} | {ln}" for i, ln in enumerate(t.splitlines(), 1)) or "ERROR: not saved"
            elif name == "run_sim":
                res = self.run_sim()
            elif name == "submit":
                probs = check_submission(self.a, self.files, self.keys)
                self.accepted = not probs
                res = "ACCEPTED. The job is done; end the session." if not probs else \
                    "NOT ACCEPTED. Fix these and submit again:\n- " + "\n- ".join(probs)
            else:
                res = f"ERROR: unknown tool {name!r}"
        except Exception as e:   # never crash the session
            res = f"ERROR: internal {type(e).__name__}: {e}"
        self.trace.append({"t": time.time(), "tool": name, "result": res[:2000]})
        self.flush()
        return res

    def run_sim(self):
        from v3 import sandbox
        if not self.files.get("sim.py"):
            return "ERROR: write sim.py first"
        self.runs += 1
        out, err, timed_out, rc = sandbox.run(DRIVER, {"sim.py": self.files["sim.py"]}, SIM_TIMEOUT)
        if timed_out:
            self.keys = {"error": "timeout"}
            return f"ERROR: sim.py ran longer than {SIM_TIMEOUT} s; make it faster.\nstderr tail:\n{err[-1500:]}"
        if rc != 0:
            self.keys = {"error": f"exit {rc}"}
            return f"ERROR: sim.py failed (exit {rc}).\nstderr tail:\n{err[-3000:]}\nstdout head:\n{out[:1000]}"
        from v3.jsonx import last_object
        try:
            parsed = last_object(out)                    # code review: log lines with braces before it are fine
        except ValueError:
            self.keys = {"error": "no JSON"}
            return "ERROR: sim.py must print one JSON object to stdout.\nstdout head:\n" + out[:2000]
        self.keys = {"output": parsed}
        return "run_sim output (kept as the computed keys):\n" + json.dumps(parsed, indent=1)[:12000]

    def flush(self):
        doc = {"assignment": self.a, "files": self.files, "keys": self.keys, "accepted": self.accepted,
               "sim_runs": self.runs, "tool_calls": self.calls, "trace": self.trace}
        tmp = self.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        os.replace(tmp, self.out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--max-calls", type=int, default=80)
    ap.add_argument("--resume", action="store_true", help="start from the files an earlier session saved")
    a = ap.parse_args(argv)
    files = {}
    if a.resume and os.path.exists(a.out):
        files = json.load(open(a.out, encoding="utf-8")).get("files") or {}
    s = WriterSession(json.load(open(a.assignment, encoding="utf-8")), a.out, a.max_calls)
    if files:
        s.files = files          # the keys are not carried over: run_sim must run again
        s.flush()
    serve(s)


if __name__ == "__main__":
    main()
