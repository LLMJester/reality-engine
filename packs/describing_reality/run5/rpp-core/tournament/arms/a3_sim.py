"""A3: deterministic simulation (see tournament/DESIGN.md).

The model writes Python implementing a fixed API:

    class Model:
        def init_state(self, scenario: dict) -> dict: ...
        def step(self, state: dict, t: int) -> dict: ...   # advances from time t to t+1
        def observe(self, state: dict) -> dict: ...

The code runs only inside the existing sandbox (harness/sandbox_runner.py: audit hook blocking network,
processes and writes outside a temp dir; memory watchdog), in a fresh process, with a 60 s timeout.
Graceful degradation: if `step` raises at step k the trajectory up to k is returned with the error
location (line, function, source line) and a hint. Missing `init_state` / `observe` fall back to
defaults (state = scenario copy; observation = state) and are reported as holes.
"""
import ast
import difflib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tournament.arms.base import Arm, ArmState

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = ROOT / "harness" / "sandbox_runner.py"
# Runtime dependencies for the final lock (harness.orchestrator.lock_files; recheck item 1): files this module
# executes in a subprocess or reads at run time, as repo-relative globs. Kept in sync by
# tests/test_recheck_lock.py (every subprocess launch target / cwd must be covered).
RUNTIME_DEPS = ["harness/sandbox_runner.py"]
MAX_CODE = 60_000
MAX_OUT_CHARS = 8000

# ---------------------------------------------------------------------------------------------------
# Driver executed inside the sandbox as main.py. It reads model.py, scenario.json and params.json from
# the working directory and writes tagged JSON lines to the real stdout as it goes, so a timeout still
# leaves the partial trajectory.
DRIVER = r'''
import copy, io, json, math, random, sys, traceback, inspect
P = json.load(open("params.json", encoding="utf-8"))
SCEN = json.load(open("scenario.json", encoding="utf-8"))
SRC = open("model.py", encoding="utf-8").read()
LINES = SRC.splitlines()
OUT = sys.stdout
CAP = io.StringIO()
sys.stdout = CAP

def _j(x, d=0):
    if d > 12:
        return repr(x)[:80]
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return x if abs(x) < 2**53 else float(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, dict):
        return {str(k): _j(v, d + 1) for k, v in list(x.items())[:500]}
    if isinstance(x, (list, tuple, set, frozenset)):
        xs = list(x) if not isinstance(x, (set, frozenset)) else sorted(x, key=repr)
        return [_j(v, d + 1) for v in xs[:500]]
    try:
        f = float(x)
        return f if math.isfinite(f) else str(f)
    except Exception:
        return repr(x)[:120]

def emit(tag, obj):
    OUT.write("@@" + tag + " " + json.dumps(obj) + "\n")
    OUT.flush()

def where(e, phase, t=None):
    d = {"phase": phase, "type": type(e).__name__, "msg": str(e)[:300], "t": t}
    if isinstance(e, SyntaxError) and (e.filename in ("model.py", "<unknown>", None)):
        d["line"] = e.lineno
        d["col"] = e.offset
        d["code"] = (e.text or "").strip()[:200]
        return d
    fr = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "model.py"]
    if fr:
        f = fr[-1]
        d["line"] = f.lineno
        d["func"] = f.name
        d["code"] = LINES[f.lineno - 1].strip()[:200] if 0 < f.lineno <= len(LINES) else ""
    return d

def nargs(fn):
    try:
        return len([p for p in inspect.signature(fn).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except Exception:
        return -1

def flat(x, pre="", out=None):
    out = {} if out is None else out
    if len(out) > 300:
        return out
    if isinstance(x, dict):
        for k, v in x.items():
            flat(v, f"{pre}.{k}" if pre else str(k), out)
    elif isinstance(x, list):
        if len(x) <= 20:
            for i, v in enumerate(x):
                flat(v, f"{pre}.{i}" if pre else str(i), out)
        else:
            out[(pre + ".len") if pre else "len"] = len(x)
    elif isinstance(x, (bool, int, float, str)) or x is None:
        out[pre or "value"] = x
    return out

def load():
    ns = {"__name__": "model"}
    try:
        exec(compile(SRC, "model.py", "exec"), ns)
    except BaseException as e:
        emit("ERR", where(e, "load"))
        return None
    M = ns.get("Model")
    if M is None:
        emit("ERR", {"phase": "load", "type": "MissingModel", "msg": "no class named Model"})
        return None
    try:
        return M()
    except BaseException as e:
        emit("ERR", where(e, "Model()"))
        return None

def trajectory(m, rng, mode):
    """Returns (observations, error_or_None)."""
    obs = []
    obs_fn = getattr(m, "observe", None)
    def observe(s, t):
        if obs_fn is None:
            o = copy.deepcopy(s)
        else:
            o = obs_fn(s)
        if not isinstance(o, dict):
            raise TypeError(f"observe must return a dict, got {type(o).__name__}")
        return _j(o) if mode == "det" else flat(_j(o))
    try:
        init = getattr(m, "init_state", None)
        if init is None:
            state = copy.deepcopy(SCEN)
        elif mode == "prob" and nargs(init) >= 2:
            state = init(copy.deepcopy(SCEN), rng)
        else:
            state = init(copy.deepcopy(SCEN))
        if not isinstance(state, dict):
            raise TypeError(f"init_state must return a dict, got {type(state).__name__}")
    except BaseException as e:
        return obs, where(e, "init_state", 0)
    try:
        obs.append(observe(state, 0))
    except BaseException as e:
        return obs, where(e, "observe", 0)
    step = getattr(m, "step", None)
    if step is None:
        return obs, {"phase": "step", "type": "MissingStep", "msg": "Model has no step method", "t": 0}
    for t in range(P["steps"]):
        try:
            new = step(state, t, rng) if mode == "prob" else step(state, t)
            if new is None:
                new = state
            if not isinstance(new, dict):
                raise TypeError(f"step must return a dict (or None after mutating state), got {type(new).__name__}")
            state = new
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            return obs, where(e, "step", t)
        try:
            obs.append(observe(state, t + 1))
        except BaseException as e:
            return obs, where(e, "observe", t + 1)
    return obs, None

m = load()
if m is not None:
    if P["mode"] == "det":
        random.seed(P.get("seed", 0))
        obs, err = trajectory(m, None, "det")
        for i, o in enumerate(obs):
            emit("OBS", {"t": i, "obs": o})
        if err:
            emit("ERR", err)
    else:
        for i in range(P["n_seeds"]):
            s = P.get("seed", 0) * 1000003 + i
            random.seed(s)
            obs, err = trajectory(m, random.Random(s), "prob")
            emit("RUN", {"i": i, "obs": obs, "err": err})
txt = CAP.getvalue()
if txt:
    emit("PRINT", txt[:600])
emit("DONE", {})
'''

HINTS = {
    "KeyError": "a key is missing from the dict; set it in init_state or use .get(key, default)",
    "ZeroDivisionError": "a divisor became zero; guard it",
    "TypeError": "check argument counts and value types (step(self, state, t) must return a dict)",
    "NameError": "a name is undefined: typo or missing import",
    "AttributeError": "an attribute or method does not exist on that object",
    "IndexError": "a list index is out of range",
    "ValueError": "a value has the wrong form for the operation",
    "RecursionError": "unbounded recursion",
    "MemoryError": "the model uses too much memory",
    "SyntaxError": "fix the syntax at the indicated line",
    "IndentationError": "fix the indentation at the indicated line",
    "MissingModel": "define `class Model:` with init_state, step, observe",
    "MissingStep": "add `def step(self, state, t):` returning the next state",
    "PermissionError": "the sandbox blocks files, network and processes; keep the model self-contained",
    "ImportError": "only the standard library (and numpy) is available",
    "ModuleNotFoundError": "only the standard library (and numpy) is available",
}


def run_in_sandbox(files, timeout):
    """Run DRIVER with the given files in a fresh sandboxed process. Returns (stdout, stderr, timed_out, rc)."""
    with tempfile.TemporaryDirectory(prefix="rpp_tour_") as d:
        for name, content in files.items():
            Path(d, name).write_text(content, encoding="utf-8")
        main = Path(d, "main.py")
        main.write_text(DRIVER, encoding="utf-8")
        env = {"PYTHONIOENCODING": "utf-8", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        p = subprocess.Popen([sys.executable, "-I", str(SANDBOX), d, str(main)], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=d, env=env, text=True, encoding="utf-8",
                             errors="replace")
        try:
            out, err = p.communicate(timeout=timeout)
            return out or "", err or "", False, p.returncode
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            return out or "", err or "", True, p.returncode


def parse_lines(out):
    recs = []
    for line in out.splitlines():
        if not line.startswith("@@"):
            continue
        tag, _, body = line[2:].partition(" ")
        try:
            recs.append((tag, json.loads(body)))
        except Exception:
            continue
    return recs


def fmt_err(e):
    if not e:
        return ""
    loc = []
    if e.get("line"):
        loc.append(f"model.py line {e['line']}" + (f", col {e['col']}" if e.get("col") else ""))
    if e.get("func"):
        loc.append(f"in {e['func']}()")
    s = f"{e.get('type', 'Error')} during {e.get('phase', '?')}"
    if e.get("t") is not None and e.get("phase") in ("step", "observe"):
        s += f" at step t={e['t']}"
    s += f": {e.get('msg', '')}"
    if loc:
        s += " [" + " ".join(loc) + "]"
    if e.get("code"):
        s += f"\n    > {e['code']}"
    h = HINTS.get(e.get("type", ""))
    if h:
        s += f"\n    hint: {h}"
    return s


def static_check(code):
    """Parse (never execute) the code; return (holes, errors)."""
    holes, errors = [], []
    if not isinstance(code, str) or not code.strip():
        return ["no model code"], []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"SyntaxError at model.py line {e.lineno}, col {e.offset}: {e.msg}")
        return ["model code does not parse"], errors
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model"]
    if not cls:
        return ["no class Model"], errors
    meths = {n.name: n for n in cls[0].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "step" not in meths:
        holes.append("Model.step missing (only the initial state can be observed)")
    if "init_state" not in meths:
        holes.append("Model.init_state missing (state defaults to a copy of the scenario)")
    if "observe" not in meths:
        holes.append("Model.observe missing (observation defaults to the full state)")
    return holes, errors


def _tool(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


def _maybe_json(x):
    if isinstance(x, str) and x.strip()[:1] in "{[":
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _short(o, limit=400):
    s = json.dumps(o, separators=(",", ":"))
    return s if len(s) <= limit else s[:limit] + "...}"


class SimArm(Arm):
    name = "A3"
    buildable = True
    executable = True
    mode = "det"
    guide_file = "a3_guide.md"
    mutating_tools = ("write_model", "patch_model")
    TIMEOUT = 60
    MAX_STEPS = 10000

    def guide(self):
        p = ROOT / "tournament" / "prompts" / self.guide_file
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def _run_props(self):
        return {"scenario": {"type": "object", "description": "passed to init_state(scenario)"},
                "steps": {"type": "integer", "description": f"number of steps (max {self.MAX_STEPS})"},
                "keys": {"type": "array", "items": {"type": "string"},
                         "description": "optional: only show these observation keys"}}

    def tools(self):
        return [
            _tool("write_model", "Replace the model code (Python defining class Model).",
                  {"code": {"type": "string"}}, ["code"]),
            _tool("patch_model", "Replace one exact occurrence of `old` with `new` in the model code.",
                  {"old": {"type": "string"}, "new": {"type": "string"}}, ["old", "new"]),
            _tool("run_scenario", "Run the model in the sandbox (60 s limit) and return observations per step.",
                  self._run_props(), ["scenario", "steps"]),
            _tool("show_model", "Show the current model code with line numbers.", {}),
        ]

    def probe_tools(self):
        return [t for t in self.tools() if t["function"]["name"] in ("run_scenario", "show_model")]

    def new_state(self):
        return ArmState(rep={"code": ""})

    def load(self, rep, given_gold=False):
        st = self.new_state()
        if isinstance(rep, dict):
            st.rep = dict(rep)
            if not isinstance(st.rep.get("code"), str):
                st.rep["code"] = ""
        elif isinstance(rep, str):
            st.rep = {"code": rep}
        st.given_gold = given_gold
        return st

    # ------------------------------------------------------------------ dispatch
    def handle(self, state, name, args):
        try:
            args = _maybe_json(args) if args is not None else {}
            if not isinstance(args, dict):
                return f"ERROR: arguments must be an object (hint: see the {name} schema)"
            fn = getattr(self, "_t_" + str(name), None)
            if fn is None:
                return (f"ERROR: unknown tool '{name}' (hint: available: "
                        + ", ".join(t["function"]["name"] for t in self.tools()) + ")")
            if state.given_gold and name in ("write_model", "patch_model"):
                return "ERROR: the model is read-only in this mode (hint: use run_scenario, show_model)"
            out = fn(state, args)
        except Exception as e:
            out = f"ERROR: internal {type(e).__name__}: {e} (hint: check the arguments)"
        state.log.append((name, {k: (v if k != "code" else f"<{len(str(v))} chars>") for k, v in args.items()}
                          if isinstance(args, dict) else args, out[:200]))
        return out

    def _code(self, state):
        c = state.rep.get("code")
        return c if isinstance(c, str) else ""

    def _after_edit(self, state, verb):
        holes, errors = static_check(self._code(state))
        n = len(self._code(state).splitlines())
        msg = f"model {verb} ({n} lines)"
        for e in errors:
            msg += f"\n  error: {e}\n    hint: fix it with patch_model or write_model"
        for h in holes:
            msg += f"\n  hole: {h}"
        return msg

    def _t_write_model(self, state, args):
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return "ERROR: code must be a non-empty string (hint: define class Model with init_state, step, observe)"
        if len(code) > MAX_CODE:
            return f"ERROR: code too long ({len(code)} chars, max {MAX_CODE})"
        state.rep["code"] = code
        return self._after_edit(state, "written")

    def _t_patch_model(self, state, args):
        old, new = args.get("old"), args.get("new")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            return "ERROR: patch_model needs non-empty string 'old' and string 'new'"
        code = self._code(state)
        n = code.count(old)
        if n == 0:
            first = old.strip().splitlines()[0] if old.strip() else old
            close = difflib.get_close_matches(first.strip(), [l.strip() for l in code.splitlines()], n=3, cutoff=0.5)
            hint = ("closest lines: " + " | ".join(close)) if close else "use show_model to copy the exact text"
            return f"ERROR: `old` text not found in the model (hint: {hint})"
        if n > 1:
            return f"ERROR: `old` text occurs {n} times (hint: include more surrounding lines so it is unique)"
        state.rep["code"] = code.replace(old, new, 1)
        if len(state.rep["code"]) > MAX_CODE:
            state.rep["code"] = code
            return "ERROR: patched code too long; patch not applied"
        return self._after_edit(state, "patched")

    def _t_show_model(self, state, args):
        return self.render(state)

    def render(self, state):
        code = self._code(state)
        if not code:
            return "(no model code yet)"
        lines = code.splitlines()
        return "\n".join(f"{i + 1:4d}| {l}" for i, l in enumerate(lines))[:60000]

    # ------------------------------------------------------------------ running
    def _check_run_args(self, args):
        scen = _maybe_json(args.get("scenario", {}))
        if scen is None:
            scen = {}
        if not isinstance(scen, dict):
            return None, None, "ERROR: scenario must be an object (hint: scenario={\"param\": 3})"
        steps = args.get("steps", 10)
        try:
            steps = int(steps)
        except (TypeError, ValueError):
            return None, None, "ERROR: steps must be an integer"
        if steps < 0:
            return None, None, "ERROR: steps must be >= 0"
        note = ""
        if steps > self.MAX_STEPS:
            note = f"note: steps capped at {self.MAX_STEPS}\n"
            steps = self.MAX_STEPS
        try:
            json.dumps(scen)
        except (TypeError, ValueError) as e:
            return None, None, f"ERROR: scenario is not JSON-serialisable: {e}"
        return scen, steps, note

    def _execute(self, state, scen, params):
        files = {"model.py": self._code(state), "scenario.json": json.dumps(scen), "params.json": json.dumps(params)}
        out, err, timed_out, rc = run_in_sandbox(files, self.TIMEOUT)
        recs = parse_lines(out)
        done = any(t == "DONE" for t, _ in recs)
        notes = []
        if timed_out:
            notes.append(f"[timed out after {self.TIMEOUT}s; partial results shown]"
                         "\n    hint: reduce steps or remove unbounded loops")
        elif not done:
            tail = err.strip().splitlines()[-3:] if err.strip() else []
            msg = "memory limit exceeded" if "memory limit" in err else f"process ended unexpectedly (exit code {rc})"
            notes.append(f"[{msg}; partial results shown]" + ("\n    " + "\n    ".join(tail) if tail else ""))
        prints = [b for t, b in recs if t == "PRINT"]
        if prints:
            notes.append("model printed: " + prints[0][:300].replace("\n", " | "))
        return recs, notes

    def _t_run_scenario(self, state, args):
        if not self._code(state).strip():
            return "ERROR: no model code yet (hint: call write_model first)"
        scen, steps, note = self._check_run_args(args)
        if scen is None:
            return note
        keys = args.get("keys")
        recs, notes = self._execute(state, scen, {"mode": "det", "steps": steps, "seed": 0})
        obs = [(b["t"], b["obs"]) for t, b in recs if t == "OBS"]
        errs = [b for t, b in recs if t == "ERR"]
        lines = [note.rstrip()] if note else []
        if isinstance(keys, list) and keys:
            obs = [(t, {k: o.get(k) for k in keys} if isinstance(o, dict) else o) for t, o in obs]
        lines.append(f"scenario {_short(scen, 200)}, steps requested {steps}, observations {len(obs)}")
        lines += self._fmt_traj(obs)
        for e in errs:
            lines.append("ERROR: " + fmt_err(e))
            if e.get("phase") in ("step", "observe"):
                lines.append(f"    (trajectory shown up to t={len(obs) - 1})")
        lines += notes
        self._record_run(state, ok=not errs and len(obs) == steps + 1, scen=scen)
        if not obs:
            lines.insert(0, "ERROR: the run produced no observations")
        out = "\n".join(lines)
        return out if len(out) <= MAX_OUT_CHARS else out[:MAX_OUT_CHARS - 200] + "\n...[truncated]...\n" + out[-180:]

    def _record_run(self, state, ok, scen):
        h = hashlib.sha1(self._code(state).encode("utf-8")).hexdigest()
        state.__dict__.setdefault("_runs", {})[h] = {"ok": ok, "scenario": scen}
        state.__dict__["_last_scenario"] = scen

    def _fmt_traj(self, obs):
        if not obs:
            return ["(no observations)"]
        n = len(obs)
        if n <= 25:
            idx = list(range(n))
        else:
            mid = list(range(5, n - 5, max(1, (n - 10) // 12)))
            idx = sorted(set(list(range(5)) + mid + list(range(n - 5, n))))
        per = max(120, min(600, (MAX_OUT_CHARS - 1500) // max(1, len(idx))))
        out, last = [], -1
        for i in idx:
            if last >= 0 and i != last + 1:
                out.append("  ...")
            t, o = obs[i]
            out.append(f"  t={t}: {_short(o, per if i != n - 1 else 1500)}")
            last = i
        return out

    # ------------------------------------------------------------------ validity
    def validity(self, state):
        code = self._code(state)
        holes, errors = static_check(code)
        usable = bool(code.strip()) and not errors and "no class Model" not in holes
        executable = False
        if usable:
            h = hashlib.sha1(code.encode("utf-8")).hexdigest()
            runs = state.__dict__.get("_runs", {})
            if h in runs and runs[h]["ok"]:
                executable = True
            else:
                scen = runs.get(h, {}).get("scenario") or state.__dict__.get("_last_scenario") or {}
                executable, err = self._smoke(state, scen)
                if err:
                    errors.append(err)
        return {"usable": usable, "executable": executable, "holes": holes, "errors": errors}

    def _smoke(self, state, scen):
        params = {"mode": self.mode, "steps": 2, "seed": 0}
        if self.mode == "prob":
            params["n_seeds"] = 2
        recs, notes = self._execute(state, scen, params)
        errs = [b for t, b in recs if t == "ERR"] + [b["err"] for t, b in recs if t == "RUN" and b.get("err")]
        if errs:
            return False, fmt_err(errs[0])
        if notes and any(n.startswith("[") for n in notes):
            return False, notes[0]
        return any(t in ("OBS", "RUN") for t, _ in recs), None


A3 = SimArm   # registry name (tournament/arms/__init__.py: ARMS["A3"] = "a3_sim:A3")
