"""G1C1: parameterised probabilistic program, submit-and-run (A4 modified; runs/discovery/gen1/specs/G1C1.md).

Representation: a Python program with the unchanged A4 API (`Model.init_state(scenario[, rng])`,
`step(state, t, rng)`, `observe(state)`), preceded by two declared literal dictionaries `PARAMS` (numbers) and
`RULES` (discrete switches / policies) and an int `HORIZON`. What-ifs are interventions on those declarations:
a re-run with `overrides`, not a rewrite. Baseline and intervention runs share their seeds (common random
numbers), so effects are reported as paired per-seed differences. A what-if the switches do not anticipate may
use a per-run `what_if_patch` that never changes the stored program.

Construction protocol (document channel, `custom_episode`): the program travels as raw text in a fenced
```model block of the assistant message (or the `source` argument); `submit_model` checks it and immediately
runs the baseline plus up to 4 requested runs. A response with a model block and no tool call is executed as an
implicit `submit_model`. Every executor result ends with a budget status line.

Execution reuses the A3 sandbox (`run_in_sandbox`, audit hook, memory watchdog, 60 s per tool call for all runs
together), A4's `summarize` and the `_expr` events. The driver below is a copy of A3's DRIVER, extended with
override merging / PARAMS-RULES rebinding, per-run source text (ephemeral patches) and several named runs
interleaved seed by seed in one process (same seed list, so a deadline cut keeps the runs paired). To keep the
output small the driver returns per-step means, final observations and only the event keys' trajectories.
"""
import ast
import difflib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from tournament.arms import _expr
from tournament.arms.a3_sim import (MAX_CODE, ROOT, SANDBOX, _maybe_json, _short, _tool, fmt_err, parse_lines,
                                    static_check)
from tournament.arms.a4_prob import _g, _parse_events, _quant, summarize
from tournament.arms.base import ERROR_PREFIX, Arm, ArmState

DEFAULT_SEEDS = 100
MAX_SEEDS = 200
MAX_STEPS = 1000
# Runtime execution dependency (D-060/D-061): models run in the A3 sandbox subprocess; the final lock covers it.
RUNTIME_DEPS = ["harness/sandbox_runner.py"]
MAX_RUNS = 4
NO_HORIZON_STEPS = 50        # steps used when HORIZON is not declared and the run gives none
MAX_OUT = 7800               # executor text; + the budget line stays within 8,000 characters
SERIES_KEYS = 8
SERIES_POINTS = 60
DEADLINE_S = 45              # inside the sandbox: stop starting new seeds after this (TIMEOUT is 60 s)
BUDGET_LINE = "[budget: {k} of {n} responses used; your last response must be the FINAL ANSWER without tool calls]"

# ---------------------------------------------------------------------------------------------------
# Sandbox driver (copied from a3_sim.DRIVER and extended; a3_sim/a4_prob are imported by other arms and stay
# unchanged). Reads params.json = {"runs": [...], "deadline": s}; each run names its own source file.
DRIVER = r'''
import copy, io, json, math, random, sys, time, traceback, inspect
T0 = time.time()
P = json.load(open("params.json", encoding="utf-8"))
OUT = sys.stdout
CAP = io.StringIO()
sys.stdout = CAP
LINES = []

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

def load(R):
    ns = {"__name__": "model"}
    try:
        exec(compile(R["_src"], "model.py", "exec"), ns)
    except BaseException as e:
        return None, where(e, "load")
    # extension 1: rebind the module globals to the overridden copies before Model() is built
    if R.get("params") is not None:
        ns["PARAMS"] = R["params"]
    if R.get("rules") is not None:
        ns["RULES"] = R["rules"]
    M = ns.get("Model")
    if M is None:
        return None, {"phase": "load", "type": "MissingModel", "msg": "no class named Model"}
    try:
        return M(), None
    except BaseException as e:
        return None, where(e, "Model()")

def trajectory(m, scen, steps, rng):
    """Returns (observations, error_or_None)."""
    obs = []
    obs_fn = getattr(m, "observe", None)
    def observe(s, t):
        o = copy.deepcopy(s) if obs_fn is None else obs_fn(s)
        if not isinstance(o, dict):
            raise TypeError(f"observe must return a dict, got {type(o).__name__}")
        return flat(_j(o))
    try:
        init = getattr(m, "init_state", None)
        if init is None:
            state = copy.deepcopy(scen)
        elif nargs(init) >= 2:
            state = init(copy.deepcopy(scen), rng)
        else:
            state = init(copy.deepcopy(scen))
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
    for t in range(steps):
        try:
            new = step(state, t, rng)
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

def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)

RUNS = P["runs"]
models = []
for k, R in enumerate(RUNS):
    R["_src"] = open(R["file"], encoding="utf-8").read()     # extension 2: per-run source (ephemeral patch)
    R["_lines"] = R["_src"].splitlines()
    LINES = R["_lines"]
    m, err = load(R)
    if err:
        emit("ERR", {"r": k, "err": err})
    models.append(m)
acc = [{"keys": [], "sum": {}, "cnt": {}} for _ in RUNS]
maxn = max([R["n"] for R in RUNS] + [0])
cut = None
for i in range(maxn):              # extension 3: runs interleaved seed by seed, same seed list for every run
    if i > 0 and time.time() - T0 > P["deadline"]:
        cut = i
        break
    for k, R in enumerate(RUNS):
        m = models[k]
        if m is None or i >= R["n"]:
            continue
        LINES = R["_lines"]
        s = R["seed"] * 1000003 + i
        random.seed(s)
        obs, err = trajectory(m, R["scen"], R["steps"], random.Random(s))
        A = acc[k]
        for t, o in enumerate(obs):
            for key, v in o.items():
                if key not in A["sum"]:
                    if len(A["keys"]) >= 300:
                        continue
                    A["keys"].append(key)
                    A["sum"][key] = [0.0] * (R["steps"] + 1)
                    A["cnt"][key] = [0] * (R["steps"] + 1)
                if num(v):
                    A["sum"][key][t] += v
                    A["cnt"][key][t] += 1
        full = len(obs) == R["steps"] + 1
        emit("RUN", {"r": k, "i": i, "n": len(obs), "final": obs[-1] if full else None,
                     "last": obs[-1] if (obs and not full) else None,
                     "ev": {key: [o.get(key) for o in obs] for key in R["evkeys"]}, "err": err})
for k, A in enumerate(acc):
    mean = {key: [(A["sum"][key][t] / A["cnt"][key][t]) if A["cnt"][key][t] else None
                  for t in range(len(A["sum"][key]))] for key in A["keys"] if any(A["cnt"][key])}
    emit("AGG", {"r": k, "keys": A["keys"], "mean": mean})
if cut is not None:
    emit("CUT", {"i": cut})
txt = CAP.getvalue()
if txt:
    emit("PRINT", txt[:600])
emit("DONE", {})
'''


# ---------------------------------------------------------------------------------------------------
# Static analysis of the declarations

_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
           ast.Pow: lambda a, b: a ** b}


def _const(node, depth=0):
    """Literal value of a node: ast.literal_eval plus constant arithmetic (e.g. 1/6, 24*7). Raises ValueError."""
    if depth > 40:
        raise ValueError("too deep")
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _const(node.left, depth + 1), _const(node.right, depth + 1)
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b)):
            raise ValueError("non-numeric arithmetic")
        if isinstance(node.op, ast.Pow) and abs(b) > 64:
            raise ValueError("exponent too large")
        return _BINOPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _const(node.operand, depth + 1)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError("non-numeric unary")
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Dict):
        return {_const(k, depth + 1): _const(v, depth + 1) for k, v in zip(node.keys, node.values) if k is not None}
    if isinstance(node, (ast.List, ast.Tuple)):
        xs = [_const(e, depth + 1) for e in node.elts]
        return xs if isinstance(node, ast.List) else tuple(xs)
    raise ValueError(f"not a literal ({type(node).__name__})")


def _jsonable(v):
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def _fragments(node):
    """String pieces a key may be built from at run time: operands of `+`, f-string parts, `%` / `.format`
    templates (e.g. p["mix_" + k], p[f"mix_{k}"])."""
    out = []
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        for x in (node.left, node.right):
            if isinstance(x, ast.Constant) and isinstance(x.value, str):
                out += re.split(r"%[sdr]", x.value)
    elif isinstance(node, ast.JoinedStr):
        out += [x.value for x in node.values if isinstance(x, ast.Constant) and isinstance(x.value, str)]
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format" \
            and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str):
        out += re.split(r"\{[^{}]*\}", node.func.value.value)
    return [f for f in out if len(f) >= 2]


def unread_names(tree, decl_values, names):
    """QA-0182: declared names that never occur in the program outside their declaration, neither as a string
    constant (scenario["x"], p.get("x")), an identifier (def f(x), f(x=...), ns.x) nor as a key built from a
    string fragment (p["mix_" + k]). Overriding such a name cannot change a run (a silent no-op what-if)."""
    skip = {id(n) for v in decl_values for n in ast.walk(v)}
    consts, idents, frags = set(), set(), []
    for n in ast.walk(tree):
        if id(n) in skip:
            continue
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            consts.add(n.value)
        elif isinstance(n, ast.Name):
            idents.add(n.id)
        elif isinstance(n, ast.arg):
            idents.add(n.arg)
        elif isinstance(n, ast.keyword) and n.arg:
            idents.add(n.arg)
        elif isinstance(n, ast.Attribute):
            idents.add(n.attr)
        frags += _fragments(n)
    return [k for k in names if k not in consts and k not in idents
            and not any(k.startswith(f) or k.endswith(f) for f in frags)]


def analyse(src):
    """Static check of a program. Returns a dict: parses, errors, holes, warnings, params, rules, horizon, unread
    (declared PARAMS/RULES names the program never reads)."""
    info = {"parses": False, "errors": [], "holes": [], "warnings": [], "params": None, "rules": None,
            "horizon": None, "has_model": False, "unread": []}
    if not isinstance(src, str) or not src.strip():
        info["holes"] = ["no model code"]
        return info
    holes, errors = static_check(src)
    info["holes"], info["errors"] = holes, errors
    if errors:
        return info
    info["parses"] = True
    info["has_model"] = "no class Model" not in holes
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id in ("PARAMS", "RULES", "HORIZON"):
                found[t.id] = value
    for name, key in (("PARAMS", "params"), ("RULES", "rules")):
        if name not in found:
            info["warnings"].append(f"{name} is not declared: overrides of its entries are impossible "
                                    f"(hint: declare `{name} = {{...}}` at the top)")
            continue
        try:
            v = _const(found[name])
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            info["warnings"].append(f"{name} is not a literal dict (only literal values and constant arithmetic "
                                    f"are allowed): overrides of its entries are impossible")
            continue
        if not isinstance(v, dict) or not all(isinstance(k, str) for k in v):
            info["warnings"].append(f"{name} must be a dict with string keys: overrides are impossible")
            continue
        if not _jsonable(v):
            info["warnings"].append(f"{name} has values that are not plain data (numbers, strings, booleans, "
                                    f"lists, dicts): overrides are impossible")
            continue
        info[key] = v
    if "HORIZON" not in found:
        info["warnings"].append(f"HORIZON is not declared: runs use their `steps` (default {NO_HORIZON_STEPS})")
    else:
        try:
            h = _const(found["HORIZON"])
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            h = None
        if isinstance(h, int) and not isinstance(h, bool) and h >= 0:
            info["horizon"] = h
        else:
            info["warnings"].append(f"HORIZON must be a non-negative int literal: runs use their `steps` "
                                    f"(default {NO_HORIZON_STEPS})")
    both = sorted(set(info["params"] or {}) & set(info["rules"] or {}))
    if both:
        info["warnings"].append(f"names declared in both PARAMS and RULES: {both} (an override sets both)")
    declared = list(info["params"] or {}) + [k for k in (info["rules"] or {}) if k not in (info["params"] or {})]
    info["unread"] = unread_names(tree, list(found.values()), declared)
    if info["unread"]:
        info["warnings"].append(f"declared but never read by the program, so an override of them has no effect: "
                                f"{info['unread'][:40]}" + (" ..." if len(info["unread"]) > 40 else "")
                                + " (hint: make the code branch on scenario[\"name\"] where a what-if may change it)")
    return info


def extract_block(text, label="model"):
    """Body of the LAST fenced block labelled exactly `label` (``` or ~~~ fences; CRLF; a missing closing fence
    runs to the end of the message). None if there is none."""
    if not isinstance(text, str) or not text:
        return None
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    open_re = re.compile(r"^\s{0,3}(`{3,}|~{3,})\s*" + re.escape(label) + r"\s*$")
    last, i = None, 0
    while i < len(lines):
        mo = open_re.match(lines[i])
        if not mo:
            i += 1
            continue
        fence = mo.group(1)
        close_re = re.compile(r"^\s{0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$")
        body, j = [], i + 1
        while j < len(lines) and not close_re.match(lines[j]):
            body.append(lines[j])
            j += 1
        if j >= len(lines):          # no closing fence: to the end of the message, without trailing blank lines
            while body and not body[-1].strip():
                body.pop()
        last = "\n".join(body)
        i = j + 1
    return last


def _ev_vars(node, out):
    if isinstance(node, tuple):
        if node and node[0] == "var":
            out.add(node[1])
        for x in node[1:]:
            _ev_vars(x, out)
    elif isinstance(node, list):
        for x in node:
            _ev_vars(x, out)
    return out


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _s3(v):
    """Compact number: 3 significant figures (integers >= 1000 in full)."""
    if v is None:
        return "-"
    if not _isnum(v):
        return str(v)
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return str(int(round(v)))
    s = f"{v:.3g}"
    return s


def _rle(tokens):
    out, i = [], 0
    while i < len(tokens):
        j = i
        while j + 1 < len(tokens) and tokens[j + 1] == tokens[i]:
            j += 1
        n = j - i + 1
        out.append(f"{tokens[i]}×{n}" if n >= 3 else " ".join([tokens[i]] * n))
        i = j + 1
    return " ".join(out)


def _signed(v):
    s = _g(float(v))
    return s if s.startswith("-") else "+" + s


def shape(series):
    """SHAPE text of a mean trajectory (list of floats/None, index = step)."""
    pts = [(t, v) for t, v in enumerate(series) if v is not None]
    if not pts:
        return "no values"
    t0, v0 = pts[0]
    tn, vn = pts[-1]
    tmax, vmax = max(pts, key=lambda p: p[1])
    tmin, vmin = min(pts, key=lambda p: p[1])
    vals = [v for _, v in pts]
    scale = max(abs(vmax), abs(vmin), 1e-12)
    if vmax - vmin <= 1e-9 * scale:
        return f"constant {_s3(v0)}"
    w = max(1, len(vals) // 12)   # smoothing half-width; steps of < 1% of the range count as flat (Monte Carlo noise)
    sm = [sum(vals[max(0, i - w):i + w + 1]) / len(vals[max(0, i - w):i + w + 1]) for i in range(len(vals))]
    eps = 0.01 * (vmax - vmin)
    signs = [1 if d > eps else -1 for d in (b - a for a, b in zip(sm, sm[1:])) if abs(d) > eps]
    changes = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    if changes == 0:
        kind = "monotone " + ("rising" if (signs and signs[0] > 0) else "falling")
    elif changes == 1:
        kind = "single-peak (rises, then falls)" if signs[0] > 0 else "single-trough (falls, then rises)"
    else:
        kind = f"oscillating ({changes} turns)"
    s = (f"start {_s3(v0)} (t={t0}) -> end {_s3(vn)} (t={tn}); max {_s3(vmax)} at t={tmax}, "
         f"min {_s3(vmin)} at t={tmin}; {kind}")
    total = vn - v0
    if abs(total) > 1e-9 * scale:
        def reach(f):
            for t, v in pts:
                if (v - v0) / total >= f:
                    return t
            return None
        s += f"; 50% of the change by t={reach(.5)}, 90% by t={reach(.9)}"
    return s


def _sample_idx(n_steps, series_list):
    """Steps to show in SERIES: all when <= 60 steps, else 60 uniform points + each extreme + the final step."""
    if n_steps + 1 <= SERIES_POINTS + 1:
        return list(range(n_steps + 1))
    idx = {round(n_steps * i / (SERIES_POINTS - 1)) for i in range(SERIES_POINTS)}
    idx.add(n_steps)
    for s in series_list:
        pts = [(t, v) for t, v in enumerate(s) if v is not None]
        if pts:
            idx.add(max(pts, key=lambda p: p[1])[0])
            idx.add(min(pts, key=lambda p: p[1])[0])
    return sorted(idx)


class G1C1Arm(Arm):
    name = "G1C1"
    buildable = True
    executable = True
    guide_file = "g1c1_guide.md"
    mutating_tools = ("submit_model", "patch_model")
    custom_episode = True
    TIMEOUT = 60

    def guide(self):
        p = ROOT / "tournament" / "prompts" / self.guide_file
        return p.read_text(encoding="utf-8") if p.exists() else ""

    # ------------------------------------------------------------------ schemas
    @staticmethod
    def _run_schema():
        return {
            "name": {"type": "string", "description": "label of this run"},
            "overrides": {"type": "object", "description": "values for declared PARAMS/RULES names, e.g. "
                          "{\"max_queue\": 5}; unknown names are an error"},
            "steps": {"type": "integer", "description": f"steps to run (default HORIZON, max {MAX_STEPS})"},
            "n_seeds": {"type": "integer", "description": f"seeded runs (default {DEFAULT_SEEDS}, max {MAX_SEEDS})"},
            "seed": {"type": "integer", "description": "base seed (default 0)"},
            "events": {"type": "array", "description": "event expressions over observation keys, e.g. \"queue > 10\" "
                       "or {expr, when: final|any|all|<step>}; 'any' events also give first-passage times"},
            "keys": {"type": "array", "items": {"type": "string"}, "description": "observation keys to show first"},
            "compare": {"type": "boolean", "description": "compare with the baseline under common random numbers "
                        "(default true)"},
            "what_if_patch": {"type": "object", "description": "{old, new}: a code change applied to THIS run only "
                              "(the stored model is unchanged)"},
        }

    def _runs_prop(self):
        return {"type": "array", "items": {"type": "object", "properties": self._run_schema()},
                "description": f"up to {MAX_RUNS} named what-if runs, each compared with the baseline"}

    def tools(self):
        return [
            _tool("submit_model", "Store the program (from `source`, or else from the last ```model fenced block in "
                  "this response), check it, and immediately run the baseline (HORIZON steps, 100 seeds) plus each "
                  "entry of `runs`.",
                  {"source": {"type": "string", "description": "the whole program (optional when it is in a "
                              "```model block of the same response)"}, "runs": self._runs_prop()}),
            _tool("patch_model", "Replace one exact, unique occurrence of `old` with `new` in the stored program, then "
                  "re-run the baseline and `runs`.",
                  {"old": {"type": "string"}, "new": {"type": "string"}, "runs": self._runs_prop()}, ["old", "new"]),
            _tool("run_scenario", "Run the stored program: either one run's fields or `runs` (up to 4). Each run "
                  "with overrides or a what_if_patch is compared with the baseline under common random numbers "
                  "(paired differences).", {**self._run_schema(), "runs": self._runs_prop()}),
            _tool("show_model", "Show the program with line numbers and the declared PARAMS/RULES table.", {}),
        ]

    def probe_tools(self):
        return [t for t in self.tools() if t["function"]["name"] in ("run_scenario", "show_model")]

    # ------------------------------------------------------------------ state / export
    def new_state(self):
        return ArmState(rep={"source": ""})

    def load(self, rep, given_gold=False):
        st = self.new_state()
        if isinstance(rep, dict):
            src = rep.get("source")
            if not isinstance(src, str):
                src = rep.get("code") if isinstance(rep.get("code"), str) else ""
            st.rep = {"source": src}
        elif isinstance(rep, str):
            st.rep = {"source": rep}
        st.given_gold = given_gold
        return st

    def _src(self, state):
        s = state.rep.get("source")
        return s if isinstance(s, str) else ""

    def export(self, state):
        src = self._src(state)
        info = analyse(src)
        return {"source": src, "params": info["params"], "rules": info["rules"], "horizon": info["horizon"]}

    def render(self, state):
        src = self._src(state)
        if not src:
            return "(no model code yet)"
        return "\n".join(f"{i + 1:4d}| {l}" for i, l in enumerate(src.splitlines()))[:60000]

    def _decl_table(self, info):
        out = []
        for name, key in (("PARAMS", "params"), ("RULES", "rules")):
            d = info[key]
            if d is None:
                out.append(f"{name}: (not declared)")
                continue
            out.append(f"{name} ({len(d)}):")
            for k, v in d.items():
                out.append(f"  {k} = {json.dumps(v)}"
                           + ("   [not read by the program: overriding it has no effect]"
                              if k in info.get("unread", ()) else ""))
        out.append(f"HORIZON: {info['horizon'] if info['horizon'] is not None else '(not declared)'}")
        return out

    # ------------------------------------------------------------------ dispatch
    def handle(self, state, name, args):
        try:
            args = _maybe_json(args) if args is not None else {}
            if not isinstance(args, dict):
                return f"{ERROR_PREFIX}: arguments must be an object (hint: see the {name} schema)"
            fn = getattr(self, "_t_" + str(name), None)
            if fn is None:
                return (f"{ERROR_PREFIX}: unknown tool '{name}' (hint: available: "
                        + ", ".join(t["function"]["name"] for t in self.tools()) + ")")
            if state.given_gold and name in self.mutating_tools:
                return (f"{ERROR_PREFIX}: the model is read-only in this mode (hint: use run_scenario with "
                        f"overrides or what_if_patch, and show_model)")
            out = fn(state, args)
        except Exception as e:
            out = f"{ERROR_PREFIX}: internal {type(e).__name__}: {e} (hint: check the arguments)"
        state.log.append((name, {k: (v if k != "source" else f"<{len(str(v))} chars>") for k, v in args.items()}
                          if isinstance(args, dict) else args, out[:200]))
        return out

    def _t_show_model(self, state, args):
        src = self._src(state)
        if not src:
            return "(no model code yet)"
        decl = "\n".join(["", "DECLARED"] + self._decl_table(analyse(src)))[:MAX_OUT // 2]
        code = self.render(state)
        room = MAX_OUT - len(decl) - 120
        if len(code) > room:
            cut = code[:room].rsplit("\n", 1)[0]
            code = cut + f"\n... [program truncated after line {cut.count(chr(10)) + 1} of {len(src.splitlines())}]"
        return code + "\n" + decl

    def _store_and_run(self, state, src, verb, args):
        if len(src) > MAX_CODE:
            return f"{ERROR_PREFIX}: program too long ({len(src)} chars, max {MAX_CODE})"
        state.rep["source"] = src
        info = analyse(src)
        head = [f"model {verb} ({len(src.splitlines())} lines); PARAMS: "
                f"{len(info['params']) if info['params'] is not None else 'none'}, RULES: "
                f"{len(info['rules']) if info['rules'] is not None else 'none'}, HORIZON: "
                f"{info['horizon'] if info['horizon'] is not None else 'none'}"]
        for e in info["errors"]:
            head.append(f"  error: {e}\n    hint: fix it with patch_model (or submit the corrected program)")
        for h in info["holes"]:
            head.append(f"  hole: {h}")
        for w in info["warnings"]:
            head.append(f"  warning: {w}")
        if not info["parses"] or not info["has_model"]:
            return f"{ERROR_PREFIX}: the program is stored but cannot run\n" + "\n".join(head)
        runs = args.get("runs")
        runs = _maybe_json(runs) if runs is not None else []
        if isinstance(runs, dict):
            runs = [runs]
        if not isinstance(runs, list):
            head.append(f"{ERROR_PREFIX}: runs must be a list of run objects; only the baseline was run")
            runs = []
        return self._run_many(state, [{"name": "baseline"}] + list(runs), head, explicit_baseline=True)

    def _t_submit_model(self, state, args):
        src = args.get("source")
        origin = "source"
        if not isinstance(src, str) or not src.strip():
            src = extract_block(state.__dict__.get("_last_text", ""))
            origin = "fenced block"
        if not isinstance(src, str) or not src.strip():
            return (f"{ERROR_PREFIX}: no program found (hint: put the whole program in a fenced block labelled "
                    f"model, i.e. a line ```model ... a line ```, in the same response, or pass it as `source`)")
        state.__dict__["_last_origin"] = origin
        return self._store_and_run(state, src, f"submitted from {origin}", args)

    def _t_patch_model(self, state, args):
        old, new = args.get("old"), args.get("new")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            return f"{ERROR_PREFIX}: patch_model needs non-empty string 'old' and string 'new'"
        src = self._src(state)
        if not src.strip():
            return f"{ERROR_PREFIX}: no program yet (hint: submit_model first)"
        n = src.count(old)
        if n != 1:
            return f"{ERROR_PREFIX}: " + self._patch_problem(src, old, n)
        return self._store_and_run(state, src.replace(old, new, 1), "patched", args)

    @staticmethod
    def _patch_problem(src, old, n):
        if n == 0:
            first = old.strip().splitlines()[0] if old.strip() else old
            close = difflib.get_close_matches(first.strip(), [l.strip() for l in src.splitlines()], n=3, cutoff=0.5)
            hint = ("closest lines: " + " | ".join(close)) if close else "use show_model to copy the exact text"
            return f"`old` text not found in the model (hint: {hint})"
        return f"`old` text occurs {n} times (hint: include more surrounding lines so it is unique)"

    def _t_run_scenario(self, state, args):
        if not self._src(state).strip():
            return f"{ERROR_PREFIX}: no model code yet (hint: call submit_model first)"
        info = analyse(self._src(state))
        if not info["parses"] or not info["has_model"]:
            return f"{ERROR_PREFIX}: the stored program cannot run: " + "; ".join(info["errors"] + info["holes"])
        runs = args.get("runs")
        if runs is not None:
            runs = _maybe_json(runs)
            if isinstance(runs, dict):
                runs = [runs]
            if not isinstance(runs, list) or not runs:
                return f"{ERROR_PREFIX}: runs must be a non-empty list of run objects"
            defaults = {k: v for k, v in args.items() if k in ("steps", "n_seeds", "seed", "events", "keys",
                                                                "compare")}
            runs = [{**defaults, **r} if isinstance(r, dict) else r for r in runs]
        else:
            runs = [{k: v for k, v in args.items() if k != "runs"}]
        return self._run_many(state, runs, [], explicit_baseline=False)

    # ------------------------------------------------------------------ run specification
    def _spec(self, src_info_cache, stored, r, k, horizon_default):
        """Normalise one run object. Returns (spec, error_text_or_None, notes)."""
        notes = []
        if not isinstance(r, dict):
            return None, f"run {k + 1} must be an object", notes
        name = str(r.get("name") or f"run{k}")[:40]
        ov = {}
        for field in ("overrides", "params", "rules", "scenario"):
            d = _maybe_json(r.get(field))
            if d is None:
                continue
            if not isinstance(d, dict):
                return None, f"run '{name}': {field} must be an object of name -> value", notes
            ov.update(d)
        patch = _maybe_json(r.get("what_if_patch"))
        src = stored
        if patch is not None:
            if not isinstance(patch, dict) or not isinstance(patch.get("old"), str) or not patch["old"] \
                    or not isinstance(patch.get("new"), str):
                return None, f"run '{name}': what_if_patch needs non-empty string 'old' and string 'new'", notes
            n = stored.count(patch["old"])
            if n != 1:
                return None, f"run '{name}': what_if_patch " + self._patch_problem(stored, patch["old"], n), notes
            src = stored.replace(patch["old"], patch["new"], 1)
        info = src_info_cache.get(src) or analyse(src)
        src_info_cache[src] = info
        if not info["parses"] or not info["has_model"]:
            return None, f"run '{name}': the patched program cannot run: " + "; ".join(info["errors"] + info["holes"]), \
                notes
        params, rules = info["params"], info["rules"]
        declared = list(params or {}) + [x for x in (rules or {}) if x not in (params or {})]
        unknown = [x for x in ov if x not in declared]
        if unknown:
            parts = []
            for u in unknown:
                close = difflib.get_close_matches(str(u), declared, n=2, cutoff=0.6)
                parts.append(repr(u) + (f" (did you mean {', '.join(map(repr, close))}?)" if close else ""))
            return None, (f"run '{name}': unknown override key(s) {', '.join(parts)}. Declared PARAMS: "
                          f"{list(params or {}) if params is not None else '(none declared)'}; RULES: "
                          f"{list(rules or {}) if rules is not None else '(none declared)'} (hint: declare the "
                          f"quantity in PARAMS/RULES, or use what_if_patch for a one-run code change)"), notes
        inert = [x for x in ov if x in info.get("unread", ())]
        if inert:
            notes.append(f"warning: run '{name}': {', '.join(map(repr, inert))} declared but never read by the "
                         f"program, so this override has no effect (hint: use what_if_patch to change the code "
                         f"that implements it)")
        for x, v in ov.items():
            dv = (params or {}).get(x, (rules or {}).get(x))
            if dv is not None and v is not None and type(dv) is not type(v) and not (_isnum(dv) and _isnum(v)):
                notes.append(f"note: run '{name}': override {x} changes type {type(dv).__name__} -> "
                             f"{type(v).__name__}")
        try:
            hz = info["horizon"] if info["horizon"] is not None else horizon_default
            steps = int(r.get("steps", hz) if r.get("steps") is not None else hz)
            n = int(r.get("n_seeds", DEFAULT_SEEDS) or DEFAULT_SEEDS)
            seed = int(r.get("seed", 0) or 0)
        except (TypeError, ValueError):
            return None, f"run '{name}': steps, n_seeds and seed must be integers", notes
        if steps < 0:
            return None, f"run '{name}': steps must be >= 0", notes
        if steps > MAX_STEPS:
            notes.append(f"note: run '{name}': steps capped at {MAX_STEPS}")
            steps = MAX_STEPS
        if n > MAX_SEEDS:
            notes.append(f"note: run '{name}': n_seeds capped at {MAX_SEEDS}")
            n = MAX_SEEDS
        n = max(1, n)
        try:
            json.dumps(ov)
        except (TypeError, ValueError) as e:
            return None, f"run '{name}': overrides are not JSON-serialisable: {e}", notes
        events = _parse_events(r.get("events"))
        keys = r.get("keys") if isinstance(r.get("keys"), list) else None
        compare = r.get("compare", True) is not False
        p2 = None if params is None else {**params, **{x: v for x, v in ov.items() if x in params}}
        r2 = None if rules is None else {**rules, **{x: v for x, v in ov.items() if x in rules}}
        scen = {**(params or {}), **(rules or {}), **ov}
        spec = {"name": name, "overrides": ov, "patched": patch is not None, "src": src, "params": p2,
                "rules": r2, "scen": scen, "steps": steps, "n": n, "seed": seed, "events": events,
                "keys": [str(x) for x in keys] if keys else None,
                "changed": bool(ov) or patch is not None, "compare": compare, "stored": src == stored}
        return spec, None, notes

    def _run_many(self, state, raw_runs, head, explicit_baseline):
        stored = self._src(state)
        lines = list(head)
        hz_default = NO_HORIZON_STEPS
        if len(raw_runs) - (1 if explicit_baseline else 0) > MAX_RUNS:
            extra = len(raw_runs) - (1 if explicit_baseline else 0) - MAX_RUNS
            lines.append(f"note: at most {MAX_RUNS} runs per call; the last {extra} were not run")
            raw_runs = raw_runs[:MAX_RUNS + (1 if explicit_baseline else 0)]
        cache, specs, problems = {}, [], []
        for k, r in enumerate(raw_runs):
            s, err, notes = self._spec(cache, stored, r, k if explicit_baseline else k + 1, hz_default)
            lines += notes
            if err:
                problems.append(err)
            else:
                specs.append(s)
        names = set()
        for s in specs:
            base, j = s["name"], 2
            while s["name"] in names:
                s["name"] = f"{base}_{j}"
                j += 1
            names.add(s["name"])
        if problems and not explicit_baseline:
            return f"{ERROR_PREFIX}: " + "\n".join(problems)
        # comparison baselines: an unchanged run on the stored source with the same (steps, seed), else a hidden one
        all_events = []
        for s in specs:
            for ev in s["events"]:
                if ev[0] not in [e[0] for e in all_events]:
                    all_events.append(ev)
        for s in specs:
            if not s["changed"]:
                s["events"] = s["events"] + [e for e in all_events if e[0] not in [x[0] for x in s["events"]]]
        for s in list(specs):
            if not (s["changed"] and s["compare"]):
                continue
            ref = next((b for b in specs if not b["changed"] and b["stored"] and b["steps"] == s["steps"]
                        and b["seed"] == s["seed"]), None)
            if ref is None:
                base_r = {"name": "baseline" if "baseline" not in names else f"baseline(steps={s['steps']},"
                          f"seed={s['seed']})", "steps": s["steps"], "n_seeds": s["n"], "seed": s["seed"]}
                ref, _, _ = self._spec(cache, stored, base_r, 0, hz_default)
                ref["events"] = list(all_events)
                ref["keys"] = s["keys"]
                names.add(ref["name"])
                specs.insert(0, ref) if not any(not b["changed"] for b in specs) else specs.append(ref)
            s["ref"] = ref["name"]
        if not specs:
            return f"{ERROR_PREFIX}: " + "\n".join(problems or ["nothing to run"])
        res = self._execute(specs)
        report = self._report(specs, res, lines, problems)
        ok_base = [s for s in specs if not s["changed"] and s["stored"]]
        if ok_base:
            r = res["runs"][specs.index(ok_base[0])]
            self._record_run(state, ok=not r["load_err"] and not r["failed"] and r["n_full"] == ok_base[0]["n"]
                             and not res["cut"])
        return report

    # ------------------------------------------------------------------ execution
    def _execute(self, specs, deadline=DEADLINE_S):
        files, runs = {}, []
        for k, s in enumerate(specs):
            fname = "model.py" if s["stored"] else f"model_r{k}.py"
            files[fname] = s["src"]
            evkeys = sorted(set().union(*[_ev_vars(ev[1], set()) for ev in s["events"] if ev[1] is not None])
                            if s["events"] else set())
            runs.append({"file": fname, "params": s["params"], "rules": s["rules"], "scen": s["scen"],
                         "steps": s["steps"], "n": s["n"], "seed": s["seed"], "evkeys": evkeys})
        files["params.json"] = json.dumps({"runs": runs, "deadline": deadline})
        out, err, timed_out, rc = _run_driver(files, self.TIMEOUT)
        recs = parse_lines(out)
        R = [{"load_err": None, "seeds": {}, "failed": [], "n_runs": 0, "n_full": 0, "keys": [], "mean": {},
              "ev": {}} for _ in specs]
        for tag, b in recs:
            if tag == "ERR" and 0 <= b.get("r", -1) < len(R):
                R[b["r"]]["load_err"] = b.get("err")
            elif tag == "RUN" and 0 <= b.get("r", -1) < len(R):
                x = R[b["r"]]
                x["n_runs"] += 1
                if b.get("err"):
                    x["failed"].append((b["i"], b["err"]))
                if b.get("final") is not None:
                    x["n_full"] += 1
                x["seeds"][b["i"]] = {"final": b.get("final"), "ev": b.get("ev") or {}, "n": b.get("n", 0)}
                for key in (b.get("final") or b.get("last") or {}):
                    if key not in x["keys"]:
                        x["keys"].append(key)
            elif tag == "AGG" and 0 <= b.get("r", -1) < len(R):
                R[b["r"]]["mean"] = b.get("mean") or {}
                agg_keys = b.get("keys") or []
                R[b["r"]]["keys"] = agg_keys + [k for k in R[b["r"]]["keys"] if k not in agg_keys]
        notes = []
        cut = next((b["i"] for t, b in recs if t == "CUT"), None)
        done = any(t == "DONE" for t, _ in recs)
        if timed_out:
            notes.append(f"[timed out after {self.TIMEOUT}s; partial results shown]"
                         "\n    hint: reduce steps, n_seeds or the number of runs, or remove unbounded loops")
        elif not done:
            tail = err.strip().splitlines()[-3:] if err.strip() else []
            msg = "memory limit exceeded" if "memory limit" in err else f"process ended unexpectedly (exit code {rc})"
            notes.append(f"[{msg}; partial results shown]" + ("\n    " + "\n    ".join(tail) if tail else ""))
        if cut is not None:
            notes.append(f"note: time limit: seeds reduced to {cut} per run (all runs stopped at the same seed, so "
                         f"comparisons stay paired); reduce steps or runs for more seeds")
        prints = [b for t, b in recs if t == "PRINT"]
        if prints:
            notes.append("model printed: " + prints[0][:300].replace("\n", " | "))
        return {"runs": R, "notes": notes, "cut": cut, "incomplete": timed_out or not done}

    # ------------------------------------------------------------------ report
    def _report(self, specs, res, pre_lines, problems):
        R = res["runs"]
        header = list(pre_lines)
        for p in problems:
            header.append(f"{ERROR_PREFIX}: {p}")
        for s, r in zip(specs, R):
            h = f"RUN {s['name']}: steps {s['steps']}, seeds {r['n_runs']}/{s['n']} completed, {r['n_full']} reached " \
                f"t={s['steps']}"
            if s["overrides"]:
                h += f"; overrides {_short(s['overrides'], 300)}"
            if s["patched"]:
                h += "; ephemeral patch applied (stored model unchanged)"
            if s.get("ref"):
                h += f"; compared with {s['ref']}"
            header.append(h)
            if r["load_err"]:
                header.append(f"  {ERROR_PREFIX}: " + fmt_err(r["load_err"]))
            if r["failed"]:
                i, e = r["failed"][0]
                header.append(f"  {len(r['failed'])} of {r['n_runs']} seed(s) raised; first (seed index {i}): "
                              + fmt_err(e) + "\n    (those seeds contribute observations up to the failing step)")
        header += res["notes"]

        def final_block(limit):
            out = []
            for s, r in zip(specs, R):
                full = [v["final"] for v in r["seeds"].values() if v["final"] is not None]
                if not full:
                    out.append(f"FINAL [{s['name']}]: no seed reached t={s['steps']}")
                    continue
                req = [k for k in (s["keys"] or []) if k in r["keys"]]
                others = [k for k in r["keys"] if k not in req]
                show = req + others[:limit]
                out.append(f"FINAL [{s['name']}] (t={s['steps']}, n={len(full)}):")
                for k in show:
                    out.append(f"  {k}: " + _summ([o.get(k) for o in full]))
                if len(others) > limit:
                    out.append(f"  (+{len(others) - limit} more keys: {', '.join(others[limit:limit + 12])}"
                               + (" ..." if len(others) > limit + 12 else "") + ")")
                if s["keys"]:
                    miss = [k for k in s["keys"] if k not in r["keys"]]
                    if miss:
                        out.append(f"  note: keys never observed: {miss} (observed: {r['keys'][:20]})")
            return out

        def effect_block(limit):
            out = []
            by = {s["name"]: (s, r) for s, r in zip(specs, R)}
            for s, r in zip(specs, R):
                if not s.get("ref"):
                    continue
                b_s, b_r = by[s["ref"]]
                common = sorted(i for i in r["seeds"] if r["seeds"][i]["final"] is not None
                                and i in b_r["seeds"] and b_r["seeds"][i]["final"] is not None)
                out.append(f"EFFECT [{s['name']}] vs {s['ref']} (paired over {len(common)} common seeds):")
                if not common:
                    out.append("  no common seed reached the final step")
                    continue
                keys = [k for k in (s["keys"] or []) if k in r["keys"]] + \
                       [k for k in r["keys"] if k not in (s["keys"] or [])]
                changed, same, shown = [], [], 0
                for k in keys:
                    a = [b_r["seeds"][i]["final"].get(k) for i in common]
                    c = [r["seeds"][i]["final"].get(k) for i in common]
                    pairs = [(x, y) for x, y in zip(a, c) if x is not None and y is not None]
                    if not pairs:
                        continue
                    if all(isinstance(x, bool) and isinstance(y, bool) for x, y in pairs):
                        pa, pc = sum(x for x, _ in pairs) / len(pairs), sum(y for _, y in pairs) / len(pairs)
                        if all(x == y for x, y in pairs):
                            same.append(k)
                            continue
                        changed.append(f"  {k}: P(true) {pa:.3f} -> {pc:.3f} (diff {pc - pa:+.3f}; "
                                       f"differs on {sum(x != y for x, y in pairs)} of {len(pairs)} seeds)")
                    elif all(_isnum(x) and _isnum(y) for x, y in pairs):
                        d = sorted(float(y) - float(x) for x, y in pairs)
                        if all(v == 0 for v in d):
                            same.append(k)
                            continue
                        changed.append(f"  {k}: mean diff {_signed(sum(d) / len(d))} | p05 {_signed(_quant(d, .05))} "
                                       f"p50 {_signed(_quant(d, .5))} p95 {_signed(_quant(d, .95))} | "
                                       f"P(diff>0)={sum(v > 0 for v in d) / len(d):.2f} "
                                       f"P(diff<0)={sum(v < 0 for v in d) / len(d):.2f} (n={len(d)})")
                    else:
                        nd = sum(1 for x, y in pairs if x != y)
                        if nd == 0:
                            same.append(k)
                        else:
                            changed.append(f"  {k}: differs on {nd} of {len(pairs)} seeds")
                for ln in changed:
                    if shown >= limit + len(s["keys"] or []):
                        out.append(f"  (+{len(changed) - shown} more changed keys omitted)")
                        break
                    out.append(ln)
                    shown += 1
                if same:
                    out.append(f"  unchanged on every seed: {', '.join(same[:30])}" + (" ..." if len(same) > 30 else ""))
            return out

        events, shapes, series = [], [], []
        for s, r in zip(specs, R):
            evl = []
            for label, east, when in s["events"]:
                if east is None:
                    evl.append(f"  [{s['name']}] {label}: error: {when}")
                    continue
                evl.append(f"  [{s['name']}] " + self._event_line(label, east, when, s, r))
            events.append(evl)
            mean = r["mean"]
            numk = [k for k in r["keys"] if k in mean]
            pref = [k for k in (s["keys"] or []) if k in mean]
            varying = [k for k in numk if k not in pref and _varies(mean[k])]
            rest = [k for k in numk if k not in pref and k not in varying]
            chosen = (pref + varying + rest)[:SERIES_KEYS]
            sh = [f"  [{s['name']}] {k}: " + shape(mean[k]) for k in chosen]
            if rest and len(pref + varying) < SERIES_KEYS:
                sh = [l for l, k in zip(sh, chosen) if k not in rest]
                sh.append(f"  [{s['name']}] constant: " + ", ".join(f"{k}={_s3(next((v for v in mean[k] if v is not None), None))}"
                                                                    for k in rest[:20]))
            shapes.append(sh)
            ser_keys = [k for k in chosen if k not in rest] or chosen
            if ser_keys and s["steps"] > 0:
                idx = _sample_idx(s["steps"], [mean[k] for k in ser_keys])
                blk = [f"  [{s['name']}] t: " + _rle([str(i) for i in idx]) if len(idx) < s["steps"] + 1
                       else f"  [{s['name']}] t = 0..{s['steps']} (every step)"]
                for k in ser_keys:
                    blk.append(f"  [{s['name']}] {k}: " + _rle([_s3(mean[k][i]) if i < len(mean[k]) else "-"
                                                                for i in idx]))
                series.append(blk)
        return _assemble(header, final_block, effect_block,
                         [("EVENTS", events), ("SHAPE (mean trajectory)", shapes), ("SERIES (mean per step)", series)],
                         bool(problems) or any(r["load_err"] or r["failed"] for r in R)
                         or not any(r["n_runs"] for r in R) or res["incomplete"])

    def _event_line(self, label, east, when, s, r):
        def ev(o):
            try:
                return bool(_expr.eval_on(east, {k: v for k, v in o.items() if v is not None}))
            except (_expr.EvalError, _expr.ExprError):
                return None
        rs, fpt = [], []
        for i, sd in sorted(r["seeds"].items()):
            evd = sd["ev"]
            if when == "final" and sd["final"] is None:
                continue
            obs = [{k: (vs[t] if t < len(vs) else None) for k, vs in evd.items()} for t in range(sd["n"])]
            if not obs:
                rs.append(None)
                continue
            if when == "final":
                rs.append(ev(obs[-1]))
            elif isinstance(when, int):
                rs.append(ev(obs[when]) if 0 <= when < len(obs) else None)
            else:
                vals = [ev(o) for o in obs]
                if when == "any":
                    if any(v is True for v in vals):
                        rs.append(True)
                        fpt.append(next(t for t, v in enumerate(vals) if v is True))
                    else:
                        rs.append(False if all(v is False for v in vals) else None)
                else:
                    rs.append(False if any(v is False for v in vals) else (True if all(v is True for v in vals)
                                                                            else None))
        defined = [x for x in rs if x is not None]
        if not defined:
            return (f"P({label}) [{when}] = undefined (no run defines it; check key names: "
                    f"{r['keys'][:15]})")
        p = sum(defined) / len(defined)
        se = math.sqrt(p * (1 - p) / len(defined))
        und = len(rs) - len(defined)
        out = f"P({label}) [{when}] = {p:.3f} +/- {se:.3f} (n={len(defined)}" + (f", {und} undefined" if und else "") + ")"
        if when == "any" and fpt:
            xs = sorted(float(x) for x in fpt)
            out += (f"; first passage t: mean {_g(sum(xs) / len(xs))} p05 {_g(_quant(xs, .05))} "
                    f"p50 {_g(_quant(xs, .5))} p95 {_g(_quant(xs, .95))} (n={len(xs)})")
        return out

    # ------------------------------------------------------------------ validity
    def _record_run(self, state, ok):
        h = hashlib.sha1(self._src(state).encode("utf-8")).hexdigest()
        prev = state.__dict__.setdefault("_runs", {}).get(h, {})
        state.__dict__["_runs"][h] = {"ok": ok or prev.get("ok", False)}

    def validity(self, state):
        src = self._src(state)
        info = analyse(src)
        holes, errors = list(info["holes"]), list(info["errors"])
        usable = bool(src.strip()) and info["parses"] and info["has_model"]
        executable = False
        if usable:
            h = hashlib.sha1(src.encode("utf-8")).hexdigest()
            if state.__dict__.get("_runs", {}).get(h, {}).get("ok"):
                executable = True
            else:
                executable, err = self._smoke(state)
                if err:
                    errors.append(err)
        return {"usable": usable, "executable": executable, "holes": holes, "errors": errors,
                "warnings": info["warnings"]}

    def _smoke(self, state):
        s, err, _ = self._spec({}, self._src(state), {"name": "smoke", "steps": 2, "n_seeds": 2}, 0, 2)
        if err:
            return False, err
        res = self._execute([s])
        r = res["runs"][0]
        if r["load_err"]:
            return False, fmt_err(r["load_err"])
        if r["failed"]:
            return False, fmt_err(r["failed"][0][1])
        if res["incomplete"]:
            return False, res["notes"][0] if res["notes"] else "smoke run did not complete"
        return r["n_runs"] > 0, None

    # ------------------------------------------------------------------ episode (document channel)
    def run_episode(self, ep, state, messages):
        """Generic tool_loop plus: system-prompt parity, the fenced ```model channel with implicit submit, and a
        budget status line on every executor result. Returns (final_text, messages, trace)."""
        from tournament.episode import FORCE_FINAL_MSG, MAX_TOOL_CALLS, _assistant, execute_tool_call
        from tournament.run import system_prompt
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_prompt(self, True, ep.budget_calls, ep.budget_tokens)
        gold = bool(state.given_gold)
        tools_fn = self.gold_tools if gold else self.current_tools
        trace, n_tools = [], 0

        def budget():
            return "\n" + BUDGET_LINE.format(k=len(ep.calls), n=ep.budget_calls)

        while True:
            tools = tools_fn(state) or None
            if tools and not ep.can_continue():
                break
            m = ep.call(messages, tools=tools, label=f"turn{len(ep.calls)}", final=not tools)
            content = m.get("content") or ""
            state.__dict__["_last_text"] = content
            tcs = (m.get("tool_calls") or []) if tools else []
            am = _assistant(m)
            block = None if gold else extract_block(content)
            if not tcs:
                am.pop("tool_calls", None)
                messages.append(am)
                if block is None or not block.strip() or not tools:
                    return am["content"], messages, trace
                # a model block without a tool call: implicit submit_model (one tool call), then continue
                if n_tools >= MAX_TOOL_CALLS:
                    res = f"{ERROR_PREFIX}: tool-call limit ({MAX_TOOL_CALLS}) reached; give your FINAL ANSWER."
                else:
                    try:
                        res = self.handle(state, "submit_model", {"source": block})
                    except Exception as e:  # handle never raises; belt and braces
                        res = f"{ERROR_PREFIX}: internal error in submit_model: {type(e).__name__}: {str(e)[:300]}"
                n_tools += 1
                res = res + budget()
                trace.append({"turn": len(ep.calls), "tool": "submit_model",
                              "args": {"source": "fenced block", "chars": len(block)}, "result": res[:3000],
                              "error": res.startswith(ERROR_PREFIX)})
                messages.append({"role": "user", "content": "EXECUTOR RESULT (submit_model)\n" + res})
                continue
            messages.append(am)
            offered = {t["function"]["name"] for t in tools}
            submitted = False
            for tc in tcs:
                name, args, res = execute_tool_call(self, state, tc, offered, n_tools, MAX_TOOL_CALLS)
                n_tools += 1
                submitted = submitted or name == "submit_model"
                res = res + budget()
                trace.append({"turn": len(ep.calls), "tool": name, "args": args, "result": res[:3000],
                              "error": res.startswith(ERROR_PREFIX)})
                messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call{len(trace)}", "content": res})
            if block and not submitted and not gold and block.strip() != self._src(state).strip() and messages:
                messages[-1]["content"] += ("\nnote: the ```model block in this response was not submitted "
                                            "(call submit_model to store and run it)")
        ep.forced_final = True
        messages.append({"role": "user", "content": FORCE_FINAL_MSG})
        m = ep.call(messages, tools=None, label="forced_final", final=True)
        messages.append({"role": "assistant", "content": m.get("content") or ""})
        return m.get("content") or "", messages, trace


def _summ(vals):
    """A4's summarize, except that a value identical on every seed (deterministic paths) is printed once."""
    ok = [v for v in vals if v is not None]
    if len(ok) > 1 and all(type(v) is type(ok[0]) and v == ok[0] for v in ok):
        return f"{_g(ok[0]) if _isnum(ok[0]) else repr(ok[0])} on all {len(ok)} seeds"
    return summarize(vals)


def _varies(xs):
    v = [x for x in xs if x is not None]
    return bool(v) and (max(v) - min(v)) > 1e-9 * max(1e-12, abs(max(v)), abs(min(v)))


def _run_driver(files, timeout):
    """a3_sim.run_in_sandbox with this arm's DRIVER as main.py (same sandbox runner, audit hook, memory watchdog
    and isolation; a3_sim.run_in_sandbox hard-wires a3_sim.DRIVER, which other arms use concurrently)."""
    with tempfile.TemporaryDirectory(prefix="rpp_g1c1_") as d:
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


def _assemble(header, final_block, effect_block, optional, is_error):
    """Bounded output: header + FINAL + EFFECT always (shrinking the 'other keys' lists if needed); then EVENTS,
    SHAPE, SERIES items in that order while they fit. Dropped items are reported."""
    core = None
    for limit in (40, 20, 10, 5, 2, 0):
        core = header + final_block(limit) + effect_block(limit)
        if len("\n".join(core)) <= MAX_OUT - 400:
            break
    text = "\n".join(core)
    if len(text) > MAX_OUT - 200:
        text = text[:MAX_OUT - 300] + "\n...[FINAL/EFFECT cut at the output limit; request fewer runs or keys]"
    omitted = []
    stop = False
    for title, groups in optional:
        items = [g for g in groups if g]
        if not items:
            continue
        if stop:
            omitted.append(f"{title}: {len(items)} run(s)")
            continue
        add = "\n" + title + ":"
        for n_done, g in enumerate(items):
            blk = add + "\n" + "\n".join(g)        # the title goes out only together with its first item
            if len(text) + len(blk) > MAX_OUT - 200:
                stop = True
                omitted.append(f"{title}: {len(items) - n_done} run(s)")
                break
            text += blk
            add = ""
    if omitted:
        text += "\n[output limit: omitted " + "; ".join(omitted) + "]"
    if is_error and not text.startswith(ERROR_PREFIX):
        text = f"{ERROR_PREFIX}: see the problems below\n" + text
    return text[:MAX_OUT]


G1C1 = G1C1Arm
