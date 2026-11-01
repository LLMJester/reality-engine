"""G3C1: parameterised probabilistic program, run-first acceptance and whole-section editing
(G2C2 modified, construction only; runs/discovery/gen3/specs/G3C1.md).

Representation (unchanged from G2C2 / G1C1): a Python program with the A4 API (`Model.init_state(scenario[, rng])`,
`step(state, t, rng)`, `observe(state)`), preceded by two declared literal dictionaries `PARAMS` and `RULES` and an
int `HORIZON`, each entry with a provenance comment. What-ifs are re-runs with overrides under common random numbers;
`what_if_patch` is an ephemeral one-run code change. Every run result carries the ASSUMED block.

This file is a copy of tournament/arms/g2c2.py (which stays unchanged as an archived parent) with only the four
construction changes of the spec:
1. narrower acceptance gate: only fatal problems (syntax errors, no class Model) and rejections (undefined names,
   forbidden imports) stop a run. Declared-but-unread PARAMS/RULES entries are WARNINGS and the program runs. An
   undeclared override key fails only its own run. The no-op what-if rejection (and its `no_effect_expected`
   exemption and escape valve) is the only rejection left after the baseline has run.
2. `edit_model` replaces whole named sections (PARAMS, RULES, HORIZON, Model, Model.<method>, HELPERS, <function>)
   located on the AST of the syntax-repaired program; indentation is normalised by the executor. `patch_model` and
   its line-number / old-new edits are gone (old/new survives only inside the ephemeral `what_if_patch`).
3. every static problem is listed at once, each tagged with its section (`[Model.step] L22:29 ...`).
4. `show_model` with no stored program returns a short instruction to submit the whole draft (not an error).
The episode loop, lenient argument parse, sandbox driver, summaries and validity are G2C2's.

Gold (spec section 5): G2C2's gold programs are reused unchanged when they pass G3C1's static report (0 fatal,
0 reject); tournament/gold/g3c1/<ID>.notes.md records the reuse.
"""
import ast
import builtins
import copy
import difflib
import io
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import tokenize
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
MAX_SYNTAX_REPAIRS = 10      # independent syntax errors listed per submission
MAX_ASSUMED = 1400           # characters of the ASSUMED block
LINES_HINT = "send the program as `lines`, an array of strings; single quotes only"
REPAIR_NOTE = "note: arguments repaired (lenient parse)"
FIX_HINT = "fix them all in one `edit_model` call; each entry replaces a whole section"
EMPTY_SHOW = ("No program yet. Call submit_model with your whole draft in `lines` now; the result lists every problem "
              "at once.")
EDIT_HINT = ("use edit_model with `sections`: [{\"name\": \"Model.step\", \"lines\": [...]}], each entry replacing a "
             "whole section; line-number and old/new edits do not exist (a full rewrite is submit_model)")
MAX_WARNINGS_SHOWN = 12      # warning items listed in a run result (the rest are counted)
DECL_NAMES = ("PARAMS", "RULES", "HORIZON")
# modules whose purpose the sandbox blocks (network, processes, file-system tools): an import is a static problem
FORBIDDEN_MODULES = ("subprocess", "socket", "multiprocessing", "urllib", "http", "requests", "ftplib", "smtplib",
                     "telnetlib", "shutil", "webbrowser")

# ---------------------------------------------------------------------------------------------------
# Sandbox driver (copied from a3_sim.DRIVER and extended; a3_sim/a4_prob are imported by other arms and stay
# unchanged). Reads params.json = {"runs": [...], "deadline": s}; each run names its own source file.
DRIVER = r'''
import copy, hashlib, io, json, math, random, sys, time, traceback, inspect
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
        # G2C2 extension 4: digest of every observed key on every step (exact no-op what-if detection)
        h = hashlib.md5(json.dumps(obs, sort_keys=True, default=repr).encode("utf-8")).hexdigest()
        emit("RUN", {"r": k, "i": i, "n": len(obs), "final": obs[-1] if full else None,
                     "last": obs[-1] if (obs and not full) else None,
                     "ev": {key: [o.get(key) for o in obs] for key in R["evkeys"]}, "err": err, "h": h})
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
    # G2C2: the provenance comment of every entry is parsed (`# stated`, `# derived`, `# assumed: why`, `unused:`)
    info["prov"] = provenance(src, found)
    info["unread_exempt"] = [k for k in info["unread"] if (info["prov"].get(k) or {}).get("unused") is not None]
    return info


_PROV_RE = re.compile(r"\b(stated|derived|assumed|assumption|assume)\b", re.I)


def _comments(src):
    """{line number: comment text without '#'} via tokenize (a '#' inside a string is not a comment)."""
    out = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                out.setdefault(tok.start[0], tok.string.lstrip("#").strip())
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def provenance(src, found):
    """{entry name: {"decl": "PARAMS"|"RULES", "line", "col", "kind": stated|derived|assumed|None, "reason",
    "unused": reason or None}} for the top-level entries of literal PARAMS / RULES dicts. The comment on the
    entry's lines (key line to value end) is used; an entry without stated/derived/assumed counts as assumed."""
    com = _comments(src)
    out = {}
    for decl in ("PARAMS", "RULES"):
        node = found.get(decl)
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                continue
            c = next((com[ln] for ln in range(k.lineno, (v.end_lineno or k.lineno) + 1) if ln in com), "")
            mo = _PROV_RE.search(c)
            kind = None
            reason = ""
            if mo:
                kind = mo.group(1).lower()
                kind = "assumed" if kind.startswith("assum") else kind
                reason = c[mo.end():].strip().lstrip(":-–— ").strip()
            mu = re.search(r"\bunused\s*:\s*(.*)", c, re.I)
            if k.value not in out:
                out[k.value] = {"decl": decl, "line": k.lineno, "col": k.col_offset + 1, "kind": kind,
                                "reason": reason, "comment": c,
                                "unused": (mu.group(1).strip() or "no reason given") if mu else None}
    return out


def assumed_entries(info):
    """[(name, value, reason)] for every declared entry whose provenance is `assumed` or missing."""
    out = []
    prov = info.get("prov") or {}
    for key in ("params", "rules"):
        for k, v in (info.get(key) or {}).items():
            p = prov.get(k)
            if p is None or p["kind"] in (None, "assumed"):
                why = (p or {}).get("reason") if p and p["kind"] == "assumed" else None
                out.append((k, v, why or "no provenance comment"))
    return out


# ---------------------------------------------------------------------------------------------------
# G2C2 change 3: every static problem at once, with line references

def repair_syntax(src, limit=MAX_SYNTAX_REPAIRS):
    """Parse; on a SyntaxError record (line, col, msg), replace the offending line by `pass` (or `if True:` when
    it opened a block) at the same indentation and re-parse, up to `limit` times. Returns (errors, repaired_src,
    tree_or_None)."""
    lines = src.split("\n")
    errs, seen = [], set()
    for _ in range(limit + 1):
        cur = "\n".join(lines)
        try:
            return errs, cur, ast.parse(cur)
        except SyntaxError as e:
            if len(errs) >= limit:
                break
            ln = e.lineno
            if ln in seen:                  # the repair of this line did not help: stop, do not list it twice
                break
            errs.append((ln, e.offset, e.msg if not isinstance(e, IndentationError) else f"{e.msg} (indentation)"))
            if not ln or ln < 1 or ln > len(lines):
                break
            seen.add(ln)
            old = lines[ln - 1]
            ind = old[:len(old) - len(old.lstrip())]
            if isinstance(e, IndentationError):     # re-indent relative to the previous code line
                prev = next((l for l in reversed(lines[:ln - 1]) if l.strip() and not l.lstrip().startswith("#")),
                            "")
                pind = prev[:len(prev) - len(prev.lstrip())]
                ind = pind + "    " if prev.split("#", 1)[0].rstrip().endswith(":") else pind
            code = old.split("#", 1)[0].rstrip()
            lines[ln - 1] = ind + ("if True:" if code.endswith(":") else "pass")
        except (ValueError, RecursionError, MemoryError) as e:
            errs.append((None, None, f"{type(e).__name__}: {e}"))
            break
    return errs, "\n".join(lines), None


def _bound_names(tree):
    bound = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.Import):
            bound.update((a.asname or a.name.split(".")[0]) for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            bound.update((a.asname or a.name) for a in n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound.update(n.names)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            bound.add(n.rest)
    return bound


_BUILTIN_NAMES = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__builtins__", "__spec__"}


def undefined_names(tree, extra=()):
    """Names loaded anywhere that are never bound anywhere (scope-insensitive, so it only errs towards silence),
    imported, built in or a parameter. Returns [(name, line, col, only_as_attribute_base, suggestions)]."""
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
        return []
    bound = _bound_names(tree)
    parents = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[id(c)] = p
    uses = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound \
                and n.id not in _BUILTIN_NAMES:
            par = parents.get(id(n))
            attr = isinstance(par, ast.Attribute) and par.value is n
            uses.setdefault(n.id, []).append((n.lineno, n.col_offset + 1, attr))
    out = []
    pool = sorted((bound | set(extra)) - _BUILTIN_NAMES)
    for name, us in uses.items():
        us.sort()
        sugg = difflib.get_close_matches(name, pool, n=2, cutoff=0.75)
        out.append((name, us[0][0], us[0][1], all(a for _, _, a in us), sugg))
    return sorted(out, key=lambda x: (x[1], x[2]))


def forbidden_imports(tree):
    out = []
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.Import):
            mods = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods = [n.module]
        for m in mods:
            if m.split(".")[0] in FORBIDDEN_MODULES:
                out.append((m, n.lineno, n.col_offset + 1))
    return out


# ---------------------------------------------------------------------------------------------------
# G3C1 change 2: named sections, located on the AST of the syntax-repaired program

def _decl_target(node):
    """'PARAMS' / 'RULES' / 'HORIZON' when a top-level statement assigns one of them, else None."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        targets = [node.target]
    else:
        return None
    for t in targets:
        if isinstance(t, ast.Name) and t.id in DECL_NAMES:
            return t.id
    return None


def _with_comments_above(lines, start):
    """Extend a 1-based start line upwards over the comment lines directly above it (no blank line between)."""
    while start > 1 and lines[start - 2].strip().startswith("#"):
        start -= 1
    return start


def _node_span(lines, node):
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", None) or []])
    return _with_comments_above(lines, first), node.end_lineno or node.lineno


def locate_sections(src, tree):
    """Sections of a program whose AST is `tree` (line numbers of `src`; repair_syntax keeps them). Returns a dict:
      spans:   {name: (start, end)} 1-based inclusive, for PARAMS / RULES / HORIZON, Model, Model.<method> and every
               top-level function (the last definition of a name wins, as in Python)
      helpers: [(start, end)] every top-level statement that is not a declaration or the Model class
      methods: [method names]; functions: [top-level function names]; model: span of Model or None
      anchor:  line before which HELPERS and new helper functions go (PARAMS, else the first declaration or Model)
      decl_anchor: line before which a missing declaration goes (the Model class, else the end)"""
    lines = src.split("\n")
    spans, helpers, methods, functions = {}, [], [], []
    model, model_node = None, None
    for node in tree.body:
        d = _decl_target(node)
        if d:
            spans[d] = _node_span(lines, node)
        elif isinstance(node, ast.ClassDef) and node.name == "Model":
            model, model_node = _node_span(lines, node), node
        else:
            helpers.append(_node_span(lines, node))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans[node.name] = helpers[-1]
                if node.name not in functions:
                    functions.append(node.name)
    if model_node is not None:
        spans["Model"] = model
        for node in model_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans["Model." + node.name] = _node_span(lines, node)
                if node.name not in methods:
                    methods.append(node.name)
    starts = [spans[k][0] for k in DECL_NAMES if k in spans] + ([model[0]] if model else [])
    anchor = spans["PARAMS"][0] if "PARAMS" in spans else (min(starts) if starts else len(lines) + 1)
    return {"spans": spans, "helpers": helpers, "methods": methods, "functions": functions, "model": model,
            "anchor": anchor, "decl_anchor": model[0] if model else len(lines) + 1}


def section_names(sec):
    """Section names in program order (HELPERS first when there are helpers)."""
    names = [k for k in DECL_NAMES if k in sec["spans"]]
    if sec["model"]:
        names += ["Model"] + ["Model." + m for m in sec["methods"]]
    return (["HELPERS"] if sec["helpers"] else []) + sec["functions"] + names


def _text_tags(lines):
    """Rough per-line section tags from indentation alone (used only when the program cannot be parsed even after
    the syntax repair)."""
    tags, cur, in_model, mind = [], "HELPERS", False, None
    for l in lines:
        code = l.split("#", 1)[0].rstrip()
        if code.strip():
            ind = len(l) - len(l.lstrip())
            word = code.strip()
            if ind == 0:
                in_model, mind = False, None
                m = re.match(r"(?:async\s+)?def\s+(\w+)", word)
                if word.startswith("class Model"):
                    cur, in_model = "Model", True
                elif m:
                    cur = m.group(1)
                elif re.match(r"(PARAMS|RULES|HORIZON)\b", word):
                    cur = word.split()[0].split("=")[0].split(":")[0]
                elif not word[0] in ")]}":
                    cur = "HELPERS"
            elif in_model:
                m = re.match(r"(?:async\s+)?def\s+(\w+)", word)
                if m and (mind is None or ind <= mind):
                    mind, cur = ind, "Model." + m.group(1)
                elif mind is not None and ind < mind:
                    cur = "Model"
        tags.append(cur)
    return tags


def line_tags(src, tree):
    """Section tag of every line (index line - 1)."""
    lines = src.split("\n")
    if tree is None:
        return _text_tags(lines)
    sec = locate_sections(src, tree)
    tags = ["HELPERS"] * len(lines)
    order = [("HELPERS", s) for s in sec["helpers"]] + \
            [(k, v) for k, v in sec["spans"].items() if k in DECL_NAMES or k in sec["functions"]] + \
            ([("Model", sec["model"])] if sec["model"] else []) + \
            [(k, v) for k, v in sec["spans"].items() if k.startswith("Model.")]
    for name, (a, b) in order:          # later (more specific) sections overwrite earlier ones
        for i in range(a, b + 1):
            if 1 <= i <= len(lines):
                tags[i - 1] = name
    return tags


# ---------------------------------------------------------------------------------------------------
# G3C1 changes 1 and 3: the narrower gate; every static problem at once, tagged by section

def static_report(src, runs=None):
    """All static problems of a program at once. Problems are (line, col, msg, key, section). Returns a dict:
      fatal:   the program cannot run (syntax errors, no class Model): stored as a draft, not run
      reject:  undefined names (not attribute-only), forbidden imports: stored as a draft, not run
      warning: declared-but-unread PARAMS/RULES entries (without `# unused: why`): the program runs
      notes:   [str] not blocking (quotes/backslashes, attribute-base names, holes, declaration warnings)
      info:    analyse() of the repaired program
    `runs` is accepted for compatibility and ignored: an undeclared override key fails only its own run."""
    rep = {"fatal": [], "reject": [], "warning": [], "notes": [], "info": None}
    if not isinstance(src, str) or not src.strip():
        rep["fatal"].append((None, None, "no program code", "fatal:empty", "program"))
        return rep
    errs, fixed, tree = repair_syntax(src)
    tags = line_tags(fixed, tree)

    def tag(ln, default="program"):
        return tags[ln - 1] if ln and 1 <= ln <= len(tags) else default
    for ln, col, msg in errs:
        rep["fatal"].append((ln, col, f"syntax error: {msg}", f"syntax:{ln}", tag(ln)))
    if tree is None:
        rep["fatal"].append((None, None, "further syntax errors could not be located (fix the ones above first)",
                             "syntax:more", "program"))
        return rep
    info = analyse(fixed)
    rep["info"] = info
    if not info["has_model"]:
        rep["fatal"].append((None, None, "no class Model (hint: define class Model with init_state, step, observe)",
                             "fatal:model", "Model"))
    for h in info["holes"]:
        if h != "no class Model":
            rep["notes"].append(f"hole: {h}")
    for w in info["warnings"]:
        rep["notes"].append(f"warning: {w}")
    prov = info.get("prov") or {}
    for k in info["unread"]:
        p = prov.get(k) or {}
        if p.get("unused") is not None:
            continue
        decl = p.get("decl", "PARAMS/RULES")
        rep["warning"].append((p.get("line"), p.get("col"),
                               f"WARNING unread: {decl} entry '{k}'" + (f" (L{p['line']})" if p.get("line") else "")
                               + " is declared but never read, so overriding it will change nothing. If a what-if "
                                 "needs it, wire it into the mechanism.", f"unread:{k}", tag(p.get("line"), decl)))
    declared = list(info["params"] or {}) + [k for k in (info["rules"] or {}) if k not in (info["params"] or {})]
    for name, ln, col, attr_only, sugg in undefined_names(tree, declared):
        msg = f"undefined name '{name}'" + (f" (did you mean {' or '.join(repr(s) for s in sugg)}?)" if sugg else "")
        if attr_only:
            rep["notes"].append(f"note: [{tag(ln)}] L{ln}:{col} {msg}; it is only used as `{name}.<attribute>`, so "
                                f"this may be a false alarm (did you forget an import?)")
        else:
            rep["reject"].append((ln, col, msg, f"undefined:{name}", tag(ln)))
    for m, ln, col in forbidden_imports(tree):
        rep["reject"].append((ln, col, f"import of '{m}' is not allowed (no files, network or processes; "
                                       f"standard library computation only)", f"import:{m}", tag(ln)))
    bad = [i + 1 for i, l in enumerate(src.split("\n")) if '"' in l or "\\" in l]
    if bad:
        rep["notes"].append(f"note: double quotes or backslashes on line(s) {_short_list(bad)}; use single quotes "
                            f"and no backslashes, so the program needs no escaping inside the JSON arguments")
    return rep


def _short_list(xs, n=12):
    return ", ".join(map(str, xs[:n])) + (f" (+{len(xs) - n} more)" if len(xs) > n else "")


def _loc(ln, col):
    if not ln:
        return ""
    return f"L{ln}:{col} " if col else f"L{ln} "


def problem_list(items):
    """Numbered list of (line, col, msg, key, section) problems."""
    return [f"  {i}. [{p[4]}] {_loc(p[0], p[1])}{p[2]}" for i, p in enumerate(items, 1)]


# ---------------------------------------------------------------------------------------------------
# G2C2 change 1: argument channel (lines) and the arm's own lenient parse of raw tool arguments

def lines_to_source(lines):
    """`lines` (array of strings, one program line each; elements may themselves contain newlines) -> source."""
    if isinstance(lines, str):
        return lines
    if not isinstance(lines, list):
        return None
    return "\n".join("" if x is None else str(x) for x in lines)


def _unescape(s):
    """Decode a JSON string body leniently: json first (strict=False), else the common escapes by hand."""
    try:
        v = json.loads('"' + s + '"', strict=False)
        if isinstance(v, str):
            return v
    except ValueError:
        pass
    out, i = [], 0
    table = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "/": "/", "r": "", "'": "'"}
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s) and s[i + 1] in table:
            out.append(table[s[i + 1]])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


_TAIL_RE = r'"\s*{close}\s*(?:,\s*"(?:runs|source|lines|edits|old|new)"\s*:|\}}\s*$)'


def _rest_object(raw, end):
    """Parse what follows the recovered payload (`, "runs": [...]}`) as an object, leniently; {} if impossible."""
    rest = raw[end:].lstrip()
    if rest.startswith(","):
        rest = rest[1:]
    rest = rest.strip()
    if not rest or rest == "}":
        return {}
    try:
        v = json.loads("{" + rest if not rest.startswith("{") else rest, strict=False)
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


def recover_args(raw):
    """Lenient recovery of the `lines` or `source` payload from raw arguments that are not valid JSON: locate the
    key, take the text up to the closing `"]` / `"` before the next key or the final `}`, split `lines` on `","`
    (and raw newlines). Returns a dict, or None if nothing can be recovered."""
    if not isinstance(raw, str):
        return None
    m = re.search(r'"lines"\s*:\s*\[\s*"', raw)
    if m:
        start = m.end()
        t = re.compile(_TAIL_RE.format(close=r"\]")).search(raw, start)
        end = t.start() if t else raw.rfind('"]')
        if end >= start:
            body = raw[start:end]
            parts = re.split(r'"\s*,\s*"', body)
            lines = []
            for p in parts:
                lines += _unescape(p).split("\n")
            close = raw.find("]", end)
            out = _rest_object(raw, close + 1) if close >= 0 else {}
            out = {k: v for k, v in out.items() if k not in ("lines", "source")}
            out["lines"] = lines
            return out
    m = re.search(r'"source"\s*:\s*"', raw)
    if m:
        start = m.end()
        t = re.compile(_TAIL_RE.format(close="")).search(raw, start)
        end = t.start() if t else raw.rfind('"}')
        if end >= start:
            src = _unescape(raw[start:end])
            out = _rest_object(raw, end + 1)
            out = {k: v for k, v in out.items() if k not in ("lines", "source")}
            out["source"] = src
            return out
    return None


_SEC_BODY = re.compile(r'"(lines|source)"\s*:\s*')
_SEC_NAME = re.compile(r'"name"\s*:\s*"([^"\\]{1,80})"')


def recover_sections(raw):
    """Lenient recovery of edit_model's `sections` from raw arguments that are not valid JSON: for each `lines` /
    `source` payload inside the `sections` array, the text up to its closing `"]` / `"` before the end of the
    section object, and the section's `name` (before or after the payload). `runs` after the sections is parsed
    leniently. Returns a dict, or None if nothing can be recovered."""
    if not isinstance(raw, str):
        return None
    k = raw.find('"sections"')
    if k < 0:
        return None
    out, pos, prev_end = [], k, k
    while True:
        m = _SEC_BODY.search(raw, pos)
        if not m:
            break
        kind, start = m.group(1), m.end()
        if kind == "lines":
            if re.match(r"\[\s*\]", raw[start:]):
                end, text, nxt = start, "", raw.find("]", start) + 1
            else:
                mo = re.match(r'\[\s*"', raw[start:])
                t = re.compile(r'"\s*\]\s*(?=[,}])').search(raw, start + (mo.end() if mo else 0)) if mo else None
                if not t:
                    break
                body = raw[start + mo.end():t.start()]
                ls = []
                for part in re.split(r'"\s*,\s*"', body):
                    ls += _unescape(part).split("\n")
                end, text, nxt = t.start(), ls, t.end()
        else:
            if not raw[start:].startswith('"'):
                break
            t = re.compile(r'"\s*(?=,\s*"name"|\})').search(raw, start + 1)
            if not t:
                break
            end, text, nxt = t.start(), _unescape(raw[start + 1:t.start()]), t.end()
        before = list(_SEC_NAME.finditer(raw, prev_end, m.start()))
        after = _SEC_NAME.match(raw, nxt + len(raw[nxt:]) - len(raw[nxt:].lstrip(" \t\r\n,")))
        name = before[-1].group(1) if before else (after.group(1) if after else None)
        if name is not None:
            out.append({"name": name, kind: text})
        pos = prev_end = max(nxt, end + 1)
        if after:
            pos = prev_end = after.end()
    if not out:
        return None
    res = {"sections": out}
    r = re.search(r'"runs"\s*:\s*(\[.*\])\s*\}\s*$', raw[prev_end:], re.S)
    if r:
        try:
            v = json.loads(r.group(1), strict=False)
            if isinstance(v, list):
                res["runs"] = v
        except ValueError:
            pass
    return res


def parse_args(raw, tool=None):
    """Raw-argument pre-parse. Returns (args_or_None, repaired: bool). Valid JSON -> (args, False); otherwise
    json.loads(strict=False) (literal newlines/tabs in strings), then recover_sections (edit_model) or recover_args;
    (None, False) if all fail."""
    if raw is None or not isinstance(raw, str):
        return raw, False
    if not raw.strip():
        return {}, False
    try:
        return json.loads(raw), False
    except ValueError:
        pass
    try:
        v = json.loads(raw, strict=False)
        if isinstance(v, dict):
            return v, True
    except ValueError:
        pass
    v = recover_sections(raw) if tool == "edit_model" else None
    if v is None:
        v = recover_args(raw)
    return (v, True) if v is not None else (None, False)


# ---------------------------------------------------------------------------------------------------
# G3C1 change 2: edit_model applies whole-section replacements

def normalise_section(text, level):
    """Executor-side indentation: tabs -> 4 spaces, a first line indented deeper than the rest is brought to the
    rest's level, textwrap.dedent, then re-indented to `level` (0 top level, 4 methods). Relative indentation inside
    the section is kept, so an edit cannot shift anything outside it. Returns a list of lines."""
    ls = text.expandtabs(4).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while ls and not ls[0].strip():
        ls.pop(0)
    while ls and not ls[-1].strip():
        ls.pop()
    if not ls:
        return []
    rest = [len(l) - len(l.lstrip()) for l in ls[1:] if l.strip()]
    ind0 = len(ls[0]) - len(ls[0].lstrip())
    if rest:
        opener = ls[0].split("#", 1)[0].rstrip().endswith(":")     # a block header must sit left of its body
        if opener and ind0 >= min(rest):
            ls[0] = " " * max(0, min(rest) - 4) + ls[0].lstrip()
        elif not opener and ind0 > min(rest):
            ls[0] = " " * min(rest) + ls[0].lstrip()
    ls = textwrap.dedent("\n".join(ls)).split("\n")
    return [(" " * level + l) if l.strip() else "" for l in ls]


def _first_code(ls):
    """First line that is neither blank, a comment nor a decorator (stripped), or ''."""
    return next((l.strip() for l in ls if l.strip() and not l.strip().startswith(("#", "@"))), "")


def _header_problem(name, kind, ls):
    """None if the replacement text starts as its section must, else a problem text."""
    first = _first_code(ls)
    if kind in ("method", "function"):
        want = name.split(".", 1)[-1]
        m = re.match(r"(?:async\s+)?def\s+(\w+)\s*\(", first)
        if not m or m.group(1) != want:
            return (f"section '{name}': the text must be the whole {'method' if kind == 'method' else 'function'}, "
                    f"starting with `def {want}(` (it starts with: {first[:60]!r})")
    elif kind == "class":
        if not re.match(r"class\s+Model\b", first):
            return f"section 'Model': the text must be the whole class, starting with `class Model` (it starts with: " \
                   f"{first[:60]!r})"
    elif kind == "decl":
        if not re.match(name + r"\s*(?::[^=]*)?=(?!=)", first):
            return f"section '{name}': the text must be the whole `{name} = ...` statement (it starts with: " \
                   f"{first[:60]!r})"
    return None


def _canonical(name, sec):
    """Case-tolerant section name: 'params' -> 'PARAMS', 'model.step' -> 'Model.step', 'helpers' -> 'HELPERS'."""
    up = name.upper()
    if up in DECL_NAMES or up == "HELPERS":
        return up
    if name.lower() == "model":
        return "Model"
    if name.lower().startswith("model."):
        return "Model." + name.split(".", 1)[1]
    return name


def _broken_header(name, kind, lines, repaired):
    """1-based line of the stored program that repair_syntax replaced and that was the header of section `name`
    (`def name(`, `class Model`, `NAME =`), else None. Such a section is not on the repaired tree, so adding it as
    new would leave the broken lines in place (QA-0285)."""
    if kind == "method":
        pat = r"\s+(?:async\s+)?def\s+" + re.escape(name[6:]) + r"\b"
    elif kind == "function":
        pat = r"(?:async\s+)?def\s+" + re.escape(name) + r"\b"
    elif kind == "class":
        pat = r"class\s+Model\b"
    elif kind == "decl":
        pat = re.escape(name) + r"\s*[:=]"
    else:
        return None
    for ln in sorted(repaired):
        if 1 <= ln <= len(lines) and re.match(pat, lines[ln - 1]):
            return ln
    return None


def apply_sections(src, entries):
    """Replace whole named sections of `src` in one call: entries = [{name, lines | source}]. Names: PARAMS, RULES,
    HORIZON (a missing one is added before the Model class), Model, Model.<method> (a new name is appended to the
    class), HELPERS (every top-level statement that is not a declaration or Model, replaced by one block before
    PARAMS) or a top-level function name (a new name, whose text defines it, is added to HELPERS). Sections are
    located on the AST of the syntax-repaired program, so a broken method can be replaced by name. All-or-nothing:
    returns (new_src, names, None) or (None, None, [problems])."""
    entries = _maybe_json(entries)
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list) or not entries:
        return None, None, ["`sections` must be a non-empty list of {\"name\": ..., \"lines\": [...]} objects"]
    errs, fixed, tree = repair_syntax(src)
    if tree is None:
        return None, None, ["the stored program cannot be split into sections (its syntax errors could not be "
                            "located); resubmit the whole program with submit_model"]
    lines = src.split("\n")
    sec = locate_sections(fixed, tree)
    avail = section_names(sec)
    repaired = {ln for ln, _, _ in errs if ln}

    def unlocatable(nm, kd):
        ln = _broken_header(nm, kd, lines, repaired)
        if ln is None:
            return None
        return (f"section '{nm}' cannot be located: its header (line {ln}) has a syntax error; locatable sections: "
                f"{', '.join(avail) or '(none)'}; resubmit the whole program with submit_model")
    probs, ops, names, ranges = [], [], [], []
    for i, e in enumerate(entries, 1):
        e = _maybe_json(e)
        if not isinstance(e, dict):
            probs.append(f"entry {i}: must be an object {{\"name\": ..., \"lines\": [...]}}")
            continue
        raw_name = e.get("name", e.get("section"))
        if not isinstance(raw_name, str) or not raw_name.strip():
            probs.append(f"entry {i}: needs `name` (one of: {', '.join(avail)})")
            continue
        name = _canonical(raw_name.strip(), sec)
        body = _maybe_json(e["lines"]) if "lines" in e else e.get("source", e.get("code"))
        text = lines_to_source(body) if isinstance(body, (list, str)) else None
        if text is None:
            probs.append(f"section '{name}': needs `lines` (array of strings) or `source` (string)")
            continue
        if name in names:
            probs.append(f"section '{name}' is given twice (send it once)")
            continue
        if name in DECL_NAMES:
            kind, level = "decl", 0
        elif name == "Model":
            kind, level = "class", 0
        elif name.startswith("Model."):
            kind, level = "method", 4
            if not name[6:].isidentifier():
                probs.append(f"section '{name}': '{name[6:]}' is not a method name")
                continue
            if not sec["model"]:
                probs.append(unlocatable("Model", "class") or
                             f"section '{name}': the program has no class Model to add it to (send the whole "
                             f"'Model' section)")
                continue
        elif name == "HELPERS":
            kind, level = "helpers", 0
        elif name in sec["methods"] and name not in sec["functions"]:
            probs.append(f"unknown section '{name}' (did you mean 'Model.{name}'? a method is named Model.<method>)")
            continue
        elif name in sec["functions"] or (name.isidentifier() and re.match(
                r"(?:async\s+)?def\s+" + re.escape(name) + r"\s*\(", _first_code(text.split("\n")))):
            kind, level = "function", 0
        else:
            close = difflib.get_close_matches(name, avail, n=3, cutoff=0.5)
            if name in sec["methods"]:
                close = ["Model." + name] + [c for c in close if c != "Model." + name]
            probs.append(f"unknown section '{name}'" + (f" (did you mean {' or '.join(map(repr, close))}?)"
                                                        if close else "")
                         + f"; available: {', '.join(avail) or '(none)'}. A new helper function is added by "
                           f"naming it and sending its whole `def`; a new method by `Model.<name>`")
            continue
        if name not in sec["spans"]:
            up = unlocatable(name, kind)
            if up:
                probs.append(up)
                continue
        new = normalise_section(text, level)
        if not new and kind != "helpers":
            probs.append(f"section '{name}': the text is empty (a section cannot be deleted; send its whole new text)")
            continue
        hp = _header_problem(name, kind, new)
        if hp:
            probs.append(hp)
            continue
        names.append(name)
        if kind == "helpers":
            for a, b in sec["helpers"]:
                ops.append((a, 1, b, []))
                ranges.append((a, b, name))
            if new:
                ops.append((sec["anchor"], 0, None, new + [""]))
        elif name in sec["spans"]:
            a, b = sec["spans"][name]
            ops.append((a, 1, b, new))
            ranges.append((a, b, name))
        elif kind == "method":
            ops.append((sec["model"][1] + 1, 0, None, [""] + new))
        elif kind == "class":
            ops.append((len(lines) + 1, 0, None, [""] + new))
        elif kind == "decl":
            ops.append((sec["decl_anchor"], 0, None, new + [""]))
        else:
            ops.append((sec["anchor"], 0, None, new + ["", ""]))
    if "Model" in names and any(n.startswith("Model.") for n in names):
        probs.append("send either the whole 'Model' section or some of its methods, not both")
    for x, (a1, b1, n1) in enumerate(ranges):
        for a2, b2, n2 in ranges[x + 1:]:
            if n1 != n2 and a1 <= b2 and a2 <= b1:
                probs.append(f"sections '{n1}' and '{n2}' overlap (send only one of them)")
    if probs:
        return None, None, probs
    seq = sorted(enumerate(ops), key=lambda x: (x[1][0], x[1][1], x[0]), reverse=True)
    for _, (a, kind, b, new) in seq:
        if kind == 1:
            lines[a - 1:b] = new
        else:
            lines[a - 1:a - 1] = new
    return "\n".join(lines), names, None


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


class G3C1Arm(Arm):
    name = "G3C1"
    buildable = True
    executable = True
    guide_file = "g3c1_guide.md"
    mutating_tools = ("submit_model", "edit_model")
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
            "no_effect_expected": {"type": "string", "description": "reason why this what-if should change nothing "
                                   "(only when you are sure; otherwise a what-if that changes no observation is "
                                   "rejected by submit_model / edit_model)"},
        }

    def _runs_prop(self):
        return {"type": "array", "items": {"type": "object", "properties": self._run_schema()},
                "description": f"up to {MAX_RUNS} named what-if runs, each compared with the baseline"}

    @staticmethod
    def _sections_prop():
        return {"type": "array", "description": "sections to replace in one call (all or none); each entry is one "
                "whole section", "items": {"type": "object", "properties": {
                    "name": {"type": "string", "description": "PARAMS, RULES, HORIZON, Model, Model.<method> (a new "
                             "method name adds the method), HELPERS (imports, constants, helper functions) or a helper "
                             "function's name"},
                    "lines": {"type": "array", "items": {"type": "string"}, "description": "the whole new section, "
                              "one string per line; indentation is normalised (relative indentation is kept)"},
                    "source": {"type": "string", "description": "alternatively the section as one string"}}}}

    def tools(self):
        return [
            _tool("submit_model", "Store the program, check it (every problem is listed at once with its line), and "
                  "if it is accepted run the baseline (HORIZON steps, 100 seeds) plus each entry of `runs`.",
                  {"lines": {"type": "array", "items": {"type": "string"}, "description": "the whole program, one "
                             "string per program line with its indentation; use single quotes and no backslashes "
                             "in the code"},
                   "source": {"type": "string", "description": "alternatively the whole program as one string"},
                   "runs": self._runs_prop()}),
            _tool("edit_model", "Replace whole named sections of the stored program in one call (each problem is "
                  "tagged with its section, e.g. [Model.step]), then check it and re-run the baseline and `runs`.",
                  {"sections": self._sections_prop(), "runs": self._runs_prop()}),
            _tool("run_scenario", "Run the stored program: either one run's fields or `runs` (up to 4). Each run "
                  "with overrides or a what_if_patch is compared with the baseline under common random numbers "
                  "(paired differences).", {**self._run_schema(), "runs": self._runs_prop()}),
            _tool("show_model", "Show the program with line numbers, its section names and the declared "
                  "PARAMS/RULES table (or one `section` only).",
                  {"section": {"type": "string", "description": "optional: show only this section"}}),
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
            if not isinstance(src, str) and isinstance(rep.get("lines"), list):
                src = lines_to_source(rep["lines"])
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
        return {"source": src, "params": info["params"], "rules": info["rules"], "horizon": info["horizon"],
                "lines": src.splitlines()}

    def render(self, state):
        src = self._src(state)
        if not src:
            return "(no model code yet)"
        return "\n".join(f"{i + 1:4d}| {l}" for i, l in enumerate(src.splitlines()))[:60000]

    def _decl_table(self, info):
        out = []
        prov = info.get("prov") or {}
        for name, key in (("PARAMS", "params"), ("RULES", "rules")):
            d = info[key]
            if d is None:
                out.append(f"{name}: (not declared)")
                continue
            out.append(f"{name} ({len(d)}):")
            for k, v in d.items():
                p = prov.get(k)
                tag = ""
                if p is None or p["kind"] in (None, "assumed"):
                    tag += "   [assumed" + (f": {p['reason'][:80]}" if p and p["kind"] and p["reason"] else
                                            ("" if p and p["kind"] else ": no provenance comment")) + "]"
                if k in info.get("unread", ()):
                    tag += (f"   [declared unused: {p['unused'][:80]}]" if p and p.get("unused") is not None
                            else "   [not read by the program: overriding it has no effect]")
                out.append(f"  {k} = {json.dumps(v)}" + tag)
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
        state.log.append((name, {k: (f"<{len(str(v))} chars>" if k in ("source", "lines") else
                                     [x.get("name") if isinstance(x, dict) else "?" for x in v]
                                     if k == "sections" and isinstance(v, list) else v)
                                 for k, v in args.items()}
                          if isinstance(args, dict) else args, out[:200]))
        return out

    def _t_show_model(self, state, args):
        src = self._src(state)
        if not src:
            return EMPTY_SHOW                   # G3C1 change 4: an instruction, not an error or an empty listing
        errs, fixed, tree = repair_syntax(src)
        names = section_names(locate_sections(fixed, tree)) if tree is not None else []
        want = args.get("section") if isinstance(args, dict) else None
        if isinstance(want, str) and want.strip():
            return self._show_section(src, fixed, tree, want.strip(), names)
        info = analyse(src)
        if not info["parses"]:              # declarations of the syntax-repaired program (line numbers kept)
            info = analyse(fixed)
        decl = "\n".join(["", "SECTIONS: " + (", ".join(names) if names else "(cannot be located: the syntax "
                                              "errors could not be repaired)"), "DECLARED"]
                         + self._decl_table(info))[:MAX_OUT // 2]
        code = self.render(state)
        room = MAX_OUT - len(decl) - 120
        if len(code) > room:
            cut = code[:room].rsplit("\n", 1)[0]
            code = cut + f"\n... [program truncated after line {cut.count(chr(10)) + 1} of {len(src.splitlines())}]"
        return code + "\n" + decl

    def _show_section(self, src, fixed, tree, want, names):
        if tree is None:
            return (f"{ERROR_PREFIX}: the sections cannot be located (the syntax errors could not be repaired); "
                    f"call show_model without `section`, or resubmit the whole program with submit_model")
        sec = locate_sections(fixed, tree)
        name = _canonical(want, sec)
        lines = src.split("\n")
        if name == "HELPERS":
            spans = sec["helpers"]
        elif name in sec["spans"]:
            spans = [sec["spans"][name]]
        else:
            close = difflib.get_close_matches(name, names, n=3, cutoff=0.5)
            return (f"{ERROR_PREFIX}: unknown section '{name}'" + (f" (did you mean {' or '.join(map(repr, close))}?)"
                                                                     if close else "")
                    + f"; available: {', '.join(names) or '(none)'}")
        out = [f"SECTION {name}:"]
        for a, b in spans:
            out += [f"{i:4d}| {lines[i - 1]}" for i in range(a, b + 1) if i <= len(lines)]
        return self._bounded(out)

    def _store_and_run(self, state, src, verb, args):
        if len(src) > MAX_CODE:
            return f"{ERROR_PREFIX}: program too long ({len(src)} chars, max {MAX_CODE})"
        runs = args.get("runs")
        runs = _maybe_json(runs) if runs is not None else []
        if isinstance(runs, dict):
            runs = [runs]
        bad_runs = not isinstance(runs, list)
        runs = [] if bad_runs else list(runs)
        state.rep["source"] = src          # stored as a draft even when rejected, so sections can be replaced
        rep = static_report(src)
        info = rep["info"] or analyse(src)
        head = [f"model {verb} ({len(src.splitlines())} lines); PARAMS: "
                f"{len(info['params']) if info['params'] is not None else 'none'}, RULES: "
                f"{len(info['rules']) if info['rules'] is not None else 'none'}, HORIZON: "
                f"{info['horizon'] if info['horizon'] is not None else 'none'}"]
        notes = [f"  {n}" for n in rep["notes"]]
        blocking = rep["fatal"] + rep["reject"]
        if blocking:
            # G3C1 change 1: only fatal problems and rejections stop the run; warnings are listed with them
            self._streak(state, None)
            what = ("the program is stored but cannot run" if rep["fatal"] else
                    "submission rejected; the program is stored as a draft but was NOT run")
            w = f" and {len(rep['warning'])} warning(s)" if rep["warning"] else ""
            return self._bounded([f"{ERROR_PREFIX}: {what}; {len(blocking)} problem(s){w}"] + head
                                 + self._numbered(src, blocking + rep["warning"])
                                 + [f"  hint: {FIX_HINT}"] + notes)
        if rep["warning"]:
            ws = rep["warning"]
            head.append("WARNINGS (the program runs; wire an entry in only if a what-if needs it):")
            head += self._numbered(src, ws[:MAX_WARNINGS_SHOWN], code=False)
            if len(ws) > MAX_WARNINGS_SHOWN:
                head.append(f"  (+{len(ws) - MAX_WARNINGS_SHOWN} more warnings; see show_model)")
        head += notes
        if bad_runs:
            head.append(f"{ERROR_PREFIX}: runs must be a list of run objects; only the baseline was run")
        return self._run_many(state, [{"name": "baseline"}] + runs, head, explicit_baseline=True, reject_noop=True)

    @staticmethod
    def _streak(state, keys):
        """Escape valve bookkeeping (no-op what-ifs only): count of identical no-op sets in a row (None resets)."""
        prev = state.__dict__.get("_rej") or {"keys": None, "n": 0}
        if keys is None:
            state.__dict__["_rej"] = {"keys": None, "n": 0}
            return 0
        n = prev["n"] + 1 if prev["keys"] == keys else 1
        state.__dict__["_rej"] = {"keys": keys, "n": n}
        return n

    @staticmethod
    def _numbered(src, items, code=True):
        """Numbered problem list; each item carries its section tag, and each item with a line also shows that
        program line."""
        lines = src.split("\n")
        out = []
        for i, p in enumerate(items, 1):
            out.append(f"  {i}. [{p[4]}] {_loc(p[0], p[1])}{p[2]}")
            if code and p[0] and 1 <= p[0] <= len(lines):
                out.append(f"     {p[0]:4d}| {lines[p[0] - 1].strip()[:160]}")
        return out

    @staticmethod
    def _bounded(lines):
        text = "\n".join(lines)
        if len(text) > MAX_OUT:
            text = text[:MAX_OUT - 80].rsplit("\n", 1)[0] + "\n...[problem list cut at the output limit]"
        return text

    def _assumed_block(self, info):
        """G2C2 change 4 (kept): the entries whose provenance is `assumed` (or missing), with value and reason."""
        ents = assumed_entries(info)
        if not ents:
            return []
        out = ["ASSUMED (not stated in the description):"]
        used = len(out[0])
        for i, (k, v, why) in enumerate(ents):
            ln = f"  {k} = {_short(v, 80)} ({why[:120]})"
            if used + len(ln) + 1 > MAX_ASSUMED:
                out.append(f"  (+{len(ents) - i} more; see show_model)")
                break
            out.append(ln)
            used += len(ln) + 1
        return out

    def _t_submit_model(self, state, args):
        src, origin = None, None
        lines = args.get("lines")
        if isinstance(lines, str):
            lines = _maybe_json(lines)
        if isinstance(lines, (list, str)):
            src, origin = lines_to_source(lines), "lines"
        if not isinstance(src, str) or not src.strip():
            src, origin = args.get("source"), "source"
        if not isinstance(src, str) or not src.strip():
            src, origin = extract_block(state.__dict__.get("_last_text", "")), "fenced block"
        if not isinstance(src, str) or not src.strip():
            return f"{ERROR_PREFIX}: no program found (hint: {LINES_HINT})"
        state.__dict__["_last_origin"] = origin
        return self._store_and_run(state, src, f"submitted from {origin}", args)

    def _t_edit_model(self, state, args):
        secs = args.get("sections")
        if secs is None and isinstance(args.get("name"), str) and ("lines" in args or "source" in args):
            secs = [{k: args[k] for k in ("name", "lines", "source") if k in args}]
        if secs is None:
            if any(k in args for k in ("edits", "line", "set", "after", "insert", "old", "new")):
                return f"{ERROR_PREFIX}: edit_model takes `sections`, not edits (hint: {EDIT_HINT})"
            return f"{ERROR_PREFIX}: edit_model needs `sections` (hint: {EDIT_HINT})"
        src = self._src(state)
        if not src.strip():
            return f"{ERROR_PREFIX}: no program yet (hint: call submit_model with your whole draft in `lines`)"
        new_src, names, probs = apply_sections(src, secs)
        if probs:
            return (f"{ERROR_PREFIX}: no section was replaced (the stored program is unchanged):\n"
                    + "\n".join(f"  {p}" for p in probs) + "\n  hint: show_model lists the section names")
        return self._store_and_run(state, new_src, "edited (" + ", ".join(names) + ")", args)

    def _t_patch_model(self, state, args):
        """Removed in G3C1 (spec change 2); kept only to answer a call by the old name with the right hint."""
        return f"{ERROR_PREFIX}: patch_model does not exist (hint: {EDIT_HINT})"

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
                "changed": bool(ov) or patch is not None, "compare": compare, "stored": src == stored,
                "no_effect": str(r["no_effect_expected"]) if r.get("no_effect_expected") else None,
                "same_as_declared": bool(ov) and all(
                    x in (params or {}) and (params or {})[x] == v or x in (rules or {}) and (rules or {})[x] == v
                    for x, v in ov.items())}
        return spec, None, notes

    def _run_many(self, state, raw_runs, head, explicit_baseline, reject_noop=False):
        stored = self._src(state)
        lines = list(head) + self._assumed_block(analyse(stored))
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
        ok_base = [s for s in specs if not s["changed"] and s["stored"]]
        if ok_base:
            r = res["runs"][specs.index(ok_base[0])]
            self._record_run(state, ok=not r["load_err"] and not r["failed"] and r["n_full"] == ok_base[0]["n"]
                             and not res["cut"])
        if reject_noop:
            specs, res = self._noop_rejections(state, specs, res, lines, problems)
        return self._report(specs, res, lines, problems)

    def _noop_rejections(self, state, specs, res, lines, problems):
        """The only dynamic rejection (kept from G2C2): a requested override run (no what_if_patch, no
        `no_effect_expected`) that leaves every observed key identical on every seed and step is rejected: the
        baseline stays, its results are withheld. After two identical no-op sets in a row the next one runs as
        UNRESOLVED. G3C1: if an overridden key is also declared-but-unread, the message says so."""
        by = {s["name"]: (s, r) for s, r in zip(specs, res["runs"])}
        noop = []
        for s, r in zip(specs, res["runs"]):
            if not s["overrides"] or s["patched"] or not s.get("ref") or s["ref"] not in by:
                continue
            if s.get("no_effect"):
                lines.append(f"note: run '{s['name']}': no effect expected ({s['no_effect'][:160]})")
                continue
            _, b = by[s["ref"]]
            common = [i for i in r["seeds"] if i in b["seeds"]]
            if r["load_err"] or r["failed"] or b["load_err"] or b["failed"] or not common:
                continue
            if all(r["seeds"][i].get("h") is not None and r["seeds"][i]["h"] == b["seeds"][i].get("h")
                   for i in common):
                noop.append(s)
        if not noop:
            self._streak(state, None)
            return specs, res
        unread = set(analyse(self._src(state))["unread"])
        keys = frozenset(f"noop:{s['name']}:{','.join(sorted(map(str, s['overrides'])))}" for s in noop)
        if self._streak(state, keys) > 2:
            self._streak(state, None)
            for s in noop:
                lines.append(f"UNRESOLVED: " + self._noop_msg(s, unread)
                             + " (shown anyway after two identical rejections)")
            return specs, res
        for s in noop:
            problems.append(self._noop_msg(s, unread) + "; its results are withheld (hint: wire "
                            + ", ".join(map(str, s["overrides"])) + " into the mechanism it describes, or set "
                            "\"no_effect_expected\": \"<reason>\" on the run if no effect is the true answer)")
        keep = [k for k, s in enumerate(specs) if s not in noop]
        res = {**res, "runs": [res["runs"][k] for k in keep]}
        return [specs[k] for k in keep], res

    @staticmethod
    def _noop_msg(s, unread=()):
        ks = ", ".join(map(str, s["overrides"]))
        dead = [k for k in s["overrides"] if k in unread]
        return (f"no-op what-if {s['name']}: overriding {ks} changed no observation; the program does not read "
                f"{ks} where it affects the outputs"
                + (" (note: the override equals the declared value)" if s.get("same_as_declared") else "")
                + "".join(f"\n  override of '{k}' changed nothing: '{k}' is declared but never read" for k in dead))

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
                x["seeds"][b["i"]] = {"final": b.get("final"), "ev": b.get("ev") or {}, "n": b.get("n", 0),
                                      "h": b.get("h")}
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

    # ------------------------------------------------------------------ episode (argument channel)
    def run_episode(self, ep, state, messages):
        """Generic tool_loop plus: system-prompt parity, the raw-argument pre-parse (strict JSON, strict=False,
        lenient `lines`/`source` recovery; repairs are noted in the result and the trace), the untaught fenced
        ```model fallback with implicit submit, and a budget status line on every executor result.
        Returns (final_text, messages, trace)."""
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
                # G2C2 change 1 (kept): the arm's own raw-argument pre-parse before dispatch (section-aware for
                # edit_model)
                fn = tc.get("function", {}) or {}
                parsed, repaired = parse_args(fn.get("arguments", "{}"), fn.get("name"))
                if parsed is None or not isinstance(parsed, dict):
                    name, args = fn.get("name", ""), fn.get("arguments")
                    res = (f"{ERROR_PREFIX}: tool-call limit ({MAX_TOOL_CALLS}) reached; give your FINAL ANSWER."
                           if n_tools >= MAX_TOOL_CALLS else
                           f"{ERROR_PREFIX}: the arguments of {name} are not a valid JSON object and could not be "
                           f"repaired (hint: {LINES_HINT})")
                else:
                    tc2 = {**tc, "function": {**fn, "arguments": parsed}} if repaired else tc
                    name, args, res = execute_tool_call(self, state, tc2, offered, n_tools, MAX_TOOL_CALLS)
                    if name == "patch_model" and name not in offered and res.startswith(f"{ERROR_PREFIX}: unknown tool"):
                        res += f" (hint: {EDIT_HINT})"
                    if repaired:
                        res = REPAIR_NOTE + "\n" + res
                n_tools += 1
                submitted = submitted or name == "submit_model"
                res = res + budget()
                entry = {"turn": len(ep.calls), "tool": name, "args": args, "result": res[:3000],
                         "error": res.startswith(ERROR_PREFIX) or (repaired and res[len(REPAIR_NOTE) + 1:]
                                                                   .startswith(ERROR_PREFIX))}
                if repaired:
                    entry["repaired"] = True
                trace.append(entry)
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
    with tempfile.TemporaryDirectory(prefix="rpp_g3c1_") as d:
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


G3C1 = G3C1Arm
