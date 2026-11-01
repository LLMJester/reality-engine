"""G1C4: stochastic event net, a typed guarded-transition core in continuous time (runs/discovery/gen1/specs/G1C4.md).

Representation: one text document (fenced block labelled `net`) with one declaration per line, 8 primitive kinds
(set, param, var, event, flow, invariant, observe, whatif) and settings (unit, horizon, grid, seeds). Expressions use
Python expression syntax over a whitelist; they are compiled once from the AST into closures (no code execution).

Semantics (spec §2): a generalised semi-Markov process with continuous flows (PDMP extended with general clocks).
- Rate events are simulated exactly. Without flows this uses the modified next-reaction method (Anderson 2007):
  statistically identical to Gillespie's direct method, with propensities recomputed after every state change, but
  each event instance draws its firing times from its OWN random stream, which is what common random numbers (§2d)
  need. With flows, hazards are piecewise constant over the RK4 step h = min(grid/4, 0.1 units).
- `after` clocks start when the guard becomes true and are discarded when it becomes false (enabling memory reset);
  `every` / `at` schedules; `immediate` events are resolved by the vanishing closure (highest priority first, ties
  uniformly at random, livelock reported after 10,000 firings in one instant). Simultaneous timed instants are
  ordered by priority, then declaration order.
- Poisson tau-leaping (tau = h) replaces exact rate firing when the expected number of rate firings in a run exceeds
  200,000; firings inside a leap are applied one by one with guards re-checked. Reported when used.
- `mode: fluid` replaces rate events with constant increments by their mean-field ODE (Kurtz); `mode: both` reports
  the stochastic-minus-fluid discrepancy.
- P-invariants: exact rational left null space of the incidence matrix of constant-increment count/level variables.
- `check_invariants`: BFS of the untimed transition system for enumerable models, otherwise a sampled check.

The executor is pure Python and runs in-process (no subprocess, no files read at run time), like A2 and A8.

Build notes (choices where the spec leaves room; none changes the representation or the scoring):
- Exact rate firing uses the modified next-reaction method instead of Gillespie's direct method (same law; it gives
  each event instance its own random stream, which the common-random-number pairing of §2(d) requires).
- Default grid: max(1, horizon/100) for horizons >= 10, horizon/10 below (the spec's "at least 1 unit" would leave a
  single interval for very short horizons).
- Guards, rates, clock enablings and invariants are re-evaluated only when a var they read has changed (or they read
  `now` / fired()); this is exact (tested against full re-evaluation) and keeps large models within the 60 s budget.
- Magic-number warnings skip unit-conversion constants (0, 1, 60, 24, 100, 1000, 3600) and priority expressions.
- The t = 0 observation is recorded after events scheduled at t = 0 (initialisations) have fired.
"""
import ast
import difflib
import hashlib
import itertools
import json
import math
import random
import re
import time
from collections import Counter, deque
from fractions import Fraction
from pathlib import Path

from tournament.arms.a4_prob import _quant
from tournament.arms.base import ERROR_PREFIX, Arm, ArmState

ROOT = Path(__file__).resolve().parents[2]
# Runtime execution dependencies (D-060): none. The executor runs in-process and reads no files at run time.
RUNTIME_DEPS = []

DEFAULT_SEEDS = 100
MAX_SEEDS = 200
MIN_SEEDS = 20
MAX_RUNS = 4
MAX_WHATIFS = 4
MAX_OUT = 7800                 # executor text; + the budget line stays within 8,000 characters
MAX_DOC = 40000
TIME_BUDGET = 60.0             # seconds of wall clock per tool call, all runs together
MAX_HORIZON = 1e6
MAX_EVENTS = 5_000_000
LEAP_THRESHOLD = 200_000
MAX_CLOSURE = 10_000
MAX_FIFO = 10 ** 6
MAX_GRID_POINTS = 2001
MAX_REC_KEYS = 120
SERIES_POINTS = 60
SERIES_KEYS = 8
BFS_MAX_STATES = 200_000
DEP_TRACKING = True            # re-evaluate guards/rates/invariants only when a var they read changed (exact)
UNIT_CONSTANTS = (0.0, 1.0, 60.0, 24.0, 100.0, 1000.0, 3600.0)   # not flagged as magic numbers (unit conversions)
BUDGET_LINE = "[budget: {k} of {n} responses used; your last response must be the FINAL ANSWER without tool calls]"

KINDS = ("set", "param", "var", "event", "flow", "invariant", "observe", "whatif")
SETTINGS = ("unit", "horizon", "grid", "seeds")
TIMINGS = ("rate", "after", "every", "at", "immediate")
MATH_FUNCS = ("min", "max", "abs", "round", "floor", "ceil", "sqrt", "exp", "log", "clip")
AGG_FUNCS = ("sum", "any", "all", "count")
FIFO_FUNCS = ("len", "oldest_age", "mean_age")
DRAW_FUNCS = ("bernoulli", "binomial", "poisson", "normal", "uniform", "choice")
PROC_FUNCS = ("push", "pop", "drop_older")
DIST_FUNCS = ("exp", "uniform", "normal", "lognormal", "erlang", "choice")
ALL_FUNCS = MATH_FUNCS + AGG_FUNCS + FIFO_FUNCS + DRAW_FUNCS + PROC_FUNCS + ("fired",)
CONSTS = {"true": True, "false": False, "True": True, "False": False, "inf": math.inf, "pi": math.pi}
ASSIGN_OPS = (":=", "+=", "-=", "*=", "/=", "=")


class NetError(Exception):
    """A construction error in one line (the line becomes a hole)."""


class EvalError(Exception):
    """A run-time evaluation error."""


class Livelock(Exception):
    pass


class Budget(Exception):
    pass


def _dym(name, pool):
    close = difflib.get_close_matches(str(name), [str(p) for p in pool], n=2, cutoff=0.6)
    return f" (did you mean {' or '.join(repr(c) for c in close)}?)" if close else ""


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


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


# ================================================================================================ text helpers
def _split_comment(line):
    q = None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
            continue
        if ch in "\"'":
            q = ch
        elif ch == "#":
            return line[:i].rstrip(), line[i + 1:].strip()
    return line.rstrip(), ""


def _split_top(s, sep=","):
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur))
    return parts


def _find_header_colon(s):
    depth = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and depth == 0 and s[i + 1:i + 2] != "=":
            return i
    return -1


def _wordch(c):
    return c.isalnum() or c == "_"


def _kw_split(s, kws):
    """Split `s` at top-level whole-word keywords. -> (head, {kw: text})."""
    pos, depth = [], 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
            continue
        if ch in ")]}":
            depth -= 1
            continue
        if depth == 0 and (i == 0 or not _wordch(s[i - 1])):
            for kw in kws:
                j = i + len(kw)
                if s.startswith(kw, i) and (j == len(s) or not _wordch(s[j])):
                    pos.append((i, kw))
                    break
    head = s[:pos[0][0]] if pos else s
    parts = {}
    for n, (i, kw) in enumerate(pos):
        end = pos[n + 1][0] if n + 1 < len(pos) else len(s)
        if kw in parts:
            raise NetError(f"'{kw}' given twice")
        parts[kw] = s[i + len(kw):end].strip()
    return head.strip(), parts


def _find_assign(s):
    depth = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            two = s[i:i + 2]
            if two in (":=", "+=", "-=", "*=", "/="):
                return i, two
            if ch == "=" and s[i + 1:i + 2] != "=" and (i == 0 or s[i - 1] not in "=<>!"):
                return i, "="
    return -1, None


def extract_block(text, label="net"):
    """Body of the LAST fenced block labelled exactly `label` (``` or ~~~; CRLF; a missing closing fence runs to the
    end of the message). None if there is none."""
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
        if j >= len(lines):
            while body and not body[-1].strip():
                body.pop()
        last = "\n".join(body)
        i = j + 1
    return last


def _parse_atoms(text):
    atoms = []
    for part in _split_top(text):
        p = part.strip()
        if not p:
            continue
        m = re.match(r"^(-?\d+)\s*\.\.\s*(-?\d+)$", p)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b - a > 10000:
                raise NetError(f"range {p} is too large (max 10,000 atoms)")
            atoms += list(range(a, b + 1))
        elif re.match(r"^-?\d+$", p):
            atoms.append(int(p))
        elif re.match(r"^[A-Za-z_]\w*$", p):
            atoms.append(p)
        else:
            raise NetError(f"set element '{p}' must be a name, an integer or a range like 1..5")
    if len(set(atoms)) != len(atoms):
        raise NetError("a set lists the same element twice")
    return atoms


def _keystr(name, key):
    return f"{name}[{','.join(str(k) for k in key)}]" if key is not None else name


# ================================================================================================ FIFO with ages
class Fifo:
    """First-in-first-out multiset of items stamped with their entry time (run-length encoded)."""
    __slots__ = ("runs", "n", "tsum", "cap", "over")

    def __init__(self, cap=MAX_FIFO):
        self.runs = deque()
        self.n = 0
        self.tsum = 0.0
        self.cap = cap
        self.over = 0

    def push(self, t, k):
        k = int(round(k))
        if k <= 0:
            return 0
        room = self.cap - self.n
        if k > room:
            self.over += k - max(room, 0)
            k = max(room, 0)
            if k == 0:
                return 0
        if self.runs and self.runs[-1][0] == t:
            self.runs[-1][1] += k
        else:
            self.runs.append([t, k])
        self.n += k
        self.tsum += t * k
        return k

    def pop(self, k=1):
        k = min(int(round(k)), self.n)
        removed = 0
        while removed < k:
            r = self.runs[0]
            take = min(r[1], k - removed)
            r[1] -= take
            removed += take
            self.tsum -= r[0] * take
            if r[1] == 0:
                self.runs.popleft()
        self.n -= removed
        if self.n == 0:
            self.tsum = 0.0
        return removed

    def drop_older(self, t, age):
        removed = 0
        while self.runs and t - self.runs[0][0] > age + 1e-12:
            r = self.runs.popleft()
            removed += r[1]
            self.tsum -= r[0] * r[1]
        self.n -= removed
        if self.n == 0:
            self.tsum = 0.0
        return removed

    def oldest_age(self, t):
        return t - self.runs[0][0] if self.n else 0.0

    def mean_age(self, t):
        return t - self.tsum / self.n if self.n else 0.0

    def __len__(self):
        return self.n

    def clone(self):
        f = Fifo(self.cap)
        f.runs = deque([list(r) for r in self.runs])
        f.n, f.tsum, f.over = self.n, self.tsum, self.over
        return f

    def key(self):
        return tuple((t, k) for t, k in self.runs)


# ================================================================================================ random draws
def _poisson(rng, m):
    if m <= 0:
        return 0
    if m < 30:
        lim, k, p = math.exp(-m), 0, 1.0
        while True:
            p *= rng.random()
            if p <= lim:
                return k
            k += 1
    slam, loglam = math.sqrt(m), math.log(m)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2)
    while True:                                   # PTRS (Hoermann 1993)
        u = rng.random() - 0.5
        v = rng.random()
        us = 0.5 - abs(u)
        k = math.floor((2 * a / us + b) * u + m + 0.43)
        if us >= 0.07 and v <= vr:
            return k
        if k < 0 or (us < 0.013 and v > us):
            continue
        if math.log(v) + math.log(invalpha) - math.log(a / (us * us) + b) <= -m + k * loglam - math.lgamma(k + 1):
            return k


def _num(x, what):
    if isinstance(x, bool):
        return int(x)
    if not _isnum(x):
        raise EvalError(f"{what} needs a number, got {x!r}")
    return x


def _draw(fn, R, args):
    fl = R.fluid
    if fn == "bernoulli":
        p = min(1.0, max(0.0, float(_num(args[0], "bernoulli"))))
        return p if fl else (R.rng.random() < p)
    if fn == "binomial":
        n, p = int(round(_num(args[0], "binomial n"))), min(1.0, max(0.0, float(_num(args[1], "binomial p"))))
        if n <= 0:
            return 0
        return n * p if fl else R.rng.binomialvariate(n, p)
    if fn == "poisson":
        m = float(_num(args[0], "poisson"))
        return max(0.0, m) if fl else _poisson(R.rng, m)
    if fn == "normal":
        m, s = float(_num(args[0], "normal")), float(_num(args[1], "normal"))
        return m if fl or s <= 0 else R.rng.gauss(m, s)
    if fn == "uniform":
        a, b = float(_num(args[0], "uniform")), float(_num(args[1], "uniform"))
        return (a + b) / 2 if fl else R.rng.uniform(a, b)
    if fn == "choice":
        vals, w = args[0], (args[1] if len(args) > 1 else None)
        if not vals:
            raise EvalError("choice needs a non-empty list")
        if w is not None and (len(w) != len(vals) or sum(w) <= 0 or any(x < 0 for x in w)):
            raise EvalError("choice weights must be as many as the values, non-negative, not all zero")
        if fl:
            ws = w or [1] * len(vals)
            if all(_isnum(v) or isinstance(v, bool) for v in vals):
                return sum(float(v) * x for v, x in zip(vals, ws)) / sum(ws)
            return vals[max(range(len(vals)), key=lambda i: ws[i])]
        return R.rng.choices(vals, weights=w)[0]
    raise EvalError(f"unknown draw {fn}")


# ================================================================================================ runtime env
class Env:
    __slots__ = ("state", "params", "sets", "t", "loc", "rng", "fired", "fluid", "rec", "coerce", "clips",
                 "rounded", "fifo_over")

    def __init__(self):
        self.state, self.params, self.sets, self.loc = {}, {}, {}, {}
        self.t, self.rng, self.fired, self.fluid, self.rec = 0.0, None, Counter(), False, None
        self.coerce, self.clips, self.rounded, self.fifo_over = {}, Counter(), Counter(), Counter()


def _get(d, key, name):
    try:
        return d[key]
    except KeyError:
        raise EvalError(f"{name} has no entry {key[0] if len(key) == 1 else key}") from None
    except TypeError:
        raise EvalError(f"{name} is not indexed") from None


def _safe_div(a, b):
    if b == 0:
        raise EvalError("division by zero")
    return a / b


def _safe_floordiv(a, b):
    if b == 0:
        raise EvalError("division by zero")
    return a // b


def _safe_mod(a, b):
    if b == 0:
        raise EvalError("modulo by zero")
    return a % b


def _safe_pow(a, b):
    if isinstance(a, int) and isinstance(b, int) and abs(b) > 64:
        return float(a) ** b
    return a ** b


_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: _safe_div, ast.FloorDiv: _safe_floordiv, ast.Mod: _safe_mod, ast.Pow: _safe_pow}
_CMP = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}


def _mathf(fn, args):
    try:
        if fn == "min":
            return min(args)
        if fn == "max":
            return max(args)
        if fn == "abs":
            return abs(args[0])
        if fn == "round":
            return round(args[0]) if len(args) == 1 else round(args[0], int(args[1]))
        if fn == "floor":
            return math.floor(args[0])
        if fn == "ceil":
            return math.ceil(args[0])
        if fn == "sqrt":
            return math.sqrt(args[0])
        if fn == "exp":
            return math.exp(args[0])
        if fn == "log":
            return math.log(args[0]) if len(args) == 1 else math.log(args[0], args[1])
        if fn == "clip":
            return min(max(args[0], args[1]), args[2])
    except (ValueError, OverflowError, TypeError) as e:
        raise EvalError(f"{fn}{tuple(args)}: {e}") from None
    raise EvalError(f"unknown function {fn}")


_ARITY = {"min": (1, 99), "max": (1, 99), "abs": (1, 1), "round": (1, 2), "floor": (1, 1), "ceil": (1, 1),
          "sqrt": (1, 1), "exp": (1, 1), "log": (1, 2), "clip": (3, 3), "bernoulli": (1, 1), "binomial": (2, 2),
          "poisson": (1, 1), "normal": (2, 2), "uniform": (2, 2), "choice": (1, 2), "push": (1, 2), "pop": (1, 2),
          "drop_older": (2, 2), "len": (1, 1), "oldest_age": (1, 1), "mean_age": (1, 1), "fired": (1, 1)}


# ================================================================================================ compiler
class Compiled:
    __slots__ = ("fn", "kind", "deps", "text", "node", "reads", "writes")

    def __init__(self, fn, kind, deps, text, node, reads=frozenset(), writes=frozenset()):
        self.fn, self.kind, self.deps, self.text, self.node = fn, kind, deps, text, node
        self.reads = reads             # names of the vars the expression reads (for dependency-tracked re-evaluation)
        self.writes = writes           # queues changed by push/pop/drop_older inside the expression


class Compiler:
    """Whitelisted expression AST -> closure f(R). ctx: 'param' (params, atoms), 'state' (+vars, now, fired),
    'update' (+draws, push/pop/drop_older), 'delay' (+draws), 'record' (observed keys, params, atoms)."""

    def __init__(self, net, locals_=None, ctx="state", extra_atoms=(), rec_keys=None):
        self.net, self.ctx = net, ctx
        self.locals = dict(locals_ or {})
        self.atoms = set(net.atoms) | set(extra_atoms)
        self.rec_keys = rec_keys
        self.deps = set()
        self.reads = set()
        self.writes = set()

    def compile(self, text):
        text = text.strip()
        if not text:
            raise NetError("empty expression")
        try:
            node = ast.parse(text, mode="eval").body
        except SyntaxError as e:
            raise NetError(f"syntax error in '{text[:80]}': {e.msg}"
                           + (" (hint: write updates after ':' and separate them with ';')" if ":=" in text else ""))
        fn, kind = self.c(node)
        return Compiled(fn, kind, frozenset(self.deps), text, node, frozenset(self.reads), frozenset(self.writes))

    # -------------------------------------------------------------- nodes
    def c(self, n):
        if isinstance(n, ast.Constant):
            v = n.value
            if isinstance(v, bool) or _isnum(v):
                return (lambda R, v=v: v), ("bool" if isinstance(v, bool) else "num")
            raise NetError("text constants are not allowed (use a bare atom name declared in a set or enum)")
        if isinstance(n, ast.Name):
            return self._name(n.id)
        if isinstance(n, ast.Subscript):
            return self._subscript(n)
        if isinstance(n, ast.BinOp):
            op = _BIN.get(type(n.op))
            if op is None:
                raise NetError(f"operator {type(n.op).__name__} is not allowed")
            a, _ = self.c(n.left)
            b, _ = self.c(n.right)
            return (lambda R: op(a(R), b(R))), "num"
        if isinstance(n, ast.UnaryOp):
            a, k = self.c(n.operand)
            if isinstance(n.op, ast.Not):
                return (lambda R: not a(R)), "bool"
            if isinstance(n.op, ast.USub):
                return (lambda R: -a(R)), "num"
            if isinstance(n.op, ast.UAdd):
                return a, k
            raise NetError("operator ~ is not allowed")
        if isinstance(n, ast.BoolOp):
            fs = [self.c(v)[0] for v in n.values]
            if isinstance(n.op, ast.And):
                return (lambda R: all(f(R) for f in fs)), "bool"
            return (lambda R: any(f(R) for f in fs)), "bool"
        if isinstance(n, ast.Compare):
            first, _ = self.c(n.left)
            ops = []
            for o, comp in zip(n.ops, n.comparators):
                f = _CMP.get(type(o))
                if f is None:
                    raise NetError(f"comparison {type(o).__name__} is not allowed (use ==, !=, <, <=, >, >=)")
                ops.append((f, self.c(comp)[0]))

            def cmp(R):
                a = first(R)
                for f, g in ops:
                    b = g(R)
                    if not f(a, b):
                        return False
                    a = b
                return True
            return cmp, "bool"
        if isinstance(n, ast.IfExp):
            t, kt = self.c(n.test)
            a, ka = self.c(n.body)
            b, kb = self.c(n.orelse)
            return (lambda R: a(R) if t(R) else b(R)), (ka if ka == kb else "any")
        if isinstance(n, ast.Call):
            return self._call(n)
        if isinstance(n, (ast.GeneratorExp, ast.ListComp, ast.List, ast.Tuple, ast.Set, ast.Dict)):
            raise NetError("lists and comprehensions are allowed only inside sum/any/all/count/min/max/choice")
        if isinstance(n, ast.NamedExpr):
            raise NetError("':=' is an update: write it after the event's ':' (e.g. `event E rate r : x := 1`)")
        if isinstance(n, ast.Attribute):
            raise NetError("attribute access ('.') is not allowed")
        raise NetError(f"{type(n).__name__} is not allowed in expressions")

    def _name(self, nm):
        net = self.net
        if nm in self.locals:
            return (lambda R: R.loc[nm]), "atom"
        if self.ctx == "record":
            if nm in self.rec_keys:
                return (lambda R: R.rec[nm]), "any"
        elif nm in net.vars and self.ctx != "param":
            v = net.vars[nm]
            if v["index"]:
                raise NetError(f"{nm} is indexed by {', '.join(v['index'])}: write {nm}[...]")
            if v["type"] == "fifo":
                raise NetError(f"{nm} is a fifo: use len({nm}), oldest_age({nm}) or mean_age({nm})")
            self.deps.add("state")
            self.reads.add(nm)
            return (lambda R: R.state[nm]), {"bool": "bool", "enum": "atom"}.get(v["type"], "num")
        if nm in net.params:
            if net.params[nm]["index"]:
                raise NetError(f"param {nm} is indexed by {', '.join(net.params[nm]['index'])}: write {nm}[...]")
            self.deps.add("param")

            def pget(R):
                try:
                    return R.params[nm]
                except KeyError:
                    raise EvalError(f"param {nm} is used before its declaration line") from None
            return pget, "any"
        if nm == "now":
            if self.ctx in ("param",):
                raise NetError("`now` cannot be used here (constants only)")
            self.deps.add("time")
            return (lambda R: R.t), "num"
        if nm in CONSTS:
            v = CONSTS[nm]
            return (lambda R: v), ("bool" if isinstance(v, bool) else "num")
        if nm in self.atoms:
            return (lambda R: nm), "atom"
        if nm in net.sets:
            raise NetError(f"set {nm} can only be iterated, e.g. sum(x[d] for d in {nm})")
        if nm in net.events:
            raise NetError(f"{nm} is an event: use fired({nm}) for its firing count")
        if nm in net.hole_names:
            raise NetError(f"references {nm}, which is a hole (its declaration failed)")
        if nm in net.vars and self.ctx == "param":
            raise NetError(f"{nm} is a var: only params and numbers can be used here")
        if nm in net.declared:
            raise NetError(f"{nm} is declared later (line {net.declared[nm][1]}); declare it before it is used here")
        pool = list(net.vars) + list(net.params) + list(self.atoms) + list(self.locals) + ["now"]
        if self.ctx == "record":
            pool += list(self.rec_keys)
        raise NetError(f"unknown name '{nm}'{_dym(nm, pool)}")

    def _index_fns(self, n):
        sl = n.slice
        elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        return [self.c(e)[0] for e in elts], elts

    def _subscript(self, n):
        if not isinstance(n.value, ast.Name):
            raise NetError("only declared indexed names can be subscripted, e.g. load[d]")
        nm = n.value.id
        fns, elts = self._index_fns(n)
        net = self.net
        if self.ctx == "record" and (nm in self.rec_keys or nm in net.vars):
            def rget(R):
                k = _keystr(nm, tuple(f(R) for f in fns))
                if k not in R.rec:
                    raise EvalError(f"{k} is not observed")
                return R.rec[k]
            return rget, "any"
        if nm in net.vars and self.ctx != "param":
            v = net.vars[nm]
            if not v["index"]:
                raise NetError(f"{nm} is not indexed: write {nm} without [...]")
            if len(fns) != len(v["index"]):
                raise NetError(f"{nm} has {len(v['index'])} index(es) ({', '.join(v['index'])}), got {len(fns)}")
            if v["type"] == "fifo":
                raise NetError(f"{nm} is a fifo: use len({nm}[...]), oldest_age(...) or mean_age(...)")
            self._check_static_index(nm, v["index"], elts)
            self.deps.add("state")
            self.reads.add(nm)
            kind = {"bool": "bool", "enum": "atom"}.get(v["type"], "num")
            if len(fns) == 1:
                f0 = fns[0]
                return (lambda R: _get(R.state[nm], (f0(R),), nm)), kind
            return (lambda R: _get(R.state[nm], tuple(f(R) for f in fns), nm)), kind
        if nm in net.params:
            p = net.params[nm]
            if not p["index"]:
                raise NetError(f"param {nm} is not indexed: write {nm} without [...]")
            if len(fns) != len(p["index"]):
                raise NetError(f"param {nm} has {len(p['index'])} index(es), got {len(fns)}")
            self._check_static_index(nm, p["index"], elts)
            self.deps.add("param")
            return (lambda R: _get(R.params[nm], tuple(f(R) for f in fns), nm)), "any"
        if nm in net.hole_names:
            raise NetError(f"references {nm}, which is a hole (its declaration failed)")
        raise NetError(f"'{nm}[...]': {nm} is not an indexed var or param{_dym(nm, list(net.vars) + list(net.params))}")

    def _check_static_index(self, nm, sets, elts):
        for e, s in zip(elts, sets):
            if isinstance(e, ast.Name) and e.id not in self.locals and e.id in self.atoms:
                if s in self.net.sets and e.id not in self.net.sets[s]["atoms"] and e.id not in self.atoms - set(
                        self.net.atoms):
                    raise NetError(f"{e.id} is not an element of {s} (index of {nm}){_dym(e.id, self.net.sets[s]['atoms'])}")

    def _fifo_ref(self, n):
        if isinstance(n, ast.Name) and n.id in self.net.vars and self.net.vars[n.id]["type"] == "fifo":
            if self.net.vars[n.id]["index"]:
                raise NetError(f"{n.id} is indexed: write {n.id}[...]")
            nm = n.id
            self.deps.add("state")
            self.reads.add(nm)
            return lambda R: R.state[nm]
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in self.net.vars \
                and self.net.vars[n.value.id]["type"] == "fifo":
            nm = n.value.id
            fns, _ = self._index_fns(n)
            if len(fns) != len(self.net.vars[nm]["index"]):
                raise NetError(f"{nm} has {len(self.net.vars[nm]['index'])} index(es)")
            self.deps.add("state")
            self.reads.add(nm)
            return lambda R: _get(R.state[nm], tuple(f(R) for f in fns), nm)
        fifos = [v for v, d in self.net.vars.items() if d["type"] == "fifo"]
        raise NetError(f"expected a fifo var here{(' (declared fifos: ' + ', '.join(fifos) + ')') if fifos else ' (no fifo var is declared: `var line : fifo = []`)'}")

    def _call(self, n):
        if not isinstance(n.func, ast.Name):
            raise NetError("only plain function calls are allowed")
        fn = n.func.id
        if n.keywords:
            raise NetError(f"{fn}: keyword arguments are not allowed")
        args = n.args
        if fn in AGG_FUNCS or (fn in ("min", "max") and len(args) == 1
                               and isinstance(args[0], (ast.GeneratorExp, ast.ListComp, ast.List))):
            return self._agg(fn, args)
        if fn not in _ARITY:
            raise NetError(f"function '{fn}' is not allowed{_dym(fn, ALL_FUNCS)} (allowed: {', '.join(ALL_FUNCS)})")
        lo, hi = _ARITY[fn]
        if not lo <= len(args) <= hi:
            raise NetError(f"{fn} takes {lo if lo == hi else f'{lo}-{hi}'} argument(s), got {len(args)}")
        if self.ctx == "record" and fn in FIFO_FUNCS + PROC_FUNCS + DRAW_FUNCS + ("fired",):
            raise NetError(f"{fn} cannot be used in event expressions over observations")
        if fn in FIFO_FUNCS:
            q = self._fifo_ref(args[0])
            if fn == "len":
                return (lambda R: q(R).n), "num"
            self.deps.add("time")
            if fn == "oldest_age":
                return (lambda R: q(R).oldest_age(R.t)), "num"
            return (lambda R: q(R).mean_age(R.t)), "num"
        if fn == "fired":
            a = args[0]
            if not isinstance(a, ast.Name) or a.id not in self.net.declared_events:
                raise NetError(f"fired(...) takes an event name{_dym(getattr(a, 'id', ''), self.net.declared_events)}")
            if self.ctx == "param":
                raise NetError("fired() cannot be used here (constants only)")
            ev = a.id
            self.deps.add("fired")
            return (lambda R: R.fired[ev]), "num"
        if fn in PROC_FUNCS:
            if self.ctx != "update":
                raise NetError(f"{fn}(...) changes a queue: use it only in an event's updates (after ':')")
            q = self._fifo_ref(args[0])
            self.deps.add("proc")
            a0 = args[0].value if isinstance(args[0], ast.Subscript) else args[0]
            self.writes.add(a0.id)
            if fn == "drop_older":
                age, _ = self.c(args[1])
                return (lambda R: q(R).drop_older(R.t, age(R))), "num"
            k = self.c(args[1])[0] if len(args) > 1 else (lambda R: 1)
            if fn == "push":
                return (lambda R: q(R).push(R.t, _num(k(R), "push"))), "num"
            return (lambda R: q(R).pop(_num(k(R), "pop"))), "num"
        if fn in DRAW_FUNCS:
            if self.ctx not in ("update", "delay"):
                raise NetError(f"random draw {fn}(...) is allowed only in updates and delays")
            self.deps.add("draw")
            if fn == "choice":
                lists = []
                for a in args:
                    if not isinstance(a, ast.List):
                        raise NetError("choice takes literal lists: choice([v1, v2], [w1, w2])")
                    lists.append([self.c(e)[0] for e in a.elts])
                vals = lists[0]
                ws = lists[1] if len(lists) > 1 else None
                return (lambda R: _draw("choice", R, ([f(R) for f in vals], [f(R) for f in ws] if ws else None))), "any"
            fs = [self.c(a)[0] for a in args]
            kind = "bool" if fn == "bernoulli" else "num"
            return (lambda R: _draw(fn, R, [f(R) for f in fs])), kind
        fs = [self.c(a)[0] for a in args]
        return (lambda R: _mathf(fn, [f(R) for f in fs])), "num"

    def _agg(self, fn, args):
        if len(args) != 1:
            raise NetError(f"{fn} takes one comprehension, e.g. {fn}(x[d] for d in Set)")
        g = args[0]
        if isinstance(g, ast.List):
            fs = [self.c(e)[0] for e in g.elts]

            def vals(R):
                return [f(R) for f in fs]
        elif isinstance(g, (ast.GeneratorExp, ast.ListComp)):
            gens, saved = [], dict(self.locals)
            for comp in g.generators:
                if not isinstance(comp.target, ast.Name) or not isinstance(comp.iter, ast.Name):
                    raise NetError("comprehensions must have the form `for d in SetName`")
                s = comp.iter.id
                if s not in self.net.sets:
                    raise NetError(f"comprehensions run only over declared sets; '{s}' is not a set"
                                   f"{_dym(s, self.net.sets)}")
                self.locals[comp.target.id] = s
                ifs = [self.c(x)[0] for x in comp.ifs]
                gens.append((comp.target.id, s, ifs))
            elt = self.c(g.elt)[0]
            self.locals = saved
            e = g.elt
            whole = (len(gens) == 1 and not gens[0][2] and isinstance(e, ast.Subscript)
                     and isinstance(e.value, ast.Name) and e.value.id in self.net.vars
                     and self.net.vars[e.value.id]["index"] == [gens[0][1]]
                     and self.net.vars[e.value.id]["type"] in ("int", "real", "bool")
                     and isinstance(e.slice, ast.Name) and e.slice.id == gens[0][0])
            if whole:
                # x[d] for d in S where x is indexed by exactly S: every entry of x, in set order (exact fast path)
                vn = e.value.id

                def vals(R):
                    return list(R.state[vn].values())
            elif len(gens) == 1 and not gens[0][2]:
                nm1, s1, _ = gens[0]

                def vals(R):
                    loc = R.loc
                    old = loc.get(nm1, _MISSING)
                    out = []
                    try:
                        for a in R.sets[s1]:
                            loc[nm1] = a
                            out.append(elt(R))
                    finally:
                        if old is _MISSING:
                            loc.pop(nm1, None)
                        else:
                            loc[nm1] = old
                    return out
        else:
            raise NetError(f"{fn} takes a comprehension `{fn}(expr for d in Set)` or a list")
        if isinstance(g, (ast.GeneratorExp, ast.ListComp)) and not (whole or (len(gens) == 1 and not gens[0][2])):

            def vals(R):
                out = []
                loc = R.loc
                old = {nm: loc.get(nm, _MISSING) for nm, _, _ in gens}

                def rec(i):
                    if i == len(gens):
                        out.append(elt(R))
                        return
                    nm, s, ifs = gens[i]
                    for a in R.sets[s]:
                        loc[nm] = a
                        if all(f(R) for f in ifs):
                            rec(i + 1)
                try:
                    rec(0)
                finally:
                    for nm, v in old.items():
                        if v is _MISSING:
                            loc.pop(nm, None)
                        else:
                            loc[nm] = v
                return out
        if fn == "sum":
            return (lambda R: sum(vals(R))), "num"
        if fn == "count":
            return (lambda R: sum(1 for v in vals(R) if v)), "num"
        if fn == "any":
            return (lambda R: any(vals(R))), "bool"
        if fn == "all":
            return (lambda R: all(vals(R))), "bool"

        def mm(R):
            v = vals(R)
            if not v:
                raise EvalError(f"{fn} of an empty set")
            return min(v) if fn == "min" else max(v)
        return mm, "num"


_MISSING = object()


# ================================================================================================ the parsed net
class EventDecl:
    def __init__(self, name, line):
        self.name, self.line = name, line
        self.idx = []                  # [(local, set)]
        self.kind = None               # rate | after | every | at | immediate
        self.rate = self.delay = self.period = self.start = self.until = None
        self.at = []
        self.guard = None
        self.prio = None
        self.updates = []              # dicts
        self.writes = frozenset()      # vars changed by the updates
        self.src = ""
        self.magic = []


class Net:
    """Parsed document. Bad lines become holes; the rest is compiled."""

    def __init__(self, src):
        self.src = src if isinstance(src, str) else ""
        self.unit = "unit"
        self.horizon = None            # Compiled
        self.grid = None
        self.seeds = DEFAULT_SEEDS
        self.sets, self.params, self.vars, self.events, self.observes, self.whatifs = {}, {}, {}, {}, {}, {}
        self.flows, self.invariants = [], []
        self.holes, self.warnings, self.hole_names = [], [], set()
        self.atoms = set()
        self.declared = {}             # name -> (kind, line)
        self.declared_events = set()
        self.provenance = Counter()
        self.lines = []
        self._parse()

    # -------------------------------------------------------------- helpers
    def hole(self, no, code, msg, name=None):
        self.holes.append({"line": no, "text": code[:160], "error": msg})
        if name:
            self.hole_names.add(name)

    def comp(self, text, ctx="state", locals_=None, extra_atoms=()):
        return Compiler(self, locals_, ctx, extra_atoms).compile(text)

    # -------------------------------------------------------------- parse
    def _parse(self):
        raw = self.src.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for i, line in enumerate(raw, 1):
            code, com = _split_comment(line)
            code = code.strip()
            if code:
                self.lines.append((i, code, com))
                m = re.match(r"^(stated|derived|assumed)\b", com, re.I)
                if m:
                    self.provenance[m.group(1).lower()] += 1
        by_kind = {k: [] for k in KINDS + SETTINGS}
        # pass 1: classify, register names, collect atoms
        for no, code, com in self.lines:
            w = re.match(r"^([A-Za-z_]\w*)", code)
            w = w.group(1) if w else ""
            if w not in by_kind:
                self.hole(no, code, f"unknown declaration '{w or code[:20]}'{_dym(w, KINDS + SETTINGS)} (each line "
                                    f"starts with one of: {', '.join(KINDS + SETTINGS)})")
                continue
            if w in ("param", "var", "event", "observe", "set"):
                m = re.match(r"^\w+\s+([A-Za-z_]\w*)", code)
                if not m:
                    self.hole(no, code, f"{w}: missing name")
                    continue
                nm = m.group(1)
                if nm in self.declared:
                    self.hole(no, code, f"'{nm}' is already declared on line {self.declared[nm][1]}", None)
                    continue
                if nm in CONSTS or nm == "now" or nm in ALL_FUNCS or nm in TIMINGS or nm in KINDS:
                    self.hole(no, code, f"'{nm}' is a reserved word; choose another name", nm)
                    continue
                self.declared[nm] = (w, no)
                if w == "event":
                    self.declared_events.add(nm)
                if w == "set":
                    m2 = re.match(r"^set\s+([A-Za-z_]\w*)\s*=\s*\{(.*)\}\s*$", code)
                    try:
                        if not m2:
                            raise NetError("set syntax: set Name = {a, b, c}  (or {1..5})")
                        atoms = _parse_atoms(m2.group(2))
                        if not atoms:
                            raise NetError("a set needs at least one element")
                        self.sets[nm] = {"atoms": atoms, "line": no}
                        self.atoms.update(a for a in atoms if isinstance(a, str))
                    except NetError as e:
                        self.hole(no, code, str(e), nm)
                    continue
                if w == "var":
                    m2 = re.match(r"^var\s+[A-Za-z_]\w*\s*(?:\[[^\]]*\])?\s*:\s*\{([^}]*)\}", code)
                    if m2:
                        try:
                            self.atoms.update(a for a in _parse_atoms(m2.group(1)) if isinstance(a, str))
                        except NetError:
                            pass
            by_kind[w].append((no, code, com))
        for nm in sorted(self.atoms & (set(self.declared) - set(self.sets))):   # sorted: hash-seed free (QA-0194)
            self.warnings.append(f"'{nm}' is both an atom (set/enum element) and a declared name; the declared "
                                 f"name wins in expressions")
        for no, code, com in by_kind["unit"] + by_kind["horizon"] + by_kind["grid"] + by_kind["seeds"]:
            self._try(no, code, None, self._setting)
        for no, code, com in by_kind["param"]:
            self._try(no, code, re.match(r"^\w+\s+([A-Za-z_]\w*)", code).group(1), self._param)
        for no, code, com in by_kind["var"]:
            self._try(no, code, re.match(r"^\w+\s+([A-Za-z_]\w*)", code).group(1), self._var)
        for no, code, com in by_kind["event"]:
            nm = re.match(r"^\w+\s+([A-Za-z_]\w*)", code).group(1)
            try:
                ev = self.parse_event(no, code)
                self.events[nm] = ev
            except NetError as e:
                self.hole(no, code, str(e), nm)
        for no, code, com in by_kind["flow"]:
            self._try(no, code, None, self._flow)
        for no, code, com in by_kind["observe"]:
            self._try(no, code, re.match(r"^\w+\s+([A-Za-z_]\w*)", code).group(1), self._observe)
        for no, code, com in by_kind["invariant"]:
            self._try(no, code, None, self._invariant)
        for no, code, com in by_kind["whatif"]:
            self._try(no, code, None, self._whatif)
        if self.horizon is None:
            self.warnings.append("no `horizon` line: runs use horizon 10 (hint: add `horizon <number>`)")

    def _try(self, no, code, name, fn):
        try:
            fn(no, code)
        except NetError as e:
            self.hole(no, code, str(e), name)

    def _setting(self, no, code):
        w, _, rest = code.partition(" ")
        rest = rest.strip()
        if not rest:
            raise NetError(f"{w} needs a value")
        if w == "unit":
            self.unit = rest[:30]
        elif w == "horizon":
            self.horizon = self.comp(rest, "param")
        elif w == "grid":
            self.grid = self.comp(rest, "param")
        elif w == "seeds":
            if not re.match(r"^\d+$", rest):
                raise NetError("seeds needs a whole number")
            self.seeds = max(1, min(MAX_SEEDS, int(rest)))

    def _index_sets(self, text):
        sets = [s.strip() for s in _split_top(text) if s.strip()]
        for s in sets:
            if s not in self.sets:
                raise NetError(f"unknown set '{s}'{_dym(s, self.sets)} (declare it with `set {s} = {{...}}`)")
        return sets

    def _valuespec(self, text, idx, ctx="param"):
        """A scalar expression or a dict literal {a: v, ...} for an indexed declaration."""
        text = text.strip()
        if idx and text.startswith("{"):
            try:
                node = ast.parse(text, mode="eval").body
            except SyntaxError as e:
                raise NetError(f"syntax error in '{text[:60]}': {e.msg}")
            if not isinstance(node, ast.Dict):
                raise NetError("expected {atom: value, ...}")
            out = {}
            for k, v in zip(node.keys, node.values):
                if k is None:
                    raise NetError("** is not allowed in a value table")
                key = self._atom_key(k, idx)
                out[key] = self.comp(ast.unparse(v), ctx)
            return {"dict": out}
        return {"expr": self.comp(text, ctx)}

    def _atom_key(self, k, idx):
        elts = k.elts if isinstance(k, ast.Tuple) else [k]
        if len(elts) != len(idx):
            raise NetError(f"key {ast.unparse(k)} needs {len(idx)} element(s) ({', '.join(idx)})")
        key = []
        for e, s in zip(elts, idx):
            if isinstance(e, ast.Name):
                a = e.id
            elif isinstance(e, ast.Constant) and isinstance(e.value, int) and not isinstance(e.value, bool):
                a = e.value
            elif isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub) and isinstance(e.operand, ast.Constant):
                a = -e.operand.value
            else:
                raise NetError(f"key {ast.unparse(e)} must be an element of {s}")
            if a not in self.sets[s]["atoms"]:
                raise NetError(f"{a} is not an element of {s}{_dym(a, self.sets[s]['atoms'])}")
            key.append(a)
        return tuple(key)

    def _param(self, no, code):
        m = re.match(r"^param\s+([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?\s*=\s*(.+)$", code, re.S)
        if not m:
            raise NetError("param syntax: param name = value   or   param name[Set] = {a: 1, b: 2}")
        nm, idx, rest = m.group(1), m.group(2), m.group(3)
        idx = self._index_sets(idx) if idx is not None else []
        rest = rest.strip()
        # an unknown bare name as a whole value is an atom-valued param (e.g. param policy = fifo)
        if re.match(r"^[A-Za-z_]\w*$", rest) and rest not in self.params and rest not in CONSTS \
                and rest not in self.declared:
            self.atoms.add(rest)
        spec = self._valuespec(rest, idx)
        self.params[nm] = {"index": idx, "line": no, **spec}

    def _var(self, no, code):
        m = re.match(r"^var\s+([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?\s*:\s*(int|real|bool|fifo|\{[^}]*\})\s*(.*)$",
                     code, re.S)
        if not m:
            t = re.match(r"^var\s+[A-Za-z_]\w*\s*(?:\[[^\]]*\])?\s*:\s*(\S+)", code)
            hint = f" (type '{t.group(1)}' is unknown{_dym(t.group(1), ['int', 'real', 'bool', 'fifo'])})" if t else ""
            raise NetError("var syntax: var name : int|real|bool|fifo|{a, b} [lo, hi] = init" + hint)
        nm, idx, typ, rest = m.groups()
        idx = self._index_sets(idx) if idx is not None else []
        d = {"index": idx, "line": no, "enum": None, "lo": None, "hi": None, "init": None}
        if typ.startswith("{"):
            d["type"] = "enum"
            d["enum"] = _parse_atoms(typ[1:-1])
            if not d["enum"]:
                raise NetError("an enum type needs at least one value")
        else:
            d["type"] = typ
        rest = rest.strip()
        if rest.startswith("["):
            depth, j = 0, None
            for i, ch in enumerate(rest):
                depth += ch in "([{"
                depth -= ch in ")]}"
                if depth == 0:
                    j = i
                    break
            if j is None:
                raise NetError("unclosed bounds '[...'")
            b = _split_top(rest[1:j])
            if len(b) != 2:
                raise NetError("bounds need two values: [lo, hi] (use inf for no upper bound)")
            if d["type"] in ("bool", "enum"):
                raise NetError(f"a {d['type']} var has no numeric bounds")
            if d["type"] == "fifo" and b[0].strip() not in ("0", "_"):
                raise NetError("a fifo's bounds are [0, capacity]")
            d["lo"] = None if b[0].strip() in ("_", "") else self.comp(b[0], "param")
            d["hi"] = None if b[1].strip() in ("_", "") else self.comp(b[1], "param")
            rest = rest[j + 1:].strip()
        if rest:
            if not rest.startswith("=") or rest.startswith("=="):
                raise NetError(f"unexpected '{rest[:30]}' (expected '= <initial value>')")
            init = rest[1:].strip()
            if d["type"] == "fifo":
                if init in ("[]", ""):
                    init = None
                else:
                    d["init"] = self._valuespec(init, idx)
                    init = None
            elif d["type"] == "enum" and re.match(r"^[A-Za-z_]\w*$", init) and init not in self.params:
                if init not in d["enum"]:
                    raise NetError(f"initial value {init} is not one of {d['enum']}")
                d["init"] = {"expr": self.comp(init, "param")}
                init = None
            if init:
                d["init"] = self._valuespec(init, idx)
        self.vars[nm] = d

    def _update(self, text, locals_, evname):
        text = text.strip()
        i, op = _find_assign(text)
        if i < 0:
            try:
                node = ast.parse(text, mode="eval").body
            except SyntaxError as e:
                raise NetError(f"update '{text[:60]}': {e.msg}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in PROC_FUNCS:
                c = Compiler(self, locals_, "update").compile(text)
                return {"kind": "proc", "fn": c.fn, "text": text, "var": self._proc_target(node), "deps": c.deps,
                        "writes": set(c.writes), "reads": set(c.reads)}
            raise NetError(f"'{text[:60]}' is not an update (write x := e, x += e, x -= e, or push/pop/drop_older"
                           f"(queue, n)); an update that changes nothing is not allowed")
        lhs, rhs = text[:i].strip(), text[i + len(op):].strip()
        if op == "=":
            op = ":="
        try:
            tnode = ast.parse(lhs, mode="eval").body
        except SyntaxError:
            raise NetError(f"cannot assign to '{lhs}'")
        cmp = Compiler(self, locals_, "update")
        if isinstance(tnode, ast.Name):
            vn, keyfns, kelts = tnode.id, None, []
        elif isinstance(tnode, ast.Subscript) and isinstance(tnode.value, ast.Name):
            vn = tnode.value.id
            keyfns, kelts = cmp._index_fns(tnode)
        else:
            raise NetError(f"cannot assign to '{lhs}' (only vars, e.g. q or load[d])")
        if vn not in self.vars:
            if vn in self.hole_names:
                raise NetError(f"references {vn}, which is a hole (its declaration failed)")
            if vn in self.params:
                raise NetError(f"{vn} is a param (constant); only vars can be updated")
            raise NetError(f"unknown var '{vn}'{_dym(vn, self.vars)}")
        v = self.vars[vn]
        if bool(v["index"]) != (keyfns is not None):
            raise NetError(f"{vn} is {'indexed: write ' + vn + '[...]' if v['index'] else 'not indexed'}")
        if keyfns is not None and len(keyfns) != len(v["index"]):
            raise NetError(f"{vn} has {len(v['index'])} index(es)")
        key_deps = frozenset(cmp.deps)
        if v["type"] == "fifo":
            raise NetError(f"{vn} is a fifo: change it with push({vn}, n), pop({vn}, n) or drop_older({vn}, age)")
        if v["type"] in ("bool", "enum") and op != ":=":
            raise NetError(f"'{op}' on a {v['type']} var {vn} (use :=)")
        val = Compiler(self, locals_, "update").compile(rhs)
        if v["type"] == "enum" and val.kind not in ("atom", "any"):
            raise NetError(f"{vn} takes one of {v['enum']}, not a {val.kind}")
        if v["type"] == "enum" and isinstance(val.node, ast.Name) and val.node.id not in locals_ \
                and val.node.id not in self.params and val.node.id not in v["enum"]:
            raise NetError(f"{val.node.id} is not a value of {vn} {v['enum']}")
        if v["type"] in ("int", "real") and val.kind == "atom":
            raise NetError(f"{vn} is numeric; '{rhs}' is an atom")
        return {"kind": "assign", "var": vn, "keyfns": keyfns, "op": op, "val": val, "text": text,
                "key_static": not (key_deps - {"param"}), "deps": val.deps | key_deps,
                "writes": {vn} | set(val.writes), "reads": set(val.reads) | set(cmp.reads)}

    def _proc_target(self, node):
        a = node.args[0] if node.args else None
        if isinstance(a, ast.Subscript):
            a = a.value
        return a.id if isinstance(a, ast.Name) else None

    def parse_event(self, no, code, extra_atoms=()):
        m = re.match(r"^event\s+([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?\s*(.*)$", code, re.S)
        if not m:
            raise NetError("event syntax: event Name[d in Set] <timing> [when <guard>] [priority k] : <updates>")
        name, idxs, rest = m.groups()
        ev = EventDecl(name, no)
        ev.src = code
        if idxs is not None:
            for part in _split_top(idxs):
                mm = re.match(r"^\s*([A-Za-z_]\w*)\s+in\s+([A-Za-z_]\w*)\s*$", part)
                if not mm:
                    raise NetError(f"index '{part.strip()}': write `d in SetName`")
                if mm.group(2) not in self.sets:
                    raise NetError(f"unknown set '{mm.group(2)}'{_dym(mm.group(2), self.sets)}")
                ev.idx.append((mm.group(1), mm.group(2)))
        loc = {d: s for d, s in ev.idx}
        ci = _find_header_colon(rest)
        if ci < 0:
            raise NetError("missing ':' between the timing/guard and the updates")
        header, body = rest[:ci].strip(), rest[ci + 1:].strip()
        mt = re.match(r"^([A-Za-z_]\w*)\b\s*(.*)$", header, re.S)
        if not mt or mt.group(1) not in TIMINGS:
            got = mt.group(1) if mt else header[:20]
            raise NetError(f"timing '{got}' is unknown{_dym(got, TIMINGS)}: use rate <expr>, after <delay>, "
                           f"every <period> [from t0] [until t1], at <t1, t2>, or immediate")
        ev.kind, trest = mt.group(1), mt.group(2)
        kws = ["when", "priority"] + (["from", "until"] if ev.kind == "every" else [])
        texpr, parts = _kw_split(trest, kws)
        C = lambda text, ctx: Compiler(self, loc, ctx, extra_atoms).compile(text)  # noqa: E731
        if ev.kind != "immediate" and not texpr:
            raise NetError(f"'{ev.kind}' needs a value")
        if ev.kind == "rate":
            ev.rate = C(texpr, "state")
        elif ev.kind == "after":
            ev.delay = self._delay(texpr, loc, extra_atoms)
        elif ev.kind == "every":
            ev.period = C(texpr, "param")
            ev.start = C(parts["from"], "param") if parts.get("from") else None
            ev.until = C(parts["until"], "param") if parts.get("until") else None
        elif ev.kind == "at":
            ev.at = [C(x, "param") for x in _split_top(texpr) if x.strip()]
        elif texpr:
            raise NetError(f"'immediate' takes no value (found '{texpr[:30]}'); a condition goes after `when`")
        if "when" in parts:
            if not parts["when"]:
                raise NetError("`when` needs a condition")
            ev.guard = C(parts["when"], "state")
        if "priority" in parts:
            ev.prio = C(parts["priority"], "param")
        for u in _split_top(body, ";"):
            if u.strip():
                ev.updates.append(self._update(u, loc, name))
        ev.writes = frozenset(set().union(*[u["writes"] for u in ev.updates]) if ev.updates else ())
        # magic numbers: stated-looking numbers inside the event line that are not params (priorities excluded)
        scan = code[m.start(3):]
        if parts.get("priority"):
            scan = scan.replace(parts["priority"], " ", 1)
        scan = re.sub(r"[A-Za-z_]\w*", " ", scan)
        nums = [x for x in re.findall(r"(?<![\w.])(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+)", scan)
                if float(x) not in UNIT_CONSTANTS]
        ev.magic = nums
        return ev

    def _delay(self, text, loc, extra_atoms=()):
        try:
            node = ast.parse(text, mode="eval").body
        except SyntaxError as e:
            raise NetError(f"syntax error in delay '{text[:60]}': {e.msg}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DIST_FUNCS:
            fn = node.func.id
            need = {"exp": 1, "uniform": 2, "normal": 2, "lognormal": 2, "erlang": 2}.get(fn)
            if fn == "choice":
                c = Compiler(self, loc, "delay", extra_atoms).compile(text)
                return {"dist": "const", "f": c}
            if len(node.args) != need:
                raise NetError(f"delay {fn} takes {need} argument(s)")
            args = [Compiler(self, loc, "state", extra_atoms).compile(ast.unparse(a)) for a in node.args]
            return {"dist": fn, "args": args}
        return {"dist": "const", "f": Compiler(self, loc, "delay", extra_atoms).compile(text)}

    def _flow(self, no, code):
        m = re.match(r"^flow\s+([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?\s*'\s*=\s*(.+)$", code, re.S)
        if not m:
            raise NetError("flow syntax: flow level' = <rate of change per time unit>")
        vn, idx, rhs = m.groups()
        if vn not in self.vars:
            raise NetError(f"unknown var '{vn}'{_dym(vn, self.vars)}")
        v = self.vars[vn]
        if v["type"] != "real":
            raise NetError(f"flows need a real var; {vn} is {v['type']} (declare `var {vn} : real ...`)")
        locs = [x.strip() for x in _split_top(idx)] if idx is not None else []
        if len(locs) != len(v["index"]):
            raise NetError(f"{vn} has {len(v['index'])} index(es): write flow {vn}"
                           + (f"[{', '.join('d' + str(i) for i in range(len(v['index'])))}]" if v["index"] else "")
                           + "' = ...")
        for x in locs:
            if not re.match(r"^[A-Za-z_]\w*$", x):
                raise NetError(f"flow index '{x}' must be a plain name")
        loc = dict(zip(locs, v["index"]))
        if any(f["var"] == vn for f in self.flows):
            raise NetError(f"{vn} already has a flow")
        c = Compiler(self, loc, "state").compile(rhs)
        self.flows.append({"var": vn, "locs": locs, "f": c, "line": no})

    def _observe(self, no, code):
        m = re.match(r"^observe\s+([A-Za-z_]\w*)\s*=\s*(.+)$", code, re.S)
        if not m:
            raise NetError("observe syntax: observe name = <expression>")
        nm = m.group(1)
        if nm in self.vars or nm in self.params:
            raise NetError(f"'{nm}' is already a var/param name")
        self.observes[nm] = {"f": self.comp(m.group(2), "state"), "line": no}

    def _invariant(self, no, code):
        m = re.match(r"^invariant\s+(conserved\s+)?(.+)$", code, re.S)
        if not m:
            raise NetError("invariant syntax: invariant <condition>   or   invariant conserved <linear total>")
        c = self.comp(m.group(2), "state")
        inv = {"text": m.group(2).strip(), "f": c, "conserved": bool(m.group(1)), "line": no}
        if inv["conserved"]:
            inv["linear"] = self._linear(c.node)
        self.invariants.append(inv)

    def _linear(self, node):
        """Coefficients {(var, key|None): coefficient-expr-value} of a linear total; None if not linear."""
        out = {}

        def add(k, c):
            out[k] = out.get(k, 0) + c

        def const(n):
            try:
                v = Compiler(self, None, "param").compile(ast.unparse(n)).fn(_param_env(self))
                return v if _isnum(v) else None
            except (NetError, EvalError, Exception):
                return None

        def walk(n, c):
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
                return walk(n.left, c) and walk(n.right, c if isinstance(n.op, ast.Add) else -c)
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
                return walk(n.operand, -c if isinstance(n.op, ast.USub) else c)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
                a, b = const(n.left), const(n.right)
                if a is not None:
                    return walk(n.right, c * a)
                if b is not None:
                    return walk(n.left, c * b)
                return False
            if isinstance(n, ast.Name) and n.id in self.vars and not self.vars[n.id]["index"] \
                    and self.vars[n.id]["type"] in ("int", "real"):
                add((n.id, None), c)
                return True
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in self.vars:
                vn = n.value.id
                sl = n.slice
                elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
                key = []
                for e in elts:
                    if isinstance(e, ast.Name) and e.id in self.atoms:
                        key.append(e.id)
                    elif isinstance(e, ast.Constant) and isinstance(e.value, int):
                        key.append(e.value)
                    else:
                        return False
                add((vn, tuple(key)), c)
                return True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "sum" and len(n.args) == 1 \
                    and isinstance(n.args[0], ast.GeneratorExp) and len(n.args[0].generators) == 1:
                g = n.args[0]
                comp = g.generators[0]
                if comp.ifs or not isinstance(comp.target, ast.Name) or not isinstance(comp.iter, ast.Name) \
                        or comp.iter.id not in self.sets:
                    return False
                d, s = comp.target.id, comp.iter.id
                e = g.elt
                if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) and e.value.id in self.vars \
                        and isinstance(e.slice, ast.Name) and e.slice.id == d:
                    for a in self.sets[s]["atoms"]:
                        add((e.value.id, (a,)), c)
                    return True
                return False
            if isinstance(n, ast.Constant) and _isnum(n.value):
                return True
            if const(n) is not None:
                return True
            return False
        return out if walk(node, 1) else None

    def _whatif(self, no, code):
        m = re.match(r"^whatif\s+([A-Za-z_]\w*)\s*:\s*(.+)$", code, re.S)
        if not m:
            raise NetError("whatif syntax: whatif name: <clause>; <clause>  (clauses: param p = v, set S = {..}, "
                           "init x = v, scale E k, disable E, horizon T, replace <event line>)")
        nm = m.group(1)
        if nm in self.whatifs:
            raise NetError(f"whatif {nm} is declared twice")
        self.whatifs[nm] = {"clauses": self.parse_clauses(m.group(2)), "line": no, "text": m.group(2).strip()}

    def parse_clauses(self, text):
        if isinstance(text, str):
            parts = []
            rest = text
            while rest.strip():
                if rest.strip().startswith("replace"):
                    parts.append(rest.strip())
                    break
                seg = _split_top(rest, ";")
                parts.append(seg[0])
                rest = ";".join(seg[1:])
            items = [p for p in parts if p.strip()]
        else:
            items = [str(x) for x in text]
        clauses, extra = [], set()
        # set clauses first, so their new atoms are known to the others
        for it in sorted(items, key=lambda s: 0 if s.strip().startswith("set ") else 1):
            cl = self.parse_clause(it, extra)
            if cl["kind"] == "set":
                extra.update(a for a in cl["atoms"] if isinstance(a, str))
            clauses.append(cl)
        return clauses

    def parse_clause(self, text, extra=()):
        t = text.strip()
        w = re.match(r"^([A-Za-z_]\w*)", t)
        w = w.group(1) if w else ""
        kinds = ("param", "set", "init", "scale", "disable", "horizon", "replace")
        if w == "param" or w == "init":
            m = re.match(r"^\w+\s+([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?\s*=\s*(.+)$", t, re.S)
            if not m:
                raise NetError(f"clause syntax: {w} name = value   or   {w} name[a] = value")
            nm, key, val = m.groups()
            pool = self.params if w == "param" else self.vars
            if nm not in pool:
                raise NetError(f"{w} clause: unknown {'param' if w == 'param' else 'var'} '{nm}'{_dym(nm, pool)} "
                               f"(declared: {', '.join(list(pool)[:30])})")
            idx = pool[nm]["index"]
            keyv = None
            if key is not None:
                if not idx:
                    raise NetError(f"{nm} is not indexed")
                ks = []
                for part, s in zip(_split_top(key), idx):
                    p = part.strip()
                    a = int(p) if re.match(r"^-?\d+$", p) else p
                    if a not in self.sets[s]["atoms"] and a not in extra:
                        raise NetError(f"{a} is not an element of {s}{_dym(a, self.sets[s]['atoms'])} (add it with "
                                       f"a `set {s} = {{...}}` clause)")
                    ks.append(a)
                if len(ks) != len(idx):
                    raise NetError(f"{nm} has {len(idx)} index(es)")
                keyv = tuple(ks)
            if w == "init" and self.vars[nm]["type"] == "enum" and re.match(r"^[A-Za-z_]\w*$", val.strip()):
                c = Compiler(self, None, "param", extra).compile(val)
            elif idx and keyv is None and val.strip().startswith("{"):
                vs = self._valuespec(val, idx)
                return {"kind": w, "name": nm, "key": None, "table": vs["dict"], "text": t}
            else:
                c = Compiler(self, None, "param", extra).compile(val)
            return {"kind": w, "name": nm, "key": keyv, "f": c, "text": t}
        if w == "set":
            m = re.match(r"^set\s+([A-Za-z_]\w*)\s*=\s*\{(.*)\}\s*$", t, re.S)
            if not m:
                raise NetError("clause syntax: set Name = {a, b, c}")
            if m.group(1) not in self.sets:
                raise NetError(f"unknown set '{m.group(1)}'{_dym(m.group(1), self.sets)}")
            return {"kind": "set", "name": m.group(1), "atoms": _parse_atoms(m.group(2)), "text": t}
        if w in ("scale", "disable"):
            m = re.match(r"^\w+\s+([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?\s*(.*)$", t, re.S)
            if not m:
                raise NetError(f"clause syntax: {w} EventName" + (" factor" if w == "scale" else ""))
            nm, key, rest = m.groups()
            if nm not in self.events:
                raise NetError(f"{w}: unknown event '{nm}'{_dym(nm, self.events)} (events: "
                               f"{', '.join(list(self.events)[:30])})")
            keyv = tuple((int(p.strip()) if re.match(r"^-?\d+$", p.strip()) else p.strip())
                         for p in _split_top(key)) if key is not None else None
            if w == "scale":
                if not rest.strip():
                    raise NetError("scale needs a factor: scale EventName 0.8")
                return {"kind": "scale", "name": nm, "key": keyv,
                        "f": Compiler(self, None, "param", extra).compile(rest), "text": t}
            if rest.strip():
                raise NetError(f"unexpected '{rest.strip()[:20]}' after disable {nm}")
            return {"kind": "disable", "name": nm, "key": keyv, "text": t}
        if w == "horizon":
            return {"kind": "horizon", "f": Compiler(self, None, "param", extra).compile(t[len("horizon"):]),
                    "text": t}
        if w == "replace":
            line = t[len("replace"):].strip()
            mm = re.match(r"^event\s+([A-Za-z_]\w*)", line)
            if not mm:
                raise NetError("replace takes a whole event line: replace event Name ... : ...")
            if mm.group(1) not in self.events:
                raise NetError(f"replace: unknown event '{mm.group(1)}'{_dym(mm.group(1), self.events)} (replace "
                               f"changes an existing event, matched by its name)")
            return {"kind": "replace", "name": mm.group(1), "event": self.parse_event(0, line, extra), "text": t}
        raise NetError(f"unknown what-if clause '{t[:40]}'{_dym(w, kinds)} (use: param p = v; set S = {{..}}; "
                       f"init x = v; scale E k; disable E; horizon T; replace event ...)")

    # -------------------------------------------------------------- summaries
    def usable(self):
        return bool(self.vars) and bool(self.events)

    def timing_text(self, ev):
        return ev.kind


def _param_env(net, sets=None):
    R = Env()
    R.sets = sets or {k: v["atoms"] for k, v in net.sets.items()}
    R.params = {}
    for nm, d in net.params.items():
        try:
            R.params[nm] = _eval_param(R, d)
        except (EvalError, Exception):
            pass
    return R


def _eval_param(R, d):
    if d["index"]:
        keys = list(itertools.product(*[R.sets[s] for s in d["index"]]))
        if "dict" in d:
            out = {}
            for k in keys:
                if k not in d["dict"]:
                    raise EvalError(f"no value for {k[0] if len(k) == 1 else k}")
                out[k] = d["dict"][k].fn(R)
            return out
        v = d["expr"].fn(R)
        return {k: v for k in keys}
    return d["expr"].fn(R)


# ================================================================================================ instantiation
class Inst:
    """One event instance (an event declaration with its index values bound)."""
    __slots__ = ("name", "ev", "loc", "prio", "order", "kind", "scale", "guard", "rate", "updates", "rng_t", "rng_u",
                 "clock", "T", "P", "next_t", "k", "period", "start", "until", "at", "at_i", "fluid", "incs",
                 "dreads", "dalways", "gval", "force")

    def __init__(self):
        self.clock = None
        self.T = 0.0
        self.P = 0.0
        self.next_t = math.inf
        self.k = 0
        self.at_i = 0
        self.fluid = False
        self.incs = None


class Prog:
    """A model ready to run: the parsed net with one run's interventions applied."""

    def __init__(self, net, clauses=(), horizon=None, mode="stochastic"):
        self.net, self.mode = net, mode
        self.fluid = mode == "fluid"
        self.notes = []
        sets = {k: list(v["atoms"]) for k, v in net.sets.items()}
        events = dict(net.events)
        disabled, scale = set(), {}
        p_over, i_over, h_over = [], [], None
        for cl in clauses:
            k = cl["kind"]
            if k == "set":
                sets[cl["name"]] = list(cl["atoms"])
            elif k == "param":
                p_over.append(cl)
            elif k == "init":
                i_over.append(cl)
            elif k == "disable":
                disabled.add(cl["name"] if cl["key"] is None else _keystr(cl["name"], cl["key"]))
            elif k == "scale":
                scale[cl["name"] if cl["key"] is None else _keystr(cl["name"], cl["key"])] = cl
            elif k == "horizon":
                h_over = cl
            elif k == "replace":
                events[cl["name"]] = cl["event"]
        self.sets = sets
        R = Env()
        R.sets = sets
        params = {}
        R.params = params
        for nm, d in net.params.items():
            try:
                full = [c for c in p_over if c["name"] == nm and c["key"] is None and "f" in c]
                table = [c for c in p_over if c["name"] == nm and "table" in c]
                if full and not d["index"]:
                    v = full[-1]["f"].fn(R)
                elif d["index"] and (full or table):
                    keys = list(itertools.product(*[sets[s] for s in d["index"]]))
                    if full:
                        x = full[-1]["f"].fn(R)
                        v = {k: x for k in keys}
                    else:
                        tb = table[-1]["table"]
                        v = {}
                        for k in keys:
                            if k not in tb:
                                raise EvalError(f"no value for {k[0] if len(k) == 1 else k}")
                            v[k] = tb[k].fn(R)
                else:
                    if d["index"] and "dict" in d:
                        keys = list(itertools.product(*[sets[s] for s in d["index"]]))
                        v = {}
                        for k in keys:
                            if k in d["dict"]:
                                v[k] = d["dict"][k].fn(R)
                            elif any(c["name"] == nm and c["key"] == k for c in p_over):
                                v[k] = None
                            else:
                                raise EvalError(f"no value for {k[0] if len(k) == 1 else k}"
                                                + (" (hint: add `param " + nm + "[" + ",".join(map(str, k))
                                                   + "] = ...` to the what-if)" if clauses else ""))
                    else:
                        v = _eval_param(R, d)
                for c in p_over:
                    if c["name"] == nm and c["key"] is not None:
                        if not isinstance(v, dict) or c["key"] not in v:
                            raise EvalError(f"{_keystr(nm, c['key'])} is not an entry of {nm}")
                        v[c["key"]] = c["f"].fn(R)
            except EvalError as e:
                raise EvalError(f"param {nm} (line {d['line']}): {e}") from None
            except (TypeError, ValueError, OverflowError, ZeroDivisionError) as e:
                raise EvalError(f"param {nm} (line {d['line']}): {type(e).__name__}: {e}") from None
            vals = v.values() if isinstance(v, dict) else [v]
            for x in vals:
                if not (isinstance(x, (bool, str)) or _isnum(x)):
                    raise EvalError(f"param {nm} (line {d['line']}): value {x!r} is not a number, bool or atom")
            params[nm] = v
        self.params = params
        # horizon and grid
        try:
            H = (h_over["f"].fn(R) if h_over else (net.horizon.fn(R) if net.horizon else 10))
            if horizon is not None:
                H = horizon
            H = float(H)
        except (EvalError, TypeError, ValueError) as e:
            raise EvalError(f"horizon: {e}") from None
        if not (H > 0) or not math.isfinite(H):
            raise EvalError(f"horizon must be a positive number (got {H})")
        if H > MAX_HORIZON:
            raise EvalError(f"horizon {H:g} exceeds the limit of {MAX_HORIZON:g} {net.unit}s; refused (hint: use a "
                            f"coarser time unit)")
        self.horizon = H
        try:
            g = float(net.grid.fn(R)) if net.grid else None
        except (EvalError, TypeError, ValueError):
            g = None
        if not g or g <= 0:
            g = max(1.0, H / 100) if H >= 10 else H / 10
        n_pts = int(math.floor(H / g + 1e-9)) + 1
        if n_pts > MAX_GRID_POINTS:
            self.notes.append(f"grid {g:g} gives too many points; using {H / (MAX_GRID_POINTS - 1):g}")
            g = H / (MAX_GRID_POINTS - 1)
            n_pts = MAX_GRID_POINTS
        self.grid = g
        times = [round(i * g, 12) for i in range(n_pts)]
        if times[-1] < H - 1e-9 * max(1, H):
            times.append(H)
        self.grid_times = times
        self.h = min(g / 4, 0.1)
        # vars
        self.vars = net.vars
        self.bounds, self.init, self.coerce_kind = {}, {}, {}
        for nm, d in net.vars.items():
            try:
                lo = d["lo"].fn(R) if d["lo"] else None
                hi = d["hi"].fn(R) if d["hi"] else None
                self.bounds[nm] = (lo, hi)
                keys = list(itertools.product(*[sets[s] for s in d["index"]])) if d["index"] else [None]
                vals = {}
                for k in keys:
                    vals[k] = self._init_value(nm, d, k, R)
                for c in i_over:
                    if c["name"] == nm:
                        if c["key"] is None:
                            if "table" in c:
                                for k in keys:
                                    if k in c["table"]:
                                        vals[k] = c["table"][k].fn(R)
                            else:
                                x = c["f"].fn(R)
                                for k in keys:
                                    vals[k] = x
                        else:
                            if c["key"] not in vals:
                                raise EvalError(f"{_keystr(nm, c['key'])} is not an entry of {nm}")
                            vals[c["key"]] = c["f"].fn(R)
                self.init[nm] = vals
            except EvalError as e:
                raise EvalError(f"var {nm} (line {d['line']}): {e}") from None
            except (TypeError, ValueError, OverflowError) as e:
                raise EvalError(f"var {nm} (line {d['line']}): {type(e).__name__}: {e}") from None
        # events -> instance templates
        self.events = events
        self.templates = []
        order = 0
        for evn in [e for e in net.events]:
            ev = events[evn]
            combos = list(itertools.product(*[sets[s] for _, s in ev.idx])) if ev.idx else [()]
            for combo in combos:
                loc = {d: a for (d, _), a in zip(ev.idx, combo)}
                iname = _keystr(evn, combo) if ev.idx else evn
                if evn in disabled or iname in disabled:
                    continue
                R.loc = loc
                try:
                    prio = float(ev.prio.fn(R)) if ev.prio else 0.0
                    sc = scale.get(iname) or scale.get(evn)
                    factor = float(sc["f"].fn(R)) if sc else 1.0
                    if sc and ev.kind in ("at", "immediate"):
                        raise EvalError(f"scale does not apply to '{ev.kind}' events")
                    if factor < 0:
                        raise EvalError("scale factor must be >= 0")
                    tpl = {"name": iname, "ev": ev, "loc": loc, "prio": prio, "order": order, "scale": factor}
                    if ev.kind == "every":
                        per = float(ev.period.fn(R)) / (factor if factor else math.inf)
                        if factor == 0:
                            continue
                        if not per > 0:
                            raise EvalError(f"period must be > 0 (got {per:g})")
                        tpl["period"] = per
                        tpl["start"] = float(ev.start.fn(R)) if ev.start else per
                        tpl["until"] = float(ev.until.fn(R)) if ev.until else math.inf
                    elif ev.kind == "at":
                        tpl["at"] = sorted(float(a.fn(R)) for a in ev.at)
                    elif ev.kind in ("rate", "after") and factor == 0:
                        continue
                except EvalError as e:
                    raise EvalError(f"event {iname} (line {ev.line}): {e}") from None
                except (TypeError, ValueError) as e:
                    raise EvalError(f"event {iname} (line {ev.line}): {type(e).__name__}: {e}") from None
                self.templates.append(tpl)
                order += 1
        # fluid conversion
        self.fluid_events = set()
        self.realvars = set()
        if self.fluid:
            for tpl in self.templates:
                ev = tpl["ev"]
                if ev.kind == "rate" and _const_incr(ev, net):
                    tpl["fluid"] = True
                    self.fluid_events.add(ev.name)
                    for u in ev.updates:
                        self.realvars.add(u["var"])
            kept = sorted({t["ev"].name for t in self.templates if t["ev"].kind == "rate" and not t.get("fluid")})
            if kept:
                self.notes.append(f"fluid mode: rate events kept stochastic (updates are not constant increments): "
                                  f"{', '.join(kept)}")
        self.flows = net.flows
        self.continuous = bool(net.flows) or bool(self.fluid_events)
        self.deterministic = self.fluid and not any(t["ev"].kind == "rate" and not t.get("fluid")
                                                    for t in self.templates)
        # recorded keys
        keys = []
        for nm in net.observes:
            keys.append(("obs", nm, None))
        for nm, d in net.vars.items():
            if d["index"]:
                for k in self.init[nm]:
                    keys.append(("var", nm, k))
            else:
                keys.append(("var", nm, None))
        if len(keys) > MAX_REC_KEYS:
            self.notes.append(f"{len(keys)} recorded quantities; only the first {MAX_REC_KEYS} are kept "
                              f"(observes first)")
            keys = keys[:MAX_REC_KEYS]
        self.rec_keys = keys
        self.rec_names = [k[1] if k[0] == "obs" or k[2] is None else _keystr(k[1], k[2]) for k in keys]

    def _init_value(self, nm, d, k, R):
        t = d["type"]
        if d["init"] is None:
            if t == "fifo":
                return 0
            return {"int": 0, "real": 0.0, "bool": False, "enum": d["enum"][0] if d["enum"] else None}[t]
        if "dict" in d["init"]:
            if k not in d["init"]["dict"]:
                return {"int": 0, "real": 0.0, "bool": False, "fifo": 0}.get(t, d["enum"][0] if d["enum"] else None)
            return d["init"]["dict"][k].fn(R)
        return d["init"]["expr"].fn(R)


def _const_incr(ev, net):
    """True if every update of the event is `x += c` / `x -= c` with c and the index free of state and draws."""
    for u in ev.updates:
        if u["kind"] != "assign" or u["op"] not in ("+=", "-=") or not u["key_static"]:
            return False
        if u["deps"] - {"param"}:
            return False
        if net.vars[u["var"]]["type"] not in ("int", "real"):
            return False
    return True


# ================================================================================================ the simulator
def _stream(base, name, tag):
    h = hashlib.blake2b(f"{base}|{name}|{tag}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(h, "big"))


def _make_coerce(prog, nm, d, R):
    lo, hi = prog.bounds.get(nm, (None, None))
    t = "real" if nm in prog.realvars else d["type"]
    clips, rounded = R.clips, R.rounded
    if t == "int":
        def co(v):
            if isinstance(v, bool):
                v = int(v)
            elif isinstance(v, float):
                if not math.isfinite(v):
                    raise EvalError(f"{nm} got a non-finite value {v}")
                r = round(v)
                if abs(r - v) > 1e-9 * max(1.0, abs(v)):
                    rounded[nm] += 1
                v = int(r)
            elif not isinstance(v, int):
                raise EvalError(f"{nm} is an int var; got {v!r}")
            if lo is not None and v < lo:
                clips[nm] += 1
                v = int(math.ceil(lo))
            elif hi is not None and v > hi:
                clips[nm] += 1
                v = int(math.floor(hi))
            return v
    elif t == "real":
        def co(v):
            if isinstance(v, bool) or not _isnum(v):
                raise EvalError(f"{nm} is a real var; got {v!r}")
            v = float(v)
            if not math.isfinite(v):
                raise EvalError(f"{nm} got a non-finite value {v}")
            if lo is not None and v < lo:
                clips[nm] += 1
                v = float(lo)
            elif hi is not None and v > hi:
                clips[nm] += 1
                v = float(hi)
            return v
    elif t == "bool":
        def co(v):
            if isinstance(v, bool):
                return v
            if _isnum(v):
                return bool(v)
            raise EvalError(f"{nm} is a bool var; got {v!r}")
    elif t == "enum":
        vals = set(d["enum"])

        def co(v):
            if v not in vals:
                raise EvalError(f"{nm} takes one of {d['enum']}; got {v!r}")
            return v
    else:
        def co(v):
            return v
    return co


def _make_update(u, R_coerce):
    if u["kind"] == "proc":
        return u["fn"]
    vn, keyfns, op, val = u["var"], u["keyfns"], u["op"], u["val"].fn
    co = R_coerce[vn]
    opf = {":=": None, "+=": lambda c, v: c + v, "-=": lambda c, v: c - v, "*=": lambda c, v: c * v,
           "/=": _safe_div}[op]
    if keyfns is None:
        if opf is None:
            def up(R):
                R.state[vn] = co(val(R))
        else:
            def up(R):
                R.state[vn] = co(opf(R.state[vn], val(R)))
        return up
    if len(keyfns) == 1:
        k0 = keyfns[0]

        def key(R):
            return (k0(R),)
    else:
        def key(R):
            return tuple(f(R) for f in keyfns)

    def upi(R):
        d = R.state[vn]
        k = key(R)
        if k not in d:
            raise EvalError(f"{vn} has no entry {k[0] if len(k) == 1 else k}")
        v = val(R)
        d[k] = co(v if opf is None else opf(d[k], v))
    return upi


class SeedResult:
    __slots__ = ("grid", "final", "counts", "first", "err", "err_t", "clips", "rounded", "negrate", "leap",
                 "viol", "enabled", "n_fire", "fifo_over", "complete")


class Sim:
    """Runs one Prog for one seed. All stochastic choices come from per-instance streams (CRN)."""

    def __init__(self, prog, seed, index, deadline=math.inf, check=False):
        self.P, self.check = prog, check
        self.base = f"{seed}:{index}"
        self.deadline = deadline

    def run(self):
        P = self.P
        net = P.net
        R = Env()
        R.params, R.sets, R.fluid = P.params, P.sets, P.fluid
        res = SeedResult()
        res.grid = {k: [] for k in P.rec_names}
        res.final, res.counts, res.first, res.err, res.err_t = None, Counter(), {}, None, None
        res.negrate, res.leap, res.viol, res.enabled, res.n_fire = Counter(), False, {}, set(), 0
        res.clips, res.rounded, res.complete = R.clips, R.rounded, False
        res.fifo_over = Counter()
        # state
        for nm, d in net.vars.items():
            R.coerce[nm] = _make_coerce(P, nm, d, R)
        st = {}
        for nm, d in net.vars.items():
            vals = P.init[nm]
            if d["type"] == "fifo":
                lo, hi = P.bounds.get(nm, (None, None))
                cap = int(hi) if hi is not None and math.isfinite(hi) else MAX_FIFO
                cap = min(cap, MAX_FIFO)
                out = {}
                for k, v in vals.items():
                    f = Fifo(cap)
                    f.push(0.0, _num(v, f"initial {nm}") if v else 0)
                    out[k] = f
            else:
                out = {k: R.coerce[nm](v) for k, v in vals.items()}
            st[nm] = out[None] if not d["index"] else out
        R.state = st
        # instances
        rate, after, sched, imm = [], [], [], []
        insts = []
        for tpl in P.templates:
            ev = tpl["ev"]
            I = Inst()
            I.name, I.ev, I.loc, I.prio, I.order, I.kind = tpl["name"], ev, dict(tpl["loc"]), tpl["prio"], \
                tpl["order"], ev.kind
            I.scale = tpl["scale"]
            I.guard = ev.guard.fn if ev.guard else None
            I.rate = ev.rate.fn if ev.rate else None
            I.updates = [_make_update(u, R.coerce) for u in ev.updates]
            I.rng_t = _stream(self.base, I.name, "t")
            I.rng_u = _stream(self.base, I.name, "u")
            I.fluid = bool(tpl.get("fluid"))
            if I.fluid:
                I.incs = []
                for u in ev.updates:
                    R.loc = I.loc
                    key = tuple(f(R) for f in u["keyfns"]) if u["keyfns"] else None
                    dv = float(u["val"].fn(R)) * (1 if u["op"] == "+=" else -1)
                    I.incs.append((u["var"], key, dv))
            if ev.kind == "every":
                I.period, I.start, I.until = tpl["period"], tpl["start"], tpl["until"]
                I.next_t = I.start if I.start <= I.until else math.inf
            elif ev.kind == "at":
                I.at = [a for a in tpl["at"] if 0 <= a]
                I.next_t = I.at[0] if I.at else math.inf
            insts.append(I)
            {"rate": rate, "after": after, "every": sched, "at": sched, "immediate": imm}[ev.kind].append(I)
        stoch_rate = [I for I in rate if not I.fluid]
        fluid_rate = [I for I in rate if I.fluid]
        for I in stoch_rate:
            I.P = I.rng_t.expovariate(1.0)
        # dependency tracking: an expression is re-evaluated only when a var it reads was written (or it reads
        # `now` / fired()); this is an exact optimisation, not an approximation
        for I in insts:
            cs = [c for c in (I.ev.guard, I.ev.rate) if c is not None]
            I.dreads = frozenset().union(*[c.reads for c in cs]) if cs else frozenset()
            I.dalways = any(c.deps & {"time", "fired"} for c in cs)
            I.gval = None
            I.force = True
        dirty_r, dirty_c, dirty_m, dirty_i = set(), set(), set(), set()
        cache_a = [0.0] * len(stoch_rate)
        inv_info = [(c.reads, bool(c.deps & {"time", "fired"})) for c in (inv["f"] for inv in net.invariants)]
        flags = {"r": False, "i": False}
        closure_rng = _stream(self.base, "__closure__", "c")
        hist = deque(maxlen=5) if self.check else None
        H = P.horizon
        eps = 1e-9 * max(1.0, H)
        fired = R.fired
        counts, first = res.counts, res.first
        invs = [(i, inv) for i, inv in enumerate(net.invariants)]
        inv0 = {}
        perf = time.perf_counter

        def en(I):
            if I.guard is None:
                return True
            R.loc = I.loc
            return bool(I.guard(R))

        def fire(I):
            R.loc = I.loc
            R.rng = I.rng_u
            for u in I.updates:
                u(R)
            w = I.ev.writes
            dirty_r.update(w)
            dirty_c.update(w)
            dirty_m.update(w)
            dirty_i.update(w)
            I.force = True
            nm = I.ev.name
            fired[nm] += 1
            counts[nm] += 1
            if nm not in first:
                first[nm] = R.t
            res.n_fire += 1
            res.enabled.add(nm)
            if hist is not None:
                hist.append((R.t, I.name))
            if res.n_fire > MAX_EVENTS:
                raise Budget(f"more than {MAX_EVENTS:,} events in one run; refused (hint: use a coarser time unit, "
                             f"a shorter horizon, or fewer individual events)")

        def closure():
            n = 0
            seq = []
            while True:
                for I in imm:
                    if not DEP_TRACKING or I.force or I.dalways or (I.dreads & dirty_m):
                        I.gval = en(I)
                        I.force = False
                dirty_m.clear()
                cand = [I for I in imm if I.gval]
                if not cand:
                    return
                top = max(I.prio for I in cand)
                cand = [I for I in cand if I.prio == top]
                I = cand[0] if len(cand) == 1 else cand[closure_rng.randrange(len(cand))]
                fire(I)
                n += 1
                if n > MAX_CLOSURE - 40:
                    seq.append(I.name)
                if n > MAX_CLOSURE:
                    raise Livelock(_cycle(seq))

        def clocks():
            t = R.t
            for I in after:
                if DEP_TRACKING and not (I.force or I.dalways or (I.dreads & dirty_c)):
                    continue                 # nothing it reads has changed: its enabling is unchanged
                I.force = False
                e = en(I)
                if e and I.clock is None:
                    res.enabled.add(I.ev.name)
                    I.clock = t + self._delay(I, R)
                elif not e and I.clock is not None:
                    I.clock = None
            dirty_c.clear()

        def check_invs():
            first_check = not flags["i"] or not DEP_TRACKING
            flags["i"] = True
            for i, inv in invs:
                reads, always = inv_info[i]
                if not (first_check or always or (reads & dirty_i)):
                    continue
                ok = True
                try:
                    R.loc = {}
                    v = inv["f"].fn(R)
                    if inv["conserved"]:
                        if i not in inv0:
                            inv0[i] = v
                        ok = abs(v - inv0[i]) <= 1e-9 * max(1.0, abs(inv0[i]))
                    else:
                        ok = bool(v)
                except (EvalError, TypeError, ValueError, ZeroDivisionError) as e:
                    raise EvalError(f"invariant '{inv['text']}': {e}") from None
                if not ok:
                    c, t0, last = res.viol.get(i, (0, R.t, list(hist) if hist is not None else []))
                    res.viol[i] = (c + 1, t0, last)
            dirty_i.clear()

        def record():
            R.loc = {}
            g = res.grid
            for (kind, nm, k), key in zip(P.rec_keys, P.rec_names):
                if kind == "obs":
                    v = net.observes[nm]["f"].fn(R)
                else:
                    v = R.state[nm] if k is None else R.state[nm][k]
                    if isinstance(v, Fifo):
                        v = v.n
                g[key].append(v)

        # continuous part: slots and derivative
        slots = []
        flow_terms = []
        if P.continuous:
            seen = set()
            for f in P.flows:
                d = net.vars[f["var"]]
                keys = list(R.state[f["var"]].keys()) if d["index"] else [None]
                for k in keys:
                    loc = dict(zip(f["locs"], k)) if k else {}
                    flow_terms.append(((f["var"], k), f["f"].fn, loc))
                    if (f["var"], k) not in seen:
                        seen.add((f["var"], k))
                        slots.append((f["var"], k))
            for I in fluid_rate:
                for vn, k, _ in I.incs:
                    if (vn, k) not in seen:
                        seen.add((vn, k))
                        slots.append((vn, k))
                    if net.vars[vn]["type"] == "int" and not isinstance(R.state[vn] if k is None else
                                                                        R.state[vn][k], float):
                        if k is None:
                            R.state[vn] = float(R.state[vn])
                        else:
                            R.state[vn][k] = float(R.state[vn][k])

        slot_vars = {s[0] for s in slots}

        def sget(s):
            return R.state[s[0]] if s[1] is None else R.state[s[0]][s[1]]

        def sset(s, v):
            if s[1] is None:
                R.state[s[0]] = v
            else:
                R.state[s[0]][s[1]] = v

        def deriv():
            dd = dict.fromkeys(slots, 0.0)
            for s, f, loc in flow_terms:
                R.loc = loc
                dd[s] += float(f(R))
            for I in fluid_rate:
                if not en(I):
                    continue
                R.loc = I.loc
                a = I.rate(R) * I.scale
                if a < 0:
                    res.negrate[I.ev.name] += 1
                    continue
                if a:
                    for vn, k, dv in I.incs:
                        dd[(vn, k)] += a * dv
            return dd

        def integrate(t0, dt):
            y0 = {s: sget(s) for s in slots}
            R.t = t0
            k1 = deriv()
            for s in slots:
                sset(s, y0[s] + dt / 2 * k1[s])
            R.t = t0 + dt / 2
            k2 = deriv()
            for s in slots:
                sset(s, y0[s] + dt / 2 * k2[s])
            k3 = deriv()
            for s in slots:
                sset(s, y0[s] + dt * k3[s])
            R.t = t0 + dt
            k4 = deriv()
            for s in slots:
                v = y0[s] + dt / 6 * (k1[s] + 2 * k2[s] + 2 * k3[s] + k4[s])
                if not math.isfinite(v):
                    raise EvalError(f"{_keystr(*s)} became non-finite during integration")
                sset(s, R.coerce[s[0]](v))
            for d in (dirty_r, dirty_c, dirty_m, dirty_i):
                d.update(slot_vars)

        def rates():
            out = cache_a
            valid = flags["r"] and DEP_TRACKING
            flags["r"] = True
            for j, I in enumerate(stoch_rate):
                if valid and not I.dalways and not (I.dreads & dirty_r):
                    continue
                if I.guard is not None:
                    R.loc = I.loc
                    if not I.guard(R):
                        out[j] = 0.0
                        continue
                R.loc = I.loc
                a = I.rate(R)
                if not _isnum(a) and not isinstance(a, bool):
                    raise EvalError(f"rate of {I.name} is not a number ({a!r})")
                a = float(a) * I.scale
                if a < 0 or a != a:
                    res.negrate[I.ev.name] += 1
                    a = 0.0
                if a > 0:
                    res.enabled.add(I.ev.name)
                out[j] = a
            dirty_r.clear()
            return out

        grid = P.grid_times
        gi = 0
        t = 0.0
        R.t = 0.0
        try:
            closure()
            clocks()
            check_invs()
            # the t = 0 grid point is recorded after any events scheduled at t = 0 (initialisations) have fired
            if not any(I.next_t <= eps for I in sched) and not any(I.clock is not None and I.clock <= eps for I in after):
                while gi < len(grid) and grid[gi] <= t + eps:
                    record()
                    gi += 1
            leap = False
            a0 = rates()
            if sum(a0) * H > LEAP_THRESHOLD:
                leap = True
            n_rate = 0
            tau = P.h
            it = 0
            same_t, last_t = 0, -1.0
            while True:
                it += 1
                if it % 512 == 0 and perf() > self.deadline:
                    raise Budget("time budget exceeded during this run")
                tc = min((I.clock for I in after if I.clock is not None), default=math.inf)
                ts = min((I.next_t for I in sched), default=math.inf)
                tg = grid[gi] if gi < len(grid) else math.inf
                tlim = min(tc, ts, tg, H)
                if P.continuous or leap:
                    tlim = min(tlim, t + tau)
                a = rates()
                kfire = None
                if not leap:
                    best = math.inf
                    for j, I in enumerate(stoch_rate):
                        if a[j] > 0:
                            dtj = (I.P - I.T) / a[j]
                            if dtj < best:
                                best, kfire = dtj, j
                    if kfire is not None and t + best < tlim:
                        tnext = t + best
                    else:
                        kfire = None
                        tnext = tlim
                else:
                    tnext = tlim
                dt = tnext - t
                if dt < 0:
                    dt, tnext = 0.0, t
                if P.continuous and dt > 0:
                    integrate(t, dt)
                if not leap:
                    for j, I in enumerate(stoch_rate):
                        if a[j]:
                            I.T += a[j] * dt
                t = tnext
                R.t = t
                if t == last_t:
                    same_t += 1
                    if same_t > MAX_CLOSURE:
                        raise Livelock("timed events keep firing at the same instant (zero delays?)")
                else:
                    same_t, last_t = 0, t
                changed = P.continuous
                if leap and dt > 0:
                    todo = []
                    for j, I in enumerate(stoch_rate):
                        if a[j] > 0:
                            n = _poisson(I.rng_t, a[j] * dt)
                            todo += [I] * n
                    if todo:
                        closure_rng.shuffle(todo)
                        for I in todo:
                            if en(I):
                                fire(I)
                                n_rate += 1
                        changed = True
                elif kfire is not None:
                    I = stoch_rate[kfire]
                    fire(I)
                    I.P += I.rng_t.expovariate(1.0)
                    n_rate += 1
                    changed = True
                    if n_rate > LEAP_THRESHOLD and not leap:
                        leap = True
                if kfire is None:
                    due = [I for I in after if I.clock is not None and I.clock <= t + eps] + \
                          [I for I in sched if I.next_t <= t + eps]
                    if due:
                        due.sort(key=lambda I: (-I.prio, I.order))
                        for I in due:
                            if I.kind == "after":
                                if I.clock is None:
                                    continue                     # discarded by an earlier simultaneous firing
                                I.clock = None
                                fire(I)
                            else:
                                if I.kind == "every":
                                    I.k += 1
                                    nt = I.start + I.k * I.period
                                    I.next_t = nt if nt <= I.until + eps else math.inf
                                else:
                                    I.at_i += 1
                                    I.next_t = I.at[I.at_i] if I.at_i < len(I.at) else math.inf
                                if en(I):
                                    fire(I)
                            closure()
                            clocks()
                        changed = True
                if changed:
                    closure()
                    clocks()
                    if invs:
                        check_invs()
                while gi < len(grid) and grid[gi] <= t + eps:
                    record()
                    gi += 1
                if t >= H - eps:
                    break
            res.leap = leap
            res.complete = True
            res.final = {k: v[-1] for k, v in res.grid.items() if v}
        except Livelock as e:
            res.err = f"livelock: {e}"
            res.err_t = R.t
        except (EvalError, Budget) as e:
            res.err = str(e)
            res.err_t = R.t
        except (TypeError, ValueError, ZeroDivisionError, OverflowError, KeyError, RecursionError) as e:
            res.err = f"{type(e).__name__}: {e}"
            res.err_t = R.t
        for nm, d in net.vars.items():
            if d["type"] == "fifo":
                fs = [R.state[nm]] if not d["index"] else list(R.state[nm].values())
                for f in fs:
                    if f.over:
                        res.fifo_over[nm] += f.over
        return res

    def _delay(self, I, R):
        d = I.ev.delay
        R.loc = I.loc
        R.rng = I.rng_t
        fl = R.fluid
        kind = d["dist"]
        if kind == "const":
            v = d["f"].fn(R)
        else:
            args = [float(_num(a.fn(R), f"delay {kind}")) for a in d["args"]]
            rng = I.rng_t
            if kind == "exp":
                m = args[0]
                v = m if fl or m <= 0 else rng.expovariate(1.0 / m)
            elif kind == "uniform":
                v = (args[0] + args[1]) / 2 if fl else rng.uniform(args[0], args[1])
            elif kind == "normal":
                m, s = args
                if fl or s <= 0:
                    v = max(0.0, m)
                else:
                    v = -1
                    for _ in range(50):
                        v = rng.gauss(m, s)
                        if v >= 0:
                            break
                    v = max(0.0, v)
            elif kind == "lognormal":
                m, s = args
                if fl or s <= 0 or m <= 0:
                    v = max(0.0, m)
                else:
                    sig2 = math.log(1 + (s * s) / (m * m))
                    v = rng.lognormvariate(math.log(m) - sig2 / 2, math.sqrt(sig2))
            elif kind == "erlang":
                k, m = int(round(args[0])), args[1]
                if k < 1:
                    raise EvalError("erlang needs k >= 1")
                v = m if fl or m <= 0 else rng.gammavariate(k, m / k)
            else:
                raise EvalError(f"unknown delay {kind}")
        v = _num(v, f"delay of {I.name}")
        if v < 0 or v != v:
            raise EvalError(f"delay of {I.name} is negative or undefined ({v})")
        return float(v) / I.scale


def _cycle(seq):
    s = seq[-40:]
    for p in range(1, 21):
        if len(s) >= 2 * p and s[-p:] == s[-2 * p:-p]:
            return (f"immediate events keep firing in zero time; cycle: {' -> '.join(s[-p:])} (hint: make each "
                    f"immediate event's update falsify its own guard, e.g. add `and not <flag>` to the condition)")
    return f"immediate events keep firing in zero time; last: {' -> '.join(s[-8:])}"


# ================================================================================================ P-invariants
def structural(net):
    """Incidence matrix over constant-increment count/level vars and baseline event instances; P-invariant basis;
    conserved-claim proofs. Returns a dict."""
    out = {"invariants": [], "excluded": {}, "unchanged": [], "claims": [], "sources": [], "sinks": [], "error": None}
    try:
        prog = Prog(net)
    except EvalError as e:
        out["error"] = str(e)
        return out
    R = Env()
    R.params, R.sets = prog.params, prog.sets
    # declaration order, not a set: excl's order reaches the reasoner via structural_lines() (QA-0194)
    flows = list(dict.fromkeys(f["var"] for f in net.flows))
    cols = {}                       # instance -> {(var, key): Fraction}
    excl = {}
    for vn in flows:
        excl[vn] = "has a flow"
    for vn, d in net.vars.items():
        if d["type"] not in ("int", "real"):
            excl.setdefault(vn, f"{d['type']} var")
    for tpl in prog.templates:
        ev = tpl["ev"]
        col = {}
        for u in ev.updates:
            vn = u["var"]
            if vn is None:
                continue
            if u["kind"] != "assign":
                excl.setdefault(vn, f"changed by {u['text'][:30]} ({ev.name})")
                continue
            if u["op"] not in ("+=", "-=") or (u["deps"] - {"param"}) or not u["key_static"]:
                excl.setdefault(vn, f"non-constant update `{u['text'][:40]}` ({ev.name})")
                continue
            R.loc = tpl["loc"]
            try:
                key = tuple(f(R) for f in u["keyfns"]) if u["keyfns"] else None
                v = u["val"].fn(R)
                q = Fraction(str(float(v))).limit_denominator(10 ** 9) if not isinstance(v, int) else Fraction(v)
            except Exception:
                excl.setdefault(vn, f"increment of {ev.name} cannot be evaluated")
                continue
            col[(vn, key)] = col.get((vn, key), 0) + (q if u["op"] == "+=" else -q)
        cols[tpl["name"]] = col
    places = []
    for vn, d in net.vars.items():
        if vn in excl:
            continue
        keys = list(prog.init[vn].keys())
        for k in keys:
            places.append((vn, k))
    changed = {p for c in cols.values() for p, v in c.items() if v}
    out["unchanged"] = [_keystr(*p) for p in places if p not in changed]
    places = [p for p in places if p in changed]
    out["excluded"] = excl
    names = list(cols)
    if places:
        # left null space: y^T C = 0  <=>  C^T y = 0 ; rows = events, columns = places
        M = [[cols[e].get(p, Fraction(0)) for p in places] for e in names]
        basis = _nullspace(M, len(places))
        for y in basis:
            terms = [(c, p) for c, p in zip(y, places) if c]
            init = sum(c * Fraction(str(float(prog.init[p[0]][p[1]]))) for c, p in terms)
            out["invariants"].append({"terms": [(c, _keystr(*p)) for c, p in terms], "value": init})
        # sources / sinks: events with a nonzero net change of the unweighted total
        for e in names:
            s = sum(cols[e].get(p, 0) for p in places)
            if s > 0:
                out["sources"].append(e)
            elif s < 0:
                out["sinks"].append(e)
    for inv in net.invariants:
        if not inv["conserved"]:
            continue
        lin = inv.get("linear")
        claim = {"text": inv["text"], "status": None, "detail": ""}
        if lin is None:
            claim["status"] = "not linear"
            claim["detail"] = "only linear totals can be proved; it is checked at run time"
        else:
            bad = [vn for (vn, k) in lin if vn in excl]
            if bad:
                claim["status"] = "cannot prove"
                claim["detail"] = "; ".join(f"{vn}: {excl[vn]}" for vn in sorted(set(bad))) + \
                    " (checked at run time only)"
            else:
                breaks = []
                for e in names:
                    s = sum(Fraction(str(float(c))) * cols[e].get(p, 0) for p, c in lin.items())
                    if s:
                        breaks.append(f"{e} ({'+' if s > 0 else ''}{_frac(s)})")
                if breaks:
                    claim["status"] = "broken"
                    claim["detail"] = "changed by " + ", ".join(_compress(breaks))
                else:
                    claim["status"] = "proved"
        out["claims"].append(claim)
    return out


def _nullspace(M, n):
    """Basis of {y : M y = 0} over the rationals (M: list of rows of Fractions, n columns)."""
    A = [list(r) for r in M if any(r)]
    piv_cols, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, len(A)) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        piv_cols.append(c)
        r += 1
        if r == len(A):
            break
    free = [c for c in range(n) if c not in piv_cols]
    basis = []
    for fc in free:
        y = [Fraction(0)] * n
        y[fc] = Fraction(1)
        for i, pc in enumerate(piv_cols):
            y[pc] = -A[i][fc]
        den = 1
        for x in y:
            den = den * x.denominator // math.gcd(den, x.denominator)
        y = [x * den for x in y]
        g = 0
        for x in y:
            g = math.gcd(g, int(x))
        if g > 1:
            y = [x / g for x in y]
        if next(x for x in y if x) < 0:
            y = [-x for x in y]
        basis.append(y)
    return basis


def _frac(q):
    q = Fraction(q)
    return str(q.numerator) if q.denominator == 1 else f"{float(q):.4g}"


def _compress(names):
    """['F[a] (+1)', 'F[b] (+1)'] -> ['F[*] (+1)'] when all instances of an event agree."""
    groups = {}
    for s in names:
        m = re.match(r"^([A-Za-z_]\w*)(\[[^\]]*\])?(.*)$", s)
        base, rest = m.group(1), m.group(3)
        groups.setdefault((base, rest), []).append(m.group(2))
    out = []
    for (base, rest), idx in groups.items():
        if len(idx) > 1 and all(idx):
            out.append(f"{base}[*]{rest}")
        else:
            out += [f"{base}{i or ''}{rest}" for i in idx]
    return out


def _lin_text(terms):
    s = ""
    for c, name in terms:
        c = Fraction(c)
        mag = abs(c)
        t = name if mag == 1 else f"{_frac(mag)}*{name}"
        s += (" - " if c < 0 else " + ") + t if s else ("-" if c < 0 else "") + t
    return s


def structural_lines(net, sres=None):
    s = sres or structural(net)
    out = []
    if s["error"]:
        return [f"P-invariants: not computed ({s['error']})"]
    if s["invariants"]:
        groups = {}                      # the same pattern over every index value is shown once
        for i in s["invariants"]:
            text = _lin_text(i["terms"])
            pat = re.sub(r"\[[^\]]*\]", "[*]", text)
            idx = sorted(set(re.findall(r"\[([^\]]*)\]", text)))
            key = pat if len(idx) == 1 else text
            groups.setdefault(key, []).append((idx[0] if len(idx) == 1 else None, _frac(i["value"])))
        parts = []
        for pat, members in groups.items():
            if len(members) > 1:
                parts.append(f"for each index [*] in ({', '.join(m[0] for m in members)}): {pat} = "
                             + ("/".join(m[1] for m in members) if len({m[1] for m in members}) > 1 else members[0][1]))
            else:
                parts.append(f"{members[0][0] and pat.replace('[*]', '[' + members[0][0] + ']') or pat} = "
                             f"{members[0][1]}")
        out.append("P-invariants (conserved weighted totals of constant-increment vars): " + "; ".join(parts[:8])
                   + (f" (+{len(parts) - 8} more)" if len(parts) > 8 else ""))
    else:
        out.append("P-invariants: none" + (f" (sources: {', '.join(_compress(s['sources'])[:8])}"
                                           if s["sources"] else "")
                                         + (f"; sinks: {', '.join(_compress(s['sinks'])[:8])}" if s["sinks"] else "")
                                         + (")" if s["sources"] else ""))
    if s["invariants"] and (s["sources"] or s["sinks"]):
        out.append("  sources (net +): " + (", ".join(_compress(s["sources"])[:8]) or "none") + "; sinks (net -): "
                   + (", ".join(_compress(s["sinks"])[:8]) or "none"))
    if s["excluded"]:
        out.append("  not in the matrix: " + "; ".join(f"{k} ({v})" for k, v in list(s["excluded"].items())[:8])
                   + (" ..." if len(s["excluded"]) > 8 else ""))
    for c in s["claims"]:
        out.append(f"  conserved {c['text']}: {c['status'].upper()}" + (f" - {c['detail']}" if c["detail"] else ""))
    return out


# ================================================================================================ bounded check
def _enumerable(net):
    why = []
    for vn, d in net.vars.items():
        if d["type"] in ("fifo", "real"):
            why.append(f"{vn} is {d['type']}")
        elif d["type"] == "int" and (d["lo"] is None or d["hi"] is None):
            why.append(f"{vn} has no [lo, hi] bounds")
    if net.flows:
        why.append("flows")
    for ev in net.events.values():
        for u in ev.updates:
            if "draw" in u["deps"]:
                why.append(f"{ev.name} draws random numbers")
                break
        for c in [ev.guard] + [u.get("val") for u in ev.updates if u["kind"] == "assign"]:
            if c is not None and (("time" in c.deps) or ("fired" in c.deps)):
                why.append(f"{ev.name} uses now/fired()")
                break
    return why


def bfs_check(net, depth, prog):
    """Breadth-first search of the untimed transition system (every enabled event is a choice)."""
    R = Env()
    R.params, R.sets = prog.params, prog.sets
    for nm, d in net.vars.items():
        R.coerce[nm] = _make_coerce(prog, nm, d, R)
    insts = []
    for tpl in prog.templates:
        I = Inst()
        I.name, I.ev, I.loc, I.prio = tpl["name"], tpl["ev"], tpl["loc"], tpl["prio"]
        I.guard = I.ev.guard.fn if I.ev.guard else None
        I.updates = [_make_update(u, R.coerce) for u in I.ev.updates]
        insts.append(I)
    names = list(net.vars)

    def freeze(st):
        return tuple(tuple(sorted(st[n].items(), key=repr)) if isinstance(st[n], dict) else st[n] for n in names)

    def thaw(st):
        return {n: (dict(v) if isinstance(v, dict) else v) for n, v in st.items()}

    s0 = {nm: (prog.init[nm][None] if not d["index"] else dict(prog.init[nm])) for nm, d in net.vars.items()}
    s0 = {nm: (R.coerce[nm](v) if not isinstance(v, dict) else {k: R.coerce[nm](x) for k, x in v.items()})
          for nm, v in s0.items()}
    invs = [inv for inv in net.invariants]

    def violated(st):
        R.state = st
        R.loc = {}
        for inv in invs:
            try:
                v = inv["f"].fn(R)
                if inv["conserved"]:
                    R.state = s0
                    v0 = inv["f"].fn(R)
                    R.state = st
                    if abs(v - v0) > 1e-9 * max(1, abs(v0)):
                        return f"conserved {inv['text']} changed ({v0} -> {v})"
                elif not v:
                    return f"invariant {inv['text']} is false"
            except (EvalError, TypeError, ValueError, ZeroDivisionError) as e:
                return f"invariant {inv['text']} cannot be evaluated ({e})"
        return None

    v = violated(s0)
    if v:
        return {"ok": False, "trace": ["init"], "msg": v, "states": 1, "covered": False}
    seen = {freeze(s0): None}
    parent = {}
    frontier = [s0]
    covered = True
    for dep in range(depth):
        nxt = []
        for st in frontier:
            fz = freeze(st)
            for I in insts:
                R.state = st
                R.loc = I.loc
                try:
                    if I.guard is not None and not I.guard(R):
                        continue
                    ns = thaw(st)
                    R.state = ns
                    R.clips.clear()
                    R.rng = random.Random(0)
                    for u in I.updates:
                        u(R)
                except (EvalError, TypeError, ValueError, ZeroDivisionError) as e:
                    path = _path(parent, fz) + [I.name]
                    return {"ok": False, "trace": path, "msg": f"evaluation error: {e}", "states": len(seen),
                            "covered": False}
                nf = freeze(ns)
                if nf in seen:
                    continue
                seen[nf] = None
                parent[nf] = (fz, I.name)
                msg = violated(ns)
                if R.clips:
                    msg = msg or ("bound reached: " + ", ".join(f"{k} was clipped" for k in R.clips))
                if msg:
                    return {"ok": False, "trace": _path(parent, nf), "msg": msg, "states": len(seen),
                            "covered": False, "state": {k: ns[k] for k in names[:12]}}
                nxt.append(ns)
                if len(seen) > BFS_MAX_STATES:
                    return {"ok": True, "trace": None, "msg": f"stopped at {BFS_MAX_STATES:,} states", "states": len(seen),
                            "covered": False, "depth": dep + 1}
        frontier = nxt
        if not frontier:
            break
    else:
        covered = not frontier
    return {"ok": True, "trace": None, "msg": "", "states": len(seen), "covered": not frontier, "depth": depth}


def _path(parent, fz):
    out = []
    while fz in parent:
        fz, nm = parent[fz]
        out.append(nm)
    return ["init"] + out[::-1]


# ================================================================================================ formatting
def _g(v):
    """Readable number: integers in full, otherwise 4 significant digits without exponent below 1e6."""
    if isinstance(v, bool) or not _isnum(v):
        return str(v)
    if isinstance(v, int) or (v == int(v) and abs(v) < 1e15):
        return str(int(v))
    a = abs(v)
    if a >= 1e6 or a < 1e-4:
        return f"{v:.4g}"
    d = max(0, 3 - math.floor(math.log10(a)))
    s = f"{v:.{d}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def summarize(vals):
    """A4's summary layout (mean sd | quantiles | min max (n)) with readable numbers."""
    ok = [v for v in vals if v is not None]
    if not ok:
        return "no values"
    if all(isinstance(v, bool) for v in ok):
        return f"P(true)={sum(ok) / len(ok):.3f} (n={len(ok)})"
    nums = [float(v) for v in ok if _isnum(v)]
    if len(nums) == len(ok):
        xs = sorted(nums)
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
        return (f"mean {_g(m)} sd {_g(sd)} | p05 {_g(_quant(xs, .05))} p25 {_g(_quant(xs, .25))} "
                f"p50 {_g(_quant(xs, .5))} p75 {_g(_quant(xs, .75))} p95 {_g(_quant(xs, .95))} | "
                f"min {_g(xs[0])} max {_g(xs[-1])} (n={len(xs)})")
    cnt = Counter(str(v) for v in ok)
    return ", ".join(f"{k!r}: {c / len(ok):.3f}" for k, c in cnt.most_common(8)) + f" (n={len(ok)})"


def _compact(vals):
    ok = [v for v in vals if v is not None]
    if not ok:
        return "no values"
    if len(ok) == 1:
        return _summ(ok)
    if all(type(v) is type(ok[0]) and v == ok[0] for v in ok):
        return f"{_g(ok[0])} on all {len(ok)}"
    if all(isinstance(v, bool) for v in ok):
        return f"P(true)={sum(ok) / len(ok):.3f}"
    if all(_isnum(v) for v in ok):
        xs = sorted(float(v) for v in ok)
        return f"mean {_g(sum(xs) / len(xs))} [p05 {_g(_quant(xs, .05))}, p95 {_g(_quant(xs, .95))}]"
    return summarize(vals)


def _s3(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "T" if v else "F"
    if not _isnum(v):
        return str(v)
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return str(int(round(v)))
    return f"{v:.3g}"


def _rle(tokens):
    out, i = [], 0
    while i < len(tokens):
        j = i
        while j + 1 < len(tokens) and tokens[j + 1] == tokens[i]:
            j += 1
        n = j - i + 1
        out.append(f"{tokens[i]}x{n}" if n >= 3 else " ".join([tokens[i]] * n))
        i = j + 1
    return " ".join(out)


def _signed(v):
    s = _g(float(v))
    return s if s.startswith("-") else "+" + s


def _summ(vals):
    ok = [v for v in vals if v is not None]
    if len(ok) == 1:
        v = ok[0]
        return (("true" if v else "false") if isinstance(v, bool) else _g(v) if _isnum(v) else str(v)) + " (single run)"
    if len(ok) > 1 and all(type(v) is type(ok[0]) and v == ok[0] for v in ok):
        return f"{_g(ok[0]) if _isnum(ok[0]) else repr(ok[0])} on all {len(ok)} seeds"
    return summarize(vals)


def shape(times, series):
    pts = [(t, v) for t, v in zip(times, series) if v is not None]
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
    w = max(1, len(vals) // 12)
    sm = [sum(vals[max(0, i - w):i + w + 1]) / len(vals[max(0, i - w):i + w + 1]) for i in range(len(vals))]
    eps = 0.01 * (vmax - vmin)
    signs = [1 if d > 0 else -1 for d in (b - a for a, b in zip(sm, sm[1:])) if abs(d) > eps]
    changes = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    if changes == 0:
        kind = "monotone " + ("rising" if (signs and signs[0] > 0) else "falling")
    elif changes == 1:
        kind = "single-peak (rises, then falls)" if signs[0] > 0 else "single-trough (falls, then rises)"
    else:
        kind = f"oscillating ({changes} turns)"
    onset = next((t for t, v in pts if abs(v - v0) > 0.05 * (vmax - vmin)), None)
    s = (f"start {_s3(v0)} (t={_g(t0)}) -> end {_s3(vn)} (t={_g(tn)}); max {_s3(vmax)} at t={_g(tmax)}, "
         f"min {_s3(vmin)} at t={_g(tmin)}; {kind}; onset t={_g(onset) if onset is not None else '-'}")
    total = vn - v0
    if abs(total) > 1e-9 * scale and changes == 0:
        def reach(f):
            for t, v in pts:
                if (v - v0) / total >= f:
                    return _g(t)
            return "-"
        s += f"; 50% of the change by t={reach(.5)}, 90% by t={reach(.9)}"
    return s


def _sample_idx(times, series_list):
    n = len(times)
    if n <= SERIES_POINTS:
        return list(range(n))
    idx = {round((n - 1) * i / (SERIES_POINTS - 1)) for i in range(SERIES_POINTS)}
    idx.add(n - 1)
    for s in series_list:
        pts = [(i, v) for i, v in enumerate(s) if v is not None]
        if pts:
            idx.add(max(pts, key=lambda p: p[1])[0])
            idx.add(min(pts, key=lambda p: p[1])[0])
    return sorted(idx)


def _numeric(vs):
    return [float(v) if not isinstance(v, bool) else float(v) for v in vs if v is not None and
            (_isnum(v) or isinstance(v, bool))]


def _parse_run_events(events, comp_fn):
    out = []
    if events is None:
        return out
    if isinstance(events, (str, dict)):
        events = [events]
    if not isinstance(events, list):
        return [("events", None, "events must be a list of expressions or {expr, when}")]
    for ev in events[:12]:
        when = "final"
        src = ev.get("expr") if isinstance(ev, dict) else ev
        if isinstance(ev, dict):
            when = ev.get("when", "final")
        if isinstance(when, str) and when not in ("final", "any", "all"):
            try:
                when = float(when)
            except ValueError:
                out.append((str(src), None, "when must be final, any, all or a time"))
                continue
        if not (when in ("final", "any", "all") or _isnum(when)):
            out.append((str(src), None, "when must be final, any, all or a time"))
            continue
        try:
            out.append((str(src), comp_fn(str(src)), when))
        except NetError as e:
            out.append((str(src), None, str(e)))
    return out


# ================================================================================================ the arm
class G1C4Arm(Arm):
    name = "G1C4"
    buildable = True
    executable = True
    guide_file = "g1c4_guide.md"
    mutating_tools = ("submit_model", "patch_model")
    custom_episode = True
    TIME_BUDGET = TIME_BUDGET

    def guide(self):
        p = ROOT / "tournament" / "prompts" / self.guide_file
        return p.read_text(encoding="utf-8") if p.exists() else ""

    # ------------------------------------------------------------------ schemas
    @staticmethod
    def _run_schema():
        return {
            "name": {"type": "string", "description": "label of this run"},
            "whatif": {"type": "string", "description": "name of a stored `whatif` line to apply"},
            "changes": {"type": "array", "items": {"type": "string"},
                        "description": "what-if clauses for this run only, e.g. [\"param rate_in = 8\", "
                                       "\"disable Audit\", \"scale Serve 0.8\", \"init q = 5\", \"set Desk = {a, b, c}\", "
                                       "\"horizon 48\", \"replace event ...\"]"},
            "horizon": {"type": "number", "description": "run length in time units (default: the model's horizon)"},
            "seeds": {"type": "integer", "description": f"random runs (default {DEFAULT_SEEDS}, max {MAX_SEEDS})"},
            "seed": {"type": "integer", "description": "base seed (default 0)"},
            "mode": {"type": "string", "enum": ["stochastic", "fluid", "both"],
                     "description": "stochastic (default), fluid (average-flow ODE) or both (reports the discrepancy)"},
            "events": {"type": "array", "description": "event expressions over observed names and vars, e.g. "
                       "\"waiting > 10\" or {expr, when: final|any|all|<time>}; 'any' gives first-passage times"},
            "keys": {"type": "array", "items": {"type": "string"}, "description": "observed names/vars to show first"},
            "compare": {"type": "boolean", "description": "compare with the baseline under common random numbers "
                        "(default true)"},
        }

    def _runs_prop(self):
        return {"type": "array", "items": {"type": "object", "properties": self._run_schema()},
                "description": f"up to {MAX_RUNS} runs, each compared with the baseline"}

    def tools(self):
        return [
            _tool("submit_model", "Store the net (from `source`, or else from the last ```net fenced block of this "
                  "response), check it (holes, warnings, conserved totals) and immediately run the baseline, every "
                  "stored whatif (up to 4) and each entry of `runs`.",
                  {"source": {"type": "string", "description": "the whole net document (optional when it is in a "
                              "```net block of the same response)"}, "runs": self._runs_prop()}),
            _tool("patch_model", "Replace one exact, unique occurrence of `old` with `new` in the stored net, then "
                  "check and run it like submit_model.",
                  {"old": {"type": "string"}, "new": {"type": "string"}, "runs": self._runs_prop()}, ["old", "new"]),
            _tool("run_scenario", "Run the stored net: either one run's fields or `runs` (up to 4). Runs with a "
                  "whatif or changes are compared with the baseline under common random numbers (paired "
                  "differences).", {**self._run_schema(), "runs": self._runs_prop()}),
            _tool("check_invariants", "Check the invariants: structural conserved totals (P-invariants), then a "
                  "bounded search of all event orders (small discrete models) or a sampled check over random runs.",
                  {"depth": {"type": "integer", "description": "search depth (default 8)"},
                   "runs": {"type": "integer", "description": "sampled runs (default 100)"}}),
            _tool("show_model", "Show the numbered net, the parsed summary, the holes and the P-invariants.", {}),
        ]

    def probe_tools(self):
        return [t for t in self.tools() if t["function"]["name"] in ("run_scenario", "check_invariants",
                                                                       "show_model")]

    # ------------------------------------------------------------------ state / export
    def new_state(self):
        return ArmState(rep={"source": ""})

    def load(self, rep, given_gold=False):
        st = self.new_state()
        if isinstance(rep, dict):
            src = rep.get("source")
            st.rep = {"source": src if isinstance(src, str) else ""}
        elif isinstance(rep, str):
            st.rep = {"source": rep}
        st.given_gold = given_gold
        return st

    def _src(self, state):
        s = state.rep.get("source")
        return s if isinstance(s, str) else ""

    def _net(self, state):
        src = self._src(state)
        cache = state.__dict__.setdefault("_netcache", {})
        if cache.get("src") != src:
            cache.clear()
            cache["src"] = src
            cache["net"] = Net(src)
        return cache["net"]

    def _structural(self, state):
        cache = state.__dict__.setdefault("_netcache", {})
        net = self._net(state)
        if "struct" not in cache:
            cache["struct"] = structural(net)
        return cache["struct"]

    def summary(self, net):
        try:
            prog = Prog(net)
            pvals = {k: ({",".join(map(str, kk)): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
                     for k, v in prog.params.items()}
        except EvalError:
            pvals = {k: None for k in net.params}
        return {"sets": {k: v["atoms"] for k, v in net.sets.items()}, "params": pvals,
                "vars": {k: {"type": v["type"] if v["type"] != "enum" else "{" + ", ".join(map(str, v["enum"])) + "}",
                             "index": v["index"]} for k, v in net.vars.items()},
                "events": [{"name": e.name, "timing": e.kind, "index": [s for _, s in e.idx]}
                           for e in net.events.values()],
                "observes": list(net.observes), "whatifs": {k: v["text"] for k, v in net.whatifs.items()},
                "unit": net.unit}

    def export(self, state):
        src = self._src(state)
        net = self._net(state)
        s = self._structural(state)
        return {"source": src, "summary": self.summary(net),
                "p_invariants": [f"{_lin_text(i['terms'])} = {_frac(i['value'])}" for i in s["invariants"]]}

    def render(self, state):
        src = self._src(state)
        if not src.strip():
            return "(no net yet)"
        net = self._net(state)
        doc = "\n".join(f"{i + 1:4d}| {l}" for i, l in enumerate(src.splitlines()))
        return (doc + "\n\n" + "\n".join(self._summary_lines(net) + self._hole_lines(net)
                                         + structural_lines(net, self._structural(state))))[:60000]

    def _summary_lines(self, net):
        out = ["PARSED: unit " + net.unit + "; sets " + (", ".join(f"{k}={{{', '.join(map(str, v['atoms']))}}}"
                                                               for k, v in net.sets.items()) or "none")]
        try:
            prog = Prog(net)
            out.append("params: " + (", ".join(f"{k}={_short(v)}" for k, v in prog.params.items()) or "none"))
            out.append(f"horizon {_g(prog.horizon)} {net.unit}, grid {_g(prog.grid)}")
        except EvalError as e:
            out.append(f"params: cannot be evaluated: {e}")
        out.append("vars: " + (", ".join(f"{k}{'[' + ','.join(v['index']) + ']' if v['index'] else ''}:"
                                         f"{v['type'] if v['type'] != 'enum' else '{' + ','.join(map(str, v['enum'])) + '}'}"
                                         for k, v in net.vars.items()) or "none"))
        out.append("events: " + (", ".join(f"{e.name}{'[' + ','.join(s for _, s in e.idx) + ']' if e.idx else ''}"
                                           f"({e.kind})" for e in net.events.values()) or "none"))
        if net.flows:
            out.append("flows: " + ", ".join(f["var"] for f in net.flows))
        out.append("observes: " + (", ".join(net.observes) or "none") + "; whatifs: "
                   + (", ".join(net.whatifs) or "none"))
        return out

    def _hole_lines(self, net):
        if not net.holes:
            return ["holes: 0"]
        out = [f"holes: {len(net.holes)}"]
        for h in net.holes[:12]:
            out.append(f"  line {h['line']}: {h['error']}\n    > {h['text']}")
        if len(net.holes) > 12:
            out.append(f"  (+{len(net.holes) - 12} more)")
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
                return (f"{ERROR_PREFIX}: the net is read-only in this mode (hint: use run_scenario with a whatif "
                        f"or changes, check_invariants and show_model)")
            out = fn(state, args)
        except Exception as e:
            out = f"{ERROR_PREFIX}: internal {type(e).__name__}: {e} (hint: check the arguments)"
        state.log.append((name, {k: (v if k != "source" else f"<{len(str(v))} chars>") for k, v in args.items()}
                          if isinstance(args, dict) else args, out[:200]))
        return out

    def _t_show_model(self, state, args):
        src = self._src(state)
        if not src.strip():
            return "(no net yet)"
        net = self._net(state)
        tail = "\n".join([""] + self._summary_lines(net) + self._hole_lines(net)
                         + structural_lines(net, self._structural(state)))[:MAX_OUT // 2]
        doc = "\n".join(f"{i + 1:4d}| {l}" for i, l in enumerate(src.splitlines()))
        room = MAX_OUT - len(tail) - 120
        if len(doc) > room:
            cut = doc[:room].rsplit("\n", 1)[0]
            doc = cut + f"\n... [document truncated after line {cut.count(chr(10)) + 1} of {len(src.splitlines())}]"
        return doc + "\n" + tail

    def _t_submit_model(self, state, args):
        src = args.get("source")
        origin = "source"
        if not isinstance(src, str) or not src.strip():
            src = extract_block(state.__dict__.get("_last_text", ""))
            origin = "fenced block"
        if not isinstance(src, str) or not src.strip():
            return (f"{ERROR_PREFIX}: no net found (hint: put the whole net in a fenced block labelled net, i.e. a "
                    f"line ```net ... a line ```, in the same response, or pass it as `source`)")
        return self._store_and_run(state, src, f"submitted from {origin}", args)

    def _t_patch_model(self, state, args):
        old, new = args.get("old"), args.get("new")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            return f"{ERROR_PREFIX}: patch_model needs non-empty string 'old' and string 'new'"
        src = self._src(state)
        if not src.strip():
            return f"{ERROR_PREFIX}: no net yet (hint: submit_model first)"
        n = src.count(old)
        if n != 1:
            if n == 0:
                first = old.strip().splitlines()[0] if old.strip() else old
                close = difflib.get_close_matches(first.strip(), [l.strip() for l in src.splitlines()], n=3,
                                                  cutoff=0.5)
                hint = ("closest lines: " + " | ".join(close)) if close else "use show_model to copy the exact text"
                return f"{ERROR_PREFIX}: `old` text not found in the net (hint: {hint})"
            return f"{ERROR_PREFIX}: `old` text occurs {n} times (hint: include more of the line so it is unique)"
        return self._store_and_run(state, src.replace(old, new, 1), "patched", args)

    def _store_and_run(self, state, src, verb, args):
        if len(src) > MAX_DOC:
            return f"{ERROR_PREFIX}: net too long ({len(src)} chars, max {MAX_DOC})"
        state.rep["source"] = src
        net = self._net(state)
        head = [f"net {verb} ({len(src.splitlines())} lines): {len(net.sets)} sets, {len(net.params)} params, "
                f"{len(net.vars)} vars, {len(net.events)} events, {len(net.flows)} flows, {len(net.observes)} "
                f"observes, {len(net.invariants)} invariants, {len(net.whatifs)} whatifs"]
        if not net.usable():
            return (f"{ERROR_PREFIX}: the net is stored but cannot run (it needs at least one var and one event "
                    f"that are not holes)\n" + "\n".join(head + self._hole_lines(net)))
        runs = args.get("runs")
        runs = _maybe_json(runs) if runs is not None else []
        if isinstance(runs, dict):
            runs = [runs]
        extra = []
        if not isinstance(runs, list):
            extra.append(f"{ERROR_PREFIX}: runs must be a list of run objects; they were not run")
            runs = []
        wis = list(net.whatifs)[:MAX_WHATIFS]
        if len(net.whatifs) > MAX_WHATIFS:
            extra.append(f"note: only the first {MAX_WHATIFS} stored whatifs were run on submit "
                         f"({', '.join(list(net.whatifs)[MAX_WHATIFS:])} can be run with run_scenario)")
        auto = [{"name": w, "whatif": w} for w in wis]
        return self._run_many(state, [{"name": "baseline"}] + auto + list(runs)[:MAX_RUNS], head + extra,
                              check=True)

    def _t_run_scenario(self, state, args):
        if not self._src(state).strip():
            return f"{ERROR_PREFIX}: no net yet (hint: call submit_model first)"
        net = self._net(state)
        if not net.usable():
            return f"{ERROR_PREFIX}: the stored net cannot run:\n" + "\n".join(self._hole_lines(net))
        runs = args.get("runs")
        if runs is not None:
            runs = _maybe_json(runs)
            if isinstance(runs, dict):
                runs = [runs]
            if not isinstance(runs, list) or not runs:
                return f"{ERROR_PREFIX}: runs must be a non-empty list of run objects"
            defaults = {k: v for k, v in args.items() if k in ("horizon", "seeds", "seed", "events", "keys",
                                                                "compare", "mode")}
            runs = [{**defaults, **r} if isinstance(r, dict) else r for r in runs]
        else:
            runs = [{k: v for k, v in args.items() if k != "runs"}]
        head = []
        if len(runs) > MAX_RUNS:
            head.append(f"note: at most {MAX_RUNS} runs per call; the last {len(runs) - MAX_RUNS} were not run")
            runs = runs[:MAX_RUNS]
        return self._run_many(state, runs, head, check=False)

    def _t_check_invariants(self, state, args):
        if not self._src(state).strip():
            return f"{ERROR_PREFIX}: no net yet (hint: call submit_model first)"
        net = self._net(state)
        out = ["CHECK_INVARIANTS"] + self._hole_lines(net) + structural_lines(net, self._structural(state))
        try:
            depth = max(1, min(30, int(args.get("depth", 8) or 8)))
            nruns = max(1, min(MAX_SEEDS, int(args.get("runs", 100) or 100)))
        except (TypeError, ValueError):
            return f"{ERROR_PREFIX}: depth and runs must be integers"
        if not net.usable():
            return f"{ERROR_PREFIX}: the net cannot run\n" + "\n".join(out)
        try:
            prog = Prog(net)
        except EvalError as e:
            return f"{ERROR_PREFIX}: {e}\n" + "\n".join(out)
        why = _enumerable(net)
        if not net.invariants:
            out.append("note: no invariant lines; the search only looks for bound violations (clipping)")
        if not why:
            r = bfs_check(net, depth, prog)
            if r["ok"]:
                out.append(f"bounded check (enumerable model, every enabled event is a choice, timing ignored), "
                           f"depth {depth}: OK - no violation in {r['states']:,} reachable states ("
                           + ("state space covered" if r["covered"] else f"bounded at depth {depth}"
                              + (f"; {r['msg']}" if r["msg"] else "")) + ")")
            else:
                out.append(f"bounded check, depth {depth}: VIOLATION after {len(r['trace']) - 1} event(s): "
                           f"{r['msg']}")
                out.append("  shortest trace: " + " -> ".join(r["trace"]))
                if r.get("state"):
                    out.append("  state: " + _short(r["state"], 400))
        else:
            out.append(f"sampled check ({nruns} runs; not enumerable because: {'; '.join(why[:4])})")
            t0 = time.perf_counter()
            deadline = t0 + self.TIME_BUDGET
            res = []
            for i in range(nruns):
                if i >= MIN_SEEDS and time.perf_counter() > t0 + 0.75 * self.TIME_BUDGET:
                    out.append(f"note: time limit: {i} runs done")
                    break
                res.append(Sim(prog, 0, i, deadline, check=True).run())
            errs = [r for r in res if r.err]
            for i, inv in enumerate(net.invariants):
                hit = [r for r in res if i in r.viol]
                label = ("conserved " if inv["conserved"] else "") + inv["text"]
                if not hit:
                    out.append(f"  {label}: held after every event in all {len(res)} runs")
                    continue
                f = min(hit, key=lambda r: r.viol[i][1])
                c, t1, last = f.viol[i]
                out.append(f"  {label}: VIOLATED in {len(hit)}/{len(res)} runs; earliest at t={_g(t1)}; last events "
                           f"before it: " + ", ".join(f"{n}@{_g(t)}" for t, n in last))
            clips = Counter()
            for r in res:
                for k, v in r.clips.items():
                    clips[k] += 1
            for k, n in clips.items():
                out.append(f"  bound of {k} reached (clipped) in {n}/{len(res)} runs")
            if errs:
                out.append(f"  {len(errs)} run(s) stopped with an error; first: {errs[0].err} (t={_g(errs[0].err_t)})")
        return "\n".join(out)[:MAX_OUT]

    # ------------------------------------------------------------------ runs
    def _spec(self, net, r, k, check_names):
        if not isinstance(r, dict):
            return None, f"run {k + 1} must be an object"
        name = str(r.get("name") or (r.get("whatif") if isinstance(r.get("whatif"), str) else None) or f"run{k}")[:40]
        texts = []
        clauses = []
        wi = r.get("whatif")
        if wi:
            if not isinstance(wi, str) or wi not in net.whatifs:
                return None, (f"run '{name}': unknown whatif '{wi}'{_dym(wi, net.whatifs)} (stored: "
                              f"{', '.join(net.whatifs) or 'none'})")
            clauses += net.whatifs[wi]["clauses"]
            texts.append(f"whatif {wi}")
        ch = _maybe_json(r.get("changes"))
        if ch is not None:
            if isinstance(ch, str):
                ch = [x for x in _split_top(ch, ";") if x.strip()] if not ch.strip().startswith("replace") else [ch]
            if not isinstance(ch, list):
                return None, f"run '{name}': changes must be a list of clause strings"
            try:
                clauses += net.parse_clauses([str(x) for x in ch])
            except NetError as e:
                return None, f"run '{name}': {e}"
            texts += [str(x) for x in ch]
        try:
            hz = r.get("horizon")
            hz = float(hz) if hz is not None else None
            n = int(r.get("seeds") or r.get("n_seeds") or net.seeds)
            seed = int(r.get("seed", 0) or 0)
        except (TypeError, ValueError):
            return None, f"run '{name}': horizon must be a number, seeds and seed integers"
        notes = []
        if n > MAX_SEEDS:
            notes.append(f"seeds capped at {MAX_SEEDS}")
            n = MAX_SEEDS
        n = max(1, n)
        mode = r.get("mode") or "stochastic"
        if mode not in ("stochastic", "fluid", "both"):
            return None, f"run '{name}': mode must be stochastic, fluid or both"
        try:
            prog = Prog(net, clauses, hz, "fluid" if mode == "fluid" else "stochastic")
        except EvalError as e:
            return None, f"run '{name}': {e}"
        fprog = None
        if mode == "both":
            fprog = Prog(net, clauses, hz, "fluid")
        rec = set(prog.rec_names) | set(net.vars)
        events = _parse_run_events(r.get("events"),
                                   lambda s: Compiler(net, None, "record", rec_keys=rec).compile(s))
        keys = [str(x) for x in r.get("keys")] if isinstance(r.get("keys"), list) else []
        return {"name": name, "clauses": clauses, "texts": texts, "prog": prog, "fprog": fprog, "n": n, "seed": seed,
                "mode": mode, "events": events, "keys": keys, "changed": bool(clauses),
                "compare": r.get("compare", True) is not False, "notes": notes, "hz": hz}, None

    def _run_many(self, state, raw, head, check):
        net = self._net(state)
        lines = list(head)
        specs, problems = [], []
        for k, r in enumerate(raw):
            s, err = self._spec(net, r, k, None)
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
        if not specs:
            return f"{ERROR_PREFIX}: " + "\n".join(problems or ["nothing to run"])
        # union of event expressions on unchanged runs; hidden baselines for comparisons
        all_ev = []
        for s in specs:
            for e in s["events"]:
                if e[0] not in [x[0] for x in all_ev]:
                    all_ev.append(e)
        for s in specs:
            if not s["changed"]:
                s["events"] = s["events"] + [e for e in all_ev if e[0] not in [x[0] for x in s["events"]]]
        for s in list(specs):
            if not (s["changed"] and s["compare"]):
                continue
            want_mode = "fluid" if s["mode"] == "fluid" else "stochastic"
            ref = next((b for b in specs if not b["changed"] and b["seed"] == s["seed"]
                        and b["prog"].horizon == s["prog"].horizon and b["prog"].mode == want_mode), None)
            if ref is None:
                bname = "baseline" if "baseline" not in names else f"baseline(h={_g(s['prog'].horizon)})"
                ref, err = self._spec(net, {"name": bname, "horizon": s["hz"], "seeds": s["n"], "seed": s["seed"],
                                            "mode": want_mode, "keys": s["keys"]}, 0, None)
                if err:
                    problems.append(err)
                    continue
                ref["events"] = list(all_ev)
                names.add(ref["name"])
                specs.insert(0, ref)
            s["ref"] = ref["name"]
            ref["n"] = max(ref["n"], s["n"])
        res = self._execute(specs)
        base = next((s for s in specs if not s["changed"] and s["prog"].mode == "stochastic"), None)
        if base is not None and not base["hz"]:
            R = res[specs.index(base)]
            ok = R["runs"] and not any(r.err for r in R["runs"])
            self._record_run(state, ok)
        return self._report(net, state, specs, res, lines, problems, check)

    def _execute(self, specs):
        t0 = time.perf_counter()
        budget = self.TIME_BUDGET
        soft, hard = t0 + 0.75 * budget, t0 + budget
        out = [{"runs": [], "fluid": [], "cut": None} for _ in specs]
        # fluid parts first (deterministic: one run)
        for s, o in zip(specs, out):
            fp = s["fprog"] if s["mode"] == "both" else (s["prog"] if s["mode"] == "fluid" else None)
            if fp is None:
                continue
            nf = 1 if fp.deterministic else s["n"]
            for i in range(nf):
                if i >= MIN_SEEDS and time.perf_counter() > soft:
                    break
                o["fluid"].append(Sim(fp, s["seed"], i, hard).run())
            if s["mode"] == "fluid":
                o["runs"] = o["fluid"]
        stoch = [(s, o) for s, o in zip(specs, out) if s["mode"] != "fluid"]
        maxn = max([s["n"] for s, _ in stoch] + [0])
        cut = None
        for i in range(maxn):
            now = time.perf_counter()
            if i > 0 and ((i >= MIN_SEEDS and now > soft) or now > hard):
                cut = i
                break
            for s, o in stoch:
                if i < s["n"]:
                    o["runs"].append(Sim(s["prog"], s["seed"], i, hard).run())
        for o in out:
            o["cut"] = cut
        return out

    def _record_run(self, state, ok):
        h = hashlib.sha1(self._src(state).encode("utf-8")).hexdigest()
        prev = state.__dict__.setdefault("_runs", {}).get(h, {})
        state.__dict__["_runs"][h] = {"ok": ok or prev.get("ok", False)}

    # ------------------------------------------------------------------ report
    def _report(self, net, state, specs, res, pre, problems, check):
        header = list(pre)
        for p in problems:
            header.append(f"{ERROR_PREFIX}: {p}")
        any_err = bool(problems)
        # CHECK
        chk = []
        if check:
            chk.append("CHECK:")
            chk += ["  " + l for l in self._hole_lines(net)]
            for w in net.warnings[:6]:
                chk.append(f"  warning: {w}")
            magic = [(e.name, e.magic) for e in net.events.values() if e.magic]
            if magic:
                chk.append("  warning (magic numbers - stated-looking numbers inside event lines that are not params; "
                           "hint: make each a param): " + "; ".join(f"{n}: {', '.join(m[:4])}" for n, m in magic[:6]))
            base = next((s for s in specs if s["name"] == "baseline"), None)
            if base is not None:
                br = res[specs.index(base)]["runs"]
                if br:
                    enabled = set().union(*[r.enabled for r in br])
                    firedset = {k for r in br for k, v in r.counts.items() if v}
                    never = [e for e in net.events if e not in enabled and e in {t["ev"].name
                                                                                   for t in base["prog"].templates}]
                    if never:
                        chk.append("  warning: never enabled in the baseline: " + ", ".join(never[:12])
                                   + " (hint: check their conditions and timing)")
                    nf = [e for e in net.events if e in enabled and e not in firedset]
                    if nf:
                        chk.append("  note: enabled but never fired in the baseline: " + ", ".join(nf[:12]))
            chk += ["  " + l for l in structural_lines(net, self._structural(state))]
        approx = set()
        for s, R in zip(specs, res):
            P = s["prog"]
            if s["mode"] in ("fluid", "both"):
                approx.add(f"fluid ODE by RK4, h={_g(P.h)}")
            if P.continuous and s["mode"] != "fluid":
                approx.add(f"flows by RK4 with h={_g(P.h)}; hazards piecewise constant over h; guards on flowing "
                           f"vars checked every h")
            if any(r.leap for r in R["runs"]):
                approx.add(f"tau-leaping used (tau={_g(P.h)}) in run {s['name']}")
        if not any("RK4" in a for a in approx) and not any("tau" in a for a in approx):
            approx.add("exact (next-reaction method; event-driven clocks and schedules)")
        if any(e.kind in ("every", "at", "after") for e in net.events.values()):
            approx.add("simultaneous timed events: higher priority first, then declaration order")
        (chk if check else header).append(("  " if check else "") + "approximations: " + "; ".join(sorted(approx)))
        # RUN lines
        for s, R in zip(specs, res):
            runs = R["runs"]
            nok = sum(1 for r in runs if r.complete)
            h = f"RUN {s['name']}: horizon {_g(s['prog'].horizon)} {net.unit}, mode {s['mode']}, " \
                f"{nok}/{len(runs)} runs completed"
            if s["texts"]:
                h += "; " + "; ".join(s["texts"])[:300]
            if s.get("ref"):
                h += f"; compared with {s['ref']}"
            header.append(h)
            for n in s["notes"] + s["prog"].notes + (s["fprog"].notes if s["fprog"] else []):
                header.append(f"  note: {n}")
            errs = [r for r in runs if r.err]
            if errs:
                any_err = True
                header.append(f"  {ERROR_PREFIX}: {len(errs)} of {len(runs)} run(s) stopped: {errs[0].err} "
                              f"(t={_g(errs[0].err_t)})")
            tot_clip = Counter()
            n_clip = Counter()
            for r in runs:
                for k, v in r.clips.items():
                    tot_clip[k] += v
                    n_clip[k] += 1
            if tot_clip:
                header.append("  clipped: " + "; ".join(f"{k} clipped {tot_clip[k]} times in {n_clip[k]}/{len(runs)} "
                                                        f"runs" for k in list(tot_clip)[:8]))
            rnd = Counter()
            for r in runs:
                rnd.update(r.rounded)
            if rnd:
                header.append("  rounded to integers: " + ", ".join(f"{k} {v} times" for k, v in list(rnd.items())[:6]))
            neg = Counter()
            for r in runs:
                neg.update(r.negrate)
            if neg:
                header.append("  negative rates treated as 0: " + ", ".join(f"{k} ({v})" for k, v in neg.items()))
            over = Counter()
            for r in runs:
                over.update(r.fifo_over)
            if over:
                header.append("  fifo capacity reached (items not added): " + ", ".join(f"{k} {v}"
                                                                                         for k, v in over.items()))
            vio = Counter()
            for r in runs:
                for i in r.viol:
                    vio[i] += 1
            for i, c in vio.items():
                header.append(f"  invariant '{net.invariants[i]['text']}' violated in {c}/{len(runs)} runs")
        cut = res[0]["cut"] if res else None
        if cut is not None:
            header.append(f"note: time budget ({int(self.TIME_BUDGET)} s): seeds reduced to {cut} per run (all runs "
                          f"stopped at the same seed, so comparisons stay paired)")
        if any(s["mode"] == "fluid" for s in specs):
            pass

        def keys_of(s, R):
            names = s["prog"].rec_names
            obs = [k for k in names if k in net.observes]
            req = [k for k in s["keys"] if k in names]
            rest = [k for k in names if k not in obs and k not in req]
            return req, [k for k in obs if k not in req], rest

        def final_block(limits):
            compact, olim, limit = limits
            out = []
            for s, R in zip(specs, res):
                full = [r.final for r in R["runs"] if r.final is not None]
                if not full:
                    out.append(f"FINAL [{s['name']}]: no run reached the horizon")
                    continue
                req, obs, rest = keys_of(s, R)
                if compact == 3 and s.get("ref"):
                    out.append(f"FINAL [{s['name']}]: (n={len(full)}) = FINAL [{s['ref']}] + the paired differences "
                               f"in EFFECT [{s['name']}] (shown in full with run_scenario for this run alone)")
                    continue
                rest = obs[olim:] + rest
                short = compact and (s["changed"] or compact >= 2)
                out.append(f"FINAL [{s['name']}] (t={_g(s['prog'].horizon)}, n={len(full)}"
                           + ("; mean [p05, p95]" if short else "") + "):")
                for k in req + obs[:olim] + rest[:limit]:
                    vals = [o.get(k) for o in full]
                    out.append(f"  {k}: " + (_compact(vals) if short else _summ(vals)))
                if len(rest) > limit:
                    out.append(f"  (+{len(rest) - limit} more: {', '.join(rest[limit:limit + 10])}"
                               + (" ..." if len(rest) > limit + 10 else "") + "; ask for them with `keys`)")
                miss = [k for k in s["keys"] if k not in s["prog"].rec_names]
                if miss:
                    out.append(f"  note: unknown keys {miss} (observed: {', '.join(s['prog'].rec_names[:20])})")
            return out

        def effect_block(limits):
            compact, olim, limit = limits
            out = []
            by = {s["name"]: (s, R) for s, R in zip(specs, res)}
            for s, R in zip(specs, res):
                if not s.get("ref"):
                    continue
                bs, bR = by[s["ref"]]
                a_runs = {i: r for i, r in enumerate(bR["runs"]) if r.final is not None}
                c_runs = {i: r for i, r in enumerate(R["runs"]) if r.final is not None}
                common = sorted(set(a_runs) & set(c_runs))
                out.append(f"EFFECT [{s['name']}] vs {s['ref']} (paired over {len(common)} common seeds):")
                if not common:
                    out.append("  no common seed reached the horizon")
                    continue
                req, obs, rest = keys_of(s, R)
                rest = obs[olim:] + rest
                obs = obs[:olim]
                changed, same = [], []
                for k in req + obs + rest:
                    pairs = [(a_runs[i].final.get(k), c_runs[i].final.get(k)) for i in common]
                    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                    if not pairs:
                        continue
                    if all(isinstance(x, bool) and isinstance(y, bool) for x, y in pairs):
                        if all(x == y for x, y in pairs):
                            same.append(k)
                            continue
                        pa = sum(x for x, _ in pairs) / len(pairs)
                        pc = sum(y for _, y in pairs) / len(pairs)
                        changed.append((k, f"  {k}: P(true) {pa:.3f} -> {pc:.3f} (diff {pc - pa:+.3f})"))
                    elif all((_isnum(x) or isinstance(x, bool)) and (_isnum(y) or isinstance(y, bool))
                             for x, y in pairs):
                        d = sorted(float(y) - float(x) for x, y in pairs)
                        if all(abs(float(y) - float(x)) <= 1e-9 * max(1.0, abs(float(x))) for x, y in pairs):
                            same.append(k)
                            continue
                        pp, pn = sum(v > 0 for v in d) / len(d), sum(v < 0 for v in d) / len(d)
                        if len(d) == 1:
                            changed.append((k, f"  {k}: diff {_signed(d[0])} ({_g(float(pairs[0][0]))} -> "
                                               f"{_g(float(pairs[0][1]))})"))
                        elif compact:
                            changed.append((k, f"  {k}: mean diff {_signed(sum(d) / len(d))} [p05 "
                                               f"{_signed(_quant(d, .05))}, p95 {_signed(_quant(d, .95))}] "
                                               f"P(>0)={pp:.2f} P(<0)={pn:.2f}"))
                        else:
                            changed.append((k, f"  {k}: mean diff {_signed(sum(d) / len(d))} | p05 "
                                               f"{_signed(_quant(d, .05))} p95 {_signed(_quant(d, .95))} | "
                                               f"P(diff>0)={pp:.2f} P(diff<0)={pn:.2f} (n={len(d)})"))
                    else:
                        nd = sum(1 for x, y in pairs if x != y)
                        if nd == 0:
                            same.append(k)
                        else:
                            changed.append((k, f"  {k}: differs on {nd} of {len(pairs)} seeds"))
                shown = 0
                for k, ln in changed:
                    if k in rest and shown >= len(req) + len(obs) + limit:
                        out.append(f"  (+{len(changed) - shown} more changed quantities omitted)")
                        break
                    out.append(ln)
                    shown += 1
                if same:
                    cap = 25 if limit >= 8 else 8
                    out.append(f"  unchanged on every seed: {', '.join(same[:cap])}"
                               + (f" (+{len(same) - cap} more)" if len(same) > cap else ""))
            return out

        events_b, counts_b, shapes_b, fluid_b, series_b = [], [], [], [], []
        base_counts = None
        for s, R in zip(specs, res):
            runs = R["runs"]
            P = s["prog"]
            # EVENTS
            evl = []
            for label, c, when in s["events"]:
                if c is None:
                    evl.append(f"  [{s['name']}] {label}: error: {when}")
                    continue
                evl.append(f"  [{s['name']}] " + self._event_line(label, c, when, P, runs))
            events_b.append(evl)
            # EVENT COUNTS
            if runs:
                n = len(runs)
                cl = []
                evnames = list(dict.fromkeys(t["ev"].name for t in P.templates))
                rows = {}
                never = []
                for e in evnames:
                    if e in P.fluid_events:
                        rows[e] = f"{e}: continuous (fluid)"
                        continue
                    cs = [r.counts.get(e, 0) for r in runs]
                    ft = sorted(r.first[e] for r in runs if e in r.first)
                    if not ft:
                        never.append(e)
                        continue
                    if n == 1:
                        row = f"{e}: {cs[0]} firings, first t={_g(ft[0])}"
                    else:
                        row = (f"{e}: mean {_g(sum(cs) / n)}, P(fired)={len(ft) / n:.2f}, first t mean "
                               f"{_g(sum(ft) / len(ft))} p50 {_g(_quant(ft, .5))} p95 {_g(_quant(ft, .95))}")
                    rows[e] = row
                if never:
                    rows["__never"] = "never fired: " + ", ".join(never)
                if base_counts is None and not s["changed"]:
                    base_counts = rows
                    cl = [f"  [{s['name']}] " + r for r in rows.values()]
                else:
                    diff = [r for e, r in rows.items() if base_counts is None or base_counts.get(e) != r]
                    cl = [f"  [{s['name']}] " + r for r in diff] or [f"  [{s['name']}] same as baseline"]
                counts_b.append(cl)
            # SHAPE and SERIES
            times = P.grid_times
            mean = {}
            for k in P.rec_names:
                cols = []
                for i in range(len(times)):
                    vs = _numeric([r.grid[k][i] for r in runs if i < len(r.grid.get(k, []))])
                    cols.append(sum(vs) / len(vs) if vs else None)
                mean[k] = cols
            req, obs, rest = keys_of(s, R)
            varying = [k for k in rest if _varies(mean[k])]
            chosen = (req + obs + varying)[:SERIES_KEYS] or rest[:SERIES_KEYS]
            if s["changed"] or len(specs) == 1 or s is next((x for x in specs if not x["changed"]), None):
                sh = [f"  [{s['name']}] {k}: " + shape(times, mean[k]) for k in chosen]
                shapes_b.append(sh)
                if chosen:
                    idx = _sample_idx(times, [mean[k] for k in chosen])
                    blk = [f"  [{s['name']}] t: " + _rle([_g(times[i]) for i in idx])]
                    for k in chosen:
                        blk.append(f"  [{s['name']}] {k}: " + _rle([_s3(mean[k][i]) for i in idx]))
                    series_b.append(blk)
            # FLUID vs STOCHASTIC
            if s["mode"] == "both" and R["fluid"]:
                fl = []
                fr = R["fluid"]
                fprog = s["fprog"]
                for k in (req + obs + varying)[:SERIES_KEYS]:
                    if k not in fprog.rec_names:
                        continue
                    fm = []
                    for i in range(len(fprog.grid_times)):
                        vs = _numeric([r.grid[k][i] for r in fr if i < len(r.grid.get(k, []))])
                        fm.append(sum(vs) / len(vs) if vs else None)
                    sm_ = mean[k]
                    if not fm or fm[-1] is None or not sm_ or sm_[-1] is None:
                        continue
                    diffs = [(times[i], sm_[i] - fm[i]) for i in range(min(len(sm_), len(fm)))
                             if sm_[i] is not None and fm[i] is not None]
                    tmx, dmx = max(diffs, key=lambda p: abs(p[1]))
                    scale = max([abs(v) for v in sm_ + fm if v is not None] + [1e-12])
                    flag = " LARGE (small-number effects: extinction, saturation or queue variability; averaged-rate " \
                           "reasoning would mislead)" if abs(dmx) > 0.1 * scale else ""
                    fl.append(f"  [{s['name']}] {k}: stochastic mean {_g(sm_[-1])} vs fluid {_g(fm[-1])} at the "
                              f"horizon (diff {_signed(sm_[-1] - fm[-1])}, {100 * (sm_[-1] - fm[-1]) / max(abs(fm[-1]), 1e-12):+.1f}%); "
                              f"max |diff| {_g(abs(dmx))} at t={_g(tmx)}{flag}")
                fluid_b.append(fl or [f"  [{s['name']}] no numeric observed quantity to compare"])
        text = _assemble(header + chk, final_block, effect_block,
                         [("EVENTS", events_b), ("EVENT COUNTS (mean firings per run, P(fired at least once), first "
                                                 "firing time)", counts_b),
                          ("SHAPE (mean trajectory)", shapes_b), ("FLUID vs STOCHASTIC", fluid_b),
                          ("SERIES (mean on the grid)", series_b)],
                         any_err)
        return text

    def _event_line(self, label, c, when, P, runs):
        times = P.grid_times
        rs, fpt = [], []
        names = P.rec_names
        for r in runs:
            n = min((len(v) for v in r.grid.values()), default=0)
            if n == 0:
                rs.append(None)
                continue

            def ev(i):
                R = Env()
                R.params, R.sets = P.params, P.sets
                R.rec = {k: r.grid[k][i] for k in names}
                for k in list(R.rec):
                    m = re.match(r"^([A-Za-z_]\w*)\[", k)
                    if m and m.group(1) not in R.rec:
                        pass
                R.t = times[i]
                try:
                    return bool(c.fn(R))
                except (EvalError, TypeError, ValueError, ZeroDivisionError, KeyError):
                    return None
            if when == "final":
                rs.append(ev(n - 1) if r.final is not None else None)
            elif _isnum(when):
                i = next((j for j, t in enumerate(times) if t >= when - 1e-9), None)
                rs.append(ev(i) if i is not None and i < n else None)
            else:
                vals = [ev(i) for i in range(n)]
                if when == "any":
                    if any(v is True for v in vals):
                        rs.append(True)
                        fpt.append(times[next(i for i, v in enumerate(vals) if v is True)])
                    else:
                        rs.append(False if all(v is False for v in vals) else None)
                else:
                    rs.append(False if any(v is False for v in vals) else (True if all(v is True for v in vals)
                                                                            else None))
        defined = [x for x in rs if x is not None]
        if not defined:
            return f"P({label}) [{when}] = undefined (no run defines it; observed: {', '.join(names[:15])})"
        p = sum(defined) / len(defined)
        se = math.sqrt(p * (1 - p) / len(defined))
        und = len(rs) - len(defined)
        out = f"P({label}) [{when}] = {p:.3f} +/- {se:.3f} (n={len(defined)}" + (f", {und} undefined" if und else "") + ")"
        if when == "any" and fpt:
            xs = sorted(fpt)
            out += (f"; first passage t (grid {_g(P.grid)}): mean {_g(sum(xs) / len(xs))} p05 {_g(_quant(xs, .05))} "
                    f"p50 {_g(_quant(xs, .5))} p95 {_g(_quant(xs, .95))} (n={len(xs)})")
        return out

    # ------------------------------------------------------------------ validity
    def validity(self, state):
        src = self._src(state)
        net = self._net(state)
        holes = [f"line {h['line']}: {h['error']}" for h in net.holes]
        errors = []
        usable = bool(src.strip()) and net.usable()
        executable = False
        if usable:
            h = hashlib.sha1(src.encode("utf-8")).hexdigest()
            if state.__dict__.get("_runs", {}).get(h, {}).get("ok"):
                executable = True
            else:
                executable, err = self._smoke(net)
                if err:
                    errors.append(err)
        return {"usable": usable, "executable": executable, "holes": holes, "errors": errors,
                "warnings": list(net.warnings)}

    def _smoke(self, net):
        try:
            prog = Prog(net)
            prog = Prog(net, horizon=min(prog.horizon, 2.0))
        except EvalError as e:
            return False, str(e)
        deadline = time.perf_counter() + self.TIME_BUDGET
        for i in range(2):
            r = Sim(prog, 0, i, deadline).run()
            if r.err:
                return False, r.err
        return True, None

    # ------------------------------------------------------------------ episode (document channel)
    def run_episode(self, ep, state, messages):
        """Generic tool loop plus: system-prompt parity, the fenced ```net channel with implicit submit, and a
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
                messages[-1]["content"] += ("\nnote: the ```net block in this response was not submitted "
                                            "(call submit_model to store and run it)")
        ep.forced_final = True
        messages.append({"role": "user", "content": FORCE_FINAL_MSG})
        m = ep.call(messages, tools=None, label="forced_final", final=True)
        messages.append({"role": "assistant", "content": m.get("content") or ""})
        return m.get("content") or "", messages, trace


def _short(o, limit=120):
    try:
        if isinstance(o, dict):
            o = {",".join(map(str, k)) if isinstance(k, tuple) else str(k): v for k, v in o.items()}
        s = json.dumps(o, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        s = repr(o)
    return s if len(s) <= limit else s[:limit] + "..."


def _varies(xs):
    v = [x for x in xs if x is not None]
    return bool(v) and (max(v) - min(v)) > 1e-9 * max(1e-12, abs(max(v)), abs(min(v)))


def _assemble(header, final_block, effect_block, optional, is_error):
    """Bounded output: header (with CHECK) + FINAL + EFFECT always (shrinking the 'other vars' lists if needed);
    then the optional sections in order while they fit. Dropped blocks are reported."""
    core = None
    for limit in ((0, 99, 12), (1, 99, 12), (1, 99, 6), (2, 99, 6), (2, 99, 2), (2, 99, 0),
                  (3, 99, 0), (3, 12, 0), (3, 6, 0), (3, 3, 0), (3, 1, 0)):
        core = header + final_block(limit) + effect_block(limit)
        if len("\n".join(core)) <= MAX_OUT - 400:
            break
    text = "\n".join(core)
    if len(text) > MAX_OUT - 200:
        text = text[:MAX_OUT - 300] + "\n...[FINAL/EFFECT cut at the output limit; request fewer runs or keys]"
    omitted, stop = [], False
    for title, groups in optional:
        items = [g for g in groups if g]
        if not items:
            continue
        if stop:
            omitted.append(f"{title.split(' (')[0]}: {len(items)} run(s)")
            continue
        add = "\n" + title + ":"
        for n_done, g in enumerate(items):
            blk = add + "\n" + "\n".join(g)
            if len(text) + len(blk) > MAX_OUT - 200:
                stop = True
                omitted.append(f"{title.split(' (')[0]}: {len(items) - n_done} run(s)")
                break
            text += blk
            add = ""
    if omitted:
        text += "\n[output limit: omitted " + "; ".join(omitted) + "]"
    if is_error and not text.startswith(ERROR_PREFIX):
        text = f"{ERROR_PREFIX}: see the problems below\n" + text
    return text[:MAX_OUT]


G1C4 = G1C4Arm
