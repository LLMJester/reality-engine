"""G2C1: stochastic structural-equation document with declared parameters, rule switches and evidence checks
(G1C3 + G1C1 recombined).

Spec: runs/discovery/gen2/specs/G2C1.md. One line-oriented document (fence label `sse`), no double quotes or
backslashes needed:

  time steps=60 dt=1 unit=minute
  seeds 200
  targets: queue, served_cum
  param arrival_rate = 1.2                 # stated 'about 1.2 a minute'
  rule clerks = 2                          # stated
  arrivals : int := poisson(arrival_rate)
  queue : int = 0 := queue[t-1] + arrivals - served
  event long_queue: queue >= 10
  check q_at_30: queue at 30 ~ 8 +- 3      # stated 'around 8 waiting by half past'
  whatif third_clerk: rule clerks = 3 ; targets queue ; events long_queue

Semantics: a discrete-time stochastic structural causal model. Endogenous nodes are recomputed every step in
same-step dependency order (cycles are an error); `= v0` is the step-0 value (A2's step-0 rule); exogenous nodes are
fixed, or sampled once per run. Per-step stochastic functions draw from their own stream, keyed by (replication seed,
node, call index, step), by inverse-CDF from one uniform, so what-ifs are paired with the baseline under common random
numbers. Checks compare runs with stated observations and never weight runs.

The evaluator is pure Python and runs in process (no sandbox, no subprocess). It reuses G1C3's line helpers and
`_expr`'s tokenizer/parser (the stochastic functions are added in a parser subclass); neither module is modified.
"""
import copy
import difflib
import hashlib
import json
import math
import re
import time as _time
from statistics import NormalDist

from tournament.arms import _expr
from tournament.arms.base import ERROR_PREFIX, Arm, ArmState
from tournament.arms.g1c3 import _did_you_mean, _g3, _split_comment, _split_top, compact_series, fenced_block

# in-process executor: nothing runs in a subprocess; the only run-time code is the imported arm modules (D-060)
RUNTIME_DEPS = ["tournament/arms/_expr.py", "tournament/arms/g1c3.py", "tournament/arms/a2_causal.py"]

FENCE_LABEL = "sse"
MAX_OUT = 7800                  # tool result bound (the episode adds the budget line; the runner caps at 8,000)
MAX_RUNS = 6                    # runs per `run` call; stored what-ifs run at submit
MAX_SEEDS = 400
DEFAULT_SEEDS = 200
SENS_SEEDS = 50
SENS_MAX = 3
WORK_CAP = 3_000_000            # node evaluations per tool call
WALL_S = 45.0                   # wall-clock guard per tool call
MAX_STEPS = 5000
RENDER_MAX = 60000
MUTATING = ("submit", "edit")
BUDGET_LINE = "[budget: {k} of {n} responses used; your last response must be the FINAL ANSWER without tool calls]"
ARG_HINT = "send the document as `lines`, an array of strings; use single quotes only"

STOCH = {"bernoulli": (1, 1), "binomial": (2, 2), "poisson": (1, 1), "normal": (2, 2), "uniform": (2, 2),
         "exponential": (1, 1), "choice": (1, 99), "choice_w": (2, 98)}
RESERVED_NAMES = _expr.RESERVED | _expr.FUNCS | set(STOCH)
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
VAR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(:=|:|=|~)")
CHANGE_RE = re.compile(r"^\s*(?:(rule|param|do)\s+)?([A-Za-z_]\w*)\s*(:=|=(?!=))")
CHECK_RE = re.compile(r"^(.*?)\s+at\s+(final|end|any|\d+)\s*(?:~\s*(\S+)\s*(?:\+-|\+/-|±)\s*(\S+)|is\s+(true|false))\s*$",
                      re.I | re.S)
TYPES = ("real", "int", "bool", "category")
TYPE_SYNONYMS = {"float": "real", "number": "real", "num": "real", "double": "real", "integer": "int",
                 "count": "int", "boolean": "bool", "categorical": "category", "cat": "category", "string": "category",
                 "str": "category", "enum": "category"}
SECTION_ORDER = ("HEADER", "ASSUMED", "TARGETS", "CHECKS", "EFFECTS", "EVENTS", "SERIES")


# ============================================================================ expression language (+ stochastic)
class _Parser(_expr._Parser):
    """_expr's recursive-descent parser plus per-step stochastic calls ("rcall", fn, args, call_index)."""

    def __init__(self, src):
        super().__init__(src)
        self.sites = 0

    def call(self, fn, col):
        if fn not in STOCH:
            if fn not in _expr.FUNCS:
                raise _expr.ExprError(f"unknown function '{fn}'", col,
                                      hint="available: " + ", ".join(sorted(_expr.FUNCS | set(STOCH))))
            return super().call(fn, col)
        self.expect_op("(")
        args = []
        if not self.is_op(")"):
            args.append(self.expr())
            while self.is_op(","):
                self.next()
                args.append(self.expr())
        self.expect_op(")")
        lo, hi = STOCH[fn]
        if not lo <= len(args) <= hi:
            raise _expr.ExprError(f"{fn}() takes {lo}" + (f"-{hi}" if hi != lo else "") + f" argument(s), got "
                                  f"{len(args)}", col)
        if fn == "choice_w" and len(args) % 2:
            raise _expr.ExprError("choice_w('a', w1, 'b', w2, ...) takes option/weight pairs", col)
        idx = self.sites
        self.sites += 1
        return ("rcall", fn, args, idx)


def parse_expr(src):
    try:
        return _Parser(src).parse()
    except RecursionError:
        raise _expr.ExprError("expression nested too deeply")


def expr_refs(ast, out=None):
    """{(name, lag)} referenced (t and dt excluded)."""
    out = set() if out is None else out
    k = ast[0]
    if k == "var":
        if ast[1] not in ("t", "dt"):
            out.add((ast[1], ast[2]))
    elif k in ("neg", "not"):
        expr_refs(ast[1], out)
    elif k == "bin":
        expr_refs(ast[2], out)
        expr_refs(ast[3], out)
    elif k in ("and", "or"):
        expr_refs(ast[1], out)
        expr_refs(ast[2], out)
    elif k == "if":
        for a in ast[1:]:
            expr_refs(a, out)
    elif k in ("call", "rcall"):
        for a in ast[2]:
            expr_refs(a, out)
    return out


def has_random(ast):
    k = ast[0]
    if k == "rcall":
        return True
    if k in ("neg", "not"):
        return has_random(ast[1])
    if k == "bin":
        return has_random(ast[2]) or has_random(ast[3])
    if k in ("and", "or"):
        return has_random(ast[1]) or has_random(ast[2])
    if k == "if":
        return any(has_random(a) for a in ast[1:])
    if k == "call":
        return any(has_random(a) for a in ast[2])
    return False


def _value(src):
    """A literal: number, true/false, 'text' or a bare word (category); constant arithmetic is allowed."""
    s = src.strip()
    if not s:
        raise ValueError("missing value")
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    if s in ("true", "false"):
        return s == "true"
    if NAME_RE.fullmatch(s):
        return s
    try:
        a = parse_expr(s)
    except _expr.ExprError as e:
        raise ValueError(str(e))
    if expr_refs(a) or has_random(a):
        raise ValueError(f"a value must be a constant ({s!r})")
    try:
        v = _expr.evaluate(a, lambda n, l: (_ for _ in ()).throw(_expr.EvalError("no names in a constant")))
    except _expr.EvalError as e:
        raise ValueError(str(e))
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15 and re.fullmatch(r"[-+]?\d+", s):
        v = int(v)
    return v


def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ============================================================================ static types
def _join(a, b):
    if a == b:
        return a
    if {a, b} <= {"int", "real"}:
        return "real"
    if {a, b} <= {"int", "bool"}:
        return "int"
    return None


def _typeof(ast, env, errs):
    """Static type of an expression ('int' | 'real' | 'bool' | 'cat' | None); type errors appended to errs."""
    k = ast[0]
    if k == "num":
        return "int" if float(ast[1]).is_integer() else "real"
    if k == "str":
        return "cat"
    if k == "bool":
        return "bool"
    if k == "var":
        return "int" if ast[1] == "t" else "real" if ast[1] == "dt" else env.get(ast[1])
    if k == "neg":
        t = _typeof(ast[1], env, errs)
        if t == "cat":
            errs.append("a category value cannot be negated")
        return "int" if t in ("int", "bool") else "real" if t == "real" else None
    if k in ("not", "and", "or"):
        for a in ast[1:]:
            if _typeof(a, env, errs) == "cat":
                errs.append("a category value is used as a condition (compare it: x == 'a')")
        return "bool"
    if k == "if":
        if _typeof(ast[1], env, errs) == "cat":
            errs.append("a category value is used as a condition (compare it: x == 'a')")
        return _join(_typeof(ast[2], env, errs), _typeof(ast[3], env, errs))
    if k == "bin":
        op, ta, tb = ast[1], _typeof(ast[2], env, errs), _typeof(ast[3], env, errs)
        if op in ("==", "!=", "<", "<=", ">", ">="):
            if (ta == "cat") != (tb == "cat") and None not in (ta, tb):
                errs.append(f"'{op}' compares a category with a number")
            return "bool"
        if "cat" in (ta, tb):
            errs.append(f"arithmetic '{op}' on a category value")
            return None
        if op in ("/", "^"):
            return "real"
        if ta in ("int", "bool") and tb in ("int", "bool"):
            return "int"
        return "real" if ta and tb else None
    if k == "call":
        fn, ts = ast[1], [_typeof(a, env, errs) for a in ast[2]]
        if fn in ("floor", "ceil", "sign") or (fn == "round" and len(ts) == 1):
            return "int"
        if fn in ("min", "max", "clip", "abs"):
            t = ts[0]
            for x in ts[1:]:
                t = _join(t, x)
            return t
        return "real"
    if k == "rcall":
        fn, args = ast[1], ast[2]
        ts = [_typeof(a, env, errs) for a in args]
        if fn == "binomial":
            if ts[0] == "real":
                errs.append("binomial(n, p): the count n is real-valued (hint: declare the count as int, or wrap "
                            "it in round(...))")
            elif ts[0] == "cat":
                errs.append("binomial(n, p): the count n is a category")
            return "int"
        if fn == "poisson":
            return "int"
        if fn == "bernoulli":
            return "bool"
        if fn in ("choice", "choice_w"):
            opts = ts if fn == "choice" else ts[0::2]
            if fn == "choice_w" and any(x == "cat" for x in ts[1::2]):
                errs.append("choice_w: weights must be numbers")
            if all(x == "cat" for x in opts):
                return "cat"
            t = opts[0]
            for x in opts[1:]:
                t = _join(t, x)
            return t
        if any(x == "cat" for x in ts):
            errs.append(f"{fn}(): arguments must be numbers")
        return "real"
    return None


_DECL_T = {"real": "real", "int": "int", "bool": "bool", "category": "cat"}


def _vtype(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, str):
        return "cat"
    if isinstance(v, int):
        return "int"
    return "real"


# ============================================================================ the document
def _prov_kind(com):
    c = (com or "").strip().lower()
    for k in ("stated", "derived", "assumed"):
        if c.startswith(k):
            return k
    for k in ("assumed", "stated", "derived"):
        if re.search(r"\b" + k + r"\b", c):
            return k
    return "none"


def _norm_lines(text_or_lines):
    if isinstance(text_or_lines, str):
        t = text_or_lines.replace("\r\n", "\n").replace("\r", "\n")
        inner = fenced_block(t, FENCE_LABEL)
        if inner is not None:
            t = inner
        out = t.split("\n")
    else:
        out = []
        for x in text_or_lines or []:
            out += str(x).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while out and not out[-1].strip():
        out.pop()
    return out


class Doc:
    """A parsed and statically checked document. `problems` lists every static problem with line numbers."""

    def __init__(self, lines):
        self.lines = _norm_lines(lines)
        self.text = "\n".join(self.lines)
        self.hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
        self.time = None
        self.seeds = DEFAULT_SEEDS
        self.targets, self.targets_line, self.targets_declared = [], None, False
        self.params, self.rules, self.nodes = {}, {}, {}
        self.events, self.checks, self.whatifs = [], [], []
        self.problems = []
        self._model = None
        self._parse()
        if self.lines:
            self._check()

    # ------------------------------------------------------------ helpers
    def prob(self, line, kind, msg, hint="", col=None):
        self.problems.append({"line": line, "col": col, "kind": kind, "msg": msg, "hint": hint})

    @property
    def decl(self):
        return set(self.params) | set(self.rules) | set(self.nodes)

    def kind_of(self, name):
        if name in self.params:
            return "param"
        if name in self.rules:
            return "rule"
        if name in self.nodes:
            return self.nodes[name]["kind"]
        return None

    def _expr_at(self, src, no, raw, what):
        try:
            return parse_expr(src)
        except _expr.ExprError as e:
            col = raw.find(src.strip()[:20]) + 1 if src.strip() else None
            self.prob(no, "syntax", f"{what}: {e.msg}" + (f" at column {e.col}" if e.col else ""),
                      e.hint or "check the expression syntax", col if col and col > 0 else None)
            return None

    def _name_ok(self, name, no):
        if name in RESERVED_NAMES:
            self.prob(no, "syntax", f"invalid name '{name}' (a keyword, function or t/dt)", "rename it")
            return False
        if name in self.decl:
            first = (self.params.get(name) or self.rules.get(name) or self.nodes.get(name))["line"]
            self.prob(no, "duplicate", f"'{name}' is declared twice (first on line {first}); this line is ignored",
                      "remove or rename one")
            return False
        return True

    # ------------------------------------------------------------ parsing
    def _parse(self):
        for no, raw in enumerate(self.lines, 1):
            code, com = _split_comment(raw)
            s = code.strip()
            if not s or s.startswith("```") or s.startswith("~~~"):
                continue
            try:
                self._line(no, raw, s, com)
            except Exception as e:  # never raise on a line
                self.prob(no, "syntax", f"could not parse the line ({type(e).__name__}: {e})", "see the syntax table")

    def _line(self, no, raw, s, com):
        low = s.lower()
        meta = {"line": no, "prov": com, "pk": _prov_kind(com),
                "unused_ok": bool(re.search(r"\bunused\s*:", com, re.I)),
                "noeffect_ok": bool(re.search(r"no effect expected\s*:", com, re.I))}
        if re.match(r"^time\s*:?\s*(steps|dt|unit)\b", low):
            if self.time is not None:
                self.prob(no, "duplicate", "a second time line; it is ignored", "one time line per document")
                return
            kv = dict((k.lower(), v) for k, v in re.findall(r"(steps|dt|unit)\s*[=:]?\s*([^\s,]+)", s, re.I))
            try:
                st = float(kv["steps"])
                dt = float(kv.get("dt", 1))
            except (KeyError, ValueError):
                self.prob(no, "syntax", "the time line needs steps=N (an integer) and optionally dt and unit",
                          "time steps=60 dt=1 unit=minute")
                return
            if st != int(st) or not 1 <= st <= MAX_STEPS or dt <= 0:
                self.prob(no, "syntax", f"steps must be an integer 1..{MAX_STEPS} and dt > 0",
                          "time steps=60 dt=1")
                return
            self.time = {"steps": int(st), "dt": int(dt) if dt == int(dt) else dt, "unit": kv.get("unit", "step"),
                         "line": no}
            return
        m = re.match(r"^seeds?\s*[=:]?\s*(\S+)\s*$", low)
        if m:
            try:
                n = int(float(m.group(1)))
            except ValueError:
                self.prob(no, "syntax", "seeds needs an integer", "seeds 200")
                return
            if not 1 <= n <= MAX_SEEDS:
                self.prob(no, "syntax", f"seeds must be 1..{MAX_SEEDS}", "seeds 200")
                n = max(1, min(MAX_SEEDS, n))
            self.seeds = n
            return
        if re.match(r"^targets?\s*:", low):
            self.targets_declared = True
            self.targets_line = no
            for x in s.split(":", 1)[1].split(","):
                x = x.strip()
                if not x:
                    continue
                if not NAME_RE.fullmatch(x):
                    self.prob(no, "syntax", f"bad target name {x!r}", "targets: a, b, c")
                elif x not in self.targets:
                    self.targets.append(x)
            return
        m = re.match(r"^(param|rule)\s+([A-Za-z_]\w*)\s*=\s*(.*)$", s, re.S)
        if m:
            kind, name, vsrc = m.group(1), m.group(2), m.group(3)
            if not self._name_ok(name, no):
                return
            try:
                v = _value(vsrc)
            except ValueError as e:
                self.prob(no, "syntax", f"bad value for {kind} '{name}': {e}",
                          "a number" + ("" if kind == "param" else ", true/false or 'mode'"))
                return
            if kind == "param" and not _isnum(v):
                self.prob(no, "type", f"param '{name}' must be a number (got {v!r})",
                          "use a rule for true/false or 'mode' switches")
                return
            if kind == "param" and isinstance(v, int) and re.search(r"[.eE]", vsrc):
                v = float(v)
            (self.params if kind == "param" else self.rules)[name] = dict(meta, value=v, src=vsrc.strip())
            return
        m = re.match(r"^event\s+([A-Za-z_]\w*)\s*:\s*(.*)$", s, re.S)
        if m:
            name, src = m.group(1), m.group(2).strip()
            if any(e["name"] == name for e in self.events):
                self.prob(no, "duplicate", f"event '{name}' declared twice; this line is ignored", "rename one")
                return
            ast = self._expr_at(src, no, s, f"event {name}") if src else None
            if not src:
                self.prob(no, "syntax", f"event '{name}' has no predicate", "event name: queue >= 10")
            self.events.append(dict(meta, name=name, src=src, ast=ast))
            return
        m = re.match(r"^check\s+([A-Za-z_]\w*)\s*:\s*(.*)$", s, re.S)
        if m:
            name, body = m.group(1), m.group(2).strip()
            if any(c["name"] == name for c in self.checks):
                self.prob(no, "duplicate", f"check '{name}' declared twice; this line is ignored", "rename one")
                return
            mm = CHECK_RE.match(body)
            if not mm:
                self.prob(no, "syntax", f"bad check '{name}'",
                          "check name: expr at T ~ value +- tol | check name: expr at T is true|false "
                          "(T = a step, final or any)")
                return
            src, at = mm.group(1).strip(), mm.group(2).lower()
            at = "final" if at == "end" else at if at in ("final", "any") else int(at)
            c = dict(meta, name=name, src=src, at=at)
            if mm.group(5):
                c.update(mode="bool", expect=mm.group(5).lower() == "true")
            else:
                try:
                    c.update(mode="num", value=float(mm.group(3)), tol=abs(float(mm.group(4))))
                except ValueError:
                    self.prob(no, "syntax", f"check '{name}': value and tolerance must be numbers",
                              "check name: queue at 30 ~ 8 +- 3")
                    return
            c["ast"] = self._expr_at(src, no, s, f"check {name}")
            self.checks.append(c)
            return
        m = re.match(r"^whatif\s+([A-Za-z_]\w*)\s*:\s*(.*)$", s, re.S)
        if m:
            self._whatif(no, s, m.group(1), m.group(2), meta)
            return
        if low.startswith(("param ", "rule ", "event ", "check ", "whatif ")):
            w = low.split()[0]
            hint = {"param": "param name = 1.5", "rule": "rule name = 2 | true | 'mode'",
                    "event": "event name: predicate", "check": "check name: expr at T ~ value +- tol",
                    "whatif": "whatif name: rule r = 3 ; targets a ; events e"}[w]
            self.prob(no, "syntax", f"bad {w} line", hint)
            return
        m = VAR_RE.match(s)
        if not m:
            self.prob(no, "syntax", "unrecognised line",
                      "lines are: time | seeds | targets: | param | rule | name : type [= v0] := expr | "
                      "name : type = v | name : type ~ dist(...) | event | check | whatif")
            return
        self._node(no, s, m, meta)

    def _node(self, no, s, m, meta):
        name, op = m.group(1), m.group(2)
        if not self._name_ok(name, no):
            return
        rest = s[m.end(1):].strip()
        if op == ":":
            rest = rest[1:]
        eq_src = None
        if ":=" in rest:
            left, eq_src = rest.split(":=", 1)
            eq_src = eq_src.strip()
        else:
            left = rest
        dist_src = val_src = None
        if "~" in left:
            left, dist_src = left.split("~", 1)
        if "=" in left:
            left, val_src = left.split("=", 1)
        words = left.split()
        typ = "real"
        if words and (words[0].lower() in TYPES or words[0].lower() in TYPE_SYNONYMS):
            w0 = words.pop(0).lower()
            typ = TYPE_SYNONYMS.get(w0, w0)
        spec = dict(meta, name=name, type=typ, unit=" ".join(words))
        if eq_src is not None:
            if dist_src is not None:
                self.prob(no, "syntax", f"'{name}' has both ~ and := (a node is either sampled once per run or "
                          "computed every step)", "use := with a per-step function, e.g. := normal(m, s)")
            spec["kind"] = "endo"
            spec["eq_src"] = eq_src
            if not eq_src:
                self.prob(no, "syntax", f"empty equation for '{name}'", f"{name} : real := <expression>")
                spec["eq_ast"] = None
            else:
                spec["eq_ast"] = self._expr_at(eq_src, no, s, f"equation of {name}")
            if val_src is not None:
                spec["v0_src"] = val_src.strip()
                spec["v0_ast"] = self._expr_at(val_src, no, s, f"initial value of {name}") if val_src.strip() \
                    else None
        elif dist_src is not None or val_src is not None:
            spec["kind"] = "exo"
            src = (dist_src if dist_src is not None else val_src).strip()
            spec["init_src"] = src
            spec["sampled"] = dist_src is not None
            spec["init_ast"] = None
            if not src:
                self.prob(no, "syntax", f"missing value for '{name}'", f"{name} : real = 5")
            elif typ == "category" and NAME_RE.fullmatch(src) and src not in ("true", "false") and dist_src is None \
                    and src not in self.params and src not in self.rules:
                spec["init_ast"] = ("str", src)
            else:
                spec["init_ast"] = self._expr_at(src, no, s, f"value of {name}")
        else:
            self.prob(no, "syntax", f"'{name}' has no value, distribution or equation",
                      f"{name} : real = 5 | {name} : real ~ normal(5, 1) | {name} : real := expr")
            return
        self.nodes[name] = spec

    def _whatif(self, no, s, name, body, meta):
        if any(w["name"] == name for w in self.whatifs):
            self.prob(no, "duplicate", f"whatif '{name}' declared twice; this line is ignored", "rename one")
            return
        wi = dict(meta, name=name, changes=[], targets=[], events=[], set={}, do={})
        for seg in _split_top(body, ";"):
            seg = seg.strip()
            if not seg:
                continue
            w = seg.split(None, 1)
            key, rest = w[0].lower(), (w[1] if len(w) > 1 else "")
            if key in ("targets", "target"):
                wi["targets"] += [x.strip() for x in rest.split(",") if x.strip()]
                continue
            if key in ("events", "event"):
                wi["events"] += [x.strip() for x in _split_top(rest, ",") if x.strip()]
                continue
            if key in ("seeds", "steps"):
                try:
                    wi[key] = int(float(rest))
                except ValueError:
                    self.prob(no, "syntax", f"whatif {name}: {key} needs an integer", f"{key} 100")
                continue
            parts = []
            for p in _split_top(seg, ","):
                if parts and not CHANGE_RE.match(p):
                    parts[-1] += "," + p          # a comma inside an expression (do x := if a then b, ...)
                else:
                    parts.append(p)
            for p in parts:
                mm = CHANGE_RE.match(p)
                if not mm:
                    self.prob(no, "syntax", f"whatif {name}: bad change {p.strip()!r}",
                              "rule r = 3 | param p = 0.5 | do x = 5 [from 10] | do x := expr")
                    continue
                kw, var, op, rest2 = mm.group(1), mm.group(2), mm.group(3), p[mm.end():].strip()
                ch = {"kw": kw, "name": var}
                if op == ":=":
                    ch["eq"] = rest2
                    ch["ast"] = self._expr_at(rest2, no, s, f"whatif {name}: equation for {var}") if rest2 else None
                    if not rest2:
                        self.prob(no, "syntax", f"whatif {name}: empty equation for {var}", "do x := expr")
                        continue
                else:
                    frm = 0
                    m3 = re.fullmatch(r"(.+?)\s+from\s+(\d+)", rest2, re.S)
                    if m3:
                        rest2, frm = m3.group(1), int(m3.group(2))
                    try:
                        ch["value"] = _value(rest2)
                    except ValueError as e:
                        self.prob(no, "syntax", f"whatif {name}: bad value for {var}: {e}", "a number, true/false or "
                                  "'mode'")
                        continue
                    ch["from"] = frm
                wi["changes"].append(ch)
        if not wi["changes"]:
            self.prob(no, "whatif", f"whatif '{name}' changes nothing (no change clause)",
                      "whatif name: rule r = 3 ; targets a")
        self.whatifs.append(wi)

    # ------------------------------------------------------------ static checks
    def _check(self):
        decl = self.decl
        known = sorted(decl)
        env = {n: _vtype(p["value"]) for n, p in self.params.items()}
        env.update({n: _vtype(r["value"]) for n, r in self.rules.items()})
        env.update({n: _DECL_T[s["type"]] for n, s in self.nodes.items()})
        reads = set()
        if self.time is None:
            self.prob(None, "time", "no time line", "time steps=60 dt=1 unit=minute")
            self.time = {"steps": 1, "dt": 1, "unit": "step", "line": None}
        if not self.targets_declared:
            self.prob(None, "targets", "no 'targets:' line (the quantities the question asks about)",
                      "targets: a, b, c")
        consts = set(self.params) | set(self.rules)

        def use(ast, no, where, allowed=None, owner_type=None):
            if ast is None:
                return
            for r, _ in sorted(expr_refs(ast)):
                reads.add(r)
                if r not in decl:
                    self.prob(no, "undefined", f"undefined name '{r}' in {where}{_did_you_mean(r, known)}",
                              "declare it (param, rule or node) or fix the spelling")
                elif allowed is not None and r not in allowed:
                    self.prob(no, "syntax", f"{where} may only use params, rules" +
                              (" and exogenous values" if "exo" in where or "initial" in where else "") +
                              f"; '{r}' is a node", "compute it with := instead")
            errs = []
            t = _typeof(ast, env, errs)
            if owner_type == "cat" and t in ("int", "real", "bool"):
                errs.append("a category node is given a numeric expression")
            elif owner_type in ("int", "real", "bool") and t == "cat":
                errs.append(f"a {owner_type} node is given a category expression (declare it as category)")
            for e in dict.fromkeys(errs):
                self.prob(no, "type", f"{where}: {e}", "")

        exo_names = {n for n, s in self.nodes.items() if s["kind"] == "exo"}
        for n, s in self.nodes.items():
            ot = _DECL_T[s["type"]]
            if s["kind"] == "endo":
                use(s.get("eq_ast"), s["line"], f"equation of {n}", owner_type=ot)
                if s.get("v0_ast") is not None:
                    use(s["v0_ast"], s["line"], f"initial value of {n}", allowed=consts | exo_names, owner_type=ot)
            else:
                use(s.get("init_ast"), s["line"], f"value of exo {n}", allowed=consts, owner_type=ot)
        steps = self.time["steps"]
        for e in self.events:
            use(e["ast"], e["line"], f"event {e['name']}")
            if e["ast"] is not None and has_random(e["ast"]):
                self.prob(e["line"], "syntax", f"event {e['name']}: random draws are not allowed in an event", "")
        for c in self.checks:
            use(c["ast"], c["line"], f"check {c['name']}")
            if c["ast"] is not None and has_random(c["ast"]):
                self.prob(c["line"], "syntax", f"check {c['name']}: random draws are not allowed in a check", "")
            if isinstance(c["at"], int) and c["at"] > steps:
                self.prob(c["line"], "syntax", f"check {c['name']}: step {c['at']} is after the last step {steps}",
                          f"use a step 0..{steps}, final or any")
        for t in self.targets:
            reads.add(t)
            if t not in decl:
                self.prob(self.targets_line, "undefined", f"undefined target '{t}'{_did_you_mean(t, known)}",
                          "every target must be a declared param, rule or node")
        ev_names = {e["name"] for e in self.events}
        for w in self.whatifs:
            where = f"whatif {w['name']}"
            for ch in w["changes"]:
                n = ch["name"]
                k = self.kind_of(n)
                if k is None:
                    self.prob(w["line"], "undefined", f"{where}: unknown name '{n}'{_did_you_mean(n, known)}",
                              "declared params/rules: " + (", ".join(sorted(consts))[:200] or "none"))
                    continue
                if "eq" in ch:
                    if k in ("param", "rule"):
                        self.prob(w["line"], "whatif", f"{where}: '{n}' is a {k}; set it with {k} {n} = value", "")
                        continue
                    use(ch["ast"], w["line"], f"{where} (equation for {n})", owner_type=_DECL_T[self.nodes[n]["type"]])
                    w["do"][n] = {"eq": ch["eq"], "ast": ch["ast"]}
                elif k in ("param", "rule"):
                    v = ch["value"]
                    err = _set_type_error(n, k, (self.params.get(n) or self.rules.get(n))["value"], v)
                    if err:
                        self.prob(w["line"], "type", f"{where}: {err}", "")
                        continue
                    if ch.get("from"):
                        self.prob(w["line"], "whatif", f"{where}: 'from' applies to do on a node, not to {k} {n}",
                                  "do x = v from k")
                    w["set"][n] = v
                else:
                    if ch["kw"] in ("param", "rule"):
                        pass                                   # a node named with param/rule: treat as do
                    w["do"][n] = {"value": ch["value"], "from": ch.get("from", 0)}
            for t in w["targets"]:
                reads.add(t)
                if t not in decl:
                    self.prob(w["line"], "undefined", f"{where}: unknown target '{t}'{_did_you_mean(t, known)}", "")
            for e in w["events"]:
                if e in ev_names:
                    continue
                try:
                    a = parse_expr(e)
                except _expr.ExprError as x:
                    self.prob(w["line"], "undefined", f"{where}: '{e}' is neither a declared event nor a predicate "
                              f"({x.msg}){_did_you_mean(e, ev_names)}", "events name1, name2")
                    continue
                use(a, w["line"], f"{where} (event {e})")
        # same-step cycles
        endo = [n for n, s in self.nodes.items() if s["kind"] == "endo"]
        deps = {n: sorted({r for r, lag in expr_refs(self.nodes[n]["eq_ast"]) if lag == 0 and r in self.nodes
                           and self.nodes[r]["kind"] == "endo"}) if self.nodes[n].get("eq_ast") else [] for n in endo}
        self.cycles = _cycles(endo, deps)
        for cyc in self.cycles:
            path = " -> ".join(cyc + [cyc[0]])
            self.prob(self.nodes[cyc[0]]["line"], "cycle", f"same-step cycle {path}",
                      "use a lag x[t-1] for one of these references")
        # declared but unread
        for n, p in list(self.params.items()) + list(self.rules.items()):
            if n not in reads and not p["unused_ok"]:
                k = "param" if n in self.params else "rule"
                self.prob(p["line"], "unread", f"{k} '{n}' is declared but no equation, check, event or what-if reads "
                          f"it", "use it in the equation it belongs to, or mark it '# unused: <reason>'")
        for n in exo_names:
            s = self.nodes[n]
            if n not in reads and not s["unused_ok"]:
                self.prob(s["line"], "unread", f"exogenous '{n}' is declared but nothing reads it",
                          "use it, or mark it '# unused: <reason>'")
        self.problems.sort(key=lambda p: (p["line"] is None and -1 or p["line"] or 0))

    # ------------------------------------------------------------ outputs
    def problem_text(self, p):
        loc = f"line {p['line']}" if p.get("line") else "document"
        if p.get("col"):
            loc += f" col {p['col']}"
        return f"{loc}: {p['msg']}" + (f" (hint: {p['hint']})" if p.get("hint") else "")

    def model(self):
        if self._model is None:
            self._model = Model(self)
        return self._model

    def export(self):
        def prov(d):
            return {n: x["prov"] for n, x in d.items() if x.get("prov")}
        return {"source": self.text, "lines": list(self.lines),
                "params": {n: p["value"] for n, p in self.params.items()},
                "rules": {n: r["value"] for n, r in self.rules.items()},
                "provenance": {**prov(self.params), **prov(self.rules), **prov(self.nodes)},
                "targets": list(self.targets),
                "whatifs": [{"name": w["name"], "line": self.lines[w["line"] - 1].strip()} for w in self.whatifs],
                "checks": [{"name": c["name"], "line": self.lines[c["line"] - 1].strip()} for c in self.checks],
                "events": [{"name": e["name"], "predicate": e["src"]} for e in self.events],
                "time": {k: v for k, v in (self.time or {}).items() if k != "line"}}


def _set_type_error(name, kind, cur, v):
    if kind == "param":
        return None if _isnum(v) else f"param '{name}' needs a number (got {v!r})"
    if isinstance(cur, bool):
        return None if isinstance(v, bool) else f"rule '{name}' is true/false (got {v!r})"
    if isinstance(cur, str):
        return None if isinstance(v, str) else f"rule '{name}' is a mode like '{cur}' (got {v!r})"
    return None if _isnum(v) else f"rule '{name}' needs a number (got {v!r})"


def _cycles(nodes, deps):
    """Strongly connected components with a cycle (Tarjan), each as a list in declaration order."""
    index, low, stack, on, out, k = {}, {}, [], set(), [], [0]

    def strong(v):
        index[v] = low[v] = k[0]
        k[0] += 1
        stack.append(v)
        on.add(v)
        for w in deps.get(v, []):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1 or v in deps.get(v, []):
                out.append([n for n in nodes if n in comp])
    for v in nodes:
        if v not in index:
            strong(v)
    return out


# ============================================================================ random streams (common random numbers)
M64 = (1 << 64) - 1
_INV53 = 1.0 / 9007199254740992.0
_ND = NormalDist()


def _mix(z):
    z = (z + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def _site_key(owner, idx):
    return int.from_bytes(hashlib.sha256(f"{owner}|{idx}".encode()).digest()[:8], "big")


def _seed_key(i):
    return _mix(0xC0FFEE ^ ((i * 0xD1B54A32D192ED03) & M64))


def _step_key(seedkey, t):
    return _mix(seedkey ^ (((t + 7) * 0x9E3779B97F4A7C15) & M64))


def _binom_inv(n, p, u):
    if n <= 0 or p <= 0:
        return 0
    if p >= 1:
        return n
    if p > 0.5:
        return n - _binom_inv(n, 1 - p, 1 - u)
    q = 1 - p
    var = n * p * q
    if var > 50:
        x = round(n * p + math.sqrt(var) * _ND.inv_cdf(u))
        return int(min(n, max(0, x)))
    pmf = q ** n
    c, k, r = pmf, 0, p / q
    while u > c and k < n:
        pmf *= (n - k) / (k + 1) * r
        k += 1
        c += pmf
    return k


def _poisson_inv(lam, u):
    if lam <= 0:
        return 0
    if lam > 50:
        return max(0, int(round(lam + math.sqrt(lam) * _ND.inv_cdf(u))))
    pmf = math.exp(-lam)
    c, k = pmf, 0
    while u > c and k < 10000:
        k += 1
        pmf *= lam / k
        c += pmf
    return k


# ============================================================================ compiled model and runs
class EvalError(_expr.EvalError):
    pass


def _numv(x, what):
    t = type(x)
    if t is float or t is int:
        return x
    if t is bool:
        return 1 if x else 0
    raise EvalError(f"{what}: expected a number, got {t.__name__} {x!r}")


def _truth(v):
    if isinstance(v, str):
        raise EvalError(f"a category value ({v!r}) cannot be used as a condition")
    return bool(v)


class _R:
    """The state of one replication."""
    __slots__ = ("t", "dt", "theta", "H", "failed", "exo", "v0", "seedkey", "stepkey", "rounds", "clips", "cur")

    def u(self, key):
        return ((_mix(key ^ self.stepkey) >> 11) + 0.5) * _INV53

    def why(self, name):
        if name in self.failed:
            return f"depends on '{name}', which has no value ({self.failed[name][1][:120]})"
        return f"'{name}' has no value at step {self.t}"

    def init(self, name):
        if name in self.v0:
            return self.v0[name]
        if name in self.exo and self.exo[name] is not None:
            return self.exo[name]
        if name in self.failed:
            raise EvalError(self.why(name))
        raise EvalError(f"{name}[t-k] reaches before step 0 and '{name}' has no initial value (hint: declare "
                        f"'{name} : type = v0 := ...')")

    def round_count(self, what):
        self.rounds[what] = self.rounds.get(what, 0) + 1


def _clip_p(R, p):
    p = _numv(p, "probability")
    if p < 0 or p > 1:
        R.clips += 1
        p = 0.0 if p < 0 else 1.0
    return p


def _coerce(R, name, typ, v):
    tv = type(v)
    if typ == "int":
        if tv is int:
            return v
        if tv is bool:
            return int(v)
        if tv is float:
            if v != v or v in (math.inf, -math.inf):
                raise EvalError(f"non-finite result {v}")
            if v.is_integer():
                return int(v)
            R.round_count(name)
            return int(round(v))
        raise EvalError(f"an int node got {v!r}")
    if typ == "real":
        if tv is float:
            if v != v or v in (math.inf, -math.inf):
                raise EvalError(f"non-finite result {v}")
            return v
        if tv is int or tv is bool:
            return float(v)
        raise EvalError(f"a real node got {v!r} (declare it as category?)")
    if typ == "bool":
        if tv is bool:
            return v
        if tv is int or tv is float:
            return v != 0
        raise EvalError(f"a bool node got {v!r}")
    if tv is str:
        return v
    raise EvalError(f"a category node got {v!r} (category values are 'text')")


class Model:
    """Closures compiled from a Doc; runs replications under common random numbers."""

    def __init__(self, doc):
        self.doc = doc
        self.nodes = doc.nodes
        self.theta0 = {**{n: p["value"] for n, p in doc.params.items()},
                       **{n: r["value"] for n, r in doc.rules.items()}}
        self.exo = [n for n, s in self.nodes.items() if s["kind"] == "exo"]
        self.endo = [n for n, s in self.nodes.items() if s["kind"] == "endo"]
        self.steps, self.dt = doc.time["steps"], doc.time["dt"]
        self.fn, self.v0fn, self.initfn = {}, {}, {}
        for n in self.endo:
            s = self.nodes[n]
            self.fn[n] = (self.compile(s["eq_ast"], n) if s.get("eq_ast") is not None
                          else _raiser(f"line {s['line']}: the equation of '{n}' does not parse"))
            if "v0_src" in s:
                self.v0fn[n] = (self.compile(s["v0_ast"], n + "@v0") if s.get("v0_ast") is not None
                                else _raiser(f"line {s['line']}: the initial value of '{n}' does not parse"))
        for n in self.exo:
            s = self.nodes[n]
            self.initfn[n] = (self.compile(s["init_ast"], n + "@init") if s.get("init_ast") is not None
                              else _raiser(f"line {s['line']}: the value of '{n}' does not parse"))
        self.pre_failed = {}
        for cyc in getattr(doc, "cycles", []):
            for n in cyc:
                self.pre_failed[n] = f"line {self.nodes[n]['line']}: same-step cycle " + " -> ".join(cyc + [cyc[0]])
        self._orders = {}
        self.events = {e["name"]: (self.compile(e["ast"], "@event") if e["ast"] is not None
                                   else _raiser(f"line {e['line']}: the event does not parse")) for e in doc.events}
        self.checks = {c["name"]: (self.compile(c["ast"], "@check") if c["ast"] is not None
                                   else _raiser(f"line {c['line']}: the check does not parse")) for c in doc.checks}

    # ------------------------------------------------------------ compiler
    def compile(self, ast, owner):
        c = lambda a: self.compile(a, owner)
        k = ast[0]
        if k == "num":
            v = ast[1]
            v = int(v) if float(v).is_integer() and abs(v) < 1e15 else v
            return lambda R: v
        if k in ("str", "bool"):
            v = ast[1]
            return lambda R: v
        if k == "var":
            return self._var(ast[1], ast[2])
        if k == "neg":
            f = c(ast[1])
            return lambda R: -_numv(f(R), "unary minus")
        if k == "not":
            f = c(ast[1])
            return lambda R: not _truth(f(R))
        if k == "and":
            fa, fb = c(ast[1]), c(ast[2])
            return lambda R: _truth(fa(R)) and _truth(fb(R))
        if k == "or":
            fa, fb = c(ast[1]), c(ast[2])
            return lambda R: _truth(fa(R)) or _truth(fb(R))
        if k == "if":
            fc, fa, fb = c(ast[1]), c(ast[2]), c(ast[3])
            return lambda R: fa(R) if _truth(fc(R)) else fb(R)
        if k == "bin":
            return _binop(ast[1], c(ast[2]), c(ast[3]))
        if k == "call":
            return _call(ast[1], [c(a) for a in ast[2]])
        if k == "rcall":
            return _rcall(ast[1], [c(a) for a in ast[2]], _site_key(owner, ast[3]))
        return _raiser(f"unknown expression node {k}")

    def _var(self, name, lag):
        if name == "t":
            return lambda R: R.t
        if name == "dt":
            return lambda R: R.dt
        if name in self.theta0:
            return lambda R: R.theta[name]
        if name in self.nodes:
            if lag == 0:
                def f(R):
                    v = R.H[name][R.t]
                    if v is None:
                        raise EvalError(R.why(name))
                    return v
                return f

            def g(R):
                ts = R.t - lag
                if ts < 0:
                    return R.init(name)
                v = R.H[name][ts]
                if v is None:
                    raise EvalError(R.why(name))
                return v
            return g
        return _raiser(f"undefined name '{name}'")

    # ------------------------------------------------------------ ordering
    def order_for(self, patched):
        """Same-step evaluation order of the per-step nodes, with `patched` {node: ast} equations replacing f."""
        key = tuple(sorted((n, repr(a)) for n, a in patched.items()))
        if key in self._orders:
            return self._orders[key]
        stepped = list(self.endo) + [n for n in self.exo if n in patched]
        asts = {n: patched.get(n, self.nodes[n].get("eq_ast")) for n in stepped}
        sset = set(stepped)
        deps = {n: {r for r, lag in expr_refs(a) if lag == 0 and r in sset and r != n} if a is not None else set()
                for n, a in asts.items()}
        self_loops = {n for n, a in asts.items() if a is not None and (n, 0) in expr_refs(a)}
        order, done, pending = [], set(), [n for n in stepped]
        while pending:
            progressed = False
            for n in list(pending):
                if deps[n] <= done:
                    order.append(n)
                    done.add(n)
                    pending.remove(n)
                    progressed = True
            if not progressed:
                break
        cyc = {}
        if pending or self_loops:
            for comp in _cycles(stepped, {n: sorted(deps[n] | ({n} if n in self_loops else set())) for n in stepped}):
                for n in comp:
                    cyc[n] = "same-step cycle " + " -> ".join(comp + [comp[0]])
            order += pending
        self._orders[key] = (order, cyc)
        return order, cyc

    # ------------------------------------------------------------ simulation
    def simulate(self, theta=None, do=None, steps=None, seeds=1, deadline=None):
        """Replications 0..seeds-1 (paired with any other call by seed index). Returns a Sim."""
        steps = self.steps if steps is None else steps
        th = dict(self.theta0)
        th.update(theta or {})
        do = do or {}
        vals = {n: d for n, d in do.items() if "value" in d}
        eqs = {n: self.compile(d["ast"], n) for n, d in do.items() if "ast" in d}
        order, cyc = self.order_for({n: d["ast"] for n, d in do.items() if "ast" in d})
        fns = dict(self.fn)
        fns.update(eqs)
        pre = dict(self.pre_failed)
        for n in eqs:
            pre.pop(n, None)
        pre.update({n: m for n, m in cyc.items() if n not in self.pre_failed})
        types = {n: s["type"] for n, s in self.nodes.items()}
        static_exo = [n for n in self.exo if n not in eqs]
        sim = Sim(steps)
        rounds = sim.rounds
        for i in range(seeds):
            if deadline is not None and i >= 2 and _time.time() > deadline:
                sim.time_hit = True
                break
            R = _R()
            R.t, R.dt, R.theta = 0, self.dt, th
            R.H = {n: [None] * (steps + 1) for n in self.nodes}
            R.failed = {n: (0, m) for n, m in pre.items()}
            R.exo, R.v0, R.rounds, R.clips = {}, {}, rounds, 0
            R.seedkey = _seed_key(i)
            R.stepkey = _step_key(R.seedkey, -1)
            for n in self.exo:
                R.cur = n
                try:
                    v = _coerce(R, n, types[n], self.initfn[n](R))
                except (_expr.EvalError, ArithmeticError, ValueError, TypeError) as e:
                    v = None
                    if n not in eqs:
                        R.failed[n] = (0, str(e))
                R.exo[n] = v
            for n in static_exo:
                iv = vals.get(n)
                row = R.H[n]
                base = R.exo[n]
                if iv is not None:
                    try:
                        dv = _coerce(R, n, types[n], iv["value"])
                    except _expr.EvalError as e:
                        R.failed[n] = (0, f"do {n}: {e}")
                        continue
                    k = iv.get("from", 0) or 0
                    for t in range(steps + 1):
                        row[t] = dv if t >= k else base
                    if k == 0:
                        R.exo[n] = dv
                        R.failed.pop(n, None)
                elif n not in R.failed:
                    for t in range(steps + 1):
                        row[t] = base
            for n, f in self.v0fn.items():
                R.cur = n
                try:
                    R.v0[n] = _coerce(R, n, types[n], f(R))
                except (_expr.EvalError, ArithmeticError, ValueError, TypeError) as e:
                    R.failed.setdefault(n, (0, f"initial value: {e}"))
            for n, iv in vals.items():
                if n in self.v0fn or n not in fns or (iv.get("from", 0) or 0) > 0:
                    continue
                try:
                    R.v0[n] = _coerce(R, n, types[n], iv["value"])
                except _expr.EvalError:
                    pass
            H, failed, v0 = R.H, R.failed, R.v0
            for t in range(steps + 1):
                R.t = t
                R.stepkey = _step_key(R.seedkey, t)
                for n in order:
                    if n in failed:
                        continue
                    R.cur = n
                    try:
                        iv = vals.get(n)
                        if iv is not None and t >= (iv.get("from", 0) or 0):
                            v = iv["value"]
                        elif t == 0 and n in self.v0fn:
                            if n not in v0:
                                raise EvalError("no initial value")
                            v = v0[n]
                        else:
                            v = fns[n](R)
                        H[n][t] = _coerce(R, n, types[n], v)
                    except (_expr.EvalError, ArithmeticError, ValueError, TypeError, RecursionError) as e:
                        failed[n] = (t, str(e)[:200])
            sim.clips += R.clips
            sim.reps.append(R)
        sim.work = len(sim.reps) * (steps + 1) * max(1, len(order))
        return sim


class Sim:
    def __init__(self, steps):
        self.steps = steps
        self.reps = []
        self.rounds = {}
        self.clips = 0
        self.time_hit = False
        self.work = 0


def _raiser(msg):
    def f(R):
        raise EvalError(msg)
    return f


def _binop(op, fa, fb):
    if op == "+":
        return lambda R: _numv(fa(R), "'+'") + _numv(fb(R), "'+'")
    if op == "-":
        return lambda R: _numv(fa(R), "'-'") - _numv(fb(R), "'-'")
    if op == "*":
        return lambda R: _numv(fa(R), "'*'") * _numv(fb(R), "'*'")
    if op == "/":
        def div(R):
            x, y = _numv(fa(R), "'/'"), _numv(fb(R), "'/'")
            if y == 0:
                raise EvalError("division by zero")
            return x / y
        return div
    if op == "%":
        def mod(R):
            x, y = _numv(fa(R), "'%'"), _numv(fb(R), "'%'")
            if y == 0:
                raise EvalError("modulo by zero")
            return x % y
        return mod
    if op == "^":
        def pw(R):
            x, y = float(_numv(fa(R), "'^'")), float(_numv(fb(R), "'^'"))
            if x < 0 and not y.is_integer():
                raise EvalError("negative number to a fractional power")
            if x == 0 and y < 0:
                raise EvalError("zero to a negative power")
            try:
                return math.pow(x, y)
            except OverflowError:
                raise EvalError("numeric overflow in '^'")
        return pw
    if op in ("==", "!="):
        eq = op == "=="

        def cmp_eq(R):
            a, b = fa(R), fb(R)
            r = (a == b) if (isinstance(a, str) or isinstance(b, str)) else (_numv(a, op) == _numv(b, op))
            return r if eq else not r
        return cmp_eq

    def cmp(R):
        a, b = fa(R), fb(R)
        if not (isinstance(a, str) and isinstance(b, str)):
            a, b = _numv(a, op), _numv(b, op)
        return a < b if op == "<" else a <= b if op == "<=" else a > b if op == ">" else a >= b
    return cmp


def _call(fn, fs):
    if fn in ("min", "max"):
        agg = min if fn == "min" else max
        return lambda R: agg(_numv(f(R), fn) for f in fs)
    f0 = fs[0]
    if fn == "abs":
        return lambda R: abs(_numv(f0(R), "abs"))
    if fn == "sign":
        return lambda R: (lambda x: (x > 0) - (x < 0))(_numv(f0(R), "sign"))
    if fn == "floor":
        return lambda R: math.floor(_numv(f0(R), "floor"))
    if fn == "ceil":
        return lambda R: math.ceil(_numv(f0(R), "ceil"))
    if fn == "round":
        if len(fs) == 1:
            return lambda R: round(_numv(f0(R), "round"))
        f1 = fs[1]
        return lambda R: float(round(_numv(f0(R), "round"), int(_numv(f1(R), "round digits"))))
    if fn == "clip":
        f1, f2 = fs[1], fs[2]
        return lambda R: min(max(_numv(f0(R), "clip"), _numv(f1(R), "clip lo")), _numv(f2(R), "clip hi"))
    if fn == "sqrt":
        def sq(R):
            x = _numv(f0(R), "sqrt")
            if x < 0:
                raise EvalError("sqrt of a negative number")
            return math.sqrt(x)
        return sq
    if fn == "exp":
        def ex(R):
            try:
                return math.exp(_numv(f0(R), "exp"))
            except OverflowError:
                raise EvalError("numeric overflow in exp()")
        return ex
    if fn == "log":
        def lg(R):
            x = _numv(f0(R), "log")
            if x <= 0:
                raise EvalError("log of a non-positive number")
            if len(fs) == 2:
                b = _numv(fs[1](R), "log base")
                if b <= 0 or b == 1:
                    raise EvalError("bad log base")
                return math.log(x, b)
            return math.log(x)
        return lg
    return _raiser(f"unknown function {fn}")


def _rcall(fn, fs, key):
    if fn == "bernoulli":
        f0 = fs[0]
        return lambda R: R.u(key) < _clip_p(R, f0(R))
    if fn == "binomial":
        f0, f1 = fs

        def binom(R):
            n = _numv(f0(R), "binomial count")
            if n < 0:
                raise EvalError(f"binomial(n, p): n is negative ({n})")
            if type(n) is float:
                if not n.is_integer():
                    R.round_count(f"binomial count in {R.cur}")
                n = int(round(n))
            p = _clip_p(R, f1(R))
            return _binom_inv(n, p, R.u(key))
        return binom
    if fn == "poisson":
        f0 = fs[0]

        def pois(R):
            lam = _numv(f0(R), "poisson rate")
            if lam < 0:
                raise EvalError(f"poisson(rate): negative rate ({lam})")
            return _poisson_inv(lam, R.u(key))
        return pois
    if fn == "normal":
        f0, f1 = fs

        def norm(R):
            m, s = _numv(f0(R), "normal mean"), _numv(f1(R), "normal sd")
            if s < 0:
                raise EvalError("normal(m, s): negative sd")
            return m + s * _ND.inv_cdf(R.u(key))
        return norm
    if fn == "uniform":
        f0, f1 = fs
        return lambda R: (lambda a, b: a + (b - a) * R.u(key))(_numv(f0(R), "uniform"), _numv(f1(R), "uniform"))
    if fn == "exponential":
        f0 = fs[0]

        def expo(R):
            r = _numv(f0(R), "exponential rate")
            if r <= 0:
                raise EvalError("exponential(rate): rate must be > 0")
            return -math.log(1.0 - R.u(key)) / r
        return expo
    if fn == "choice":
        def ch(R):
            u = R.u(key)
            return fs[min(len(fs) - 1, int(u * len(fs)))](R)
        return ch
    if fn == "choice_w":
        def chw(R):
            ws = [_numv(fs[j](R), "choice_w weight") for j in range(1, len(fs), 2)]
            if any(w < 0 for w in ws) or sum(ws) <= 0:
                raise EvalError("choice_w: weights must be >= 0 with a positive sum")
            x, c = R.u(key) * sum(ws), 0.0
            for j, w in enumerate(ws):
                c += w
                if x < c:
                    return fs[2 * j](R)
            return fs[2 * (len(ws) - 1)](R)
        return chw
    return _raiser(f"unknown function {fn}")


# ============================================================================ statistics and formatting
def _stats(xs):
    xs = sorted(xs)
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0

    def q(p):
        return xs[min(n - 1, max(0, int(math.ceil(p * n)) - 1))]
    return {"mean": m, "sd": sd, "p05": q(0.05), "p50": q(0.5), "p95": q(0.95), "n": n}


def _g(v):
    if isinstance(v, str):
        return repr(v)
    if _isnum(v) and 1000 <= abs(v) < 1e7:
        return f"{v:.0f}"                      # 2027, 1540 rather than 2.03e+03, 1.54e+03
    return _g3(v)


def _dist_text(xs, typ):
    """Distribution of per-seed values (None = failed)."""
    ok = [x for x in xs if x is not None]
    if not ok:
        return "no value"
    nf = len(xs) - len(ok)
    tail = f" ({nf} seeds failed)" if nf else ""
    if isinstance(ok[0], str):
        cnt = {}
        for x in ok:
            cnt[x] = cnt.get(x, 0) + 1
        return ", ".join(f"'{k}' {v / len(ok):.2f}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:6]) + tail
    if all(isinstance(x, bool) for x in ok):
        return f"P(true) {sum(ok) / len(ok):.2f}" + tail
    s = _stats([float(x) for x in ok])
    if s["sd"] == 0:
        return f"{_g(ok[0])} on all {len(ok)} seeds" + tail
    out = (f"mean {_g(s['mean'])} sd {_g(s['sd'])} [p05 {_g(s['p05'])} | p50 {_g(s['p50'])} | "
           f"p95 {_g(s['p95'])}]")
    if typ == "int":
        out += f"; P(>0) {sum(1 for x in ok if x > 0) / len(ok):.2f}"
    return out + tail


def _cut(text, limit):
    if len(text) <= limit:
        return text
    out, n = [], 0
    for ln in text.split("\n"):
        if n + len(ln) + 1 > limit - 34:
            out.append("  [... cut to the output bound]")
            break
        out.append(ln)
        n += len(ln) + 1
    return "\n".join(out)


def _bound(text, limit=MAX_OUT):
    return text if len(text) <= limit else text[:limit - 40] + "\n[... output cut to the bound]"


def _fmt_val(v):
    return f"'{v}'" if isinstance(v, str) else ("true" if v is True else "false" if v is False else _g(v))


def _desc_changes(spec):
    parts = [f"{n}={_fmt_val(v)}" for n, v in spec.get("set", {}).items()]
    for n, d in spec.get("do", {}).items():
        if "ast" in d:
            parts.append(f"do {n} := {d['eq']}")
        else:
            parts.append(f"do {n}={_fmt_val(d['value'])}" + (f" from {d['from']}" if d.get("from") else ""))
    return ", ".join(parts) or "no change"


# ============================================================================ the arm
def _tool(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


class G2C1State(ArmState):
    def __init__(self, rep=None, given_gold=False):
        super().__init__(rep=rep if rep is not None else {}, given_gold=given_gold)
        self.doc = None
        self.accepted = False          # the stored document passed (or was run past) the rejection rule
        self.ran_hash = None           # text hash of the last baseline that ran without a run-time error
        self.ran_any = None            # (hash, at least one target computed)
        self.unresolved = []           # problems the document ran with (escape valve), incl. no-op what-ifs
        self.rej_sig, self.rej_count = None, 0


class G2C1Arm(Arm):
    name = "G2C1"
    buildable = True
    executable = True
    guide_file = "g2c1_guide.md"
    mutating_tools = MUTATING
    custom_episode = True

    # ------------------------------------------------------------ tools
    def tools(self):
        run_item = {"description": "a stored whatif name, or {name?, set?: {param or rule: value}, do?: {node: value "
                                   "| 'v from k' | ':= expr'}, steps?, seeds?, targets?: [..], events?: [..], "
                                   "at?: 'final'|'all'|step}"}
        return [
            _tool("submit", "Store the whole document and check it; every problem is listed at once with line "
                  "numbers. If accepted, runs the baseline, every stored whatif and every check in the same call. "
                  "Send `lines`: an array of strings, one document line each (single quotes only).",
                  {"lines": {"type": "array", "items": {"type": "string"}}, "source": {"type": "string"}}),
            _tool("edit", "Apply ALL edits in one call to the numbered document last shown, then re-check and re-run "
                  "as submit. Each edit: {line: n, set: [lines]} (replace line n; [] deletes it), "
                  "{after: n, insert: [lines]} (0 = top), or {old: text, new: text} (exact unique replacement).",
                  {"edits": {"type": "array", "items": {"type": "object"}}}, ["edits"]),
            _tool("run", "Run up to 6 scenarios on the stored document, paired with the baseline under common "
                  "random numbers (no runs = the baseline report).",
                  {"runs": {"type": "array", "items": run_item}}),
            _tool("show", "The numbered document, the parameter/rule table with provenance and the dependency "
                  "summary.", {}),
        ]

    def probe_tools(self):
        return [t for t in self.tools() if t["function"]["name"] in ("run", "show")]

    # ------------------------------------------------------------ state
    def new_state(self):
        st = G2C1State()
        self._set_doc(st, Doc([]))
        return st

    def load(self, rep, given_gold=False):
        st = G2C1State(given_gold=given_gold)
        if isinstance(rep, dict) and isinstance(rep.get("source"), str):
            doc = Doc(rep["source"])
        elif isinstance(rep, dict) and isinstance(rep.get("lines"), list):
            doc = Doc(rep["lines"])
        elif isinstance(rep, list):
            doc = Doc(rep)
        elif isinstance(rep, str):
            doc = Doc(rep)
        else:
            doc = Doc([])
            st.log.append(("load", None, f"representation was {type(rep).__name__}; started empty"))
        self._set_doc(st, doc)
        st.accepted = not doc.problems
        return st

    def _set_doc(self, st, doc):
        st.doc = doc
        st.rep = doc.export()

    def _doc(self, st):
        if getattr(st, "doc", None) is None:
            r = st.rep if isinstance(st.rep, dict) else {}
            st.doc = Doc(r.get("source") if isinstance(r.get("source"), str) else r.get("lines") or [])
        return st.doc

    def export(self, state):
        return self._doc(state).export()

    # ------------------------------------------------------------ dispatch
    def handle(self, state, name, args):
        try:
            if isinstance(args, str):
                args, _ = preparse_args(args)
                if args is None:
                    return f"{ERROR_PREFIX}: the arguments are not a valid JSON object (hint: {ARG_HINT})"
            args = args or {}
            if not isinstance(args, dict):
                return f"{ERROR_PREFIX}: arguments must be an object (hint: see the {name} schema)"
            if name not in {t["function"]["name"] for t in self.tools()}:
                return (f"{ERROR_PREFIX}: unknown tool '{name}' (hint: available: "
                        + ", ".join(t["function"]["name"] for t in self.tools()) + ")")
            if state.given_gold and name in MUTATING:
                return f"{ERROR_PREFIX}: the representation is read-only in this mode (hint: use run and show)"
            out = getattr(self, "_t_" + name)(state, args)
        except Exception as e:  # never raise
            out = f"{ERROR_PREFIX}: internal {type(e).__name__}: {e} (hint: check the arguments)"
        state.log.append((name, args if isinstance(args, dict) else str(args)[:200], out[:200]))
        return out

    # ------------------------------------------------------------ submit / edit
    def _t_submit(self, state, args):
        lines, src = args.get("lines"), args.get("source")
        if isinstance(lines, str):
            try:
                v = json.loads(lines)
                lines = v if isinstance(v, list) else lines
            except ValueError:
                pass
        if isinstance(lines, list) and lines:
            doc = Doc([x if isinstance(x, str) else json.dumps(x) for x in lines])
        elif isinstance(lines, str) and lines.strip():
            doc = Doc(lines)
        elif isinstance(src, str) and src.strip():
            doc = Doc(src)
        else:
            return f"{ERROR_PREFIX}: no document (hint: {ARG_HINT})"
        if not doc.lines:
            return f"{ERROR_PREFIX}: the document is empty (hint: {ARG_HINT})"
        return self._accept(state, doc, "submitted")

    def _t_edit(self, state, args):
        doc = self._doc(state)
        if not doc.lines:
            return f"{ERROR_PREFIX}: no document to edit (hint: submit the whole document first)"
        edits = args.get("edits")
        if isinstance(edits, str):
            try:
                edits = json.loads(edits)
            except ValueError:
                return f"{ERROR_PREFIX}: edits must be a list of edit objects (hint: {{\"line\": 3, \"set\": [...]}})"
        if isinstance(edits, dict):
            edits = [edits]
        if not isinstance(edits, list) or not edits:
            if any(k in args for k in ("line", "after", "old")):
                edits = [args]
            else:
                return (f"{ERROR_PREFIX}: edits must be a non-empty list (hint: [{{\"line\": 3, \"set\": "
                        f"[\"param a = 2\"]}}, {{\"after\": 5, \"insert\": [\"...\"]}}, {{\"old\": \"x\", \"new\": \"y\"}}])")
        lines = list(doc.lines)
        n = len(lines)
        line_edits, text_edits = [], []
        for k, e in enumerate(edits, 1):
            if not isinstance(e, dict):
                return f"{ERROR_PREFIX}: edit {k} is not an object; nothing was applied"
            if "old" in e:
                if not isinstance(e.get("old"), str) or not e["old"] or not isinstance(e.get("new", ""), str):
                    return f"{ERROR_PREFIX}: edit {k}: old must be non-empty text and new text; nothing was applied"
                text_edits.append((k, e["old"], e.get("new", "")))
                continue
            new = e.get("set", e.get("insert", e.get("lines")))
            if isinstance(new, str):
                new = [new]
            if new is None or not isinstance(new, list):
                return (f"{ERROR_PREFIX}: edit {k}: give set: [lines] with line, or insert: [lines] with after; "
                        f"nothing was applied")
            new = [str(x) for x in new]
            if "line" in e:
                try:
                    ln = int(e["line"])
                except (TypeError, ValueError):
                    return f"{ERROR_PREFIX}: edit {k}: line must be a number; nothing was applied"
                if not 1 <= ln <= n:
                    return f"{ERROR_PREFIX}: edit {k}: line {ln} is outside 1..{n}; nothing was applied"
                line_edits.append((ln, 1, k, new))
            elif "after" in e:
                try:
                    ln = int(e["after"])
                except (TypeError, ValueError):
                    return f"{ERROR_PREFIX}: edit {k}: after must be a number; nothing was applied"
                if not 0 <= ln <= n:
                    return f"{ERROR_PREFIX}: edit {k}: after {ln} is outside 0..{n}; nothing was applied"
                line_edits.append((ln, 0, k, new))
            else:
                return f"{ERROR_PREFIX}: edit {k} needs line, after or old; nothing was applied"
        seen = [x[0] for x in line_edits if x[1] == 1]
        if len(seen) != len(set(seen)):
            return f"{ERROR_PREFIX}: two edits replace the same line; combine them; nothing was applied"
        for ln, is_set, k, new in sorted(line_edits, key=lambda x: (x[0], x[1]), reverse=True):
            if is_set:
                lines[ln - 1:ln] = new
            else:
                lines[ln:ln] = new
        text = "\n".join(lines)
        for k, old, new in text_edits:
            c = text.count(old)
            if c != 1:
                near = difflib.get_close_matches(old.strip().split("\n")[0], [x.strip() for x in doc.lines], n=1,
                                                 cutoff=0.4)
                return (f"{ERROR_PREFIX}: edit {k}: 'old' occurs {c} times" + (f" (closest line: {near[0]!r})" if near
                                                                              and c == 0 else "")
                        + "; nothing was applied (hint: copy it exactly and make it unique, or use line edits)")
            text = text.replace(old, new, 1)
        return self._accept(state, Doc(text), f"{len(edits)} edit(s) applied")

    def _accept(self, state, doc, how):
        """The rejection rule, then the full run report."""
        self._set_doc(state, doc)
        state.accepted, state.unresolved = False, []
        probs = list(doc.problems)
        if probs:
            sig = _sig(probs)
            if not (state.rej_sig == sig and state.rej_count >= 2):
                return self._reject(state, doc, probs, how)
            state.unresolved = probs
        sec, noops = self._report(state, doc, base_note=how, sens=True)
        if noops and not state.unresolved:
            sig = _sig(noops)
            if not (state.rej_sig == sig and state.rej_count >= 2):
                return self._reject(state, doc, noops, how)
        state.unresolved = state.unresolved + noops
        state.accepted = True
        if not state.unresolved:
            state.rej_sig, state.rej_count = None, 0
        return self._text(state, doc, sec, stored=True)

    def _text(self, state, doc, sec, stored):
        if stored and state.unresolved:
            u = [f"UNRESOLVED ({len(state.unresolved)} problems; run anyway because the same problems were rejected "
                 f"twice):"] + [f"  - {doc.problem_text(p)}" for p in state.unresolved[:12]]
            sec = dict(sec, HEADER="\n".join(u) + "\n" + sec["HEADER"].replace("; 0 problems", ""))
        return _assemble(sec)

    def _reject(self, state, doc, probs, how):
        sig = _sig(probs)
        state.rejected = list(probs)
        state.rej_count = state.rej_count + 1 if state.rej_sig == sig else 1
        state.rej_sig = sig
        txt = [f"{ERROR_PREFIX}: submission rejected ({len(probs)} problems; {how}, {len(doc.lines)} lines stored as a "
               f"draft, not run)"]
        txt += [f"  - {doc.problem_text(p)}" for p in probs[:40]]
        if len(probs) > 40:
            txt.append(f"  ... {len(probs) - 40} more")
        txt.append("Fix them all in ONE edit call (line numbers refer to the document as stored; see show)."
                   + (" The same problems a third time run anyway, listed as UNRESOLVED." if state.rej_count >= 2
                      else ""))
        return _bound("\n".join(txt))

    # ------------------------------------------------------------ run / show
    def _t_run(self, state, args):
        doc = self._doc(state)
        if not doc.lines:
            return f"{ERROR_PREFIX}: no document yet (hint: submit the whole document first)"
        if not state.given_gold and not state.accepted:
            probs = getattr(state, "rejected", None) or doc.problems
            return (f"{ERROR_PREFIX}: the stored document is a rejected draft ({len(probs)} problems: "
                    + "; ".join(doc.problem_text(p) for p in probs[:3])[:400] + "); fix them with edit first")
        runs = args.get("runs")
        if isinstance(runs, str):
            try:
                runs = json.loads(runs)
            except ValueError:
                runs = [runs]
        if isinstance(runs, dict):
            runs = [runs]
        if runs is None:
            extra = {k: args[k] for k in ("set", "do", "name", "steps", "seeds", "targets", "events", "at") if k in args}
            runs = [extra] if extra else []
        if not isinstance(runs, list):
            return f"{ERROR_PREFIX}: runs must be a list (hint: runs=[\"stored_whatif\", {{\"set\": {{\"p\": 2}}}}])"
        note = ""
        if len(runs) > MAX_RUNS:
            note = f"note: only the first {MAX_RUNS} of {len(runs)} runs were run (limit per call)\n"
            runs = runs[:MAX_RUNS]
        specs, errs = [], []
        for k, r in enumerate(runs, 1):
            s = self._run_spec(doc, r, k)
            if isinstance(s, str):
                errs.append(s)
            else:
                specs.append(s)
        if errs:
            return _bound(f"{ERROR_PREFIX}: " + "; ".join(errs) + _declared(doc))
        if not specs:
            sec, _ = self._report(state, doc, base_note="baseline report", sens=True)
        else:
            sec, _ = self._report(state, doc, base_note="run", runs=specs, sens=False)
        return _bound(note + self._text(state, doc, sec, stored=False))

    def _run_spec(self, doc, r, k):
        stored = {w["name"]: w for w in doc.whatifs}
        if isinstance(r, str):
            if r not in stored:
                return (f"run {k}: '{r}' is not a stored whatif{_did_you_mean(r, stored)} (stored: "
                        f"{', '.join(stored) or 'none'})")
            r = {"name": r}
        if not isinstance(r, dict):
            return f"run {k}: must be a whatif name or an object"
        name = str(r.get("name") or f"run{k}")
        spec = {"name": name, "set": {}, "do": {}, "targets": [], "events": [], "at": "final"}
        if name in stored and not r.get("set") and not r.get("do"):
            w = stored[name]
            spec.update(set=dict(w["set"]), do=copy.deepcopy(w["do"]), targets=list(w["targets"]),
                        events=list(w["events"]))
            for key in ("seeds", "steps"):
                if key in w:
                    spec[key] = w[key]
        sets = r.get("set") or {}
        do = r.get("do") or {}
        if isinstance(sets, str):
            try:
                sets = json.loads(sets)
            except ValueError:
                return f"run {k}: set must be an object {{name: value}}"
        if isinstance(do, str):
            try:
                do = json.loads(do)
            except ValueError:
                return f"run {k}: do must be an object {{node: value}}"
        if not isinstance(sets, dict) or not isinstance(do, dict):
            return f"run {k}: set and do must be objects"
        known = sorted(doc.decl)
        for n, v in sets.items():
            kind = doc.kind_of(n)
            v = _jvalue(v)
            if kind in ("param", "rule"):
                cur = (doc.params.get(n) or doc.rules.get(n))["value"]
                err = _set_type_error(n, kind, cur, v)
                if err:
                    return f"run {k}: {err}"
                spec["set"][n] = v
            elif kind is None:
                return f"run {k}: set: '{n}' is not a declared param or rule{_did_you_mean(n, known)}"
            else:
                spec["do"][n] = {"value": v, "from": 0}
        for n, v in do.items():
            kind = doc.kind_of(n)
            if kind is None:
                return f"run {k}: do: '{n}' is not declared{_did_you_mean(n, known)}"
            frm = 0
            if isinstance(v, dict):
                if "eq" in v or "expr" in v:
                    v = ":= " + str(v.get("eq", v.get("expr")))
                else:
                    frm = int(v.get("from") or 0)
                    v = v.get("value")
            if isinstance(v, str) and v.strip().startswith(":="):
                if kind in ("param", "rule"):
                    return f"run {k}: do: '{n}' is a {kind}; give it a value with set"
                src = v.strip()[2:].strip()
                try:
                    a = parse_expr(src)
                except _expr.ExprError as e:
                    return f"run {k}: do {n} := {src}: {e}"
                und = [x for x, _ in expr_refs(a) if x not in doc.decl]
                if und:
                    return f"run {k}: do {n}: undefined name '{und[0]}'{_did_you_mean(und[0], known)}"
                spec["do"][n] = {"eq": src, "ast": a}
                continue
            if isinstance(v, str):
                m = re.fullmatch(r"\s*(.+?)\s+from\s+(\d+)\s*", v)
                if m:
                    v, frm = m.group(1), int(m.group(2))
                try:
                    v = _value(v)
                except ValueError as e:
                    return f"run {k}: do {n}: {e}"
            if kind in ("param", "rule"):
                cur = (doc.params.get(n) or doc.rules.get(n))["value"]
                err = _set_type_error(n, kind, cur, v)
                if err:
                    return f"run {k}: {err}"
                spec["set"][n] = v
                continue
            typ = doc.nodes[n]["type"]
            if (typ == "category") != isinstance(v, str) or v is None:
                return f"run {k}: do {n}: value {v!r} does not fit the {typ} node"
            spec["do"][n] = {"value": v, "from": frm}
        for key in ("targets", "events"):
            x = r.get(key)
            if isinstance(x, str):
                x = [y.strip() for y in x.split(",") if y.strip()]
            if x:
                spec[key] = [str(y) for y in x]
        bad = [t for t in spec["targets"] if t not in doc.decl]
        if bad:
            return f"run {k}: unknown target '{bad[0]}'{_did_you_mean(bad[0], known)}"
        for key, hi in (("steps", MAX_STEPS), ("seeds", MAX_SEEDS)):
            if r.get(key) is not None:
                try:
                    spec[key] = max(1, min(hi, int(r[key])))
                except (TypeError, ValueError):
                    return f"run {k}: {key} must be an integer"
        at = r.get("at", "final")
        if isinstance(at, str) and at.strip().lstrip("-").isdigit():
            at = int(at)
        if at not in ("final", "all") and not (isinstance(at, int) and not isinstance(at, bool)):
            return f"run {k}: at must be 'final', 'all' or a step number"
        spec["at"] = at
        return spec

    def _t_show(self, state, args):
        doc = self._doc(state)
        if not doc.lines:
            return "no document yet (submit the whole document as lines)"
        out = [self.render(state)]
        if doc.problems:
            out.append(f"PROBLEMS ({len(doc.problems)}):")
            out += [f"  - {doc.problem_text(p)}" for p in doc.problems[:30]]
        elif state.unresolved:
            out.append("UNRESOLVED: " + "; ".join(doc.problem_text(p) for p in state.unresolved[:10]))
        m = doc.model()
        order, _ = m.order_for({})
        out.append("DEPENDENCIES (same-step evaluation order; reads, lag k = x[t-k]):")
        for n in order:
            a = doc.nodes[n].get("eq_ast")
            rs = sorted(expr_refs(a)) if a is not None else []
            out.append(f"  {n} <- " + (", ".join(f"{x}[t-{lg}]" if lg else x for x, lg in rs) or "(none)"))
        return _bound("\n".join(out))

    def render(self, state):
        doc = self._doc(state)
        out = ["DOCUMENT (sse)"]
        out += [f"{i:3d}| {ln}" for i, ln in enumerate(doc.lines, 1)]
        head = "\n".join(out)
        out = []
        rows = []
        for n, p in doc.params.items():
            rows.append(("param", n, p))
        for n, r in doc.rules.items():
            rows.append(("rule", n, r))
        for n, s in doc.nodes.items():
            if s["kind"] == "exo":
                rows.append(("exo", n, dict(s, value=s.get("init_src"))))
        if rows:
            out.append("PARAMETERS AND RULES (kind | name | value | provenance; ASSUMED = not stated or derived)")
            for k, n, x in rows:
                v = x["value"] if k == "exo" else _fmt_val(x["value"])
                tag = "ASSUMED" if x["pk"] not in ("stated", "derived") else x["pk"]
                out.append(f"  {k} | {n} | {v} | {tag}" + (f" | {x['prov'][:110]}" if x.get("prov") else ""))
        table = "\n".join(out)
        if len(head) + len(table) + 1 <= RENDER_MAX:          # the document is the representation: the table is dropped first
            return head + ("\n" + table if table else "")
        if len(head) <= RENDER_MAX:
            return head + "\n[parameter table omitted to fit the render bound; provenance is in the line comments]"
        return head[:RENDER_MAX - 60] + "\n[... document cut to the render bound]"

    def validity(self, state):
        doc = self._doc(state)
        holes = [doc.problem_text(p) for p in doc.problems]
        holes += [doc.problem_text(p) for p in state.unresolved if p.get("kind") == "noop"]
        v = {"usable": False, "executable": False, "holes": holes, "errors": [], "targets": list(doc.targets),
             "problems": len(doc.problems), "accepted": bool(state.accepted)}
        if not doc.lines:
            v["errors"].append("no document")
            return v
        if state.ran_hash == doc.hash:
            v["executable"] = True
            v["usable"] = True
            return v
        try:
            m = doc.model()
            sim = m.simulate(steps=min(2, m.steps), seeds=2, deadline=_time.time() + 10)
            failed = set()
            for R in sim.reps:
                failed |= set(R.failed)
            tg = [t for t in doc.targets if t in doc.decl]
            computed = [t for t in tg if t in m.theta0 or t not in failed]
            v["usable"] = bool(computed)
            v["executable"] = bool(computed) and not failed
            if failed:
                R = sim.reps[0]
                v["errors"] += [f"{n}: {R.failed[n][1]}" for n in sorted(R.failed)[:10]]
        except Exception as e:
            v["errors"].append(f"smoke run failed: {type(e).__name__}: {e}")
        return v

    # ------------------------------------------------------------ the run report
    def _report(self, state, doc, base_note, runs=None, sens=True):
        """(report text, no-op problems). runs=None: the stored what-ifs (submit/edit/baseline report)."""
        t0 = _time.time()
        deadline = t0 + WALL_S
        m = doc.model()
        steps = m.steps
        stored_mode = runs is None
        notes = []
        if stored_mode:
            runs = []
            for w in doc.whatifs[:MAX_RUNS]:
                s = {"name": w["name"], "set": dict(w["set"]), "do": w["do"], "targets": list(w["targets"]),
                     "events": list(w["events"]), "at": "final", "line": w["line"], "noeffect_ok": w["noeffect_ok"]}
                for key in ("seeds", "steps"):
                    if key in w:
                        s[key] = max(1, min(MAX_SEEDS if key == "seeds" else MAX_STEPS, w[key]))
                runs.append(s)
            if len(doc.whatifs) > MAX_RUNS:
                notes.append(f"only the first {MAX_RUNS} stored whatifs were run; run others with run")
        seeds = doc.seeds
        assumed = self._assumed(doc)
        sens_params = [n for n, kind, x in assumed if kind == "param" and _isnum(x["value"]) and x["value"] != 0][
            :SENS_MAX] if sens else []
        n_nodes = max(1, len(doc.nodes))
        # compute plan: baseline per distinct steps value, each run, sensitivity runs
        plan = {"base": seeds * (steps + 1)}
        for r in runs:
            st = r.get("steps", steps)
            plan[r["name"]] = r.get("seeds", seeds) * (st + 1) * (2 if st != steps else 1)
        if sens_params:
            plan["sens"] = 2 * len(sens_params) * min(SENS_SEEDS, seeds) * (steps + 1)
        total = sum(plan.values()) * n_nodes
        f = 1.0
        if total > WORK_CAP:
            f = WORK_CAP / total
            notes.append(f"compute cap: seeds reduced to {max(2, int(seeds * f))} of {seeds} (about "
                         f"{WORK_CAP:,} node evaluations per call)")
        bseeds = max(2, int(seeds * f)) if f < 1 else seeds
        base = m.simulate(seeds=bseeds, deadline=deadline)
        bases = {steps: base}
        results = []
        for r in runs:
            st = r.get("steps", steps)
            rs = max(2, int(r.get("seeds", seeds) * f)) if f < 1 else r.get("seeds", seeds)
            if st not in bases or len(bases[st].reps) < rs:
                bases[st] = m.simulate(steps=st, seeds=rs, deadline=deadline)
            sim = m.simulate(theta=r["set"], do=r["do"], steps=st, seeds=rs, deadline=deadline)
            results.append((r, sim, bases[st]))
        noops = []
        if stored_mode:
            for r, sim, b in results:
                k = min(len(sim.reps), len(b.reps))
                same = all(sim.reps[i].H == b.reps[i].H and set(sim.reps[i].failed) == set(b.reps[i].failed)
                           for i in range(k))
                if same and not r.get("noeffect_ok"):
                    what = ", ".join(list(r["set"]) + list(r["do"])) or "nothing"
                    noops.append({"line": r.get("line"), "col": None, "kind": "noop",
                                  "msg": f"no-op what-if {r['name']}: setting {what} changes nothing; the equations do "
                                         f"not read {what} where it matters",
                                  "hint": "make the mechanism depend on it, or mark the line "
                                          "'# no effect expected: <reason>'"})
        sens_res = {}
        if sens_params:
            ss = max(2, int(min(SENS_SEEDS, seeds) * f)) if f < 1 else min(SENS_SEEDS, seeds)
            for p in sens_params:
                v = doc.params[p]["value"]
                sens_res[p] = [(fac, m.simulate(theta={p: v * fac}, seeds=ss, deadline=deadline)) for fac in (0.5, 2)]
        if base.time_hit or any(s.time_hit for _, s, _ in results):
            notes.append(f"time guard ({WALL_S:.0f} s): some runs stopped early; seeds completed are shown")
        if base.time_hit:
            pass
        if not any(R.failed for R in base.reps):
            state.ran_hash = doc.hash
        # ---------------------------------------------------------------- sections
        sec = {}
        h = []
        if stored_mode:
            h.append(f"ACCEPTED ({base_note}): {len(doc.lines)} lines; {len(doc.params)} params, {len(doc.rules)} "
                     f"rules, {len(doc.nodes)} nodes, {len(doc.events)} events, {len(doc.checks)} checks, "
                     f"{len(doc.whatifs)} whatifs; 0 problems")
        tm = doc.time
        h.append(f"BASELINE: steps 0..{steps} (dt {tm['dt']} {tm['unit']}), seeds {len(base.reps)} of {seeds}; "
                 + _rounding_text(base))
        h += [f"  note: {x}" for x in notes]
        errs = _errors_text(doc, base)
        if errs:
            h.append("RUN-TIME ERRORS (baseline):")
            h += errs
        sec["HEADER"] = "\n".join(h)
        # ASSUMED
        if stored_mode or sens:
            a = []
            if assumed:
                a.append("ASSUMED (no stated or derived source; report these with your numbers):")
                for n, kind, x in assumed[:20]:
                    v = x.get("init_src") if kind == "exo" else _fmt_val(x["value"])
                    a.append(f"  {kind} {n} = {v} (line {x['line']}" + (f"; {x['prov'][:80]}" if x.get("prov") else
                                                                         "; no provenance") + ")")
                    if n in sens_res:
                        a.append("    " + _sens_text(doc, base, sens_res[n]))
                if len(assumed) > 20:
                    a.append(f"  ... {len(assumed) - 20} more")
            else:
                a.append("ASSUMED: none (every param, rule and exogenous value is stated or derived)")
            sec["ASSUMED"] = "\n".join(a)
        else:
            sec["ASSUMED"] = ("ASSUMED: " + ", ".join(f"{n}={_fmt_val(x['value']) if kind != 'exo' else x.get('init_src')}"
                                                      for n, kind, x in assumed[:12])) if assumed else ""
        # TARGETS
        t = [f"TARGETS (baseline, final step {steps}):"]
        for n in doc.targets:
            t.append(f"  {n}: " + _target_text(doc, m, base, n, steps))
        if not doc.targets:
            t.append("  (no targets declared)")
        sec["TARGETS"] = "\n".join(t)
        # CHECKS
        if doc.checks:
            cols = [("baseline", base)] + [(r["name"], sim) for r, sim, _ in results]
            c = ["CHECKS (model mean [p05, p95] at the stated time vs value +- tol; bool: share of seeds matching, "
                 "OK if >= 0.5):"]
            miss = []
            for ck in doc.checks:
                cells = []
                for cname, sim in cols:
                    cell, ok = _check_cell(m, ck, sim)
                    cells.append(f"{cname[:18]}: {cell}")
                    if cname == "baseline" and ok is False:
                        miss.append(ck["name"])
                spec = (f"~ {_g(ck['value'])} +- {_g(ck['tol'])}" if ck["mode"] == "num"
                        else f"is {'true' if ck['expect'] else 'false'}")
                c.append(f"  {ck['name']} ({ck['src'][:50]} at {ck['at']} {spec}): " + " | ".join(cells))
            if miss:
                c.append(f"  BASELINE MISS ({', '.join(miss)}): the baseline does not reproduce a stated observation; "
                         "check the mechanism or the encoding before answering")
            sec["CHECKS"] = "\n".join(c)
        # EFFECTS
        if results:
            e = ["EFFECTS (paired with the baseline under common random numbers: same seeds, same draws):"]
            for r, sim, b in results:
                e += _effects_text(doc, m, r, sim, b)
            sec["EFFECTS"] = "\n".join(e)
        # EVENTS
        if doc.events:
            ev = ["EVENTS (baseline: P(ever), P(at end), first step it holds among seeds where it occurs):"]
            for e_ in doc.events:
                ev.append(f"  {e_['name']} ({e_['src'][:50]}): " + _event_text(m, e_["name"], base))
            sec["EVENTS"] = "\n".join(ev)
        # SERIES
        sr = ["SERIES (mean over seeds; step:value, a-b:v = steps a..b all v)"]
        for n in doc.targets:
            if n in doc.nodes:
                row, shape = compact_series(_mean_series(base, n))
                sr.append(f"  {n}: {row}" + (f"\n    {shape}" if shape else ""))
        for r, sim, b in results:
            tg = r["targets"] or doc.targets
            if r.get("at") == "all" or stored_mode or not runs:
                for n in tg:
                    if n in doc.nodes:
                        row, shape = compact_series(_mean_series(sim, n))
                        sr.append(f"  [{r['name']}] {n}: {row}" + (f"\n    {shape}" if shape else ""))
        if len(sr) > 1:
            sec["SERIES"] = "\n".join(sr)
        return sec, noops

    def _assumed(self, doc):
        out = []
        for n, p in doc.params.items():
            if p["pk"] not in ("stated", "derived"):
                out.append((n, "param", p))
        for n, r in doc.rules.items():
            if r["pk"] not in ("stated", "derived"):
                out.append((n, "rule", r))
        for n, s in doc.nodes.items():
            if s["kind"] == "exo" and s["pk"] not in ("stated", "derived"):
                out.append((n, "exo", s))
        return out

    # ------------------------------------------------------------ episode loop (argument channel)
    def run_episode(self, ep, state, messages):
        """G1C1's loop structure: system-prompt parity, budget line on every result, reserved answer call; plus the
        arm's own argument pre-parse (strict JSON, strict=False, lenient recovery) and the untaught fenced ```sse
        fallback (implicit submit while no document has been accepted)."""
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
            tcs = (m.get("tool_calls") or []) if tools else []
            am = _assistant(m)
            if not tcs:
                am.pop("tool_calls", None)
                messages.append(am)
                block = None if (gold or state.accepted or not tools) else fenced_block(content, FENCE_LABEL)
                if block is None or not block.strip():
                    return am["content"], messages, trace
                if n_tools >= MAX_TOOL_CALLS:
                    res = f"{ERROR_PREFIX}: tool-call limit ({MAX_TOOL_CALLS}) reached; give your FINAL ANSWER."
                else:
                    res = self.handle(state, "submit", {"source": block})
                n_tools += 1
                res = res + budget()
                trace.append({"turn": len(ep.calls), "tool": "submit",
                              "args": {"source": "fenced block", "chars": len(block)}, "result": res[:3000],
                              "error": res.startswith(ERROR_PREFIX)})
                messages.append({"role": "user", "content": "EXECUTOR RESULT (submit)\n" + res})
                continue
            messages.append(am)
            offered = {t["function"]["name"] for t in tools}
            for tc in tcs:
                fn = tc.get("function", {}) or {}
                name = fn.get("name", "")
                args, how = preparse_args(fn.get("arguments", "{}"))
                if args is None:
                    res = (f"{ERROR_PREFIX}: the arguments of {name} are not a valid JSON object ({how}) "
                           f"(hint: {ARG_HINT})")
                    if n_tools >= MAX_TOOL_CALLS:
                        res = f"{ERROR_PREFIX}: tool-call limit ({MAX_TOOL_CALLS}) reached; give your FINAL ANSWER."
                    targs = fn.get("arguments")
                else:
                    tc2 = dict(tc, function=dict(fn, arguments=args))
                    name, targs, res = execute_tool_call(self, state, tc2, offered, n_tools, MAX_TOOL_CALLS)
                    if how == "repaired":
                        res = "note: arguments repaired (lenient parse)\n" + res
                n_tools += 1
                res = res + budget()
                ent = {"turn": len(ep.calls), "tool": name, "args": targs, "result": res[:3000],
                       "error": res.startswith(ERROR_PREFIX)}
                if how == "repaired":
                    ent["repaired"] = True
                trace.append(ent)
                messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call{len(trace)}", "content": res})
        ep.forced_final = True
        messages.append({"role": "user", "content": FORCE_FINAL_MSG})
        m = ep.call(messages, tools=None, label="forced_final", final=True)
        messages.append({"role": "assistant", "content": m.get("content") or ""})
        return m.get("content") or "", messages, trace


# ============================================================================ report helpers
def _sig(probs):
    return tuple(sorted((p["kind"], p["msg"]) for p in probs))


def _declared(doc):
    ps = ", ".join(list(doc.params)[:30]) or "none"
    rs = ", ".join(list(doc.rules)[:30]) or "none"
    ns = ", ".join(list(doc.nodes)[:40]) or "none"
    return f" (declared params: {ps}; rules: {rs}; nodes: {ns})"


def _jvalue(v):
    if isinstance(v, str):
        try:
            return _value(v)
        except ValueError:
            return v
    return v


def _rounding_text(sim):
    r = {k: v for k, v in sim.rounds.items()}
    rt = ("int roundings: " + ", ".join(f"{k} {v}" for k, v in sorted(r.items(), key=lambda kv: -kv[1])[:6])
          if r else "int roundings: none")
    return rt + f"; probabilities clipped to [0, 1]: {sim.clips}"


def _errors_text(doc, sim):
    agg = {}
    for R in sim.reps:
        for n, (t, msg) in R.failed.items():
            a = agg.setdefault(n, {"n": 0, "t": t, "msg": msg})
            a["n"] += 1
            a["t"] = min(a["t"], t)
    if not agg:
        return []
    roots = sorted(agg, key=lambda n: (agg[n]["msg"].startswith("depends on"), agg[n]["t"]))
    out = []
    for n in roots[:6]:
        a = agg[n]
        ln = doc.nodes.get(n, {}).get("line")
        out.append(f"  line {ln} {n}: {a['msg'][:160]} (in {a['n']}/{len(sim.reps)} seeds, from t={a['t']})")
    if len(roots) > 6:
        out.append(f"  ... {len(roots) - 6} more nodes without a value")
    return out


def _vals(m, sim, name, at):
    out = []
    for R in sim.reps:
        if name in R.theta:
            out.append(R.theta[name])
            continue
        h = R.H.get(name)
        if h is None:
            out.append(None)
            continue
        k = len(h) - 1 if at in ("final", "all") else at
        out.append(h[k] if 0 <= k < len(h) else None)
    return out


def _target_text(doc, m, sim, n, at):
    if n not in doc.decl:
        return "undefined"
    if n in m.theta0:
        return f"{_fmt_val(sim.reps[0].theta[n] if sim.reps else m.theta0[n])} ({doc.kind_of(n)})"
    xs = _vals(m, sim, n, "final" if at == m.steps else at)
    out = _dist_text(xs, doc.nodes[n]["type"])
    if out == "no value" and sim.reps:
        R = sim.reps[0]
        if n in R.failed:
            out += f" ({R.failed[n][1][:120]})"
    return out


def _mean_series(sim, name):
    if not sim.reps:
        return []
    L = len(sim.reps[0].H[name])
    out = []
    for t in range(L):
        xs = [R.H[name][t] for R in sim.reps if R.H[name][t] is not None and not isinstance(R.H[name][t], str)]
        out.append(sum(float(x) for x in xs) / len(xs) if xs else None)
    while out and out[-1] is None:
        out.pop()
    return out


def _check_eval(m, name, R, t):
    R.t = t
    return m.checks[name](R)


def _check_cell(m, ck, sim):
    """(text, ok) for one check on one run."""
    if not sim.reps:
        return "no seeds", None
    steps = sim.steps
    at = ck["at"]
    if isinstance(at, int) and at > steps:
        return "n/a (after the last step)", None
    per, err = [], None
    for R in sim.reps:
        try:
            if at == "any":
                hit = False
                for t in range(steps + 1):
                    try:
                        v = _check_eval(m, ck["name"], R, t)
                    except _expr.EvalError:
                        continue
                    if ck["mode"] == "num":
                        if _isnum(v) or isinstance(v, bool):
                            if abs(float(v) - ck["value"]) <= ck["tol"]:
                                hit = True
                                break
                    elif bool(v) == ck["expect"]:
                        hit = True
                        break
                per.append(hit)
            else:
                v = _check_eval(m, ck["name"], R, steps if at == "final" else at)
                if ck["mode"] == "num":
                    per.append(float(_numv(v, "check")))
                else:
                    per.append(_truth(v) == ck["expect"])
        except (_expr.EvalError, ArithmeticError, ValueError, TypeError) as e:
            err = err or str(e)
    if not per:
        return f"n/a ({(err or 'no value')[:80]})", None
    if at == "any" or ck["mode"] == "bool":
        share = sum(per) / len(per)
        ok = share >= 0.5
        return f"{share:.2f} of seeds {'OK' if ok else 'MISS'}", ok
    s = _stats(per)
    ok = abs(s["mean"] - ck["value"]) <= ck["tol"]
    return f"{_g(s['mean'])} [{_g(s['p05'])}, {_g(s['p95'])}] {'OK' if ok else 'MISS'}", ok


def _event_series(m, name, R):
    out = []
    for t in range(len(next(iter(R.H.values()))) if R.H else 1):
        R.t = t
        try:
            out.append(bool(_truth(m.events[name](R))) if name in m.events else False)
        except (_expr.EvalError, ArithmeticError, ValueError, TypeError):
            out.append(None)
    return out


def _event_stats(m, name, sim):
    ever, end, first = 0, 0, []
    n = 0
    for R in sim.reps:
        s = _event_series(m, name, R)
        if all(x is None for x in s):
            continue
        n += 1
        if any(s):
            ever += 1
            first.append(next(i for i, x in enumerate(s) if x))
        if s and s[-1]:
            end += 1
    return n, ever, end, first


def _event_text(m, name, sim):
    n, ever, end, first = _event_stats(m, name, sim)
    if not n:
        return "undefined (no value)"
    out = f"P(ever) {ever / n:.2f}, P(at end) {end / n:.2f}"
    if first:
        s = _stats(first)
        out += (f", first t mean {_g(s['mean'])} [p05 {_g(s['p05'])} | p50 {_g(s['p50'])} | p95 {_g(s['p95'])}] "
                f"(n={len(first)})")
    return out


def _pred_event(m, src):
    """An ad-hoc event predicate (a what-if's events entry that is not a declared event)."""
    key = "@" + src
    if key not in m.events:
        try:
            m.events[key] = m.compile(parse_expr(src), "@event")
        except _expr.ExprError as e:
            m.events[key] = _raiser(str(e))
    return key


def _effects_text(doc, m, r, sim, base):
    tg = r["targets"] or doc.targets
    at = r.get("at", "final")
    k = min(len(sim.reps), len(base.reps))
    at_txt = f"at step {at}" if isinstance(at, int) else f"final step {sim.steps}"
    out = [f"  {r['name']} [{_desc_changes(r)}] ({k} paired seeds, {at_txt}):"]
    pick = "final" if at in ("final", "all") else at
    for n in tg:
        if n not in doc.decl:
            out.append(f"    {n}: undefined")
            continue
        a = _vals(m, sim, n, pick)[:k]
        b = _vals(m, base, n, pick)[:k]
        typ = doc.nodes[n]["type"] if n in doc.nodes else "real"
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if pairs and all(_isnum(x) and _isnum(y) for x, y in pairs):
            d = [float(x) - float(y) for x, y in pairs]
            sd = _stats(d)
            ma = sum(float(x) for x, _ in pairs) / len(pairs)
            mb = sum(float(y) for _, y in pairs) / len(pairs)
            out.append(f"    {n}: {_g(mb)} -> {_g(ma)}; diff mean {_g(sd['mean'])} [p05 {_g(sd['p05'])} | p95 "
                       f"{_g(sd['p95'])}]; P(diff>0) {sum(1 for x in d if x > 0) / len(d):.2f}, "
                       f"P(diff<0) {sum(1 for x in d if x < 0) / len(d):.2f}")
        else:
            out.append(f"    {n}: baseline {_dist_text(b, typ)} -> {_dist_text(a, typ)}")
    evs = r["events"] or [e["name"] for e in doc.events]
    for e in evs[:8]:
        key = e if e in m.events else _pred_event(m, e)
        nb, eb, endb, _ = _event_stats(m, key, base)
        na, ea, enda, fa = _event_stats(m, key, sim)
        if nb and na:
            out.append(f"    P(ever {e}) {eb / nb:.2f} -> {ea / na:.2f} (diff {ea / na - eb / nb:+.2f}); "
                       f"P(at end) {endb / nb:.2f} -> {enda / na:.2f}")
        else:
            out.append(f"    event {e}: undefined")
    errs = _errors_text(doc, sim)
    if errs and _errors_text(doc, base) != errs:
        out.append("    run-time errors in this run:")
        out += ["  " + x for x in errs[:3]]
    if sim.rounds != base.rounds and sim.rounds:
        out.append("    " + _rounding_text(sim))
    return out


def _sens_text(doc, base, res):
    m = doc.model()
    num_t = [n for n in doc.targets if n in doc.nodes and doc.nodes[n]["type"] in ("int", "real")][:4]
    if not num_t:
        return "halve/double: no numeric target"
    parts = []
    for fac, sim in res:
        k = min(len(sim.reps), len(base.reps))
        eff = []
        for n in num_t:
            a = _vals(m, sim, n, "final")[:k]
            b = _vals(m, base, n, "final")[:k]
            d = [float(x) - float(y) for x, y in zip(a, b) if _isnum(x) and _isnum(y)]
            eff.append(f"{n} {sum(d) / len(d):+.3g}" if d else f"{n} n/a")
        parts.append(f"x{_g(fac)} -> " + ", ".join(eff))
    return f"halve/double ({min(len(s.reps) for _, s in res)} paired seeds, change in final mean): " + "; ".join(parts)


def _assemble(sec):
    """Blocks in priority order within MAX_OUT; lower blocks are cut or dropped first, and that is stated."""
    blocks = [(k, sec[k]) for k in SECTION_ORDER if sec.get(k)]
    limit = MAX_OUT - 120
    total = sum(len(b) + 1 for _, b in blocks)
    dropped, cut = [], []
    i = len(blocks) - 1
    while total > limit and i > 0:
        k, b = blocks[i]
        over = total - limit
        if len(b) - over >= 400 and k not in ("SERIES",):
            nb = _cut(b, len(b) - over)
            blocks[i] = (k, nb)
            cut.append(k)
            total -= len(b) - len(nb)
        else:
            blocks.pop(i)
            dropped.append(k)
            total -= len(b) + 1
        i -= 1
    text = "\n".join(b for _, b in blocks)
    if dropped or cut:
        text += ("\n[report bound 7,800 characters: " + "; ".join(
            ([f"dropped {', '.join(reversed(dropped))}"] if dropped else [])
            + ([f"cut {', '.join(reversed(cut))}"] if cut else [])) + "; use run with fewer runs or show]")
    return _bound(text)


# ============================================================================ argument pre-parse (lenient)
def _unesc(s):
    return s.replace('\\"', '"').replace("\\t", "\t").replace("\\\\", "\\")


def lenient_args(raw):
    """Recover {"lines": [...]} or {"source": "..."} from a broken JSON argument string, or None."""
    m = re.search(r'"(lines|source)"\s*:\s*', raw)
    if not m:
        return None
    key, rest = m.group(1), raw[m.end():].lstrip()
    if key == "lines":
        if not rest.startswith("["):
            return None
        body = rest[1:]
        end = body.rfind('"]')
        if end < 0:
            end = body.rfind('"')
            if end < 0:
                return None
        body = body[:end + 1].strip()
        if body.startswith('"'):
            body = body[1:]
        if body.endswith('"'):
            body = body[:-1]
        parts = re.split(r'"\s*,\s*"', body)
        if len(parts) == 1 and "\n" in body:
            parts = body.split("\n")
        out = []
        for p in parts:
            out += _unesc(p).replace("\\n", "\n").split("\n")
        return {"lines": out} if any(x.strip() for x in out) else None
    if not rest.startswith('"'):
        return None
    body = rest[1:]
    end = body.rfind('"}')
    if end < 0:
        end = body.rfind('"')
    if end < 0:
        return None
    body = _unesc(body[:end]).replace("\\n", "\n")
    return {"source": body} if body.strip() else None


def preparse_args(raw):
    """(dict, how) with how in (None, 'repaired'), or (None, reason)."""
    if isinstance(raw, dict):
        return raw, None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if not isinstance(raw, str):
        return None, f"arguments are a {type(raw).__name__}"
    try:
        v = json.loads(raw)
        return (v, None) if isinstance(v, dict) else (None, "arguments must be a JSON object")
    except ValueError:
        pass
    try:
        v = json.loads(raw, strict=False)
        if isinstance(v, dict):
            return v, "repaired"
    except ValueError:
        pass
    v = lenient_args(raw)
    if v is not None:
        return v, "repaired"
    return None, "not valid JSON and no lines/source payload could be recovered"


G2C1 = G2C1Arm
