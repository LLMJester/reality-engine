"""G2C3: evidence-conditioned model ensemble (spec: runs/discovery/gen2/specs/G2C3.md).

Representation: a line-oriented `ece` document (single quotes, `#` comments). Stated numbers are `param` lines,
stated switches `rule` lines, open or vague quantities `unknown name ~ dist`, competing explanations
`hypothesis name: 'a' w | 'b' w`, mechanisms are node lines `name : type [= v0] := expr` (the G1C3/A2 syntax: lags,
step-0 rule), stated observations are `fact` lines, the question's conclusions `claim` lines and stored interventions
`whatif` lines.

Semantics: the document denotes a model family (prior pi over unknowns x hypotheses; aleatory per-step noise inside
each member) conditioned on the facts, P(u, eps | F) ~ pi(u) P(eps) prod_j L_j(u, eps). The executor approximates it
by importance sampling (randomly shifted Halton draws for continuous unknowns, stratified categorical draws, one
aleatory replicate per member, vectorised over members with NumPy) with one Gaussian-kernel refinement round when
ESS < 50, and reports functionals of the conditioned family: HYPOTHESES, TARGETS, EFFECTS (paired, common random
numbers, evidence-weighted), SENSITIVITY (first-order variance shares), CLAIMS (probability, threshold stump,
PRIM-lite box), FACTS (pass rates) and INCOHERENT (minimal unsatisfiable fact sets: exact with z3 in static mode,
empirical via the MUS/MCS hitting-set duality in dynamic mode).

Runs in process (no subprocess). Its only runtime import outside the standard library, NumPy and z3 is `_expr`
(declared in RUNTIME_DEPS for the final lock, D-060).

Build record (spec section 9): all blocks were delivered. That covers the vectorised evaluator, conditioning with
one refinement round, and the HYPOTHESES / TARGETS / EFFECTS / FACTS / SENSITIVITY blocks. It also covers the static
INCOHERENT block (z3, MARCO-style, up to 5 MUSes, 10 s budget, dynamic fallback), CLAIMS (stump + PRIM-lite) and the
dynamic INCOHERENT block (hitting-set duality plus bounded local search).

Interpretation choices beyond the spec text:
- The given-data sensitivity estimator is bias-corrected by (bins - 1) / ESS.
- The no-op hypothesis check compares targets, facts and claims ("where it matters").
- In read-only (gold) mode, a `run` entry without changes returns the full report.
- A lag x[t-k] needs an initial value only if its line is evaluated before step k (A2 step-0 rule).
"""
import copy
import difflib
import hashlib
import json
import math
import re
import time
import zlib
from functools import reduce

import numpy as np

from tournament.arms import _expr
from tournament.arms.base import ERROR_PREFIX, Arm, ArmState

# Runtime dependency (D-060): the executor runs in process; it imports the shared expression parser.
RUNTIME_DEPS = ["tournament/arms/_expr.py"]

FENCE_LABEL = "ece"
MAX_OUT = 7800                 # tool result bound; the episode adds the budget line (runner cap 8,000)
DEFAULT_SAMPLES = 2000
MAX_SAMPLES = 4000
MIN_SAMPLES = 50
MAX_STEPS = 5000
COMPUTE_CAP = 2e7              # members x steps x nodes per tool call
WALL_S = 45.0                  # wall-clock guard per tool call
MIN_ESS = 50.0
MAX_RUNS = 6
MAX_MUS_FACTS = 16
Z3_BUDGET_S = 10.0
LOCAL_SEARCH_RUNS = 200
SEED = 7919
BUDGET_LINE = "[budget: {k} of {n} responses used; your last response must be the FINAL ANSWER without tool calls]"
ARG_HINT = "send the document as `lines`, an array of strings; use single quotes only"
MUTATING = ("submit", "edit")
TYPES = {"real": "real", "float": "real", "number": "real", "num": "real", "int": "int", "integer": "int",
         "count": "int", "bool": "bool", "boolean": "bool", "category": "category", "cat": "category",
         "string": "category", "enum": "category"}
STOCH = {"bernoulli": (1, 1), "binomial": (2, 2), "poisson": (1, 1), "normal": (2, 2), "uniform": (2, 2),
         "exponential": (1, 1), "choice": (1, 99), "choice_w": (2, 198)}
Z3_FUNCS = {"min", "max", "abs", "floor", "ceil", "clip"}
CONT_DISTS = ("uniform", "loguniform", "triangular", "normal")
LINE_KEYWORDS = {"time", "samples", "targets", "target", "param", "rule", "unknown", "hypothesis", "fact", "claim",
                 "whatif"}
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CMP_OPS = ("<=", ">=", "==", "!=", "<", ">")


class _Timeout(Exception):
    pass


# ============================================================================ small helpers
def _split_comment(line):
    """(code, comment) splitting at the first '#' outside quotes."""
    q = None
    for i, c in enumerate(line):
        if q:
            if c == q:
                q = None
        elif c in "'\"":
            q = c
        elif c == "#":
            return line[:i], line[i + 1:].strip()
    return line, ""


def _split_top(s, sep):
    """Split at `sep` (one character) outside quotes, parentheses and brackets."""
    out, depth, q, cur = [], 0, None, []
    for c in s:
        if q:
            if c == q:
                q = None
        elif c in "'\"":
            q = c
        elif c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == sep and depth == 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(c)
    out.append("".join(cur))
    return out


def _dym(name, known):
    m = difflib.get_close_matches(str(name), [str(k) for k in known], n=2, cutoff=0.6)
    return f" (did you mean {' or '.join(repr(x) for x in m)}?)" if m else ""


def _g(v):
    """Compact number: 3 significant figures."""
    if v is None:
        return "-"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(v):
        return str(v)
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return str(int(round(v)))
    s = f"{v:.3g}"
    return "0" if s in ("-0", "-0.0") else s


def _pct(p):
    return f"{100 * p:.1f}%" if p >= 0.001 or p == 0 else f"{100 * p:.2f}%"


def _prob(line, msg, hint="", fatal=True):
    return {"line": line, "msg": msg, "hint": hint, "fatal": fatal}


def fenced_block(text, label=FENCE_LABEL):
    """The last block fenced with ```label (or ~~~label); a missing closing fence runs to the end."""
    if not isinstance(text, str) or not text:
        return None
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    found, i = None, 0
    while i < len(lines):
        m = re.match(r"^\s*(```+|~~~+)\s*([A-Za-z0-9_+-]*)\s*$", lines[i])
        if not m:
            i += 1
            continue
        fence, lab = m.group(1), m.group(2)
        j = i + 1
        while j < len(lines) and not re.match(r"^\s*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$",
                                              lines[j]):
            j += 1
        if lab == label:
            found = "\n".join(lines[i + 1:j])
        i = j + 1
    return found


# ============================================================================ expressions
class _EParser(_expr._Parser):
    """_expr's parser plus the per-step stochastic functions (FUNCS / arity are module-level in _expr)."""

    def call(self, fn, col):
        if fn not in STOCH:
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
        if fn == "choice" and not all(a[0] == "str" for a in args):
            raise _expr.ExprError("choice() takes quoted values", col, hint="choice('a', 'b')")
        if fn == "choice_w":
            if len(args) % 2 or not all(a[0] == "str" for a in args[0::2]):
                raise _expr.ExprError("choice_w() takes value, weight pairs", col, hint="choice_w('a', 0.7, 'b', 0.3)")
        return ("call", fn, args)


def parse_expr(src):
    try:
        return _EParser(src).parse()
    except RecursionError:
        raise _expr.ExprError("expression nested too deeply")


def _has_stoch(ast):
    if not isinstance(ast, tuple):
        return False
    if ast[0] == "call":
        return ast[1] in STOCH or any(_has_stoch(a) for a in ast[2])
    return any(_has_stoch(a) for a in ast[1:] if isinstance(a, tuple))


def _const(src, bare_word=False):
    s = src.strip()
    if not s:
        raise ValueError("missing value")
    if bare_word and NAME_RE.fullmatch(s) and s not in ("true", "false"):
        return s
    try:
        a = parse_expr(s)
    except _expr.ExprError as e:
        raise ValueError(str(e))
    if _expr.refs(a) or _has_stoch(a):
        raise ValueError(f"a value must be a constant ({s!r})")

    def lookup(name, lag):
        raise _expr.EvalError(f"'{name}' is not available in a constant")
    try:
        return _expr.evaluate(a, lookup)
    except _expr.EvalError as e:
        raise ValueError(str(e))


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ============================================================================ distributions
def _parse_dist(src):
    m = re.fullmatch(r"\s*([A-Za-z_]+)\s*\((.*)\)\s*", src, re.S)
    if not m:
        raise ValueError("expected a distribution: uniform(a, b) | loguniform(a, b) | int_uniform(a, b) | "
                         "triangular(a, mode, b) | normal(m, s) | bernoulli(p) | choice('x' w, 'y' w)")
    kind, inner = m.group(1).lower(), m.group(2)
    parts = [p.strip() for p in _split_top(inner, ",")] if inner.strip() else []
    if kind == "choice":
        opts, ws = [], []
        for p in parts:
            if p.startswith("'") or p.startswith('"'):
                j = p.find(p[0], 1)
                if j < 0:
                    raise ValueError("unterminated quote in choice")
                val, rest = p[1:j], p[j + 1:].strip()
            else:
                w = p.split(None, 1)
                val, rest = w[0], (w[1] if len(w) > 1 else "")
            wt = _const(rest) if rest else 1.0
            if not _isnum(wt) or wt < 0:
                raise ValueError(f"choice weight for {val!r} must be a non-negative number")
            if val in opts:
                raise ValueError(f"choice value {val!r} listed twice")
            opts.append(val)
            ws.append(float(wt))
        if len(opts) < 2:
            raise ValueError("empty range: choice needs at least 2 values (a single value is a rule)")
        if sum(ws) <= 0:
            raise ValueError("choice weights sum to 0")
        return {"kind": "choice", "options": opts, "weights": [w / sum(ws) for w in ws]}
    want = {"uniform": 2, "loguniform": 2, "int_uniform": 2, "triangular": 3, "normal": 2, "bernoulli": 1}
    if kind not in want:
        raise ValueError(f"unknown distribution {kind!r} (uniform, loguniform, int_uniform, triangular, normal, "
                         f"bernoulli, choice)")
    if len(parts) != want[kind]:
        raise ValueError(f"{kind}() takes {want[kind]} numbers")
    a = []
    for p in parts:
        v = _const(p)
        if not _isnum(v):
            raise ValueError(f"{kind}() arguments must be numbers")
        a.append(float(v))
    if kind == "uniform" and not a[0] < a[1]:
        raise ValueError(f"empty or inverted range uniform({_g(a[0])}, {_g(a[1])}) (a stated single value is a param)")
    if kind == "loguniform" and not 0 < a[0] < a[1]:
        raise ValueError("loguniform(a, b) needs 0 < a < b")
    if kind == "int_uniform":
        if a[0] != int(a[0]) or a[1] != int(a[1]):
            raise ValueError("int_uniform(a, b) needs integers")
        if not a[0] < a[1]:
            raise ValueError(f"empty or inverted range int_uniform({_g(a[0])}, {_g(a[1])})")
    if kind == "triangular" and not (a[0] <= a[1] <= a[2] and a[0] < a[2]):
        raise ValueError("triangular(a, mode, b) needs a <= mode <= b and a < b")
    if kind == "normal" and not a[1] > 0:
        raise ValueError("normal(m, s) needs s > 0")
    if kind == "bernoulli" and not 0 < a[0] < 1:
        raise ValueError("bernoulli(p) needs 0 < p < 1 (a certain value is a rule)")
    return {"kind": kind, "args": a}


def _dist_text(d):
    if d["kind"] == "choice":
        return "choice(" + ", ".join(f"'{o}' {_g(w)}" for o, w in zip(d["options"], d["weights"])) + ")"
    return f"{d['kind']}(" + ", ".join(_g(x) for x in d["args"]) + ")"


def _dist_type(d):
    return {"choice": "cat", "bernoulli": "bool"}.get(d["kind"], "num")


def _ndtri(u):
    """Inverse standard normal CDF (Acklam's rational approximation, |rel err| < 1.2e-9)."""
    u = np.clip(np.asarray(u, float), 1e-15, 1 - 1e-15)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    out = np.empty_like(u)
    lo, hi = u < plow, u > 1 - plow
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(u[lo]))
    out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
              ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = np.sqrt(-2 * np.log(1 - u[hi]))
    out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
              ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = u[mid] - 0.5
    r = q * q
    out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    return out


_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
           103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173]


def halton(n, d, rng):
    """n points of the d-dimensional Halton sequence (indices 1..n), randomly shifted modulo 1 (Cranley-Patterson)."""
    out = np.empty((n, d))
    idx0 = np.arange(1, n + 1)
    for j in range(d):
        base = _PRIMES[j % len(_PRIMES)]
        i = idx0 * (1 + j // len(_PRIMES))
        f, r = 1.0 / base, np.zeros(n)
        while np.any(i > 0):
            r += f * (i % base)
            i = i // base
            f /= base
        out[:, j] = r
    out = (out + rng.random(d)) % 1.0
    return np.clip(out, 1e-12, 1 - 1e-12)


def _inv_dist(d, u):
    k, a = d["kind"], d.get("args")
    if k == "uniform":
        return a[0] + (a[1] - a[0]) * u
    if k == "loguniform":
        return np.exp(math.log(a[0]) + (math.log(a[1]) - math.log(a[0])) * u)
    if k == "int_uniform":
        return np.clip(np.floor(a[0] + (a[1] - a[0] + 1) * u), a[0], a[1])
    if k == "triangular":
        lo, m, hi = a
        fc = (m - lo) / (hi - lo)
        return np.where(u < fc, lo + np.sqrt(u * (hi - lo) * (m - lo)), hi - np.sqrt((1 - u) * (hi - lo) * (hi - m)))
    if k == "normal":
        return a[0] + a[1] * _ndtri(u)
    if k == "bernoulli":
        return (u < a[0]).astype(float)
    raise ValueError(k)


def _log_pdf(d, x):
    k, a = d["kind"], d["args"]
    with np.errstate(all="ignore"):
        if k == "uniform":
            return np.where((x >= a[0]) & (x <= a[1]), -math.log(a[1] - a[0]), -np.inf)
        if k == "loguniform":
            ok = (x >= a[0]) & (x <= a[1])
            return np.where(ok, -np.log(np.where(ok, x, 1.0)) - math.log(math.log(a[1] / a[0])), -np.inf)
        if k == "triangular":
            lo, m, hi = a
            p = np.where(x < m, 2 * (x - lo) / ((hi - lo) * max(m - lo, 1e-300)),
                         2 * (hi - x) / ((hi - lo) * max(hi - m, 1e-300)))
            p = np.where((x >= lo) & (x <= hi), p, 0.0)
            return np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), -np.inf)
        if k == "normal":
            return -0.5 * ((x - a[0]) / a[1]) ** 2 - math.log(a[1] * math.sqrt(2 * math.pi))
    raise ValueError(k)


def _stratified(values_codes, weights, n, rng):
    w = np.asarray(weights, float)
    w = w / w.sum()
    raw = w * n
    cnt = np.floor(raw).astype(int)
    rest = n - cnt.sum()
    if rest > 0:
        cnt[np.argsort(-(raw - cnt), kind="stable")[:rest]] += 1
    arr = np.repeat(np.asarray(values_codes, float), cnt)
    return rng.permutation(arr)


# ============================================================================ document parser
def _parse_obs(body, no, static, steps, problems, what="fact"):
    """Fact / claim body -> dict(kind soft|interval|pred, src, T, v, tol, lo, hi, truth, trust)."""
    o = {"trust": None, "T": "final", "truth": True}
    s = body.strip()
    m = re.search(r"\btrust\s+([0-9.eE+\-]+)\s*$", s)
    if m:
        try:
            r = float(m.group(1))
        except ValueError:
            r = -1
        if not 0 < r < 1:
            problems.append(_prob(no, f"{what}: trust must be between 0 and 1 (exclusive)", "e.g. trust 0.7"))
            return None
        o["trust"] = r
        s = s[:m.start()].strip()
    m = re.search(r"\s+is\s+(true|false)\s*$", " " + s, re.I)
    if m:
        o["truth"] = m.group(1).lower() == "true"
        s = (" " + s)[:m.start()].strip()
        o["_is"] = True
    left, rest = s, ""
    m = re.search(r"\bat\s+(final|any|all|end|last|\d+)\b", s)
    if m:
        T = m.group(1)
        T = "final" if T in ("end", "last") else (int(T) if T.isdigit() else T)
        o["T"] = T
        left, rest = s[:m.start()].strip(), s[m.end():].strip()
        if isinstance(T, int) and not static and T > steps:
            problems.append(_prob(no, f"{what}: step {T} is beyond the time grid (steps={steps})",
                                  f"use a step between 0 and {steps}, or final"))
            return None
    else:
        rest = ""
    if static:
        o["T"] = "final"
    whole = (left + " " + rest).strip() if rest and rest[:1] in "<>=!" else None

    def soft(txt):
        mm = re.fullmatch(r"~\s*(.+?)\s*(?:\+-|\+/-|±)\s*(.+)", txt.strip(), re.S)
        if not mm:
            return None
        return mm.group(1), mm.group(2)

    def interval(txt):
        mm = re.fullmatch(r"in\s*\[(.+)\]", txt.strip(), re.S)
        if not mm:
            return None
        ps = _split_top(mm.group(1), ",")
        return ps if len(ps) == 2 else None

    expr_src, kind = left, "pred"
    if whole is not None:
        expr_src = whole
    elif rest:
        if rest.startswith("~"):
            sv = soft(rest)
            if not sv:
                problems.append(_prob(no, f"{what}: bad soft form", "expr at T ~ value +- tol"))
                return None
            kind, (vs, ts) = "soft", sv
        elif rest.lower().startswith("in"):
            iv = interval(rest)
            if not iv:
                problems.append(_prob(no, f"{what}: bad interval form", "expr at T in [a, b]"))
                return None
            kind, (vs, ts) = "interval", iv
        else:
            problems.append(_prob(no, f"{what}: cannot read {rest!r} after 'at'",
                                  "forms: expr at T ~ v +- tol | expr at T in [a, b] | pred at T is true|false | "
                                  "expr at T <= v"))
            return None
    else:
        top = _split_top(left, "~")
        mm = re.search(r"\s+in\s*\[", left)
        if len(top) == 2:
            sv = soft("~" + top[1])
            if not sv:
                problems.append(_prob(no, f"{what}: bad soft form", "expr ~ value +- tol"))
                return None
            expr_src, kind, (vs, ts) = top[0].strip(), "soft", sv
        elif mm:
            iv = interval(left[mm.start():].strip())
            if not iv:
                problems.append(_prob(no, f"{what}: bad interval form", "expr in [a, b]"))
                return None
            expr_src, kind, (vs, ts) = left[:mm.start()].strip(), "interval", iv
    try:
        if kind == "soft":
            o["v"], o["tol"] = float(_const(vs)), float(_const(ts))
            if not o["tol"] > 0:
                raise ValueError("tolerance must be > 0")
        elif kind == "interval":
            o["lo"], o["hi"] = float(_const(vs[0] if isinstance(vs, list) else vs)), float(_const(ts))
            if not o["lo"] <= o["hi"]:
                raise ValueError("interval [a, b] needs a <= b")
    except (ValueError, TypeError) as e:
        problems.append(_prob(no, f"{what}: bad number: {e}", "numbers or constant expressions"))
        return None
    if kind != "pred" and o.pop("_is", False):
        problems.append(_prob(no, f"{what}: 'is true/false' only applies to a predicate", "drop it"))
        return None
    o.pop("_is", None)
    if not expr_src:
        problems.append(_prob(no, f"{what}: missing expression", "e.g. queue at 30 ~ 8 +- 3"))
        return None
    try:
        o["ast"] = parse_expr(expr_src)
    except _expr.ExprError as e:
        problems.append(_prob(no, f"{what} expression: {e.msg}", e.hint or "check the expression"))
        return None
    if _has_stoch(o["ast"]):
        problems.append(_prob(no, f"{what}: random functions are not allowed in an observation",
                              "compute the random part in a node and refer to the node"))
        return None
    o["kind"], o["src"] = kind, expr_src
    return o


def _obs_text(o):
    T = o["T"]
    at = "" if o.get("static") else f" at {T}"
    if o["kind"] == "soft":
        s = f"{o['src']}{at} ~ {_g(o['v'])} +- {_g(o['tol'])}"
    elif o["kind"] == "interval":
        s = f"{o['src']}{at} in [{_g(o['lo'])}, {_g(o['hi'])}]"
    else:
        s = f"{o['src']}{at}" + ("" if o["truth"] else " is false")
    return s + (f" trust {_g(o['trust'])}" if o.get("trust") else "")


def _parse_value(txt):
    """A param/rule/set value: number, true/false, 'mode' or a bare word (category)."""
    if isinstance(txt, bool):
        return txt
    if _isnum(txt):
        return float(txt)
    s = str(txt).strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    v = _const(s, bare_word=True)
    return float(v) if _isnum(v) else v


def _parse_change(c, no=None):
    """One what-if change -> (kind, name, payload). kinds: set, do, doexpr, fix, any."""
    s = c.strip()
    m = re.fullmatch(r"(param|rule|set|fix|do)?\s*([A-Za-z_]\w*)\s*(:=|=)\s*(.+)", s, re.S)
    if not m:
        raise ValueError(f"bad change {s!r} (forms: rule x = v | param x = v | do x = v [from k] | do x := expr | "
                         f"fix u = v)")
    kw, name, op, rest = (m.group(1) or "any"), m.group(2), m.group(3), m.group(4).strip()
    if op == ":=":
        return ("doexpr", name, (parse_expr(rest), rest))
    frm = 0
    m2 = re.fullmatch(r"(.+?)\s+from\s+(\d+)", rest, re.S)
    if m2:
        rest, frm = m2.group(1), int(m2.group(2))
    v = _parse_value(rest)
    kind = {"param": "set", "rule": "set", "set": "set", "fix": "fix", "do": "do", "any": "any"}[kw]
    return (kind, name, (v, frm))


def analyse(lines):
    """Parse and statically check a document. Returns the doc dict with `problems` (all at once)."""
    d = {"time": None, "samples": DEFAULT_SAMPLES, "targets": [], "params": {}, "rules": {}, "unknowns": {},
         "hyps": {}, "nodes": {}, "facts": [], "claims": [], "whatifs": [], "problems": [],
         "fully_determined": False, "decl": {}, "unused": set(), "noeffect": set()}
    P, decl = d["problems"], d["decl"]

    def declare(name, no, kind):
        if not NAME_RE.fullmatch(name):
            P.append(_prob(no, f"invalid name {name!r}", "letters, digits and _"))
            return False
        if name in _expr.RESERVED or name in _expr.FUNCS or name in STOCH or name in LINE_KEYWORDS:
            P.append(_prob(no, f"{name!r} is a reserved word", "rename it"))
            return False
        if name in decl:
            P.append(_prob(no, f"duplicate name {name!r} (first declared on line {decl[name][0]})",
                           "rename or remove one"))
            return False
        decl[name] = (no, kind)
        return True

    names_fc = {}
    for no, raw in enumerate(lines, 1):
        if not isinstance(raw, str):
            raw = str(raw)
        code, com = _split_comment(raw)
        s = code.strip()
        lc = com.lower()
        if "fully determined:" in lc:
            d["fully_determined"] = True
        if not s or s.startswith("```") or s.startswith("~~~"):
            continue
        km = NAME_RE.match(s)
        kw = km.group(0).lower() if km else ""
        try:
            if kw == "time":
                rest = s[4:].strip().lstrip(":").strip()
                if re.fullmatch(r"static", rest, re.I):
                    d["time"] = {"steps": 0, "dt": 1.0, "unit": "", "static": True, "line": no}
                    continue
                kv = {k.lower(): v for k, v in re.findall(r"(\w+)\s*[=:]\s*([^\s,;]+)", rest)}
                if "steps" not in kv:
                    P.append(_prob(no, "time line needs steps (or 'static')", "time steps=60 dt=1 unit=minute"))
                    continue
                st = float(kv["steps"])
                if st != int(st) or st < 0:
                    P.append(_prob(no, "steps must be a non-negative integer", "time steps=60 dt=1"))
                    continue
                if st > MAX_STEPS:
                    P.append(_prob(no, f"too many steps ({int(st)}; max {MAX_STEPS})", "use a coarser dt"))
                    continue
                dt = float(kv.get("dt", 1))
                d["time"] = {"steps": int(st), "dt": dt, "unit": kv.get("unit", ""), "static": int(st) == 0,
                             "line": no}
                continue
            if kw == "samples":
                mm = re.fullmatch(r"samples\s*[=:]?\s*(\d+)", s, re.I)
                if not mm:
                    P.append(_prob(no, "bad samples line", "samples 2000"))
                    continue
                d["samples"] = max(MIN_SAMPLES, min(MAX_SAMPLES, int(mm.group(1))))
                continue
            if kw in ("targets", "target") and re.match(r"targets?\s*:", s, re.I):
                for x in s.split(":", 1)[1].split(","):
                    x = x.strip()
                    if not x:
                        continue
                    if not NAME_RE.fullmatch(x):
                        P.append(_prob(no, f"bad target name {x!r}", "targets: a, b"))
                    elif x not in [t for t, _ in d["targets"]]:
                        d["targets"].append((x, no))
                continue
            if kw == "param":
                mm = re.fullmatch(r"param\s+([A-Za-z_]\w*)\s*=\s*(.+)", s, re.S)
                if not mm:
                    P.append(_prob(no, "bad param line", "param name = 1.5   # stated '...'"))
                    continue
                try:
                    v = _const(mm.group(2))
                except ValueError as e:
                    P.append(_prob(no, f"param {mm.group(1)}: {e}", "a number"))
                    continue
                if not _isnum(v):
                    P.append(_prob(no, f"param {mm.group(1)} must be a number", "a stated switch or mode is a rule"))
                    continue
                if declare(mm.group(1), no, "param"):
                    d["params"][mm.group(1)] = {"value": float(v), "line": no, "prov": com}
                    if "unused:" in lc:
                        d["unused"].add(mm.group(1))
                continue
            if kw == "rule":
                mm = re.fullmatch(r"rule\s+([A-Za-z_]\w*)\s*=\s*(.+)", s, re.S)
                if not mm:
                    P.append(_prob(no, "bad rule line", "rule name = 2 | true | false | 'mode'"))
                    continue
                try:
                    v = _parse_value(mm.group(2))
                except ValueError as e:
                    P.append(_prob(no, f"rule {mm.group(1)}: {e}", "a number, true/false or 'mode'"))
                    continue
                if declare(mm.group(1), no, "rule"):
                    d["rules"][mm.group(1)] = {"value": v, "line": no, "prov": com}
                    if "unused:" in lc:
                        d["unused"].add(mm.group(1))
                continue
            if kw == "unknown":
                mm = re.fullmatch(r"unknown\s+([A-Za-z_]\w*)\s*(?::\s*\w+\s*)?~\s*(.+)", s, re.S)
                if not mm:
                    P.append(_prob(no, "bad unknown line", "unknown name ~ uniform(a, b)   # why it is open"))
                    continue
                try:
                    dist = _parse_dist(mm.group(2))
                except ValueError as e:
                    P.append(_prob(no, f"unknown {mm.group(1)}: {e}", "e.g. uniform(0.1, 0.4)"))
                    continue
                if declare(mm.group(1), no, "unknown"):
                    d["unknowns"][mm.group(1)] = {"dist": dist, "line": no, "prov": com}
                    if "unused:" in lc:
                        d["unused"].add(mm.group(1))
                continue
            if kw == "hypothesis":
                mm = re.fullmatch(r"hypothesis\s+([A-Za-z_]\w*)\s*:\s*(.*)", s, re.S)
                if not mm:
                    P.append(_prob(no, "bad hypothesis line", "hypothesis cause: 'a' 0.5 | 'b' 0.5"))
                    continue
                vals, ws, bad = [], [], False
                for alt in _split_top(mm.group(2), "|"):
                    alt = alt.strip().rstrip(",")
                    if not alt:
                        continue
                    if alt[0] in "'\"":
                        j = alt.find(alt[0], 1)
                        if j < 0:
                            P.append(_prob(no, "unterminated quote in hypothesis", "'value' weight"))
                            bad = True
                            break
                        val, rest = alt[1:j], alt[j + 1:].strip()
                    else:
                        w = alt.split(None, 1)
                        val, rest = w[0], (w[1] if len(w) > 1 else "")
                    try:
                        wt = float(_const(rest)) if rest else None
                    except (ValueError, TypeError):
                        P.append(_prob(no, f"hypothesis weight for {val!r} must be a number", "'a' 0.5 | 'b' 0.5"))
                        bad = True
                        break
                    if val in vals:
                        P.append(_prob(no, f"hypothesis value {val!r} listed twice", "remove one"))
                        bad = True
                        break
                    vals.append(val)
                    ws.append(wt)
                if bad:
                    continue
                if len(vals) < 2:
                    P.append(_prob(no, f"hypothesis {mm.group(1)} needs at least 2 values (got {len(vals)})",
                                   "add the alternatives, including one you doubt"))
                    continue
                if all(w is None for w in ws):
                    ws = [1.0] * len(vals)
                elif any(w is None for w in ws) or any(w < 0 for w in ws) or sum(ws) <= 0:
                    P.append(_prob(no, "hypothesis weights must all be given, >= 0, with a positive sum",
                                   "'a' 0.5 | 'b' 0.5 (or no weights at all: equal)"))
                    continue
                if declare(mm.group(1), no, "hypothesis"):
                    d["hyps"][mm.group(1)] = {"values": vals, "weights": [w / sum(ws) for w in ws], "line": no,
                                              "prov": com}
                    if "unused:" in lc:
                        d["unused"].add(mm.group(1))
                    if "no effect expected:" in lc:
                        d["noeffect"].add(mm.group(1))
                continue
            if kw in ("fact", "claim"):
                mm = re.fullmatch(kw + r"\s+([A-Za-z_]\w*)\s*:\s*(.+)", s, re.S | re.I)
                if not mm:
                    P.append(_prob(no, f"bad {kw} line", f"{kw} name: queue at 30 ~ 8 +- 3" if kw == "fact"
                                   else "claim name: queue at final <= 5"))
                    continue
                nm = mm.group(1)
                if nm in names_fc:
                    P.append(_prob(no, f"duplicate {kw} name {nm!r} (line {names_fc[nm]})", "rename one"))
                    continue
                names_fc[nm] = no
                tm = d["time"] or {"static": False, "steps": MAX_STEPS}
                o = _parse_obs(mm.group(2), no, tm["static"], tm["steps"], P, kw)
                if o is None:
                    continue
                if kw == "claim":
                    if o["kind"] == "soft":
                        P.append(_prob(no, "a claim is a predicate, not a soft observation",
                                       "claim name: queue at final <= 5"))
                        continue
                    if o.get("trust"):
                        P.append(_prob(no, "a claim has no trust (it is not evidence)", "drop trust"))
                        continue
                o.update(name=nm, line=no, prov=com, static=tm["static"])
                d["facts" if kw == "fact" else "claims"].append(o)
                continue
            if kw == "whatif":
                mm = re.fullmatch(r"whatif\s+([A-Za-z_]\w*)\s*:\s*(.*)", s, re.S)
                if not mm:
                    P.append(_prob(no, "bad whatif line", "whatif name: rule clerks = 3 ; targets queue"))
                    continue
                wi = {"name": mm.group(1), "changes": [], "targets": [], "at": None, "line": no,
                      "noeffect": "no effect expected:" in lc, "src": mm.group(2).strip()}
                if any(w["name"] == wi["name"] for w in d["whatifs"]):
                    P.append(_prob(no, f"duplicate whatif name {wi['name']!r}", "rename one"))
                    continue
                ok = True
                for seg in _split_top(mm.group(2), ";"):
                    seg = seg.strip()
                    if not seg:
                        continue
                    w0 = seg.split(None, 1)
                    key = w0[0].lower().rstrip(":")
                    rest = w0[1] if len(w0) > 1 else ""
                    if key in ("targets", "target"):
                        wi["targets"] += [x.strip() for x in rest.replace(":", " ").split(",") if x.strip()]
                    elif key == "at":
                        r = rest.strip()
                        wi["at"] = int(r) if r.isdigit() else "final"
                    else:
                        for c in _split_top(seg, ","):
                            if not c.strip():
                                continue
                            try:
                                wi["changes"].append(_parse_change(c, no))
                            except (ValueError, _expr.ExprError) as e:
                                P.append(_prob(no, f"whatif {wi['name']}: {e}",
                                               "rule x = v | param x = v | do x = v [from k] | do x := expr | "
                                               "fix u = v"))
                                ok = False
                if ok and not wi["changes"]:
                    P.append(_prob(no, f"whatif {wi['name']} changes nothing", "whatif name: rule clerks = 3"))
                    ok = False
                if ok:
                    d["whatifs"].append(wi)
                continue
            # node line
            mm = re.fullmatch(r"([A-Za-z_]\w*)\s*(:=|:)(.*)", s, re.S)
            if not mm:
                P.append(_prob(no, "unrecognised line",
                               "lines are: time | samples | targets: | param | rule | unknown | hypothesis | "
                               "name : type [= v0] := expr | fact | claim | whatif"))
                continue
            name, op, rest = mm.group(1), mm.group(2), mm.group(3)
            if op == ":=":
                left, eq = "", rest
            else:
                parts = rest.split(":=", 1)
                left, eq = parts[0], (parts[1] if len(parts) > 1 else None)
            v0src = None
            if "=" in left:
                left, v0src = left.split("=", 1)
            words = left.split()
            typ = "real"
            if words and words[0].lower() in TYPES:
                typ = TYPES[words.pop(0).lower()]
            nd = {"type": typ, "unit": " ".join(words), "v0": None, "line": no, "prov": com, "src": None,
                  "ast": None}
            if v0src is not None:
                try:
                    v0 = _const(v0src, bare_word=typ == "category")
                except ValueError as e:
                    P.append(_prob(no, f"bad initial value for {name!r}: {e}", "a number, true/false or 'mode'"))
                    continue
                nd["v0"] = v0
            if eq is None:
                if v0src is None:
                    P.append(_prob(no, f"node {name!r} has no equation",
                                   f"{name} : real := <expression>; a stated number is a param, an open one an "
                                   f"unknown"))
                    continue
                nd["src"], nd["ast"] = None, ("str", nd["v0"]) if isinstance(nd["v0"], str) else \
                    ("bool", nd["v0"]) if isinstance(nd["v0"], bool) else ("num", float(nd["v0"]))
                nd["const"] = True
            else:
                eq = eq.strip()
                if not eq:
                    P.append(_prob(no, f"empty equation for {name!r}", f"{name} : real := <expression>"))
                    continue
                try:
                    nd["ast"] = parse_expr(eq)
                    nd["src"] = eq
                except _expr.ExprError as e:
                    P.append(_prob(no, f"equation for {name!r}: {e.msg}" + (f" at column {e.col}" if e.col else ""),
                                   e.hint or "check the expression syntax"))
                    declare(name, no, "node")
                    continue
            if declare(name, no, "node"):
                d["nodes"][name] = nd
        except Exception as e:  # never raise on a line
            P.append(_prob(no, f"could not parse line: {type(e).__name__}: {e}", "see the document syntax"))
    _check(d)
    return d


# ---------------------------------------------------------------------------- static checks
def _kind(d, name):
    return d["decl"].get(name, (None, None))[1]


def _var_type(d, name):
    k = _kind(d, name)
    if k == "param":
        return "num"
    if k == "rule":
        v = d["rules"][name]["value"]
        return "bool" if isinstance(v, bool) else ("cat" if isinstance(v, str) else "num")
    if k == "unknown":
        return _dist_type(d["unknowns"][name]["dist"])
    if k == "hypothesis":
        return "cat"
    if k == "node" and name in d["nodes"]:  # a node whose equation failed to parse is declared but has no entry
        return {"real": "num", "int": "num", "bool": "bool", "category": "cat"}[d["nodes"][name]["type"]]
    return None


def _value_set(d, name):
    k = _kind(d, name)
    if k == "hypothesis":
        return d["hyps"][name]["values"]
    if k == "unknown" and d["unknowns"][name]["dist"]["kind"] == "choice":
        return d["unknowns"][name]["dist"]["options"]
    return None


def _infer(ast, d, errs):
    """Static type of an expression: 'num' | 'bool' | 'cat' | None (unknown); appends messages to errs."""
    k = ast[0]
    if k == "num":
        return "num"
    if k == "bool":
        return "bool"
    if k == "str":
        return "cat"
    if k == "var":
        if ast[1] in ("t", "dt"):
            return "num"
        return _var_type(d, ast[1])
    if k in ("neg",):
        t = _infer(ast[1], d, errs)
        if t == "cat":
            errs.append("a category cannot be negated")
        return "num"
    if k in ("not", "and", "or"):
        for a in ast[1:]:
            if _infer(a, d, errs) == "cat":
                errs.append(f"a category cannot be used with '{k}' (compare it: x == 'value')")
        return "bool"
    if k == "if":
        if _infer(ast[1], d, errs) == "cat":
            errs.append("an if condition cannot be a category (compare it: x == 'value')")
        ta, tb = _infer(ast[2], d, errs), _infer(ast[3], d, errs)
        if (ta == "cat") != (tb == "cat") and ta and tb:
            errs.append("if-branches mix a category and a number")
        return "cat" if "cat" in (ta, tb) else ("bool" if ta == tb == "bool" else "num")
    if k == "bin":
        op, a, b = ast[1], ast[2], ast[3]
        ta, tb = _infer(a, d, errs), _infer(b, d, errs)
        if op in ("==", "!="):
            if (ta == "cat") != (tb == "cat") and ta and tb:
                errs.append(f"'{op}' compares a category with a number")
            for x, y in ((a, b), (b, a)):
                if x[0] == "var" and y[0] == "str":
                    vs = _value_set(d, x[1])
                    if vs is not None and y[1] not in vs:
                        errs.append(f"'{y[1]}' is not a value of {x[1]}{_dym(y[1], vs)}; values: "
                                    + ", ".join(repr(v) for v in vs))
            return "bool"
        if "cat" in (ta, tb):
            errs.append(f"'{op}' cannot be applied to a category")
        return "bool" if op in CMP_OPS else "num"
    if k == "call":
        fn = ast[1]
        if fn == "choice":
            return "cat"
        if fn == "choice_w":
            for w in ast[2][1::2]:
                if _infer(w, d, errs) == "cat":
                    errs.append("choice_w weights must be numbers")
            return "cat"
        for a in ast[2]:
            if _infer(a, d, errs) == "cat":
                errs.append(f"{fn}() cannot take a category")
        return "bool" if fn == "bernoulli" else "num"
    return None


def _walk_strs(ast, out):
    if isinstance(ast, tuple):
        if ast[0] == "str":
            out.append(ast[1])
        for a in ast[1:]:
            if isinstance(a, (tuple, list)):
                _walk_strs(a, out)
    elif isinstance(ast, list):
        for a in ast:
            _walk_strs(a, out)
    return out


def _topo(nodes, deps):
    """Topological order of nodes over same-step dependencies; returns (order, cycle or None)."""
    order, state = [], {}
    for start in nodes:
        if state.get(start):
            continue
        stack = [(start, iter(sorted(deps.get(start, ()))))]
        state[start] = 1
        path = [start]
        while stack:
            n, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                path.pop()
                state[n] = 2
                order.append(n)
                continue
            if nxt not in nodes:
                continue
            s = state.get(nxt)
            if s == 1:
                return None, path[path.index(nxt):] + [nxt]
            if not s:
                state[nxt] = 1
                path.append(nxt)
                stack.append((nxt, iter(sorted(deps.get(nxt, ())))))
    return order, None


def _check(d):
    P = d["problems"]
    tm = d["time"]
    if tm is None:
        P.append(_prob(None, "missing time line", "time steps=60 dt=1 unit=day   (or: time static)"))
        tm = d["time"] = {"steps": 1, "dt": 1.0, "unit": "", "static": False, "line": None, "missing": True}
    static = tm["static"]
    declared = set(d["decl"])
    reads = set()

    def check_refs(ast, no, where, allow_lag=True, first_step=0):
        """first_step: the earliest step at which this expression is evaluated (a lag k reaches before step 0 only
        if first_step < k; A2's step-0 rule: a node with an initial value is computed from step 1)."""
        bad = False
        for name, lag in sorted(_expr.refs(ast)):
            if name not in declared:
                P.append(_prob(no, f"{where}: undefined name {name!r}{_dym(name, declared)}",
                               "declare it as param, rule, unknown, hypothesis or a node"))
                bad = True
                continue
            reads.add(name)
            if lag:
                if static:
                    P.append(_prob(no, f"{where}: lag {name}[t-{lag}] in a static document",
                                   "static documents have no time; remove the lag or use time steps=N"))
                    bad = True
                elif not allow_lag or lag <= first_step:
                    pass
                elif name in d["nodes"] and d["nodes"][name]["v0"] is None:
                    P.append(_prob(no, f"{where}: {name}[t-{lag}] at step 0 needs an initial value for {name!r}",
                                   f"declare it as {name} : {d['nodes'][name]['type']} = <v0> := ..."))
                    bad = True
        errs = []
        _infer(ast, d, errs)
        for e in dict.fromkeys(errs):
            P.append(_prob(no, f"{where}: type error: {e}", "categories compare only with == / != to 'values'"))
            bad = True
        return not bad

    deps = {}
    for n, nd in d["nodes"].items():
        if nd["ast"] is None:
            continue
        ok = check_refs(nd["ast"], nd["line"], f"node {n}", first_step=1 if nd["v0"] is not None else 0)
        deps[n] = {x for x, lag in _expr.refs(nd["ast"]) if lag == 0 and x in d["nodes"]}
        if ok:
            errs = []
            t = _infer(nd["ast"], d, errs)
            if nd["type"] == "category" and t in ("num", "bool"):
                P.append(_prob(nd["line"], f"node {n} is a category but its equation gives a number",
                               "use : real / : int, or give 'values'"))
            elif nd["type"] != "category" and t == "cat":
                P.append(_prob(nd["line"], f"node {n} is {nd['type']} but its equation gives a category",
                               f"declare it {n} : category := ..."))
            if nd["v0"] is not None and isinstance(nd["v0"], str) != (nd["type"] == "category"):
                P.append(_prob(nd["line"], f"initial value of {n} does not fit type {nd['type']}", "fix v0"))
    order, cyc = _topo(list(d["nodes"]), deps)
    if cyc:
        P.append(_prob(d["nodes"][cyc[0]]["line"], "same-step cycle: " + " -> ".join(cyc),
                       "break it with a lag, e.g. x[t-1]"))
        order = list(d["nodes"])
    d["order"] = order
    d["deps"] = deps
    for o in d["facts"] + d["claims"]:
        T = o["T"]
        fs = tm["steps"] if T == "final" else (T if isinstance(T, int) else 0)
        check_refs(o["ast"], o["line"], f"{'fact' if o in d['facts'] else 'claim'} {o['name']}", first_step=fs)
    if not d["targets"]:
        P.append(_prob(None, "missing targets: line", "targets: a, b   (the quantities the question asks about)"))
    for t, no in d["targets"]:
        if t not in declared:
            P.append(_prob(no, f"target {t!r} is not declared{_dym(t, declared)}", "declare it or fix the name"))
        else:
            reads.add(t)
    for wi in d["whatifs"]:
        _resolve_changes(d, wi["changes"], wi["line"], f"whatif {wi['name']}", P)
        for t in wi["targets"]:
            if t not in declared:
                P.append(_prob(wi["line"], f"whatif {wi['name']}: target {t!r} is not declared{_dym(t, declared)}",
                               "fix the name"))
        for ch in wi["changes"]:
            if ch[0] == "doexpr":
                for x, _l in _expr.refs(ch[2][0]):
                    reads.add(x)
    # declared but never read (a what-if on it would be a silent no-op)
    for kind, table in (("param", d["params"]), ("rule", d["rules"]), ("unknown", d["unknowns"]),
                        ("hypothesis", d["hyps"])):
        for n, rec in table.items():
            if n not in reads and n not in d["unused"]:
                P.append(_prob(rec["line"], f"{kind} {n} is declared but never read by any equation, fact or claim",
                               f"use it in the mechanism, remove it, or mark the line '# unused: <reason>'",
                               fatal=False))
    if not d["unknowns"] and not d["hyps"] and not d["fully_determined"]:
        P.append(_prob(None, "nothing is uncertain: the document has no unknown and no hypothesis",
                       "give each open or vague quantity a range (unknown x ~ uniform(a, b)) and each competing "
                       "explanation a hypothesis; or add '# fully determined: <reason>'", fatal=False))


def _resolve_changes(d, changes, no, where, P):
    """Validate and normalise changes in place: kind any/set/do/fix resolved by the declared kind of the name."""
    declared = d["decl"]
    out = []
    for ch in changes:
        kind, name, payload = ch
        k = _kind(d, name)
        if k is None:
            ov = list(d["params"]) + list(d["rules"]) + list(d["unknowns"]) + list(d["hyps"]) + list(d["nodes"])
            P.append(_prob(no, f"{where}: {name!r} is not declared{_dym(name, ov)}; declared params: "
                           f"{list(d['params'])}, rules: {list(d['rules'])}, unknowns: {list(d['unknowns'])}, "
                           f"hypotheses: {list(d['hyps'])}", "change a declared name"))
            continue
        if kind == "doexpr":
            if k != "node":
                P.append(_prob(no, f"{where}: do {name} := ... needs a node ({name} is a {k})", "use = for a value"))
                continue
            ast = payload[0]
            bad = False
            for x, _l in _expr.refs(ast):
                if x not in declared:
                    P.append(_prob(no, f"{where}: undefined name {x!r}{_dym(x, declared)}", "fix the expression"))
                    bad = True
            if not bad:
                out.append(("doexpr", name, payload))
            continue
        v, frm = payload
        if k in ("param", "rule"):
            if k == "param" and not _isnum(v):
                P.append(_prob(no, f"{where}: param {name} needs a number", "param x = 2.5"))
                continue
            out.append(("set", name, v))
        elif k in ("unknown", "hypothesis"):
            vs = _value_set(d, name)
            if vs is not None and v not in vs:
                P.append(_prob(no, f"{where}: {v!r} is not a value of {name}{_dym(v, vs)}", "one of " + ", ".join(
                    repr(x) for x in vs)))
                continue
            if vs is None and not (_isnum(v) or isinstance(v, bool)):
                P.append(_prob(no, f"{where}: fix {name} needs a number", "fix u = 0.3"))
                continue
            out.append(("fix", name, v))
        else:
            nd = d["nodes"].get(name)
            if nd is None:  # its equation failed to parse; that problem is already reported
                continue
            if isinstance(v, str) != (nd["type"] == "category"):
                P.append(_prob(no, f"{where}: do {name} = {v!r} does not fit type {nd['type']}", "fix the value"))
                continue
            out.append(("do", name, (v, frm)))
    changes[:] = out
    return out


# ============================================================================ vectorised evaluator
class Codes:
    """Category values as float codes (one table per compiled document)."""

    def __init__(self):
        self.tab, self.names = {}, []

    def code(self, s):
        s = str(s)
        if s not in self.tab:
            self.tab[s] = float(len(self.names))
            self.names.append(s)
        return self.tab[s]

    def name(self, c):
        try:
            i = int(c)
        except (TypeError, ValueError):
            return "?"
        return self.names[i] if 0 <= i < len(self.names) else "?"


def _tr(x):
    return np.not_equal(x, 0)


def _fl(x):
    return np.asarray(x, dtype=float)


_BIN = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "%": np.mod, "^": np.power}
_CMP = {"==": np.equal, "!=": np.not_equal, "<": np.less, "<=": np.less_equal, ">": np.greater,
        ">=": np.greater_equal}


class _CC:
    def __init__(self, codes, owner):
        self.codes, self.owner, self.k = codes, owner, 0
        self.oid = zlib.crc32(str(owner).encode("utf-8"))


def compile_expr(ast, codes, owner="expr"):
    """AST -> closure f(ctx) giving a float array over members (or a scalar)."""
    return _compile(ast, _CC(codes, owner))


def _compile(ast, cc):
    k = ast[0]
    if k == "num":
        v = float(ast[1])
        return lambda c: v
    if k == "bool":
        v = 1.0 if ast[1] else 0.0
        return lambda c: v
    if k == "str":
        v = cc.codes.code(ast[1])
        return lambda c: v
    if k == "var":
        name, lag = ast[1], ast[2]
        if name == "t":
            return lambda c: float(c.t)
        if name == "dt":
            return lambda c: c.dt
        if lag == 0:
            return lambda c: c.cur[name]
        return lambda c: c.lag(name, lag)
    if k == "neg":
        f = _compile(ast[1], cc)
        return lambda c: np.negative(f(c))
    if k == "not":
        f = _compile(ast[1], cc)
        return lambda c: _fl(np.logical_not(_tr(f(c))))
    if k in ("and", "or"):
        fa, fb = _compile(ast[1], cc), _compile(ast[2], cc)
        op = np.logical_and if k == "and" else np.logical_or
        return lambda c: _fl(op(_tr(fa(c)), _tr(fb(c))))
    if k == "if":
        fc, fa, fb = _compile(ast[1], cc), _compile(ast[2], cc), _compile(ast[3], cc)
        return lambda c: np.where(_tr(fc(c)), fa(c), fb(c))
    if k == "bin":
        op = ast[1]
        fa, fb = _compile(ast[2], cc), _compile(ast[3], cc)
        if op in _CMP:
            f = _CMP[op]
            return lambda c: _fl(f(fa(c), fb(c)))
        f = _BIN[op]
        return lambda c: f(_fl(fa(c)), _fl(fb(c)))
    if k == "call":
        fn, args = ast[1], ast[2]
        if fn in STOCH:
            site = cc.k
            cc.k += 1
            return _compile_stoch(fn, args, cc, site)
        fs = [_compile(a, cc) for a in args]
        if fn in ("min", "max"):
            g = np.minimum if fn == "min" else np.maximum
            return lambda c: reduce(g, [_fl(f(c)) for f in fs])
        f0 = fs[0]
        if fn == "abs":
            return lambda c: np.abs(f0(c))
        if fn == "sqrt":
            return lambda c: np.sqrt(_fl(f0(c)))
        if fn == "exp":
            return lambda c: np.exp(_fl(f0(c)))
        if fn == "log":
            if len(fs) == 2:
                f1 = fs[1]
                return lambda c: np.log(_fl(f0(c))) / np.log(_fl(f1(c)))
            return lambda c: np.log(_fl(f0(c)))
        if fn == "floor":
            return lambda c: np.floor(f0(c))
        if fn == "ceil":
            return lambda c: np.ceil(f0(c))
        if fn == "round":
            if len(fs) == 2:
                f1 = fs[1]

                def rnd(c):
                    nd = f1(c)
                    nd = int(np.asarray(nd).flat[0]) if np.ndim(nd) == 0 or np.all(nd == np.asarray(nd).flat[0]) \
                        else 0
                    return np.round(_fl(f0(c)), nd)
                return rnd
            return lambda c: np.rint(f0(c))
        if fn == "clip":
            f1, f2 = fs[1], fs[2]
            return lambda c: np.minimum(np.maximum(f0(c), f1(c)), f2(c))
        if fn == "sign":
            return lambda c: np.sign(f0(c))
    raise ValueError(f"cannot compile {k}")


def _binom_inv(u, n, p):
    """Binomial(n, p) by inversion of one uniform per member (monotone coupling: common random numbers)."""
    N = u.shape[0]
    n = np.broadcast_to(_fl(n), (N,)).copy()
    p = np.broadcast_to(_fl(p), (N,)).copy()
    flip = p > 0.5
    q = np.where(flip, 1 - p, p)
    uu = np.where(flip, 1 - u, u)
    y = np.zeros(N)
    exact = (n <= 200) & (q > 0)
    if exact.any():
        nn, qq, us = n[exact], q[exact], uu[exact]
        pmf = (1 - qq) ** nn
        cdf = pmf.copy()
        k = np.zeros_like(us)
        done = us < cdf
        ratio = qq / (1 - qq)
        for kk in range(1, int(nn.max()) + 1):
            act = ~done
            if not act.any():
                break
            pmf = pmf * np.maximum(nn - kk + 1, 0) / kk * ratio
            cdf = cdf + pmf
            k[act] = kk
            done |= act & (us < cdf)
        y[exact] = np.minimum(k, nn)
    big = (n > 200) & (q > 0)
    if big.any():
        nn, qq, us = n[big], q[big], uu[big]
        lam = nn * qq
        small = lam <= 50
        yb = np.zeros_like(us)
        if small.any():
            yb[small] = np.minimum(_pois_inv(us[small], lam[small]), nn[small])
        lg = ~small
        if lg.any():
            m, sd = lam[lg], np.sqrt(lam[lg] * (1 - qq[lg]))
            yb[lg] = np.clip(np.rint(m + sd * _ndtri(us[lg])), 0, nn[lg])
        y[big] = yb
    return np.where(flip, n - y, y)


def _pois_inv(u, lam):
    N = u.shape[0]
    lam = np.broadcast_to(_fl(lam), (N,)).copy()
    y = np.zeros(N)
    ex = (lam <= 200) & (lam > 0)
    if ex.any():
        ll, us = lam[ex], u[ex]
        pmf = np.exp(-ll)
        cdf = pmf.copy()
        k = np.zeros_like(us)
        done = us < cdf
        kmax = int(ll.max() + 12 * math.sqrt(ll.max()) + 25)
        for kk in range(1, kmax + 1):
            act = ~done
            if not act.any():
                break
            pmf = pmf * ll / kk
            cdf = cdf + pmf
            k[act] = kk
            done |= act & (us < cdf)
        y[ex] = k
    big = lam > 200
    if big.any():
        y[big] = np.maximum(0, np.rint(lam[big] + np.sqrt(lam[big]) * _ndtri(u[big])))
    return y


def _compile_stoch(fn, args, cc, site):
    oid = cc.oid
    if fn == "choice":
        codes = np.array([cc.codes.code(a[1]) for a in args])

        def ch(c):
            u = c.gen(oid, site).random(c.N)
            return codes[np.minimum((u * len(codes)).astype(int), len(codes) - 1)]
        return ch
    if fn == "choice_w":
        codes = np.array([cc.codes.code(a[1]) for a in args[0::2]])
        wf = [_compile(a, cc) for a in args[1::2]]

        def chw(c):
            u = c.gen(oid, site).random(c.N)
            W = np.stack([np.broadcast_to(np.maximum(_fl(f(c)), 0), (c.N,)) for f in wf], axis=1)
            tot = W.sum(axis=1)
            cum = np.cumsum(W, axis=1) / np.where(tot > 0, tot, 1)[:, None]
            idx = (u[:, None] >= cum).sum(axis=1)
            out = codes[np.minimum(idx, len(codes) - 1)].astype(float)
            out[tot <= 0] = np.nan
            return out
        return chw
    fs = [_compile(a, cc) for a in args]
    owner = cc.owner

    def clip01(c, p):
        p = _fl(p)
        bad = (p < 0) | (p > 1)
        if np.any(bad):
            c.clips[owner] = c.clips.get(owner, 0) + int(np.sum(np.broadcast_to(bad, (c.N,))))
        return np.clip(p, 0, 1)

    if fn == "bernoulli":
        return lambda c: _fl(c.gen(oid, site).random(c.N) < clip01(c, fs[0](c)))
    if fn == "binomial":
        def binom(c):
            u = c.gen(oid, site).random(c.N)
            n = _fl(fs[0](c))
            n = np.where(np.isfinite(n), np.maximum(np.rint(n), 0), np.nan)
            res = _binom_inv(u, np.nan_to_num(n), clip01(c, np.nan_to_num(fs[1](c))))
            return np.where(np.isnan(np.broadcast_to(n, (c.N,))), np.nan, res)
        return binom
    if fn == "poisson":
        def pois(c):
            u = c.gen(oid, site).random(c.N)
            lam = _fl(fs[0](c))
            res = _pois_inv(u, np.nan_to_num(np.maximum(lam, 0)))
            return np.where(np.isnan(np.broadcast_to(lam, (c.N,))), np.nan, res)
        return pois
    if fn == "normal":
        return lambda c: _fl(fs[0](c)) + _fl(fs[1](c)) * c.gen(oid, site).standard_normal(c.N)
    if fn == "uniform":
        return lambda c: _fl(fs[0](c)) + (_fl(fs[1](c)) - _fl(fs[0](c))) * c.gen(oid, site).random(c.N)
    if fn == "exponential":
        return lambda c: c.gen(oid, site).standard_exponential(c.N) / _fl(fs[0](c))
    raise ValueError(fn)


class _Ctx:
    def __init__(self, N, dt, seed):
        self.N, self.dt, self.seed, self.t = N, dt, int(seed), 0
        self.cur, self.hist, self.init, self.clips = {}, {}, {}, {}

    def gen(self, oid, site):
        return np.random.default_rng([self.seed, oid, site, self.t])

    def lag(self, name, k):
        ts = self.t - k
        h = self.hist.get(name)
        if h is None:
            return self.cur[name]
        if ts < 0:
            if name in self.init:
                return self.init[name]
            return np.full(self.N, np.nan)
        return h[ts]


# ============================================================================ compiled document and simulation
class Compiled:
    """A statically clean document compiled to closures."""

    def __init__(self, d):
        self.d = d
        self.codes = Codes()
        tm = d["time"]
        self.static, self.steps, self.dt, self.unit = tm["static"], (0 if tm["static"] else tm["steps"]), \
            tm["dt"], tm.get("unit", "")
        self.params = {n: r["value"] for n, r in d["params"].items()}
        self.rules = {n: r["value"] for n, r in d["rules"].items()}
        self.unknowns = d["unknowns"]
        self.hyps = d["hyps"]
        for h in self.hyps.values():
            h["codes"] = [self.codes.code(v) for v in h["values"]]
        for u in self.unknowns.values():
            if u["dist"]["kind"] == "choice":
                u["codes"] = [self.codes.code(v) for v in u["dist"]["options"]]
        self.nodes = d["nodes"]
        self.order = d["order"]
        self.fn = {n: compile_expr(nd["ast"], self.codes, n) for n, nd in self.nodes.items()}
        self.v0 = {n: self._val(nd["v0"]) for n, nd in self.nodes.items() if nd["v0"] is not None}
        self.facts = d["facts"]
        self.claims = d["claims"]
        for o in self.facts + self.claims:
            o["fn"] = compile_expr(o["ast"], self.codes, "fact:" + o["name"])
        self.targets = [t for t, _ in d["targets"]]
        self.whatifs = d["whatifs"]
        self.track = list(dict.fromkeys(self.targets + [t for w in self.whatifs for t in w["targets"]
                                                        if t in d["decl"]]))
        self.cont = [n for n, u in self.unknowns.items() if u["dist"]["kind"] in CONT_DISTS]
        self.maxlag = {}
        for nd in self.nodes.values():
            for x, lag in _expr.refs(nd["ast"]):
                self.maxlag[x] = max(self.maxlag.get(x, 0), lag)
        for o in self.facts + self.claims:
            for x, lag in _expr.refs(o["ast"]):
                self.maxlag[x] = max(self.maxlag.get(x, 0), lag)
        self.stoch = any(_has_stoch(nd["ast"]) for nd in self.nodes.values())
        rng = np.random.default_rng(12345)
        self.coef = rng.random(self.steps + 1) + 0.5
        self.type_of = {n: _var_type(d, n) for n in d["decl"]}

    def _val(self, v):
        if isinstance(v, str):
            return self.codes.code(v)
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        return float(v)

    def fmt(self, name, v):
        if self.type_of.get(name) == "cat":
            return "'" + self.codes.name(v) + "'"
        return _g(v)

    # -------------------------------------------------------------- prior draws
    def draw_prior(self, N, seed, only_discrete=False):
        rng = np.random.default_rng([int(seed), 1])
        U = {}
        qd = [n for n, u in self.unknowns.items() if u["dist"]["kind"] != "choice"]
        if only_discrete:
            qd = [n for n in qd if n not in self.cont]
        if qd:
            H = halton(N, len(qd), rng) if not only_discrete else rng.random((N, len(qd)))
            for j, n in enumerate(qd):
                U[n] = _fl(_inv_dist(self.unknowns[n]["dist"], H[:, j]))
        for n, u in self.unknowns.items():
            if u["dist"]["kind"] == "choice":
                U[n] = _stratified(u["codes"], u["dist"]["weights"], N, rng)
        for n, h in self.hyps.items():
            U[n] = _stratified(h["codes"], h["weights"], N, rng)
        return U

    # -------------------------------------------------------------- simulation
    def simulate(self, U, N, seed, *, sets=None, do=None, keep_all=False, deadline=None, extra_obs=()):
        """Run all members. sets: {param/rule: value}; do: {node: ('const', v, from) | ('expr', fn, from)}.
        Returns a result dict (targets' trajectories, fact/claim reductions, checksums, failures)."""
        sets, do = sets or {}, do or {}
        S = self.steps
        c = _Ctx(N, self.dt, seed)
        for n, v in self.params.items():
            c.cur[n] = self._val(sets.get(n, v))
        for n, v in self.rules.items():
            c.cur[n] = self._val(sets.get(n, v))
        for n in list(self.unknowns) + list(self.hyps):
            c.cur[n] = U[n]
        for n, v in self.v0.items():
            c.init[n] = v
        keep = {n: (S + 1 if keep_all or n in self.track else self.maxlag.get(n, 0) + 1) for n in self.nodes}
        for n in self.nodes:
            c.hist[n] = []
        tgt = {n: np.empty((S + 1, N)) for n in self.track if n in self.nodes}
        cs = {n: np.zeros(N) for n in self.nodes}
        failed = np.zeros(N, bool)
        first_err, nfail_node, rounds = None, {}, {}
        obs = list(self.facts) + list(self.claims) + list(extra_obs)
        acc = [_ObsAcc(o, N, S) for o in obs]
        full = {n: [] for n in self.nodes} if keep_all else None
        with np.errstate(all="ignore"):
            for t in range(S + 1):
                if deadline is not None and time.monotonic() > deadline:
                    raise _Timeout()
                c.t = t
                for n in self.order:
                    nd = self.nodes[n]
                    dv = do.get(n)
                    if dv is not None and dv[0] == "const" and t >= dv[2]:
                        v = dv[1]
                    elif dv is not None and dv[0] == "expr" and t >= dv[2]:
                        v = dv[1](c)
                    elif t == 0 and n in self.v0 and not self.static:
                        v = self.v0[n]
                    else:
                        v = self.fn[n](c)
                    v = np.array(np.broadcast_to(_fl(v), (N,)), dtype=float)
                    if nd["type"] == "int":
                        r = np.rint(v)
                        nr = int(np.sum(np.abs(v - r) > 1e-9))
                        if nr:
                            rounds[n] = rounds.get(n, 0) + nr
                        v = r
                    elif nd["type"] == "bool":
                        v = np.where(np.isnan(v), np.nan, _fl(v != 0))
                    bad = ~np.isfinite(v)
                    if bad.any():
                        newly = bad & ~failed
                        if newly.any():
                            nfail_node[n] = nfail_node.get(n, 0) + int(newly.sum())
                            if first_err is None:
                                first_err = (n, nd["line"], t)
                        failed |= bad
                        v[bad] = 0.0
                    c.cur[n] = v
                    h = c.hist[n]
                    h.append(v)
                    if len(h) > keep[n] and not keep_all:
                        h[len(h) - keep[n] - 1] = None
                    cs[n] += self.coef[t] * v
                    if n in tgt:
                        tgt[n][t] = v
                    if keep_all:
                        full[n].append(v)
                for a in acc:
                    a.update(c, t)
        res = {"N": N, "tgt": tgt, "cs": cs, "failed": failed, "first_err": first_err, "nfail": nfail_node,
               "rounds": rounds, "clips": dict(c.clips), "obs": [a.finish(failed) for a in acc],
               "final": {n: np.broadcast_to(_fl(v), (N,)) for n, v in c.cur.items()}}
        for t in self.track:
            if t not in self.nodes:
                v = np.broadcast_to(_fl(c.cur[t]), (N,))
                res["tgt"][t] = np.broadcast_to(v, (S + 1, N))
        if keep_all:
            res["full"] = {n: np.stack(v) for n, v in full.items()}
        return res


class _ObsAcc:
    """Accumulates one fact/claim over a run: likelihood, satisfied bit, violation, value shown, checksum."""

    def __init__(self, o, N, S):
        self.o, self.N, self.S = o, N, S
        T = o["T"]
        self.T = S if T == "final" else T
        self.lik = None
        self.sat = None
        self.viol = None
        self.g = None
        self.cs = np.zeros(N)

    def _when(self, t):
        return self.T == "any" or self.T == "all" or t == self.T

    def update(self, c, t):
        if not self._when(t):
            return
        o = self.o
        g = np.array(np.broadcast_to(_fl(o["fn"](c)), (self.N,)))
        g = np.where(np.isfinite(g), g, np.nan)
        self.cs += (t + 1.37) * np.nan_to_num(g, nan=-7.77)
        if o["kind"] == "soft":
            z = (g - o["v"]) / o["tol"]
            lik, sat, viol = np.exp(-0.5 * z * z), np.abs(z) <= 2, 0.5 * z * z
        elif o["kind"] == "interval":
            sat = (g >= o["lo"]) & (g <= o["hi"])
            scale = max(o["hi"] - o["lo"], 1e-6 * max(1.0, abs(o["lo"]), abs(o["hi"])))
            viol = np.maximum(np.maximum(o["lo"] - g, g - o["hi"]), 0) / scale
            lik = _fl(sat)
        else:
            lik = sat = _tr(g) & ~np.isnan(g)       # the predicate value; truth applied at the end
            viol = None
        lik = np.nan_to_num(lik)
        sat = np.asarray(sat, bool) & ~np.isnan(g)
        if viol is not None:
            viol = np.where(np.isnan(viol), 1e9, viol)
        if self.lik is None:
            self.lik, self.sat, self.viol, self.g = lik, sat, viol, g
            return
        if self.T == "any":
            self.lik, self.sat = np.maximum(self.lik, lik), self.sat | sat
            self.viol = None if viol is None else np.minimum(self.viol, viol)
        elif self.T == "all":
            self.lik, self.sat = np.minimum(self.lik, lik), self.sat & sat
            self.viol = None if viol is None else np.maximum(self.viol, viol)
        self.g = g

    def finish(self, failed):
        o = self.o
        if self.lik is None:
            z = np.zeros(self.N)
            return {"lik": z, "sat": z.astype(bool), "viol": z + 1, "g": z + np.nan, "cs": self.cs}
        if o["kind"] == "pred":
            ok = self.sat == bool(o.get("truth", True))
            lik, sat, viol = _fl(ok), ok, 1.0 - _fl(ok)
            val = _fl(self.sat)
        else:
            lik, sat, viol, val = self.lik, self.sat, self.viol, self.g
        lik = np.where(failed, 0.0, lik)
        sat = sat & ~failed
        viol = np.where(failed, 1e9, viol)
        return {"lik": lik, "sat": sat, "viol": viol, "g": val, "cs": self.cs}


def _fact_weight(o, lik):
    r = o.get("trust")
    return lik if not r else r * lik + (1 - r)


# ============================================================================ weighted statistics
def wmean(x, w):
    s = w.sum()
    return float((x * w).sum() / s) if s > 0 else float("nan")


def wquant(x, w, qs):
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = w > 0
    if not ok.any():
        return [float("nan")] * len(qs)
    x, w = x[ok], w[ok]
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    cw = np.cumsum(w) / w.sum()
    return [float(x[min(np.searchsorted(cw, q, side="left"), len(x) - 1)]) for q in qs]


def ess(w):
    s, s2 = w.sum(), (w * w).sum()
    return float(s * s / s2) if s2 > 0 else 0.0


def first_order(y, x, w, discrete=False, bins=10):
    """Given-data estimator of S = Var(E[Y | X]) / Var(Y) under weights w (quantile bins or discrete values).
    Returns (S, groups) with groups = [(label, weight share, weighted mean, weighted median)]."""
    W = w.sum()
    if W <= 0:
        return float("nan"), []
    m = (y * w).sum() / W
    var = (w * (y - m) ** 2).sum() / W
    if discrete:
        keys = np.unique(x[w > 0])
        idx = np.searchsorted(keys, x)
        labels = list(keys)
    else:
        edges = np.unique(wquant(x, w, [i / bins for i in range(1, bins)]))
        idx = np.searchsorted(edges, x, side="right")
        labels = list(range(len(edges) + 1))
    groups, between = [], 0.0
    for j, lab in enumerate(labels):
        sel = (idx == j) & (w > 0)
        wj = w[sel].sum()
        if wj <= 0:
            continue
        mj = (y[sel] * w[sel]).sum() / wj
        between += wj * (mj - m) ** 2
        groups.append((lab, wj / W, mj, wquant(y[sel], w[sel], [0.5])[0]))
    if var <= 1e-15 * max(1.0, m * m):
        return 0.0, groups
    raw = between / W / var
    bias = (len(groups) - 1) / max(ess(w), 1.0)     # expected between-bin share from noise alone (given-data estimator)
    return float(min(1.0, max(0.0, (raw - bias) / (1 - bias)) if bias < 1 else 0.0)), groups


def _entropy(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def best_stump(c, X, w, discrete):
    """Best single split of the claim indicator c on one variable: (name, kind, threshold/value, p_in, p_out, IG).
    X: {name: values}; discrete: set of names split by value."""
    W = w.sum()
    if W <= 0:
        return None
    p = float((c * w).sum() / W)
    H = _entropy(p)
    best = None
    for name, x in X.items():
        if name in discrete:
            cands = [("==", v) for v in np.unique(x[w > 0])]
        else:
            qs = np.unique(wquant(x, w, [i / 20 for i in range(1, 20)]))
            cands = [(">", v) for v in qs]
        for op, v in cands:
            inn = (x == v) if op == "==" else (x > v)
            wi, wo = w[inn].sum(), w[~inn].sum()
            if wi <= 0 or wo <= 0:
                continue
            pi, po = (c[inn] * w[inn]).sum() / wi, (c[~inn] * w[~inn]).sum() / wo
            ig = H - (wi * _entropy(pi) + wo * _entropy(po)) / W
            if best is None or ig > best[5] + 1e-12:
                best = (name, op, float(v), float(pi), float(po), float(ig))
    return best


def prim_box(c, X, w, names, peel=0.05, min_cov=0.5):
    """PRIM-lite on 1-2 continuous variables: peel `peel` of the box mass from one side per step, choosing the
    peel with the highest density, while coverage (share of claim-true weight inside) stays >= min_cov.
    Returns dict(box={name: [lo, hi]}, density, coverage, mass) or None."""
    W = w.sum()
    Wc = (c * w).sum()
    if W <= 0 or Wc <= 0 or not names:
        return None
    box = {n: [float(X[n][w > 0].min()), float(X[n][w > 0].max())] for n in names}

    def inside(b):
        m = np.ones(len(w), bool)
        for n, (lo, hi) in b.items():
            m &= (X[n] >= lo) & (X[n] <= hi)
        return m

    def stats(m):
        wb = w[m].sum()
        return (float((c[m] * w[m]).sum() / wb) if wb > 0 else 0.0, float((c[m] * w[m]).sum() / Wc), float(wb / W))

    cur = inside(box)
    dens, cov, mass = stats(cur)
    for _ in range(200):
        best = None
        for n in names:
            x = X[n]
            sel = cur & (w > 0)
            if sel.sum() < 4:
                continue
            lo_q, hi_q = wquant(x[sel], w[sel], [peel, 1 - peel])
            for side, val in (("lo", lo_q), ("hi", hi_q)):
                nb = {k: list(v) for k, v in box.items()}
                if side == "lo":
                    if val <= nb[n][0]:
                        continue
                    nb[n][0] = val
                else:
                    if val >= nb[n][1]:
                        continue
                    nb[n][1] = val
                m = inside(nb)
                dn, cv, ms = stats(m)
                if cv < min_cov or ms <= 0:
                    continue
                if best is None or dn > best[0]:
                    best = (dn, cv, ms, nb, m)
        if best is None or best[0] <= dens + 1e-9:
            break
        dens, cov, mass, box, cur = best
    return {"box": box, "density": dens, "coverage": cov, "mass": mass}


def minimal_hitting_sets(k, sets_to_hit, limit=50):
    """All minimal subsets of {0..k-1} (bitmasks) that intersect every mask in sets_to_hit (k <= 16)."""
    if any(s == 0 for s in sets_to_hit):
        return []
    masks = np.arange(1 << k, dtype=np.int64)
    ok = np.ones(1 << k, bool)
    for s in set(sets_to_hit):
        ok &= (masks & s) != 0
    pc = np.array([bin(int(m)).count("1") for m in masks]) if k <= 16 else None
    out = []
    for size in range(1, k + 1):
        cand = masks[ok & (pc == size)]
        for m in cand:
            m = int(m)
            if any((m & f) == f for f in out):
                continue
            out.append(m)
            if len(out) >= limit:
                return out
        for f in out:
            ok &= (masks & f) != f
    return out


def mus_from_vectors(k, vectors):
    """Empirical MUSes of k facts from the satisfied-fact bit vectors of members (MUS/MCS hitting-set duality)."""
    full = (1 << k) - 1
    vs = set(int(v) for v in vectors)
    if full in vs:
        return []
    if not vs:
        vs = {0}
    maximal = [v for v in vs if not any((v | o) == o and v != o for o in vs)]
    comps = [full & ~v for v in maximal]
    return minimal_hitting_sets(k, comps)


def _bits(lst):
    return [i for i in range(64) if lst >> i & 1]


# ============================================================================ static coherence (z3)
class _NoZ3(Exception):
    pass


def static_coherence(cd, budget_s=None, max_mus=5):
    """Exact coherence of the facts in a static document. Returns dict(status sat|unsat|fallback, muses=[[names]],
    soft=[names], note)."""
    t0 = time.monotonic()
    budget_s = Z3_BUDGET_S if budget_s is None else budget_s
    try:
        import z3
    except Exception:
        return {"status": "fallback", "note": "z3 is not available"}
    if budget_s <= 0:
        return {"status": "fallback", "note": "solver budget exhausted before the first check"}
    if cd.stoch:
        return {"status": "fallback", "note": "the document uses random functions, which z3 cannot translate"}
    vars_, base = {}, []

    def real_of(e):
        return z3.If(e, z3.RealVal(1), z3.RealVal(0)) if z3.is_bool(e) else (z3.ToReal(e) if z3.is_int(e) else e)

    def bool_of(e):
        return e if z3.is_bool(e) else real_of(e) != 0

    for n, u in cd.unknowns.items():
        dd = u["dist"]
        k, a = dd["kind"], dd.get("args")
        if k == "choice":
            v = z3.Int(n)
            base.append(z3.Or(*[v == int(cc) for cc, wt in zip(u["codes"], dd["weights"]) if wt > 0]))
        elif k == "int_uniform":
            v = z3.Int(n)
            base += [v >= int(a[0]), v <= int(a[1])]
        elif k == "bernoulli":
            v = z3.Bool(n)
        else:
            v = z3.Real(n)
            lo, hi = (a[0], a[1]) if k in ("uniform", "loguniform") else (a[0], a[2]) if k == "triangular" else \
                (a[0] - 4 * a[1], a[0] + 4 * a[1])
            base += [v >= z3.RealVal(repr(float(lo))), v <= z3.RealVal(repr(float(hi)))]
        vars_[n] = v
    for n, h in cd.hyps.items():
        v = z3.Int(n)
        base.append(z3.Or(*[v == int(cc) for cc, wt in zip(h["codes"], h["weights"]) if wt > 0]))
        vars_[n] = v
    memo = {}

    def tr(ast):
        k = ast[0]
        if k == "num":
            return z3.RealVal(repr(float(ast[1])))
        if k == "bool":
            return z3.BoolVal(bool(ast[1]))
        if k == "str":
            return z3.IntVal(int(cd.codes.code(ast[1])))
        if k == "var":
            n = ast[1]
            if n == "t":
                return z3.RealVal(0)
            if n == "dt":
                return z3.RealVal(repr(float(cd.dt)))
            if n in vars_:
                return vars_[n]
            if n in cd.params:
                return z3.RealVal(repr(float(cd.params[n])))
            if n in cd.rules:
                v = cd.rules[n]
                return z3.BoolVal(v) if isinstance(v, bool) else (z3.IntVal(int(cd.codes.code(v))) if
                                                                   isinstance(v, str) else z3.RealVal(repr(float(v))))
            if n in cd.nodes:
                if n not in memo:
                    nd = cd.nodes[n]
                    e = tr(nd["ast"])
                    if nd["type"] == "int":
                        e = z3.ToReal(z3.ToInt(real_of(e) + z3.RealVal("0.5")))
                    elif nd["type"] == "bool":
                        e = bool_of(e)
                    memo[n] = e
                return memo[n]
            raise _NoZ3(f"name {n}")
        if k == "neg":
            return -real_of(tr(ast[1]))
        if k == "not":
            return z3.Not(bool_of(tr(ast[1])))
        if k == "and":
            return z3.And(bool_of(tr(ast[1])), bool_of(tr(ast[2])))
        if k == "or":
            return z3.Or(bool_of(tr(ast[1])), bool_of(tr(ast[2])))
        if k == "if":
            a, b = tr(ast[2]), tr(ast[3])
            if z3.is_int(a) and z3.is_int(b):
                return z3.If(bool_of(tr(ast[1])), a, b)
            if z3.is_bool(a) and z3.is_bool(b):
                return z3.If(bool_of(tr(ast[1])), a, b)
            return z3.If(bool_of(tr(ast[1])), real_of(a), real_of(b))
        if k == "bin":
            op = ast[1]
            a, b = tr(ast[2]), tr(ast[3])
            if op in ("==", "!="):
                if z3.is_int(a) and z3.is_int(b):
                    return (a == b) if op == "==" else (a != b)
                if z3.is_bool(a) and z3.is_bool(b):
                    return (a == b) if op == "==" else (a != b)
                a, b = real_of(a), real_of(b)
                return (a == b) if op == "==" else (a != b)
            a, b = real_of(a), real_of(b)
            if op == "<":
                return a < b
            if op == "<=":
                return a <= b
            if op == ">":
                return a > b
            if op == ">=":
                return a >= b
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                return a / b
            if op == "^":
                ex = ast[3]
                if ex[0] == "num" and float(ex[1]) == int(ex[1]) and 0 <= ex[1] <= 6:
                    r = z3.RealVal(1)
                    for _ in range(int(ex[1])):
                        r = r * a
                    return r
            raise _NoZ3(f"operator {op}")
        if k == "call":
            fn, args = ast[1], [tr(x) for x in ast[2]]
            if fn not in Z3_FUNCS:
                raise _NoZ3(f"function {fn}")
            args = [real_of(x) for x in args]
            if fn == "min":
                return reduce(lambda x, y: z3.If(x <= y, x, y), args)
            if fn == "max":
                return reduce(lambda x, y: z3.If(x >= y, x, y), args)
            if fn == "abs":
                return z3.If(args[0] >= 0, args[0], -args[0])
            if fn == "floor":
                return z3.ToReal(z3.ToInt(args[0]))
            if fn == "ceil":
                return -z3.ToReal(z3.ToInt(-args[0]))
            if fn == "clip":
                x = z3.If(args[0] >= args[1], args[0], args[1])
                return z3.If(x <= args[2], x, args[2])
        raise _NoZ3(f"node {k}")

    try:
        cons = []
        for o in cd.facts:
            g = tr(o["ast"])
            if o["kind"] == "soft":
                gr = real_of(g)
                v, tol = z3.RealVal(repr(o["v"])), z3.RealVal(repr(2 * o["tol"]))
                cons.append(z3.And(gr >= v - tol, gr <= v + tol))
            elif o["kind"] == "interval":
                gr = real_of(g)
                cons.append(z3.And(gr >= z3.RealVal(repr(o["lo"])), gr <= z3.RealVal(repr(o["hi"]))))
            else:
                cons.append(bool_of(g) if o["truth"] else z3.Not(bool_of(g)))
    except _NoZ3 as e:
        return {"status": "fallback", "note": f"not translatable to z3 ({e})"}
    names = [o["name"] for o in cd.facts]
    k = len(names)
    sel = [z3.Bool(f"__sel_{i}") for i in range(k)]
    s = z3.Solver()
    s.add(*base)
    for i in range(k):
        s.add(z3.Implies(sel[i], cons[i]))

    def left():
        return budget_s - (time.monotonic() - t0)

    def check(assume):
        lft = left()
        if lft <= 0:
            raise _Timeout()
        s.set("timeout", max(1, int(lft * 1000)))
        r = s.check(*assume)
        if r == z3.unknown:
            raise _Timeout()
        return r == z3.sat

    soft = [o["name"] for o in cd.facts if o.get("trust")]
    try:
        if check(sel):
            return {"status": "sat", "muses": [], "soft": soft}
        muses, blocked = [], []
        mp = z3.Solver()
        for _ in range(64):
            if len(muses) >= max_mus or left() <= 0:
                break
            if mp.check() != z3.sat:
                break
            m = mp.model()
            # unconstrained map variables count as true, so seeds start maximal (MARCO)
            seed = [i for i in range(k) if not z3.is_false(m.eval(z3.Bool(f"__m_{i}"), model_completion=False))]
            if check([sel[i] for i in seed]):
                # grow to a maximal satisfiable set, then block it from below
                grown = list(seed)
                for i in range(k):
                    if i not in grown and check([sel[j] for j in grown + [i]]):
                        grown.append(i)
                rest = [i for i in range(k) if i not in grown]
                if not rest:
                    break
                mp.add(z3.Or(*[z3.Bool(f"__m_{i}") for i in rest]))
            else:
                core = [i for i in range(k) if any(z3.eq(sel[i], c) for c in s.unsat_core())] or seed
                mus = list(core)
                for i in list(mus):
                    trial = [j for j in mus if j != i]
                    if trial and not check([sel[j] for j in trial]):
                        mus = trial
                muses.append([names[i] for i in mus])
                mp.add(z3.Not(z3.And(*[z3.Bool(f"__m_{i}") for i in mus])))
        return {"status": "unsat", "muses": muses, "soft": soft}
    except _Timeout:
        return {"status": "fallback", "note": f"solver budget of {budget_s:g} s exhausted"}


# ============================================================================ ensemble, conditioning
class Ensemble:
    pass


def _weights(cd, res, extra=None):
    w = np.where(res["failed"], 0.0, 1.0)
    for o, r in zip(cd.facts, res["obs"][:len(cd.facts)]):
        w = w * _fact_weight(o, r["lik"])
    if extra:
        for o, r in extra:
            w = w * _fact_weight(o, r["lik"])
    return w


def _violation(cd, res, idx=None):
    v = np.zeros(res["N"])
    for i, (o, r) in enumerate(zip(cd.facts, res["obs"][:len(cd.facts)])):
        if idx is not None and i not in idx:
            continue
        v = v + r["viol"] * (o.get("trust") or 1.0)
    return np.where(res["failed"], 1e12, v)


def build_ensemble(cd, N, deadline, seed=SEED, allow_refine=True):
    E = Ensemble()
    E.N, E.notes = N, []
    U0 = cd.draw_prior(N, seed)
    r0 = cd.simulate(U0, N, seed, deadline=deadline)
    w0 = _weights(cd, r0)
    E.U0, E.res0, E.w0 = U0, r0, w0
    E.U, E.res, E.w, E.seed = U0, r0, w0, seed
    E.corr = np.ones(N)               # importance correction pi/q of the members (1 for prior draws)
    E.ess0 = E.ess = ess(w0)
    E.refined = False
    if E.ess < MIN_ESS and cd.facts and allow_refine:
        if not cd.cont:
            E.notes.append(f"ESS {E.ess:.1f} < {MIN_ESS:g}; no continuous unknown to refine")
        else:
            try:
                rr = _refine(cd, E, N, seed + 1, deadline)
            except np.linalg.LinAlgError:
                rr = None
            if rr is not None:
                U1, r1, w1, corr = rr
                e1 = ess(w1)
                E.notes.append(f"refinement round (ESS {E.ess0:.1f} < {MIN_ESS:g}): re-drew {N} members from a "
                               f"Gaussian kernel fitted to the top 5% on {', '.join(cd.cont)}, importance-corrected "
                               f"against the prior; ESS now {e1:.1f}")
                if e1 > E.ess:
                    E.U, E.res, E.w, E.seed, E.ess, E.refined, E.corr = U1, r1, w1, seed + 1, e1, True, corr
                else:
                    E.notes.append("refined ensemble not better; the original ensemble is kept")
    E.low = E.ess < MIN_ESS
    return E


def _refine(cd, E, N, seed, deadline):
    X = np.column_stack([E.U0[n] for n in cd.cont])
    if E.w0.sum() > 0:
        order = np.argsort(-E.w0, kind="stable")
    else:
        order = np.argsort(_violation(cd, E.res0), kind="stable")
    K = max(10, int(0.05 * N))
    top = X[order[:K]]
    mu = top.mean(axis=0)
    C = np.atleast_2d(np.cov(top.T)) if top.shape[0] > 1 else np.zeros((len(cd.cont),) * 2)
    span = np.array([_spread(cd.unknowns[n]["dist"]) for n in cd.cont])
    C = C + np.diag((1e-4 * span) ** 2 + 1e-12)
    rng = np.random.default_rng([int(seed), 7])
    Lc = np.linalg.cholesky(C)
    Z = rng.standard_normal((N, len(cd.cont)))
    Xn = mu + Z @ Lc.T
    logq = -0.5 * (Z ** 2).sum(axis=1) - np.log(np.diag(Lc)).sum() - 0.5 * len(cd.cont) * math.log(2 * math.pi)
    logpi = np.zeros(N)
    for j, n in enumerate(cd.cont):
        logpi = logpi + _log_pdf(cd.unknowns[n]["dist"], Xn[:, j])
    lr = logpi - logq
    fin = np.isfinite(lr)
    corr = np.zeros(N)
    if fin.any():
        corr[fin] = np.exp(lr[fin] - lr[fin].max())
    U1 = cd.draw_prior(N, seed, only_discrete=True)
    for j, n in enumerate(cd.cont):
        U1[n] = np.where(fin, Xn[:, j], mu[j])
    r1 = cd.simulate(U1, N, seed, deadline=deadline)
    return U1, r1, corr * _weights(cd, r1), corr


def _spread(d):
    a = d["args"]
    if d["kind"] in ("uniform", "loguniform"):
        return a[1] - a[0]
    if d["kind"] == "triangular":
        return a[2] - a[0]
    return 4 * a[1]


def _local_search(cd, E, idx, deadline, budget=LOCAL_SEARCH_RUNS):
    """Bounded random-restart coordinate search for a member satisfying all facts in idx. Returns its bit vector."""
    names = list(cd.unknowns) + list(cd.hyps)
    if not names:
        return None
    want = sum(1 << i for i in idx)
    v0 = _violation(cd, E.res0, set(idx))
    o = np.argsort(v0, kind="stable")[:10]
    elite = {n: E.U0[n][o].copy() for n in names}
    ev = v0[o]
    rng = np.random.default_rng(99)
    rounds, per = 4, max(1, budget // 4)
    for r in range(rounds):
        if time.monotonic() > deadline:
            return None
        pick = rng.integers(0, len(ev), per)
        cand = {n: elite[n][pick].copy() for n in names}
        fresh = cd.draw_prior(per, 1000 + r)
        for i in range(per):
            n = names[rng.integers(0, len(names))]       # coordinate move, or a random restart of that coordinate
            if rng.random() < 0.3 or n not in cd.unknowns or cd.unknowns[n]["dist"]["kind"] not in CONT_DISTS:
                cand[n][i] = fresh[n][i]
            else:
                cand[n][i] = np.clip(cand[n][i] + rng.normal(0, 0.1 * _spread(cd.unknowns[n]["dist"])),
                                     *_support(cd.unknowns[n]["dist"]))
        res = cd.simulate(cand, per, 5000 + r, deadline=deadline)
        bits = _bitvec(cd, res)
        hit = np.nonzero((bits >= 0) & ((bits & want) == want))[0]
        if hit.size:
            return int(bits[hit[0]])
        v = _violation(cd, res, set(idx))
        allv = np.concatenate([ev, v])
        allc = {n: np.concatenate([elite[n], cand[n]]) for n in names}
        o = np.argsort(allv, kind="stable")[:10]
        elite, ev = {n: allc[n][o] for n in names}, allv[o]
    return None


def _support(d):
    a = d["args"]
    if d["kind"] in ("uniform", "loguniform"):
        return a[0], a[1]
    if d["kind"] == "triangular":
        return a[0], a[2]
    return -np.inf, np.inf


def _bitvec(cd, res):
    k = min(len(cd.facts), MAX_MUS_FACTS)
    b = np.zeros(res["N"], dtype=np.int64)
    for i in range(k):
        b |= res["obs"][i]["sat"].astype(np.int64) << i
    return np.where(res["failed"], -1, b)


def dynamic_incoherence(cd, E, deadline):
    k = min(len(cd.facts), MAX_MUS_FACTS)
    vecs = set()
    for res in ([E.res0, E.res] if E.refined else [E.res0]):
        vecs |= set(int(v) for v in np.unique(_bitvec(cd, res)) if v >= 0)
    muses = mus_from_vectors(k, vecs)
    searched = 0
    for _ in range(5):
        found_new = False
        for m in muses[:5]:
            if time.monotonic() > deadline:
                break
            searched += 1
            v = _local_search(cd, E, _bits(m), deadline)
            if v is not None and v >= 0:
                vecs.add(v)
                found_new = True
        if not found_new:
            break
        muses = mus_from_vectors(k, vecs)
    return {"muses": [[cd.facts[i]["name"] for i in _bits(m)] for m in muses], "k": k, "searched": searched}


# ============================================================================ run specs
def _changes_from_run(d, r):
    """A `run` entry (dict) -> list of raw changes."""
    out = []
    for field, kind in (("set", "set"), ("params", "set"), ("rules", "set"), ("overrides", "set"), ("do", "do"),
                        ("fix", "fix")):
        v = r.get(field)
        if v is None:
            continue
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except (ValueError, TypeError):
                raise ValueError(f"{field} must be an object of name -> value")
        if not isinstance(v, dict):
            raise ValueError(f"{field} must be an object of name -> value")
        for name, val in v.items():
            if kind == "do" and isinstance(val, str) and val.strip().startswith(":="):
                src = val.strip()[2:].strip()
                out.append(("doexpr", str(name), (parse_expr(src), src)))
            elif kind == "do" and isinstance(val, dict):
                out.append(("do", str(name), (_parse_value(val.get("value")), int(val.get("from", 0) or 0))))
            else:
                vv = _parse_value(val) if isinstance(val, str) else (float(val) if _isnum(val) else val)
                out.append((kind, str(name), (vv, 0)))
    return out


def _apply(cd, changes):
    """Resolved changes -> (sets, do, fix) for simulate; raises ValueError on a same-step cycle."""
    sets, do, fix = {}, {}, {}
    deps = {n: set(v) for n, v in cd.d["deps"].items()}
    for kind, name, payload in changes:
        if kind == "set":
            sets[name] = payload
        elif kind == "fix":
            fix[name] = payload
        elif kind == "do":
            v, frm = payload
            do[name] = ("const", cd._val(v), int(frm))
            deps[name] = set()
        elif kind == "doexpr":
            ast, src = payload
            do[name] = ("expr", compile_expr(ast, cd.codes, name), 0)
            deps[name] = {x for x, lag in _expr.refs(ast) if lag == 0 and x in cd.nodes}
    order, cyc = _topo(list(cd.nodes), deps)
    if cyc:
        raise ValueError("the intervention creates a same-step cycle: " + " -> ".join(cyc))
    return sets, do, fix, order


def _fixed_U(cd, U, fix, N):
    if not fix:
        return U
    U2 = dict(U)
    for n, v in fix.items():
        U2[n] = np.full(N, cd._val(v))
    return U2


def run_changes(cd, E, changes, deadline, U=None, N=None, seed=None):
    sets, do, fix, order = _apply(cd, changes)
    U = E.U if U is None else U
    N = E.N if N is None else N
    saved = cd.order
    cd.order = order
    try:
        return cd.simulate(_fixed_U(cd, U, fix, N), N, E.seed if seed is None else seed, sets=sets, do=do,
                           deadline=deadline)
    finally:
        cd.order = saved


def _change_text(changes):
    out = []
    for kind, name, p in changes:
        if kind == "doexpr":
            out.append(f"do {name} := {p[1]}")
        elif kind == "do":
            out.append(f"do {name} = {_vt(p[0])}" + (f" from {p[1]}" if p[1] else ""))
        elif kind == "fix":
            out.append(f"fix {name} = {_vt(p)}")
        else:
            out.append(f"set {name} = {_vt(p)}")
    return ", ".join(out)


def _vt(v):
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, bool):
        return "true" if v else "false"
    return _g(v)


# ============================================================================ report
def _target_at(res, t, at, S):
    if t not in res["tgt"]:
        if at not in (None, "final") and int(at) != S:
            raise ValueError(f"'{t}' at step {at}: only targets and whatif targets keep their trajectories (use "
                             f"at: final, or add {t} to targets:)")
        return np.asarray(res["final"][t], float)
    h = res["tgt"][t]
    k = S if at in (None, "final") else max(0, min(S, int(at)))
    return np.asarray(h[k], float)


def _dist_line(cd, name, x, w, is_int):
    if np.asarray(w).sum() <= 0:
        return "undefined (no member is compatible with the facts)"
    if cd.type_of.get(name) == "cat":
        W = w.sum()
        vals = np.unique(x)
        parts = sorted(((float(w[x == v].sum() / W), v) for v in vals), reverse=True)
        return ", ".join(f"'{cd.codes.name(v)}' {p:.2f}" for p, v in parts[:6])
    if cd.type_of.get(name) == "bool":
        return f"P(true)={wmean(x, w):.3f}"
    q = wquant(x, w, [0.05, 0.5, 0.95])
    s = f"mean {_g(wmean(x, w))}, p05 {_g(q[0])} p50 {_g(q[1])} p95 {_g(q[2])}"
    if is_int:
        s += f", P(>0)={wmean(_fl(x > 0), w):.2f}"
    return s


def _is_int(cd, name):
    k = _kind(cd.d, name)
    if k == "node":
        return cd.nodes[name]["type"] == "int"
    return k == "unknown" and cd.unknowns[name]["dist"]["kind"] == "int_uniform"


def _sens_vars(cd):
    return list(cd.unknowns) + list(cd.hyps)


def _is_discrete_var(cd, n):
    if n in cd.hyps:
        return True
    k = cd.unknowns[n]["dist"]["kind"]
    return k in ("choice", "bernoulli") or (k == "int_uniform" and
                                           cd.unknowns[n]["dist"]["args"][1] - cd.unknowns[n]["dist"]["args"][0] < 10)


def _sens_label(S):
    return "matters" if S >= 0.10 else ("minor" if S >= 0.02 else "negligible")


def build_report(cd, E, *, problems_note=None, whatif_results=(), incoherent=None, static_coh=None,
                 header_extra=(), compute_note=None, at=None):
    S = cd.steps
    w = E.w
    W = w.sum()
    low = E.low
    lowtag = " [LOW-ESS (unreliable)]" if low else ""
    head = []
    if problems_note:
        head += problems_note
    mode = "static" if cd.static else f"dynamic, {S} steps of {_g(cd.dt)} {cd.unit}".rstrip()
    d = cd.d
    head.append(f"DOCUMENT accepted: {mode}; {len(cd.params)} params, {len(cd.rules)} rules, {len(cd.unknowns)} "
                f"unknowns, {len(cd.hyps)} hypotheses, {len(cd.nodes)} nodes, {len(cd.facts)} facts, "
                f"{len(cd.claims)} claims, {len(cd.whatifs)} whatifs")
    res = E.res
    hard = [i for i, o in enumerate(cd.facts) if o["kind"] != "soft" and not o.get("trust")]
    hp = float(np.mean(np.all([E.res0["obs"][i]["sat"] for i in hard], axis=0))) if hard else 1.0
    nf = int(res["failed"].sum())
    head.append(f"ENSEMBLE: N={E.N} members; ESS {E.ess:.1f} ({100 * E.ess / E.N:.1f}% of N)"
                + (f"; hard facts passed by {_pct(hp)} of the prior family" if hard else "")
                + f"; run-time errors in {nf} members" + ("" if not nf else _err_text(cd, res))
                + ("; int rounding: " + ", ".join(f"{n} {c}" for n, c in res["rounds"].items())
                   if res["rounds"] else "")
                + ("; p clipped to [0,1]: " + ", ".join(f"{n} {c}" for n, c in res["clips"].items())
                   if res["clips"] else ""))
    if compute_note:
        head.append(compute_note)
    head += [f"note: {x}" for x in E.notes]
    if low:
        head.append(f"LOW-ESS (unreliable): ESS {E.ess:.1f} < {MIN_ESS:g}. The evidence is nearly incompatible with "
                    f"the family as built; posterior blocks below are unreliable. Read FACTS and INCOHERENT.")
    for i, o in enumerate(cd.facts):
        pr = float(np.mean(E.res0["obs"][i]["sat"]))
        if pr < 0.02:
            head.append(f"WARNING: fact {o['name']} is satisfied by {_pct(pr)} of the family; either the family "
                        f"lacks the mechanism behind it, or a range is wrong")
    head += list(header_extra)
    blocks = []
    # INCOHERENT
    inc = []
    if static_coh is not None:
        if static_coh["status"] == "sat":
            inc.append(f"COHERENCE (z3, exact): the {len(cd.facts)} facts can all hold together under the stated "
                       f"rules and ranges.")
        elif static_coh["status"] == "unsat":
            inc.append("INCOHERENT (z3, exact): the facts cannot all hold. Minimal sets of statements that cannot "
                       "all be true:")
            for m in static_coh["muses"]:
                inc.append("  {" + ", ".join(m) + "}: " + "; ".join(_fact_src(cd, n) for n in m))
            if len(static_coh["muses"]) >= 5:
                inc.append("  (the search stops at 5 sets; more may exist: check the remaining facts with `run` and "
                           "`given`)")
            if static_coh.get("soft"):
                inc.append("  (soft, i.e. trust-marked: " + ", ".join(static_coh["soft"]) + ")")
        else:
            inc.append(f"COHERENCE: z3 not used ({static_coh.get('note')}); falling back to the dynamic (empirical) "
                       f"method.")
        if static_coh["status"] in ("sat", "unsat"):
            unusual = []
            inmus = {n for m in static_coh.get("muses", []) for n in m}
            for i, o in enumerate(cd.facts):
                pr = float(np.mean(E.res0["obs"][i]["sat"]))
                point = o["kind"] == "interval" and o["lo"] == o["hi"]     # a stated exact figure is not "unusual"
                if pr < 0.05 and o["name"] not in inmus and not point:
                    unusual.append(f"{o['name']} ({_pct(pr)} of the ranges)")
            if unusual:
                inc.append("  unusual but possible (NOT incoherent; satisfied only by a corner of the ranges): "
                           + ", ".join(unusual))
    if incoherent is not None:
        if not incoherent["muses"]:
            inc.append(f"INCOHERENT check (empirical): some member satisfies every fact jointly; no incoherent set "
                       f"found.")
        else:
            inc.append(f"INCOHERENT (empirical: unsatisfied by {E.N} members plus local search on "
                       f"{incoherent['searched']} set(s); not a proof). Minimal sets no member satisfies together:")
            for m in incoherent["muses"][:8]:
                inc.append("  {" + ", ".join(m) + "}: " + "; ".join(_fact_src(cd, n) for n in m))
        if len(cd.facts) > MAX_MUS_FACTS:
            inc.append(f"  (only the first {MAX_MUS_FACTS} facts were analysed)")
    if inc:
        blocks.append(("COHERENCE CHECK:", inc, True))
    # HYPOTHESES
    hyp = []
    for hn, h in cd.hyps.items():
        x0, x = E.U0[hn], E.U[hn]
        post = [float(w[x == c].sum() / W) if W > 0 else float("nan") for c in h["codes"]]
        hyp.append(f"  {hn}: " + " | ".join(f"'{v}' {pw:.2f} -> {pp:.2f}" for v, pw, pp in
                                              zip(h["values"], h["weights"], post)) + lowtag)
        rows, disc = [], []
        for i, o in enumerate(cd.facts):
            lik = _fact_weight(o, E.res0["obs"][i]["lik"])
            L = [float(lik[x0 == c].mean()) if (x0 == c).any() else float("nan") for c in h["codes"]]
            rows.append(f"{o['name']}: " + ", ".join(f"{v} {_g(l)}" for v, l in zip(h["values"], L)))
            fin = [(l, v) for l, v in zip(L, h["values"]) if math.isfinite(l)]
            if len(fin) >= 2:
                hi, lo = max(fin), min(fin)
                if hi[0] > 0 and (lo[0] <= 0 or hi[0] / lo[0] >= 3):
                    lr = "inf" if lo[0] <= 0 else _g(hi[0] / lo[0])
                    zero = [v for l, v in fin if l <= 1e-12]
                    disc.append(f"{o['name']} ('{hi[1]}' over '{lo[1]}', LR {lr}"
                                + (f"; rules out {', '.join(repr(z) for z in zero)}" if zero else "") + ")")
        if rows:
            hyp.append("    P(fact | value): " + " | ".join(rows[:12]) + (f" | (+{len(rows) - 12} more facts)"
                                                                         if len(rows) > 12 else ""))
        hyp.append("    discriminating facts (LR >= 3): " + ("; ".join(disc[:12]) + (f"; (+{len(disc) - 12} more)" if
                                                                                  len(disc) > 12 else "")
                                                            if disc else "none: the stated facts do not separate the "
                                                                         "values"))
    if hyp:
        blocks.append(("HYPOTHESES (prior -> posterior):", hyp, True))
    # TARGETS
    tg = []
    for t in cd.targets:
        xi = _target_at(res, t, at, S)
        x0 = _target_at(E.res0, t, at, S)
        isint = _is_int(cd, t)
        tg.append(f"  {t}: posterior {_dist_line(cd, t, xi, w, isint)}{lowtag} | prior predictive "
                  f"{_dist_line(cd, t, x0, np.where(E.res0['failed'], 0.0, 1.0), isint)}")
    if tg:
        when = "final step" if at in (None, "final") else f"step {at}"
        blocks.append((f"TARGETS (at {when}" + ("" if cd.static else f" t={S if at in (None, 'final') else at}")
                       + "; posterior | prior predictive):", tg, True))
    # EFFECTS
    ef = []
    for wr in whatif_results:
        ef += wr
    if ef:
        blocks.append(("EFFECTS (paired: same members and noise; weighted by the evidence):", ef, False))
    # SENSITIVITY
    sv = _sensitivity_block(cd, E, at)
    if sv:
        blocks.append(("SENSITIVITY (posterior first-order variance share S of each target; swing = target p50 at "
                       "the unknown's low vs high decile; narrowing = posterior IQR / prior IQR):", sv, False))
    # CLAIMS
    cl = _claims_block(cd, E, lowtag)
    if cl:
        blocks.append(("CLAIMS:", cl, False))
    # FACTS
    fa = []
    for i, o in enumerate(cd.facts):
        sat0 = float(np.mean(E.res0["obs"][i]["sat"]))
        sat1 = wmean(_fl(res["obs"][i]["sat"]), w) if W > 0 else float("nan")
        g = res["obs"][i]["g"]
        val = ""
        if o["kind"] != "pred" and W > 0:
            q = wquant(np.nan_to_num(g), w, [0.05, 0.5, 0.95])
            val = f"; model value p50 {_g(q[1])} [p05 {_g(q[0])}, p95 {_g(q[2])}]"
        fa.append(f"  {o['name']} [{_obs_text(o)}]: pass rate prior {_pct(sat0)} -> posterior "
                  f"{_pct(sat1) if math.isfinite(sat1) else 'undefined'}{val}")
    if fa:
        blocks.append(("FACTS (soft facts pass within 2 tol):", fa, False))
    # SERIES
    if not cd.static and S > 0:
        se = []
        ww = w if W > 0 else np.where(E.res0["failed"], 0.0, 1.0)
        rr = res if W > 0 else E.res0
        for t in cd.targets:
            if cd.type_of.get(t) == "cat":
                continue
            h = rr["tgt"][t]
            m = (np.asarray(h) * ww).sum(axis=1) / ww.sum() if ww.sum() > 0 else np.zeros(S + 1)
            row, shp = compact_series(list(m))
            se.append(f"  {t}: {row}")
            se.append(f"    shape: {shp}")
        if se:
            blocks.append(("SERIES (posterior mean per step):", se, False))
    return _assemble(head, blocks)


def _err_text(cd, res):
    fe = res["first_err"]
    if not fe:
        return ""
    n, line, t = fe
    return (f" (first: node {n}, line {line}, t={t}: non-finite value, e.g. division by zero, log of 0 or "
            f"overflow; those members get weight 0)")


def _fact_src(cd, name):
    for o in cd.facts:
        if o["name"] == name:
            return f"line {o['line']} '{_obs_text(o)}'" + (f" # {o['prov'][:60]}" if o.get("prov") else "")
    return name


def _sensitivity_block(cd, E, at):
    names = _sens_vars(cd)
    if not names or E.w.sum() <= 0:
        return []
    out = []
    w = E.w
    w0 = np.where(E.res0["failed"], 0.0, 1.0)
    for t in cd.targets:
        if cd.type_of.get(t) in ("cat", "bool") or t in names:
            continue
        y = _target_at(E.res, t, at, cd.steps)
        rows, tot = [], 0.0
        for n in names:
            x = E.U[n]
            disc = _is_discrete_var(cd, n)
            Si, groups = first_order(y, x, w, discrete=disc)
            if not math.isfinite(Si):
                continue
            tot += Si
            if disc:
                parts = ", ".join(f"{cd.fmt(n, lab) if n in cd.hyps or cd.type_of.get(n) == 'cat' else _g(lab)}: "
                                  f"{_g(med)}" for lab, sh, mj, med in groups[:8])
                extra = f"target p50 by value: {parts}"
            else:
                if len(groups) >= 2:
                    extra = f"swing {_g(groups[0][3])} -> {_g(groups[-1][3])} ({_signed(groups[-1][3] - groups[0][3])})"
                else:
                    extra = "swing 0"
                q1 = wquant(x, w, [0.25, 0.75])
                q0 = wquant(E.U0[n], w0, [0.25, 0.75])
                if q0[1] - q0[0] > 0:
                    extra += f"; narrowing {_g((q1[1] - q1[0]) / (q0[1] - q0[0]))}"
            rows.append((Si, f"    {n}{' (hypothesis)' if n in cd.hyps else ''}: S={Si:.2f} {_sens_label(Si)}; "
                             f"{extra}"))
        if not rows:
            continue
        rows.sort(key=lambda r: -r[0])
        var = float(np.var(y[w > 0])) if (w > 0).any() else 0.0
        if var <= 1e-15 * max(1.0, float(np.mean(y[w > 0]) ** 2) if (w > 0).any() else 1.0):
            out.append(f"  {t}: constant under the posterior (no unknown matters)")
            continue
        out.append(f"  {t}:")
        shown = [r for s_, r in rows if s_ >= 0.02][:6] or [r for _, r in rows[:1]]
        out += shown
        rest = [n for (s_, r) in rows if r not in shown for n in [r.split(":")[0].strip().split(" ")[0]]]
        if rest:
            out.append(f"    others negligible or minor (S < {min(s_ for s_, r in rows if r in shown):.2f}): "
                       + ", ".join(rest[:12]))
        if var > 1e-15:
            out.append(f"    sum of S = {tot:.2f}" + ((" (the rest is per-step randomness or" if cd.stoch else " (the rest is")
                                                     + " interactions between unknowns; read the CLAIMS boxes or use "
                                                     "`run` with `fix`)" if tot < 0.6 else ""))
    return out


def _signed(v):
    s = _g(v)
    return s if s.startswith("-") else "+" + s


def _claims_block(cd, E, lowtag):
    if not cd.claims:
        return []
    out = []
    w = E.w
    W = w.sum()
    nf = len(cd.facts)
    X = {n: E.U[n] for n in _sens_vars(cd)}
    disc = {n for n in X if _is_discrete_var(cd, n)}
    for j, o in enumerate(cd.claims):
        c = _fl(E.res["obs"][nf + j]["sat"])
        if W <= 0:
            out.append(f"  {o['name']} [{_obs_text(o)}]: P undefined (no member is compatible with the facts)")
            continue
        p = float((c * w).sum() / W)
        se = math.sqrt(max(p * (1 - p), 0) / max(E.ess, 1))
        out.append(f"  {o['name']} [{_obs_text(o)}]: P(claim | facts) = {p:.3f} +/- {se:.3f}{lowtag}")
        if p < 1e-3 or p > 1 - 1e-3:
            p = 0.0 if p < 1e-3 else 1.0
        if 0 < p < 1 and X:
            st = best_stump(c, X, w, disc)
            if st and st[5] < 0.01:
                out.append("    no single unknown separates holds from fails (information gain < 0.01 bits): the "
                           "outcome is driven by per-step randomness or by several unknowns together")
                st = None
            if st:
                n, op, v, pi, po, ig = st
                cond = f"{n} == {cd.fmt(n, v)}" if op == "==" else f"{n} > {_g(v)}"
                out.append(f"    best single condition: {cond} (P(claim) {pi:.2f} there vs {po:.2f} otherwise; "
                           f"information gain {ig:.2f} bits)")
            cont = [n for n in X if n not in disc]
            if cont:
                ranked = []
                for n in cont:
                    s1 = best_stump(c, {n: X[n]}, w, set())
                    ranked.append((s1[5] if s1 else 0, n))
                ranked.sort(reverse=True)
                pick = [n for _, n in ranked[:2]]
                b = prim_box(c, X, w, pick)
                if b and b["mass"] < 0.999 and b["density"] >= max(1.5 * p, p + 0.05):
                    out.append("    best box: " + " and ".join(f"{n} in [{_g(lo)}, {_g(hi)}]" for n, (lo, hi) in
                                                                b["box"].items())
                               + f": density {b['density']:.2f} (P(claim) inside), coverage {b['coverage']:.2f} "
                                 f"(share of claim-true weight inside), mass {b['mass']:.2f}")
        elif p in (0.0, 1.0):
            out.append(f"    {'holds in every' if p == 1 else 'fails in every'} compatible member")
        for hn, h in cd.hyps.items():
            x = E.U[hn]
            parts = []
            for v, code in zip(h["values"], h["codes"]):
                sel = x == code
                ws = w[sel].sum()
                parts.append(f"'{v}' {(c[sel] * w[sel]).sum() / ws:.2f}" if ws > 0 else f"'{v}' -")
            out.append(f"    by {hn}: " + ", ".join(parts))
    return out


def effect_lines(cd, E, name, changes, rr, targets, at, w=None, noeffect=False):
    """EFFECTS lines for one intervention result rr (paired with E.res)."""
    w = E.w if w is None else w
    W = w.sum()
    S = cd.steps
    out = [f"  {name} [{_change_text(changes)}]:" + (" [LOW-ESS (unreliable)]" if E.low else "")]
    if W <= 0:
        out.append("    undefined: no member is compatible with the facts")
        return out
    nf = int((rr["failed"] & ~E.res["failed"]).sum())
    if nf:
        out.append(f"    {nf} members failed under the intervention{_err_text(cd, rr)}")
    ww = np.where(rr["failed"], 0.0, w)
    for t in targets:
        a, b = _target_at(E.res, t, at, S), _target_at(rr, t, at, S)
        if cd.type_of.get(t) == "cat":
            ch = wmean(_fl(a != b), ww)
            out.append(f"    {t}: changes value in {ch:.2f} of the weight; under the intervention: "
                       f"{_dist_line(cd, t, b, ww, False)}")
            continue
        dlt = b - a
        if np.all(dlt[ww > 0] == 0):
            out.append(f"    {t}: no change in any member")
            continue
        q = wquant(dlt, ww, [0.05, 0.5, 0.95])
        line = (f"    {t}: diff mean {_signed(wmean(dlt, ww))}, p05 {_signed(q[0])} p50 {_signed(q[1])} p95 "
                f"{_signed(q[2])}; P(diff>0)={wmean(_fl(dlt > 0), ww):.2f} P(diff<0)={wmean(_fl(dlt < 0), ww):.2f}"
                f"; level under it: {_dist_line(cd, t, b, ww, _is_int(cd, t))}")
        names = _sens_vars(cd)
        if names:
            best = max(((first_order(dlt, E.U[n], ww, _is_discrete_var(cd, n))[0], n) for n in names),
                       key=lambda z: (z[0] if math.isfinite(z[0]) else -1))
            if math.isfinite(best[0]):
                line += f"; effect most sensitive to {best[1]} (S={best[0]:.2f})"
        out.append(line)
    nfct = len(cd.facts)
    for j, o in enumerate(cd.claims):
        p0 = wmean(_fl(E.res["obs"][nfct + j]["sat"]), ww)
        p1 = wmean(_fl(rr["obs"][nfct + j]["sat"]), ww)
        out.append(f"    claim {o['name']}: P {p0:.3f} -> {p1:.3f}")
    return out


def compact_series(vals, points=60):
    """'[t:v ...]' with run-length ranges, <= `points` sampled steps incl. extremes and the final step; + SHAPE."""
    n = len(vals)
    if n == 0:
        return "[]", "no values"
    big = max([abs(float(v)) for v in vals] + [1.0])
    vals = [0.0 if abs(float(v)) < 1e-9 * big else float(v) for v in vals]      # floating-point dust
    idx = list(range(n)) if n <= points else sorted(
        {round((n - 1) * i / (points - 1)) for i in range(points)} | {int(np.argmax(vals)), int(np.argmin(vals)),
                                                                       n - 1})
    toks, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and _g(vals[idx[j + 1]]) == _g(vals[idx[i]]):
            j += 1
        toks.append(f"{idx[i]}:{_g(vals[idx[i]])}" if i == j else f"{idx[i]}-{idx[j]}:{_g(vals[idx[i]])}")
        i = j + 1
    v = np.asarray(vals, float)
    vmax, vmin = float(v.max()), float(v.min())
    tmax, tmin = int(v.argmax()), int(v.argmin())
    if vmax - vmin <= 1e-9 * max(1.0, abs(vmax)):
        shp = f"constant {_g(v[0])}"
    else:
        eps = 0.01 * (vmax - vmin)
        dif = np.diff(v)
        signs = [1 if x > 0 else -1 for x in dif if abs(x) > eps]
        turns = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        kind = ("monotone rising" if signs and signs[0] > 0 else "monotone falling") if turns == 0 else \
            ("single peak" if signs[0] > 0 else "single trough") if turns == 1 else f"oscillating ({turns} turns)"
        shp = (f"start {_g(v[0])} -> end {_g(v[-1])}; max {_g(vmax)} at t={tmax}, min {_g(vmin)} at t={tmin}; "
               f"{kind}")
    return "[" + " ".join(toks) + "]", shp


def _assemble(head, blocks):
    """Bounded output: header, then blocks in priority order; required blocks are cut, optional ones dropped."""
    text = "\n".join(head)
    if len(text) > MAX_OUT - 1500:
        text = text[:MAX_OUT - 1500] + "\n...[header cut]"
    dropped = []
    for title, lines, required in blocks:
        blk = "\n" + title + "\n" + "\n".join(lines)
        if len(text) + len(blk) <= MAX_OUT - 150:
            text += blk
        elif required:
            room = MAX_OUT - 150 - len(text)
            if room > 200:
                text += blk[:room].rsplit("\n", 1)[0] + "\n  ...[cut at the output limit]"
            else:
                dropped.append(title.split(" ")[0].rstrip(":"))
        else:
            dropped.append(title.split(" ")[0].rstrip(":"))
    if dropped:
        text += "\n[output limit: dropped " + ", ".join(dropped) + "]"
    return text[:MAX_OUT]


# ============================================================================ lenient argument parsing
def lenient_args(raw):
    """Arguments string -> (dict or None, repaired: bool). json.loads, then strict=False, then a recovery of the
    `lines` / `source` payload."""
    if isinstance(raw, dict):
        return raw, False
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, False
    if not isinstance(raw, str):
        return None, False
    try:
        v = json.loads(raw)
        return (v, False) if isinstance(v, dict) else (None, False)
    except ValueError:
        pass
    try:
        v = json.loads(raw, strict=False)
        if isinstance(v, dict):
            return v, True
    except ValueError:
        pass
    m = re.search(r'"lines"\s*:\s*\[', raw)
    if m:
        body = raw[m.end():]
        j = body.rfind("]")
        if j >= 0:
            body = body[:j]
        body = body.strip()
        if body.startswith('"'):
            body = body[1:]
        if body.endswith('"'):
            body = body[:-1]
        items = re.split(r'"\s*,\s*(?:\r?\n\s*)?"', body)
        out = []
        for it in items:
            it = it.replace('\\"', "'").replace("\\n", "\n").replace("\\t", " ")
            out += it.split("\n")
        return {"lines": out}, True
    m = re.search(r'"source"\s*:\s*"', raw)
    if m:
        body = raw[m.end():]
        j = body.rfind('"')
        if j >= 0:
            body = body[:j]
        return {"source": body.replace("\\n", "\n").replace('\\"', "'").replace("\\t", " ")}, True
    return None, False


def _as_lines(args):
    """lines/source argument -> list of strings, or None."""
    ls = args.get("lines")
    if isinstance(ls, str):
        ls = ls.split("\n")
    if isinstance(ls, list) and ls:
        out = []
        for x in ls:
            out += str(x).replace("\r\n", "\n").split("\n")
        return out
    src = args.get("source")
    if isinstance(src, str) and src.strip():
        return src.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return None


def _tool(name, desc, props, req=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req or []}}}


# ============================================================================ the arm
class G2C3Arm(Arm):
    name = "G2C3"
    buildable = True
    executable = True
    guide_file = "g2c3_guide.md"
    mutating_tools = MUTATING
    custom_episode = True

    # ------------------------------------------------------------------ schemas
    def tools(self):
        run_item = {"type": ["object", "string"], "description": "a stored whatif name, or {name?, set?: {param or "
                    "rule: value}, do?: {node: value or ':= expr'}, fix?: {unknown: value}, given?: [fact lines], "
                    "targets?: [..], at?: step|final}"}
        return [
            _tool("submit", "Store the whole ece document, check every line at once, and if it is accepted run the "
                  "evidence-conditioned ensemble with every stored whatif and claim (full report).",
                  {"lines": {"type": "array", "items": {"type": "string"},
                             "description": "the document, one string per line; single quotes only"},
                   "source": {"type": "string", "description": "alternative: the document as one string"}}),
            _tool("edit", "Apply several edits to the stored document in one call (line numbers as last shown), then "
                  "re-check and re-run like submit.",
                  {"edits": {"type": "array", "items": {"type": "object"}, "description":
                             "each {line: n, set: [lines]} (empty list deletes), {after: n, insert: [lines]} or "
                             "{old: text, new: text}"}}, ["edits"]),
            _tool("run", f"Run up to {MAX_RUNS} scenarios on the stored document under the evidence-conditioned "
                  "family: set/do interventions (paired effects), given = extra hypothetical evidence, fix = pin an "
                  "unknown. A run with no changes, e.g. {name: 'baseline'}, returns the full report (HYPOTHESES, "
                  "TARGETS, SENSITIVITY, CLAIMS, coherence, ...).", {"runs": {"type": "array", "items": run_item},
                               "samples": {"type": "integer", "description": f"members (max {MAX_SAMPLES})"}},
                  ["runs"]),
            _tool("show", "Show the numbered document, the declarations, the facts and the dependency summary.", {}),
        ]

    def probe_tools(self):
        return [t for t in self.tools() if t["function"]["name"] in ("run", "show")]

    # ------------------------------------------------------------------ state
    def new_state(self):
        return ArmState(rep={"lines": []})

    def load(self, rep, given_gold=False):
        st = self.new_state()
        lines = None
        if isinstance(rep, dict):
            if isinstance(rep.get("lines"), list) and rep["lines"]:
                lines = [str(x) for x in rep["lines"]]
            elif isinstance(rep.get("source"), str):
                lines = rep["source"].replace("\r\n", "\n").split("\n")
        elif isinstance(rep, list):
            lines = [str(x) for x in rep]
        elif isinstance(rep, str):
            lines = rep.replace("\r\n", "\n").split("\n")
        st.rep = {"lines": lines or []}
        st.given_gold = given_gold
        d, cd = self._compile(st)
        if cd is not None:
            st.__dict__["_accepted"] = self._hash(st)
        return st

    @staticmethod
    def _lines(state):
        ls = state.rep.get("lines") if isinstance(state.rep, dict) else None
        return ls if isinstance(ls, list) else []

    def _hash(self, state):
        return hashlib.sha1("\n".join(self._lines(state)).encode("utf-8")).hexdigest()

    def _compile(self, state):
        h = self._hash(state)
        cache = state.__dict__.setdefault("_cc", {})
        if h not in cache:
            d = analyse(self._lines(state))
            cd = None
            if not any(p["fatal"] for p in d["problems"]):
                try:
                    cd = Compiled(d)
                except Exception as e:  # pragma: no cover - defensive
                    d["problems"].append(_prob(None, f"internal compile error: {type(e).__name__}: {e}", ""))
            cache.clear()
            cache[h] = (d, cd)
        return cache[h]

    def export(self, state):
        lines = self._lines(state)
        d = analyse(lines)
        src = "\n".join(lines)
        return {"source": src, "lines": list(lines),
                "params": {n: r["value"] for n, r in d["params"].items()},
                "rules": {n: r["value"] for n, r in d["rules"].items()},
                "unknowns": {n: _dist_text(u["dist"]) for n, u in d["unknowns"].items()},
                "hypotheses": {n: dict(zip(h["values"], h["weights"])) for n, h in d["hyps"].items()},
                "facts": [f"{o['name']}: {_obs_text(o)}" for o in d["facts"]],
                "claims": [f"{o['name']}: {_obs_text(o)}" for o in d["claims"]],
                "targets": [t for t, _ in d["targets"]],
                "whatifs": [f"{w['name']}: {w['src']}" for w in d["whatifs"]],
                "time": {k: v for k, v in (d["time"] or {}).items() if k in ("steps", "dt", "unit", "static")},
                "samples": d["samples"]}

    def render(self, state):
        lines = self._lines(state)
        if not lines:
            return "(no document yet)"
        d = analyse(lines)
        return ("\n".join(f"{i:3d}| {l}" for i, l in enumerate(lines, 1)) + "\n\nDECLARATIONS\n"
                + "\n".join(_decl_table(d)))[:60000]

    def validity(self, state):
        lines = self._lines(state)
        d, cd = self._compile(state)
        holes = [_ptext(p) for p in d["problems"]]
        usable = bool(lines) and cd is not None and bool(cd.targets)
        executable, errors = False, []
        if usable:
            if state.__dict__.get("_ran", {}).get(self._hash(state)):
                executable = True
            else:
                try:
                    U = cd.draw_prior(20, SEED)
                    saved = cd.steps
                    cd.steps = min(2, cd.steps)
                    cd.coef = np.ones(cd.steps + 1)
                    try:
                        r = cd.simulate(U, 20, SEED, deadline=time.monotonic() + WALL_S)
                    finally:
                        cd.steps = saved
                        cd.coef = np.random.default_rng(12345).random(saved + 1) + 0.5
                    executable = not r["failed"].all()
                    if r["failed"].any():
                        errors.append("smoke run: run-time errors" + _err_text(cd, r))
                except Exception as e:
                    errors.append(f"smoke run failed: {type(e).__name__}: {e}")
        return {"usable": usable, "executable": executable, "holes": holes, "errors": errors}

    # ------------------------------------------------------------------ dispatch
    def handle(self, state, name, args):
        try:
            if isinstance(args, str):
                args, _rep = lenient_args(args)
            args = args if args is not None else {}
            if not isinstance(args, dict):
                return f"{ERROR_PREFIX}: arguments must be an object (hint: {ARG_HINT})"
            fn = getattr(self, "_t_" + str(name), None)
            if fn is None:
                return (f"{ERROR_PREFIX}: unknown tool '{name}' (hint: available: "
                        + ", ".join(t["function"]["name"] for t in self.tools()) + ")")
            if state.given_gold and name in self.mutating_tools:
                return f"{ERROR_PREFIX}: the document is read-only in this mode (hint: use run and show)"
            out = fn(state, args)
        except Exception as e:
            out = f"{ERROR_PREFIX}: internal {type(e).__name__}: {str(e)[:300]} (hint: check the arguments)"
        state.log.append((name, {k: (v if k not in ("lines", "source") else f"<{len(str(v))} chars>")
                                 for k, v in args.items()} if isinstance(args, dict) else args, out[:200]))
        return out

    def _t_show(self, state, args):
        lines = self._lines(state)
        if not lines:
            return "(no document yet; call submit with lines)"
        d, cd = self._compile(state)
        out = ["DOCUMENT"] + [f"{i:3d}| {l}" for i, l in enumerate(lines, 1)]
        out += ["", "DECLARATIONS"] + _decl_table(d)
        out += ["", "DEPENDENCIES (same-step order; lags in brackets)"]
        for n in d.get("order", []):
            nd = d["nodes"][n]
            refs = sorted(_expr.refs(nd["ast"])) if nd["ast"] else []
            out.append(f"  {n} <- " + (", ".join(f"{x}[t-{l}]" if l else x for x, l in refs) or "(constant)"))
        if d["problems"]:
            out += ["", f"PROBLEMS ({len(d['problems'])})"] + ["  " + _ptext(p) for p in d["problems"]]
        txt = "\n".join(out)
        return txt if len(txt) <= MAX_OUT else txt[:MAX_OUT - 60] + "\n...[cut at the output limit]"

    def _t_submit(self, state, args):
        lines = _as_lines(args)
        if lines is None:
            return f"{ERROR_PREFIX}: no document found (hint: {ARG_HINT})"
        return self._store_and_run(state, lines, "submitted")

    def _t_edit(self, state, args):
        lines = list(self._lines(state))
        if not lines:
            return f"{ERROR_PREFIX}: no document yet (hint: call submit with lines first)"
        edits = args.get("edits")
        if isinstance(edits, str):
            try:
                edits = json.loads(edits)
            except ValueError:
                return f"{ERROR_PREFIX}: edits must be a list of edit objects"
        if isinstance(edits, dict):
            edits = [edits]
        if not isinstance(edits, list) or not edits:
            return f"{ERROR_PREFIX}: edits must be a non-empty list of {{line, set}} / {{after, insert}} / {{old, new}}"
        line_ed, text_ed, errs = [], [], []
        for k, e in enumerate(edits):
            if not isinstance(e, dict):
                errs.append(f"edit {k + 1} is not an object")
                continue
            try:
                if "old" in e:
                    text_ed.append((str(e["old"]), str(e.get("new", ""))))
                elif "line" in e:
                    v = e.get("set", [])
                    v = [v] if isinstance(v, str) else list(v or [])
                    line_ed.append((int(e["line"]), 1, v))
                elif "after" in e:
                    v = e.get("insert", [])
                    v = [v] if isinstance(v, str) else list(v or [])
                    line_ed.append((int(e["after"]), 2, v))
                else:
                    errs.append(f"edit {k + 1}: needs line+set, after+insert or old+new")
            except (TypeError, ValueError):
                errs.append(f"edit {k + 1}: line numbers must be integers")
        for n, kind, _v in line_ed:
            if not (0 <= n <= len(lines)) or (kind == 1 and n == 0):
                errs.append(f"line {n} does not exist (the document has {len(lines)} lines)")
        if errs:
            return f"{ERROR_PREFIX}: no edit applied: " + "; ".join(errs)
        for n, kind, v in sorted(line_ed, key=lambda x: (x[0], x[1]), reverse=True):
            ins = []
            for x in v:
                ins += str(x).split("\n")
            if kind == 1:
                lines[n - 1:n] = ins
            else:
                lines[n:n] = ins
        text = "\n".join(lines)
        for old, new in text_ed:
            c = text.count(old)
            if c != 1:
                close = difflib.get_close_matches(old.strip(), [l.strip() for l in lines], n=2, cutoff=0.5)
                return (f"{ERROR_PREFIX}: no edit applied: `old` text occurs {c} times"
                        + (f" (closest lines: {' | '.join(close)})" if close and c == 0 else "")
                        + " (hint: use show to copy the exact text, or include more of the line)")
            text = text.replace(old, new, 1)
        return self._store_and_run(state, text.split("\n"), f"edited ({len(edits)} edits)")

    # ------------------------------------------------------------------ submit / edit pipeline
    def _store_and_run(self, state, lines, verb):
        state.rep = {"lines": list(lines)}
        d, cd = self._compile(state)
        probs = list(d["problems"])
        fatal = [p for p in probs if p["fatal"]]
        if fatal or cd is None:
            return self._reject(state, probs, verb, len(lines), can_run=False)
        deadline = time.monotonic() + WALL_S
        try:
            out = self._full_run(state, cd, d, deadline, policy=[p for p in probs if not p["fatal"]], verb=verb,
                                 nlines=len(lines))
        except _Timeout:
            return (f"{ERROR_PREFIX}: the run exceeded the {WALL_S:g} s time limit (hint: fewer steps, fewer samples "
                    f"or fewer whatifs)")
        return out

    def _sig(self, probs):
        return frozenset(re.sub(r"\d+", "#", p["msg"]) for p in probs)

    def _reject(self, state, probs, verb, nlines, can_run):
        rej = state.__dict__.setdefault("_rej", {"sig": None, "count": 0})
        sig = self._sig(probs)
        rej["count"] = rej["count"] + 1 if rej["sig"] == sig else 1
        rej["sig"] = sig
        state.__dict__.pop("_accepted", None)
        head = [f"{ERROR_PREFIX}: submission rejected ({len(probs)} problem{'s' if len(probs) != 1 else ''}); the "
                f"document ({nlines} lines, {verb}) is stored as a draft but not run. Fix them all in one `edit`."]
        for p in probs:
            head.append("  " + _ptext(p))
        if can_run and rej["count"] >= 2:
            head.append("note: the same problems were rejected twice; a third submission with them runs anyway, "
                        "listed as UNRESOLVED")
        txt = "\n".join(head)
        return txt if len(txt) <= MAX_OUT else txt[:MAX_OUT - 40] + "\n...[more problems cut]"

    def _plan_n(self, cd, N, runs):
        per = (cd.steps + 1) * max(1, len(cd.nodes) + len(cd.facts) + len(cd.claims))
        cap = int(COMPUTE_CAP // (per * max(1, runs)))
        if cap < N:
            n2 = max(MIN_SAMPLES, cap)
            return n2, (f"compute cap: members reduced from {N} to {n2} (members x steps x nodes <= "
                        f"{COMPUTE_CAP:.0e} per call)")
        return N, None

    def _ensemble(self, state, cd, N, deadline):
        key = (self._hash(state), N)
        ec = state.__dict__.setdefault("_ens", {})
        if key not in ec:
            ec.clear()
            ec[key] = build_ensemble(cd, N, deadline)
        return ec[key]

    def _full_run(self, state, cd, d, deadline, policy, verb, nlines):
        runs = 2 + len(cd.whatifs) + (1 if cd.hyps else 0)
        N, cnote = self._plan_n(cd, d["samples"], runs)
        E = self._ensemble(state, cd, N, deadline)
        post = []
        # no-op hypothesis values (ensemble form of the no-op what-if rule)
        for hn, h in cd.hyps.items():
            if hn in d["noeffect"]:
                continue
            M = min(N, 200)
            sub = {n: v[:M] for n, v in E.U.items()}
            sigs = []
            for code in h["codes"]:
                U2 = dict(sub)
                U2[hn] = np.full(M, code)
                r = cd.simulate(U2, M, E.seed, deadline=deadline)
                sigs.append(_sig_arrays(cd, r))
            for i in range(len(sigs)):
                for j in range(i):
                    if all(np.array_equal(a, b) for a, b in zip(sigs[i], sigs[j])):
                        post.append(_prob(h["line"], f"hypothesis {hn}: '{h['values'][i]}' produces the same "
                                          f"trajectories as '{h['values'][j]}'; the equations do not branch on it "
                                          f"where it matters", f"make the mechanism depend on {hn} == "
                                          f"'{h['values'][i]}' (or mark the line '# no effect expected: <reason>')",
                                          fatal=False))
                        break
        wres = self._whatif_effects(cd, E, deadline, post)
        allp = policy + post
        unresolved = None
        if allp:
            rej = state.__dict__.setdefault("_rej", {"sig": None, "count": 0})
            if rej["sig"] == self._sig(allp) and rej["count"] >= 2:
                unresolved = ["UNRESOLVED (rejected twice; run anyway at the escape valve):"] + \
                             ["  " + _ptext(p) for p in allp]
            else:
                return self._reject(state, allp, verb, nlines, can_run=True)
        state.__dict__["_rej"] = {"sig": None, "count": 0}
        state.__dict__["_accepted"] = self._hash(state)
        state.__dict__.setdefault("_ran", {})[self._hash(state)] = not E.res["failed"].all()
        return self._report(cd, E, deadline, wres, unresolved, cnote,
                            pre=[f"{verb.upper()}: {nlines} lines; {len(cd.whatifs)} stored whatifs run"])

    @staticmethod
    def _whatif_effects(cd, E, deadline, post=None):
        """EFFECTS lines of every stored whatif; no-op whatifs become problems in `post` (submit) or are noted."""
        wres = []
        for wi in cd.whatifs:
            try:
                rr = run_changes(cd, E, wi["changes"], deadline)
            except ValueError as e:
                if post is not None:
                    post.append(_prob(wi["line"], f"whatif {wi['name']}: {e}", "change the intervention"))
                continue
            if not wi["noeffect"] and _same(cd, E.res, rr):
                if post is not None:
                    post.append(_prob(wi["line"], f"no-op what-if {wi['name']}: {_change_text(wi['changes'])} "
                                      f"changes nothing; the equations do not read it where it matters",
                                      "make the mechanism read it, or mark the line '# no effect expected: <reason>'",
                                      fatal=False))
                    continue
            wres.append(effect_lines(cd, E, wi["name"], wi["changes"], rr, wi["targets"] or cd.targets, wi["at"]))
        return wres

    def _report(self, cd, E, deadline, wres, unresolved=None, cnote=None, pre=()):
        static_coh, incoh = None, None
        if cd.static and cd.facts:
            static_coh = static_coherence(cd)
            if static_coh["status"] == "fallback":
                incoh = dynamic_incoherence(cd, E, deadline)
        if E.low and cd.facts and incoh is None and (static_coh is None or static_coh["status"] == "fallback"):
            incoh = dynamic_incoherence(cd, E, deadline)
        return build_report(cd, E, problems_note=list(pre) + (unresolved or []), whatif_results=wres,
                            incoherent=incoh, static_coh=static_coh, compute_note=cnote)

    # ------------------------------------------------------------------ run
    def _t_run(self, state, args):
        if not self._lines(state):
            return f"{ERROR_PREFIX}: no document yet (hint: call submit with lines)"
        d, cd = self._compile(state)
        if cd is None:
            return (f"{ERROR_PREFIX}: the stored document has unresolved problems and cannot run: "
                    + "; ".join(_ptext(p) for p in d["problems"] if p["fatal"])[:2000])
        if state.__dict__.get("_accepted") != self._hash(state):
            return (f"{ERROR_PREFIX}: the stored document was rejected (draft); fix its problems with edit "
                    f"(hint: show lists them)")
        runs = args.get("runs")
        if isinstance(runs, str):
            try:
                runs = json.loads(runs)
            except ValueError:
                runs = [runs]
        if isinstance(runs, dict):
            runs = [runs]
        if runs is None and any(k in args for k in ("set", "do", "fix", "given", "name")):
            runs = [{k: v for k, v in args.items() if k != "samples"}]
        if not isinstance(runs, list) or not runs:
            return f"{ERROR_PREFIX}: runs must be a non-empty list (stored whatif names or run objects)"
        notes = []
        if len(runs) > MAX_RUNS:
            notes.append(f"note: at most {MAX_RUNS} runs per call; the last {len(runs) - MAX_RUNS} were not run")
            runs = runs[:MAX_RUNS]
        N = d["samples"]
        cnote = None
        cached = [k[1] for k in state.__dict__.get("_ens", {}) if k[0] == self._hash(state)]
        if args.get("samples") is not None:
            try:
                N = max(MIN_SAMPLES, min(MAX_SAMPLES, int(args["samples"])))
            except (TypeError, ValueError):
                return f"{ERROR_PREFIX}: samples must be an integer"
            N, cnote = self._plan_n(cd, N, 2 + len(runs))
        elif cached:
            N = cached[0]                 # the same conditioned family as the last report
        else:
            N, cnote = self._plan_n(cd, N, 2 + len(runs))
        deadline = time.monotonic() + WALL_S
        stored = {w["name"] for w in cd.whatifs}

        def is_base(r):
            if isinstance(r, str):
                return r.strip() not in stored and r.strip().lower() in ("baseline", "base", "report", "full")
            return isinstance(r, dict) and not any(k in r for k in ("set", "params", "rules", "overrides", "do",
                                                                     "fix", "given"))
        try:
            E = self._ensemble(state, cd, N, deadline)
            out = list(notes) + ([cnote] if cnote else [])
            if any(is_base(r) for r in runs):
                # a run without changes: the full report of the evidence-conditioned family (read-only mode too)
                out.append(self._report(cd, E, deadline, self._whatif_effects(cd, E, deadline),
                                        pre=["BASELINE REPORT (the evidence-conditioned family, as after submit):"]))
                runs = [r for r in runs if not is_base(r) or (isinstance(r, dict) and (r.get("targets") or
                                                                                         r.get("at") is not None))]
            else:
                out.append(f"BASELINE (evidence-conditioned): N={E.N}, ESS {E.ess:.1f}"
                           + (" [LOW-ESS (unreliable)]" if E.low else ""))
            errors = 0
            for k, r in enumerate(runs):
                try:
                    blk, err = self._one_run(cd, d, E, r, k, deadline)
                except ValueError as e:
                    blk, err = [f"{ERROR_PREFIX}: run {k + 1}: {e}"], 1
                errors += err
                out += blk
        except _Timeout:
            return f"{ERROR_PREFIX}: the run exceeded the {WALL_S:g} s time limit (hint: fewer runs or samples)"
        txt = "\n".join(out)
        if len(txt) > MAX_OUT:
            txt = txt[:MAX_OUT - 60] + "\n...[cut at the output limit; request fewer runs]"
        if runs and errors == len(runs):
            txt = f"{ERROR_PREFIX}: no run could be executed\n" + txt
        return txt

    def _one_run(self, cd, d, E, r, k, deadline):
        if isinstance(r, str):
            wi = next((w for w in cd.whatifs if w["name"] == r.strip()), None)
            if wi is None:
                return [f"{ERROR_PREFIX}: run {k + 1}: no stored whatif named {r!r}"
                        f"{_dym(r, [w['name'] for w in cd.whatifs])}; stored: {[w['name'] for w in cd.whatifs]}"], 1
            r = {"name": wi["name"], "_changes": list(wi["changes"]), "targets": wi["targets"], "at": wi["at"]}
        if not isinstance(r, dict):
            return [f"{ERROR_PREFIX}: run {k + 1} must be an object or a stored whatif name"], 1
        name = str(r.get("name") or f"run{k + 1}")[:40]
        P = []
        if "_changes" in r:
            changes = r["_changes"]           # a stored whatif: already resolved by the static check
        else:
            try:
                changes = _changes_from_run(d, r)
            except (ValueError, _expr.ExprError) as e:
                return [f"{ERROR_PREFIX}: run '{name}': {e}"], 1
            _resolve_changes(d, changes, None, f"run '{name}'", P)
        given = r.get("given") or []
        if isinstance(given, str):
            given = [given]
        gobs = []
        for gl in given:
            g = str(gl).strip()
            g = re.sub(r"^fact\s+", "", g)
            m = re.match(r"([A-Za-z_]\w*)\s*:\s*(.+)", g, re.S)
            gname, body = (m.group(1), m.group(2)) if m and not re.match(r"[A-Za-z_]\w*\s*:=", g) else \
                (f"given{len(gobs) + 1}", g)
            o = _parse_obs(body, None, cd.static, cd.steps, P, "given")
            if o is None:
                continue
            bad = [x for x, _l in _expr.refs(o["ast"]) if x not in d["decl"]]
            if bad:
                P.append(_prob(None, f"given {gname}: undefined name {bad[0]!r}{_dym(bad[0], d['decl'])}", ""))
                continue
            o.update(name=gname, line=None, prov="", static=cd.static)
            o["fn"] = compile_expr(o["ast"], cd.codes, "given:" + gname)
            gobs.append(o)
        targets = r.get("targets") or cd.targets
        if isinstance(targets, str):
            targets = [x.strip() for x in targets.split(",") if x.strip()]
        badt = [t for t in targets if t not in d["decl"]]
        if badt:
            P.append(_prob(None, f"run '{name}': unknown target {badt[0]!r}{_dym(badt[0], d['decl'])}", ""))
        at = r.get("at")
        if at not in (None, "final"):
            try:
                at = int(at)
            except (TypeError, ValueError):
                P.append(_prob(None, f"run '{name}': at must be a step number or final", ""))
        if P:
            return [f"{ERROR_PREFIX}: run '{name}': " + "; ".join(p["msg"] for p in P)], 1
        for t in targets:
            if t not in E.res["tgt"]:
                pass
        out = [f"RUN {name}" + (f" [{_change_text(changes)}]" if changes else "")
               + (f" given: " + "; ".join(_obs_text(o) for o in gobs) if gobs else "") + ":"]
        w = E.w
        if gobs:
            rb = cd.simulate(E.U, E.N, E.seed, deadline=deadline, extra_obs=gobs)
            w = E.w * reduce(lambda a, b: a * b, [_fact_weight(o, rb["obs"][len(cd.facts) + len(cd.claims) + i]["lik"])
                                                  for i, o in enumerate(gobs)])
            e2 = ess(w)
            out.append(f"  given evidence: ESS {E.ess:.1f} -> {e2:.1f}" + (" [LOW-ESS (unreliable)]" if e2 < MIN_ESS
                                                                            else ""))
            for hn, h in cd.hyps.items():
                x = E.U[hn]
                W0, W1 = E.w.sum(), w.sum()
                out.append(f"  {hn}: " + " | ".join(
                    f"'{v}' {(E.w[x == c].sum() / W0 if W0 > 0 else 0):.2f} -> {(w[x == c].sum() / W1 if W1 > 0 else 0):.2f}"
                    for v, c in zip(h["values"], h["codes"])))
        tg = [t for t in targets if t in d["decl"]]
        fixes = [c for c in changes if c[0] == "fix"]
        others = [c for c in changes if c[0] != "fix"]
        if fixes:
            # pinning an unknown: the family is re-conditioned on the facts with the unknown known
            rf = run_changes(cd, E, fixes, deadline)
            gl = np.divide(w, E.w, out=np.zeros_like(w), where=E.w > 0) if gobs else 1.0
            wf = E.corr * _weights(cd, rf) * gl
            ef = ess(wf)
            out.append(f"  with {_change_text(fixes)} (conditioned on the facts again): ESS {ef:.1f}"
                       + (" [LOW-ESS (unreliable)]" if ef < MIN_ESS else ""))
            res_tg = rf
            for t in tg:
                xb = _target_at(E.res, t, at, cd.steps)
                xf = _target_at(rf, t, at, cd.steps)
                out.append(f"    {t}: {_dist_line(cd, t, xf, wf, _is_int(cd, t))} (baseline posterior: "
                           f"{_dist_line(cd, t, xb, w, _is_int(cd, t))})")
            if others:
                rr = run_changes(cd, E, fixes + others, deadline)
                E2 = copy.copy(E)
                E2.res, E2.w, E2.low = rf, wf, ef < MIN_ESS
                out += effect_lines(cd, E2, name, others, rr, tg, at, w=wf)[1:]
            return out, 0
        if others:
            rr = run_changes(cd, E, others, deadline)
            out += effect_lines(cd, E, name, others, rr, tg, at, w=w)[1:]
            return out, 0
        for t in tg:
            x = _target_at(E.res, t, at, cd.steps)
            out.append(f"    {t}: {_dist_line(cd, t, x, w, _is_int(cd, t))}")
        nf = len(cd.facts)
        for j, o in enumerate(cd.claims):
            out.append(f"    claim {o['name']}: P {wmean(_fl(E.res['obs'][nf + j]['sat']), w):.3f}")
        return out, 0

    # ------------------------------------------------------------------ episode (argument channel)
    def run_episode(self, ep, state, messages):
        """G1C1-style loop: system-prompt parity, lenient argument pre-parse, the (untaught) fenced ```ece fallback
        while no document has been accepted, and a budget status line on every executor result."""
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
                block = None if gold or state.__dict__.get("_accepted") else fenced_block(content)
                if block is None or not block.strip() or not tools:
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
                raw = fn.get("arguments", "{}")
                note = ""
                args, repaired = lenient_args(raw)
                if args is None:
                    name = fn.get("name", "")
                    res = (f"{ERROR_PREFIX}: the arguments of {name} are not a valid JSON object (hint: {ARG_HINT})")
                    targs = raw
                else:
                    tc2 = dict(tc)
                    tc2["function"] = dict(fn, arguments=json.dumps(args))
                    name, targs, res = execute_tool_call(self, state, tc2, offered, n_tools, MAX_TOOL_CALLS)
                    if repaired:
                        note = "note: arguments repaired (lenient parse)\n"
                        if isinstance(targs, dict):
                            targs = dict(targs, _repaired=True)
                n_tools += 1
                is_err = res.startswith(ERROR_PREFIX)
                res = note + res + budget()
                if isinstance(targs, dict):
                    targs = {k: (v if k not in ("lines", "source") else f"<{len(str(v))} chars>")
                             for k, v in targs.items()}
                trace.append({"turn": len(ep.calls), "tool": name, "args": targs, "result": res[:3000],
                              "error": is_err})
                messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call{len(trace)}", "content": res})
        ep.forced_final = True
        messages.append({"role": "user", "content": FORCE_FINAL_MSG})
        m = ep.call(messages, tools=None, label="forced_final", final=True)
        messages.append({"role": "assistant", "content": m.get("content") or ""})
        return m.get("content") or "", messages, trace


def _sig_arrays(cd, r):
    out = [cd.coef @ np.asarray(r["tgt"][t]) if t in cd.nodes else np.asarray(r["tgt"][t][0]) for t in cd.targets]
    out += [o["cs"] for o in r["obs"]]
    return out


def _same(cd, a, b):
    """True if two runs agree on every node, target and observation in every member and step (checksums)."""
    for n in cd.nodes:
        if not np.array_equal(a["cs"][n], b["cs"][n]):
            return False
    for x, y in zip(_sig_arrays(cd, a), _sig_arrays(cd, b)):
        if not np.array_equal(x, y):
            return False
    return True


def _ptext(p):
    return (f"line {p['line']}: " if p.get("line") else "") + p["msg"] + (f"  -> hint: {p['hint']}" if p.get("hint")
                                                                          else "")


def _decl_table(d):
    out = []
    for n, r in d["params"].items():
        out.append(f"  param {n} = {_g(r['value'])}  [stated]" + (f"  # {r['prov']}" if r["prov"] else ""))
    for n, r in d["rules"].items():
        out.append(f"  rule {n} = {_vt(r['value'])}  [stated]" + (f"  # {r['prov']}" if r["prov"] else ""))
    for n, u in d["unknowns"].items():
        out.append(f"  unknown {n} ~ {_dist_text(u['dist'])}  [open: range]" + (f"  # {u['prov']}" if u["prov"]
                                                                                   else ""))
    for n, h in d["hyps"].items():
        out.append(f"  hypothesis {n}: " + " | ".join(f"'{v}' {_g(w)}" for v, w in zip(h["values"], h["weights"]))
                   + (f"  # {h['prov']}" if h["prov"] else ""))
    for o in d["facts"]:
        out.append(f"  fact {o['name']} ({o['kind']}): {_obs_text(o)}")
    for o in d["claims"]:
        out.append(f"  claim {o['name']}: {_obs_text(o)}")
    for w in d["whatifs"]:
        out.append(f"  whatif {w['name']}: {_change_text(w['changes'])}")
    tm = d.get("time") or {}
    out.append(f"  time: {'static' if tm.get('static') else str(tm.get('steps')) + ' steps of ' + _g(tm.get('dt')) + ' ' + str(tm.get('unit', ''))}"
               f"; samples {d['samples']}; targets: {', '.join(t for t, _ in d['targets'])}")
    return out


G2C3 = G2C3Arm
