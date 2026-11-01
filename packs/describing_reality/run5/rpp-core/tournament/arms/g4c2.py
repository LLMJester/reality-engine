"""G4C2: specification-checked probabilistic program (runs/discovery/gen4/specs/G4C2.md).

Representation: the pair (M, Phi). M is G3C1's program (PARAMS, RULES, HORIZON, class Model), unchanged in syntax and
meaning. Phi is an optional declaration `SPEC = {name: (kind, formula, quote)}` of at most 16 clauses: the description's
rules, schedules, limits and reported facts (`rule`, `fact`) and the properties the question asks about (`ask`), each a
bounded metric temporal logic formula over the run's trace (spec section 1.3). Every run result carries the SPEC block:
SPEC CONFORMANCE of the baseline (rule/fact clauses) in the header, and per run the ask results, the rules a what-if
breaks, and a facts x runs table (spec section 3.4). Without SPEC the arm behaves exactly like G3C1 plus one note line.

Implementation choices (stated as the spec asks):
- g3c1.py, _expr.py and harness/sandbox_runner.py are not edited. G3C1's module functions and G3C1Arm's methods are
  re-bound (same code objects) to a copy of g3c1's namespace in which DECL_NAMES also holds 'SPEC'. So SPEC is a
  whole-section declaration for edit_model / show_model / problem tags exactly like PARAMS, and G3C1's own code runs
  for everything else. In that namespace `unread_names` also excludes the SPEC literal, so SPEC strings never count as
  reads of a PARAMS/RULES entry, and a SPEC entry can never raise the unread-declaration warning.
- Dotted observation keys (`a.b`, `a.0`) need no alias: _expr's tokeniser already reads `.` inside a name.
- Traces: the observation keys a run's clauses read are added to that run's `evkeys` (one internal pseudo-event that
  never reaches the output), so G3C1's unchanged driver returns their per-step series. Names that are PARAMS/RULES
  are resolved in process from the run's scenario (an observed key of the same name takes precedence), as is `t`.
- Seeds that do not reach the final step give an undefined verdict (counted and reported, never dropped).
- Facts are judged (conformance) on the baseline only; with two or more runs their per-run share is also shown in the
  facts x runs table (spec 1.9), which is information, not a violation.
- Truncation (spec 1.4): a clause is "truncated" on a seed when its verdict at position 0 would become unknown if the
  steps after the horizon were unknown (a three-valued re-evaluation in which a bounded future window that passes H,
  or `next` at H, contributes "unknown"). The reported verdict is still the finite-trace one.
- Undefined atoms follow Kleene's three-valued logic (False and undefined = False; True or undefined = True).
"""
import ast
import contextvars
import difflib
import itertools
import math
import time
import types
from collections import deque
from functools import lru_cache

from tournament.arms import _expr
from tournament.arms import g3c1 as _g3
from tournament.arms.a4_prob import _g, _quant
from tournament.arms.g3c1 import DEADLINE_S, MAX_OUT, G3C1Arm, _s3, _signed

# Runtime execution dependency (D-060/D-061): models run in G3C1's driver inside the A3 sandbox subprocess.
RUNTIME_DEPS = ["harness/sandbox_runner.py"]
DECL_NAMES = _g3.DECL_NAMES + ("SPEC",)
KINDS = ("rule", "fact", "ask")
MAX_CLAUSES = 16
MAX_TEMPORAL = 12            # temporal operators per clause
MAX_SPEC_KEYS = 24           # distinct observation keys read by SPEC clauses per run
MONITOR_CAP_S = 5.0          # in-process monitor time per tool call
MAX_CONF = 700               # characters of SPEC CONFORMANCE
MAX_SPEC_NOTES = 400         # characters of SPEC NOTES
MAX_CEX_KEYS = 6
NO_SPEC_NOTE = "SPEC: none (no SPEC: nothing is checked against the description)"
SPEC_TITLE = "SPEC (per run)"
MARGIN_NA = "margin n/a (several quantities)"

# ---------------------------------------------------------------------------------------------------
# G3C1's code, re-bound to a namespace in which SPEC is a declaration (g3c1.py itself is unchanged)


def _rebind(f, g):
    nf = types.FunctionType(f.__code__, g, f.__name__, f.__defaults__, f.__closure__)
    nf.__kwdefaults__ = f.__kwdefaults__
    nf.__doc__ = f.__doc__
    nf.__qualname__ = f.__qualname__
    return nf


_G = dict(vars(_g3))
for _n, _f in list(vars(_g3).items()):
    if isinstance(_f, types.FunctionType) and _f.__module__ == _g3.__name__:
        _G[_n] = _rebind(_f, _G)
_G["DECL_NAMES"] = DECL_NAMES


def _spec_value_nodes(tree):
    """Value nodes of the top-level `SPEC = ...` assignments (the last one wins, as in Python)."""
    out = []
    for node in getattr(tree, "body", []) or []:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SPEC" for t in node.targets):
            out.append(node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name) \
                and node.target.id == "SPEC":
            out.append(node.value)
    return out


_unread_g3 = _G["unread_names"]


def _unread_names(tree, decl_values, names):
    """G3C1's unread check with the SPEC literal excluded like PARAMS/RULES (spec 3.2: SPEC keys are not state)."""
    return _unread_g3(tree, list(decl_values) + _spec_value_nodes(tree), names)


def _text_tags(lines):
    """G3C1's indentation-only section tags (used only when the program cannot be parsed), with SPEC."""
    import re
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
                elif re.match(r"(PARAMS|RULES|HORIZON|SPEC)\b", word):
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


_SPEC_GROUPS = contextvars.ContextVar("g4c2_spec_groups", default=None)
_assemble_g3 = _g3._assemble


def _assemble(header, final_block, effect_block, optional, is_error):
    """G3C1's bounded assembly with the SPEC per-run block first among the optional blocks (spec 3.4: SERIES is cut
    first, then SHAPE, EVENTS and SPEC per run). The header (with SPEC CONFORMANCE) is never cut."""
    groups = _SPEC_GROUPS.get()
    if groups:
        optional = [(SPEC_TITLE, groups)] + list(optional)
    return _assemble_g3(header, final_block, effect_block, optional, is_error)


_G["unread_names"] = _unread_names
_G["_text_tags"] = _text_tags
_G["_assemble"] = _assemble

analyse = _G["analyse"]
static_report = _G["static_report"]
apply_sections = _G["apply_sections"]
locate_sections = _G["locate_sections"]
section_names = _G["section_names"]
line_tags = _G["line_tags"]


def _method(name):
    f = G3C1Arm.__dict__[name]
    if isinstance(f, staticmethod):
        return staticmethod(_rebind(f.__func__, _G))
    return _rebind(f, _G)


_BASE = {n: _method(n) for n, f in G3C1Arm.__dict__.items() if isinstance(f, (types.FunctionType, staticmethod))}


# ---------------------------------------------------------------------------------------------------
# Formula language (spec 1.3): a recursive-descent parser; every maximal non-temporal sub-formula is an atom whose
# text is handed to _expr.parse (G3C1 event syntax).

T, U, F = 2, 1, 0            # Kleene truth values: true, undefined/unknown, false (and = min, or = max, not = 2 - x)
INF = float("inf")
NAN = float("nan")
_CMP = ("==", "!=", "<", "<=", ">", ">=")
TEMPORAL_WORDS = ("always", "eventually", "next", "until", "since", "once", "historically", "at", "every")
_OPS2 = ("->", "**", "<=", ">=", "==", "!=", "&&", "||")
_TEMP_KINDS = {"G", "F", "X", "U", "O", "H", "S", "at", "every", "imp"}
_UNARY = {"always": "G", "eventually": "F", "once": "O", "historically": "H"}


class SpecError(Exception):
    def __init__(self, msg, col=None, hint=""):
        super().__init__(msg)
        self.msg, self.col, self.hint = msg, col, hint

    def __str__(self):
        return self.msg + (f" at column {self.col}" if self.col else "") + (f" (hint: {self.hint})" if self.hint else "")


def _ftokens(src):
    """Tokens (kind, value, start, end) with 0-based character offsets."""
    if not isinstance(src, str):
        raise SpecError("the formula must be a string")
    if len(src) > _expr.MAX_LEN:
        raise SpecError(f"formula too long ({len(src)} chars, max {_expr.MAX_LEN})")
    toks, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            while j < n and (src[j].isdigit() or src[j] == "."):
                j += 1
            if j < n and src[j] in "eE":
                k = j + 1
                if k < n and src[k] in "+-":
                    k += 1
                if k < n and src[k].isdigit():
                    j = k
                    while j < n and src[j].isdigit():
                        j += 1
            try:
                v = float(src[i:j])
            except ValueError:
                raise SpecError(f"bad number '{src[i:j]}'", i + 1)
            toks.append(("num", v, i, j))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_."):
                j += 1
            name = src[i:j].rstrip(".")
            toks.append(("name", name, i, i + len(name)))
            i += len(name)
            continue
        if c in "'\"":
            j = src.find(c, i + 1)
            if j < 0:
                raise SpecError("unterminated string", i + 1, f"close it with {c}")
            toks.append(("str", src[i + 1:j], i, j + 1))
            i = j + 1
            continue
        if src[i:i + 2] in _OPS2:
            toks.append(("op", src[i:i + 2], i, i + 2))
            i += 2
            continue
        if c in "+-*/%^()<>,[]!:":
            toks.append(("op", c, i, i + 1))
            i += 1
            continue
        if c == "=":
            raise SpecError("single '=' is not allowed", i + 1, "use '==' to compare")
        raise SpecError(f"unexpected character '{c}'", i + 1)
    toks.append(("end", None, n, n))
    return toks


def _desc(t):
    if t[0] == "end":
        return "end of formula"
    return f"'{t[1]}'" if t[0] != "num" else f"number {t[1]:g}"


class _N:
    __slots__ = ("k", "kids", "s", "e", "x", "temp")

    def __init__(self, k, kids, s, e, x=None):
        self.k, self.kids, self.s, self.e, self.x = k, kids, s, e, x
        self.temp = k in _TEMP_KINDS or any(c.temp for c in kids)


class _FP:
    MAX_DEPTH = 60

    def __init__(self, src):
        self.src = src
        self.toks = _ftokens(src)
        self.i = 0
        self.depth = 0

    def peek(self):
        return self.toks[min(self.i, len(self.toks) - 1)]

    def next(self):
        t = self.toks[min(self.i, len(self.toks) - 1)]
        self.i += 1
        return t

    def is_op(self, v):
        t = self.peek()
        return t[0] == "op" and t[1] == v

    def is_kw(self, v):
        t = self.peek()
        return t[0] == "name" and t[1] == v

    def expect_op(self, v):
        t = self.next()
        if not (t[0] == "op" and t[1] == v):
            raise SpecError(f"expected '{v}' but found {_desc(t)}", t[2] + 1)
        return t

    def expect_kw(self, v):
        t = self.next()
        if not (t[0] == "name" and t[1] == v):
            raise SpecError(f"expected '{v}' but found {_desc(t)}", t[2] + 1, "if <cond> then <a> else <b>")
        return t

    def int_tok(self, what):
        t = self.next()
        if t[0] != "num" or t[1] != int(t[1]) or t[1] < 0:
            raise SpecError(f"{what} must be a non-negative integer, found {_desc(t)}", t[2] + 1)
        return int(t[1])

    def _enter(self):
        self.depth += 1
        if self.depth > self.MAX_DEPTH:
            raise SpecError("formula nested too deeply", self.peek()[2] + 1)

    def parse(self):
        f = self.formula()
        t = self.peek()
        if t[0] != "end":
            raise SpecError(f"unexpected {_desc(t)}", t[2] + 1, "missing operator or unbalanced parentheses?")
        return f

    def formula(self):                      # implication (right-associative), lowest precedence
        self._enter()
        try:
            a = self.orf()
            if self.is_op("->"):
                self.next()
                b = self.formula()
                return _N("imp", [a, b], a.s, b.e)
            return a
        finally:
            self.depth -= 1

    def orf(self):
        a = self.andf()
        while self.is_kw("or") or self.is_op("||"):
            self.next()
            b = self.andf()
            a = _N("or", [a, b], a.s, b.e)
        return a

    def andf(self):
        a = self.untilf()
        while self.is_kw("and") or self.is_op("&&"):
            self.next()
            b = self.untilf()
            a = _N("and", [a, b], a.s, b.e)
        return a

    def untilf(self):
        a = self.unaryf()
        while self.is_kw("until") or self.is_kw("since"):
            t = self.next()
            w = self.window()
            b = self.unaryf()
            a = _N("U" if t[1] == "until" else "S", [a, b], a.s, b.e, w)
        return a

    def window(self):
        if not self.is_op("["):
            return (0, None)
        t0 = self.next()
        a = self.int_tok("the window start")
        self.expect_op(",")
        b = self.int_tok("the window end")
        self.expect_op("]")
        if a > b:
            raise SpecError(f"window [{a},{b}] has a > b", t0[2] + 1, "write [a,b] with a <= b, in steps from now")
        return (a, b)

    def unaryf(self):
        t = self.peek()
        if (t[0] == "name" and t[1] == "not") or (t[0] == "op" and t[1] == "!"):
            self.next()
            self._enter()
            try:
                a = self.unaryf()
            finally:
                self.depth -= 1
            return _N("not", [a], t[2], a.e)
        if t[0] == "name":
            w = t[1]
            if w in _UNARY:
                self.next()
                win = self.window()
                self._enter()
                try:
                    a = self.unaryf()
                finally:
                    self.depth -= 1
                return _N(_UNARY[w], [a], t[2], a.e, win)
            if w == "next":
                self.next()
                self._enter()
                try:
                    a = self.unaryf()
                finally:
                    self.depth -= 1
                return _N("X", [a], t[2], a.e)
            if w == "at":
                self.next()
                if self.is_kw("final"):
                    self.next()
                    kv = "final"
                else:
                    kv = self.int_tok("the step after 'at'")
                self.expect_op(":")
                a = self.formula()
                return _N("at", [a], t[2], a.e, kv)
            if w == "every":
                self.next()
                k = self.int_tok("the period after 'every'")
                if k < 1:
                    raise SpecError("the period of 'every' must be at least 1", t[2] + 1)
                s0 = 0
                if self.is_kw("from"):
                    self.next()
                    s0 = self.int_tok("the start after 'from'")
                self.expect_op(":")
                a = self.formula()
                return _N("every", [a], t[2], a.e, (k, s0))
        return self.cmp()

    @staticmethod
    def no_temp(*nodes):
        for n in nodes:
            if n.temp:
                raise SpecError("a temporal formula (or ->) cannot be part of arithmetic or a comparison", n.s + 1,
                                "compare inside the temporal operator, e.g. always (x <= 3)")

    def cmp(self):
        a = self.add()
        t = self.peek()
        if t[0] == "op" and t[1] in _CMP:
            self.next()
            b = self.add()
            self.no_temp(a, b)
            a = _N("cmp", [a, b], a.s, b.e, t[1])
            t2 = self.peek()
            if t2[0] == "op" and t2[1] in _CMP:
                raise SpecError("chained comparison", t2[2] + 1, "write (a < b) and (b < c)")
        return a

    def add(self):
        a = self.mul()
        while self.peek()[0] == "op" and self.peek()[1] in ("+", "-"):
            self.next()
            b = self.mul()
            self.no_temp(a, b)
            a = _N("arith", [a, b], a.s, b.e)
        return a

    def mul(self):
        a = self.unary()
        while self.peek()[0] == "op" and self.peek()[1] in ("*", "/", "%"):
            self.next()
            b = self.unary()
            self.no_temp(a, b)
            a = _N("arith", [a, b], a.s, b.e)
        return a

    def unary(self):
        if self.is_op("-") or self.is_op("+"):
            t = self.next()
            self._enter()
            try:
                a = self.unary()
            finally:
                self.depth -= 1
            self.no_temp(a)
            return _N("arith", [a], t[2], a.e)
        a = self.postfix()
        if self.is_op("^") or self.is_op("**"):
            self.next()
            self._enter()
            try:
                b = self.unary()
            finally:
                self.depth -= 1
            self.no_temp(a, b)
            a = _N("arith", [a, b], a.s, b.e)
        return a

    def postfix(self):
        a = self.primary()
        if self.is_op("["):
            t = self.next()
            if a.k != "name":
                raise SpecError("[t-k] can only follow a variable name", t[2] + 1)
            tt = self.next()
            if not (tt[0] == "name" and tt[1] == "t"):
                raise SpecError(f"inside [...] expected 't' or 't-k', found {_desc(tt)}", tt[2] + 1, "e.g. x[t-1]")
            if self.is_op("-"):
                self.next()
                self.int_tok("the lag")
            close = self.expect_op("]")
            a = _N("name", [], a.s, close[3], a.x)
        return a

    def primary(self):
        t = self.next()
        kind, v, s, e = t
        if kind in ("num", "str"):
            return _N(kind, [], s, e)
        if kind == "op" and v == "(":
            inner = self.formula()
            close = self.expect_op(")")
            return _N("paren", [inner], s, close[3])
        if kind == "name":
            if v in TEMPORAL_WORDS:
                raise SpecError(f"temporal operator '{v}' inside an expression", s + 1,
                                "a temporal operator applies to a whole formula, e.g. always (x <= cap)")
            if v == "if":
                c = self.formula()
                self.expect_kw("then")
                a = self.formula()
                self.expect_kw("else")
                b = self.formula()
                self.no_temp(c, a, b)
                return _N("if", [c, a, b], s, b.e)
            if v in ("then", "else", "and", "or", "not"):
                raise SpecError(f"unexpected keyword '{v}'", s + 1)
            if self.is_op("("):
                self.next()
                args = []
                if not self.is_op(")"):
                    args.append(self.formula())
                    while self.is_op(","):
                        self.next()
                        args.append(self.formula())
                close = self.expect_op(")")
                self.no_temp(*args)
                return _N("call", args, s, close[3], v)
            return _N("name", [], s, e, v)
        raise SpecError(f"unexpected {_desc(t)}", s + 1, "an operand (number, name, '(') was expected")


def _build(node, src):
    """Collapse every maximal non-temporal sub-formula into ('atom', _expr AST, text)."""
    if not node.temp:
        text = src[node.s:node.e]
        try:
            return ("atom", _expr.parse(text), text)
        except _expr.ExprError as e:
            raise SpecError(f"in '{text.strip()[:60]}': {e.msg}", node.s + (e.col or 1), e.hint)
    k, kids = node.k, node.kids
    if k == "paren":
        return _build(kids[0], src)
    if k == "not":
        return ("not", _build(kids[0], src))
    if k in ("and", "or", "imp"):
        return (k, _build(kids[0], src), _build(kids[1], src))
    if k in ("G", "F", "O", "H"):
        return (k, node.x[0], node.x[1], _build(kids[0], src))
    if k == "X":
        return ("X", _build(kids[0], src))
    if k in ("U", "S"):
        return (k, node.x[0], node.x[1], _build(kids[0], src), _build(kids[1], src))
    if k == "at":
        return ("at", node.x, _build(kids[0], src))
    if k == "every":
        return ("every", node.x[0], node.x[1], _build(kids[0], src))
    raise SpecError(f"internal: unexpected node {k}")


@lru_cache(maxsize=2048)
def parse_formula(src):
    """-> (tree, None) or (None, problem text). Trees are nested tuples:
    ('atom', ast, text) ('not', f) ('and'|'or'|'imp', f, g) ('G'|'F'|'O'|'H', a, b|None, f) ('X', f)
    ('U'|'S', a, b|None, f, g) ('at', k|'final', f) ('every', k, s, f)."""
    try:
        return _build(_FP(src).parse(), src), None
    except SpecError as e:
        return None, str(e)
    except RecursionError:
        return None, "formula nested too deeply"


def _subtrees(tree):
    yield tree
    if tree[0] == "atom":
        return
    for x in tree[1:]:
        if isinstance(x, tuple):
            yield from _subtrees(x)


def _ast_nodes(a):
    yield a
    k = a[0]
    if k in ("neg", "not"):
        yield from _ast_nodes(a[1])
    elif k == "bin":
        yield from _ast_nodes(a[2])
        yield from _ast_nodes(a[3])
    elif k in ("and", "or"):
        yield from _ast_nodes(a[1])
        yield from _ast_nodes(a[2])
    elif k == "if":
        for x in a[1:]:
            yield from _ast_nodes(x)
    elif k == "call":
        for x in a[2]:
            yield from _ast_nodes(x)


def tree_info(tree):
    """(names read, number of numeric comparisons, number of temporal operators, largest bounded future window or
    None, uses `next`)."""
    names, ncmp, nops, bmax, nxt = set(), 0, 0, None, False
    for st in _subtrees(tree):
        k = st[0]
        if k == "atom":
            for a in _ast_nodes(st[1]):
                if a[0] == "var" and a[1] != "t":
                    names.add(a[1])
                elif a[0] == "bin" and a[1] in _CMP:
                    ncmp += 1
        else:
            if k in ("G", "F", "X", "U", "O", "H", "S", "at", "every"):
                nops += 1
            if k in ("G", "F", "U") and st[2] is not None:
                bmax = st[2] if bmax is None else max(bmax, st[2])
            if k == "X":
                nxt = True
    return names, ncmp, nops, bmax, nxt


# ---------------------------------------------------------------------------------------------------
# Clauses

_CID = itertools.count(1)


class Clause:
    __slots__ = ("cid", "name", "kind", "formula", "quote", "tree", "problems", "notes", "names", "ncmp", "nops",
                 "bmax", "nxt", "origin")

    def __init__(self, name, kind, formula, quote, origin="spec"):
        self.cid = (origin, next(_CID)) if origin != "spec" else ("spec", name)
        self.name, self.kind, self.formula, self.quote, self.origin = name, kind, formula, quote, origin
        self.tree, self.problems, self.notes = None, [], []
        self.names, self.ncmp, self.nops, self.bmax, self.nxt = set(), 0, 0, None, False

    @property
    def ok(self):
        return self.tree is not None and not self.problems

    def export(self):
        return {"name": self.name, "kind": self.kind, "formula": self.formula, "quote": self.quote,
                "problems": list(self.problems) + list(self.notes)}


def make_clause(name, ent, origin="spec", default_kind=None):
    """Clause from a SPEC entry: (kind, formula, quote), [kind, formula], {kind, formula, quote} or (for one-off
    asks) a formula string. Problems are recorded on the clause (it is then skipped), never raised."""
    kind, formula, quote, probs = default_kind, None, "", []
    if isinstance(ent, str) and default_kind:
        formula = ent
    elif isinstance(ent, (tuple, list)) and len(ent) >= 2:
        kind, formula = ent[0], ent[1]
        quote = ent[2] if len(ent) >= 3 else ""
        if len(ent) > 3:
            probs.append(f"has {len(ent)} parts; (kind, formula, quote) expected")
    elif isinstance(ent, dict):
        kind, formula, quote = ent.get("kind", default_kind), ent.get("formula"), ent.get("quote", "")
    else:
        c = Clause(str(name), str(kind or "?"), str(ent)[:120], "", origin)
        c.problems.append("must be (kind, formula, quote)")
        return c
    kind_s = str(kind).strip().lower() if isinstance(kind, str) else str(kind)
    c = Clause(str(name), kind_s, formula if isinstance(formula, str) else str(formula),
               quote if isinstance(quote, str) else str(quote), origin)
    c.problems += probs
    if kind_s not in KINDS:
        c.problems.append(f"unknown kind '{kind_s}' (use rule, fact or ask)")
    if not isinstance(formula, str) or not formula.strip():
        c.problems.append("the formula must be a non-empty string")
        return c
    if origin == "spec" and not c.quote.strip():
        c.notes.append("no quote (each clause should quote the description)")
    tree, err = parse_formula(formula)
    if err:
        c.problems.append(f"malformed: {err}")
        return c
    c.tree = tree
    c.names, c.ncmp, c.nops, c.bmax, c.nxt = tree_info(tree)
    if c.nops > MAX_TEMPORAL:
        c.problems.append(f"{c.nops} temporal operators (at most {MAX_TEMPORAL} per clause)")
    return c


@lru_cache(maxsize=64)
def parse_spec(src):
    """-> (present, clauses tuple, notes tuple). Static and never raising; a clause with a problem is kept with its
    problems (and skipped by the monitor)."""
    if not isinstance(src, str) or "SPEC" not in src:
        return False, (), ()
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        tree = _g3.repair_syntax(src)[2]
    nodes = _spec_value_nodes(tree) if tree is not None else []
    if not nodes:
        return False, (), ()
    try:
        val = _g3._const(nodes[-1])
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
        return True, (), ("SPEC is not a literal dict (only literal strings, tuples and lists): nothing is checked",)
    if not isinstance(val, dict):
        return True, (), ("SPEC must be a dict {name: (kind, formula, quote)}: nothing is checked",)
    clauses, notes = [], []
    items = list(val.items())
    for name, ent in items[:MAX_CLAUSES]:
        clauses.append(make_clause(name, ent))
    if len(items) > MAX_CLAUSES:
        notes.append(f"more than {MAX_CLAUSES} clauses: {', '.join(str(n) for n, _ in items[MAX_CLAUSES:])} skipped")
    return True, tuple(clauses), tuple(notes)


def parse_asks(raw, run_name):
    """One-off `ask` clauses of a run: a formula string, a list of strings or {name: formula}."""
    raw = _g3._maybe_json(raw)
    if raw is None:
        return [], []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = [(f"ask{i}", x) for i, x in enumerate(raw, 1)]
    else:
        return [], [f"note: run '{run_name}': ask must be a list of formulas or {{name: formula}}"]
    out, notes = [], []
    for name, f in items[:MAX_CLAUSES]:
        c = make_clause(name, f, origin="ask", default_kind="ask")
        if c.problems:
            notes.append(f"note: run '{run_name}': ask '{c.name}' skipped: " + "; ".join(c.problems))
        out.append(c)
    return out, notes


# ---------------------------------------------------------------------------------------------------
# Monitor (spec 1.4, 1.5, 3.3): bottom-up dynamic programming over the trace, Kleene verdicts and robustness

_MISSING = object()


def _swin(vals, los, his, use_min, empty):
    """out[t] = min (or max) of vals[los[t] .. his[t]] (inclusive), `empty` when los[t] > his[t]; los and his are
    non-decreasing in t (sliding-window extrema with a monotone deque, O(len))."""
    out, dq, nxt = [], deque(), 0
    for lo, hi in zip(los, his):
        if lo > hi:
            out.append(empty)
            continue
        while nxt <= hi:
            v = vals[nxt]
            if use_min:
                while dq and vals[dq[-1]] >= v:
                    dq.pop()
            else:
                while dq and vals[dq[-1]] <= v:
                    dq.pop()
            dq.append(nxt)
            nxt += 1
        while dq[0] < lo:
            dq.popleft()
        out.append(vals[dq[0]])
    return out


def _kcomb(fn, v, va, ra, vb, rb):
    if v == U:
        return NAN
    rs = [r for vv, r in ((va, ra), (vb, rb)) if vv != U]
    return fn(rs)


def _av(node, lookup, errs):
    """(verdict, robustness) of an atom AST at one position. Comparisons give the margin in their own units;
    other boolean-valued expressions +-inf; anything that cannot be evaluated is undefined."""
    k = node[0]
    try:
        if k == "bin" and node[1] in _CMP:
            op = node[1]
            a = _expr.evaluate(node[2], lookup)
            b = _expr.evaluate(node[3], lookup)
            if isinstance(a, str) or isinstance(b, str):
                if op in ("==", "!="):
                    ok = (a == b) == (op == "==")
                elif isinstance(a, str) and isinstance(b, str):
                    ok = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
                else:
                    raise _expr.EvalError(f"cannot compare {a!r} {op} {b!r}")
                return (T if ok else F), (INF if ok else -INF)
            x, y = _expr._num(a, op), _expr._num(b, op)
            if math.isnan(x) or math.isnan(y):
                raise _expr.EvalError("not a number")
            ok = {"<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y, "==": x == y, "!=": x != y}[op]
            r = {"<": y - x, "<=": y - x, ">": x - y, ">=": x - y, "==": -abs(x - y), "!=": abs(x - y)}[op]
            return (T if ok else F), (NAN if math.isnan(r) else r)
        if k == "not":
            v, r = _av(node[1], lookup, errs)
            return 2 - v, -r
        if k in ("and", "or"):
            va, ra = _av(node[1], lookup, errs)
            vb, rb = _av(node[2], lookup, errs)
            fn = min if k == "and" else max
            v = fn(va, vb)
            return v, _kcomb(fn, v, va, ra, vb, rb)
        if k == "if":
            vc, _ = _av(node[1], lookup, errs)
            if vc == U:
                return U, NAN
            return _av(node[2] if vc == T else node[3], lookup, errs)
        ok = _expr._truth(_expr.evaluate(node, lookup))
        return (T if ok else F), (INF if ok else -INF)
    except (_expr.EvalError, _expr.ExprError, TypeError, ValueError, ZeroDivisionError, OverflowError,
            RecursionError) as e:
        if not errs:
            errs.append(str(e)[:120])
        return U, NAN


def _atom_refs(a):
    refs = sorted({(x[1], x[2]) for x in _ast_nodes(a) if x[0] == "var" and x[1] != "t"})
    uses_t = any(x[0] == "var" and x[1] == "t" for x in _ast_nodes(a))
    return tuple(refs), uses_t


class _Trace:
    """Verdict / robustness arrays of every sub-formula of a clause on one finite trace o_0 .. o_H."""

    def __init__(self, series, scen, H, acache):
        self.series, self.scen, self.H, self.acache = series, scen, H, acache
        self.memo = {}
        self.why = []

    def node(self, n, open_=False):
        key = (id(n), False if n[0] == "atom" else open_)
        hit = self.memo.get(key)
        if hit is None:
            hit = self.memo[key] = self._node(n, open_)
        return hit

    def value(self, name, t):
        xs = self.series.get(name)
        if xs is not None and 0 <= t < len(xs) and xs[t] is not None:
            return xs[t]
        return self.scen.get(name, _MISSING)

    def _atom(self, n):
        a, H = n[1], self.H
        refs, uses_t = _atom_refs(a)
        cache = self.acache.setdefault(id(a), {})
        cols = []
        for name, lag in refs:
            xs = self.series.get(name)
            sv = self.scen.get(name, _MISSING)
            if xs is None:
                cols.append([sv] * (H + 1))
            else:
                col = []
                for t in range(H + 1):
                    j = t - lag if t >= lag else 0
                    x = xs[j] if j < len(xs) else None
                    col.append(sv if x is None else x)
                cols.append(col)
        V, R = [], []
        rows = zip(*cols) if cols else (() for _ in range(H + 1))
        for t, vals in enumerate(rows):
            key = (vals, t) if uses_t else vals
            try:
                hit = cache.get(key)
            except TypeError:               # an unhashable value: evaluate without the cache
                key, hit = None, None
            if hit is None:
                env = dict(zip(refs, vals))

                def lookup(name, lag, env=env, t=t):
                    if name == "t":
                        return t
                    v = env.get((name, lag), _MISSING)
                    if v is _MISSING:
                        raise _expr.EvalError(f"unknown name '{name}'" + (f" (lag {lag})" if lag else ""))
                    return v
                errs = []
                v, r = _av(a, lookup, errs)
                hit = (v, r, errs[0] if errs else None)
                if key is not None and len(cache) < 200000:
                    cache[key] = hit
            if hit[2] and not self.why:
                self.why.append(hit[2])
            V.append(hit[0])
            R.append(hit[1])
        return V, R

    def _node(self, n, op):
        H, k, wr = self.H, n[0], not op
        if k == "atom":
            return self._atom(n)
        if k == "not":
            v, r = self.node(n[1], op)
            return [2 - x for x in v], ([-x for x in r] if wr else None)
        if k in ("and", "or", "imp"):
            va, ra = self.node(n[1], op)
            vb, rb = self.node(n[2], op)
            if k == "imp":
                va = [2 - x for x in va]
                ra = [-x for x in ra] if wr else None
            fn = min if k == "and" else max
            V = [fn(x, y) for x, y in zip(va, vb)]
            R = [_kcomb(fn, v, x, rx, y, ry) for v, x, rx, y, ry in zip(V, va, ra, vb, rb)] if wr else None
            return V, R
        if k in ("G", "F", "O", "H"):
            a, b = n[1], n[2]
            v, r = self.node(n[3], op)
            use_min = k in ("G", "H")
            if k in ("G", "F"):
                los = [t + a for t in range(H + 1)]
                his = [H if b is None else min(t + b, H) for t in range(H + 1)]
            else:
                los = [0 if b is None else max(0, t - b) for t in range(H + 1)]
                his = [t - a for t in range(H + 1)]
            V = _swin(v, los, his, use_min, T if use_min else F)
            if op and k in ("G", "F") and b is not None:     # unknown beyond the horizon
                for t in range(max(0, H - b + 1), H + 1):
                    V[t] = min(V[t], U) if use_min else max(V[t], U)
            R = None
            if wr:
                fill = INF if use_min else -INF
                R = _swin([x if vv != U else fill for x, vv in zip(r, v)], los, his, use_min, fill)
                R = [NAN if vv == U else x for x, vv in zip(R, V)]
            return V, R
        if k == "X":
            v, r = self.node(n[1], op)
            return v[1:] + [U if op else F], ((r[1:] + [-INF]) if wr else None)
        if k in ("U", "S"):
            return self._until(n, op)
        if k == "at":
            kk = H if n[1] == "final" else n[1]
            v, r = self.node(n[2], op)
            if 0 <= kk <= H:
                return [v[kk]] * (H + 1), ([r[kk]] * (H + 1) if wr else None)
            msg = f"at {n[1]}: step {kk} is after the horizon {H}"
            if msg not in self.why:
                self.why.insert(0, msg)
            return [U] * (H + 1), ([NAN] * (H + 1) if wr else None)
        if k == "every":
            per, s0 = n[1], n[2]
            v, r = self.node(n[3], op)
            js = range(s0, H + 1, per)
            vv = min((v[j] for j in js), default=T)
            rr = NAN
            if wr and vv != U:
                rr = min((r[j] if v[j] != U else INF for j in js), default=INF)
            return [vv] * (H + 1), ([rr] * (H + 1) if wr else None)
        raise ValueError(f"unknown node {k}")

    def _until(self, n, op):
        """phi U[a,b] psi (future) / phi S[a,b] psi (past), Kleene verdicts and robustness."""
        H, wr = self.H, not op
        k, a, b = n[0], n[1], n[2]
        vl, rl = self.node(n[3], op)
        vr, rr = self.node(n[4], op)
        if wr:
            rlm = [x if v != U else INF for x, v in zip(rl, vl)]       # inside min
            rrm = [x if v != U else -INF for x, v in zip(rr, vr)]      # inside max
        V, R = [F] * (H + 1), ([-INF] * (H + 1) if wr else None)
        if b is None:
            V0, R0 = [F] * (H + 1), [-INF] * (H + 1)
            order = range(H, -1, -1) if k == "U" else range(H + 1)
            prev = None
            for t in order:
                if prev is None:
                    V0[t] = vr[t]
                    if wr:
                        R0[t] = rrm[t]
                else:
                    V0[t] = max(vr[t], min(vl[t], V0[prev]))
                    if wr:
                        R0[t] = max(rrm[t], min(rlm[t], R0[prev]))
                prev = t
            if a == 0:
                V, R = V0, (R0 if wr else None)
            else:
                ts = range(H + 1)
                if k == "U":                    # phi on [t, t+a-1] and U0 at t+a
                    los, his = list(ts), [t + a - 1 for t in ts]
                    src = [t + a for t in ts]
                else:                           # phi on (t-a, t] and S0 at t-a
                    los, his = [max(0, t - a + 1) for t in ts], list(ts)
                    src = [t - a for t in ts]
                pv = _swin(vl, los, [min(h, H) for h in his], True, T)
                pr = _swin(rlm, los, [min(h, H) for h in his], True, INF) if wr else None
                for t in ts:
                    j = src[t]
                    if 0 <= j <= H:
                        V[t] = min(pv[t], V0[j])
                        if wr:
                            R[t] = min(pr[t], R0[j])
        else:
            for t in range(H + 1):
                bv, br, run_v, run_r = F, -INF, T, INF
                if k == "U":
                    js = range(t, min(t + b, H) + 1)
                    lo_ok = t + a
                else:
                    js = range(t, max(0, t - b) - 1, -1)
                    lo_ok = None
                for j in js:
                    if (j >= lo_ok) if k == "U" else (j <= t - a):
                        bv = max(bv, min(vr[j], run_v))
                        if wr:
                            br = max(br, min(rrm[j], run_r))
                    run_v = min(run_v, vl[j])
                    if wr:
                        run_r = min(run_r, rlm[j])
                    elif bv == T or run_v == F:
                        break
                if op and k == "U" and t + b > H:
                    # the window continues past H: psi may still come there while phi has not yet failed (run_v is
                    # the minimum of phi over [t, H] unless the loop stopped early, in which case this adds nothing)
                    bv = max(bv, min(U, run_v))
                V[t] = bv
                if wr:
                    R[t] = br
        if wr:
            R = [NAN if v == U else x for x, v in zip(R, V)]
        return V, R


def witness(tree, tr):
    """('viol', j) for the first violating step of a top-level always / every (or not eventually), ('sat', j) for
    the first satisfying step of a top-level eventually / until (or not always), ('at', k), or None."""
    H, k = tr.H, tree[0]
    if k == "not" and tree[1][0] in ("G", "F"):
        inner = tree[1]
        v, _ = tr.node(inner[3])
        lo, hi = inner[1], H if inner[2] is None else min(inner[2], H)
        want, tag = (T, "viol") if inner[0] == "F" else (F, "sat")
        j = next((j for j in range(lo, hi + 1) if v[j] == want), None)
        return (tag, j) if j is not None else None
    if k in ("G", "F"):
        v, _ = tr.node(tree[3])
        lo, hi = tree[1], H if tree[2] is None else min(tree[2], H)
        want, tag = (F, "viol") if k == "G" else (T, "sat")
        j = next((j for j in range(lo, hi + 1) if v[j] == want), None)
        return (tag, j) if j is not None else None
    if k == "U":
        vl, _ = tr.node(tree[3])
        vr, _ = tr.node(tree[4])
        hi = H if tree[2] is None else min(tree[2], H)
        run = T
        for j in range(0, hi + 1):
            if j >= tree[1] and min(vr[j], run) == T:
                return ("sat", j)
            run = min(run, vl[j])
            if run == F:
                break
        return None
    if k == "every":
        v, _ = tr.node(tree[3])
        j = next((j for j in range(tree[2], H + 1, tree[1]) if v[j] == F), None)
        return ("viol", j) if j is not None else None
    if k == "at":
        kk = H if tree[1] == "final" else tree[1]
        return ("at", kk) if 0 <= kk <= H else None
    return None


def eval_clause(c, series, scen, H, acache):
    """One clause on one trace: {v, r, wit, trunc, why, cex} (cex: seed -> counterexample text, when violated)."""
    tr = _Trace(series, scen, H, acache)
    V, R = tr.node(c.tree)
    v, r = V[0], R[0]
    trunc = False
    if (c.bmax is not None or c.nxt) and v != U:
        Vo, _ = tr.node(c.tree, True)
        trunc = Vo[0] == U
    w = witness(c.tree, tr)
    cex = None
    if v == F:
        j = w[1] if w and w[0] in ("viol", "at") else 0
        cex = lambda seed: counterexample(c, tr, seed, j)     # noqa: E731 (built only for the reported seed)
    return {"v": v, "r": r, "wit": w, "trunc": trunc, "why": tr.why[0] if tr.why else None, "cex": cex}


# ---------------------------------------------------------------------------------------------------
# Statistics and text


def wilson(k, n, z=1.96):
    if n <= 0:
        return NAN, NAN
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


class Stat:
    """Per clause and run: verdicts and margins by seed index, first-violation / witness times, truncation."""

    def __init__(self):
        self.v, self.r, self.vt, self.wt = {}, {}, [], []
        self.trunc = 0
        self.trunc_only = 0
        self.why = None
        self.first_bad = None          # (seed, counterexample text) of the lowest-index violating seed

    @property
    def n_true(self):
        return sum(1 for x in self.v.values() if x == T)

    @property
    def n_false(self):
        return sum(1 for x in self.v.values() if x == F)

    @property
    def n_undef(self):
        return sum(1 for x in self.v.values() if x == U)

    @property
    def n_def(self):
        return self.n_true + self.n_false

    def p(self):
        return self.n_true / self.n_def if self.n_def else NAN

    def holds(self):
        return self.n_false == 0 and self.n_undef == 0 and self.n_true > 0


def _qs(xs, label):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return ""
    if xs[0] == xs[-1]:
        return f"{label} {_g(xs[0])}"
    return f"{label} p05 {_g(_quant(xs, .05))} p50 {_g(_quant(xs, .5))} p95 {_g(_quant(xs, .95))}"


def _num_s(v):
    return _s3(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else repr(v)


def counterexample(c, tr, seed, j):
    """'seed 3, t=37->38: queue 8->9, max_queue 8' (at most MAX_CEX_KEYS keys)."""
    names = sorted(c.names)[:MAX_CEX_KEYS]
    parts = []
    for nm in names:
        observed = nm in tr.series and any(x is not None for x in tr.series[nm])
        cur = tr.value(nm, j)
        if cur is _MISSING:
            parts.append(f"{nm} (missing)")
        elif observed and j > 0:
            parts.append(f"{nm} {_num_s(tr.value(nm, j - 1))}->{_num_s(cur)}")
        else:
            parts.append(f"{nm} {_num_s(cur)}")
    at = f"t={j - 1}->{j}" if j > 0 else f"t={j}"
    return f"seed {seed}, {at}: " + ", ".join(parts)


def _quote(c, n=60):
    q = (c.quote or "").strip().replace("\n", " ")
    return (q[:n] + "...") if len(q) > n else q


def _single(c):
    return c.ncmp == 1


# ---------------------------------------------------------------------------------------------------
# The arm


class G4C2Arm(G3C1Arm):
    name = "G4C2"
    buildable = True
    executable = True
    guide_file = "g4c2_guide.md"
    mutating_tools = ("submit_model", "edit_model")
    custom_episode = True
    MONITOR_CAP_S = MONITOR_CAP_S

    # ------------------------------------------------------------------ schemas
    @staticmethod
    def _run_schema():
        s = _BASE["_run_schema"].__func__()
        s["ask"] = {"type": "array", "items": {"type": "string"}, "description": "temporal properties to evaluate on "
                    "this run and its baseline, e.g. \"eventually[0,60] queue >= 5\" (or {name: formula}); the "
                    "stored SPEC is unchanged"}
        return s

    @staticmethod
    def _sections_prop():
        p = _BASE["_sections_prop"].__func__()
        p["items"]["properties"]["name"]["description"] = (
            "SPEC, PARAMS, RULES, HORIZON, Model, Model.<method> (a new method name adds the method), HELPERS "
            "(imports, constants, helper functions) or a helper function's name")
        return p

    def tools(self):
        ts = _BASE["tools"](self)
        for t in ts:
            fn = t["function"]
            if fn["name"] == "submit_model":
                fn["description"] = ("Store the program (SPEC, PARAMS, RULES, HORIZON, class Model), check it (every "
                                     "problem is listed at once with its line), and if it is accepted run the "
                                     "baseline (HORIZON steps, 100 seeds) plus each entry of `runs`; every run is "
                                     "checked against the SPEC clauses.")
            elif fn["name"] == "show_model":
                fn["description"] = ("Show the program with line numbers, its section names, the declared "
                                     "PARAMS/RULES table and the SPEC clauses (or one `section` only).")
        return ts

    # ------------------------------------------------------------------ export / show
    def export(self, state):
        rep = _BASE["export"](self, state)
        present, clauses, notes = parse_spec(self._src(state))
        rep["spec"] = [c.export() for c in clauses]
        return rep

    def _spec_listing(self, src):
        present, clauses, notes = parse_spec(src)
        if not present:
            return ["SPEC: (not declared; nothing is checked against the description)"]
        out = [f"SPEC ({len(clauses)} clause(s)):"]
        for c in clauses:
            out.append(f"  {c.name} [{c.kind}] {c.formula[:160]}" + (f"   # '{_quote(c, 80)}'" if c.quote else ""))
            if c.tree is not None:
                out.append(f"      reads: {', '.join(sorted(c.names)) or '(no keys)'}")
            for p in c.problems + c.notes:
                out.append(f"      problem: {p[:200]}" if p in c.problems else f"      note: {p[:200]}")
        out += [f"  note: {n}" for n in notes]
        return out

    def _t_show_model(self, state, args):
        base = _BASE["_t_show_model"](self, state, args)
        src = self._src(state)
        want = args.get("section") if isinstance(args, dict) else None
        if not src or (isinstance(want, str) and want.strip()) or base.startswith("ERROR"):
            return base
        listing = "\n".join(self._spec_listing(src))[:2400]
        if len(base) + len(listing) + 1 > MAX_OUT:
            k = base.find("\nSECTIONS:")
            code, decl = (base[:k], base[k:]) if k >= 0 else (base, "")
            room = MAX_OUT - len(decl) - len(listing) - 120
            cut = code[:max(0, room)].rsplit("\n", 1)[0]
            base = cut + f"\n... [program truncated after line {cut.count(chr(10)) + 1} of " \
                         f"{len(src.splitlines())}]" + decl
        return (base + "\n" + listing)[:MAX_OUT]

    # ------------------------------------------------------------------ runs
    def _t_run_scenario(self, state, args):
        if isinstance(args, dict) and args.get("runs") is not None and args.get("ask") is not None:
            runs = _g3._maybe_json(args["runs"])
            if isinstance(runs, dict):
                runs = [runs]
            if isinstance(runs, list):
                runs = [({"ask": args["ask"], **r} if isinstance(r, dict) else r) for r in runs]
                args = {**{k: v for k, v in args.items() if k != "ask"}, "runs": runs}
        return _BASE["_t_run_scenario"](self, state, args)

    def _spec(self, src_info_cache, stored, r, k, horizon_default):
        s, err, notes = _BASE["_spec"](self, src_info_cache, stored, r, k, horizon_default)
        if s is not None:
            asks, an = parse_asks(r.get("ask") if isinstance(r, dict) else None, s["name"])
            s["_ask"] = asks
            s["_stored"] = stored
            notes = list(notes) + an
        return s, err, notes

    @staticmethod
    def _plan(specs):
        """Clauses evaluated on each run and the observation keys they read (cap MAX_SPEC_KEYS per run)."""
        for s in specs:
            present, clauses, _ = parse_spec(s.get("_stored", ""))
            cands = [c for c in clauses if c.ok] + [c for c in s.get("_ask", []) if c.ok]
            for d in specs:
                if d is not s and d.get("ref") == s["name"]:
                    cands += [c for c in d.get("_ask", []) if c.ok]
            scen_names = set(s["scen"])
            keys, take, skipped = [], [], []
            for c in cands:
                new = [x for x in sorted(c.names) if x not in scen_names and x not in keys]
                if len(keys) + len(new) > MAX_SPEC_KEYS:
                    skipped.append(c.name)
                    continue
                keys += new
                take.append(c)
            read = sorted(set(keys) | {x for c in take for x in c.names})
            s["_spec_plan"] = {"clauses": take, "skipped": skipped, "read": read}

    def _execute(self, specs, deadline=DEADLINE_S):
        self._plan(specs)
        saved = [s["events"] for s in specs]
        try:
            for s in specs:
                read = s["_spec_plan"]["read"]
                if read:   # internal pseudo-event: only its variables are used (G3C1's evkeys); never reported
                    s["events"] = list(s["events"]) + [("__spec__", ("keys", [("var", x, 0) for x in read]), "final")]
            return _BASE["_execute"](self, specs, deadline)
        finally:
            for s, ev in zip(specs, saved):
                s["events"] = ev

    def _report(self, specs, res, pre_lines, problems):
        head, groups = self._spec_blocks(specs, res)
        res2 = {**res, "notes": list(res["notes"]) + head}
        tok = _SPEC_GROUPS.set(groups)
        try:
            return _BASE["_report"](self, specs, res2, pre_lines, problems)
        finally:
            _SPEC_GROUPS.reset(tok)

    # ------------------------------------------------------------------ monitor over a call's runs
    def monitor(self, specs, res):
        """Evaluate each run's planned clauses on every completed seed. Returns (stats {run index: {cid: Stat}},
        skipped-by-name notes, seeds-cap note or None). Seeds are taken in index order across all runs, so a time
        cap keeps comparisons paired."""
        R = res["runs"]
        keys_seen = set()
        for r in R:
            keys_seen.update(r["keys"])
        stats = {i: {} for i in range(len(specs))}
        notes, unknown = [], {}
        for i, s in enumerate(specs):
            plan = s.get("_spec_plan") or {"clauses": [], "skipped": []}
            for c in plan["clauses"]:
                bad = [x for x in sorted(c.names) if x not in keys_seen and x not in s["scen"]]
                if bad and keys_seen:
                    pool = sorted(keys_seen | set(s["scen"]))
                    msg = "; ".join(f"unknown name '{x}'" + (
                        f" (did you mean {' or '.join(repr(m) for m in difflib.get_close_matches(x, pool, n=2, cutoff=0.6))}?)"
                        if difflib.get_close_matches(x, pool, n=1, cutoff=0.6) else "") for x in bad)
                    unknown[c.cid] = (c.name, msg)
                    continue
                stats[i][c.cid] = Stat()
            if plan["skipped"]:
                notes.append(f"run '{s['name']}': at most {MAX_SPEC_KEYS} observation keys per run; clauses "
                             f"{', '.join(plan['skipped'])} skipped")
        for name, msg in unknown.values():
            notes.append(f"clause '{name}': {msg}: skipped")
        seeds = sorted(set().union(*[set(r["seeds"]) for r in R])) if R else []
        acaches = [{} for _ in specs]
        rcache = {}
        t0 = time.time()
        cap_note = None
        for m, idx in enumerate(seeds):
            if m > 0 and time.time() - t0 > self.MONITOR_CAP_S:
                cap_note = (f"SPEC monitor time cap ({self.MONITOR_CAP_S:g} s): clauses were evaluated on the first "
                            f"{m} seeds of each run")
                break
            for i, (s, r) in enumerate(zip(specs, R)):
                if not stats[i]:
                    continue
                sd = r["seeds"].get(idx)
                if sd is None:
                    continue
                cls = [c for c in s["_spec_plan"]["clauses"] if c.cid in stats[i]]
                for c in cls:
                    st = stats[i][c.cid]
                    if sd["final"] is None:
                        st.v[idx] = U
                        st.why = st.why or "seed did not complete"
                        continue
                    ck = (i, sd.get("h"), c.cid)
                    out = rcache.get(ck) if sd.get("h") else None
                    cex = None
                    if out is None:
                        out = eval_clause(c, sd["ev"], s["scen"], s["steps"], acaches[i].setdefault(c.cid, {}))
                        cex = out.pop("cex")        # the trace arrays are not kept beyond this seed
                        if sd.get("h"):
                            rcache[ck] = out
                    self._record(st, out, idx, cex)
        return stats, notes, cap_note

    @staticmethod
    def _record(st, out, idx, cex):
        v = out["v"]
        st.v[idx] = v
        if v == U:
            st.why = st.why or out["why"] or "cannot be evaluated"
            return
        if math.isfinite(out["r"]):
            st.r[idx] = out["r"]
        if out["trunc"]:
            st.trunc += 1
        w = out["wit"]
        if v == F:
            if out["trunc"]:
                st.trunc_only += 1
            j = w[1] if w and w[0] in ("viol", "at") else None
            if j is not None:
                st.vt.append(j)
            if st.first_bad is None and cex is not None:      # seeds come in index order: the lowest index
                st.first_bad = (idx, cex(idx))
        elif w and w[0] in ("sat", "at"):
            st.wt.append(w[1])

    def _spec_blocks(self, specs, res):
        """(header lines with SPEC CONFORMANCE and SPEC NOTES, per-run groups)."""
        stored = specs[0].get("_stored", "") if specs else ""
        present, clauses, snotes = parse_spec(stored)
        any_ask = any(s.get("_ask") for s in specs)
        if not present and not any_ask:
            return [NO_SPEC_NOTE], []
        stats, mnotes, cap_note = self.monitor(specs, res)
        R = res["runs"]
        head = []
        static = [f"clause '{c.name}': " + "; ".join(c.problems) for c in clauses if c.problems]
        static += list(snotes)
        for s in specs:
            for c in s.get("_ask", []):
                if c.problems:
                    static.append(f"ask '{c.name}': " + "; ".join(c.problems))
        base_i = next((i for i, s in enumerate(specs) if not s["changed"] and s["stored"]), None)
        rf = [c for c in clauses if c.kind in ("rule", "fact")]
        if not present:
            head.append(NO_SPEC_NOTE)
        elif base_i is None:
            head.append("SPEC CONFORMANCE: no unchanged run in this call (facts are judged on the baseline only); "
                        "rule clauses are checked per run below")
        else:
            head += self._conformance(specs[base_i], stats[base_i], rf, R[base_i])
        allnotes = static + mnotes
        if allnotes:
            txt = "SPEC NOTES (these clauses are skipped; the rest are checked): " + " | ".join(allnotes)
            head.append(txt if len(txt) <= MAX_SPEC_NOTES else txt[:MAX_SPEC_NOTES - 20].rsplit(" ", 1)[0] + " ...")
        if cap_note:
            head.append(cap_note)
        groups = self._per_run(specs, stats, base_i, clauses)
        return head, groups

    def _conformance(self, s, st_by, rf, r):
        n_seeds = len(r["seeds"])
        checked = [c for c in rf if c.cid in st_by]
        if not checked:
            return [f"SPEC CONFORMANCE [{s['name']}, {n_seeds} seeds]: no rule/fact clause could be checked"
                    + (" (SPEC has none)" if not rf else "")]
        hold = [c for c in checked if st_by[c.cid].holds()]
        viol = [c for c in checked if st_by[c.cid].n_false]
        undef = [c for c in checked if not st_by[c.cid].n_false and not st_by[c.cid].holds()]
        lines = [f"SPEC CONFORMANCE [{s['name']}, {n_seeds} seeds]: {len(hold)} of {len(checked)} rule/fact clauses "
                 f"hold on every seed"]
        tail = []
        hs = [c for c in hold]
        if hs:
            mm = [f"{c.name} {_s3(min(st_by[c.cid].r.values()))}" for c in hs
                  if _single(c) and st_by[c.cid].r]
            tr = [c.name for c in hs if st_by[c.cid].trunc]
            tail.append("  holds: " + ", ".join(c.name for c in hs) + (f" (min margin: {', '.join(mm)})" if mm else "")
                        + (f"; truncated (a window passes the horizon): {', '.join(tr)}" if tr else ""))
        for c in undef:
            st = st_by[c.cid]
            tail.insert(len(tail) - (1 if hs else 0),
                        f"  UNDEFINED {c.name} ({c.kind}): {st.n_undef}/{len(st.v)} seeds ({(st.why or '')[:100]})")
        # violations first: the 'holds' and UNDEFINED lines keep only a small reserve and are shortened to fit
        used = len(lines[0])
        reserve = min(160, sum(len(x) + 1 for x in tail))
        body = []
        for k, c in enumerate(viol):
            st = st_by[c.cid]
            ln = (f"  VIOLATED {c.name} ({c.kind}, '{_quote(c)}'): {st.n_false}/{len(st.v)} seeds"
                  + (f" ({st.trunc_only} only through a window that passes the horizon)" if st.trunc_only else ""))
            if st.vt:
                ln += "; first " + _qs(st.vt, "at t")
            if _single(c):
                bad = [x for i, x in st.r.items() if st.v.get(i) == F]
                if bad:
                    ln += f"; worst margin {_s3(min(bad))}"
            elif c.ncmp > 1:
                ln += "; " + MARGIN_NA
            if st.n_undef:
                ln += f"; {st.n_undef} undefined"
            if st.first_bad is not None:
                ln += ";\n    " + st.first_bad[1][:200]
            room = MAX_CONF - reserve - used - 1
            if len(ln) > room:
                if k > 0:
                    body.append(f"  (+{len(viol) - k} more violated: {', '.join(x.name for x in viol[k:])})"[:200])
                    used += len(body[-1]) + 1
                    break
                ln = ln[:max(room, 100)].rstrip() + " ..."          # the first violation is always shown
            body.append(ln)
            used += len(ln) + 1
        for ln in tail:
            room = MAX_CONF - used - 1
            if len(ln) > room:
                ln = ln[:max(room - 4, 40)].rstrip() + " ..."
            body.append(ln)
            used += len(ln) + 1
        return lines + body

    def _per_run(self, specs, stats, base_i, clauses):
        groups = []
        by_name = {s["name"]: i for i, s in enumerate(specs)}
        rules = [c for c in clauses if c.kind == "rule"]
        facts = [c for c in clauses if c.kind == "fact"]
        for i, s in enumerate(specs):
            st_by = stats[i]
            if not st_by:
                continue
            ref_i = by_name.get(s.get("ref")) if s.get("ref") else None
            g = []
            asks = [c for c in s["_spec_plan"]["clauses"] if c.kind == "ask" and c.cid in st_by]
            for c in asks:
                g.append(self._ask_line(s, c, st_by[c.cid], specs, stats, ref_i))
            if i != base_i and s["changed"]:
                for c in rules:
                    if c.cid not in st_by:
                        continue
                    ln = self._rule_line(s, c, st_by[c.cid], specs, stats, ref_i)
                    if ln:
                        g.append(ln)
            if g:
                groups.append(g)
        cols = [i for i, s in enumerate(specs) if any(c.cid in stats[i] for c in facts)]
        if len(cols) >= 2 and facts:
            g = ["  facts x runs (✓ holds on every seed, ✗ on none, else the share of seeds): "
                 + " | ".join(specs[i]["name"] for i in cols)]
            for c in facts:
                cells = []
                for i in cols:
                    st = stats[i].get(c.cid)
                    if st is None or not st.n_def:
                        cells.append("?")
                    elif st.n_false == 0:
                        cells.append("✓" + (f" ({st.n_undef} undef.)" if st.n_undef else ""))
                    elif st.n_true == 0:
                        cells.append("✗")
                    else:
                        cells.append(f"{st.p():.2f}")
                g.append(f"    {c.name}: " + " | ".join(cells))
            groups.append(g)
        return groups

    @staticmethod
    def _times(st):
        out = []
        if st.wt:
            out.append("first true " + _qs(st.wt, "at t") + f" (n={len(st.wt)})")
        if st.vt:
            out.append("first violated " + _qs(st.vt, "at t") + f" (n={len(st.vt)})")
        return out

    def _ask_line(self, s, c, st, specs, stats, ref_i):
        label = f"{c.name} ({c.formula[:70]})"
        if not st.n_def:
            return f"  [{s['name']}] ask {label}: undefined on every seed ({(st.why or '')[:100]})"
        lo, hi = wilson(st.n_true, st.n_def)
        parts = [f"P={st.p():.3f} [{lo:.3f}, {hi:.3f}] (n={st.n_def}" + (f", {st.n_undef} undefined" if st.n_undef
                                                                        else "") + ")"]
        b = stats[ref_i].get(c.cid) if ref_i is not None else None
        if b is not None and b.n_def:
            common = [i for i in st.v if st.v[i] != U and b.v.get(i, U) != U]
            gained = sum(1 for i in common if b.v[i] == F and st.v[i] == T)
            lost = sum(1 for i in common if b.v[i] == T and st.v[i] == F)
            parts.append(f"vs {specs[ref_i]['name']} {b.p():.3f}: ΔP {st.p() - b.p():+.3f}; gained on {gained}, lost "
                         f"on {lost} of {len(common)} paired seeds")
            if _single(c):
                d = sorted(st.r[i] - b.r[i] for i in st.r if i in b.r)
                if d:
                    parts.append(f"margin change mean {_signed(sum(d) / len(d))} p05 {_signed(_quant(d, .05))} "
                                 f"p50 {_signed(_quant(d, .5))} p95 {_signed(_quant(d, .95))}")
        elif _single(c) and st.r:
            xs = sorted(st.r.values())
            parts.append(f"margin p50 {_s3(_quant(xs, .5))} min {_s3(xs[0])} max {_s3(xs[-1])}")
        if c.ncmp > 1:
            parts.append(MARGIN_NA)
        parts += self._times(st)
        if st.trunc:
            parts.append(f"truncated on {st.trunc} seeds (a window passes the horizon)")
        return f"  [{s['name']}] ask {label}: " + "; ".join(parts)

    def _rule_line(self, s, c, st, specs, stats, ref_i):
        b = stats[ref_i].get(c.cid) if ref_i is not None else None
        if b is None:
            if st.n_false:
                return (f"  [{s['name']}] rule {c.name}: violated under {s['name']} ({st.n_false}/{len(st.v)} seeds"
                        + ("; " + _qs(st.vt, "first at t") if st.vt else "") + ")")
            return None
        if all(st.v.get(i) == b.v.get(i) for i in set(st.v) | set(b.v)):
            return None
        common = [i for i in st.v if st.v[i] != U and b.v.get(i, U) != U]
        gained = sum(1 for i in common if b.v[i] == F and st.v[i] == T)
        lost = sum(1 for i in common if b.v[i] == T and st.v[i] == F)
        rn = specs[ref_i]["name"]
        if b.holds() and st.n_false:
            head = f"holds in {rn}, violated under {s['name']} ({st.n_false}/{len(st.v)} seeds" + (
                "; " + _qs(st.vt, "first at t") if st.vt else "") + ")"
        elif b.n_false and st.holds():
            head = f"violated in {rn} ({b.n_false}/{len(b.v)} seeds), holds under {s['name']}"
        else:
            head = f"P {b.p():.2f} in {rn} -> {st.p():.2f} under {s['name']}"
        return f"  [{s['name']}] rule {c.name}: {head}; gained on {gained}, lost on {lost} of {len(common)} paired seeds"


# Methods of G3C1Arm that G4C2Arm does not override run G3C1's code with SPEC as a declaration (see the docstring).
for _n, _f in _BASE.items():
    if _n not in G4C2Arm.__dict__:
        setattr(G4C2Arm, _n, _f)

G4C2 = G4C2Arm
