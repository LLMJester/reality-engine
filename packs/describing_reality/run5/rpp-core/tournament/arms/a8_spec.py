"""A8: set-theoretic state-transition specification (inspired by Z, Alloy and TLA+; see DISCOVERY.md §3a).

Representation (JSON):
  {"given":      [{"name", "instances": [atoms]} | {"name", "range": [lo, hi]} | {"name", "value": expr},
                  each with "provenance": {"kind": stated|derived|assumed, "quote": "..."}],
   "state":      [{"name", "type"}],                       type language: see TYPE GRAMMAR below
   "invariant":  [{"name"?, "expr"}],
   "init":       [{"var", "value": expr, "provenance"} | {"predicate": expr, "provenance"}],
   "operations": [{"name", "inputs": {name: type}, "pre": expr, "post": expr, "provenance"}],
   "derived":    [{"name", "params"?: [names], "expr"}],
   "scenarios":  [{"name", "steps": [{"op", "inputs"?, "repeat"?} | {"op": "any"}],
                   "set"?: {var: expr}, "constants"?: {name: value}, "observe"?: [expr]}]}

Expression language: a hand-written tokenizer + recursive-descent parser (no Python eval). Values are
ints, booleans, atoms (strings), frozensets (sets; a relation is a set of pairs; a function is a
functional relation), pairs (tuples) and sequences (Seq).

Execution: explicit-state. A postcondition is split into alternatives (or / exists / if / =>);
in each alternative, conjuncts `v' = e` determine variables, `v' in E` give candidate values, any other
mentioned-but-undetermined primed variable is enumerated over its (finite) type, and every conjunct is
then checked. Variables not mentioned in an alternative are unchanged (frame). More than one successor
is reported as branching. Bounded BFS explores "any enabled operation" sequences for invariant
violations (check_invariants) or reachability (find), with a state limit and a timeout.

Graceful degradation: a part that fails to parse or refers to unknown names becomes a hole and is left
out; everything else still runs. Tool calls never raise; errors are returned as text.
"""
import copy
import difflib
import itertools
import json
import time

from tournament.arms.base import Arm, ArmState, ERROR_PREFIX

MAX_CARRIER = 4096        # largest finite type that is enumerated
MAX_COMBOS = 20000        # candidate next-states per alternative / input combinations per operation
MAX_BRANCHES = 512        # alternatives per postcondition
MAX_RANGE = 100000        # largest a..b set
MAX_INIT = 256
MAX_FRONTIER = 64         # states kept per scenario step
MAX_WORK = 5_000_000      # expression nodes evaluated per tool call
DEFAULT_MAX_STATES = 20000
DEFAULT_TIMEOUT = 20.0
PROV_KINDS = ("stated", "derived", "assumed")


# ============================================================================ errors
class SpecError(Exception):
    """Parse / static error with position and hint."""

    def __init__(self, msg, pos=None, hint=""):
        super().__init__(msg)
        self.msg, self.pos, self.hint = msg, pos, hint

    def text(self, src=None):
        s = self.msg
        if self.pos is not None and src is not None:
            s += f" at char {self.pos}: «{src[max(0, self.pos - 15):self.pos]}⟦{src[self.pos:self.pos + 15]}»"
        if self.hint:
            s += f" (hint: {self.hint})"
        return s


class EvalError(Exception):
    def __init__(self, msg, hint=""):
        super().__init__(msg)
        self.msg, self.hint = msg, hint

    def text(self):
        return self.msg + (f" (hint: {self.hint})" if self.hint else "")


class _Missing(EvalError):
    """A variable's value is not known yet (used while solving postconditions)."""


# ============================================================================ values
class Seq(tuple):
    """A sequence value (distinct from a pair/tuple)."""


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _eq(a, b):
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def _sort_key(v):
    if isinstance(v, bool):
        return (0, str(v))
    if _is_num(v):
        return (1, v, "")
    if isinstance(v, str):
        return (2, v)
    return (3, fmt(v))


def fmt(v, in_rel=False):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return str(int(v)) if v == int(v) and abs(v) < 1e15 else f"{v:.6g}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v if v.replace("_", "a").isalnum() and not v[:1].isdigit() else json.dumps(v)
    if isinstance(v, Seq):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, tuple):
        if len(v) == 2 and in_rel:
            return f"{fmt(v[0])} |-> {fmt(v[1])}"
        return "(" + ", ".join(fmt(x) for x in v) + ")"
    if isinstance(v, frozenset):
        rel = all(isinstance(x, tuple) and not isinstance(x, Seq) and len(x) == 2 for x in v) and v
        return "{" + ", ".join(fmt(x, bool(rel)) for x in sorted(v, key=_sort_key)) + "}"
    return str(v)


def _from_json(x):
    """JSON literal -> value (lists become sets; [a, b] pairs inside a set of pairs stay sets of 2)."""
    if isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, list):
        return frozenset(_from_json(e) for e in x)
    if isinstance(x, dict):
        return frozenset((k, _from_json(v)) for k, v in x.items())
    raise EvalError(f"unsupported literal {x!r}")


# ============================================================================ tokenizer
UNICODE = {"∈": "in", "∉": "notin", "⊆": "subseteq", "⊂": "subset", "∪": "union", "∩": "inter", "∖": "\\",
           "⊕": "++", "↦": "|->", "∀": "forall", "∃": "exists", "∧": "and", "∨": "or", "¬": "not", "⇒": "=>",
           "⇔": "<=>", "≠": "!=", "≤": "<=", "≥": ">=", "×": "cross", "◁": "<|", "⩤": "<<|", "▷": "|>",
           "⩥": "|>>", "→": "->", "⇸": "+->", "↔": "<->", "•": "@", "·": "@", "⦇": "[", "⦈": "]",
           "⟨": "[", "⟩": "]", "−": "-"}
MULTI = sorted(["<<|", "|>>", "<=>", "|->", "+->", "<->", "=>", "<=", ">=", "!=", "/=", "==", "\\/", "/\\",
                "++", "..", "<|", "|>", "->", "&&", "||"], key=len, reverse=True)
SINGLE = set("+-*/%#(){}[],:|=<>\\~@.!")
ALIAS = {"&&": "and", "||": "or", "!": "not", "==": "=", "/=": "!=", "\\/": "union", "/\\": "inter",
         "implies": "=>", "iff": "<=>", "/": "/"}
KEYWORDS = {"in", "notin", "not", "and", "or", "implies", "iff", "forall", "exists", "if", "then", "else",
            "true", "false", "union", "inter", "subseteq", "subset", "div", "mod", "cross", "sum"}


class Tok:
    __slots__ = ("kind", "val", "pos")

    def __init__(self, kind, val, pos):
        self.kind, self.val, self.pos = kind, val, pos

    def __repr__(self):
        return f"{self.kind}:{self.val}"


def tokenize(src):
    if not isinstance(src, str):
        raise SpecError(f"expected an expression string, got {type(src).__name__}", None,
                        "write expressions as strings")
    toks, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c == "∅":
            toks += [Tok("op", "{", i), Tok("op", "}", i)]
            i += 1
            continue
        if c == "ℙ":
            toks.append(Tok("name", "P", i))
            i += 1
            continue
        if c in UNICODE:
            v = UNICODE[c]
            toks.append(Tok("op", v, i))
            i += 1
            continue
        if c.isdigit():
            j = i
            while j < n and src[j].isdigit():
                j += 1
            if j + 1 < n and src[j] == "." and src[j + 1].isdigit():
                j += 1
                while j < n and src[j].isdigit():
                    j += 1
                toks.append(Tok("num", float(src[i:j]), i))
            else:
                toks.append(Tok("num", int(src[i:j]), i))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            if j < n and src[j] == "'" and word not in KEYWORDS:
                toks.append(Tok("pname", word, i))
                j += 1
            elif word in KEYWORDS:
                toks.append(Tok("op", ALIAS.get(word, word), i))
            else:
                toks.append(Tok("name", word, i))
            i = j
            continue
        if c in "'\"":
            j = src.find(c, i + 1)
            if j < 0:
                raise SpecError("unterminated string", i, "close the quote")
            toks.append(Tok("str", src[i + 1:j], i))
            i = j + 1
            continue
        for m in MULTI:
            if src.startswith(m, i):
                toks.append(Tok("op", ALIAS.get(m, m), i))
                i += len(m)
                break
        else:
            if c in SINGLE:
                toks.append(Tok("op", ALIAS.get(c, c), i))
                i += 1
            else:
                raise SpecError(f"unexpected character {c!r}", i, "see the notation table in the guide")
    toks.append(Tok("end", None, n))
    return toks


# ============================================================================ parser
CMP_OPS = ("=", "!=", "<", "<=", ">", ">=", "in", "notin", "subseteq", "subset")
SET_OPS = ("union", "inter", "\\", "++", "<|", "<<|", "|>", "|>>", "cross", "..")
BUILTINS = {"card", "dom", "ran", "min", "max", "sum", "abs", "inverse", "id", "comp", "head", "tail",
            "last", "front", "append", "concat", "elems", "first", "second", "isfun"}


class Parser:
    def __init__(self, src):
        self.src = src
        self.toks = tokenize(src)
        self.i = 0

    def peek(self, k=0):
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def at(self, *vals, k=0):
        t = self.peek(k)
        return t.kind == "op" and t.val in vals

    def next(self):
        t = self.peek()
        self.i = min(self.i + 1, len(self.toks) - 1)
        return t

    def expect(self, val, hint=""):
        if not self.at(val):
            t = self.peek()
            raise SpecError(f"expected '{val}' but found {t.val if t.kind != 'end' else 'end of expression'!s}",
                            t.pos, hint)
        return self.next()

    def parse(self):
        e = self.expr()
        if self.peek().kind != "end":
            t = self.peek()
            raise SpecError(f"unexpected {t.val!r} after a complete expression", t.pos,
                            "check operators and parentheses; comparisons cannot be chained (use 'and')")
        return e

    # precedence, lowest first
    def expr(self):
        a = self.implies()
        while self.at("<=>"):
            self.next()
            a = ("bin", "<=>", a, self.implies())
        return a

    def implies(self):
        a = self.or_()
        if self.at("=>"):
            self.next()
            return ("bin", "=>", a, self.implies())
        return a

    def or_(self):
        a = self.and_()
        while self.at("or"):
            self.next()
            a = ("bin", "or", a, self.and_())
        return a

    def and_(self):
        a = self.not_()
        while self.at("and"):
            self.next()
            a = ("bin", "and", a, self.not_())
        return a

    def not_(self):
        if self.at("not") and not self.at("in", k=1):
            self.next()
            return ("un", "not", self.not_())
        return self.cmp()

    def cmp(self):
        a = self.maplet()
        op = None
        if self.at("not") and self.at("in", k=1):
            self.next(), self.next()
            op = "notin"
        elif self.at(*CMP_OPS):
            op = self.next().val
        if op is None:
            return a
        b = self.maplet()
        if self.at(*CMP_OPS):
            raise SpecError("comparisons cannot be chained", self.peek().pos, "write a < b and b < c")
        return ("bin", op, a, b)

    def maplet(self):
        a = self.setop()
        if self.at("|->"):
            self.next()
            return ("tuple", [a, self.setop()])
        return a

    def setop(self):
        a = self.add()
        while self.at(*SET_OPS):
            op = self.next().val
            a = ("bin", op, a, self.add())
        return a

    def add(self):
        a = self.mul()
        while self.at("+", "-"):
            op = self.next().val
            a = ("bin", op, a, self.mul())
        return a

    def mul(self):
        a = self.unary()
        while self.at("*", "/", "div", "mod", "%"):
            op = self.next().val
            a = ("bin", "mod" if op == "%" else op, a, self.unary())
        return a

    def unary(self):
        if self.at("-"):
            self.next()
            return ("un", "neg", self.unary())
        if self.at("#"):
            self.next()
            return ("un", "card", self.unary())
        return self.postfix()

    def postfix(self):
        p = self.primary()
        while True:
            if self.at("("):
                self.next()
                args = [] if self.at(")") else self.exprlist()
                self.expect(")", "close the argument list")
                p = ("call", p, args)
            elif self.at("["):
                self.next()
                s = self.expr()
                self.expect("]", "relational image is r[S]")
                p = ("image", p, s)
            elif self.at("~"):
                self.next()
                p = ("un", "inv", p)
            else:
                return p

    def exprlist(self):
        out = [self.expr()]
        while self.at(","):
            self.next()
            out.append(self.expr())
        return out

    def binders(self):
        """x in S | x, y in S | x : S, ...  -> [(name, setexpr)]"""
        out = []
        while True:
            names = []
            t = self.peek()
            if t.kind != "name":
                raise SpecError("expected a bound variable name", t.pos, "forall x in S : P")
            names.append(self.next().val)
            while self.at(",") and self.peek(1).kind == "name":
                self.next()
                names.append(self.next().val)
            if not self.at("in", ":"):
                raise SpecError("expected 'in' after the bound variable(s)", self.peek().pos, "forall x in S : P")
            self.next()
            s = self.setop()
            out += [(nm, s) for nm in names]
            if self.at(",") and self.peek(1).kind == "name" and (self.at("in", ":", k=2) or self.at(",", k=2)):
                self.next()
                continue
            return out

    def primary(self):
        t = self.peek()
        if t.kind == "num":
            self.next()
            return ("num", t.val)
        if t.kind == "str":
            self.next()
            return ("str", t.val)
        if t.kind == "name":
            self.next()
            return ("name", t.val, False)
        if t.kind == "pname":
            self.next()
            return ("name", t.val, True)
        if t.kind == "op":
            v = t.val
            if v in ("true", "false"):
                self.next()
                return ("bool", v == "true")
            if v == "(":
                self.next()
                e = self.expr()
                if self.at(","):
                    items = [e]
                    while self.at(","):
                        self.next()
                        items.append(self.expr())
                    self.expect(")", "close the tuple")
                    return ("tuple", items)
                self.expect(")", "unbalanced parentheses")
                return e
            if v == "{":
                return self.braces()
            if v == "[":
                self.next()
                items = [] if self.at("]") else self.exprlist()
                self.expect("]", "close the sequence literal")
                return ("seq", items)
            if v in ("forall", "exists") or (v == "sum" and not self.at("(", k=1)):
                self.next()
                bs = self.binders()
                if not self.at(":", ".", "@"):
                    raise SpecError(f"expected ':' after the {v} binders", self.peek().pos, f"{v} x in S : P")
                self.next()
                return ("quant", v, bs, self.expr())
            if v == "sum":
                self.next()
                return ("name", "sum", False)
            if v == "if":
                self.next()
                c = self.expr()
                self.expect("then", "if C then A else B")
                a = self.expr()
                self.expect("else", "if C then A else B (else is required)")
                return ("if", c, a, self.expr())
        if t.kind == "end":
            raise SpecError("unexpected end of expression", t.pos, "an operand is missing")
        raise SpecError(f"unexpected {t.val!r}", t.pos, "an operand was expected here")

    def braces(self):
        self.expect("{")
        if self.at("}"):
            self.next()
            return ("set", [])
        save = self.i
        try:
            bs = self.binders()
            if self.at("|", "@"):
                pred, mapping = ("bool", True), None
                if self.at("|"):
                    self.next()
                    pred = self.expr()
                if self.at("@"):
                    self.next()
                    mapping = self.expr()
                self.expect("}", "comprehension is {x in S | P} or {x in S | P @ e}")
                return ("compr", bs, pred, mapping)
        except SpecError:
            pass
        self.i = save
        items = self.exprlist()
        self.expect("}", "close the set literal")
        return ("set", items)


def parse(src):
    return Parser(src).parse()


# ---------------------------------------------------------------------------- AST helpers
def free_names(node, bound=frozenset()):
    """Set of (name, primed) referenced freely."""
    out = set()

    def walk(n, b):
        k = n[0]
        if k == "name":
            if n[2] or n[1] not in b:
                out.add((n[1], n[2]))
        elif k in ("num", "str", "bool"):
            pass
        elif k in ("set", "seq", "tuple"):
            for x in n[1]:
                walk(x, b)
        elif k == "bin":
            walk(n[2], b), walk(n[3], b)
        elif k == "un":
            walk(n[2], b)
        elif k == "call":
            if not (n[1][0] == "name" and n[1][1] in BUILTINS and not n[1][2]):
                walk(n[1], b)
            for x in n[2]:
                walk(x, b)
        elif k == "image":
            walk(n[1], b), walk(n[2], b)
        elif k == "if":
            walk(n[1], b), walk(n[2], b), walk(n[3], b)
        elif k in ("quant", "compr"):
            bs = n[2] if k == "quant" else n[1]
            nb = set(b)
            for nm, s in bs:
                walk(s, frozenset(nb))
                nb.add(nm)
            nb = frozenset(nb)
            if k == "quant":
                walk(n[3], nb)
            else:
                walk(n[2], nb)
                if n[3] is not None:
                    walk(n[3], nb)

    walk(node, frozenset(bound))
    return out


def unparse(n):
    k = n[0]
    if k == "num":
        return fmt(n[1])
    if k == "str":
        return json.dumps(n[1])
    if k == "bool":
        return "true" if n[1] else "false"
    if k == "name":
        return n[1] + ("'" if n[2] else "")
    if k == "set":
        return "{" + ", ".join(map(unparse, n[1])) + "}"
    if k == "seq":
        return "[" + ", ".join(map(unparse, n[1])) + "]"
    if k == "tuple":
        return "(" + ", ".join(map(unparse, n[1])) + ")"
    if k == "bin":
        return f"({unparse(n[2])} {n[1]} {unparse(n[3])})"
    if k == "un":
        return {"not": "not ", "neg": "-", "card": "#"}.get(n[1], "") + unparse(n[2]) + ("~" if n[1] == "inv" else "")
    if k == "call":
        return unparse(n[1]) + "(" + ", ".join(map(unparse, n[2])) + ")"
    if k == "image":
        return f"{unparse(n[1])}[{unparse(n[2])}]"
    if k == "if":
        return f"(if {unparse(n[1])} then {unparse(n[2])} else {unparse(n[3])})"
    bs = ", ".join(f"{a} in {unparse(s)}" for a, s in (n[2] if k == "quant" else n[1]))
    if k == "quant":
        return f"({n[1]} {bs} : {unparse(n[3])})"
    return "{" + bs + " | " + unparse(n[2]) + (f" @ {unparse(n[3])}" if n[3] is not None else "") + "}"


# ============================================================================ types
def parse_type(src, consts):
    """Type grammar:
       T := B | B -> T (total function) | B +-> T (partial function) | B <-> T (relation) | B cross T
       B := Int | Nat | Bool | Name (given set) | lo..hi | set of B | P B | seq of B | (T)"""
    p = Parser(src)

    def bound():
        neg = False
        if p.at("-"):
            p.next()
            neg = True
        t = p.next()
        if t.kind == "num" and isinstance(t.val, int):
            v = t.val
        elif t.kind == "name" and isinstance(consts.get(t.val), int) and not isinstance(consts.get(t.val), bool):
            v = consts[t.val]
        else:
            raise SpecError(f"range bound must be an integer or integer constant, got {t.val!r}", t.pos,
                            "e.g. 0..10 or 0..capacity")
        return -v if neg else v

    def base():
        t = p.peek()
        if t.kind in ("num",) or p.at("-"):
            lo = bound()
            p.expect("..", "an integer range is lo..hi")
            return ("range", lo, bound())
        if t.kind == "name":
            w = t.val
            if w in ("set", "seq") or (w == "P" and p.peek(1).kind in ("name", "num") or (w == "P" and p.at("(", k=1))):
                p.next()
                if p.peek().kind == "name" and p.peek().val == "of":
                    p.next()
                return ("set" if w in ("set", "P") else "seq", base())
            p.next()
            if w in ("Int", "Nat", "Bool"):
                return (w.lower(),)
            if p.at(".."):
                p.i -= 1
                lo = bound()
                p.next()
                return ("range", lo, bound())
            return ("given", w)
        if p.at("("):
            p.next()
            ty = full()
            p.expect(")", "unbalanced parentheses in type")
            return ty
        raise SpecError(f"unexpected {t.val!r} in type", t.pos,
                        "types: Int, Nat, Bool, a given set, lo..hi, set of T, seq of T, A -> B, A +-> B, A <-> B")

    def full():
        a = base()
        if p.at("->", "+->", "<->", "cross"):
            op = p.next().val
            b = full()
            return {"->": ("fun", a, b, True), "+->": ("fun", a, b, False), "<->": ("rel", a, b),
                    "cross": ("cross", a, b)}[op]
        return a

    ty = full()
    if p.peek().kind != "end":
        raise SpecError(f"unexpected {p.peek().val!r} after type", p.peek().pos, "parenthesise compound types")
    return ty


def type_str(t):
    k = t[0]
    if k in ("int", "nat", "bool"):
        return k.capitalize()
    if k == "given":
        return t[1]
    if k == "range":
        return f"{t[1]}..{t[2]}"
    if k in ("set", "seq"):
        return f"{k} of {_tparen(t[1])}"
    if k == "fun":
        return f"{_tparen(t[1])} {'->' if t[3] else '+->'} {type_str(t[2])}"
    if k == "rel":
        return f"{_tparen(t[1])} <-> {type_str(t[2])}"
    return f"{_tparen(t[1])} cross {type_str(t[2])}"


def _tparen(t):
    s = type_str(t)
    return f"({s})" if t[0] in ("fun", "rel", "cross") else s


def type_names(t):
    k = t[0]
    if k == "given":
        return {t[1]}
    out = set()
    for x in t[1:]:
        if isinstance(x, tuple):
            out |= type_names(x)
    return out


# ============================================================================ compiled spec
def _prov(entry):
    """Return a warning string if provenance is missing/invalid, else None."""
    p = entry.get("provenance") if isinstance(entry, dict) else None
    if p is None:
        return "no provenance"
    if isinstance(p, str):
        kind = p.split(":")[0].strip().lower()
    elif isinstance(p, dict):
        kind = str(p.get("kind", "")).lower()
    else:
        return "provenance must be {kind, quote}"
    if kind not in PROV_KINDS:
        return f"provenance kind must be one of {'/'.join(PROV_KINDS)}"
    return None


def _as_list(x, keyname):
    """Accept a list of objects or a {name: object|value} map."""
    if x is None:
        return []
    if isinstance(x, dict):
        out = []
        for k, v in x.items():
            out.append(dict(v, **{keyname: k}) if isinstance(v, dict) else {keyname: k, "_v": v})
        return out
    return x if isinstance(x, list) else None


class Op:
    def __init__(self, name, inputs, pre, post, src):
        self.name, self.inputs, self.pre, self.post, self.src = name, inputs, pre, post, src


class Spec:
    def __init__(self, rep):
        self.holes, self.warnings = [], []
        self.sets, self.consts, self.atoms = {}, {}, {}
        self.state_types, self.state_order = {}, []
        self.derived = {}             # name -> (params, ast, src)
        self.invariants = []          # (label, ast, src)
        self.ops = {}
        self.init_states = []
        self.init_notes = []
        self.scenarios = {}
        self.work = [0]
        rep = rep if isinstance(rep, dict) else {}
        self.rep = rep
        try:
            self._build(rep)
        except Exception as e:     # never let compilation crash the arm
            self.hole("specification", f"internal {type(e).__name__}: {e}", "simplify the failing element")

    def hole(self, where, error, hint=""):
        self.holes.append({"where": where, "error": error, "hint": hint})

    def env(self, vals=None, inputs=None, nxt=None):
        return Env(self, vals or {}, inputs or {}, nxt, {})

    # ------------------------------------------------------------------ build
    def _build(self, rep):
        self._given(rep.get("given", rep.get("givens")))
        self._state(rep.get("state"))
        self._derived(rep.get("derived"))
        self._init(rep.get("init"))
        self._recheck_derived()
        self._invariants(rep.get("invariant", rep.get("invariants")))
        self._operations(rep.get("operations", rep.get("ops")))
        sc = _as_list(rep.get("scenarios"), "name") or []
        for i, s in enumerate(sc):
            if isinstance(s, dict) and s.get("name"):
                self.scenarios[str(s["name"])] = s

    def known_names(self):
        return (set(self.state_types) | set(self.consts) | set(self.sets) | set(self.derived) | set(self.atoms)
                | {"Bool"} | BUILTINS)

    def _given(self, given):
        lst = _as_list(given, "name")
        if lst is None:
            return self.hole("given", "must be a list of {name, instances|range|value}", "")
        for i, g in enumerate(lst):
            where = f"given[{i}]"
            if not isinstance(g, dict) or not isinstance(g.get("name"), str):
                self.hole(where, "must be an object with a 'name'", '{"name": "Worker", "instances": ["w1"]}')
                continue
            name = g["name"]
            where = f"given '{name}'"
            if not _valid_name(name):
                self.hole(where, "invalid name", "letters, digits, '_'; not a keyword")
                continue
            if name in self.sets or name in self.consts:
                self.hole(where, "defined twice; later definition ignored", "rename one")
                continue
            w = _prov(g)
            if w:
                self.warnings.append(f"{where}: {w}")
            try:
                if "instances" in g:
                    inst = g["instances"]
                    if not isinstance(inst, list) or not all(isinstance(a, str) and _valid_name(a) for a in inst):
                        raise SpecError("instances must be a list of identifier strings", None,
                                        '["w1", "w2"] (letters, digits, _)')
                    for a in inst:
                        if a in self.atoms and self.atoms[a] != name:
                            self.warnings.append(f"{where}: atom '{a}' is also in '{self.atoms[a]}'")
                        self.atoms.setdefault(a, name)
                    self.sets[name] = frozenset(inst)
                elif "range" in g:
                    r = g["range"]
                    if isinstance(r, str):
                        ty = parse_type(r, self.consts)
                        if ty[0] != "range":
                            raise SpecError("range must be [lo, hi] or 'lo..hi'")
                        r = [ty[1], ty[2]]
                    lo, hi = r
                    lo, hi = (self.consts.get(lo, lo), self.consts.get(hi, hi))
                    if not (isinstance(lo, int) and isinstance(hi, int)) or hi - lo > MAX_RANGE:
                        raise SpecError("range must be two integers (hi - lo <= 100000)", None, "[0, 10]")
                    self.sets[name] = frozenset(range(lo, hi + 1))
                elif "value" in g or "_v" in g:
                    v = g.get("value", g.get("_v"))
                    if isinstance(v, str):
                        val = self._eval_static(v)
                    else:
                        val = _from_json(v)
                    self.consts[name] = val
                else:
                    raise SpecError("needs 'instances', 'range' or 'value'", None,
                                    '{"name": "cap", "value": 3}')
            except SpecError as e:
                self.hole(where, e.text(g.get("value") if isinstance(g.get("value"), str) else None), "")
            except EvalError as e:
                self.hole(where, e.text())

    def _eval_static(self, src):
        ast = parse(src)
        bad = [n for n, p in free_names(ast) if p or n not in self.known_names() - set(self.state_types)]
        if bad:
            raise SpecError(f"unknown name '{bad[0]}'", None, "constants may only use earlier given sets/constants")
        return evaluate(ast, self.env())

    def _type(self, src, where):
        try:
            ty = parse_type(src, self.consts)
        except SpecError as e:
            self.hole(where, "bad type: " + e.text(src),
                      "Int, Nat, Bool, GivenSet, lo..hi, set of T, seq of T, A -> B, A +-> B, A <-> B")
            return None
        missing = [n for n in type_names(ty) if n not in self.sets]
        if missing:
            self.hole(where, f"unknown given set '{missing[0]}' in type",
                      "define it with define_given first" + _close(missing[0], self.sets))
            return None
        return ty

    def _state(self, state):
        lst = _as_list(state, "name")
        if lst is None:
            return self.hole("state", "must be a list of {name, type}", "")
        for i, s in enumerate(lst):
            if isinstance(s, dict) and "_v" in s:
                s = {"name": s["name"], "type": s["_v"]}
            where = f"state[{i}]"
            if not isinstance(s, dict) or not isinstance(s.get("name"), str):
                self.hole(where, "must be an object {name, type}", "")
                continue
            name = s["name"]
            where = f"state '{name}'"
            if not _valid_name(name) or name in BUILTINS:
                self.hole(where, "invalid name", "letters, digits, '_'; not a keyword or built-in")
                continue
            if name in self.state_types or name in self.consts or name in self.sets:
                self.hole(where, "name already used", "rename")
                continue
            if name in self.atoms:
                self.warnings.append(f"{where}: also an atom name; the variable shadows the atom")
            ty = self._type(s.get("type"), where) if isinstance(s.get("type"), str) else None
            if ty is None:
                if not isinstance(s.get("type"), str):
                    self.hole(where, "missing 'type'", "e.g. 'set of Worker' or 'Worker +-> Job'")
                continue
            self.state_types[name] = ty
            self.state_order.append(name)

    def _derived(self, derived):
        lst = _as_list(derived, "name")
        if lst is None:
            return self.hole("derived", "must be a list of {name, params?, expr}", "")
        for i, d in enumerate(lst):
            if isinstance(d, dict) and "_v" in d:
                d = {"name": d["name"], "expr": d["_v"]}
            where = f"derived[{i}]"
            if not isinstance(d, dict) or not isinstance(d.get("name"), str):
                self.hole(where, "must be an object {name, expr}", "")
                continue
            name = d["name"]
            where = f"derived '{name}'"
            params = d.get("params") or []
            if not _valid_name(name) or name in self.known_names() - set(self.atoms):
                self.hole(where, "invalid or already used name", "rename")
                continue
            if not isinstance(params, list) or not all(isinstance(p, str) and _valid_name(p) for p in params):
                self.hole(where, "params must be a list of names", '["w"]')
                continue
            src = d.get("expr")
            try:
                ast = parse(src)
            except SpecError as e:
                self.hole(where, e.text(src if isinstance(src, str) else None))
                continue
            self.derived[name] = (params, ast, src)

    def _recheck_derived(self):
        changed = True
        while changed:
            changed = False
            for name, (params, ast, src) in list(self.derived.items()):
                err = self._check_names(ast, set(params), allow_primed=False)
                if err:
                    del self.derived[name]
                    self.hole(f"derived '{name}'", err[0], err[1])
                    changed = True

    def _check_names(self, ast, extra, allow_primed):
        known = self.known_names() | extra
        for n, primed in sorted(free_names(ast)):
            if primed:
                if not allow_primed:
                    return (f"primed name {n}' is only allowed in a postcondition", "remove the prime")
                if n not in self.state_types and not (n in self.derived and not self.derived[n][0]):
                    return (f"{n}' is not a state variable", "only state variables (or derived values) can be "
                            "primed" + _close(n, self.state_types))
            elif n not in known:
                return (f"unknown name '{n}'", "declare it (state, given, derived or input)" + _close(n, known))
        return None

    def _init(self, init):
        if isinstance(init, str):
            init = [{"predicate": init}]
        elif isinstance(init, dict) and not ("var" in init or "predicate" in init):
            init = [{"var": k, "value": v} for k, v in init.items()]
        elif isinstance(init, dict):
            init = [init]
        if init is None:
            init = []
        if not isinstance(init, list):
            return self.hole("init", "must be a list of {var, value} / {predicate}", "")
        assigns, preds = [], []
        for i, e in enumerate(init):
            where = f"init[{i}]"
            if isinstance(e, str):
                e = {"predicate": e}
            if not isinstance(e, dict):
                self.hole(where, "must be an object", '{"var": "x", "value": "0", "provenance": {...}}')
                continue
            w = _prov(e)
            if w:
                self.warnings.append(f"{where}: {w}")
            try:
                if "var" in e:
                    v = e["var"]
                    where = f"init '{v}'"
                    if v not in self.state_types:
                        raise SpecError(f"'{v}' is not a state variable", None, _close(v, self.state_types).strip(" ;"))
                    val = e.get("value")
                    ast = parse(val) if isinstance(val, str) else ("lit", _from_json(val))
                    if ast[0] != "lit":
                        err = self._check_names(ast, set(), False)
                        if err:
                            raise SpecError(err[0], None, err[1])
                    assigns.append((v, ast, where))
                elif "predicate" in e:
                    ast = parse(e["predicate"])
                    err = self._check_names(ast, set(), False)
                    if err:
                        raise SpecError(err[0], None, err[1])
                    preds.append((ast, where))
                else:
                    raise SpecError("needs 'var'+'value' or 'predicate'")
            except SpecError as ex:
                src = e.get("value", e.get("predicate"))
                self.hole(where, ex.text(src if isinstance(src, str) else None))
            except EvalError as ex:
                self.hole(where, ex.text())
        # assignments, in dependency order
        vals = {}
        pending = list(assigns)
        while pending:
            progress = False
            for item in list(pending):
                v, ast, where = item
                deps = set() if ast[0] == "lit" else {n for n, p in free_names(ast) if n in self.state_types}
                if deps - set(vals):
                    continue
                pending.remove(item)
                progress = True
                try:
                    vals[v] = ast[1] if ast[0] == "lit" else evaluate(ast, self.env(vals))
                except EvalError as ex:
                    self.hole(where, ex.text())
            if not progress:
                for v, ast, where in pending:
                    self.hole(where, "depends on variables that have no initial value (or on each other)",
                              "give those variables values first")
                break
        unknown = [v for v in self.state_order if v not in vals]
        states = [dict(vals)]
        if unknown:
            mentioned = set()
            for ast, _ in preds:
                mentioned |= {n for n, p in free_names(ast) if not p}
            for v in list(unknown):
                if v not in mentioned and carrier(self.state_types[v], self) is None:
                    self.hole(f"state '{v}'", "has no initial value and its type is too large to enumerate",
                              "add set_init(var, value) or an init predicate 'v = ...'")
                    self.state_types.pop(v, None)
                    self.state_order.remove(v)
                    unknown.remove(v)
        if unknown:
            pred = None
            for ast, _ in preds:
                pred = ast if pred is None else ("bin", "and", pred, ast)
            if pred is None:
                pred = ("bool", True)
            res, errs, notes = solve(self, pred, self.env(vals), set(unknown), primed=False, cap=MAX_COMBOS)
            for er in errs[:5]:
                self.hole("init", er, "set every variable's initial value with set_init(var, value)")
            states = [dict(vals, **r) for r in res]
            self.init_notes += notes
        else:
            for ast, where in preds:
                try:
                    ok = evaluate(ast, self.env(vals))
                    if ok is not True:
                        self.hole(where, "init predicate is false for the assigned initial values", "")
                        states = []
                except EvalError as ex:
                    self.hole(where, ex.text())
        uniq = {}
        for s in states:
            uniq.setdefault(self.key(s), s)
        self.init_states = list(uniq.values())[:MAX_INIT]
        if not self.init_states and self.state_order:
            self.hole("init", "no initial state satisfies init", "check the init predicates")

    def _invariants(self, invs):
        if isinstance(invs, str):
            invs = [invs]
        lst = _as_list(invs, "name")
        if lst is None:
            return self.hole("invariant", "must be a list of expressions or {name, expr}", "")
        for i, inv in enumerate(lst):
            if isinstance(inv, str):
                inv = {"expr": inv}
            if isinstance(inv, dict) and "_v" in inv:
                inv = {"name": inv["name"], "expr": inv["_v"]}
            label = (inv.get("name") if isinstance(inv, dict) else None) or f"inv{i + 1}"
            where = f"invariant '{label}'"
            if not isinstance(inv, dict):
                self.hole(where, "must be a string or {name, expr}")
                continue
            src = inv.get("expr", inv.get("predicate"))
            try:
                ast = parse(src)
                err = self._check_names(ast, set(), False)
                if err:
                    raise SpecError(err[0], None, err[1])
            except SpecError as e:
                self.hole(where, e.text(src if isinstance(src, str) else None))
                continue
            self.invariants.append((label, ast, src))

    def _operations(self, ops):
        lst = _as_list(ops, "name")
        if lst is None:
            return self.hole("operations", "must be a list of {name, inputs, pre, post}", "")
        for i, o in enumerate(lst):
            where = f"operations[{i}]"
            if not isinstance(o, dict) or not isinstance(o.get("name"), str):
                self.hole(where, "must be an object with a 'name'", "")
                continue
            name = o["name"]
            where = f"operation '{name}'"
            if name in self.ops:
                self.hole(where, "defined twice; later definition ignored", "rename one")
                continue
            w = _prov(o)
            if w:
                self.warnings.append(f"{where}: {w}")
            inputs = o.get("inputs") or {}
            if isinstance(inputs, list):
                try:
                    inputs = {d["name"]: d["type"] for d in inputs}
                except (TypeError, KeyError):
                    self.hole(where, "inputs must be {name: type} or [{name, type}]", '{"w": "Worker"}')
                    continue
            if not isinstance(inputs, dict):
                self.hole(where, "inputs must be {name: type}", '{"w": "Worker"}')
                continue
            ins, ok = [], True
            for nm, ts in inputs.items():
                if not _valid_name(nm) or nm in self.state_types:
                    self.hole(where, f"input '{nm}' has an invalid name or clashes with a state variable", "rename")
                    ok = False
                    break
                ty = self._type(ts, f"{where} input '{nm}'") if isinstance(ts, str) else None
                if ty is None:
                    if not isinstance(ts, str):
                        self.hole(where, f"input '{nm}' needs a type string", '"Worker"')
                    ok = False
                    break
                ins.append((nm, ty))
            if not ok:
                continue
            asts = {}
            for part in ("pre", "post"):
                src = o.get(part, "true" if part == "pre" else None)
                if part == "post" and src is None:
                    self.hole(where, "missing 'post'", "post relates x and x', e.g. \"x' = x + 1\"")
                    ok = False
                    break
                try:
                    ast = parse(src) if not isinstance(src, bool) else ("bool", src)
                    err = self._check_names(ast, {n for n, _ in ins}, allow_primed=(part == "post"))
                    if err:
                        raise SpecError(err[0], None, err[1])
                    asts[part] = ast
                except SpecError as e:
                    self.hole(f"{where} {part}", e.text(src if isinstance(src, str) else None))
                    ok = False
                    break
            if ok:
                self.ops[name] = Op(name, ins, asts["pre"], asts["post"], o)

    # ------------------------------------------------------------------ runtime helpers
    def key(self, state):
        return tuple(state.get(v) for v in self.state_order)

    def type_errors(self, state):
        out = []
        for v in self.state_order:
            if v in state:
                e = check_type(state[v], self.state_types[v], self)
                if e:
                    out.append(f"type of '{v}' ({type_str(self.state_types[v])}): {e}")
        return out

    def violations(self, state):
        """[(label, message)] for false/unevaluable invariants and type violations."""
        out = [("type", m) for m in self.type_errors(state)]
        env = self.env(state)
        for label, ast, src in self.invariants:
            try:
                v = evaluate(ast, env)
                if v is not True:
                    out.append((label, f"invariant '{label}' is false: {src}"))
            except EvalError as e:
                out.append((label, f"invariant '{label}' could not be evaluated: {e.text()}"))
        return out

    def input_combos(self, op, given=None):
        """Yield input dicts; given inputs are fixed, others enumerated over their types."""
        given = given or {}
        names, choices = [], []
        for nm, ty in op.inputs:
            names.append(nm)
            if nm in given:
                choices.append([given[nm]])
            else:
                c = carrier(ty, self)
                if c is None:
                    raise EvalError(f"input '{nm}' of {op.name} has an unbounded or large type "
                                    f"({type_str(ty)}) and no value was given",
                                    "give the input a value, or a finite type such as 0..10")
                choices.append(c)
        n = 1
        for c in choices:
            n *= max(1, len(c))
        if n > MAX_COMBOS:
            raise EvalError(f"{op.name} has {n} input combinations", "narrow the input types")
        for combo in itertools.product(*choices):
            yield dict(zip(names, combo))

    def step(self, op, state, inputs):
        """-> (enabled, [next states], [errors], [notes])"""
        env = self.env(state, inputs)
        for nm, ty in op.inputs:
            e = check_type(inputs.get(nm), ty, self) if nm in inputs else "missing"
            if e:
                return False, [], [f"input '{nm}' = {fmt(inputs.get(nm)) if nm in inputs else '?'}: {e}"], []
        try:
            pre = evaluate(op.pre, env)
        except EvalError as e:
            return False, [], [f"{op.name} pre: {e.text()}"], []
        if pre is not True:
            if pre is not False:
                return False, [], [f"{op.name} pre is not a boolean ({fmt(pre)})"], []
            return False, [], [], []
        res, errs, notes = solve(self, op.post, env, set(self.state_order), primed=True, cap=MAX_COMBOS)
        uniq = {}
        for r in res:
            s = dict(state, **r)
            uniq.setdefault(self.key(s), s)
        return True, list(uniq.values()), [f"{op.name} post: {e}" for e in errs], notes


def _valid_name(s):
    return isinstance(s, str) and bool(s) and (s[0].isalpha() or s[0] == "_") and \
        all(c.isalnum() or c == "_" for c in s) and s not in KEYWORDS


def _close(name, pool):
    m = difflib.get_close_matches(str(name), [str(p) for p in pool], n=3)
    return ("; did you mean " + ", ".join(m) + "?") if m else ""


# ============================================================================ type checking / carriers
def check_type(v, t, spec):
    k = t[0]
    if k == "int":
        return None if isinstance(v, int) and not isinstance(v, bool) else f"{fmt(v)} is not an integer"
    if k == "nat":
        return None if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else f"{fmt(v)} is not a natural"
    if k == "bool":
        return None if isinstance(v, bool) else f"{fmt(v)} is not a boolean"
    if k == "range":
        ok = isinstance(v, int) and not isinstance(v, bool) and t[1] <= v <= t[2]
        return None if ok else f"{fmt(v)} is not in {t[1]}..{t[2]}"
    if k == "given":
        return None if (not isinstance(v, bool)) and v in spec.sets.get(t[1], ()) else f"{fmt(v)} is not in {t[1]}"
    if k == "set":
        if not isinstance(v, frozenset):
            return f"{fmt(v)} is not a set"
        for x in v:
            e = check_type(x, t[1], spec)
            if e:
                return e
        return None
    if k == "seq":
        if not isinstance(v, Seq):
            return f"{fmt(v)} is not a sequence"
        for x in v:
            e = check_type(x, t[1], spec)
            if e:
                return e
        return None
    if k == "cross":
        if not (isinstance(v, tuple) and not isinstance(v, Seq) and len(v) == 2):
            return f"{fmt(v)} is not a pair"
        return check_type(v[0], t[1], spec) or check_type(v[1], t[2], spec)
    if k in ("rel", "fun"):
        if not isinstance(v, frozenset):
            return f"{fmt(v)} is not a relation/function"
        dom = {}
        for x in v:
            if not (isinstance(x, tuple) and not isinstance(x, Seq) and len(x) == 2):
                return f"{fmt(x)} is not a pair"
            e = check_type(x[0], t[1], spec) or check_type(x[1], t[2], spec)
            if e:
                return e
            if k == "fun":
                if x[0] in dom:
                    return f"maps {fmt(x[0])} to more than one value"
                dom[x[0]] = 1
        if k == "fun" and t[3]:
            c = carrier(t[1], spec)
            if c is not None and len(dom) != len(c):
                missing = [fmt(a) for a in c if a not in dom][:3]
                return f"total function is undefined at {', '.join(missing)}"
        return None
    return None


def carrier(t, spec, limit=MAX_CARRIER):
    """Finite list of all values of type t, or None if infinite / larger than limit."""
    k = t[0]
    if k == "bool":
        return [False, True]
    if k in ("int", "nat", "seq"):
        return None
    if k == "range":
        return list(range(t[1], t[2] + 1)) if t[2] - t[1] < limit else None
    if k == "given":
        return sorted(spec.sets.get(t[1], ()), key=_sort_key)
    if k == "cross":
        a, b = carrier(t[1], spec, limit), carrier(t[2], spec, limit)
        if a is None or b is None or len(a) * len(b) > limit:
            return None
        return [(x, y) for x in a for y in b]
    if k == "set":
        base = carrier(t[1], spec, limit)
        return _powerset(base, limit)
    if k == "rel":
        pairs = carrier(("cross", t[1], t[2]), spec, limit)
        return _powerset(pairs, limit)
    if k == "fun":
        a, b = carrier(t[1], spec, limit), carrier(t[2], spec, limit)
        if a is None or b is None:
            return None
        opts = len(b) + (0 if t[3] else 1)
        if opts ** len(a) > limit:
            return None
        out = []
        for combo in itertools.product(*[([None] if not t[3] else []) + b for _ in a]):
            out.append(frozenset((x, y) for x, y in zip(a, combo) if y is not None or t[3]))
        return out
    return None


def _powerset(base, limit):
    if base is None or 2 ** len(base) > limit:
        return None
    out = []
    for r in range(len(base) + 1):
        out += [frozenset(c) for c in itertools.combinations(base, r)]
    return out


# ============================================================================ evaluator
class Env:
    __slots__ = ("spec", "vals", "inputs", "nxt", "binds", "depth")

    def __init__(self, spec, vals, inputs, nxt, binds, depth=0):
        self.spec, self.vals, self.inputs, self.nxt, self.binds, self.depth = spec, vals, inputs, nxt, binds, depth

    def child(self, binds):
        return Env(self.spec, self.vals, self.inputs, self.nxt, dict(self.binds, **binds), self.depth)

    def with_next(self, nxt):
        return Env(self.spec, self.vals, self.inputs, nxt, self.binds, self.depth)

    def with_vals(self, vals):
        return Env(self.spec, vals, self.inputs, self.nxt, self.binds, self.depth)

    def lookup(self, name, primed):
        sp = self.spec
        if primed:
            if name in sp.state_types:
                if self.nxt is None:
                    raise EvalError(f"{name}' used outside a postcondition")
                if name not in self.nxt:
                    raise _Missing(f"{name}' is not determined yet")
                return self.nxt[name]
            if name in sp.derived and not sp.derived[name][0]:
                if self.nxt is None or any(v not in self.nxt for v in sp.state_order):
                    raise _Missing(f"{name}' needs the whole next state")
                return self._derived(name, [], Env(sp, self.nxt, {}, None, {}, self.depth))
            raise EvalError(f"{name}' is not a state variable")
        if name in self.binds:
            return self.binds[name]
        if name in self.inputs:
            return self.inputs[name]
        if name in sp.state_types:
            if name not in self.vals:
                raise _Missing(f"state variable '{name}' has no value yet")
            return self.vals[name]
        if name in sp.consts:
            return sp.consts[name]
        if name in sp.sets:
            return sp.sets[name]
        if name in sp.derived:
            if sp.derived[name][0]:
                raise EvalError(f"derived '{name}' needs arguments", f"{name}({', '.join(sp.derived[name][0])})")
            return self._derived(name, [], self)
        if name in sp.atoms:
            return name
        if name == "Bool":
            return frozenset({False, True})
        if name in ("Int", "Nat"):
            raise EvalError(f"{name} is infinite and cannot be used as a set value", "use a range lo..hi")
        raise EvalError(f"unknown name '{name}'", _close(name, sp.known_names()).strip("; "))

    def _derived(self, name, args, base):
        params, ast, src = self.spec.derived[name]
        if len(args) != len(params):
            raise EvalError(f"derived '{name}' takes {len(params)} argument(s), got {len(args)}")
        if self.depth > 60:
            raise EvalError(f"derived '{name}' recursion too deep", "derived definitions must not be circular")
        env = Env(self.spec, base.vals, {}, base.nxt, dict(zip(params, args)), self.depth + 1)
        return evaluate(ast, env)


def _need(cond, msg, hint=""):
    if not cond:
        raise EvalError(msg, hint)


def _as_set(v, what):
    if isinstance(v, Seq):
        raise EvalError(f"{what}: {fmt(v)} is a sequence, not a set", "use elems(s)")
    _need(isinstance(v, frozenset), f"{what}: {fmt(v)} is not a set")
    return v


def _pairs(v, what):
    s = _as_set(v, what)
    for x in s:
        _need(isinstance(x, tuple) and not isinstance(x, Seq) and len(x) == 2,
              f"{what}: {fmt(x)} is not a pair, so {fmt(v)} is not a relation", "write pairs as a |-> b")
    return s


def _num(v, what):
    _need(_is_num(v), f"{what}: {fmt(v)} is not a number")
    return v


def _bool(v, what):
    _need(isinstance(v, bool), f"{what}: {fmt(v)} is not true/false")
    return v


def apply_value(f, args, what):
    if isinstance(f, Seq):
        _need(len(args) == 1 and isinstance(args[0], int) and not isinstance(args[0], bool),
              f"{what}: a sequence is indexed by one integer")
        i = args[0]
        _need(1 <= i <= len(f), f"{what}: index {i} out of range 1..{len(f)}")
        return f[i - 1]
    if isinstance(f, frozenset):
        key = args[0] if len(args) == 1 else tuple(args)
        hits = [b for (a, b) in _pairs(f, what) if _eq(a, key)]
        _need(hits, f"{what}: not defined at {fmt(key)}", "check the domain with dom(f) or a precondition")
        _need(len(hits) == 1, f"{what}: maps {fmt(key)} to several values ({', '.join(fmt(h) for h in hits)})",
              "it is a relation, not a function: use the image r[{x}]")
        return hits[0]
    raise EvalError(f"{what}: {fmt(f)} cannot be applied", "only functions, relations and sequences take (x)")


def _binder_sets(bs, env):
    vals = []
    for nm, s in bs:
        v = evaluate(s, env)
        if isinstance(v, Seq):
            v = frozenset(v)
        _as_set(v, f"bound variable '{nm}'")
        vals.append((nm, sorted(v, key=_sort_key)))
    return vals


def _iter_bindings(bs, env):
    """Yield dicts of bindings; later sets may depend on earlier variables."""
    def rec(i, b):
        if i == len(bs):
            yield b
            return
        nm, s = bs[i]
        v = evaluate(s, env.child(b))
        if isinstance(v, Seq):
            v = frozenset(v)
        _as_set(v, f"bound variable '{nm}'")
        for x in sorted(v, key=_sort_key):
            yield from rec(i + 1, dict(b, **{nm: x}))
    yield from rec(0, {})


def evaluate(n, env):
    w = env.spec.work
    w[0] += 1
    if w[0] > MAX_WORK:
        raise EvalError("evaluation budget exceeded", "reduce set sizes, depth or quantifier nesting")
    k = n[0]
    if k in ("num", "str", "bool", "lit"):
        return n[1]
    if k == "name":
        return env.lookup(n[1], n[2])
    if k == "set":
        return frozenset(evaluate(x, env) for x in n[1])
    if k == "seq":
        return Seq(evaluate(x, env) for x in n[1])
    if k == "tuple":
        return tuple(evaluate(x, env) for x in n[1])
    if k == "if":
        c = _bool(evaluate(n[1], env), "if condition")
        return evaluate(n[2] if c else n[3], env)
    if k == "un":
        op = n[1]
        v = evaluate(n[2], env)
        if op == "not":
            return not _bool(v, "not")
        if op == "neg":
            return -_num(v, "unary -")
        if op == "card":
            _need(isinstance(v, (frozenset, Seq)), f"#: {fmt(v)} is not a set or sequence")
            return len(v)
        if op == "inv":
            return frozenset((b, a) for a, b in _pairs(v, "inverse ~"))
    if k == "bin":
        return _binop(n, env)
    if k == "call":
        return _call(n, env)
    if k == "image":
        r = _pairs(evaluate(n[1], env), "image r[S]")
        s = evaluate(n[2], env)
        _as_set(s, "image r[S]: S")
        return frozenset(b for a, b in r if a in s)
    if k == "quant":
        kind, bs, body = n[1], n[2], n[3]
        if kind == "sum":
            total = 0
            for b in _iter_bindings(bs, env):
                total += _num(evaluate(body, env.child(b)), "sum body")
            return total
        for b in _iter_bindings(bs, env):
            v = _bool(evaluate(body, env.child(b)), f"{kind} body")
            if kind == "forall" and not v:
                return False
            if kind == "exists" and v:
                return True
        return kind == "forall"
    if k == "compr":
        bs, pred, mapping = n[1], n[2], n[3]
        out = set()
        for b in _iter_bindings(bs, env):
            e2 = env.child(b)
            if _bool(evaluate(pred, e2), "comprehension predicate"):
                if mapping is not None:
                    out.add(evaluate(mapping, e2))
                else:
                    out.add(b[bs[0][0]] if len(bs) == 1 else tuple(b[nm] for nm, _ in bs))
        return frozenset(out)
    raise EvalError(f"cannot evaluate node {k}")


def _binop(n, env):
    op, an, bn = n[1], n[2], n[3]
    if op == "and":
        return _bool(evaluate(an, env), "and") and _bool(evaluate(bn, env), "and")
    if op == "or":
        return _bool(evaluate(an, env), "or") or _bool(evaluate(bn, env), "or")
    if op == "=>":
        return (not _bool(evaluate(an, env), "=>")) or _bool(evaluate(bn, env), "=>")
    a, b = evaluate(an, env), evaluate(bn, env)
    if op == "<=>":
        return _bool(a, "<=>") == _bool(b, "<=>")
    if op == "=":
        return _eq(a, b)
    if op == "!=":
        return not _eq(a, b)
    if op in ("<", "<=", ">", ">="):
        _need((_is_num(a) and _is_num(b)) or (isinstance(a, str) and isinstance(b, str)),
              f"{op}: cannot compare {fmt(a)} and {fmt(b)}", "use subseteq for sets")
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
    if op in ("in", "notin"):
        if isinstance(b, Seq):
            b = frozenset(b)
        _as_set(b, f"{op}: right side")
        r = any(_eq(a, x) for x in b) if isinstance(a, (bool, int)) else a in b
        return r if op == "in" else not r
    if op in ("subseteq", "subset"):
        a, b = _as_set(a, op), _as_set(b, op)
        return a <= b if op == "subseteq" else a < b
    if op in ("+", "-", "*", "/", "div", "mod"):
        if op in ("+", "-") and isinstance(a, frozenset):
            raise EvalError(f"{op} on sets", "use union / \\ for sets")
        _num(a, op), _num(b, op)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        _need(b != 0, f"{op}: division by zero")
        if op == "div":
            return a // b
        if op == "mod":
            return a % b
        return a // b if isinstance(a, int) and isinstance(b, int) and a % b == 0 else a / b
    if op == "..":
        _need(isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool) and not isinstance(b, bool),
              f"..: bounds must be integers ({fmt(a)}, {fmt(b)})")
        _need(b - a <= MAX_RANGE, "..: range too large")
        return frozenset(range(a, b + 1))
    if op in ("union", "inter", "\\"):
        a, b = _as_set(a, op), _as_set(b, op)
        return a | b if op == "union" else (a & b if op == "inter" else a - b)
    if op == "cross":
        a, b = _as_set(a, op), _as_set(b, op)
        _need(len(a) * len(b) <= MAX_RANGE, "cross: product too large")
        return frozenset((x, y) for x in a for y in b)
    if op == "++":
        r, s = _pairs(a, "override ++ (left)"), _pairs(b, "override ++ (right)")
        d = {x for x, _ in s}
        return frozenset(p for p in r if p[0] not in d) | s
    if op in ("<|", "<<|"):
        s, r = _as_set(a, f"{op} (left: a set)"), _pairs(b, f"{op} (right: a relation)")
        return frozenset(p for p in r if (p[0] in s) == (op == "<|"))
    if op in ("|>", "|>>"):
        r, s = _pairs(a, f"{op} (left: a relation)"), _as_set(b, f"{op} (right: a set)")
        return frozenset(p for p in r if (p[1] in s) == (op == "|>"))
    raise EvalError(f"unknown operator {op}")


def _call(n, env):
    fn, argn = n[1], n[2]
    sp = env.spec
    if fn[0] == "name" and not fn[2]:
        name = fn[1]
        shadow = name in env.binds or name in env.inputs or name in sp.state_types or name in sp.consts
        if name in sp.derived and not shadow:
            return env._derived(name, [evaluate(a, env) for a in argn], env)
        if name in BUILTINS and not shadow:
            args = [evaluate(a, env) for a in argn]
            return _builtin(name, args)
    f = evaluate(fn, env)
    args = [evaluate(a, env) for a in argn]
    return apply_value(f, args, unparse(fn))


def _builtin(name, args):
    def one():
        _need(len(args) == 1, f"{name} takes one argument")
        return args[0]

    if name == "card":
        v = one()
        _need(isinstance(v, (frozenset, Seq)), f"card: {fmt(v)} is not a set or sequence")
        return len(v)
    if name == "dom":
        return frozenset(a for a, _ in _pairs(one(), "dom")) if not isinstance(args[0], Seq) else \
            frozenset(range(1, len(args[0]) + 1))
    if name == "ran":
        return frozenset(b for _, b in _pairs(one(), "ran")) if not isinstance(args[0], Seq) else frozenset(args[0])
    if name in ("min", "max"):
        xs = list(args[0]) if len(args) == 1 and isinstance(args[0], (frozenset, Seq)) else args
        _need(xs, f"{name} of an empty collection")
        for x in xs:
            _num(x, name)
        return min(xs) if name == "min" else max(xs)
    if name == "sum":
        v = one()
        _need(isinstance(v, (frozenset, Seq)), "sum(S) needs a set or sequence of numbers",
              "for a sum over elements use: sum x in S : f(x)")
        return sum(_num(x, "sum") for x in v)
    if name == "abs":
        return abs(_num(one(), "abs"))
    if name == "inverse":
        return frozenset((b, a) for a, b in _pairs(one(), "inverse"))
    if name == "id":
        return frozenset((x, x) for x in _as_set(one(), "id"))
    if name == "comp":
        _need(len(args) == 2, "comp(r, s) takes two relations (r then s)")
        r, s = _pairs(args[0], "comp"), _pairs(args[1], "comp")
        return frozenset((a, d) for a, b in r for c, d in s if _eq(b, c))
    if name == "isfun":
        r = _pairs(one(), "isfun")
        return len({a for a, _ in r}) == len(r)
    if name == "elems":
        v = one()
        _need(isinstance(v, Seq), "elems(s) needs a sequence")
        return frozenset(v)
    if name in ("head", "last", "tail", "front"):
        v = one()
        _need(isinstance(v, Seq), f"{name}(s) needs a sequence")
        _need(len(v) > 0, f"{name} of an empty sequence", "guard it with #s > 0")
        return {"head": lambda: v[0], "last": lambda: v[-1], "tail": lambda: Seq(v[1:]),
                "front": lambda: Seq(v[:-1])}[name]()
    if name == "append":
        _need(len(args) == 2 and isinstance(args[0], Seq), "append(s, x) needs a sequence and an element")
        return Seq(args[0] + (args[1],))
    if name == "concat":
        _need(len(args) == 2 and all(isinstance(a, Seq) for a in args), "concat(s, t) needs two sequences")
        return Seq(args[0] + args[1])
    if name in ("first", "second"):
        v = one()
        _need(isinstance(v, tuple) and not isinstance(v, Seq) and len(v) == 2, f"{name}: {fmt(v)} is not a pair")
        return v[0] if name == "first" else v[1]
    raise EvalError(f"unknown function {name}")


# ============================================================================ solving postconditions / init
def solve(spec, pred, env, targets, primed, cap):
    """Find all assignments to the target variables that satisfy pred.

    primed=True: targets are x' (post-state); variables not mentioned in an alternative keep their value.
    primed=False: targets are unassigned unprimed variables (init); all must be determined.
    Returns (list of {var: value} for the touched targets, errors, notes)."""
    memo = {}

    def tnames(node):
        key = id(node)
        if key not in memo:
            memo[key] = (node, {nm for nm, p in free_names(node) if p == primed and nm in targets})
        return memo[key][1]

    def is_t(node):
        return node[0] == "name" and node[2] == primed and node[1] in targets

    errors, notes = [], []
    branches = []

    def expand(node, binds, out):
        if len(out) > MAX_BRANCHES:
            raise EvalError(f"more than {MAX_BRANCHES} alternatives", "simplify the disjunctions")
        k = node[0]
        if not tnames(node):
            return [[node]]
        e = env.child(binds) if binds else env
        if k == "bin" and node[1] == "and":
            left = expand(node[2], binds, out)
            right = expand(node[3], binds, out)
            return [a + b for a in left for b in right]
        if k == "bin" and node[1] == "or":
            return expand(node[2], binds, out) + expand(node[3], binds, out)
        if k == "if" and not tnames(node[1]):
            c = _bool(evaluate(node[1], e), "if condition")
            return expand(node[2] if c else node[3], binds, out)
        if k == "bin" and node[1] == "=>" and not tnames(node[2]):
            c = _bool(evaluate(node[2], e), "=> condition")
            return expand(node[3], binds, out) if c else [[]]
        return [[node]]

    def top(node, binds):
        """Expand into (conjuncts, binds) alternatives, opening existentials that bind choices."""
        res = []
        for conj in expand(node, binds, res):
            res.append((conj, binds))
        # open 'exists' conjuncts that mention targets
        final = []
        for conj, b in res:
            ex = [c for c in conj if c[0] == "quant" and c[1] == "exists" and tnames(c[3])
                  and not any(tnames(s) for _, s in c[2])]
            if not ex:
                final.append((conj, b))
                continue
            q = ex[0]
            rest = [c for c in conj if c is not q]
            for nb in _iter_bindings(q[2], env.child(b)):
                b2 = dict(b, **nb)
                for c2, b3 in top(q[3], b2):
                    final += top_join(rest, c2, b3)
            if len(final) > MAX_BRANCHES:
                raise EvalError(f"more than {MAX_BRANCHES} alternatives", "simplify the existentials")
        return final

    def top_join(rest, conj, binds):
        # re-expand remaining conjuncts in case they contain further existentials
        node = None
        for c in rest:
            node = c if node is None else ("bin", "and", node, c)
        if node is None:
            return [(conj, binds)]
        return [(conj + c, b) for c, b in top(node, binds)]

    try:
        branches = top(pred, {})
    except EvalError as e:
        return [], [e.text()], notes

    results = []
    for bi, (conjs, binds) in enumerate(branches):
        benv = env.child(binds) if binds else env
        mentioned = set()
        for c in conjs:
            mentioned |= tnames(c)
        assigned = {}
        progress = True
        try:
            while progress:
                progress = False
                for c in conjs:
                    if c[0] == "bin" and c[1] == "=":
                        for lhs, rhs in ((c[2], c[3]), (c[3], c[2])):
                            if is_t(lhs) and lhs[1] not in assigned and tnames(rhs) <= set(assigned):
                                try:
                                    val = evaluate(rhs, _partial(benv, assigned, primed))
                                except _Missing:
                                    continue
                                assigned[lhs[1]] = val
                                progress = True
                                break
            open_vars = sorted((mentioned if primed else targets) - set(assigned),
                               key=lambda v: spec.state_order.index(v) if v in spec.state_order else 0)
            choices = []
            for v in open_vars:
                cands = None
                for c in conjs:
                    if c[0] == "bin" and c[1] == "in" and is_t(c[2]) and c[2][1] == v and \
                            tnames(c[3]) <= set(assigned):
                        try:
                            s = evaluate(c[3], _partial(benv, assigned, primed))
                        except _Missing:
                            continue
                        cands = sorted(_as_set(s if not isinstance(s, Seq) else frozenset(s), f"{v}' in"),
                                       key=_sort_key)
                        break
                if cands is None:
                    cands = carrier(spec.state_types[v], spec)
                    if cands is None:
                        raise EvalError(f"{v}{chr(39) if primed else ''} is not determined"
                                        f"{' in alternative ' + str(bi + 1) if len(branches) > 1 else ''} and its type "
                                        f"({type_str(spec.state_types[v])}) is too large to enumerate",
                                        f"add a conjunct {v}{chr(39) if primed else ''} = <expression>")
                    notes.append(f"{v}{chr(39) if primed else ''} enumerated over {len(cands)} values")
                choices.append(cands)
            total = 1
            for c in choices:
                total *= len(c)
            if total > cap:
                raise EvalError(f"{total} candidate states for {', '.join(open_vars)}",
                                "determine those variables with equations")
            for combo in itertools.product(*choices):
                full = dict(assigned, **dict(zip(open_vars, combo)))
                e2 = _partial(benv, full, primed, complete=True)
                if all(_bool(evaluate(c, e2), "postcondition conjunct" if primed else "init predicate") is True
                       for c in conjs):
                    results.append(full)
        except EvalError as e:
            errors.append(e.text())
    return results, errors, notes


def _partial(env, assigned, primed, complete=False):
    if primed:
        nxt = dict(env.vals, **assigned) if complete else dict(assigned)
        return env.with_next(nxt)
    return env.with_vals(dict(env.vals, **assigned))


# ============================================================================ exploration
def explore(spec, depth, target=None, max_states=DEFAULT_MAX_STATES, timeout=DEFAULT_TIMEOUT, stop_on_violation=True):
    """BFS over all enabled operations with all inputs. Returns a result dict."""
    t0 = time.monotonic()
    parent = {}
    states = {}
    res = {"states": 0, "transitions": 0, "depth_done": 0, "branching": 0, "deadlocks": 0, "errors": [],
           "skipped": [], "violation": None, "found": None, "exhausted": False, "limit": None}
    frontier = []
    for s in spec.init_states:
        k = spec.key(s)
        if k in states:
            continue
        states[k] = s
        parent[k] = None
        frontier.append(k)
        if target is not None:
            if _target_hit(spec, target, s, res):
                res["found"] = k
                break
        v = spec.violations(s)
        if v and res["violation"] is None:
            res["violation"] = (k, v)
            if stop_on_violation:
                break
    combos = {}
    for op in spec.ops.values():
        try:
            combos[op.name] = list(spec.input_combos(op))
        except EvalError as e:
            res["skipped"].append(f"{op.name}: {e.text()}")
    d = 0

    def halted():
        return res["found"] is not None or res["limit"] is not None or             (stop_on_violation and res["violation"] is not None)

    while frontier and d < depth and not halted():
        d += 1
        new = []
        for k in frontier:
            s = states[k]
            any_enabled = False
            for op in spec.ops.values():
                for inp in combos.get(op.name, []):
                    if time.monotonic() - t0 > timeout:
                        res["limit"] = f"timeout after {timeout:.0f}s"
                        break
                    en, succ, errs, _ = spec.step(op, s, inp)
                    for er in errs:
                        if len(res["errors"]) < 5 and er not in res["errors"]:
                            res["errors"].append(f"{er} [in {_state_line(spec, s)}]")
                    if not en:
                        continue
                    any_enabled = True
                    if len(succ) > 1:
                        res["branching"] += 1
                    for t in succ:
                        res["transitions"] += 1
                        tk = spec.key(t)
                        if tk in states:
                            continue
                        states[tk] = t
                        parent[tk] = (k, op.name, inp)
                        new.append(tk)
                        if target is not None and _target_hit(spec, target, t, res):
                            res["found"] = tk
                            break
                        v = spec.violations(t)
                        if v and res["violation"] is None:
                            res["violation"] = (tk, v)
                            if stop_on_violation:
                                break
                        if len(states) >= max_states:
                            res["limit"] = f"state limit {max_states} reached"
                            break
                    if halted():
                        break
                if halted():
                    break
            if not any_enabled and not res["limit"]:
                res["deadlocks"] += 1
            if halted():
                break
        frontier = new
        if res["limit"] is None:
            res["depth_done"] = d
    if not frontier and not halted():
        res["exhausted"] = True
    res["states"] = len(states)
    res["_parent"], res["_states"] = parent, states
    res["seconds"] = round(time.monotonic() - t0, 2)
    return res


def _target_hit(spec, target, s, res):
    try:
        v = evaluate(target, spec.env(s))
        return v is True
    except EvalError as e:
        if len(res["errors"]) < 5:
            res["errors"].append(f"target: {e.text()}")
        return False


def _trace(spec, res, k):
    parent, states = res["_parent"], res["_states"]
    chain = []
    while k is not None:
        p = parent[k]
        chain.append((k, p))
        k = p[0] if p else None
    chain.reverse()
    lines = []
    prev = None
    for i, (k, p) in enumerate(chain):
        s = states[k]
        if p is None:
            lines.append(f"  s0 (initial): {_state_line(spec, s)}")
        else:
            lines.append(f"  s{i} after {_call_str(p[1], p[2])}: {_diff(spec, prev, s)}")
        prev = s
    return lines


def _call_str(op, inputs):
    return f"{op}(" + ", ".join(f"{k}={fmt(v)}" for k, v in inputs.items()) + ")"


def _state_line(spec, s, limit=600):
    out = ", ".join(f"{v}={fmt(s[v])}" for v in spec.state_order if v in s)
    return out if len(out) <= limit else out[:limit] + "..."


def _diff(spec, a, b):
    ch = [f"{v}={fmt(b[v])}" for v in spec.state_order if a is None or not _eq(a.get(v), b.get(v))]
    return ", ".join(ch) if ch else "(no change)"


# ============================================================================ the arm
def _tool(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


_PROV = {"type": "object", "description": "{kind: stated|derived|assumed, quote: short quote or reason}",
         "properties": {"kind": {"type": "string", "enum": list(PROV_KINDS)}, "quote": {"type": "string"}}}
_SCEN = {"description": "a scenario object {steps: [{op, inputs?, repeat?} | {op: 'any'}], set?: {var: expr}, "
                        "constants?: {name: value}, observe?: [expr]} or the name of a stored scenario"}
MUTATING = ("set_spec", "define_given", "define_state", "add_invariant", "set_init", "add_operation",
            "add_derived", "add_scenario")


def _maybe_json(x):
    if isinstance(x, str):
        s = x.strip()
        if s[:1] in "{[":
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


def _upsert(lst, key, entry):
    for i, e in enumerate(lst):
        if isinstance(e, dict) and e.get(key) == entry.get(key):
            lst[i] = entry
            return "replaced"
    lst.append(entry)
    return "added"


class SpecArm(Arm):
    name = "A8"
    buildable = True
    executable = True
    guide_file = "a8_guide.md"
    mutating_tools = MUTATING

    def tools(self):
        return [
            _tool("set_spec", "Replace the whole specification with a JSON object {given, state, invariant, init, "
                  "operations, derived?, scenarios?}.", {"spec": {"type": "object"}}, ["spec"]),
            _tool("define_given", "Define a given set (instances), an integer range, or a constant (value).", {
                "name": {"type": "string"}, "instances": {"type": "array", "items": {"type": "string"}},
                "range": {"type": "array", "items": {"type": "integer"}, "description": "[lo, hi]"},
                "value": {"description": "constant: number, boolean or expression string"},
                "provenance": _PROV}, ["name"]),
            _tool("define_state", "Declare a state variable with a type (e.g. 'set of W', 'W +-> J', 'W <-> J', "
                  "'0..10', 'seq of J').", {"name": {"type": "string"}, "type": {"type": "string"}},
                  ["name", "type"]),
            _tool("add_invariant", "Add (or replace by name) a predicate that must hold in every state.",
                  {"expr": {"type": "string"}, "name": {"type": "string"}, "provenance": _PROV}, ["expr"]),
            _tool("set_init", "Set a variable's initial value (var + value), or add an init predicate.", {
                "var": {"type": "string"}, "value": {"description": "expression string or JSON value"},
                "predicate": {"type": "string"}, "provenance": _PROV}),
            _tool("add_operation", "Add or replace an operation schema: typed inputs, precondition, and a "
                  "postcondition relating x (before) and x' (after); unmentioned variables stay unchanged.", {
                    "name": {"type": "string"}, "inputs": {"type": "object", "description": "{name: type}"},
                    "pre": {"type": "string"}, "post": {"type": "string"}, "provenance": _PROV},
                  ["name", "post"]),
            _tool("add_derived", "Add or replace a derived definition (optionally with parameters).", {
                "name": {"type": "string"}, "params": {"type": "array", "items": {"type": "string"}},
                "expr": {"type": "string"}}, ["name", "expr"]),
            _tool("add_scenario", "Store a named scenario in the specification.", {"scenario": {"type": "object"}},
                  ["scenario"]),
            _tool("run_scenario", "Simulate a sequence of operation invocations from the initial state(s); "
                  "invariants are checked at every state; branching is reported.", {"scenario": _SCEN},
                  ["scenario"]),
            _tool("check_invariants", "Bounded exploration of every enabled operation (all inputs) to the given "
                  "depth; reports the first invariant violation with a trace.", {
                    "depth": {"type": "integer"}, "max_states": {"type": "integer"},
                    "timeout": {"type": "number", "description": "seconds"}}, ["depth"]),
            _tool("find", "Bounded search for a reachable state satisfying a predicate; returns a trace.", {
                "predicate": {"type": "string"}, "depth": {"type": "integer"}, "max_states": {"type": "integer"},
                "timeout": {"type": "number"}}, ["predicate", "depth"]),
            _tool("show", "Show the specification, its holes and the initial state(s).", {}),
        ]

    def probe_tools(self):
        return [t for t in self.tools() if t["function"]["name"] in ("run_scenario", "find")]

    def new_state(self):
        return ArmState(rep={"given": [], "state": [], "invariant": [], "init": [], "operations": [],
                             "derived": [], "scenarios": []})

    def load(self, rep, given_gold=False):
        st = self.new_state()
        if isinstance(rep, dict):
            st.rep = copy.deepcopy(rep)
        else:
            st.log.append(("load", None, f"representation was {type(rep).__name__}, not an object; started empty"))
        st.given_gold = given_gold
        return st

    # ---------------------------------------------------------------- dispatch
    def handle(self, state, name, args):
        try:
            args = _maybe_json(args) if args is not None else {}
            if not isinstance(args, dict):
                out = f"{ERROR_PREFIX}: arguments must be an object (hint: see the {name} schema)"
            else:
                fn = getattr(self, "_t_" + str(name), None)
                if fn is None:
                    out = (f"{ERROR_PREFIX}: unknown tool '{name}' (hint: available: "
                           + ", ".join(t["function"]["name"] for t in self.tools()) + ")")
                elif state.given_gold and name in MUTATING:
                    out = f"{ERROR_PREFIX}: the specification is read-only in this mode (hint: use run_scenario, " \
                          f"find, check_invariants, show)"
                else:
                    out = fn(state, args)
        except Exception as e:  # never raise
            out = f"{ERROR_PREFIX}: internal {type(e).__name__}: {e} (hint: check the arguments)"
        state.log.append((name, args, str(out)[:200]))
        return out

    def _rep(self, state):
        r = state.rep if isinstance(state.rep, dict) else {}
        state.rep = r
        for k, alias in (("given", "givens"), ("invariant", "invariants"), ("operations", "ops")):
            if k not in r and alias in r:
                r[k] = r.pop(alias)
        for k in ("given", "state", "invariant", "init", "operations", "derived", "scenarios"):
            v = r.get(k)
            if isinstance(v, dict):
                r[k] = [dict(x, name=n) if isinstance(x, dict) else {"name": n, "_v": x} for n, x in v.items()] \
                    if k != "init" else [{"var": n, "value": x} for n, x in v.items()]
            elif isinstance(v, str):
                r[k] = [v] if k in ("invariant", "init") else []
            elif not isinstance(v, list):
                r[k] = []
        return r

    def _status(self, state, focus=None):
        sp = Spec(state.rep)
        s = (f"{len(sp.sets)} given sets, {len(sp.consts)} constants, {len(sp.state_types)} state variables, "
             f"{len(sp.invariants)} invariants, {len(sp.ops)} operations, {len(sp.init_states)} initial state(s); "
             f"{len(sp.holes)} hole(s)")
        holes = sp.holes if focus is None else [h for h in sp.holes if focus in h["where"]] or sp.holes
        txt = _holes_text(holes, 8)
        prefix = f"{ERROR_PREFIX}: " if focus is not None and any(focus in h["where"] for h in sp.holes) else ""
        return prefix, s + txt, sp

    def _t_set_spec(self, state, args):
        spec = _maybe_json(args.get("spec", args))
        if not isinstance(spec, dict):
            return f"{ERROR_PREFIX}: spec must be a JSON object (hint: {{given, state, invariant, init, operations}})"
        state.rep = copy.deepcopy(spec)
        self._rep(state)
        _, s, sp = self._status(state)
        out = "specification set: " + s
        if sp.init_states:
            out += "\ninitial: " + _state_line(sp, sp.init_states[0])
        return out

    def _t_define_given(self, state, args):
        r = self._rep(state)
        name = args.get("name")
        if not isinstance(name, str) or not name:
            return f"{ERROR_PREFIX}: define_given needs a 'name' (hint: name='Worker', instances=['w1','w2'])"
        e = {k: args[k] for k in ("name", "instances", "range", "value", "provenance") if k in args}
        how = _upsert(r["given"], "name", e)
        pre, s, _ = self._status(state, f"given '{name}'")
        return f"{pre}given '{name}' {how}. {s}"

    def _t_define_state(self, state, args):
        r = self._rep(state)
        name = args.get("name")
        if not isinstance(name, str) or not isinstance(args.get("type"), str):
            return f"{ERROR_PREFIX}: define_state needs 'name' and 'type' (hint: type='set of Worker')"
        how = _upsert(r["state"], "name", {k: args[k] for k in ("name", "type", "provenance") if k in args})
        pre, s, _ = self._status(state, f"state '{name}'")
        return f"{pre}state '{name}' {how}. {s}"

    def _t_add_invariant(self, state, args):
        r = self._rep(state)
        if not isinstance(args.get("expr"), str):
            return f"{ERROR_PREFIX}: add_invariant needs 'expr' (hint: expr='#busy <= 3')"
        name = args.get("name") or f"inv{len(r['invariant']) + 1}"
        r["invariant"] = [x if isinstance(x, dict) else {"expr": x} for x in r["invariant"]]
        how = _upsert(r["invariant"], "name", {k: v for k, v in dict(args, name=name).items()
                                               if k in ("name", "expr", "provenance")})
        pre, s, _ = self._status(state, f"invariant '{name}'")
        return f"{pre}invariant '{name}' {how}. {s}"

    def _t_set_init(self, state, args):
        r = self._rep(state)
        r["init"] = [x if isinstance(x, dict) else {"predicate": x} for x in r["init"]]
        if "var" in args:
            if "value" not in args:
                return f"{ERROR_PREFIX}: set_init with 'var' needs a 'value' (hint: value='{{}}')"
            e = {k: args[k] for k in ("var", "value", "provenance") if k in args}
            r["init"] = [x for x in r["init"] if x.get("var") != args["var"]] + [e]
            what = f"init '{args['var']}'"
        elif isinstance(args.get("predicate"), str):
            r["init"].append({k: args[k] for k in ("predicate", "provenance") if k in args})
            what = f"init[{len(r['init']) - 1}]"
        else:
            return f"{ERROR_PREFIX}: set_init needs var+value or predicate (hint: var='busy', value='{{}}')"
        pre, s, sp = self._status(state, what)
        out = f"{pre}{what} set. {s}"
        if sp.init_states:
            out += f"\ninitial: {_state_line(sp, sp.init_states[0])}"
            if len(sp.init_states) > 1:
                out += f" (+{len(sp.init_states) - 1} more initial states)"
        return out

    def _t_add_operation(self, state, args):
        r = self._rep(state)
        name = args.get("name")
        if not isinstance(name, str) or not name:
            return f"{ERROR_PREFIX}: add_operation needs a 'name'"
        e = {k: _maybe_json(args[k]) if k == "inputs" else args[k]
             for k in ("name", "inputs", "pre", "post", "provenance") if k in args}
        how = _upsert(r["operations"], "name", e)
        pre, s, _ = self._status(state, f"operation '{name}'")
        return f"{pre}operation '{name}' {how}. {s}"

    def _t_add_derived(self, state, args):
        r = self._rep(state)
        if not isinstance(args.get("name"), str) or not isinstance(args.get("expr"), str):
            return f"{ERROR_PREFIX}: add_derived needs 'name' and 'expr'"
        how = _upsert(r["derived"], "name", {k: args[k] for k in ("name", "params", "expr", "provenance")
                                             if k in args})
        pre, s, _ = self._status(state, f"derived '{args['name']}'")
        return f"{pre}derived '{args['name']}' {how}. {s}"

    def _t_add_scenario(self, state, args):
        r = self._rep(state)
        sc = _maybe_json(args.get("scenario"))
        if not isinstance(sc, dict) or not sc.get("name") or not isinstance(sc.get("steps", []), list):
            return f"{ERROR_PREFIX}: scenario must be an object with 'name' and 'steps' list"
        how = _upsert(r["scenarios"], "name", sc)
        return f"scenario '{sc['name']}' {how}."

    def _t_show(self, state, args):
        return self.render(state)

    # ---------------------------------------------------------------- scenario
    def _t_run_scenario(self, state, args):
        sc = _maybe_json(args.get("scenario", args if "steps" in args else None))
        rep = self._rep(state)
        if isinstance(sc, str):
            sp0 = Spec(rep)
            if sc not in sp0.scenarios:
                return f"{ERROR_PREFIX}: no stored scenario named '{sc}' (hint: " \
                       f"{', '.join(sp0.scenarios) or 'pass a scenario object'})"
            sc = sp0.scenarios[sc]
        if not isinstance(sc, dict):
            return f"{ERROR_PREFIX}: scenario must be an object {{steps: [...]}} or a stored scenario name " \
                   f"(hint: {{\"steps\": [{{\"op\": \"any\"}}]}})"
        steps = sc.get("steps", [])
        if not isinstance(steps, list):
            return f"{ERROR_PREFIX}: scenario 'steps' must be a list (hint: [{{\"op\": \"Assign\", \"inputs\": {{...}}}}])"
        consts = sc.get("constants") or {}
        if consts:
            rep = copy.deepcopy(rep)
            names = {g.get("name") for g in rep["given"] if isinstance(g, dict)}
            for k, v in consts.items():
                if k not in names:
                    return f"{ERROR_PREFIX}: constants override: '{k}' is not a given (hint: {', '.join(map(str, names))})"
                for g in rep["given"]:
                    if isinstance(g, dict) and g.get("name") == k:
                        for f in ("instances", "range", "value"):
                            g.pop(f, None)
                        if isinstance(v, list) and all(isinstance(x, str) for x in v):
                            g["instances"] = v
                        else:
                            g["value"] = v
        sp = Spec(rep)
        lines = []
        if sp.holes:
            lines.append(f"note: {len(sp.holes)} hole(s) left out:" + _holes_text(sp.holes, 4))
        if not sp.init_states:
            return f"{ERROR_PREFIX}: no initial state (hint: fix init; holes:{_holes_text(sp.holes, 6)})"
        observe = sc.get("observe") or []
        if isinstance(observe, str):
            observe = [observe]
        obs_asts = []
        for o in observe:
            try:
                obs_asts.append((o, parse(o)))
            except SpecError as e:
                lines.append(f"{ERROR_PREFIX}: observe '{o}': {e.text(o)}")
        frontier = [dict(s) for s in sp.init_states]
        # state overrides
        setv = sc.get("set") or {}
        if setv:
            new = []
            for s in frontier:
                s2 = dict(s)
                for var, val in setv.items():
                    if var not in sp.state_types:
                        return f"{ERROR_PREFIX}: set: '{var}' is not a state variable (hint:" \
                               f"{_close(var, sp.state_types) or ' ' + ', '.join(sp.state_order)})"
                    try:
                        s2[var] = evaluate(parse(val), sp.env(s)) if isinstance(val, str) else _from_json(val)
                    except (SpecError, EvalError) as e:
                        return f"{ERROR_PREFIX}: set '{var}': {e.text(val) if isinstance(e, SpecError) else e.text()}"
                new.append(s2)
            frontier = new
        name = sc.get("name", "scenario")
        lines.append(f"scenario '{name}': {len(frontier)} initial state(s)")
        failed = hard = False
        failed |= self._report_states(sp, frontier, None, lines, obs_asts, "s0")
        idx = 0
        for si, st in enumerate(steps):
            if isinstance(st, str):
                st = {"op": st}
            if not isinstance(st, dict) or "op" not in st:
                lines.append(f"{ERROR_PREFIX}: step {si + 1}: must be {{op, inputs?}} (hint: {{\"op\": \"any\"}} "
                             f"for any enabled operation)")
                hard = True
                break
            reps = st.get("repeat", 1)
            if not isinstance(reps, int) or reps < 1 or reps > 1000:
                lines.append(f"{ERROR_PREFIX}: step {si + 1}: repeat must be 1..1000")
                hard = True
                break
            stop = False
            for _ in range(reps):
                idx += 1
                label = self._step_label(st)
                new, msgs, branching, err = self._apply_step(sp, frontier, st)
                if err:
                    lines.append(f"{ERROR_PREFIX}: step {idx} {label}: {err}")
                    hard, stop = True, True
                    break
                for m in msgs[:4]:
                    lines.append(f"  step {idx} {label}: {m}")
                if not new:
                    lines.append(f"step {idx} {label}: no successor (not enabled) -> scenario stops")
                    failed, stop = True, True
                    break
                extra = f" [BRANCHING: {len(frontier)} -> {len(new)} possible states]" if len(new) > len(frontier) \
                    else (f" ({len(new)} states)" if len(new) > 1 else "")
                lines.append(f"step {idx} {label}: ok{extra}")
                if len(new) > MAX_FRONTIER:
                    lines.append(f"  (keeping the first {MAX_FRONTIER} of {len(new)} states)")
                    new = new[:MAX_FRONTIER]
                failed |= self._report_states(sp, new, frontier if len(frontier) == 1 else None, lines, obs_asts,
                                              f"s{idx}")
                frontier = new
            if stop:
                break
        lines.append(f"final: {len(frontier)} state(s)")
        if len(frontier) == 1:
            lines.append("  " + _state_line(sp, frontier[0], 1500))
        out = "\n".join(lines)
        if len(out) > 12000:
            out = out[:12000] + "\n...(truncated)"
        if hard:
            out = f"{ERROR_PREFIX}: the scenario could not be run as written (details below)\n" + out
        return out

    def _step_label(self, st):
        inp = st.get("inputs") or {}
        if st["op"] == "any":
            return "any"
        return f"{st['op']}(" + ", ".join(f"{k}={v}" for k, v in inp.items()) + ")"

    def _apply_step(self, sp, frontier, st):
        opname = st["op"]
        raw = _maybe_json(st.get("inputs") or {})
        if not isinstance(raw, dict):
            return [], [], 0, "inputs must be an object {name: value}"
        if opname == "any":
            ops = list(sp.ops.values())
        elif opname in sp.ops:
            ops = [sp.ops[opname]]
        else:
            hole = [h for h in sp.holes if f"'{opname}'" in h["where"]]
            if hole:
                return [], [], 0, f"operation '{opname}' is a hole: {hole[0]['error']} (hint: {hole[0]['hint']})"
            return [], [], 0, f"unknown operation '{opname}' (hint:{_close(opname, sp.ops) or ' ' + ', '.join(sp.ops)})"
        out, msgs, uniq = [], [], {}
        disabled = 0
        for s in frontier:
            for op in ops:
                given = {}
                for k, v in (raw.items() if opname != "any" else []):
                    if k not in dict(op.inputs):
                        return [], [], 0, f"{op.name} has no input '{k}' (hint: inputs are " \
                                          f"{', '.join(n for n, _ in op.inputs) or 'none'})"
                    try:
                        given[k] = evaluate(parse(v), sp.env(s)) if isinstance(v, str) else _from_json(v)
                    except SpecError as e:
                        return [], [], 0, f"input '{k}': {e.text(v)}"
                    except EvalError as e:
                        return [], [], 0, f"input '{k}': {e.text()}"
                try:
                    combos = list(sp.input_combos(op, given))
                except EvalError as e:
                    if opname == "any":
                        msgs.append(f"skipped {op.name}: {e.text()}")
                        continue
                    return [], [], 0, e.text()
                for inp in combos:
                    en, succ, errs, notes = sp.step(op, s, inp)
                    for er in errs:
                        msgs.append(f"{ERROR_PREFIX}: {er}")
                    if not en and not errs and opname != "any":
                        if len(combos) == 1:
                            disabled += 1
                        continue
                    if en and not succ and not errs:
                        msgs.append(f"{_call_str(op.name, inp)}: precondition holds but no next state satisfies the "
                                    f"postcondition (contradictory post or type violation)")
                    if len(succ) > 1 and opname != "any":
                        msgs.append(f"{_call_str(op.name, inp)} is nondeterministic: {len(succ)} successors")
                    for t in succ:
                        uniq.setdefault(sp.key(t), (t, op.name, inp))
        for t, opn, inp in uniq.values():
            out.append(t)
        if disabled:
            where = "" if len(frontier) == 1 else f" in {disabled} of {len(frontier)} states (those branches end)"
            msgs.insert(0, f"precondition false{where}: {ops[0].src.get('pre')}")
        if opname == "any" and len(uniq) > 1:
            via = {}
            for t, opn, inp in uniq.values():
                via.setdefault(_call_str(opn, inp), 0)
                via[_call_str(opn, inp)] += 1
            msgs.append("via " + "; ".join(list(via)[:8]) + (" ..." if len(via) > 8 else ""))
        return out, msgs, len(out), None

    def _report_states(self, sp, states, prev, lines, obs_asts, label):
        failed = False
        for i, s in enumerate(states[:4]):
            tag = label if len(states) == 1 else f"{label}.{i + 1}"
            if prev is not None:
                lines.append(f"  {tag}: changed {_diff(sp, prev[0], s)}")
            else:
                lines.append(f"  {tag}: {_state_line(sp, s)}")
            for lab, msg in sp.violations(s):
                lines.append(f"  {tag}: VIOLATION {msg}")
                failed = True
            if obs_asts:
                vals = []
                for src, ast in obs_asts:
                    try:
                        vals.append(f"{src} = {fmt(evaluate(ast, sp.env(s)))}")
                    except EvalError as e:
                        vals.append(f"{src} = ERROR {e.text()}")
                lines.append(f"  {tag} observe: " + "; ".join(vals))
        if len(states) > 4:
            lines.append(f"  ... and {len(states) - 4} more states")
            bad = sum(1 for s in states[4:] if sp.violations(s))
            if bad:
                lines.append(f"  VIOLATION in {bad} of the remaining states")
                failed = True
        if obs_asts and len(states) > 1:
            for src, ast in obs_asts:
                vs = set()
                for s in states:
                    try:
                        vs.add(fmt(evaluate(ast, sp.env(s))))
                    except EvalError:
                        vs.add("ERROR")
                lines.append(f"  {label} observe '{src}' across {len(states)} states: {{{', '.join(sorted(vs))}}}")
        return failed

    # ---------------------------------------------------------------- exploration tools
    def _limits(self, args):
        depth = args.get("depth", 5)
        if not isinstance(depth, int) or depth < 0 or depth > 200:
            raise EvalError("depth must be an integer 0..200")
        ms = args.get("max_states", DEFAULT_MAX_STATES)
        to = args.get("timeout", DEFAULT_TIMEOUT)
        if not isinstance(ms, int) or ms < 1:
            ms = DEFAULT_MAX_STATES
        if not isinstance(to, (int, float)) or to <= 0:
            to = DEFAULT_TIMEOUT
        return depth, min(ms, 200000), min(float(to), 120.0)

    def _t_check_invariants(self, state, args):
        try:
            depth, ms, to = self._limits(args)
        except EvalError as e:
            return f"{ERROR_PREFIX}: {e.text()}"
        sp = Spec(self._rep(state))
        if not sp.init_states:
            return f"{ERROR_PREFIX}: no initial state (hint: fix init; holes:{_holes_text(sp.holes, 6)})"
        if not sp.invariants:
            pre = "note: no invariants defined; only types are checked\n"
        else:
            pre = ""
        res = explore(sp, depth, None, ms, to)
        lines = [pre + _explore_header(sp, res, depth)]
        if res["violation"]:
            k, v = res["violation"]
            lines[0] = "VIOLATION: " + "; ".join(m for _, m in v[:3])
            lines.append(_explore_header(sp, res, depth))
            lines.append("trace:")
            lines += _trace(sp, res, k)
        else:
            lines.append(f"OK: no invariant or type violation in {res['states']} reachable states"
                         + (" (state space exhausted: complete)" if res["exhausted"] else
                            f" up to depth {res['depth_done']}"))
        lines += _explore_footer(sp, res)
        return "\n".join(lines)

    def _t_find(self, state, args):
        try:
            depth, ms, to = self._limits(args)
        except EvalError as e:
            return f"{ERROR_PREFIX}: {e.text()}"
        src = args.get("predicate")
        sp = Spec(self._rep(state))
        try:
            ast = parse(src)
            err = sp._check_names(ast, set(), False)
            if err:
                raise SpecError(err[0], None, err[1])
        except SpecError as e:
            return f"{ERROR_PREFIX}: predicate: {e.text(src if isinstance(src, str) else None)}"
        if not sp.init_states:
            return f"{ERROR_PREFIX}: no initial state (hint: fix init; holes:{_holes_text(sp.holes, 6)})"
        res = explore(sp, depth, ast, ms, to, stop_on_violation=False)
        lines = []
        if res["found"] is not None:
            tr = _trace(sp, res, res["found"])
            lines.append(f"FOUND: '{src}' is reachable in {len(tr) - 1} step(s)")
            lines.append("trace:")
            lines += tr
        else:
            if res["exhausted"]:
                lines.append(f"NOT REACHABLE: '{src}' holds in none of the {res['states']} reachable states "
                             "(state space exhausted: complete)")
            else:
                lines.append(f"NOT FOUND within depth {res['depth_done']} ({res['states']} states"
                             + (f"; stopped: {res['limit']}" if res["limit"] else "") + ")")
        lines.append(_explore_header(sp, res, depth))
        if res["violation"]:
            k, v = res["violation"]
            lines.append("note: an invariant is violated in a reachable state: " + v[0][1])
        lines += _explore_footer(sp, res)
        return "\n".join(lines)

    # ---------------------------------------------------------------- rendering / validity
    def render(self, state):
        rep = self._rep(state)
        sp = Spec(rep)
        L = ["GIVEN"]
        for name, s in sp.sets.items():
            L.append(f"  {name} = {fmt(s) if len(s) <= 40 else f'{min(s)}..{max(s)}'}")
        for name, v in sp.consts.items():
            L.append(f"  {name} = {fmt(v)}")
        L.append("STATE")
        for v in sp.state_order:
            L.append(f"  {v} : {type_str(sp.state_types[v])}")
        if sp.derived:
            L.append("DERIVED")
            for n, (params, ast, src) in sp.derived.items():
                L.append(f"  {n}{'(' + ', '.join(params) + ')' if params else ''} == {src}")
        L.append("INVARIANT")
        for label, ast, src in sp.invariants:
            L.append(f"  [{label}] {src}")
        L.append("INIT")
        for e in rep.get("init", []):
            if isinstance(e, dict):
                L.append(f"  {e['var']} = {e.get('value')}" if "var" in e else f"  {e.get('predicate')}")
            else:
                L.append(f"  {e}")
        L.append("OPERATIONS")
        for op in sp.ops.values():
            ins = ", ".join(f"{n}: {type_str(t)}" for n, t in op.inputs)
            L.append(f"  {op.name}({ins})")
            L.append(f"    pre:  {op.src.get('pre', 'true')}")
            L.append(f"    post: {op.src.get('post')}")
        if sp.scenarios:
            L.append("SCENARIOS: " + ", ".join(sp.scenarios))
        L.append(f"INITIAL STATE(S): {len(sp.init_states)}")
        for s in sp.init_states[:3]:
            L.append("  " + _state_line(sp, s, 1500))
            for _, m in sp.violations(s):
                L.append(f"  VIOLATION {m}")
        if sp.holes:
            L.append(f"HOLES ({len(sp.holes)}; left out, the rest runs):" + _holes_text(sp.holes, 30))
        if sp.warnings:
            L.append("WARNINGS: " + "; ".join(sp.warnings[:15]))
        return "\n".join(L)[:60000]

    def validity(self, state):
        sp = Spec(state.rep if isinstance(state.rep, dict) else {})
        usable = bool(sp.state_types)
        executable = usable and bool(sp.init_states)
        errors = []
        if executable:
            for s in sp.init_states[:3]:
                errors += [m for _, m in sp.violations(s)]
            for op in list(sp.ops.values())[:20]:
                try:
                    for inp in itertools.islice(sp.input_combos(op), 50):
                        _, _, errs, _ = sp.step(op, sp.init_states[0], inp)
                        errors += errs[:1]
                        if errs:
                            break
                except EvalError as e:
                    errors.append(f"{op.name}: {e.text()}")
        return {"usable": usable, "executable": executable,
                "holes": [f"{h['where']}: {h['error']}" for h in sp.holes],
                "errors": errors[:20], "warnings": sp.warnings[:20],
                "primitives": primitives_used(sp)}


def primitives_used(sp):
    """Which primitive kinds this specification actually uses (for H-prim accounting)."""
    used = set()
    if sp.sets:
        used.add("given set")
    if sp.consts:
        used.add("constant")
    for t in sp.state_types.values():
        for k in _type_kinds(t):
            used.add({"set": "set", "rel": "relation", "fun": "function", "seq": "sequence", "cross": "pair",
                      "int": "integer", "nat": "integer", "range": "integer", "bool": "boolean",
                      "given": "atom"}[k])
    if sp.state_types:
        used.add("state variable")
    if sp.invariants:
        used.add("invariant")
    if sp.ops:
        used.add("operation")
    if sp.derived:
        used.add("derived definition")
    for op in sp.ops.values():
        for n in _walk(op.post):
            if n[0] == "quant" and n[1] == "exists" or (n[0] == "bin" and n[1] == "or"):
                used.add("nondeterministic choice")
    return sorted(used)


def _type_kinds(t):
    yield t[0]
    for x in t[1:]:
        if isinstance(x, tuple):
            yield from _type_kinds(x)


def _walk(n):
    yield n
    for x in n[1:]:
        if isinstance(x, tuple) and x and isinstance(x[0], str):
            yield from _walk(x)
        elif isinstance(x, list):
            for y in x:
                if isinstance(y, tuple) and y and isinstance(y[0], str):
                    yield from _walk(y)
                elif isinstance(y, tuple):
                    for z in y:
                        if isinstance(z, tuple) and z and isinstance(z[0], str):
                            yield from _walk(z)


def _holes_text(holes, limit):
    if not holes:
        return ""
    out = []
    for h in holes[:limit]:
        out.append(f"\n  HOLE {h['where']}: {h['error']}" + (f" (hint: {h['hint']})" if h.get("hint") else ""))
    if len(holes) > limit:
        out.append(f"\n  ... {len(holes) - limit} more")
    return "".join(out)


def _explore_header(sp, res, depth):
    return (f"explored {res['states']} states, {res['transitions']} transitions, depth {res['depth_done']}/{depth}, "
            f"{res['seconds']}s" + (f"; stopped: {res['limit']}" if res["limit"] else ""))


def _explore_footer(sp, res):
    out = []
    if res["branching"]:
        out.append(f"branching: {res['branching']} (state, operation, inputs) combinations had more than one "
                   "successor (nondeterministic postconditions)")
    if res["deadlocks"]:
        out.append(f"deadlocks: {res['deadlocks']} reachable states have no enabled operation")
    for s in res["skipped"]:
        out.append(f"skipped operation {s}")
    for e in res["errors"]:
        out.append(f"evaluation error: {e}")
    if sp.holes:
        out.append(f"{len(sp.holes)} hole(s) were left out:" + _holes_text(sp.holes, 5))
    return out
