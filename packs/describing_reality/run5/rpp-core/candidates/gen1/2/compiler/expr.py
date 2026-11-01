"""Expression sub-language of R++ Core.

Conditions and value expressions inside the JSON AST are strings in a small,
fully specified grammar (see lang/SPEC.md §4). This module parses them into
tuples:

  ("int", n) ("name", s) ("var", s) ("call", name, [args])
  ("bin", op, a, b)     op in + - * / %
  ("neg", a)
  ("cmp", op, a, b)     op in = != < <= > >=
  ("and", [..]) ("or", [..]) ("not", a)
  ("agg", "count"|"sum", var, sort, value_expr_or_None, cond_or_None)
  ("fn", "min"|"max"|"abs", [args])
  ("dec", n, d)         decimal literal n / 10**d (e.g. 0.25 -> ("dec", 25, 2))
  ("ite", cond, a, b)   if cond then a else b
  ("now",)              the current step (built-in names t / now)
"""
import re

TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<dec>\d+\.\d+)
  | (?P<int>\d+)
  | (?P<op><=|>=|!=|==|[=<>+\-*/%(),:|])
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
""", re.X)

KEYWORDS = {"and", "or", "not", "in", "if", "then", "else", "mod", "true", "false"}
TIME_NAMES = {"t", "now"}
RESERVED = KEYWORDS | TIME_NAMES
AGGS = {"count", "sum"}
FNS = {"min", "max", "abs"}


class ExprError(ValueError):
    pass


def tokenize(s):
    pos, out = 0, []
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m:
            raise ExprError(f"unexpected character {s[pos]!r} at {pos} in {s!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        val = m.group(kind)
        if kind == "op" and val == "==":
            val = "="
        out.append((kind, val))
    out.append(("end", None))
    return out


class Parser:
    def __init__(self, s):
        self.s = s
        self.toks = tokenize(s)
        self.i = 0

    def peek(self, k=0):
        return self.toks[self.i + k]

    def take(self, val=None, kind=None):
        t = self.toks[self.i]
        if val is not None and t[1] != val:
            raise ExprError(f"expected {val!r}, got {t[1]!r} in {self.s!r}")
        if kind is not None and t[0] != kind:
            raise ExprError(f"expected {kind}, got {t[1]!r} in {self.s!r}")
        self.i += 1
        return t

    def at(self, val):
        return self.toks[self.i][1] == val and self.toks[self.i][0] in ("op", "name")

    def parse(self):
        e = self.expr_()
        if self.peek()[0] != "end":
            raise ExprError(f"trailing input at {self.peek()[1]!r} in {self.s!r}")
        return e

    def expr_(self):
        # full expression; also accepts the Python form "A if C else B"
        e = self.or_()
        if self.at("if"):
            self.take()
            c = self.or_()
            self.take("else")
            return ("ite", c, e, self.expr_())
        return e

    def or_(self):
        parts = [self.and_()]
        while self.at("or"):
            self.take()
            parts.append(self.and_())
        return parts[0] if len(parts) == 1 else ("or", parts)

    def and_(self):
        parts = [self.not_()]
        while self.at("and"):
            self.take()
            parts.append(self.not_())
        return parts[0] if len(parts) == 1 else ("and", parts)

    def not_(self):
        if self.at("not"):
            self.take()
            return ("not", self.not_())
        return self.cmp()

    def cmp(self):
        a = self.sum()
        t = self.peek()
        if t[0] == "op" and t[1] in ("=", "!=", "<", "<=", ">", ">="):
            self.take()
            b = self.sum()
            return ("cmp", t[1], a, b)
        return a

    def sum(self):
        a = self.prod()
        while self.peek()[0] == "op" and self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            a = ("bin", op, a, self.prod())
        return a

    def prod(self):
        a = self.unary()
        while (self.peek()[0] == "op" and self.peek()[1] in ("*", "/", "%")) or self.at("mod"):
            op = self.take()[1]
            op = "%" if op == "mod" else op
            a = ("bin", op, a, self.unary())
        return a

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return ("neg", self.unary())
        return self.atom()

    def atom(self):
        kind, val = self.peek()
        if kind == "int":
            self.take()
            return ("int", int(val))
        if kind == "dec":
            self.take()
            whole, frac = val.split(".")
            return ("dec", int(whole + frac), len(frac))
        if (kind, val) == ("op", "("):
            self.take()
            e = self.expr_()
            self.take(")")
            return e
        if kind == "name":
            if val in ("true", "false"):
                self.take()
                return ("name", val)
            if val == "if":
                # if C then A else B   (else-branch extends as far as possible)
                self.take()
                c = self.expr_()
                self.take("then")
                a = self.expr_()
                self.take("else")
                return ("ite", c, a, self.expr_())
            if val in TIME_NAMES and self.peek(1) != ("op", "("):
                self.take()
                return ("now",)
            if val in KEYWORDS:
                raise ExprError(f"unexpected keyword {val!r} in {self.s!r}")
            self.take()
            if self.peek() == ("op", "("):
                if val in AGGS:
                    return self.agg(val)
                self.take("(")
                args = []
                if self.peek() != ("op", ")"):
                    args.append(self.expr_())
                    while self.peek() == ("op", ","):
                        self.take()
                        args.append(self.expr_())
                self.take(")")
                if val in FNS:
                    return ("fn", val, args)
                return ("call", val, args)
            if val[0].isupper():
                return ("var", val)
            return ("name", val)
        raise ExprError(f"unexpected token {val!r} in {self.s!r}")

    def agg(self, which):
        # count(X in sort: cond)   sum(X in sort: value [if cond])
        self.take("(")
        var = self.take(kind="name")[1]
        if not var[0].isupper():
            raise ExprError(f"aggregate variable must be capitalised in {self.s!r}")
        self.take("in")
        sort = self.take(kind="name")[1]
        self.take(":")
        if which == "count":
            cond = self.or_()
            self.take(")")
            return ("agg", "count", var, sort, None, cond)
        value = self.sum()
        cond = None
        if self.at("if"):
            self.take()
            cond = self.or_()
        self.take(")")
        return ("agg", "sum", var, sort, value, cond)


def parse(s):
    if isinstance(s, (int, bool)):
        s = str(s).lower()
    return Parser(str(s)).parse()


def free_vars(e, bound=frozenset()):
    k = e[0]
    if k == "var":
        return set() if e[1] in bound else {e[1]}
    if k in ("int", "name", "dec", "now"):
        return set()
    if k == "ite":
        return free_vars(e[1], bound) | free_vars(e[2], bound) | free_vars(e[3], bound)
    if k == "call" or k == "fn":
        out = set()
        for a in e[2]:
            out |= free_vars(a, bound)
        return out
    if k == "bin" or k == "cmp":
        return free_vars(e[2], bound) | free_vars(e[3], bound)
    if k == "neg" or k == "not":
        return free_vars(e[1], bound)
    if k in ("and", "or"):
        out = set()
        for a in e[1]:
            out |= free_vars(a, bound)
        return out
    if k == "agg":
        b = bound | {e[2]}
        out = set()
        if e[4] is not None:
            out |= free_vars(e[4], b)
        if e[5] is not None:
            out |= free_vars(e[5], b)
        return out
    raise ExprError(f"unknown node {k}")


def calls(e):
    """Yield every ('call', name, args) node."""
    k = e[0]
    if k == "call":
        yield e
        for a in e[2]:
            yield from calls(a)
    elif k == "fn":
        for a in e[2]:
            yield from calls(a)
    elif k in ("bin", "cmp"):
        yield from calls(e[2])
        yield from calls(e[3])
    elif k in ("neg", "not"):
        yield from calls(e[1])
    elif k == "ite":
        for a in e[1:]:
            yield from calls(a)
    elif k in ("and", "or"):
        for a in e[1]:
            yield from calls(a)
    elif k == "agg":
        if e[4] is not None:
            yield from calls(e[4])
        if e[5] is not None:
            yield from calls(e[5])


def names(e):
    """Yield every bare ('name', s) node (constants or 0-ary fluents/statics)."""
    k = e[0]
    if k == "name":
        yield e[1]
    elif k in ("call", "fn"):
        for a in e[2]:
            yield from names(a)
    elif k in ("bin", "cmp"):
        yield from names(e[2])
        yield from names(e[3])
    elif k in ("neg", "not"):
        yield from names(e[1])
    elif k == "ite":
        for a in e[1:]:
            yield from names(a)
    elif k in ("and", "or"):
        for a in e[1]:
            yield from names(a)
    elif k == "agg":
        if e[4] is not None:
            yield from names(e[4])
        if e[5] is not None:
            yield from names(e[5])


def to_text(e):
    k = e[0]
    if k == "int":
        return str(e[1])
    if k in ("name", "var"):
        return e[1]
    if k == "dec":
        return f"{e[1] / 10 ** e[2]:.{e[2]}f}"
    if k == "now":
        return "t"
    if k == "ite":
        return f"(if {to_text(e[1])} then {to_text(e[2])} else {to_text(e[3])})"
    if k in ("call", "fn"):
        return f"{e[1]}({', '.join(to_text(a) for a in e[2])})"
    if k == "bin":
        return f"({to_text(e[2])} {e[1]} {to_text(e[3])})"
    if k == "neg":
        return f"-{to_text(e[1])}"
    if k == "cmp":
        return f"{to_text(e[2])} {e[1]} {to_text(e[3])}"
    if k == "and":
        return " and ".join(_wrap(a) for a in e[1])
    if k == "or":
        return " or ".join(_wrap(a) for a in e[1])
    if k == "not":
        return f"not {_wrap(e[1])}"
    if k == "agg":
        if e[1] == "count":
            return f"count({e[2]} in {e[3]}: {to_text(e[5])})"
        tail = f" if {to_text(e[5])}" if e[5] is not None else ""
        return f"sum({e[2]} in {e[3]}: {to_text(e[4])}{tail})"
    raise ExprError(k)


def _wrap(e):
    return f"({to_text(e)})" if e[0] in ("and", "or") else to_text(e)
