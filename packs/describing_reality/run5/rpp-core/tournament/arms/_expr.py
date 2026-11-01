"""Small safe expression language used by A2 (structural equations) and A4 (event predicates).

No Python `eval`: expressions are tokenised and parsed by a recursive-descent parser into tuples,
then interpreted. Supported:

  numbers, 'strings' / "strings", true / false
  + - * / % ^ (or **), unary -, comparisons == != < <= > >=, and / or / not (also && || !)
  if c then a else b        ifelse(c, a, b)
  min(...), max(...), abs, sqrt, exp, log(x[, base]), floor, ceil, round(x[, n]), clip(x, lo, hi), sign
  variable references: x (current step), x[t-1] / prev(x) / lag(x, k) (k steps earlier)
  built-ins t (current step index) and dt (step length)

AST node shapes:
  ("num", v) ("str", s) ("bool", b) ("var", name, lag) ("neg", a) ("not", a)
  ("bin", op, a, b) ("and", a, b) ("or", a, b) ("if", c, a, b) ("call", fn, [args])
"""
import math

MAX_LEN = 4000
MAX_DEPTH = 80
KEYWORDS = {"if", "then", "else", "and", "or", "not", "true", "false"}
RESERVED = KEYWORDS | {"t", "dt"}
FUNCS = {"min", "max", "abs", "sqrt", "exp", "log", "floor", "ceil", "round", "clip", "ifelse", "sign",
         "prev", "lag"}


class ExprError(Exception):
    """Parse error; `col` is a 1-based column."""

    def __init__(self, msg, col=None, hint=""):
        super().__init__(msg)
        self.msg = msg
        self.col = col
        self.hint = hint

    def __str__(self):
        s = self.msg + (f" at column {self.col}" if self.col else "")
        return s + (f" (hint: {self.hint})" if self.hint else "")


class EvalError(Exception):
    pass


# ----------------------------------------------------------------------------------------- tokenizer
def tokenize(src):
    if not isinstance(src, str):
        raise ExprError(f"expression must be a string, got {type(src).__name__}", hint="write e.g. \"a * 2 + b\"")
    if len(src) > MAX_LEN:
        raise ExprError(f"expression too long ({len(src)} chars, max {MAX_LEN})")
    toks = []
    i, n = 0, len(src)
    two = {"**", "<=", ">=", "==", "!=", "&&", "||"}
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        col = i + 1
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
            text = src[i:j]
            try:
                v = float(text)
            except ValueError:
                raise ExprError(f"bad number '{text}'", col)
            toks.append(("num", v, col))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_."):
                j += 1
            name = src[i:j].rstrip(".")
            j = i + len(name)
            toks.append(("name", name, col))
            i = j
            continue
        if c in "'\"":
            j = src.find(c, i + 1)
            if j < 0:
                raise ExprError("unterminated string", col, hint=f"close it with {c}")
            toks.append(("str", src[i + 1:j], col))
            i = j + 1
            continue
        if src[i:i + 2] in two:
            toks.append(("op", src[i:i + 2], col))
            i += 2
            continue
        if c in "+-*/%^()<>,[]!=":
            if c == "=":
                raise ExprError("single '=' is not allowed", col, hint="use '==' to compare; equations are given as {var: expr}")
            toks.append(("op", c, col))
            i += 1
            continue
        raise ExprError(f"unexpected character '{c}'", col)
    toks.append(("end", None, n + 1))
    return toks


# ----------------------------------------------------------------------------------------- parser
class _Parser:
    def __init__(self, src):
        self.toks = tokenize(src)
        self.i = 0
        self.depth = 0

    def peek(self, k=0):
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def is_op(self, v, k=0):
        t = self.peek(k)
        return t[0] == "op" and t[1] == v

    def is_kw(self, v, k=0):
        t = self.peek(k)
        return t[0] == "name" and t[1] == v

    def expect_op(self, v):
        t = self.next()
        if not (t[0] == "op" and t[1] == v):
            raise ExprError(f"expected '{v}' but found {_desc(t)}", t[2])
        return t

    def expect_kw(self, v):
        t = self.next()
        if not (t[0] == "name" and t[1] == v):
            raise ExprError(f"expected '{v}' but found {_desc(t)}", t[2],
                            hint="conditional syntax is: if <cond> then <a> else <b>" if v in ("then", "else") else "")
        return t

    def parse(self):
        e = self.expr()
        t = self.peek()
        if t[0] != "end":
            raise ExprError(f"unexpected {_desc(t)}", t[2], hint="missing operator or unbalanced parentheses?")
        return e

    def expr(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ExprError("expression nested too deeply", self.peek()[2])
        try:
            if self.is_kw("if"):
                return self.ifexpr()
            return self.orexpr()
        finally:
            self.depth -= 1

    def ifexpr(self):
        t_if = self.next()
        if self.is_op("("):
            # either if(c, a, b) or if (c) then a else b
            save = self.i
            self.next()
            args = [self.expr()]
            while self.is_op(","):
                self.next()
                args.append(self.expr())
            self.expect_op(")")
            if len(args) == 3 and not self.is_kw("then"):
                return ("if", args[0], args[1], args[2])
            if len(args) != 1:
                raise ExprError(f"if(...) takes 3 arguments (cond, a, b), got {len(args)}", t_if[2])
            self.i = save
        c = self.orexpr()
        self.expect_kw("then")
        a = self.expr()
        self.expect_kw("else")
        b = self.expr()
        return ("if", c, a, b)

    def orexpr(self):
        a = self.andexpr()
        while self.is_kw("or") or self.is_op("||"):
            self.next()
            a = ("or", a, self.andexpr())
        return a

    def andexpr(self):
        a = self.notexpr()
        while self.is_kw("and") or self.is_op("&&"):
            self.next()
            a = ("and", a, self.notexpr())
        return a

    def notexpr(self):
        if self.is_kw("not") or self.is_op("!"):
            self.next()
            self._deeper()
            try:
                return ("not", self.notexpr())
            finally:
                self.depth -= 1
        return self.cmp()

    def cmp(self):
        a = self.add()
        t = self.peek()
        if t[0] == "op" and t[1] in ("==", "!=", "<", "<=", ">", ">="):
            self.next()
            b = self.add()
            a = ("bin", t[1], a, b)
            t2 = self.peek()
            if t2[0] == "op" and t2[1] in ("==", "!=", "<", "<=", ">", ">="):
                raise ExprError("chained comparison", t2[2], hint="write (a < b) and (b < c)")
        return a

    def add(self):
        a = self.mul()
        while self.peek()[0] == "op" and self.peek()[1] in "+-" and len(self.peek()[1]) == 1:
            op = self.next()[1]
            a = ("bin", op, a, self.mul())
        return a

    def mul(self):
        a = self.unary()
        while self.peek()[0] == "op" and self.peek()[1] in ("*", "/", "%"):
            op = self.next()[1]
            a = ("bin", op, a, self.unary())
        return a

    def _deeper(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ExprError("expression nested too deeply", self.peek()[2])

    def unary(self):
        if self.is_op("-") or self.is_op("+"):
            neg = self.next()[1] == "-"
            self._deeper()
            try:
                a = self.unary()
            finally:
                self.depth -= 1
            return ("neg", a) if neg else a
        return self.power()

    def power(self):
        a = self.postfix()
        if self.is_op("^") or self.is_op("**"):
            self.next()
            self.depth += 1
            if self.depth > MAX_DEPTH:
                raise ExprError("expression nested too deeply", self.peek()[2])
            try:
                b = self.unary()
            finally:
                self.depth -= 1
            return ("bin", "^", a, b)
        return a

    def postfix(self):
        a = self.atom()
        if self.is_op("["):
            t = self.next()
            if a[0] != "var" or a[2] != 0:
                raise ExprError("[t-k] can only follow a variable name", t[2])
            tt = self.next()
            if not (tt[0] == "name" and tt[1] == "t"):
                raise ExprError(f"inside [...] expected 't' or 't-k', found {_desc(tt)}", tt[2], hint="e.g. x[t-1]")
            lag = 0
            if self.is_op("-"):
                self.next()
                kt = self.next()
                if kt[0] != "num" or kt[1] != int(kt[1]) or kt[1] < 0:
                    raise ExprError("lag must be a non-negative integer", kt[2], hint="e.g. x[t-2]")
                lag = int(kt[1])
            self.expect_op("]")
            a = ("var", a[1], lag)
        return a

    def atom(self):
        t = self.next()
        kind, v, col = t
        if kind == "num":
            return ("num", v)
        if kind == "str":
            return ("str", v)
        if kind == "op" and v == "(":
            e = self.expr()
            self.expect_op(")")
            return e
        if kind == "name":
            if v == "true":
                return ("bool", True)
            if v == "false":
                return ("bool", False)
            if v in KEYWORDS:
                raise ExprError(f"unexpected keyword '{v}'", col)
            if self.is_op("("):
                return self.call(v, col)
            return ("var", v, 0)
        raise ExprError(f"unexpected {_desc(t)}", col, hint="an operand (number, name, '(') was expected")

    def call(self, fn, col):
        if fn not in FUNCS:
            raise ExprError(f"unknown function '{fn}'", col, hint="available: " + ", ".join(sorted(FUNCS)))
        self.expect_op("(")
        args = []
        if not self.is_op(")"):
            args.append(self.expr())
            while self.is_op(","):
                self.next()
                args.append(self.expr())
        self.expect_op(")")
        arity = {"abs": (1, 1), "sqrt": (1, 1), "exp": (1, 1), "log": (1, 2), "floor": (1, 1), "ceil": (1, 1),
                 "round": (1, 2), "clip": (3, 3), "ifelse": (3, 3), "sign": (1, 1), "prev": (1, 1),
                 "lag": (2, 2), "min": (1, 99), "max": (1, 99)}[fn]
        if not arity[0] <= len(args) <= arity[1]:
            raise ExprError(f"{fn}() takes {arity[0]}" + (f"-{arity[1]}" if arity[1] != arity[0] else "")
                            + f" argument(s), got {len(args)}", col)
        if fn in ("prev", "lag"):
            if args[0][0] != "var" or args[0][2] != 0:
                raise ExprError(f"{fn}() needs a plain variable name as first argument", col)
            k = 1
            if fn == "lag":
                if args[1][0] != "num" or args[1][1] != int(args[1][1]) or args[1][1] < 0:
                    raise ExprError("lag(x, k): k must be a non-negative integer literal", col)
                k = int(args[1][1])
            return ("var", args[0][1], k)
        if fn == "ifelse":
            return ("if", args[0], args[1], args[2])
        return ("call", fn, args)


def _desc(t):
    if t[0] == "end":
        return "end of expression"
    return f"'{t[1]}'" if t[0] != "num" else f"number {t[1]:g}"


def parse(src):
    """Parse an expression string to an AST; raises ExprError with column + hint."""
    try:
        return _Parser(src).parse()
    except RecursionError:
        raise ExprError("expression nested too deeply")


def refs(ast, out=None):
    """Set of (name, lag) referenced by the AST (excluding built-ins t, dt)."""
    out = set() if out is None else out
    k = ast[0]
    if k == "var":
        if ast[1] not in ("t", "dt"):
            out.add((ast[1], ast[2]))
    elif k in ("neg", "not"):
        refs(ast[1], out)
    elif k in ("bin",):
        refs(ast[2], out)
        refs(ast[3], out)
    elif k in ("and", "or"):
        refs(ast[1], out)
        refs(ast[2], out)
    elif k == "if":
        for a in ast[1:]:
            refs(a, out)
    elif k == "call":
        for a in ast[2]:
            refs(a, out)
    return out


# ----------------------------------------------------------------------------------------- evaluator
def _num(v, what):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    raise EvalError(f"{what}: expected a number, got {type(v).__name__} {v!r}")


def _truth(v):
    if isinstance(v, str):
        raise EvalError(f"a string ({v!r}) cannot be used as a condition")
    return bool(v)


def evaluate(ast, lookup):
    """Interpret AST. `lookup(name, lag)` returns a value or raises EvalError."""
    k = ast[0]
    if k == "num":
        return ast[1]
    if k in ("str", "bool"):
        return ast[1]
    if k == "var":
        return lookup(ast[1], ast[2])
    if k == "neg":
        return -_num(evaluate(ast[1], lookup), "unary minus")
    if k == "not":
        return not _truth(evaluate(ast[1], lookup))
    if k == "and":
        return _truth(evaluate(ast[1], lookup)) and _truth(evaluate(ast[2], lookup))
    if k == "or":
        return _truth(evaluate(ast[1], lookup)) or _truth(evaluate(ast[2], lookup))
    if k == "if":
        return evaluate(ast[2], lookup) if _truth(evaluate(ast[1], lookup)) else evaluate(ast[3], lookup)
    if k == "bin":
        op = ast[1]
        a = evaluate(ast[2], lookup)
        b = evaluate(ast[3], lookup)
        if op in ("==", "!="):
            if isinstance(a, str) or isinstance(b, str):
                r = a == b
            else:
                r = _num(a, op) == _num(b, op)
            return r if op == "==" else not r
        if op in ("<", "<=", ">", ">="):
            if isinstance(a, str) and isinstance(b, str):
                x, y = a, b
            else:
                x, y = _num(a, op), _num(b, op)
            return {"<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y}[op]
        x, y = _num(a, f"'{op}'"), _num(b, f"'{op}'")
        try:
            if op == "+":
                r = x + y
            elif op == "-":
                r = x - y
            elif op == "*":
                r = x * y
            elif op == "/":
                if y == 0:
                    raise EvalError("division by zero")
                r = x / y
            elif op == "%":
                if y == 0:
                    raise EvalError("modulo by zero")
                r = x % y
            else:  # ^
                if x < 0 and y != int(y):
                    raise EvalError("negative number to a fractional power")
                if x == 0 and y < 0:
                    raise EvalError("zero to a negative power")
                r = x ** y
        except OverflowError:
            raise EvalError(f"numeric overflow in '{op}'")
        if isinstance(r, complex) or (isinstance(r, float) and math.isinf(r) and not (math.isinf(x) or math.isinf(y))):
            raise EvalError(f"numeric overflow in '{op}'")
        return r
    if k == "call":
        fn = ast[1]
        args = [evaluate(a, lookup) for a in ast[2]]
        try:
            if fn in ("min", "max"):
                nums = [_num(a, fn) for a in args]
                return min(nums) if fn == "min" else max(nums)
            x = _num(args[0], fn)
            if fn == "abs":
                return abs(x)
            if fn == "sqrt":
                if x < 0:
                    raise EvalError("sqrt of a negative number")
                return math.sqrt(x)
            if fn == "exp":
                return math.exp(x)
            if fn == "log":
                if x <= 0:
                    raise EvalError("log of a non-positive number")
                if len(args) == 2:
                    b = _num(args[1], "log base")
                    if b <= 0 or b == 1:
                        raise EvalError("bad log base")
                    return math.log(x, b)
                return math.log(x)
            if fn == "floor":
                return float(math.floor(x))
            if fn == "ceil":
                return float(math.ceil(x))
            if fn == "round":
                nd = int(_num(args[1], "round digits")) if len(args) == 2 else 0
                return float(round(x, nd))
            if fn == "clip":
                lo, hi = _num(args[1], "clip lo"), _num(args[2], "clip hi")
                return min(max(x, lo), hi)
            if fn == "sign":
                return float((x > 0) - (x < 0))
        except OverflowError:
            raise EvalError(f"numeric overflow in {fn}()")
        except (ValueError, TypeError) as e:
            raise EvalError(f"{fn}(): {e}")
    raise EvalError(f"unknown node {k}")


def eval_on(src_or_ast, values):
    """Evaluate an expression against a flat dict of current values (no lags). Used for event predicates."""
    ast = parse(src_or_ast) if isinstance(src_or_ast, str) else src_or_ast

    def lookup(name, lag):
        if lag:
            raise EvalError("lags are not available here")
        if name not in values:
            raise EvalError(f"unknown name '{name}'")
        return values[name]
    return evaluate(ast, lookup)
