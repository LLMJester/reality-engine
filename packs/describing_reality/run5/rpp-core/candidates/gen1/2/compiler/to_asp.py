"""Compile an R++ Core AST (JSON) to a clingo program, one program per scenario.

Semantics are those of the encoding below (an action-language / event-calculus
style discrete-time encoding); see lang/SPEC.md §5 for the prose statement.
All fluent values live in val(F, V, T); statics in st(S, V).

The program is produced in three parts (SPEC §5.4):
  static  - timeless facts and rules (sorts, statics, initial values, events)
  dynamic - rules whose head carries a step; every one is guarded by now(HeadStep)
  final   - constraints and claims (read the trajectory, never feed back into it)
The monolithic program is static + dynamic + final + "now(T) :- step(T)."; the
checker evaluates the same rules step by step (checker/check.py), which the
splitting-sequence theorem makes equivalent because no dynamic rule derives an
atom at step t from an atom at a later step.

Every compiler-internal variable contains "__"; user variables may not.
Decimal quantities are fixed-point integers (SPEC §2.1).
"""
import copy
from decimal import Decimal, InvalidOperation

from . import expr as X

TV = "T__"      # time variable of a rule
TV2 = "T2__"    # second time variable (BEFORE/AFTER)

ENGINE_STATIC = r"""
% ---- R++ Core engine (discrete time, inertia, effects, clamping) ----
bounded(F) :- range(F,_,_).
"""
ENGINE_DYNAMIC = r"""
hasset(F,T) :- eff_set(F,_,_,T), now(T).
hasadd(F,T) :- eff_add(F,_,_,_,T), now(T).
setadd(F,T) :- hasset(F,T), hasadd(F,T), now(T).
raw(F,V,T) :- eff_set(F,V,_,T), step(T), T > 0, not hasadd(F,T), now(T).
raw(F,V+S,T) :- val(F,V,T-1), hasadd(F,T), not hasset(F,T), step(T), T > 0, now(T),
                S = #sum{ D,L,I : eff_add(F,D,L,I,T) }.
val(F,V,T) :- val(F,V,T-1), inertial(F), not hasset(F,T), not hasadd(F,T), step(T), T > 0, now(T).
val(F,V,T) :- raw(F,V,T), not bounded(F), now(T).
val(F,V,T) :- raw(F,V,T), range(F,Lo,Hi), Lo <= V, V <= Hi, now(T).
val(F,Lo,T) :- raw(F,V,T), range(F,Lo,Hi), V < Lo, now(T).
val(F,Hi,T) :- raw(F,V,T), range(F,Lo,Hi), V > Hi, now(T).
val(F,V,0) :- init(F,V), now(0).
multi(F,T) :- val(F,V1,T), val(F,V2,T), V1 < V2, now(T).
"""
ENGINE_FINAL = r"""
hasval0(F) :- val(F,_,0).
noinit(F) :- inertial(F), not hasval0(F).
"""
ENGINE_DYN_PREDS = {"hasset", "hasadd", "setadd", "raw", "val", "multi", "eff_set", "eff_add",
                    "hasder", "badocc", "badval"}
SHOWS = ["#show val/3.", "#show viol/3.", "#show multi/2.", "#show setadd/2.",
         "#show badocc/2.", "#show badval/3.", "#show cholds/1.", "#show noinit/1."]

KEYWORD_ALIASES = {"MUST": "ALWAYS", "MUST_NOT": "NEVER", "MUST NOT": "NEVER",
                   "AT LEAST": "AT_LEAST", "AT MOST": "AT_MOST"}
DECIMAL_TYPES = {"decimal", "real", "float", "number"}
MAX_DECIMALS = 4


class CompileError(ValueError):
    pass


def decimals_of(decl):
    """Declared number of decimal places of a fluent/static (0 for non-decimal types)."""
    if decl is None or decl.get("type") not in DECIMAL_TYPES:
        return 0
    return int(decl.get("decimals", 2))


def scaled(v, d, what="value"):
    """A literal value as an ASP constant; numbers are scaled by 10**d."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)) or (isinstance(v, str) and _numeric(v)):
        try:
            q = Decimal(str(v)) * (Decimal(10) ** d)
        except InvalidOperation:
            raise CompileError(f"{what}: {v!r} is not a number")
        if q != q.to_integral_value():
            raise CompileError(f"{what}: {v} has more decimal places than the {d} declared "
                               f"(declare \"type\": \"decimal\", \"decimals\": {_places(v)}, or round it)")
        return str(int(q))
    return str(v)


def _numeric(s):
    try:
        Decimal(s)
        return s.strip().lstrip("-").replace(".", "", 1).isdigit()
    except InvalidOperation:
        return False


def _places(v):
    s = str(v)
    return len(s.split(".")[1]) if "." in s else 0


def asp_const(v):
    return scaled(v, 0)


def rescale(term, s_from, s_to):
    if s_from == s_to:
        return term
    if s_to > s_from:
        return f"({term}*{10 ** (s_to - s_from)})"
    return f"({term}/{10 ** (s_from - s_to)})"


class Model:
    """Declarations index built from an AST."""

    def __init__(self, ast):
        self.ast = ast
        self.sorts = {}
        for s in ast.get("sorts", []):
            self.sorts[s["name"]] = s
        self.instances = {}
        for s in ast.get("sorts", []):
            for c in s.get("instances", []):
                self.instances[c] = s["name"]
        self.fluents = {f["name"]: f for f in ast.get("fluents", [])}
        self.statics = {s["name"]: s for s in ast.get("statics", [])}
        self.actions = {a["name"]: a for a in ast.get("actions", [])}

    def dom(self, var, sort):
        if sort not in self.sorts:
            raise CompileError(f"unknown sort {sort!r}")
        return f"sort_{sort}({var})"

    def decl(self, name):
        return self.fluents.get(name) or self.statics.get(name)


class Compiler:
    def __init__(self, model):
        self.m = model
        self.static = []
        self.dynamic = []
        self.final = []
        self.layer = "dynamic"
        self.dyn_preds = set(ENGINE_DYN_PREDS)
        self.n = 0

    def fresh(self, p="V"):
        self.n += 1
        return f"{p}__{self.n}"

    def fresh_pred(self, p):
        name = self.fresh(p).lower()
        if self.layer == "dynamic":
            self.dyn_preds.add(name)
        return name

    def rule(self, head, body, htime=None):
        """Emit a rule. In the dynamic layer, a rule whose head has a step is guarded by now(step)."""
        body = list(body)
        if htime is None:
            target = self.static if self.layer == "dynamic" else self.final
        elif self.layer == "dynamic":
            # step-wise solving relies on the head's step being its last argument
            if not (head.endswith(f",{htime})") or head.endswith(f"({htime})")):
                raise AssertionError(f"internal: dynamic head {head} does not end with its step {htime}")
            target = self.dynamic
            body.append(f"now({htime})")
        else:
            target = self.final
        target.append(f"{head} :- {', '.join(body)}." if body else f"{head}.")

    # ---------------- expressions ----------------
    def term_args(self, args, T, env, lits):
        out = []
        for a in args:
            t, l = self.val(a, T, env, 0)
            lits.extend(l)
            out.append(t)
        return out

    @staticmethod
    def fterm(name, args):
        return f"{name}({','.join(args)})" if args else name

    def val(self, e, T, env, scale=None):
        """Value term and body literals; rescaled to `scale` decimals if given."""
        t, lits, s = self.vals(e, T, env)
        if scale is not None:
            t = rescale(t, s, scale)
        return t, lits

    def vals(self, e, T, env):
        """Return (term, literals, scale) for a value expression at step T."""
        k = e[0]
        m = self.m
        if k == "int":
            return str(e[1]), [], 0
        if k == "dec":
            return str(e[1]), [], e[2]
        if k == "now":
            return str(T), [], 0
        if k == "var":
            if e[1] not in env:
                raise CompileError(f"unbound variable {e[1]}")
            return e[1], [], 0
        if k == "name":
            s = e[1]
            if s in ("true", "false") or s in m.instances:
                return s, [], 0
            if s in m.fluents or s in m.statics:
                return self.vals(("call", s, []), T, env)
            hint = (" (the current step is the built-in name t)"
                    if s in ("step", "time", "day", "week", "month", "minute", "hour", "year",
                             "current_step", "current_day", "current_time") else "")
            raise CompileError(f"unknown name {s!r}{hint}")
        if k == "call":
            name, args = e[1], e[2]
            lits = []
            targs = self.term_args(args, T, env, lits)
            v = self.fresh()
            if name in m.fluents:
                self._arity(m.fluents[name], args, name)
                lits.append(f"val({self.fterm(name, targs)},{v},{T})")
            elif name in m.statics:
                self._arity(m.statics[name], args, name)
                lits.append(f"st({self.fterm(name, targs)},{v})")
            else:
                hint = " (the current step is the built-in name t, not a function)" \
                    if name in ("time", "step", "now", "current_time") else ""
                raise CompileError(f"unknown fluent/static {name!r}{hint}")
            return v, lits, decimals_of(m.decl(name))
        if k == "bin":
            a, la, sa = self.vals(e[2], T, env)
            b, lb, sb = self.vals(e[3], T, env)
            op = e[1]
            s = max(sa, sb)
            if op in ("+", "-"):
                return f"({rescale(a, sa, s)}{op}{rescale(b, sb, s)})", la + lb, s
            if op == "%":
                return f"({rescale(a, sa, s)}\\{rescale(b, sb, s)})", la + lb, s
            if op == "*":
                return rescale(f"({a}*{b})", sa + sb, s), la + lb, s
            if op == "/":
                return f"({rescale(a, sa, s + sb)}/{b})", la + lb, s
            raise CompileError(f"unknown operator {op}")
        if k == "neg":
            a, la, sa = self.vals(e[1], T, env)
            return f"-({a})", la, sa
        if k == "fn":
            fn, args = e[1], e[2]
            parts = [self.vals(a, T, env) for a in args]
            lits = [l for _, ls, _ in parts for l in ls]
            s = max([p[2] for p in parts] or [0])
            ts = [rescale(t, ps, s) for t, _, ps in parts]
            if fn == "abs":
                if len(ts) != 1:
                    raise CompileError("abs takes one argument")
                return f"|{ts[0]}|", lits, s
            if len(ts) < 2:
                raise CompileError(f"{fn} takes at least two arguments")
            v = self.fresh()
            lits.append(f"{v} = #{fn}{{{';'.join(ts)}}}")
            return v, lits, s
        if k == "ite":
            return self.ite(e, T, env)
        if k == "agg":
            _, which, var, sort, value, cond = e
            if var in env:
                raise CompileError(f"aggregate variable {var} shadows an outer variable")
            inner_env = dict(env)
            inner_env[var] = sort
            elem = [self.m.dom(var, sort)]
            if cond is not None:
                elem += self.boolean(cond, T, inner_env, allow_agg=False)
            v = self.fresh("A")
            if which == "count":
                return v, [f"{v} = #count{{ {var} : {', '.join(elem)} }}"], 0
            vt, vl, vs = self.vals(value, T, inner_env)
            if any("#" in l for l in vl):
                raise CompileError("nested aggregates are not supported")
            elem = elem + vl
            return v, [f"{v} = #sum{{ {vt},{var} : {', '.join(elem)} }}"], vs
        if k in ("cmp", "and", "or", "not"):
            raise CompileError(f"boolean expression used as a value: {X.to_text(e)}")
        raise CompileError(f"unknown expression node {k}")

    def ite(self, e, T, env):
        """if C then A else B: an auxiliary predicate with one rule per branch."""
        _, c, a, b = e
        fv = sorted(v for v in X.free_vars(e) if v in env)
        name = self.fresh_pred("ite")
        at, al, sa = self.vals(a, TV, env)
        bt, bl, sb = self.vals(b, TV, env)
        s = max(sa, sb)
        doms = [self.m.dom(v, env[v]) for v in fv]
        head = lambda t: f"{name}({','.join(fv + [t, TV])})"   # step is always the last argument
        self.rule(head(rescale(at, sa, s)), doms + [f"step({TV})"] + self.boolean(c, TV, env) + al, TV)
        cond = self.aux(c, TV, env)
        self.rule(head(rescale(bt, sb, s)), doms + [f"step({TV})", f"not {cond}"] + bl, TV)
        v = self.fresh()
        return v, [f"{name}({','.join(fv + [v, str(T)])})"], s

    def _arity(self, decl, args, name):
        if len(decl.get("args", [])) != len(args):
            raise CompileError(f"{name} expects {len(decl.get('args', []))} args, got {len(args)}")

    def aux(self, e, T, env):
        """Define aux predicate for boolean e; return its atom (positive)."""
        fv = sorted(v for v in X.free_vars(e) if v in env)
        name = self.fresh_pred("aux")
        doms = [self.m.dom(v, env[v]) for v in fv]
        body = doms + [f"step({TV})"] + self.boolean(e, TV, env)
        self.rule(f"{name}({','.join(fv + [TV])})", body, TV)
        return f"{name}({','.join(fv + [str(T)])})"

    def boolean(self, e, T, env, allow_agg=True):
        k = e[0]
        m = self.m
        if k == "and":
            out = []
            for p in e[1]:
                out += self.boolean(p, T, env, allow_agg)
            return out
        if k == "or":
            fv = sorted(v for v in X.free_vars(e) if v in env)
            name = self.fresh_pred("aux")
            for p in e[1]:
                doms = [self.m.dom(v, env[v]) for v in fv]
                body = doms + [f"step({TV})"] + self.boolean(p, TV, env)
                self.rule(f"{name}({','.join(fv + [TV])})", body, TV)
            return [f"{name}({','.join(fv + [str(T)])})"]
        if k == "not":
            return [f"not {self.aux(e[1], T, env)}"]
        if k == "ite":
            _, c, a, b = e
            return self.boolean(("or", [("and", [c, a]), ("and", [("not", c), b])]), T, env, allow_agg)
        if k == "cmp":
            a, la, sa = self.vals(e[2], T, env)
            b, lb, sb = self.vals(e[3], T, env)
            s = max(sa, sb)
            lits = la + lb
            if not allow_agg and any("#" in l for l in lits):
                raise CompileError("nested aggregates are not supported")
            return lits + [f"{rescale(a, sa, s)} {e[1]} {rescale(b, sb, s)}"]
        if k == "call":
            name = e[1]
            decl = m.fluents.get(name) or m.statics.get(name)
            if decl is None:
                raise CompileError(f"unknown fluent/static {name!r}")
            if decl.get("type", "bool") != "bool":
                raise CompileError(f"non-boolean {name} used as a condition")
            v, lits = self.val(e, T, env)
            return lits + [f"{v} = true"]
        if k == "name":
            if e[1] == "true":
                return []
            if e[1] == "false":
                return ["1 != 1"]
            if e[1] in m.fluents or e[1] in m.statics:
                return self.boolean(("call", e[1], []), T, env, allow_agg)
            raise CompileError(f"{e[1]!r} is not a condition")
        raise CompileError(f"not a condition: {X.to_text(e)}")

    # ---------------- declarations ----------------
    def params_env(self, params):
        env = {}
        for v, s in (params or {}).items():
            if not v[:1].isupper():
                raise CompileError(f"parameter {v} must be capitalised")
            if "__" in v:
                raise CompileError(f"parameter {v}: names containing '__' are reserved")
            if s not in self.m.sorts:
                raise CompileError(f"unknown sort {s!r} for parameter {v}")
            env[v] = s
        return env

    def doms(self, env):
        return [self.m.dom(v, s) for v, s in env.items()]

    def declarations(self, ast):
        m = self.m
        for s in ast.get("sorts", []):
            if "instances" in s:
                for c in s["instances"]:
                    self.rule(f"sort_{s['name']}({c})", [])
            elif "range" in s:
                lo, hi = s["range"]
                self.rule(f"sort_{s['name']}({lo}..{hi})", [])
            else:
                raise CompileError(f"sort {s['name']} needs instances or range")
        for f in ast.get("fluents", []):
            vs = [f"X__{i}" for i in range(len(f.get("args", [])))]
            env = dict(zip(vs, f.get("args", [])))
            head = self.fterm(f["name"], vs)
            d = self.doms(env)
            dec = decimals_of(f)
            t = f.get("type", "bool")
            if t in DECIMAL_TYPES and not 0 <= dec <= MAX_DECIMALS:
                raise CompileError(f"fluent {f['name']}: decimals must be 0..{MAX_DECIMALS}")
            if not f.get("derived"):
                self.rule(f"inertial({head})", d)
            if (t == "int" or t in DECIMAL_TYPES) and f.get("range"):
                lo, hi = (scaled(x, dec, f"range of {f['name']}") for x in f["range"])
                self.rule(f"range({head},{lo},{hi})", d)
            if t == "bool":
                self.rule(f"badval({head},V__,{TV})", d + [f"val({head},V__,{TV})", "V__ != true", "V__ != false"],
                          TV)
            elif t in m.sorts:
                self.rule(f"badval({head},V__,{TV})", d + [f"val({head},V__,{TV})", f"not sort_{t}(V__)"], TV)
            if f.get("derived") and "default" in f:
                dv = scaled(f["default"], dec, f"default of {f['name']}")
                self.rule(f"val({head},{dv},{TV})", d + [f"step({TV})", f"not hasder({head},{TV})"], TV)
        for s in ast.get("statics", []):
            dec = decimals_of(s)
            if s.get("type") in DECIMAL_TYPES and not 0 <= dec <= MAX_DECIMALS:
                raise CompileError(f"static {s['name']}: decimals must be 0..{MAX_DECIMALS}")
            for fact in s.get("facts", []):
                args = [asp_const(a) for a in fact.get("args", [])]
                self.rule(f"st({self.fterm(s['name'], args)},{scaled(fact['value'], dec, s['name'])})", [])
            if "default" in s:
                vs = [f"X__{i}" for i in range(len(s.get("args", [])))]
                env = dict(zip(vs, s.get("args", [])))
                head = self.fterm(s["name"], vs)
                given = self.fresh("given").lower()
                for fact in s.get("facts", []):
                    args = [asp_const(a) for a in fact.get("args", [])]
                    self.rule(f"{given}({self.fterm(s['name'], args)})", [])
                d = self.doms(env)
                self.rule(f"st({head},{scaled(s['default'], dec, s['name'])})", d + [f"not {given}({head})"])

    def initial(self, entries):
        for e in entries:
            env = self.params_env(e.get("params"))
            lits = []
            targs = self.term_args([X.parse(a) for a in e.get("args", [])], 0, env, lits)
            if lits:
                raise CompileError("initial-state arguments must be constants or parameters")
            dec = decimals_of(self.m.fluents.get(e["fluent"]))
            head = f"init({self.fterm(e['fluent'], targs)},{scaled(e['value'], dec, e['fluent'])})"
            self.rule(head, self.doms(env))

    def effects(self, effects, body, env, delay, rid, inst):
        for ef in effects:
            f = ef.get("fluent")
            if f not in self.m.fluents:
                raise CompileError(f"effect on unknown fluent {f!r} in {rid}")
            if self.m.fluents[f].get("derived"):
                raise CompileError(f"effect on derived fluent {f!r} in {rid}")
            dec = decimals_of(self.m.fluents[f])
            lits = list(body)
            args = [X.parse(a) for a in ef.get("args", [])]
            self._arity(self.m.fluents[f], args, f)
            targs = self.term_args(args, TV, env, lits)
            head_f = self.fterm(f, targs)
            when = f"{TV}+{delay}"
            lits.append(f"step({when})")
            if "set" in ef:
                v, vl = self.val(X.parse(ef["set"]), TV, env, dec)
                self.rule(f"eff_set({head_f},{v},{rid},{when})", lits + vl, when)
            elif "add" in ef:
                v, vl = self.val(X.parse(ef["add"]), TV, env, dec)
                self.rule(f"eff_add({head_f},{v},{rid},{inst},{when})", lits + vl, when)
            else:
                raise CompileError(f"effect in {rid} needs 'set' or 'add'")

    def inst(self, env):
        return f"inst({','.join(env)})" if env else "inst"

    def laws(self, ast):
        for law in ast.get("laws", []):
            try:
                env = self.params_env(law.get("params"))
                body = self.doms(env) + [f"step({TV})"]
                if law.get("when") not in (None, ""):
                    body += self.boolean(X.parse(law["when"]), TV, env)
                if law.get("unless") not in (None, ""):
                    body.append(f"not {self.aux(X.parse(law['unless']), TV, env)}")
                delay = int(law.get("delay", 1))
                if delay < 1:
                    raise CompileError("delay must be >= 1")
                self.effects(law.get("effects", []), body, env, delay, law["id"], self.inst(env))
            except (CompileError, X.ExprError) as e:
                raise CompileError(f"law {law.get('id')}: {e}")

    def actions(self, ast):
        for a in ast.get("actions", []):
            try:
                env = self.params_env(a.get("params"))
                occ = f"occ({self.fterm(a['name'], list(env))},{TV})"
                body = self.doms(env) + [f"step({TV})", occ]
                if a.get("pre") not in (None, ""):
                    pre = self.aux(X.parse(a["pre"]), TV, env)
                    self.rule(f"badocc({self.fterm(a['name'], list(env))},{TV})",
                              self.doms(env) + [f"step({TV})", occ, 'not ' + pre], TV)
                    body.append(pre)
                self.effects(a.get("effects", []), body, env, 1, a["name"], self.inst(env))
            except (CompileError, X.ExprError) as e:
                raise CompileError(f"action {a.get('name')}: {e}")

    def derived(self, ast):
        for d in ast.get("derived", []):
            try:
                f = d["fluent"]
                if f not in self.m.fluents or not self.m.fluents[f].get("derived"):
                    raise CompileError(f"targets {f!r}, which is not a derived fluent")
                env = self.params_env(d.get("params"))
                body = self.doms(env) + [f"step({TV})"]
                if d.get("when") not in (None, ""):
                    body += self.boolean(X.parse(d["when"]), TV, env)
                if d.get("unless") not in (None, ""):
                    body.append(f"not {self.aux(X.parse(d['unless']), TV, env)}")
                args = [X.parse(a) for a in d.get("args", [])]
                self._arity(self.m.fluents[f], args, f)
                targs = self.term_args(args, TV, env, body)
                head = self.fterm(f, targs)
                v, vl = self.val(X.parse(d.get("value", "true")), TV, env, decimals_of(self.m.fluents[f]))
                self.rule(f"val({head},{v},{TV})", body + vl, TV)
                self.rule(f"hasder({head},{TV})", body, TV)
            except (CompileError, X.ExprError) as e:
                raise CompileError(f"derived {d.get('id')}: {e}")

    def constraints(self, ast):
        for c in ast.get("constraints", []):
            try:
                self._constraint(c)
            except (CompileError, X.ExprError) as e:
                raise CompileError(f"constraint {c.get('id')}: {e}")

    def _constraint(self, c):
        cid = c["id"]
        kw = c["keyword"].upper()
        kw = KEYWORD_ALIASES.get(kw, kw)
        env = self.params_env(c.get("params"))
        inst = self.inst(env)
        base = self.doms(env) + [f"step({TV})"]
        unless = []
        if c.get("unless") not in (None, ""):
            unless = [f"not {self.aux(X.parse(c['unless']), TV, env)}"]
        head = f"viol({cid},{inst},{TV})"
        if kw == "NEVER":
            body = base + self.boolean(X.parse(c["cond"]), TV, env) + unless
        elif kw == "ALWAYS":
            body = base + [f"not {self.aux(X.parse(c['cond']), TV, env)}"] + unless
        elif kw == "ONLY":
            body = base + self.boolean(X.parse(c["cond"]), TV, env) + \
                [f"not {self.aux(X.parse(c['if']), TV, env)}"] + unless
        elif kw in ("AT_MOST", "AT_LEAST", "EXACTLY"):
            over = c["over"].split(" in ")
            var, sort = over[0].strip(), over[1].strip()
            agg = ("agg", "count", var, sort, None, X.parse(c["cond"]))
            n, lits = self.val(agg, TV, env)
            op = {"AT_MOST": ">", "AT_LEAST": "<", "EXACTLY": "!="}[kw]
            body = base + lits + [f"{n} {op} {int(c['n'])}"] + unless
        elif kw in ("BEFORE", "AFTER"):
            a, b = (c["a"], c["b"]) if kw == "BEFORE" else (c["b"], c["a"])
            aa = self.aux(X.parse(a), TV2, env)
            seen = self.fresh("seen").lower()
            fv = list(env)
            self.rule(f"{seen}({','.join(fv + [TV])})",
                      self.doms(env) + [f"step({TV})", f"step({TV2})", f"{TV2} < {TV}", aa], TV)
            body = base + self.boolean(X.parse(b), TV, env) + \
                [f"not {seen}({','.join(fv + [TV])})"] + unless
        else:
            raise CompileError(f"unknown constraint keyword {c['keyword']!r}")
        self.rule(head, body, TV)

    def claims(self, claims):
        for c in claims:
            about = c.get("about", {})
            cid = c["id"]
            try:
                e = X.parse(about["expr"])
                if X.free_vars(e):
                    raise CompileError(f"free variables {sorted(X.free_vars(e))}")
                t = about.get("time", 0)
                if t == "any":
                    self.rule(f"cholds({cid})", [f"step({TV})", self.aux(e, TV, {})])
                elif t == "all":
                    self.rule(f"cfail({cid})", [f"step({TV})", f"not {self.aux(e, TV, {})}"])
                    self.rule(f"cholds({cid})", [f"not cfail({cid})"])
                else:
                    self.rule(f"cholds({cid})", [self.aux(e, int(t), {})])
            except (CompileError, X.ExprError) as e:
                raise CompileError(f"claim {cid}: {e}")


def apply_interventions(ast, interventions):
    ast = copy.deepcopy(ast)
    for iv in interventions or []:
        op = iv.get("op")
        if op == "set_static":
            st = next((s for s in ast.get("statics", []) if s["name"] == iv["static"]), None)
            if st is None:
                hint = ""
                if any(f["name"] == iv["static"] for f in ast.get("fluents", [])):
                    hint = " (it is a fluent: change it with the scenario's init, an event, or add_law)"
                raise CompileError(f"intervention on unknown static {iv['static']!r}{hint}")
            args = [str(a) for a in iv.get("args", [])]
            facts = [f for f in st.get("facts", []) if [str(a) for a in f.get("args", [])] != args]
            facts.append({"args": iv.get("args", []), "value": iv["value"]})
            st["facts"] = facts
        elif op == "disable":
            rid = iv["id"]
            known = ({r.get("id") for k in ("laws", "derived", "constraints") for r in ast.get(k, [])}
                     | {a.get("name") for a in ast.get("actions", [])})
            if rid not in known:
                raise CompileError(f"intervention disables unknown id {rid!r}")
            for key in ("laws", "derived", "constraints"):
                ast[key] = [r for r in ast.get(key, []) if r.get("id") != rid]
            ast["actions"] = [a for a in ast.get("actions", []) if a.get("name") != rid]
        elif op == "add_law":
            ast.setdefault("laws", []).append(iv["law"])
        elif op == "add_derived":
            ast.setdefault("derived", []).append(iv["derived"])
        elif op == "add_constraint":
            ast.setdefault("constraints", []).append(iv["constraint"])
        else:
            raise CompileError(f"unknown intervention op {op!r}")
    return ast


def scenario_horizon(ast, sc, claims):
    h = int(sc.get("horizon", ast.get("horizon", 10)))
    for ev in sc.get("events", []):
        h = max(h, int(ev["time"]) + 1)
    for c in claims:
        t = c.get("about", {}).get("time", 0)
        if isinstance(t, int):
            h = max(h, t)
    return h


class Compiled:
    """The three parts of a scenario program (see module docstring)."""

    def __init__(self, static, dynamic, final, horizon, max_delay, dyn_preds):
        self.static, self.dynamic, self.final = static, dynamic, final
        self.horizon, self.max_delay, self.dyn_preds = horizon, max_delay, dyn_preds

    def monolithic(self):
        return "\n".join([ENGINE_STATIC, ENGINE_DYNAMIC, ENGINE_FINAL, f"now({TV}) :- step({TV})."]
                         + self.static + self.dynamic + self.final + SHOWS) + "\n"

    def step_program(self):
        """Static + dynamic part; ground with now(t) and the previous steps' atoms as facts."""
        return "\n".join([ENGINE_STATIC, ENGINE_DYNAMIC] + self.static + self.dynamic) + "\n"

    def final_program(self):
        """Static + final part; ground with the whole trajectory as facts."""
        return "\n".join([ENGINE_STATIC, ENGINE_FINAL] + self.static + self.final + SHOWS) + "\n"


def compile_parts(ast, sc, claims):
    base = apply_interventions(ast, sc.get("interventions"))
    m = Model(base)
    c = Compiler(m)
    h = scenario_horizon(base, sc, claims)
    c.rule(f"step(0..{h})", [])
    c.declarations(base)
    # initial state: scenario overrides replace base entries for the same fluent+args
    over = {(e["fluent"], tuple(str(a) for a in e.get("args", []))) for e in sc.get("init", [])}
    inits = [e for e in base.get("initial", [])
             if (e["fluent"], tuple(str(a) for a in e.get("args", []))) not in over]
    c.initial(inits + list(sc.get("init", [])))
    c.laws(base)
    c.actions(base)
    c.derived(base)
    for ev in sc.get("events", []):
        a = ev["action"]
        if a not in m.actions:
            raise CompileError(f"scenario {sc['id']}: unknown action {a!r}")
        args = [asp_const(x) for x in ev.get("args", [])]
        if len(args) != len(m.actions[a].get("params", {})):
            raise CompileError(f"scenario {sc['id']}: action {a} expects "
                               f"{len(m.actions[a].get('params', {}))} args")
        c.rule(f"occ({Compiler.fterm(a, args)},{int(ev['time'])})", [])
    c.layer = "final"
    c.constraints(base)
    c.claims(claims)
    max_delay = max([int(l.get("delay", 1)) for l in base.get("laws", [])] + [1])
    return Compiled(c.static, c.dynamic, c.final, h, max_delay, c.dyn_preds)


def compile_scenario(ast, sc, claims):
    """Return the monolithic clingo program text for one scenario and its horizon."""
    p = compile_parts(ast, sc, claims)
    return p.monolithic(), p.horizon
