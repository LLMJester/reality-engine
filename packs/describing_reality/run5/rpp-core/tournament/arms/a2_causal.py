"""A2: causal graph + structural equations (see tournament/DESIGN.md, reports/TOURNAMENT_PLAN_v2.md §2).

Representation (JSON):
  {"variables": [{"name", "type": bool|int|real|category, "unit", "exogenous": bool, "value"?,
                  "distribution"?: {"kind": uniform|normal|choice, ...}}],
   "edges": [{"from", "to", "sign": "+"|"-"|"?", "lag"?: int}],
   "equations": {var: expr},
   "time"?: {"steps": int, "dt": number}}

Executor: pure Python, expressions parsed by tournament.arms._expr (no eval). Supports static and
time-stepped evaluation with lags, do-interventions, seeded Monte Carlo over exogenous distributions,
and qualitative sign propagation along edge paths where equations are missing. Broken parts become
holes; everything that does not depend on a hole still computes.
"""
import copy
import json
import math
import random
import time
from collections import deque
from pathlib import Path

from tournament.arms import _expr
from tournament.arms.base import Arm, ArmState

TYPES = ("bool", "int", "real", "category")
MAX_STEPS = 5000
MAX_SAMPLES = 2000
MAX_WORK = 3_000_000          # variable-evaluations per tool call
QUAL_SECONDS = 2.0           # wall-clock budget for qualitative propagation per tool call
QUAL_VISITS = 200_000        # node-visit budget for the in-loop path search per tool call
EXAMPLE_VISITS = 5_000       # node visits per target spent on listing example paths
QUAL_WORDS = {"+": "+", "-": "-", "up": "+", "down": "-", "increase": "+", "decrease": "-"}


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.6g}"
    if isinstance(v, int):
        return str(v)
    return repr(v) if isinstance(v, str) else str(v)


def _maybe_json(x):
    if isinstance(x, str):
        s = x.strip()
        if s[:1] in "{[":
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


# ============================================================================ compilation
class Compiled:
    """Validated view of a representation: what can run, and the holes."""

    def __init__(self, rep):
        self.vars = {}          # name -> spec dict (normalised)
        self.order_decl = []
        self.edges = []         # (from, to, sign, lag)
        self.eqs = {}           # name -> ast
        self.eq_src = {}
        self.holes = []         # dicts {where, error, hint}
        self.warnings = []
        self.broken = {}        # var -> reason it cannot be computed numerically
        self.steps = 0
        self.dt = 1.0
        self.timed = False
        self._build(rep if isinstance(rep, dict) else {})
        if not isinstance(rep, dict):
            self.hole("representation", f"expected a JSON object, got {type(rep).__name__}",
                      "use {variables:[...], edges:[...], equations:{...}}")

    def hole(self, where, error, hint=""):
        self.holes.append({"where": where, "error": error, "hint": hint})

    # ---------------------------------------------------------------- build
    def _build(self, rep):
        variables = rep.get("variables") or []
        if not isinstance(variables, list):
            self.hole("variables", "must be a list", "variables: [{name, type, ...}]")
            variables = []
        for i, v in enumerate(variables):
            self._add_var(i, v)
        edges = rep.get("edges") or []
        if not isinstance(edges, list):
            self.hole("edges", "must be a list", "edges: [{from, to, sign}]")
            edges = []
        for i, e in enumerate(edges):
            self._add_edge(i, e)
        eqs = rep.get("equations") or {}
        if not isinstance(eqs, dict):
            self.hole("equations", "must be an object {var: expr}", 'equations: {"y": "2*x"}')
            eqs = {}
        for name, src in eqs.items():
            self._add_eq(name, src)
        self._time(rep.get("time"))
        self._check_vars()
        self._cycles()

    def _add_var(self, i, v):
        where = f"variables[{i}]"
        if not isinstance(v, dict):
            return self.hole(where, "must be an object", "{name, type, unit, exogenous}")
        name = v.get("name")
        if not isinstance(name, str) or not name:
            return self.hole(where, "missing 'name'", "every variable needs a name")
        where = f"variable '{name}'"
        try:
            toks = _expr.tokenize(name)
            ok = len(toks) == 2 and toks[0][0] == "name" and toks[0][1] == name
        except _expr.ExprError:
            ok = False
        if not ok or name in _expr.RESERVED or name in _expr.FUNCS:
            return self.hole(where, "invalid name", "use letters, digits, '_' (not a keyword, function or t/dt)")
        if name in self.vars:
            return self.hole(where, "declared twice; later declaration ignored", "rename or remove one")
        typ = v.get("type", "real")
        if typ not in TYPES:
            self.hole(where, f"unknown type {typ!r}; treated as real", "type is one of " + "|".join(TYPES))
            typ = "real"
        spec = {"name": name, "type": typ, "unit": v.get("unit", ""), "exogenous": bool(v.get("exogenous", False))}
        if "value" in v and v["value"] is not None:
            val, err = _coerce(v["value"], typ)
            if err:
                self.hole(where, f"bad value: {err}", f"value must fit type {typ}")
            else:
                spec["value"] = val
        dist = v.get("distribution")
        if dist is not None:
            err = _check_dist(dist, typ)
            if err:
                self.hole(where, f"bad distribution: {err}",
                          "uniform{low,high} | normal{mean,sd[,min,max]} | choice{options[,weights]}")
            elif not spec["exogenous"]:
                self.warnings.append(f"{where}: distribution ignored (only exogenous variables are sampled)")
            else:
                spec["distribution"] = dist
        self.vars[name] = spec
        self.order_decl.append(name)

    def _add_edge(self, i, e):
        where = f"edges[{i}]"
        if not isinstance(e, dict):
            return self.hole(where, "must be an object", "{from, to, sign}")
        a, b = e.get("from"), e.get("to")
        if a not in self.vars or b not in self.vars:
            missing = [x for x in (a, b) if x not in self.vars]
            return self.hole(f"edge {a}->{b}", f"unknown variable(s) {missing}", "declare variables before edges")
        sign = str(e.get("sign", "?"))
        if sign not in ("+", "-", "?"):
            self.hole(f"edge {a}->{b}", f"bad sign {sign!r}; treated as '?'", "sign is '+', '-' or '?'")
            sign = "?"
        lag = e.get("lag", 0) or 0
        if not isinstance(lag, int) or lag < 0:
            self.hole(f"edge {a}->{b}", f"bad lag {lag!r}; treated as 0", "lag is a non-negative integer")
            lag = 0
        self.edges.append((a, b, sign, lag))

    def _add_eq(self, name, src):
        where = f"equation for '{name}'"
        if name not in self.vars:
            self.hole(where, "variable not declared; equation ignored", f"add_variable name={name} first")
            return
        try:
            ast = _expr.parse(src if not isinstance(src, (int, float)) or isinstance(src, bool) else str(src))
        except _expr.ExprError as e:
            self.eq_src[name] = src
            self.hole(where, str(e), e.hint or "check the syntax in the guide")
            self.broken[name] = "equation does not parse"
            return
        bad = sorted({n for n, _ in _expr.refs(ast) if n not in self.vars})
        self.eq_src[name] = src
        if bad:
            self.hole(where, f"references undeclared variable(s) {bad}", "declare them or fix the spelling")
            self.broken[name] = f"equation references undeclared {bad}"
            return
        self.eqs[name] = ast

    def _time(self, tm):
        if tm is None:
            return
        if not isinstance(tm, dict):
            return self.hole("time", "must be an object {steps, dt}", 'time: {"steps": 12, "dt": 1}')
        steps = tm.get("steps", 0)
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            self.hole("time", f"bad steps {steps!r}; using 0", "steps is a non-negative integer")
            steps = 0
        if steps > MAX_STEPS:
            self.hole("time", f"steps {steps} > {MAX_STEPS}; capped", "")
            steps = MAX_STEPS
        dt = tm.get("dt", 1.0)
        if not isinstance(dt, (int, float)) or isinstance(dt, bool) or dt <= 0:
            self.hole("time", f"bad dt {dt!r}; using 1", "dt is a positive number")
            dt = 1.0
        self.steps, self.dt, self.timed = steps, float(dt), True

    def _check_vars(self):
        uses_lag = any(lag > 0 for ast in self.eqs.values() for _, lag in _expr.refs(ast))
        if uses_lag and not self.timed:
            self.warnings.append("equations use lags but no time is set; evaluated at step 0 only "
                                 "(add time: {steps, dt})")
        for name, s in self.vars.items():
            if name in self.eqs or name in self.broken:
                continue
            if s["exogenous"]:
                if "value" not in s and "distribution" not in s:
                    self.hole(f"variable '{name}'", "exogenous variable has no value, distribution or equation",
                              "give it a value or a distribution")
                    self.broken[name] = "no value"
            elif "value" in s:
                self.hole(f"variable '{name}'", "endogenous variable has no equation; held constant at its value",
                          f"set_equation var={name}")
            else:
                self.hole(f"variable '{name}'", "endogenous variable has no equation (numeric value unavailable; "
                          "qualitative sign propagation still works)", f"set_equation var={name}")
                self.broken[name] = "no equation"
        # lagged references to variables without an initial value -> warn (failure shows at step 0)
        for name, ast in self.eqs.items():
            for ref, lag in _expr.refs(ast):
                if lag > 0 and "value" not in self.vars[ref] and ref in self.eqs:
                    self.warnings.append(f"equation for '{name}' uses {ref}[t-{lag}] but '{ref}' has no initial "
                                         "value; step 0 will fail for it")
        # equation dependencies not reflected by edges
        edge_set = {(a, b) for a, b, _, _ in self.edges}
        for name, ast in self.eqs.items():
            for ref in sorted({r for r, _ in _expr.refs(ast)}):
                if ref != name and (ref, name) not in edge_set:
                    self.warnings.append(f"equation for '{name}' uses '{ref}' but there is no edge {ref}->{name}")

    def deps0(self, name):
        if name not in self.eqs:
            return set()
        return {r for r, lag in _expr.refs(self.eqs[name]) if lag == 0}

    def _cycles(self):
        """Contemporaneous (lag-0) cycles cannot be computed: mark them as holes."""
        order, state = [], {}
        cyc = set()

        def visit(root):           # iterative DFS (no recursion limit on long dependency chains)
            if state.get(root) == 2:
                return
            stack, work = [root], [iter(sorted(self.deps0(root)))]
            state[root] = 1
            while work:
                d = next(work[-1], None)
                if d is None:
                    work.pop()
                    n = stack.pop()
                    state[n] = 2
                    order.append(n)
                    continue
                if d not in self.vars:
                    continue
                st = state.get(d)
                if st == 2:
                    continue
                if st == 1:
                    cyc.update(stack[stack.index(d):])
                    continue
                state[d] = 1
                stack.append(d)
                work.append(iter(sorted(self.deps0(d))))
        for n in self.order_decl:
            visit(n)
        for n in sorted(cyc):
            self.hole(f"equation for '{n}'", "part of a same-step cycle (no lag breaks it)",
                      "reference one variable at the previous step, e.g. x[t-1]")
            self.broken[n] = "same-step cycle"
        self.order = order


def _coerce(v, typ):
    try:
        if typ == "category":
            if isinstance(v, str):
                return v, None
            return None, f"category value must be a string, got {v!r}"
        if isinstance(v, str):
            return None, f"expected {typ}, got string {v!r}"
        if typ == "bool":
            return bool(v), None
        if typ == "int":
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return None, f"non-finite value {v!r}"
            return int(round(f)), None
        return float(v), None
    except (TypeError, ValueError) as e:
        return None, str(e)


def _check_dist(d, typ):
    if not isinstance(d, dict):
        return "must be an object"
    k = d.get("kind")
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    if k == "uniform":
        lo, hi = d.get("low", d.get("min")), d.get("high", d.get("max"))
        if not (num(lo) and num(hi)) or lo > hi:
            return "uniform needs numbers low <= high"
    elif k == "normal":
        if not (num(d.get("mean")) and num(d.get("sd"))) or d["sd"] < 0:
            return "normal needs mean and sd >= 0"
    elif k == "choice":
        opts = d.get("options")
        if not isinstance(opts, list) or not opts:
            return "choice needs a non-empty options list"
        w = d.get("weights")
        if w is not None and (not isinstance(w, list) or len(w) != len(opts) or not all(num(x) and x >= 0 for x in w)
                              or sum(w) <= 0):
            return "weights must be non-negative numbers, one per option"
        for o in opts:
            if _coerce(o, typ)[1]:
                return f"option {o!r} does not fit type {typ}"
    else:
        return f"unknown kind {k!r}"
    if typ == "category" and k != "choice":
        return "category variables need a choice distribution"
    return None


def _dist_point(d):
    k = d["kind"]
    if k == "uniform":
        return (d.get("low", d.get("min")) + d.get("high", d.get("max"))) / 2
    if k == "normal":
        m = d["mean"]
        if "min" in d:
            m = max(m, d["min"])
        if "max" in d:
            m = min(m, d["max"])
        return m
    w = d.get("weights") or [1] * len(d["options"])
    return d["options"][max(range(len(w)), key=lambda i: w[i])]


def _dist_sample(d, rng):
    k = d["kind"]
    if k == "uniform":
        return rng.uniform(d.get("low", d.get("min")), d.get("high", d.get("max")))
    if k == "normal":
        x = rng.gauss(d["mean"], d["sd"])
        if "min" in d:
            x = max(x, d["min"])
        if "max" in d:
            x = min(x, d["max"])
        return x
    return rng.choices(d["options"], weights=d.get("weights"))[0]


# ============================================================================ simulation
def simulate(cg, steps=None, do=None, rng=None):
    """Run the structural equations. Returns (hist: name -> list, failed: name -> (step, reason)).

    do: {var: {"value"|"scale"|"add": x, "from": step}}
    """
    steps = cg.steps if steps is None else steps
    do = do or {}
    hist = {n: [] for n in cg.vars}
    failed = {n: (0, r) for n, r in cg.broken.items()}
    base = {}
    for n, s in cg.vars.items():
        if n not in cg.eqs and n not in cg.broken:
            if s["exogenous"] and "distribution" in s:
                v = _dist_sample(s["distribution"], rng) if rng is not None else (
                    s["value"] if "value" in s else _dist_point(s["distribution"]))
                base[n] = _coerce(v, s["type"])[0]
            elif "value" in s:
                base[n] = s["value"]
    order = [n for n in cg.order if n in cg.vars]
    for t in range(steps + 1):
        for n in order:
            spec = cg.vars[n]
            iv = do.get(n)
            active = iv is not None and t >= iv.get("from", 0)
            if n in failed and not (active and "value" in iv):
                continue
            try:
                if active and "value" in iv:
                    v = iv["value"]
                else:
                    if cg.timed and t == 0 and n in cg.eqs and not spec["exogenous"] and "value" in spec:
                        v = spec["value"]
                    elif n in cg.eqs:
                        v = _expr.evaluate(cg.eqs[n], _lookup(cg, hist, failed, t, base))
                    elif n in base:
                        v = base[n]
                    else:
                        raise _expr.EvalError("no value")
                    if active:
                        if "scale" in iv:
                            v = _expr._num(v, "scale intervention") * iv["scale"]
                        if "add" in iv:
                            v = _expr._num(v, "add intervention") + iv["add"]
                v2, err = _coerce(v, spec["type"])
                if err:
                    raise _expr.EvalError(f"result does not fit type {spec['type']}: {err}")
                if isinstance(v2, float) and (math.isnan(v2) or math.isinf(v2)):
                    raise _expr.EvalError(f"non-finite result {v2}")
                hist[n].append(v2)
            except _expr.EvalError as e:
                failed[n] = (t, str(e))
                hist[n].append(None)
    return hist, failed


def _lookup(cg, hist, failed, t, base):
    def lookup(name, lag):
        if name == "t":
            return float(t)
        if name == "dt":
            return cg.dt
        ts = t - lag
        if ts < 0:
            if name in base:
                return base[name]
            s = cg.vars[name]
            if "value" in s:
                return s["value"]
            raise _expr.EvalError(f"{name}[t-{lag}] refers to before step 0 and '{name}' has no initial value "
                                  f"(hint: give '{name}' a value)")
        h = hist[name]
        if ts >= len(h) or h[ts] is None:
            if name in failed:
                fs, why = failed[name]
                raise _expr.EvalError(f"depends on '{name}', which has no value ({why})")
            raise _expr.EvalError(f"'{name}' is not available at step {ts}")
        return h[ts]
    return lookup


# ============================================================================ qualitative
# Sign algebra on {"+", "-", "?"}; None means "no path" (reported as "0"). The net effect of src on
# dst is the join of the signs of all simple directed paths src -> dst. It is computed without
# enumerating paths: dynamic programming over the condensation (DAG of strongly connected components)
# of the relevant subgraph, which is linear in its size. Only inside a feedback loop (a non-trivial
# SCC) are simple paths searched, under a shared node-visit and wall-clock budget; if that budget
# runs out, the loop is solved by fixpoint iteration over walks instead -- a sound over-approximation
# (it can only turn a definite sign into "?", never the reverse) -- and a note says so.

def _smul(a, b):
    if a == "?" or b == "?":
        return "?"
    return "+" if a == b else "-"


def _sjoin(a, b):
    if a is None:
        return b
    if b is None or a == b:
        return a
    return "?"


class QualBudget:
    """Shared budget for one tool call's qualitative work."""

    def __init__(self, seconds=QUAL_SECONDS, visits=QUAL_VISITS):
        self.deadline = time.monotonic() + seconds
        self.visits = visits
        self.seconds = seconds
        self.hit = None

    def expired(self):
        if self.hit is None and time.monotonic() > self.deadline:
            self.hit = f"{self.seconds:g} s time budget"
        return self.hit is not None

    def spend(self):
        """Count one node visit; False once a budget is exhausted."""
        if self.hit is not None:
            return False
        self.visits -= 1
        if self.visits < 0:
            self.hit = "node-visit budget"
            return False
        if not (self.visits & 255):
            return not self.expired()
        return True


def _adjacency(cg):
    adj = getattr(cg, "_qual_adj", None)
    if adj is None:
        out, inc = {}, {}
        for a, b, s, _lag in cg.edges:
            out.setdefault(a, []).append((b, s))
            inc.setdefault(b, []).append((a, s))
        adj = cg._qual_adj = (out, inc)
    return adj


def _reach(start, nbrs, allowed=None):
    seen = set(start)
    todo = list(start)
    while todo:
        n = todo.pop()
        for b, _s in nbrs.get(n, ()):
            if b not in seen and (allowed is None or b in allowed):
                seen.add(b)
                todo.append(b)
    return seen


def _sccs(src, out, rel):
    """Iterative Tarjan over rel (all of it reachable from src); components in topological order."""
    index, low, on, stack, comps = {src: 0}, {src: 0}, {src}, [src], []
    counter = 1
    work = [(src, iter(out.get(src, ())))]
    while work:
        n, it = work[-1]
        advanced = False
        for b, _s in it:
            if b not in rel:
                continue
            if b not in index:
                index[b] = low[b] = counter
                counter += 1
                stack.append(b)
                on.add(b)
                work.append((b, iter(out.get(b, ()))))
                advanced = True
                break
            if b in on:
                low[n] = min(low[n], index[b])
        if advanced:
            continue
        work.pop()
        if work:
            p = work[-1][0]
            low[p] = min(low[p], low[n])
        if low[n] == index[n]:
            comp = []
            while True:
                x = stack.pop()
                on.discard(x)
                comp.append(x)
                if x == n:
                    break
            comps.append(comp)
    comps.reverse()
    return comps


def propagate(cg, src, targets=None, budget=None):
    """Net sign ("+", "-", "?") on each relevant node of moving src up.

    Nodes without a directed path from src are absent. Restricted to nodes on some path to
    `targets` when given. Returns (signs, notes)."""
    budget = budget or QualBudget()
    out, inc = _adjacency(cg)
    rel = _reach([src], out)
    if targets is not None:
        rel = _reach([t for t in targets if t in rel], inc, allowed=rel) | {src}
    comps = _sccs(src, out, rel)
    comp_of = {n: i for i, c in enumerate(comps) for n in c}
    tset = set(targets) if targets is not None else None
    val_in = {src: "+"}
    val, notes = {}, []
    for ci, comp in enumerate(comps):
        if len(comp) == 1:
            n = comp[0]
            if val_in.get(n) is not None:
                val[n] = val_in[n]
        else:
            res, note = _solve_loop(comp, ci, comp_of, out, val_in, tset, budget)
            val.update(res)
            if note:
                notes.append(note)
        for n in comp:
            v = val.get(n)
            if v is None:
                continue
            for b, s in out.get(n, ()):
                if b in rel and comp_of[b] != ci:
                    val_in[b] = _sjoin(val_in.get(b), _smul(v, s))
    val[src] = "+"
    return val, notes


def _solve_loop(comp, ci, comp_of, out, val_in, tset, budget):
    """Signs on the members of one strongly connected component, given the entry signs."""
    inner = {n: [(b, s) for b, s in out.get(n, ()) if comp_of.get(b) == ci] for n in comp}
    interest = {n for n in comp if (tset is not None and n in tset)
                or any(comp_of.get(b, ci) != ci for b, _ in out.get(n, ()))} or set(comp)
    entries = [(e, val_in[e]) for e in comp if val_in.get(e) is not None]
    if any(s == "?" for _, s in entries):
        return {n: "?" for n in comp}, None       # every member is reachable from that entry
    acc = {}
    n_mixed = 0
    complete = True

    def add(n, s):
        nonlocal n_mixed
        old = acc.get(n)
        new = _sjoin(old, s)
        if new != old:
            acc[n] = new
            if new == "?" and n in interest:
                n_mixed += 1

    for e, s0 in entries:
        if n_mixed >= len(interest):
            break                                 # everything of interest is already "?"
        if budget.expired():
            complete = False
            break
        add(e, s0)
        on_path = {e}
        stack = [(e, s0, iter(inner[e]))]
        while stack and n_mixed < len(interest):
            n, s, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                on_path.discard(n)
                continue
            b, es = nxt
            if b in on_path:
                continue
            if not budget.spend():
                complete = False
                break
            ns = _smul(s, es)
            add(b, ns)
            on_path.add(b)
            stack.append((b, ns, iter(inner[b])))
        if not complete:
            break
    if complete:
        return acc, None
    # budget exhausted: fixpoint over walks (each value changes at most twice, so O(edges))
    w = dict(entries)
    todo = deque(w)
    while todo:
        n = todo.popleft()
        for b, es in inner[n]:
            nv = _sjoin(w.get(b), _smul(w[n], es))
            if nv != w.get(b):
                w[b] = nv
                todo.append(b)
    note = (f"{budget.hit} reached inside a feedback loop of {len(comp)} variables; signs through it are "
            "conservative ('?' where loops of both signs exist)")
    return w, note


def example_paths(cg, src, dst, budget=None, limit=6, visits=EXAMPLE_VISITS, want_signs=1):
    """Up to `limit` simple paths src -> dst, one of each sign first (searching on, within the visit
    budget, until `want_signs` different path signs are seen). Returns (paths, complete)."""
    out, inc = _adjacency(cg)
    co = _reach([dst], inc)
    if src not in co:
        return [], True
    first, rest = {}, []
    n_found = 0
    complete = True
    on_path, path, signs = {src}, [src], []
    stack = [iter(out.get(src, ()))]
    while stack:
        nxt = next(stack[-1], None)
        if nxt is None:
            stack.pop()
            on_path.discard(path.pop())
            if signs:
                signs.pop()
            continue
        b, s = nxt
        if b in on_path or b not in co:
            continue
        visits -= 1
        if visits < 0 or (budget is not None and not (visits & 63) and budget.expired()):
            complete = False
            break
        if b == dst:
            n_found += 1
            p = (path + [b], signs + [s])
            ps = _path_sign(p[1])
            if ps not in first:
                first[ps] = p
            elif len(rest) < limit:
                rest.append(p)
            if n_found > limit and len(first) >= want_signs:
                break                             # enough to show; more exist
            continue
        on_path.add(b)
        path.append(b)
        signs.append(s)
        stack.append(iter(out.get(b, ())))
    shown = (list(first.values()) + rest)[:limit]
    return shown, complete and n_found <= limit


def _path_sign(signs):
    r = "+"
    for s in signs:
        if s == "?":
            return "?"
        if s == "-":
            r = "-" if r == "+" else "+"
    return r


def _qual_text(cg, src, dst, sign, direction, notes, budget):
    if sign is None:
        return "0", f"no directed path {src} -> {dst} (no effect)"
    net = sign
    if direction == "-" and net in "+-":
        net = "-" if net == "+" else "+"
    paths, complete = example_paths(cg, src, dst, budget, want_signs=2 if sign == "?" else 1)
    shown = "; ".join("->".join(p) + f" [{_path_sign(s)}]" for p, s in paths)
    if complete:
        why = f"{len(paths)} path(s): {shown}"
    elif paths:
        why = f"paths e.g.: {shown} (more not listed)"
    else:
        why = "example paths not listed (budget)"
    if net == "?":
        why += "; mixed/unknown signs"
    if notes:
        why += "; note: " + "; ".join(notes)
    return net, why


def qualitative(cg, src, dst, direction="+", budget=None):
    """Net sign of the effect on dst of moving src in `direction` ('+' or '-'), via edge paths.

    Bounded by `budget` (a QualBudget; default 2 s and QUAL_VISITS node visits)."""
    if src == dst:
        return direction, "the intervened variable itself"
    budget = budget or QualBudget()
    signs, notes = propagate(cg, src, [dst], budget)
    return _qual_text(cg, src, dst, signs.get(dst), direction, notes, budget)


# ============================================================================ the arm
def _tool(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


_VAR_PROPS = {
    "name": {"type": "string"},
    "type": {"type": "string", "enum": list(TYPES)},
    "unit": {"type": "string"},
    "exogenous": {"type": "boolean"},
    "value": {"description": "constant (exogenous) or initial value at step 0 (endogenous, time mode)"},
    "distribution": {"type": "object", "description": "uniform{low,high} | normal{mean,sd[,min,max]} | "
                                                      "choice{options[,weights]}; exogenous only"},
}
_QUERY = {"description": "string target name, or object {targets?: [names], at?: 'final'|'all'|step, "
                         "samples?: int, seed?: int, events?: [expr], steps?: int}"}


class CausalArm(Arm):
    name = "A2"
    buildable = True
    executable = True
    guide_file = "a2_guide.md"
    mutating_tools = ("set_graph", "add_variable", "add_edge", "set_equation")

    def guide(self):
        p = Path(__file__).resolve().parent.parent / "prompts" / "a2_guide.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def tools(self):
        return [
            _tool("set_graph", "Replace the whole representation with a JSON graph "
                  "{variables, edges, equations, time?}.", {"graph": {"type": "object"}}, ["graph"]),
            _tool("add_variable", "Add or replace one variable.", _VAR_PROPS, ["name"]),
            _tool("add_edge", "Add or replace a causal edge.", {
                "from": {"type": "string"}, "to": {"type": "string"},
                "sign": {"type": "string", "enum": ["+", "-", "?"]}, "lag": {"type": "integer"}},
                ["from", "to"]),
            _tool("set_equation", "Set the structural equation for a variable (empty expr removes it).",
                  {"var": {"type": "string"}, "expr": {"type": "string"}}, ["var", "expr"]),
            _tool("evaluate", "Compute variable values (static, over time, or Monte Carlo).",
                  {"query": _QUERY}),
            _tool("intervene", "Apply do-interventions and compare targets with the baseline. "
                  "do = {var: value | {value|scale|add: x, from?: step} | '+'|'-' (qualitative)}.",
                  {"do": {"type": "object"}, "query": _QUERY}, ["do"]),
            _tool("show", "Show the current representation and its holes.", {}),
        ]

    def probe_tools(self):
        return [t for t in self.tools() if t["function"]["name"] in ("intervene", "evaluate")]

    def new_state(self):
        return ArmState(rep={"variables": [], "edges": [], "equations": {}})

    def load(self, rep, given_gold=False):
        st = self.new_state()
        st.rep = copy.deepcopy(rep) if isinstance(rep, dict) else {"variables": [], "edges": [], "equations": {}}
        if not isinstance(rep, dict):
            st.log.append(("load", None, f"representation was {type(rep).__name__}, not an object; started empty"))
        st.given_gold = given_gold
        return st

    # ---------------------------------------------------------------- dispatch
    def handle(self, state, name, args):
        try:
            args = _maybe_json(args) if args is not None else {}
            if not isinstance(args, dict):
                return f"ERROR: arguments must be an object (hint: see the {name} schema)"
            fn = getattr(self, "_t_" + str(name), None)
            if fn is None:
                return (f"ERROR: unknown tool '{name}' (hint: available: "
                        + ", ".join(t["function"]["name"] for t in self.tools()) + ")")
            if state.given_gold and name in ("set_graph", "add_variable", "add_edge", "set_equation"):
                return "ERROR: the representation is read-only in this mode (hint: use evaluate, intervene, show)"
            out = fn(state, args)
        except Exception as e:  # never raise
            out = f"ERROR: internal {type(e).__name__}: {e} (hint: check the arguments)"
        state.log.append((name, args, out[:200]))
        return out

    def _rep(self, state):
        r = state.rep
        r.setdefault("variables", [])
        r.setdefault("edges", [])
        r.setdefault("equations", {})
        if not isinstance(r["variables"], list):
            r["variables"] = []
        if not isinstance(r["edges"], list):
            r["edges"] = []
        if not isinstance(r["equations"], dict):
            r["equations"] = {}
        return r

    def _summary(self, state):
        cg = Compiled(state.rep)
        s = f"{len(cg.vars)} variables, {len(cg.edges)} edges, {len(cg.eqs)} equations; {len(cg.holes)} hole(s)"
        return s, cg

    def _t_set_graph(self, state, args):
        g = _maybe_json(args.get("graph", args))
        if not isinstance(g, dict):
            return "ERROR: graph must be an object (hint: {variables:[...], edges:[...], equations:{...}})"
        state.rep = copy.deepcopy(g)
        s, cg = self._summary(state)
        return "graph set: " + s + _holes_text(cg, 12)

    def _t_add_variable(self, state, args):
        r = self._rep(state)
        name = args.get("name")
        if not isinstance(name, str) or not name:
            return "ERROR: add_variable needs a 'name' (hint: name='price')"
        v = {k: args[k] for k in ("name", "type", "unit", "exogenous", "value", "distribution") if k in args}
        v["distribution"] = _maybe_json(v["distribution"]) if "distribution" in v else None
        if v["distribution"] is None:
            v.pop("distribution")
        idx = [i for i, x in enumerate(r["variables"]) if isinstance(x, dict) and x.get("name") == name]
        if idx:
            r["variables"][idx[0]] = v
            msg = f"variable '{name}' replaced"
        else:
            r["variables"].append(v)
            msg = f"variable '{name}' added"
        cg = Compiled(state.rep)
        mine = [h for h in cg.holes if f"'{name}'" in h["where"] and h["where"].startswith("variable")]
        return msg + ("".join(f"\n  problem: {h['error']} (hint: {h['hint']})" for h in mine) if mine else "")

    def _t_add_edge(self, state, args):
        r = self._rep(state)
        a, b = args.get("from"), args.get("to")
        e = {"from": a, "to": b, "sign": args.get("sign", "?")}
        if args.get("lag"):
            e["lag"] = args["lag"]
        r["edges"] = [x for x in r["edges"] if not (isinstance(x, dict) and x.get("from") == a and x.get("to") == b)]
        r["edges"].append(e)
        cg = Compiled(state.rep)
        mine = [h for h in cg.holes if h["where"] == f"edge {a}->{b}"]
        if mine:
            return f"edge {a}->{b} stored but unusable: " + "; ".join(f"{h['error']} (hint: {h['hint']})" for h in mine)
        return f"edge {a}->{b} ({e['sign']}" + (f", lag {e['lag']}" if e.get("lag") else "") + ") set"

    def _t_set_equation(self, state, args):
        r = self._rep(state)
        var, src = args.get("var"), args.get("expr")
        if not isinstance(var, str) or not var:
            return "ERROR: set_equation needs 'var' (hint: var='y', expr='2*x')"
        if src is None or (isinstance(src, str) and not src.strip()):
            r["equations"].pop(var, None)
            return f"equation for '{var}' removed"
        r["equations"][var] = src
        cg = Compiled(state.rep)
        mine = [h for h in cg.holes if h["where"] == f"equation for '{var}'"]
        if mine:
            return f"equation for '{var}' stored but is a hole: " + "; ".join(
                f"{h['error']} (hint: {h['hint']})" for h in mine)
        return f"equation for '{var}' set: {var} = {src}"

    def _t_show(self, state, args):
        return self.render(state)

    def render(self, state):
        cg = Compiled(state.rep)
        lines = ["VARIABLES"]
        for n in cg.order_decl:
            s = cg.vars[n]
            extra = []
            if s["unit"]:
                extra.append(f"unit={s['unit']}")
            if "value" in s:
                extra.append(f"value={_fmt(s['value'])}")
            if "distribution" in s:
                extra.append("dist=" + json.dumps(s["distribution"]))
            lines.append(f"  {n}: {s['type']}, {'exogenous' if s['exogenous'] else 'endogenous'}"
                         + (", " + ", ".join(extra) if extra else ""))
        lines.append("EDGES")
        for a, b, sg, lag in cg.edges:
            lines.append(f"  {a} -> {b} ({sg}" + (f", lag {lag}" if lag else "") + ")")
        lines.append("EQUATIONS")
        for n, src in cg.eq_src.items():
            lines.append(f"  {n} = {src}" + ("   [HOLE]" if n not in cg.eqs else ""))
        if cg.timed:
            lines.append(f"TIME steps={cg.steps} dt={_fmt(cg.dt)}")
        txt = "\n".join(lines) + _holes_text(cg, 30)
        return txt[:60000]

    # ---------------------------------------------------------------- queries
    def _parse_query(self, cg, q):
        q = _maybe_json(q)
        if q is None:
            q = {}
        if isinstance(q, str):
            q = {"targets": [x.strip() for x in q.split(",") if x.strip()]}
        if isinstance(q, list):
            q = {"targets": q}
        if not isinstance(q, dict):
            raise ValueError("query must be a name, a list of names or an object")
        tg = q.get("targets", q.get("target"))
        if isinstance(tg, str):
            tg = [tg]
        tg = list(tg) if tg else [n for n in cg.order_decl][:30]
        at = q.get("at", "final")
        steps = q.get("steps")
        if steps is not None:
            if not isinstance(steps, int) or steps < 0:
                raise ValueError("steps must be a non-negative integer")
            steps = min(steps, MAX_STEPS)
        n_steps = cg.steps if steps is None else steps
        if not cg.timed and steps is None:
            n_steps = 0
        has_dist = any("distribution" in s for s in cg.vars.values())
        samples = q.get("samples")
        if samples is None:
            samples = 500 if has_dist else 0
        samples = max(0, min(int(samples), MAX_SAMPLES))
        notes = []
        if samples:
            work = max(1, len(cg.vars) * (n_steps + 1))
            if work * samples > MAX_WORK:
                samples = max(1, MAX_WORK // work)
                notes.append(f"samples reduced to {samples} (work limit)")
        events = q.get("events") or []
        if isinstance(events, str):
            events = [events]
        return {"targets": tg, "at": at, "steps": n_steps, "samples": samples, "seed": int(q.get("seed", 0) or 0),
                "events": events, "notes": notes}

    def _t_evaluate(self, state, args):
        cg = Compiled(state.rep)
        try:
            q = self._parse_query(cg, args.get("query", {k: v for k, v in args.items() if k != "query"}))
        except (ValueError, TypeError) as e:
            return f"ERROR: bad query: {e} (hint: query={{\"targets\": [\"y\"], \"at\": \"final\"}})"
        out = self._run_and_report(cg, q, do=None)
        return out + _holes_text(cg, 8)

    def _t_intervene(self, state, args):
        cg = Compiled(state.rep)
        do_raw = _maybe_json(args.get("do"))
        if not isinstance(do_raw, dict) or not do_raw:
            return "ERROR: do must be a non-empty object (hint: do={\"price\": 12} or {\"price\": {\"scale\": 1.1}})"
        try:
            q = self._parse_query(cg, args.get("query"))
        except (ValueError, TypeError) as e:
            return f"ERROR: bad query: {e} (hint: query={{\"targets\": [\"y\"]}})"
        do, qual, errs = {}, {}, []
        for var, val in do_raw.items():
            if var not in cg.vars:
                errs.append(f"do: unknown variable '{var}' (ignored)")
                continue
            typ = cg.vars[var]["type"]
            val = _maybe_json(val)
            if isinstance(val, str) and typ != "category" and val.strip().lower() in QUAL_WORDS:
                qual[var] = QUAL_WORDS[val.strip().lower()]
                continue
            if isinstance(val, dict):
                iv = {}
                for k in ("value", "scale", "add"):
                    if k in val:
                        iv[k] = val[k]
                if not iv:
                    errs.append(f"do {var}: object needs value, scale or add (ignored)")
                    continue
                if "value" in iv:
                    c, err = _coerce(iv["value"], typ)
                    if err:
                        errs.append(f"do {var}: {err} (ignored)")
                        continue
                    iv["value"] = c
                for k in ("scale", "add"):
                    if k in iv and (not isinstance(iv[k], (int, float)) or isinstance(iv[k], bool)):
                        errs.append(f"do {var}: {k} must be a number (ignored)")
                        iv = None
                        break
                if iv is None:
                    continue
                if "from" in val:
                    iv["from"] = int(val["from"])
                do[var] = iv
            else:
                c, err = _coerce(val, typ)
                if err:
                    errs.append(f"do {var}: {err} (ignored)")
                    continue
                do[var] = {"value": c}
        if not do and not qual:
            return "ERROR: no usable intervention. " + "; ".join(errs)
        parts = [f"note: {e}" for e in errs]
        if do:
            parts.append(self._run_and_report(cg, q, do=do))
        # qualitative propagation for every intervened variable
        base_hist = simulate(cg, steps=q["steps"])[0] if do else None
        qlines = []
        budget = QualBudget()       # shared by every qualitative query in this call
        for var, iv in list(do.items()) + list(qual.items()):
            if isinstance(iv, str):
                direction = iv
            else:
                direction = "+"
                b = base_hist[var][-1] if base_hist and base_hist[var] else None
                if "value" in iv and isinstance(b, (int, float)) and isinstance(iv["value"], (int, float)):
                    d = iv["value"] - b
                    direction = "+" if d > 0 else "-" if d < 0 else "0"
                elif "scale" in iv and isinstance(b, (int, float)):
                    d = b * iv["scale"] - b
                    direction = "+" if d > 0 else "-" if d < 0 else "0"
                elif "add" in iv:
                    direction = "+" if iv["add"] > 0 else "-" if iv["add"] < 0 else "0"
            tgts = [t for t in q["targets"] if t in cg.vars and t != var]
            signs, notes = propagate(cg, var, tgts, budget) if tgts and direction != "0" else ({}, [])
            for tgt in tgts:
                if direction == "0":
                    qlines.append(f"  {var} unchanged -> {tgt}: 0")
                    continue
                net, why = _qual_text(cg, var, tgt, signs.get(tgt), direction, notes, budget)
                arrow = " up" if direction == "+" else " down"
                qlines.append(f"  {var}{arrow} -> {tgt}: {net}   ({why})")
        if qlines:
            parts.append("QUALITATIVE (edge signs):\n" + "\n".join(qlines))
        return "\n".join(parts) + _holes_text(cg, 8)

    def _run_and_report(self, cg, q, do):
        lines = list(f"note: {n}" for n in q["notes"])
        targets = [t for t in q["targets"] if t in cg.vars]
        for t in q["targets"]:
            if t not in cg.vars:
                lines.append(f"{t}: unknown variable (hint: declared: {', '.join(cg.order_decl[:20])})")
        events = []
        for ev in q["events"]:
            try:
                events.append((ev, _expr.parse(ev)))
            except _expr.ExprError as e:
                lines.append(f"event {ev!r}: {e}")
        at = q["at"]
        steps = q["steps"]

        def pick(h):
            if not h:
                return None
            if at == "final" or at is None:
                return h[-1]
            if isinstance(at, int) and not isinstance(at, bool):
                return h[at] if 0 <= at < len(h) else None
            return h[-1]

        def ev_values(hist):
            return {n: pick(h) for n, h in hist.items()}

        if not q["samples"]:
            base = simulate(cg, steps=steps) if do else None
            hist, failed = simulate(cg, steps=steps, do=do)
            if do:
                lines.append("do(" + ", ".join(f"{k}={_fmt_iv(v)}" for k, v in do.items()) + ")"
                             + (f", at step {at}" if isinstance(at, int) else ""))
            for t in targets:
                h = hist[t]
                if at == "all":
                    lines.append(f"{t}: " + _traj(h) + _fail_note(failed, t))
                    if do:
                        lines.append(f"{t} (baseline): " + _traj(base[0][t]))
                    continue
                v = pick(h)
                if do:
                    b = pick(base[0][t])
                    delta = ""
                    if isinstance(v, (int, float)) and isinstance(b, (int, float)) and not isinstance(v, bool):
                        delta = f"  (delta {_fmt(v - b)})"
                    lines.append(f"{t}: {_fmt(v)}  baseline {_fmt(b)}{delta}" + _fail_note(failed, t))
                else:
                    lines.append(f"{t} = {_fmt(v)}" + _fail_note(failed, t))
            for src, ast in events:
                lines.append(f"event {src}: {_event(ast, ev_values(hist))}"
                             + (f"  baseline {_event(ast, ev_values(base[0]))}" if do else ""))
            if cg.timed or steps:
                lines.append(f"(steps 0..{steps}, dt={_fmt(cg.dt)})")
            return "\n".join(lines)

        # Monte Carlo
        n, seed = q["samples"], q["seed"]
        samp = {t: [] for t in targets}
        bsamp = {t: [] for t in targets}
        evc = [[0, 0, 0, 0] for _ in events]   # true, n, base-true, base-n
        fails = {}
        for i in range(n):
            hist, failed = simulate(cg, steps=steps, do=do, rng=random.Random(seed * 1_000_003 + i))
            bh = simulate(cg, steps=steps, rng=random.Random(seed * 1_000_003 + i))[0] if do else None
            for t in targets:
                samp[t].append(pick(hist[t]))
                if do:
                    bsamp[t].append(pick(bh[t]))
                if t in failed and t not in fails:
                    fails[t] = failed[t]
            for j, (_, ast) in enumerate(events):
                r = _event_bool(ast, ev_values(hist))
                if r is not None:
                    evc[j][0] += r
                    evc[j][1] += 1
                if do:
                    r = _event_bool(ast, ev_values(bh))
                    if r is not None:
                        evc[j][2] += r
                        evc[j][3] += 1
        lines.append(f"Monte Carlo: {n} samples, seed {seed}"
                     + (", do(" + ", ".join(f"{k}={_fmt_iv(v)}" for k, v in do.items()) + ")" if do else "")
                     + (f", at step {at}" if isinstance(at, int) else f", at final step {steps}"))
        for t in targets:
            lines.append(f"{t}: {_summ(samp[t])}" + (_fail_note(fails, t) if fails.get(t) else ""))
            if do:
                lines.append(f"  baseline: {_summ(bsamp[t])}")
                d = [a - b for a, b in zip(samp[t], bsamp[t])
                     if isinstance(a, (int, float)) and isinstance(b, (int, float))]
                if d:
                    lines.append(f"  delta: {_summ(d)}; P(delta>0)={sum(x > 0 for x in d) / len(d):.3f} "
                                 f"P(delta<0)={sum(x < 0 for x in d) / len(d):.3f}")
        for (src, _), c in zip(events, evc):
            p = f"{c[0] / c[1]:.3f} (n={c[1]})" if c[1] else "undefined"
            lines.append(f"P({src}) = {p}" + (f"  baseline {c[2] / c[3]:.3f}" if do and c[3] else ""))
        return "\n".join(lines)

    # ---------------------------------------------------------------- validity
    def validity(self, state):
        cg = Compiled(state.rep)
        usable = bool(cg.vars)
        executable = False
        errors = []
        if usable:
            try:
                hist, failed = simulate(cg)
                executable = any(h and h[-1] is not None for h in hist.values())
                for n, (t, why) in failed.items():
                    if n not in cg.broken:
                        errors.append(f"variable '{n}' failed at step {t}: {why}")
            except Exception as e:
                errors.append(f"internal {type(e).__name__}: {e}")
        holes = [f"{h['where']}: {h['error']}" for h in cg.holes]
        return {"usable": usable, "executable": executable, "holes": holes, "errors": errors,
                "warnings": cg.warnings[:20]}


# ============================================================================ formatting helpers
def _fmt_iv(iv):
    s = ", ".join(f"{k}={_fmt(v)}" for k, v in iv.items() if k != "from")
    if list(iv) == ["value"]:
        s = _fmt(iv["value"])
    return s + (f" from step {iv['from']}" if "from" in iv else "")


def _fail_note(failed, t):
    if t in failed:
        st, why = failed[t]
        return f"   [no value from step {st}: {why}]"
    return ""


def _traj(h):
    if len(h) <= 40:
        return "[" + ", ".join(_fmt(v) for v in h) + "]"
    idx = sorted(set(list(range(10)) + list(range(0, len(h), max(1, len(h) // 20))) + list(range(len(h) - 5, len(h)))))
    return "{" + ", ".join(f"{i}: {_fmt(h[i])}" for i in idx) + "}"


def _quant(xs, p):
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _summ(vals):
    ok = [v for v in vals if v is not None]
    miss = len(vals) - len(ok)
    tail = f", {miss} sample(s) without value" if miss else ""
    if not ok:
        return "no value" + tail
    if all(isinstance(v, bool) for v in ok):
        return f"P(true)={sum(ok) / len(ok):.3f}" + tail
    if all(isinstance(v, str) for v in ok):
        cnt = {}
        for v in ok:
            cnt[v] = cnt.get(v, 0) + 1
        return ", ".join(f"{k!r}: {c / len(ok):.3f}" for k, c in sorted(cnt.items(), key=lambda x: -x[1])) + tail
    xs = sorted(float(v) for v in ok)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
    return (f"mean {_fmt(m)}, sd {_fmt(sd)}, p05 {_fmt(_quant(xs, .05))}, p50 {_fmt(_quant(xs, .5))}, "
            f"p95 {_fmt(_quant(xs, .95))}" + tail)


def _event_bool(ast, values):
    try:
        return bool(_expr.eval_on(ast, {k: v for k, v in values.items() if v is not None}))
    except (_expr.EvalError, _expr.ExprError):
        return None


def _event(ast, values):
    try:
        return _fmt(bool(_expr.eval_on(ast, {k: v for k, v in values.items() if v is not None})))
    except (_expr.EvalError, _expr.ExprError) as e:
        return f"undefined ({e})"


def _holes_text(cg, limit):
    if not cg.holes and not cg.warnings:
        return ""
    out = []
    if cg.holes:
        out.append(f"\nHOLES ({len(cg.holes)}):")
        out += [f"  - {h['where']}: {h['error']}" + (f" (hint: {h['hint']})" if h["hint"] else "")
                for h in cg.holes[:limit]]
        if len(cg.holes) > limit:
            out.append(f"  ... {len(cg.holes) - limit} more")
    if cg.warnings:
        out.append(("" if cg.holes else "\n") + "WARNINGS:")
        out += [f"  - {w}" for w in cg.warnings[:max(3, limit // 2)]]
    return "\n".join(out)


A2 = CausalArm   # registry name (tournament/arms/__init__.py: ARMS["A2"] = "a2_causal:A2")
