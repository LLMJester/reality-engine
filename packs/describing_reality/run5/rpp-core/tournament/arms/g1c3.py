"""G1C3: causal graph + structural equations, whole-graph document construction (A2 modified).

Spec: runs/discovery/gen1/specs/G1C3.md. The representation and its semantics are A2's, unchanged: the A2 engine
(`Compiled`, `simulate`, the evaluator, qualitative propagation, `_run_and_report`) is imported, not modified.
What is new is only the surface:

  * a line-oriented graph document that compiles one-to-one into A2's JSON graph (lenient parser: a bad line
    becomes a hole with line, column and hint; the rest compiles);
  * edges implied by equation references, signs inferred by a local finite-difference test (explicit `edge`
    lines override them);
  * `targets:` (answer obligations) and stored `whatif` lines (named interventions);
  * whole-graph `submit_graph` / `patch_graph` returning a coverage report, the baseline of the targets and every
    stored what-if in one result; batched `intervene` with a shared baseline; compact series;
  * its own episode loop (fenced ```graph block channel, budget status line, reserved answer call).

Document syntax (see tournament/prompts/g1c3_guide.md):
  time steps=30 dt=1
  targets: a, b
  [exo] name : type [unit] [= value] [~ distribution] [:= expression]   # provenance
  edge a -> b +|-|? [lag k]                                            # provenance
  whatif name: do x = 5 [from k], y scale 1.1, z add -2, w + ; targets a, b ; events a > 3
"""
import ast as _pyast
import copy
import difflib
import random
import re

from tournament.arms import _expr
from tournament.arms.a2_causal import (MAX_WORK, QUAL_WORDS, TYPES, CausalArm, Compiled, QualBudget, _coerce,
                                       _dist_point, _event_bool, _fmt, _fmt_iv, _holes_text, _lookup, _maybe_json,
                                       _qual_text, _summ, propagate, simulate)
from tournament.arms.base import ArmState

FENCE_LABEL = "graph"
MAX_OUT = 7800                 # tool result bound (the episode adds the budget line; the runner caps at 8,000)
MAX_BATCH = 6                  # what-ifs per intervene call
RENDER_MAX = 60000             # render bound (as A2)
SERIES_POINTS = 60
PLAIN_NUMBERS = {0.0, 1.0, 2.0, 100.0}     # not flagged as magic numbers (-1 parses as -(1))
TYPE_SYNONYMS = {"float": "real", "number": "real", "num": "real", "double": "real", "integer": "int",
                 "boolean": "bool", "categorical": "category", "cat": "category", "string": "category",
                 "str": "category", "enum": "category"}
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
VAR_RE = re.compile(r"^(exo\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(:=|:|=|~)")
EDGE_RE = re.compile(r"^edge\s+([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)\s*([+\-?])?\s*(?:lag\s*=?\s*(\d+))?\s*$")
MUTATING = ("submit_graph", "patch_graph")


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


def _const(src, typ="real"):
    """A literal value: number, true/false, 'string', a bare word (category) or a constant expression."""
    s = src.strip()
    if not s:
        raise ValueError("missing value")
    if typ == "category" and NAME_RE.fullmatch(s) and s not in ("true", "false"):
        return s
    try:
        a = _expr.parse(s)
    except _expr.ExprError as e:
        raise ValueError(str(e))
    if _expr.refs(a):
        raise ValueError(f"a value must be a constant, not an expression over variables ({s!r})")

    def lookup(name, lag):
        raise _expr.EvalError(f"'{name}' is not available in a constant")
    try:
        v = _expr.evaluate(a, lookup)
    except _expr.EvalError as e:
        raise ValueError(str(e))
    if isinstance(v, float) and v == int(v) and abs(v) < 1e15 and re.fullmatch(r"-?\d+", s):
        v = int(v)
    return v


def _literal_args(inner):
    s = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", inner))
    v = _pyast.literal_eval("(" + s + ",)") if s.strip() else ()
    return list(v)


def _parse_dist(src):
    m = re.fullmatch(r"\s*([A-Za-z_]+)\s*\((.*)\)\s*", src, re.S)
    if not m:
        raise ValueError("expected normal(mean, sd[, min, max]) | uniform(lo, hi) | choice([options][, [weights]])")
    kind, inner = m.group(1).lower(), m.group(2)
    try:
        args = _literal_args(inner)
    except Exception:
        raise ValueError("arguments must be literals (quote category options: choice(['a', 'b']))")
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    if kind == "normal":
        if len(args) not in (2, 4) or not all(num(x) for x in args):
            raise ValueError("normal(mean, sd) or normal(mean, sd, min, max) with numbers")
        d = {"kind": "normal", "mean": args[0], "sd": args[1]}
        if len(args) == 4:
            d["min"], d["max"] = args[2], args[3]
        return d
    if kind == "uniform":
        if len(args) != 2 or not all(num(x) for x in args):
            raise ValueError("uniform(lo, hi) with numbers")
        return {"kind": "uniform", "low": args[0], "high": args[1]}
    if kind == "choice":
        if args and isinstance(args[0], (list, tuple)):
            d = {"kind": "choice", "options": list(args[0])}
            if len(args) > 1:
                d["weights"] = list(args[1])
            return d
        return {"kind": "choice", "options": list(args)}
    raise ValueError(f"unknown distribution {kind!r} (normal | uniform | choice)")


def fenced_block(text, label=FENCE_LABEL):
    """The last block fenced with ```label (or ~~~label); a missing closing fence runs to the end."""
    if not text:
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


# ============================================================================ the document parser
def _did_you_mean(name, known):
    m = difflib.get_close_matches(name, list(known), n=2, cutoff=0.6)
    return f" (did you mean {' or '.join(repr(x) for x in m)}?)" if m else ""


def _parse_do_clause(c):
    """'x = 5 [from k]' | 'x scale 1.1' | 'x add -2' | 'x +' / 'x -' / 'x up' / 'x down' -> (var, A2 do value)."""
    s = c.strip()
    m = re.fullmatch(r"([A-Za-z_]\w*)\s*(.*)", s, re.S)
    if not m:
        raise ValueError(f"bad do clause {s!r}")
    var, rest = m.group(1), m.group(2).strip()
    if rest.lower() in QUAL_WORDS:
        return var, QUAL_WORDS[rest.lower()]
    m2 = re.fullmatch(r"(scale|add)\s*=?\s*(.+)", rest, re.S)
    if m2:
        v = _const(m2.group(2))
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"{m2.group(1)} needs a number")
        return var, {m2.group(1): v}
    if rest.startswith("=") and not rest.startswith("=="):
        rest = rest[1:].strip()
        frm = None
        m3 = re.fullmatch(r"(.+?)\s+from\s+(\d+)", rest, re.S)
        if m3:
            rest, frm = m3.group(1), int(m3.group(2))
        v = _const(rest, "category" if NAME_RE.fullmatch(rest.strip() or "_") and rest.strip() not in ("true", "false")
                   else "real")
        return var, ({"value": v, "from": frm} if frm is not None else {"value": v})
    raise ValueError(f"bad do clause {s!r} (forms: x = v | x = v from k | x scale f | x add d | x + | x -)")


def parse_document(text):
    """Graph document -> dict(graph=A2 JSON (no inferred edges yet), targets, whatifs, errors, explicit_edges,
    targets_declared, refs) where errors are line holes {line, col, error, hint, text}."""
    variables, eqs, eq_line, var_line, explicit = [], {}, {}, {}, []
    targets, whatifs, errors = [], [], []
    tm, targets_declared = None, False
    exo_forced = set()
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    def err(no, col, msg, hint, raw):
        errors.append({"line": no, "col": col, "error": msg, "hint": hint, "text": raw.strip()[:120]})

    for no, raw in enumerate(lines, 1):
        code, prov = _split_comment(raw)
        s = code.strip()
        if not s or s.startswith("```") or s.startswith("~~~"):
            continue
        lead = len(code) - len(code.lstrip())
        try:
            low = s.lower()
            if re.match(r"^targets?\s*:", low):
                targets_declared = True
                for x in s.split(":", 1)[1].split(","):
                    x = x.strip()
                    if not x:
                        continue
                    if not NAME_RE.fullmatch(x):
                        err(no, lead + s.find(x) + 1, f"bad target name {x!r}", "targets: a, b, c", raw)
                    elif x not in targets:
                        targets.append(x)
                continue
            if re.match(r"^time\s*:?\s*(steps|dt)\b", low):
                kv = dict(re.findall(r"(steps|dt)\s*[=:]?\s*([0-9.eE+-]+)", s))
                if "steps" not in kv:
                    err(no, lead + 1, "time line needs steps", "time steps=30 dt=1", raw)
                    continue
                st = float(kv["steps"])
                dt = float(kv.get("dt", 1))
                tm = {"steps": int(st) if st == int(st) else st, "dt": int(dt) if dt == int(dt) else dt}
                continue
            if low.startswith("edge ") or low.startswith("edge\t"):
                m = EDGE_RE.match(s)
                if not m:
                    err(no, lead + 1, "bad edge line", "edge a -> b + | - | ? [lag k]", raw)
                    continue
                e = {"from": m.group(1), "to": m.group(2), "sign": m.group(3) or "?",
                     "provenance": prov or "", "line": no}
                if m.group(4) is not None:
                    e["lag"] = int(m.group(4))
                explicit.append(e)
                continue
            if low.startswith("whatif"):
                m = re.match(r"^whatif\s+([A-Za-z_]\w*)\s*:\s*(.*)$", s, re.S)
                if not m:
                    err(no, lead + 1, "bad whatif line", "whatif name: do x = 5 ; targets a ; events a > 3", raw)
                    continue
                wi = {"name": m.group(1), "do": {}, "targets": [], "events": [], "line": no}
                if prov:
                    wi["provenance"] = prov
                bad = False
                for k, seg in enumerate(_split_top(m.group(2), ";")):
                    seg = seg.strip()
                    if not seg:
                        continue
                    w = seg.split(None, 1)
                    key, rest = w[0].lower(), (w[1] if len(w) > 1 else "")
                    if key == "do" or (k == 0 and key not in ("targets", "target", "events", "event")):
                        body = rest if key == "do" else seg
                        for c in _split_top(body, ","):
                            if not c.strip():
                                continue
                            try:
                                var, val = _parse_do_clause(c)
                                wi["do"][var] = val
                            except ValueError as e:
                                err(no, lead + s.find(c.strip()) + 1, str(e),
                                    "do x = 5 | x = 5 from 10 | x scale 1.1 | x add -2 | x + | x -", raw)
                                bad = True
                    elif key in ("targets", "target"):
                        wi["targets"] += [x.strip() for x in rest.split(",") if x.strip()]
                    elif key in ("events", "event"):
                        wi["events"] += [x.strip() for x in _split_top(rest, ",") if x.strip()]
                    elif key in ("samples", "steps", "seed"):
                        try:
                            wi[key] = int(float(rest))
                        except ValueError:
                            err(no, lead + s.find(seg) + 1, f"{key} needs an integer", f"{key} 200", raw)
                    elif key == "at":
                        r = rest.strip()
                        wi["at"] = int(r) if r.isdigit() else r
                    else:
                        err(no, lead + s.find(seg) + 1, f"unknown whatif part {key!r}",
                            "parts: do ... ; targets ... ; events ... ; at final|all|k ; samples N", raw)
                if not wi["do"]:
                    if not bad:
                        err(no, lead + 1, "whatif without a do clause", "whatif name: do x = 5", raw)
                    continue
                if any(w["name"] == wi["name"] for w in whatifs):
                    err(no, lead + 1, f"whatif {wi['name']!r} declared twice; later one ignored", "rename one", raw)
                    continue
                whatifs.append(wi)
                continue
            m = VAR_RE.match(s)
            if not m:
                err(no, lead + 1, "unrecognised line",
                    "lines are: time ... | targets: ... | name : type [unit] = value | ~ dist | := expr | "
                    "edge a -> b + | whatif name: do ...", raw)
                continue
            is_exo, name, op = bool(m.group(1)), m.group(2), m.group(3)
            if name in _expr.RESERVED or name in _expr.FUNCS:
                err(no, lead + m.start(2) + 1, f"invalid name {name!r} (keyword, function or t/dt)",
                    "rename the variable", raw)
                continue
            if name in var_line:
                err(no, lead + m.start(2) + 1, f"variable {name!r} declared twice (first on line {var_line[name]}); "
                    "this line ignored", "remove or rename one", raw)
                continue
            rest = s[m.end(2):].strip()
            if op == ":":
                rest = rest[1:]
            eq_src = None
            if ":=" in rest:
                left, eq_src = rest.split(":=", 1)
                eq_src = eq_src.strip()
            else:
                left = rest
            dist = None
            if "~" in left:
                left, dsrc = left.split("~", 1)
                try:
                    dist = _parse_dist(dsrc)
                except ValueError as e:
                    err(no, lead + s.find("~") + 2, f"bad distribution: {e}",
                        "normal(mean, sd[, min, max]) | uniform(lo, hi) | choice([opts][, [weights]])", raw)
                    continue
            value_src = None
            if "=" in left:
                left, value_src = left.split("=", 1)
            words = left.split()
            typ = "real"
            if words and (words[0].lower() in TYPES or words[0].lower() in TYPE_SYNONYMS):
                w0 = words.pop(0).lower()
                typ = TYPE_SYNONYMS.get(w0, w0)
            unit = " ".join(words)
            v = {"name": name, "type": typ, "unit": unit, "exogenous": is_exo or eq_src is None}
            if value_src is not None:
                try:
                    v["value"] = _const(value_src, typ)
                except ValueError as e:
                    err(no, lead + s.find(value_src.strip()) + 1 if value_src.strip() else lead + 1,
                        f"bad value for {name!r}: {e}", "a number, true/false or 'text'", raw)
                    continue
            if dist is not None:
                v["distribution"] = dist
            if prov:
                v["provenance"] = prov
            if eq_src is not None:
                if not eq_src:
                    err(no, len(raw.rstrip()), f"empty equation for {name!r}", f"{name} : real := <expression>", raw)
                else:
                    try:
                        _expr.parse(eq_src)
                    except _expr.ExprError as e:
                        col = raw.find(":=") + 3 + (len(eq_src) - len(eq_src.lstrip()))
                        err(no, col + (e.col or 1) - 1 if raw.find(":=") >= 0 else None,
                            f"equation for {name!r}: {e.msg}", e.hint or "check the expression syntax", raw)
                    eqs[name] = eq_src           # a broken equation stays (A2 makes it a hole; the rest computes)
                    eq_line[name] = no
            if is_exo:
                exo_forced.add(name)
            variables.append(v)
            var_line[name] = no
        except Exception as e:  # never raise on a line
            err(no, lead + 1, f"could not parse line: {type(e).__name__}: {e}", "see the document syntax", raw)

    # a bare declaration (no value, distribution or equation) with incoming edges is an endogenous variable whose
    # sign comes from its edges (qualitative-only graphs, as in A2)
    into = {e["to"] for e in explicit}
    for v in variables:
        n = v["name"]
        if (n not in eqs and n not in exo_forced and "value" not in v and "distribution" not in v and n in into):
            v["exogenous"] = False
    graph = {"variables": variables, "edges": [], "equations": eqs}
    if tm is not None:
        graph["time"] = tm
    return {"graph": graph, "targets": targets, "whatifs": whatifs, "errors": errors, "explicit_edges": explicit,
            "targets_declared": targets_declared, "var_line": var_line, "eq_line": eq_line}


# ============================================================================ edges: inference and override
def _base_values(cg):
    base = {}
    for n, s in cg.vars.items():
        if n not in cg.eqs and n not in cg.broken:
            if s["exogenous"] and "distribution" in s:
                v = s["value"] if "value" in s else _dist_point(s["distribution"])
                base[n] = _coerce(v, s["type"])[0]
            elif "value" in s:
                base[n] = s["value"]
    return base


def _num(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _local_sign(cg, hist, failed, base, var, parent, lag, t):
    """Sign of d f_var / d parent[t-lag] at step t by a finite difference (None: undetermined)."""
    look = _lookup(cg, hist, failed, t, base)
    try:
        x = look(parent, lag)
    except _expr.EvalError:
        return None
    if isinstance(x, bool):
        lo, hi = False, True
    elif isinstance(x, (int, float)):
        d = max(1e-6, 1e-3 * abs(float(x)))
        lo, hi = x, float(x) + d
    else:
        return None

    def with_value(v):
        def f(name, lg):
            return v if (name == parent and lg == lag) else look(name, lg)
        return f
    try:
        a = _num(_expr.evaluate(cg.eqs[var], with_value(lo)))
        b = _num(_expr.evaluate(cg.eqs[var], with_value(hi)))
    except _expr.EvalError:
        return None
    if a is None or b is None:
        return None
    return "+" if b > a else "-" if b < a else "0"


def infer_edges(graph, explicit):
    """Edges implied by equation references (one per parent and lag), signs inferred at the step-0 and final-step
    baseline values; explicit edge lines override. Returns the edge list (A2 format + provenance)."""
    cg = Compiled({k: v for k, v in graph.items() if k != "edges"})
    try:
        hist, failed = simulate(cg)
    except Exception:
        hist, failed = {n: [] for n in cg.vars}, {}
    base = _base_values(cg)
    last = cg.steps if cg.timed else 0
    inferred = []
    for var in cg.order_decl:
        if var not in cg.eqs:
            continue
        for parent, lag in sorted(_expr.refs(cg.eqs[var])):
            if parent == var or parent not in cg.vars:
                continue
            signs = [_local_sign(cg, hist, failed, base, var, parent, lag, t) for t in sorted({0, last})]
            s = signs[0] if all(x == signs[0] for x in signs) and signs[0] in ("+", "-") else "?"
            e = {"from": parent, "to": var, "sign": s, "provenance": "inferred (finite difference)", "inferred": True}
            if lag:
                e["lag"] = lag
            inferred.append(e)
    out, pairs = [], {(e["from"], e["to"]) for e in explicit}
    for e in inferred:
        if (e["from"], e["to"]) not in pairs:
            out.append(e)
    for x in explicit:
        lags = [e.get("lag", 0) for e in inferred if (e["from"], e["to"]) == (x["from"], x["to"])]
        if "lag" in x or not lags:
            lags = [x.get("lag", 0)]
        for lg in sorted(set(lags)):
            e = {"from": x.get("from"), "to": x.get("to"), "sign": x.get("sign", "?"),
                 "provenance": x.get("provenance") or "explicit edge line"}
            if lg:
                e["lag"] = lg
            out.append(e)
    return out


# ============================================================================ compiled document
class Doc:
    """A parsed and compiled graph document (or an A2 JSON graph given directly)."""

    def __init__(self, source=None, graph=None):
        self.source = source
        if source is not None:
            p = parse_document(source)
            g = p["graph"]
            g["edges"] = infer_edges(g, p["explicit_edges"])
            self.graph, self.targets, self.whatifs = g, p["targets"], p["whatifs"]
            self.errors, self.targets_declared = p["errors"], p["targets_declared"]
            self.var_line, self.eq_line = p["var_line"], p["eq_line"]
        else:
            g = copy.deepcopy(graph) if isinstance(graph, dict) else {}
            self.targets = [x for x in (g.pop("targets", None) or []) if isinstance(x, str)]
            self.whatifs = [w for w in (g.pop("whatifs", None) or []) if isinstance(w, dict) and w.get("name")]
            g.pop("source", None)
            if isinstance(g.get("edges", []), list) and isinstance(g.get("equations", {}), dict):
                given = [e for e in (g.get("edges") or []) if isinstance(e, dict)]
                bad = [e for e in (g.get("edges") or []) if not isinstance(e, dict)]
                try:          # equation references without an edge get an inferred one (given edges win)
                    g["edges"] = infer_edges(g, given) + bad
                except Exception:
                    pass
            self.graph = g
            self.errors, self.targets_declared = [], bool(self.targets)
            self.var_line, self.eq_line = {}, {}
        self.cg = Compiled(self.graph)

    def export(self):
        r = copy.deepcopy(self.graph)
        r["source"] = self.source
        r["targets"] = list(self.targets)
        r["whatifs"] = copy.deepcopy(self.whatifs)
        return r

    # ---------------------------------------------------------------- coverage
    def undefined(self):
        """[(name, where)] for names referenced but not declared."""
        known = set(self.cg.vars)
        out, seen = [], set()

        def add(n, where):
            if n not in known and (n, where) not in seen:
                seen.add((n, where))
                out.append((n, where))
        for v, src in (self.graph.get("equations") or {}).items():
            try:
                for r, _ in sorted(_expr.refs(_expr.parse(src))):
                    add(r, f"equation of {v}" + (f" (line {self.eq_line[v]})" if v in self.eq_line else ""))
            except (_expr.ExprError, TypeError):
                pass
            if v not in known:
                add(v, "equation target")
        for e in self.graph.get("edges") or []:
            if isinstance(e, dict):
                for k in ("from", "to"):
                    if isinstance(e.get(k), str):
                        add(e[k], f"edge {e.get('from')}->{e.get('to')}")
        for t in self.targets:
            add(t, "targets")
        for w in self.whatifs:
            where = f"whatif {w.get('name')}"
            for v in (w.get("do") or {}):
                add(v, where + " (do)")
            for t in w.get("targets") or []:
                add(t, where + " (targets)")
            for ev in w.get("events") or []:
                try:
                    for r, _ in sorted(_expr.refs(_expr.parse(ev))):
                        add(r, where + " (events)")
                except _expr.ExprError:
                    pass
        return out

    def magic_numbers(self):
        out = []
        for v, src in (self.graph.get("equations") or {}).items():
            try:
                a = _expr.parse(src)
            except (_expr.ExprError, TypeError):
                continue
            nums = []

            def walk(n):
                if n[0] == "num":
                    if float(n[1]) not in PLAIN_NUMBERS:
                        nums.append(n[1])
                elif n[0] in ("neg", "not"):
                    walk(n[1])
                elif n[0] == "bin":
                    walk(n[2]), walk(n[3])
                elif n[0] in ("and", "or"):
                    walk(n[1]), walk(n[2])
                elif n[0] == "if":
                    for x in n[1:]:
                        walk(x)
                elif n[0] == "call":
                    for x in n[2]:
                        walk(x)
            walk(a)
            if nums:
                out.append((v, nums))
        return out

    def holes(self):
        """Every hole as text: document line errors, undefined target/what-if names, A2 compile holes."""
        hs = [f"line {e['line']}" + (f" col {e['col']}" if e.get("col") else "") + f": {e['error']}"
              for e in self.errors]
        hs += [f"undefined name '{n}' in {w}" for n, w in self.undefined()
               if w.startswith("targets") or w.startswith("whatif")]
        hs += [f"{h['where']}: {h['error']}" for h in self.cg.holes]
        return hs


# ============================================================================ compact series
def _g3(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, (int, float)):
        if v == 0:
            return "0"
        return f"{v:.3g}"
    return str(v)


def compact_series(h):
    """One row: ≤ 60 points (incl. the extremes and the final step), 3 significant figures, run-length
    compressed ('a-b:v' = steps a..b all v), plus a SHAPE line for numeric series."""
    n = len(h)
    if not n:
        return "[]", ""
    nums = [(_num(v) if not isinstance(v, str) else None) for v in h]
    numeric = all(x is not None for x in nums) and not all(isinstance(v, bool) for v in h)
    fv = [_g3(v) for v in h]
    runs, i = [], 0                       # runs of equal (3-significant-figure) values over the whole series
    while i < n:
        j = i
        while j + 1 < n and fv[j + 1] == fv[i]:
            j += 1
        runs.append((i, j))
        i = j + 1
    if len(runs) > SERIES_POINTS:         # too many: sample points (keeping the extremes and the final step)
        idx = set(round(k * (n - 1) / (SERIES_POINTS - 5)) for k in range(SERIES_POINTS - 4)) | {0, n - 1}
        if numeric:
            idx |= {max(range(n), key=lambda k: nums[k]), min(range(n), key=lambda k: nums[k])}
        runs = []
        for k in sorted(idx):             # merge neighbouring samples when every step between them has the value
            if runs and runs[-1][1] < k and all(fv[x] == fv[k] for x in range(runs[-1][0], k + 1)):
                runs[-1] = (runs[-1][0], k)
            else:
                runs.append((k, k))
    parts = [f"{a}-{b}:{fv[a]}" if b > a else f"{a}:{fv[a]}" for a, b in runs]
    row = "[" + " ".join(parts) + "]"
    if not numeric:
        return row, ""
    hi = max(range(n), key=lambda i: nums[i])
    lo = min(range(n), key=lambda i: nums[i])
    scale = max(1e-12, max(abs(x) for x in nums))
    dirs = []
    for a, b in zip(nums, nums[1:]):
        d = b - a
        if abs(d) > 1e-9 * scale:
            s = 1 if d > 0 else -1
            if not dirs or dirs[-1] != s:
                dirs.append(s)
    if not dirs:
        kind = "constant"
    elif len(dirs) == 1:
        kind = "monotone rising" if dirs[0] > 0 else "monotone falling"
    elif len(dirs) == 2:
        kind = "single peak" if dirs[0] > 0 else "single trough"
    else:
        kind = f"oscillating ({len(dirs) - 1} turns)"
    shape = (f"SHAPE start {_g3(nums[0])}, end {_g3(nums[-1])}, max {_g3(nums[hi])} at t={hi}, "
             f"min {_g3(nums[lo])} at t={lo}; {kind}")
    return row, shape


def _mean_series(hists, name):
    """Mean trajectory over Monte Carlo runs (numeric), or the single run."""
    if len(hists) == 1:
        return hists[0][name]
    L = max((len(h[name]) for h in hists), default=0)
    out = []
    for t in range(L):
        xs = [_num(h[name][t]) for h in hists if t < len(h[name]) and h[name][t] is not None
              and not isinstance(h[name][t], str)]
        out.append(sum(xs) / len(xs) if xs else None)
    return out


# ============================================================================ the arm
def _tool(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


_QUERY = {"description": "string target name, or object {targets?: [names], at?: 'final'|'all'|step, samples?: int, "
                         "seed?: int, events?: [expr], steps?: int}"}


class G1C3State(ArmState):
    def __init__(self, rep=None, given_gold=False):
        super().__init__(rep=rep if rep is not None else {}, given_gold=given_gold)
        self.doc = None              # Doc
        self.pending_block = None    # last ```graph block of the current response


class G1C3Arm(CausalArm):
    name = "G1C3"
    buildable = True
    executable = True
    guide_file = "g1c3_guide.md"
    mutating_tools = MUTATING
    custom_episode = True

    def guide(self):
        from tournament.arms.base import Arm
        return Arm.guide(self)

    def tools(self):
        return [
            _tool("submit_graph", "Compile the whole graph document and replace the representation. Takes `source` "
                  "(document text) or `graph` (A2 JSON); with neither, the last ```graph block of this response. "
                  "Returns the coverage report, the baseline of the targets and every stored whatif.",
                  {"source": {"type": "string"}, "graph": {"type": "object"}}),
            _tool("patch_graph", "Exact unique text replace in the document, then recompile and rerun as submit_graph.",
                  {"old": {"type": "string"}, "new": {"type": "string"}}, ["old", "new"]),
            _tool("evaluate", "Compute variable values (static, over time, or Monte Carlo).", {"query": _QUERY}),
            _tool("intervene", "Compare interventions with the baseline (paired). Either `do` + `query`, or `whatifs`: "
                  "up to 6 of {name?, do, query?} or names of stored whatif lines, run on one shared baseline. "
                  "do = {var: value | {value|scale|add: x, from?: step} | '+'|'-' (qualitative)}.",
                  {"do": {"type": "object"}, "query": _QUERY,
                   "whatifs": {"type": "array", "items": {}, "description": "what-ifs or stored whatif names"}}),
            _tool("show", "Show the document with line numbers, the compiled graph and its holes.", {}),
        ]

    def probe_tools(self):
        return [t for t in self.tools() if t["function"]["name"] in ("evaluate", "intervene", "show")]

    # ---------------------------------------------------------------- state
    def new_state(self):
        st = G1C3State()
        self._set_doc(st, Doc(graph={"variables": [], "edges": [], "equations": {}}))
        return st

    def load(self, rep, given_gold=False):
        st = G1C3State(given_gold=given_gold)
        if isinstance(rep, dict) and isinstance(rep.get("source"), str):
            self._set_doc(st, Doc(source=rep["source"]))
        elif isinstance(rep, dict):
            self._set_doc(st, Doc(graph=rep))
        else:
            self._set_doc(st, Doc(graph={"variables": [], "edges": [], "equations": {}}))
            st.log.append(("load", None, f"representation was {type(rep).__name__}, not an object; started empty"))
        return st

    def _set_doc(self, st, doc):
        st.doc = doc
        st.rep = doc.export()

    def _doc(self, state):
        if getattr(state, "doc", None) is None:
            d = state.rep if isinstance(state.rep, dict) else {}
            state.doc = Doc(source=d["source"]) if isinstance(d.get("source"), str) else Doc(graph=d)
        return state.doc

    def export(self, state):
        return self._doc(state).export()

    # ---------------------------------------------------------------- dispatch
    def handle(self, state, name, args):
        try:
            args = _maybe_json(args) if args is not None else {}
            if not isinstance(args, dict):
                return f"ERROR: arguments must be an object (hint: see the {name} schema)"
            fn = getattr(self, "_t_" + str(name), None)
            if fn is None or name not in {t["function"]["name"] for t in self.tools()}:
                return (f"ERROR: unknown tool '{name}' (hint: available: "
                        + ", ".join(t["function"]["name"] for t in self.tools()) + ")")
            if state.given_gold and name in MUTATING:
                return "ERROR: the representation is read-only in this mode (hint: use evaluate, intervene, show)"
            out = fn(state, args)
        except Exception as e:  # never raise
            out = f"ERROR: internal {type(e).__name__}: {e} (hint: check the arguments)"
        state.log.append((name, args, out[:200]))
        return out

    # ---------------------------------------------------------------- building
    def _t_submit_graph(self, state, args):
        src = args.get("source")
        g = _maybe_json(args.get("graph"))
        if isinstance(src, str) and src.strip():
            inner = fenced_block(src)
            doc = Doc(source=inner if inner is not None else src)
            how = "source argument"
        elif isinstance(g, dict):
            doc = Doc(graph=g)
            how = "graph argument (A2 JSON)"
        elif getattr(state, "pending_block", None):
            doc = Doc(source=state.pending_block)
            how = "```graph block"
        else:
            return ("ERROR: no document (hint: put the whole graph in ONE fenced block labelled graph in this response, "
                    "or pass source=<document text>)")
        self._set_doc(state, doc)
        return self._report(doc, f"graph submitted from the {how}")

    def _t_patch_graph(self, state, args):
        doc = self._doc(state)
        old, new = args.get("old"), args.get("new")
        if doc.source is None:
            return "ERROR: no document to patch (hint: submit_graph a document first)"
        if not isinstance(old, str) or not old or not isinstance(new, str):
            return "ERROR: patch_graph needs 'old' (non-empty text) and 'new' (hint: copy 'old' exactly from show)"
        k = doc.source.count(old)
        if k == 0:
            lines = doc.source.split("\n")
            near = difflib.get_close_matches(old.strip().split("\n")[0], [x.strip() for x in lines], n=1, cutoff=0.4)
            return ("ERROR: 'old' text not found in the document" + (f" (closest line: {near[0]!r})" if near else "")
                    + " (hint: copy it exactly, or resubmit the whole document)")
        if k > 1:
            return f"ERROR: 'old' text occurs {k} times (hint: include more surrounding text so it is unique)"
        nd = Doc(source=doc.source.replace(old, new, 1))
        self._set_doc(state, nd)
        return self._report(nd, "patch applied")

    def _report(self, doc, head):
        cg = doc.cg
        sec = [f"{head}: {len(cg.vars)} variables, {len(cg.edges)} edges, {len(cg.eqs)} equations, "
               f"{len(doc.targets)} targets, {len(doc.whatifs)} whatifs"]
        # 1. undefined names
        und = doc.undefined()
        if und:
            names = {}
            for n, w in und:
                names.setdefault(n, []).append(w)
            sec.append(f"UNDEFINED NAMES ({len(names)}):\n" + "\n".join(
                f"  - '{n}' used in {', '.join(ws[:3])}{_did_you_mean(n, cg.vars)}" for n, ws in names.items()))
        else:
            sec.append("UNDEFINED NAMES: 0")
        # baseline run (shared by the target status, the edge signs and the report)
        base_runs, q = None, None
        if cg.vars:
            tg = [t for t in doc.targets if t in cg.vars] or ([] if doc.targets else cg.order_decl[:8])
            q = self._parse_query(cg, {"targets": tg} if tg else {"targets": [cg.order_decl[0]]})
            q["targets"] = tg
            _cap_samples(cg, q, 1 + min(len(doc.whatifs), MAX_BATCH))
            base_runs = self._baseline(cg, q)
        # 2. targets
        if not doc.targets_declared:
            sec.append("TARGETS: WARNING no 'targets:' line (hint: declare what the question asks for: "
                       "targets: a, b, c)")
        else:
            st = []
            for t in doc.targets:
                if t not in cg.vars:
                    st.append(f"{t}: undefined{_did_you_mean(t, cg.vars)}")
                elif t in cg.broken:
                    st.append(f"{t}: hole ({cg.broken[t]})")
                else:
                    h = base_runs[0][0][t] if base_runs else []
                    st.append(f"{t}: computed" if h and h[-1] is not None
                              else f"{t}: hole ({base_runs[0][1].get(t, (0, 'no value'))[1] if base_runs else 'no value'})")
            sec.append("TARGETS: " + "; ".join(st))
        # 3. holes
        hl = [f"  - line {e['line']}" + (f" col {e['col']}" if e.get("col") else "") + f": {e['error']}"
              + (f" (hint: {e['hint']})" if e.get("hint") else "") for e in doc.errors]
        hl += [f"  - {h['where']}: {h['error']}" + (f" (hint: {h['hint']})" if h["hint"] else "")
               for h in cg.holes if "references undeclared" not in h["error"] and "unknown variable(s)" not in h["error"]]
        if hl:
            sec.append(f"HOLES ({len(hl)}):\n" + "\n".join(hl[:15]) + (f"\n  ... {len(hl) - 15} more" if len(hl) > 15
                                                                       else ""))
        else:
            sec.append("HOLES: 0")
        warn = [w for w in cg.warnings if "there is no edge" not in w]
        if warn:
            sec.append("WARNINGS:\n" + "\n".join(f"  - {w}" for w in warn[:5]))
        # 4. intervention handles + magic numbers
        exo = [n for n in cg.order_decl if cg.vars[n]["exogenous"]]
        sec.append("INTERVENTION HANDLES (exogenous inputs a what-if can set): "
                   + (", ".join(exo[:40]) + (f" ... {len(exo) - 40} more" if len(exo) > 40 else "") if exo else "none"))
        mags = doc.magic_numbers()
        if mags:
            sec.append("MAGIC NUMBERS (consider naming each as a variable so a what-if can change it): "
                       + "; ".join(f"{', '.join(_fmt(x) for x in xs[:6])} in {v}" for v, xs in mags[:12])
                       + (f"; ... {len(mags) - 12} more equations" if len(mags) > 12 else ""))
        # 5. edges
        inf = [e for e in cg.edges]
        unknown = [f"{a}->{b}" for a, b, s, _ in inf if s == "?"]
        sec.append(f"EDGES ({len(inf)}): " + ", ".join(f"{a}->{b} {s}" + (f" lag{lg}" if lg else "")
                                                       for a, b, s, lg in inf[:40])
                   + (f" ... {len(inf) - 40} more" if len(inf) > 40 else "")
                   + (f"\n  '?' signs ({len(unknown)}): " + ", ".join(unknown[:20]) if unknown else ""))
        cov = _cut_lines("\n".join(sec), 3000)
        # baseline (its share of the bound: half of what is left when what-ifs follow)
        parts = [cov]
        room = MAX_OUT - len(cov) - 100
        run_wis = doc.whatifs[:MAX_BATCH] if cg.vars else []
        if base_runs and q["targets"]:
            if not doc.targets:
                parts.append("(no targets declared: showing the first variables)")
            parts.append(_cut_lines(self._baseline_text(cg, q, base_runs, series=cg.timed),
                                    max(800, room // 2 if run_wis else room)))
        elif cg.vars:
            parts.append("BASELINE: no computable target to report")
        # stored what-ifs
        if run_wis:
            if len(doc.whatifs) > MAX_BATCH:
                parts.append(f"(only the first {MAX_BATCH} stored whatifs were run; run others with intervene)")
            room = MAX_OUT - sum(len(p) + 1 for p in parts) - 20
            parts.append(self._batch(cg, doc, [{"name": w["name"]} for w in run_wis], None,
                                     room=max(600, room), baseline=(q, base_runs)))
        return _bound("\n".join(parts))

    # ---------------------------------------------------------------- runs
    def _baseline(self, cg, q):
        """[(hist, failed)] per Monte Carlo run (one run without samples)."""
        if not q["samples"]:
            return [simulate(cg, steps=q["steps"])]
        return [simulate(cg, steps=q["steps"], rng=random.Random(q["seed"] * 1_000_003 + i))
                for i in range(q["samples"])]

    def _baseline_text(self, cg, q, runs, series=True):
        lines = []
        targets = [t for t in q["targets"] if t in cg.vars]
        pick = _picker(q["at"])
        if q["samples"]:
            lines.append(f"BASELINE (Monte Carlo: {q['samples']} samples, seed {q['seed']}, "
                         + (f"at step {q['at']})" if isinstance(q["at"], int) else f"at final step {q['steps']})"))
            for t in targets:
                lines.append(f"  {t}: {_summ([pick(h[t]) for h, _ in runs])}" + _fail(runs, t))
        else:
            lines.append("BASELINE" + (f" (at step {q['at']})" if isinstance(q["at"], int)
                                       else f" (final step {q['steps']})" if cg.timed else ""))
            for t in targets:
                lines.append(f"  {t} = {_fmt(pick(runs[0][0][t]))}" + _fail(runs, t))
        for ev in q["events"]:
            try:
                a = _expr.parse(ev)
            except _expr.ExprError as e:
                lines.append(f"  event {ev!r}: {e}")
                continue
            r = [_event_bool(a, {n: pick(h) for n, h in hh.items()}) for hh, _ in runs]
            r = [x for x in r if x is not None]
            lines.append(f"  P({ev}) = {sum(r) / len(r):.3f} (n={len(r)})" if r else f"  P({ev}) = undefined")
        if series and cg.timed:
            lines.append("SERIES (" + ("mean, " if q["samples"] else "") + f"steps 0..{q['steps']}, step:value; "
                         "a-b:v = steps a..b all v)")
            hs = [h for h, _ in runs]
            for t in targets:
                row, shape = compact_series(_mean_series(hs, t))
                lines.append(f"  {t}: {row}" + (f"\n    {shape}" if shape else ""))
        return "\n".join(lines)

    def _t_evaluate(self, state, args):
        doc = self._doc(state)
        cg = doc.cg
        raw = args.get("query", {k: v for k, v in args.items() if k != "query"})
        raw = _maybe_json(raw)
        if raw in (None, {}, []) and doc.targets:
            raw = {"targets": [t for t in doc.targets if t in cg.vars] or doc.targets}
        elif isinstance(raw, dict) and not raw.get("targets") and not raw.get("target") and doc.targets:
            raw = dict(raw, targets=doc.targets)
        try:
            q = self._parse_query(cg, raw)
        except (ValueError, TypeError) as e:
            return f"ERROR: bad query: {e} (hint: query={{\"targets\": [\"y\"], \"at\": \"final\"}})"
        if q["at"] == "all":
            q2 = dict(q, at="final")
            out = self._baseline_text(cg, q2, self._baseline(cg, q2), series=True)
            unk = [t for t in q["targets"] if t not in cg.vars]
            if unk:
                out += "\n" + "\n".join(f"{t}: unknown variable{_did_you_mean(t, cg.vars)}" for t in unk)
        else:
            out = self._run_and_report(cg, q, do=None)
        return _bound(out + _holes_text(cg, 8))

    def _t_intervene(self, state, args):
        doc = self._doc(state)
        cg = doc.cg
        wis = _maybe_json(args.get("whatifs"))
        if wis is None and args.get("names") is not None:
            wis = _maybe_json(args.get("names"))
        if isinstance(wis, (str, dict)):
            wis = [wis]
        if wis is None:
            do = _maybe_json(args.get("do"))
            if not isinstance(do, dict) or not do:
                return ("ERROR: give do (hint: do={\"price\": 12}) or whatifs (hint: whatifs=[\"name\", "
                        "{\"name\": \"b\", \"do\": {\"x\": {\"scale\": 1.1}}}])")
            wis = [{"name": "do", "do": do, "query": args.get("query")}]
        if not isinstance(wis, list) or not wis:
            return "ERROR: whatifs must be a non-empty list (hint: whatifs=[\"stored_name\", {\"do\": {...}}])"
        note = ""
        if len(wis) > MAX_BATCH:
            note = f"note: only the first {MAX_BATCH} of {len(wis)} what-ifs were run (limit per call)\n"
            wis = wis[:MAX_BATCH]
        out = self._batch(cg, doc, wis, args.get("query"), room=MAX_OUT - len(note) - 400)
        return _bound(note + out + _holes_text(cg, 6))

    def _resolve(self, cg, doc, w, q_default):
        """A what-if spec -> (name, do_raw, query raw) or an error string."""
        stored = {x["name"]: x for x in doc.whatifs}
        if isinstance(w, str):
            if w not in stored:
                return f"whatif '{w}': not stored{_did_you_mean(w, stored)} (stored: {', '.join(stored) or 'none'})"
            w = {"name": w}
        if not isinstance(w, dict):
            return "whatif: must be a name or an object {name?, do, query?}"
        name = w.get("name") or "whatif"
        if "do" not in w and name in stored:
            s = stored[name]
            q = {k: s[k] for k in ("targets", "events", "at", "samples", "steps", "seed") if s.get(k) not in (None, [])}
            if not q.get("targets"):
                q["targets"] = list(doc.targets)
            return name, s["do"], q
        do = _maybe_json(w.get("do"))
        if not isinstance(do, dict) or not do:
            return f"whatif '{name}': do must be a non-empty object"
        q = _maybe_json(w.get("query", q_default))
        if q is None:
            q = {}
        if isinstance(q, str):
            q = {"targets": [x.strip() for x in q.split(",") if x.strip()]}
        if isinstance(q, list):
            q = {"targets": q}
        if isinstance(q, dict) and not q.get("targets") and not q.get("target"):
            q = dict(q, targets=list(doc.targets))
        return name, do, q

    def _batch(self, cg, doc, wis, q_default, room=MAX_OUT, baseline=None):
        """Run what-ifs against one shared baseline per (steps, samples, seed)."""
        specs, blocks = [], []
        for w in wis:
            r = self._resolve(cg, doc, w, q_default)
            if isinstance(r, str):
                blocks.append(("ERROR " + r, [], []))
                continue
            name, do_raw, qraw = r
            if isinstance(qraw, dict) and not qraw.get("targets") and not qraw.get("target"):
                qraw = dict(qraw, targets=None)
            try:
                q = self._parse_query(cg, qraw)
            except (ValueError, TypeError) as e:
                blocks.append((f"ERROR whatif '{name}': bad query: {e} (hint: query={{\"targets\": [\"y\"]}})", [], []))
                continue
            specs.append((name, do_raw, q))
        n_runs = len({(q["steps"], q["samples"], q["seed"]) for _, _, q in specs}) + len(specs)
        cache = {}
        if baseline and baseline[0] is not None:
            bq, bruns = baseline
            cache[(bq["steps"], bq["samples"], bq["seed"])] = bruns
            n_runs = 1 + len(specs)
        for name, do_raw, q in specs:
            _cap_samples(cg, q, n_runs)
            key = (q["steps"], q["samples"], q["seed"])
            if key not in cache:
                cache[key] = self._baseline(cg, q)
            blocks.append(self._one_whatif(cg, name, do_raw, q, cache[key]))
        n = max(1, len(blocks))
        per = max(400, room // n)
        out = []
        for head, body, series in blocks:
            txt = head
            for part in (body, series):
                for ln in part:
                    if len(txt) + len(ln) + 1 > per:
                        txt += "\n  [... cut to the output bound]"
                        break
                    txt += "\n" + ln
                else:
                    continue
                break
            out.append(txt)
        return "\n".join(out)

    def _one_whatif(self, cg, name, do_raw, q, base_runs):
        """(head, [target/event + qualitative lines], [series lines]) for one what-if."""
        do, qual, errs = {}, {}, []
        for var, val in do_raw.items():
            if var not in cg.vars:
                errs.append(f"do: unknown variable '{var}'{_did_you_mean(var, cg.vars)} (ignored)")
                continue
            typ = cg.vars[var]["type"]
            val = _maybe_json(val)
            if isinstance(val, str) and typ != "category" and val.strip().lower() in QUAL_WORDS:
                qual[var] = QUAL_WORDS[val.strip().lower()]
                continue
            if isinstance(val, dict):
                iv = {k: val[k] for k in ("value", "scale", "add") if k in val}
                if not iv:
                    errs.append(f"do {var}: object needs value, scale or add (ignored)")
                    continue
                if "value" in iv:
                    c, err = _coerce(iv["value"], typ)
                    if err:
                        errs.append(f"do {var}: {err} (ignored)")
                        continue
                    iv["value"] = c
                if any(k in iv and (not isinstance(iv[k], (int, float)) or isinstance(iv[k], bool))
                       for k in ("scale", "add")):
                    errs.append(f"do {var}: scale/add must be a number (ignored)")
                    continue
                if val.get("from") is not None:
                    iv["from"] = int(val["from"])
                do[var] = iv
            else:
                c, err = _coerce(val, typ)
                if err:
                    errs.append(f"do {var}: {err} (ignored)")
                    continue
                do[var] = {"value": c}
        desc = ", ".join([f"{k}={_fmt_iv(v)}" for k, v in do.items()] + [f"{k} {v}" for k, v in qual.items()])
        head = f"WHATIF {name}: do({desc})"
        body = [f"  note: {e}" for e in errs + q["notes"]]
        if not do and not qual:
            return "ERROR " + head + " -- no usable intervention", body, []
        targets = [t for t in q["targets"] if t in cg.vars]
        for t in q["targets"]:
            if t not in cg.vars:
                body.append(f"  {t}: unknown variable{_did_you_mean(t, cg.vars)}")
        pick = _picker(q["at"])
        runs = []
        if do:
            if q["samples"]:
                runs = [simulate(cg, steps=q["steps"], do=do, rng=random.Random(q["seed"] * 1_000_003 + i))
                        for i in range(q["samples"])]
            else:
                runs = [simulate(cg, steps=q["steps"], do=do)]
            at_txt = f"at step {q['at']}" if isinstance(q["at"], int) else (f"final step {q['steps']}" if cg.timed
                                                                             else "static")
            head += (f"  [{q['samples']} paired samples, seed {q['seed']}, {at_txt}]" if q["samples"]
                     else f"  [{at_txt}]")
            for t in targets:
                a = [pick(h[t]) for h, _ in runs]
                b = [pick(h[t]) for h, _ in base_runs]
                fail = _fail(runs, t)
                if q["samples"]:
                    d = [x - y for x, y in zip(a, b) if _isnum(x) and _isnum(y)]
                    body.append(f"  {t}: {_summ(a)}{fail}")
                    body.append(f"    baseline: {_summ(b)}")
                    if d:
                        body.append(f"    delta: {_summ(d)}; P(delta>0)={sum(x > 0 for x in d) / len(d):.3f} "
                                    f"P(delta<0)={sum(x < 0 for x in d) / len(d):.3f}")
                else:
                    dl = f"  (delta {_fmt(a[0] - b[0])})" if _isnum(a[0]) and _isnum(b[0]) else ""
                    body.append(f"  {t}: {_fmt(a[0])}  baseline {_fmt(b[0])}{dl}{fail}")
            for ev in q["events"]:
                try:
                    ast = _expr.parse(ev)
                except _expr.ExprError as e:
                    body.append(f"  event {ev!r}: {e}")
                    continue
                ra = [x for x in (_event_bool(ast, {n: pick(h) for n, h in hh.items()}) for hh, _ in runs)
                      if x is not None]
                rb = [x for x in (_event_bool(ast, {n: pick(h) for n, h in hh.items()}) for hh, _ in base_runs)
                      if x is not None]
                fa = f"{sum(ra) / len(ra):.3f}" if ra else "undefined"
                fb = f"{sum(rb) / len(rb):.3f}" if rb else "undefined"
                body.append(f"  P({ev}) = {fa}  baseline {fb}")
        # qualitative (A2 rules: direction from the baseline value of the intervened variable)
        qlines = []
        budget = QualBudget()
        b0 = base_runs[0][0] if base_runs else {}
        for var, iv in list(do.items()) + list(qual.items()):
            if isinstance(iv, str):
                direction = iv
            else:
                direction = "+"
                b = b0.get(var, [None])[-1] if b0.get(var) else None
                if "value" in iv and _isnum(b) and _isnum(iv["value"]):
                    dd = iv["value"] - b
                    direction = "+" if dd > 0 else "-" if dd < 0 else "0"
                elif "scale" in iv and _isnum(b):
                    dd = b * iv["scale"] - b
                    direction = "+" if dd > 0 else "-" if dd < 0 else "0"
                elif "add" in iv:
                    direction = "+" if iv["add"] > 0 else "-" if iv["add"] < 0 else "0"
            tg = [t for t in targets if t != var]
            signs, notes = propagate(cg, var, tg, budget) if tg and direction != "0" else ({}, [])
            for t in tg:
                if direction == "0":
                    qlines.append(f"    {var} unchanged -> {t}: 0")
                    continue
                net, why = _qual_text(cg, var, t, signs.get(t), direction, notes, budget)
                qlines.append(f"    {var} {'up' if direction == '+' else 'down'} -> {t}: {net}   ({why})")
        if qlines:
            body.append("  QUALITATIVE (edge signs):")
            body += qlines
        series = []
        if do and cg.timed and targets:
            series.append("  SERIES (intervened" + (", mean" if q["samples"] else "") + "):")
            hs = [h for h, _ in runs]
            for t in targets:
                row, shape = compact_series(_mean_series(hs, t))
                series.append(f"    {t}: {row}" + (f"\n      {shape}" if shape else ""))
        return head, body, series

    # ---------------------------------------------------------------- show / render / validity
    def _t_show(self, state, args):
        return _bound(self.render(state), MAX_OUT)

    def render(self, state):
        doc = self._doc(state)
        out = []
        if doc.source is not None:
            out.append("DOCUMENT")
            out += [f"{i:3d}| {ln}" for i, ln in enumerate(doc.source.replace("\r\n", "\n").split("\n"), 1)]
        else:
            out.append("DOCUMENT: none (graph given as A2 JSON)")
        if doc.targets:
            out.append("TARGETS: " + ", ".join(doc.targets))
        if doc.whatifs:
            out.append("WHATIFS: " + "; ".join(
                f"{w['name']}: do(" + ", ".join(f"{k}={v if isinstance(v, str) else _fmt_iv(v)}"
                                               for k, v in w["do"].items()) + ")" for w in doc.whatifs))
        if doc.errors:
            out.append("DOCUMENT HOLES:")
            out += [f"  - line {e['line']}" + (f" col {e['col']}" if e.get("col") else "") + f": {e['error']}"
                    + (f" (hint: {e['hint']})" if e.get("hint") else "") for e in doc.errors]
        head = "\n".join(out) + _holes_text(doc.cg, 30)
        # the document is the representation and is never cut; the compiled-graph listing repeats it, so it goes
        # last and is trimmed to the render bound
        compiled = "\nCOMPILED GRAPH\n" + CausalArm.render(self, ArmState(rep=doc.graph))
        if doc.source is None:
            return (head + compiled)[:RENDER_MAX]
        room = RENDER_MAX - len(head)
        if len(compiled) > room:
            cut = compiled[:max(0, room - 80)].rsplit("\n", 1)[0] if room > 200 else ""
            compiled = cut + "\n[compiled graph listing omitted to fit the render bound; the document above is complete]"
        return head + compiled

    def validity(self, state):
        doc = self._doc(state)
        v = CausalArm.validity(self, ArmState(rep=doc.graph))
        extra = [f"line {e['line']}: {e['error']}" for e in doc.errors]
        extra += [f"undefined name '{n}' in {w}" for n, w in doc.undefined()
                  if w.startswith("targets") or w.startswith("whatif")]
        v["holes"] = extra + v["holes"]
        v["targets"] = list(doc.targets)
        return v

    # ---------------------------------------------------------------- episode loop (document channel)
    def run_episode(self, ep, state, messages):
        from tournament.episode import FORCE_FINAL_MSG, MAX_TOOL_CALLS, ERROR_PREFIX, execute_tool_call
        from tournament.run import system_prompt
        messages[0]["content"] = system_prompt(self, True, ep.budget_calls, ep.budget_tokens)
        tools_fn = self.gold_tools if state.given_gold else self.current_tools
        trace, n_tools = [], 0

        def status():
            return (f"\n[budget: {len(ep.calls)} of {ep.budget_calls} responses used; your last response must be "
                    f"the FINAL ANSWER without tool calls]")
        while ep.can_continue():
            tools = tools_fn(state)
            m = ep.call(messages, tools=tools, label=f"turn{len(ep.calls)}")
            content = m.get("content") or ""
            block = fenced_block(content)
            state.pending_block = None if state.given_gold else block
            tcs = m.get("tool_calls") or []
            am = {"role": "assistant", "content": content}
            if tcs:
                am["tool_calls"] = tcs
            messages.append(am)
            if not tcs:
                if block is None:
                    return content, messages, trace
                # a graph block without a tool call: run it (implicit submit_graph), then continue
                if n_tools >= MAX_TOOL_CALLS:
                    res = f"{ERROR_PREFIX}: tool-call limit ({MAX_TOOL_CALLS}) reached; give your FINAL ANSWER."
                elif state.given_gold:
                    res = (f"{ERROR_PREFIX}: in this task the representation is given; it cannot be rebuilt or "
                           f"modified (the graph block was not run). Use the query tools.")
                else:
                    res = self.handle(state, "submit_graph", {})
                n_tools += 1
                res = res[:7900] + status()
                trace.append({"turn": len(ep.calls), "tool": "submit_graph",
                              "args": {"source": "fenced block", "chars": len(block)}, "result": res[:3000],
                              "error": res.startswith(ERROR_PREFIX)})
                messages.append({"role": "user", "content": "EXECUTOR RESULT (submit_graph)\n" + res})
                continue
            offered = {t["function"]["name"] for t in tools}
            for tc in tcs:
                name, args, res = execute_tool_call(self, state, tc, offered, n_tools, MAX_TOOL_CALLS)
                n_tools += 1
                res = res + status()
                trace.append({"turn": len(ep.calls), "tool": name, "args": args, "result": res[:3000],
                              "error": res.startswith(ERROR_PREFIX)})
                messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call{len(trace)}",
                                 "content": res})
        ep.forced_final = True
        messages.append({"role": "user", "content": FORCE_FINAL_MSG})
        m = ep.call(messages, tools=None, label="forced_final", final=True)
        messages.append({"role": "assistant", "content": m.get("content") or ""})
        return m.get("content") or "", messages, trace


# ============================================================================ helpers
def _picker(at):
    def pick(h):
        if not h:
            return None
        if isinstance(at, int) and not isinstance(at, bool):
            return h[at] if 0 <= at < len(h) else None
        return h[-1]
    return pick


def _cap_samples(cg, q, n_runs):
    """A batch of n_runs paired runs may use the work of one A2 intervene call (intervened + baseline)."""
    if q["samples"]:
        work = max(1, len(cg.vars) * (q["steps"] + 1))
        cap = max(1, (2 * MAX_WORK) // (work * max(2, n_runs)))
        if q["samples"] > cap:
            q["notes"].append(f"samples reduced to {cap} (work limit for this batch)")
            q["samples"] = cap


def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _fail(runs, t):
    for _, failed in runs:
        if t in failed:
            st, why = failed[t]
            return f"   [no value from step {st}: {why}]"
    return ""


def _cut_lines(text, limit):
    """Keep whole lines up to `limit` characters."""
    if len(text) <= limit:
        return text
    out, n = [], 0
    for ln in text.split("\n"):
        if n + len(ln) + 1 > limit - 30:
            out.append("  [... cut to the output bound]")
            break
        out.append(ln)
        n += len(ln) + 1
    return "\n".join(out)


def _bound(text, limit=MAX_OUT):
    if len(text) <= limit:
        return text
    return text[:limit - 40] + "\n[... output cut to the bound]"


G1C3 = G1C3Arm
