"""G3C2: counterfactual channel model -- structural equations queried at the counterfactual layer (G1C3 modified).

Spec: runs/discovery/gen3/specs/G3C2.md. The document, compiler, A2 simulator and construction protocol are G1C3's,
imported unchanged (`g1c3.py`, `a2_causal.py` and `_expr.py` are not edited). What is new is what the executor
computes from a run model:

  * CHANNELS: for each what-if do(X = x') and target Y, the effect is split over **mechanisms** (disjoint groups of
    mediators): W_S is the intervened run in which every channel outside S is held, per Monte Carlo member and per
    step, to its own baseline trajectory; v(S) = E[Y_{W_S} - Y_base]. v(C) is the total effect T (the G1C3 paired
    delta), v({}) the direct effect D, and the Shapley value phi_c gives T = D + sum phi_c exactly. Alone shares
    a_c = v({c}) - v({}) and but-for shares b_c = v(C) - v(C - {c}) expose interactions.
  * EXPLAIN: modified Halpern-Pearl actual causation (HP 2015) on the unrolled model: but-for causes, causes under
    a contingency W (|W| <= 2 channel variables held at their actual values), joint causes (pairs), safeguards (a
    candidate whose contrast is its working state), and which stated observations each candidate accounts for.
  * Two optional document lines: `mechanism name: v1, v2` and
    `explain name: <event> ; candidates c1=v1, ... [; observed e1, ...] [; at final|any|k]`.
    They are parsed in a pre-pass and blanked out before G1C3's parser, so every G1C3 document is a G3C2 document
    with the same numbers. Problems with them are notes, never rejections.
  * `patch_graph` also takes `upsert` (whole lines replacing the line that declares the same name, or appended) and
    `delete` (names); all entries are applied in one call, all-or-nothing, with no line numbers.

The arm runs in process and launches no subprocess; its run-time imports are declared in RUNTIME_DEPS (D-060).

Build notes (where this implementation departs from, or fixes a detail left open by, the spec):
  * Golds: spec §5 proposed copying G1C3's golds. The build instructions require golds built from the packets per
    tournament/GOLD_BUILDER_PROMPT.md, which forbids opening other arms' gold files, so tournament/gold/g3c2/ was
    built from scratch. The §6 compatibility test therefore runs on G3C2's own golds: each, with its mechanism/explain
    lines blanked, must give byte-identical evaluate/intervene output from G1C3Arm.
  * The ~900-character CHANNELS bound applies per (what-if, target) block.
  * Light mode computes exact shares when k <= 3 (its 2k + 2 worlds are then all 2^k worlds); otherwise it reports
    alone/but-for only, and `opposes` uses their midpoint.
  * The effort target (<= 1.3x A0+ output tokens) cannot be measured on the fake fixture (fakes return a fixed token
    count per call); it is measured at evaluation, as for every arm.
"""
import copy
import difflib
import itertools
import math
import random
import re
import time

from tournament.arms import _expr
from tournament.arms.a2_causal import QUAL_WORDS, _coerce, _dist_point, _dist_sample, _fmt, _lookup, _maybe_json
from tournament.arms.g1c3 import (EDGE_RE, MAX_BATCH, MAX_OUT, NAME_RE, VAR_RE, Doc, G1C3Arm, G1C3State, _const,
                                  _cut_lines, _did_you_mean, _split_comment, _split_top, _tool, _QUERY, fenced_block)

RUNTIME_DEPS = ["tournament/arms/_expr.py", "tournament/arms/g1c3.py", "tournament/arms/a2_causal.py"]

MUTATING = ("submit_graph", "patch_graph")
MAX_CHANNELS = 6               # more channels: the 5 largest |but-for| are kept, the rest merge into `other`
PATH_CAP = 400                 # simple X->Y paths enumerated for automatic channels
PATH_VISITS = 50_000           # node visits for that enumeration (a cap that binds is stated)
MEMBERS_MAX = 40               # decomposition members with Monte Carlo
MEMBERS_MIN = 8                # fewest members before switching to a cheaper mode
INTERVENE_BUDGET_S = 20.0      # CHANNELS work per intervene call (shared by its what-ifs)
LIGHT_BUDGET_S = 6.0           # light CHANNELS work per submit/patch (shared by the stored what-ifs)
EXPLAIN_BUDGET_S = 20.0        # EXPLAIN work per explain call
SUBMIT_EXPLAIN_BUDGET_S = 6.0  # stored explains run at submit/patch
CHANNEL_CHARS = 900            # bound per CHANNELS block (one what-if, one target)
EXPLAIN_CHARS = 2000           # bound per EXPLAIN block
MAX_CONTINGENCY_VARS = 8
MAX_CONTINGENCY_RUNS = 40      # per candidate and member (1 but-for + singles + pairs over <= 8 variables = 37)
MAX_JOINT_PAIRS = 15
CAVEAT = "note: shares are counterfactual accounting (a frozen mechanism follows its baseline path)"
MECH_RE = re.compile(r"^\s*mechanism\s+([A-Za-z_]\w*)\s*:(.*)$", re.S)
EXPL_RE = re.compile(r"^\s*explain\s+([A-Za-z_]\w*)\s*:(.*)$", re.S)
AT_RE = re.compile(r"^(.*?)\s+at\s+(final|any|\d+)\b\s*(.*)$", re.S)


# ============================================================================ counterfactual simulator
def simulate_cf(cg, steps=None, do=None, frozen=None, rng=None):
    """a2_causal.simulate with one added branch: a variable in `frozen` takes frozen[n][t] at step t (its baseline
    trajectory in the same member). The exogenous sampling code and its order are copied verbatim, so the same seed
    gives the same exogenous draw u. Returns (hist, failed)."""
    steps = cg.steps if steps is None else steps
    do = do or {}
    frozen = frozen or {}
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
            if n in frozen:                                     # the added branch: hold to the baseline path
                fz = frozen[n]
                v = fz[t] if t < len(fz) else None
                if v is None:
                    failed.setdefault(n, (t, "frozen at a baseline step without a value"))
                hist[n].append(v)
                continue
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


# ============================================================================ pre-pass: mechanism and explain lines
def _split_event_at(src):
    """'revenue at final < 500' -> ('revenue < 500', 'final'); no inline `at` -> (src, None)."""
    m = AT_RE.match(src.strip())
    if not m:
        return src.strip(), None
    at = m.group(2)
    return (m.group(1) + " " + m.group(3)).strip(), (int(at) if at.isdigit() else at)


def _parse_explain(name, body):
    """Body of an explain line -> spec dict, or raise ValueError (the line is then skipped with a note)."""
    spec = {"name": name, "event": None, "at": None, "candidates": [], "observed": [], "samples": None}
    for k, seg in enumerate(_split_top(body, ";")):
        seg = seg.strip()
        if not seg:
            continue
        w = seg.split(None, 1)
        key, rest = w[0].lower(), (w[1] if len(w) > 1 else "")
        if key in ("candidates", "candidate"):
            for c in _split_top(rest, ","):
                c = c.strip()
                if not c:
                    continue
                m = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(.+)", c, re.S)
                if not m:
                    raise ValueError(f"bad candidate {c!r} (form: name=contrast, e.g. heater_ok=true)")
                vs = m.group(2).strip()
                typ = "category" if NAME_RE.fullmatch(vs) and vs not in ("true", "false") else "real"
                spec["candidates"].append([m.group(1), _const(vs, typ)])
        elif key in ("observed", "observations", "observation"):
            spec["observed"] += [x.strip() for x in _split_top(rest, ",") if x.strip()]
        elif key == "at":
            r = rest.strip()
            if not (r in ("final", "any") or r.isdigit()):
                raise ValueError(f"bad at {r!r} (final | any | step)")
            spec["at"] = int(r) if r.isdigit() else r
        elif key == "samples":
            try:
                spec["samples"] = int(float(rest))
            except ValueError:
                raise ValueError("samples needs an integer")
        elif k == 0 or key == "event":
            ev = rest if key == "event" else seg
            if spec["event"] is not None:
                raise ValueError("two events")
            spec["event"] = ev.strip()
        else:
            raise ValueError(f"unknown explain part {key!r} (parts: <event> ; candidates ... ; observed ... ; at ...)")
    if not spec["event"]:
        raise ValueError("explain without an event")
    if not spec["candidates"]:
        raise ValueError("explain without candidates (; candidates name=contrast, ...)")
    _expr.parse(_split_event_at(spec["event"])[0])
    for o in spec["observed"]:
        _expr.parse(_split_event_at(o)[0])
    return spec


def prepass(text):
    """Find `mechanism name:` and `explain name:` lines, blank them (line numbers are kept for G1C3's messages).
    Returns (cleaned text, mechanisms, explains, notes). A G1C3 variable named `mechanism` or `explain`
    (`mechanism : real = 1`) does not match: the pre-pass needs a name before the colon."""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    mechs, expls, notes, out = [], [], [], []
    for no, raw in enumerate(lines, 1):
        code, prov = _split_comment(raw)
        m, e = MECH_RE.match(code), EXPL_RE.match(code)
        if not m and not e:
            out.append(raw)
            continue
        out.append("")
        if m:
            names = [x.strip() for x in m.group(2).split(",") if x.strip()]
            bad = [x for x in names if not NAME_RE.fullmatch(x)]
            if bad or not names:
                notes.append(f"line {no}: mechanism {m.group(1)!r}: " + (f"bad names {bad}" if bad else "no variables")
                             + " (line skipped; form: mechanism name: v1, v2)")
                continue
            if any(x["name"] == m.group(1) for x in mechs):
                notes.append(f"line {no}: mechanism {m.group(1)!r} declared twice; later one ignored")
                continue
            mechs.append({"name": m.group(1), "vars": names, "line": no, "provenance": prov})
        else:
            try:
                spec = _parse_explain(e.group(1), e.group(2))
            except (ValueError, _expr.ExprError) as ex:
                notes.append(f"line {no}: explain {e.group(1)!r} skipped: {ex}")
                continue
            if any(x["name"] == spec["name"] for x in expls):
                notes.append(f"line {no}: explain {spec['name']!r} declared twice; later one ignored")
                continue
            spec["line"] = no
            if prov:
                spec["provenance"] = prov
            expls.append(spec)
    return "\n".join(out), mechs, expls, notes


class G3Doc(Doc):
    """A G1C3 Doc plus the mechanism and explain lines (and their notes)."""

    def __init__(self, source=None, graph=None):
        if source is not None:
            cleaned, mechs, expls, notes = prepass(source)
            Doc.__init__(self, source=cleaned)
            self.source = source
        else:
            g = copy.deepcopy(graph) if isinstance(graph, dict) else {}
            mechs = [x for x in (g.pop("mechanisms", None) or []) if isinstance(x, dict) and x.get("name")]
            expls = [x for x in (g.pop("explains", None) or []) if isinstance(x, dict) and x.get("name")]
            notes = []
            Doc.__init__(self, graph=g)
        self.notes = notes
        self.mechanisms, self.explains = [], []
        self._check(mechs, expls)

    def _check(self, mechs, expls):
        cg, owner = self.cg, {}
        for m in mechs:
            keep = []
            where = f"line {m['line']}: " if m.get("line") else ""
            for v in m.get("vars") or []:
                if v not in cg.vars:
                    self.notes.append(f"{where}mechanism {m['name']!r}: unknown variable '{v}'"
                                      f"{_did_you_mean(v, cg.vars)} (ignored)")
                elif v in owner:
                    self.notes.append(f"{where}mechanism {m['name']!r}: '{v}' is already in mechanism "
                                      f"{owner[v]!r} (a variable belongs to one mechanism; ignored here)")
                else:
                    if v not in cg.eqs:
                        self.notes.append(f"{where}mechanism {m['name']!r}: '{v}' has no equation, so it cannot "
                                          "carry an effect")
                    owner[v] = m["name"]
                    keep.append(v)
            self.mechanisms.append(dict(m, vars=keep))
        for x in expls:
            where = f"line {x['line']}: " if x.get("line") else ""
            bad = []
            for src in [x.get("event") or ""] + list(x.get("observed") or []):
                try:
                    bad += [r for r, _ in sorted(_expr.refs(_expr.parse(_split_event_at(src)[0])))
                            if r not in cg.vars and r not in ("t", "dt")]
                except _expr.ExprError as e:
                    bad.append(f"<{e}>")
            cands = []
            for c in x.get("candidates") or []:
                if not (isinstance(c, (list, tuple)) and len(c) == 2 and c[0] in cg.vars):
                    nm = c[0] if isinstance(c, (list, tuple)) and c else c
                    self.notes.append(f"{where}explain {x['name']!r}: unknown candidate {nm!r}"
                                      f"{_did_you_mean(str(nm), cg.vars)} (ignored)")
                    continue
                cands.append(list(c))
            for b in sorted(set(bad)):
                self.notes.append(f"{where}explain {x['name']!r}: unknown name {b!r}{_did_you_mean(b, cg.vars)}")
            if not cands:
                self.notes.append(f"{where}explain {x['name']!r}: no usable candidate (skipped)")
                continue
            self.explains.append(dict(x, candidates=cands))

    def export(self):
        r = Doc.export(self)
        r["mechanisms"] = copy.deepcopy(self.mechanisms)
        r["explains"] = copy.deepcopy(self.explains)
        return r


# ============================================================================ the causal structure
def _children(cg):
    """Numerical dependencies (equation references, any lag): parent -> {children}."""
    ch = {n: set() for n in cg.vars}
    for v, ast in cg.eqs.items():
        for r, _ in _expr.refs(ast):
            if r in cg.vars and r != v:
                ch[r].add(v)
    return ch


def _reach(starts, nbrs):
    seen, stack = set(starts), list(starts)
    while stack:
        n = stack.pop()
        for m in nbrs.get(n, ()):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return seen


def mediators(cg, xs, y, ch=None):
    """Endogenous variables (with an equation), other than X and Y, on a directed path X -> Y (lags included)."""
    ch = ch if ch is not None else _children(cg)
    par = {n: set() for n in cg.vars}
    for a, bs in ch.items():
        for b in bs:
            par[b].add(a)
    down = _reach([x for x in xs if x in cg.vars], ch)
    up = _reach([y], par) if y in cg.vars else set()
    return [n for n in cg.order_decl if n in down and n in up and n not in xs and n != y and n in cg.eqs]


def _simple_paths(ch, xs, y, allowed):
    """Simple paths X -> Y through `allowed` nodes; (paths, capped)."""
    paths, visits = [], 0
    for x in xs:
        stack = [(x, iter(sorted(ch.get(x, ()))))]
        on = [x]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                on.pop()
                continue
            visits += 1
            if visits > PATH_VISITS:
                return paths, True
            if nxt == y:
                paths.append(tuple(on[1:]))
                if len(paths) >= PATH_CAP:
                    return paths, True
                continue
            if nxt in on or nxt not in allowed:
                continue
            stack.append((nxt, iter(sorted(ch.get(nxt, ())))))
            on.append(nxt)
    return paths, False


def channels_for(cg, mechs, xs, y, ch=None):
    """(channels, notes): channels are [{name, vars, declared}] over the mediators of X -> Y. Declared mechanisms come
    first (restricted to the mediators); the remaining mediators are grouped by path equivalence."""
    ch = ch if ch is not None else _children(cg)
    med = mediators(cg, xs, y, ch)
    mset, notes, out, used = set(med), [], [], set()
    for m in mechs or []:
        vs = [v for v in m.get("vars") or [] if v in mset and v not in used]
        if vs:
            out.append({"name": m["name"], "vars": vs, "declared": True})
            used.update(vs)
    rest = [v for v in med if v not in used]
    if rest:
        paths, capped = _simple_paths(ch, [x for x in xs if x in cg.vars], y, mset)
        groups = {}
        if capped:
            notes.append(f"path cap reached ({PATH_CAP} paths): each undeclared mediator is its own channel")
            for v in rest:
                groups[(v,)] = [v]
        else:
            for v in rest:
                key = frozenset(i for i, p in enumerate(paths) if v in p)
                groups.setdefault(key, []).append(v)
        names = {c["name"] for c in out}
        for vs in groups.values():
            nm = vs[0] if vs[0] not in names else vs[0] + "_chain"
            names.add(nm)
            out.append({"name": nm, "vars": vs, "declared": False})
    return out, notes


# ============================================================================ worlds and the Shapley game
def _num(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


class OutOfTime(Exception):
    pass


class Worlds:
    """Paired runs for one intervention: a baseline per member and worlds keyed by the set of frozen variables
    (cached, so targets and queries share runs). Runs stop at the deadline (OutOfTime)."""

    def __init__(self, cg, steps, do, n, seed, deadline):
        self.cg, self.steps, self.do, self.n, self.seed, self.deadline = cg, steps, do, n, seed, deadline
        self.mc = n > 0
        self.members = max(1, n)
        self.cache, self.runs = {}, 0
        t0 = time.perf_counter()
        self.base = [self._run(i, None, None) for i in range(self.members)]
        self.t_run = max(1e-5, (time.perf_counter() - t0) / self.members)

    def _rng(self, i):
        return random.Random(self.seed * 1_000_003 + i) if self.mc else None

    def _run(self, i, do, frozen):
        self.runs += 1
        return simulate_cf(self.cg, steps=self.steps, do=do, frozen=frozen, rng=self._rng(i))[0]

    def world(self, frozen_vars, do=None):
        do = self.do if do is None else do
        key = (frozenset(frozen_vars), tuple(sorted((k, repr(v)) for k, v in do.items())))
        if key not in self.cache:
            out = []
            for i in range(self.members):
                if time.perf_counter() > self.deadline:
                    raise OutOfTime()
                fz = {v: self.base[i][v] for v in frozen_vars}
                out.append(self._run(i, do, fz))
            self.cache[key] = out
        return self.cache[key]

    def time_left(self):
        return self.deadline - time.perf_counter()


def shapley_exact(k, v):
    """Exact Shapley values of a k-player game v(frozenset) -> float."""
    fact = math.factorial
    phi = []
    for c in range(k):
        others = [j for j in range(k) if j != c]
        s = 0.0
        for r in range(len(others) + 1):
            w = fact(r) * fact(k - r - 1) / fact(k)
            for S in itertools.combinations(others, r):
                S = frozenset(S)
                s += w * (v(S | {c}) - v(S))
        phi.append(s)
    return phi


def shapley_perm(k, v, n_pairs, rng, more=lambda: True):
    """Permutation-sampling Shapley (Castro, Gomez & Tejada 2009) with antithetic pairs (a permutation and its
    reverse). Returns (phi, se, pairs used). Each permutation's marginals sum to v(C) - v({}), so efficiency is exact."""
    sums = [[] for _ in range(k)]
    used = 0
    for _ in range(n_pairs):
        if not more():
            break
        p = list(range(k))
        rng.shuffle(p)
        pair = []
        try:
            for perm in (p, p[::-1]):
                marg, S, prev = [0.0] * k, frozenset(), v(frozenset())
                for c in perm:
                    S2 = S | {c}
                    cur = v(S2)
                    marg[c] = cur - prev
                    S, prev = S2, cur
                pair.append(marg)
        except OutOfTime:
            break
        used += 1
        for c in range(k):
            sums[c].append((pair[0][c] + pair[1][c]) / 2)
    if not used:
        return None, None, 0
    phi = [sum(x) / used for x in sums]
    se = [math.sqrt(sum((a - m) ** 2 for a in x) / (used - 1) / used) if used > 1 else float("nan")
          for x, m in zip(sums, phi)]
    return phi, se, used


def _sg(x, ref=1.0):
    """Signed compact number; float noise relative to `ref` prints as 0."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if abs(x) <= 1e-9 * max(1.0, abs(ref)):
        return "0"
    s = _fmt(float(x))
    return s if s.startswith("-") else "+" + s


def _pick_step(h, at):
    if not h:
        return None
    if isinstance(at, int) and not isinstance(at, bool):
        return h[at] if 0 <= at < len(h) else None
    return h[-1]


def decompose(W, y, chans, mode, at="final"):
    """The channel game for target y over the worlds W. mode: 'exact' | 'perm' | 'light' | 'auto'.
    Returns a result dict (see channels_text); never raises OutOfTime (a partial result says so)."""
    k = len(chans)
    res = {"target": y, "channels": chans, "mode": mode, "notes": [], "members": W.members, "mc": W.mc}
    vals = {}                                               # S -> per-member y values

    def frozen_of(S):
        return frozenset(v for j, c in enumerate(chans) if j not in S for v in c["vars"])

    def get(S):
        if S not in vals:
            hs = W.world(frozen_of(S))
            vals[S] = [_num(_pick_step(h[y], step)) for h in hs]
        return vals[S]

    full = frozenset(range(k))
    try:
        hs_full = W.world(frozenset())
    except OutOfTime:
        res["mode"] = "none"
        res["notes"].append("not computed (time)")
        return res
    # the query step
    step = at
    if at == "peak":
        L = min(len(hs_full[0][y]), len(W.base[0][y]))
        best, step = -1.0, L - 1
        for t in range(L):
            ds = [_num(h[y][t]) - _num(b[y][t]) for h, b in zip(hs_full, W.base)
                  if _num(h[y][t]) is not None and _num(b[y][t]) is not None]
            if ds and abs(sum(ds) / len(ds)) > best + 1e-12:
                best, step = abs(sum(ds) / len(ds)), t
        res["peak_step"] = step
    res["step"] = step
    vals[full] = [_num(_pick_step(h[y], step)) for h in hs_full]
    base = [_num(_pick_step(b[y], step)) for b in W.base]
    try:
        get(frozenset())
    except OutOfTime:
        res["mode"] = "none"
        res["notes"].append("not computed (time)")
        return res
    if k == 0:
        mode = "exact"
    elif mode == "light" and 2 ** k <= 2 * k + 2:
        mode = "exact"                                      # k <= 3: the light worlds are all the worlds
    res["mode"] = mode
    computed = True
    try:
        if mode == "exact":
            for r in range(1, k):
                for S in itertools.combinations(range(k), r):
                    get(frozenset(S))
        else:
            for c in range(k):                              # alone and but-for worlds
                get(frozenset([c]))
                get(full - {c})
    except OutOfTime:
        computed = False
    perm_rng = random.Random(20261019)
    # members valid in every computed world
    ok = [i for i in range(W.members) if base[i] is not None and all(vs[i] is not None for vs in vals.values())]
    if not ok:
        res["mode"] = "none"
        res["notes"].append(f"target {y} has no numeric value in the runs")
        return res
    if len(ok) < W.members:
        res["notes"].append(f"{W.members - len(ok)} member(s) without a value dropped")
    res["n_used"] = len(ok)

    def v(S):
        xs = get(S)
        return sum(xs[i] - base[i] for i in ok) / len(ok)

    def vk(S):                                                # v over already computed worlds only
        return v(S) if S in vals else None

    res["T"], res["D"] = v(full), v(frozenset())
    res["alone"] = [None if vk(frozenset([c])) is None else vk(frozenset([c])) - res["D"] for c in range(k)]
    res["butfor"] = [None if vk(full - {c}) is None else res["T"] - vk(full - {c}) for c in range(k)]
    if not computed:
        res["mode"] = "minimal"
        res["phi"] = None
        return res
    if mode == "exact":
        res["phi"] = shapley_exact(k, v)
    elif mode == "perm":
        def safe_v(S):                                    # a world without a value for a used member drops the pair
            xs = get(S)
            if any(xs[i] is None for i in ok):
                raise OutOfTime()
            return v(S)
        phi, se, used = shapley_perm(k, safe_v, 500, perm_rng, more=lambda: W.time_left() > 0)
        if phi is None:
            res["mode"] = "minimal"
            res["phi"] = None
            return res
        res["phi"], res["se"], res["perms"] = phi, se, 2 * used
    else:                                                     # light
        res["phi"] = None
    return res


def _flags(res):
    """Per channel: 'opposes' / 'interacts' flags (on phi, or on the alone/but-for midpoint in light mode)."""
    T, k = res["T"], len(res["channels"])
    shares = res.get("phi") or [((a if a is not None else 0) + (b if b is not None else 0)) / 2
                                for a, b in zip(res["alone"], res["butfor"])]
    ref = abs(T) if abs(T) > 1e-12 else max([abs(s) for s in shares] + [0.0])
    out = []
    for c in range(k):
        f = []
        s, a, b = shares[c], res["alone"][c], res["butfor"][c]
        if abs(T) > 1e-12 and s * T < 0 and abs(s) >= 0.10 * abs(T):
            f.append("opposes")
        if a is not None and b is not None and abs(a - b) > 0.25 * ref and ref > 0:
            f.append("interacts")
        out.append(f)
    return shares, out, ref


def _members_text(res):
    if not res["mc"]:
        return "deterministic"
    return f"{res.get('n_used', res['members'])} paired members, mean"


def channels_text(res, label, chars=CHANNEL_CHARS):
    """The CHANNELS block for one what-if and one target."""
    y, chans = res["target"], res["channels"]
    step = res.get("step", "final")
    st = (f"peak step {step}" if "peak_step" in res else f"step {step}" if isinstance(step, int) else "final step")
    if res["mode"] == "none":
        return f"CHANNELS [{label}] {y}: " + "; ".join(res["notes"])
    k = len(chans)
    T, D = res["T"], res["D"]
    if k == 0:
        return (f"CHANNELS [{label}] {y} ({st}; {_members_text(res)}): no mediators -- the whole effect is direct "
                f"({_sg(T)})")
    mode = res["mode"]
    how = {"exact": f"exact Shapley over {k} channel{'s' if k > 1 else ''}",
           "perm": f"approximate Shapley, {res.get('perms', 0)} permutations",
           "minimal": "Shapley shares not computed (time); alone and but-for only",
           "light": "light: alone and but-for only; full shares: call intervene"}[mode]
    shares, flags, ref = _flags(res)
    order = sorted(range(k), key=lambda c: -abs(shares[c]))
    neg = [c for c in order if res.get("phi") is not None and abs(shares[c]) < 0.05 * ref]
    main = [c for c in order if c not in neg]
    head = f"CHANNELS [{label}] {y} ({st}; {how}; {_members_text(res)})"
    lines = [head]
    if res.get("phi") is not None:
        lines.append(f"  total {_sg(T, ref)} = direct {_sg(D, ref)}"
                     + "".join(f" + {chans[c]['name']} {_sg(shares[c], ref)}" for c in main)
                     + (" + negligible" if neg else ""))
    else:
        lines.append(f"  total {_sg(T, ref)} = direct {_sg(D, ref)} + channels {_sg(T - D, ref)}")
    w = max(len(chans[c]["name"]) + 3 + len(", ".join(chans[c]["vars"][:4])) for c in main) if main else 10
    w = min(w, 46)
    fw = max([len(" ".join(flags[c])) for c in main] + [0])
    for c in main:
        mem = ", ".join(chans[c]["vars"][:4]) + (f" +{len(chans[c]['vars']) - 4}" if len(chans[c]["vars"]) > 4 else "")
        nm = f"{chans[c]['name']} ({mem})"
        share = ""
        if res.get("phi") is not None:
            share = _sg(shares[c], ref)
            if res.get("se"):
                share += f" ±{_fmt(res['se'][c])}" if not math.isnan(res["se"][c]) else ""
        fl = " ".join(flags[c])
        ab = (f"alone {_sg(res['alone'][c], ref) if res['alone'][c] is not None else 'n/a'}  "
              f"but-for {_sg(res['butfor'][c], ref) if res['butfor'][c] is not None else 'n/a'}")
        cols = [nm.ljust(w), share.rjust(8)] + ([fl.ljust(fw)] if fw else []) + [ab]
        lines.append(("  " + "  ".join(cols)).rstrip())
    if neg:
        lines.append("  negligible: " + ", ".join(chans[c]["name"] for c in neg))
    for n in res["notes"]:
        lines.append(f"  note: {n}")
    if abs(T) <= 1e-12 and any(abs(s) > 1e-9 for s in shares):
        lines.append("  note: the total is 0 because the channels cancel")
    lines.append("  " + CAVEAT)
    out, n = [], 0
    for i, ln in enumerate(lines):
        if n + len(ln) + 1 > chars - 60 and i < len(lines) - 1:
            out.append(f"  [... {len(lines) - 1 - i} lower-ranked line(s) cut to the bound]")
            out.append("  " + CAVEAT)
            break
        out.append(ln)
        n += len(ln) + 1
    return "\n".join(out)


# ============================================================================ actual causation
def _ev_at(ast, h, at):
    """Event value in one run: at final | step k | any step."""
    def at_step(t):
        vals = {n: hh[t] for n, hh in h.items() if t < len(hh) and hh[t] is not None}
        try:
            return bool(_expr.eval_on(ast, vals))
        except (_expr.EvalError, _expr.ExprError):
            return None
    L = max((len(x) for x in h.values()), default=0)
    if not L:
        return None
    if at == "any":
        r = [at_step(t) for t in range(L)]
        return True if any(x is True for x in r) else (None if all(x is None for x in r) else False)
    if isinstance(at, int) and not isinstance(at, bool):
        return at_step(at) if 0 <= at < L else None
    return at_step(L - 1)


def _at_text(at):
    return "any step" if at == "any" else f"step {at}" if isinstance(at, int) else "final step"


def _frac(xs):
    xs = [x for x in xs if x is not None]
    return (sum(1 for x in xs if x) / len(xs)) if xs else None


def _p(x):
    return "n/a" if x is None else f"{x:.2f}"


def explain_run(doc, spec, n, seed, steps, budget_s):
    """The EXPLAIN block for one explain spec."""
    cg = doc.cg
    deadline = time.perf_counter() + budget_s
    name = spec.get("name") or "explain"
    ev_src, ev_at = _split_event_at(spec["event"])
    at = ev_at if ev_at is not None else (spec.get("at") or "final")
    try:
        ev = _expr.parse(ev_src)
    except _expr.ExprError as e:
        return f"EXPLAIN [{name}]: ERROR bad event {ev_src!r}: {e}"
    W = Worlds(cg, steps, {}, n, seed, deadline)
    lines = [f"EXPLAIN [{name}] event: {ev_src} ({_at_text(at)}; "
             + (f"{W.members} paired members" if W.mc else "deterministic") + ")"]
    top = []
    phi_actual = [_ev_at(ev, h, at) for h in W.base]
    p_act = _frac(phi_actual)
    # observations
    obs = []
    for o in spec.get("observed") or []:
        osrc, oat = _split_event_at(o)
        try:
            obs.append((osrc, oat if oat is not None else at, _expr.parse(osrc)))
        except _expr.ExprError as e:
            top.append(f"  note: observation {o!r} does not parse ({e})")
    obs_actual = [[_ev_at(a, h, oat) for h in W.base] for _, oat, a in obs]
    for (osrc, oat, _), xs in zip(obs, obs_actual):
        f = _frac(xs)
        if not f:
            top.append(f"  WARNING: the model's actual world does not reproduce observation '{osrc}' "
                       f"({_at_text(oat)}; P {_p(f)})")
    lines += top
    if p_act is None:
        lines.append("  actual world: the event has no value (check its variables)")
    else:
        lines.append(f"  actual world: event {'happens' if p_act > 0 else 'does not happen'} (P {_p(p_act)})")
    ch_all = _children(cg)
    ev_vars = [r for r, _ in sorted(_expr.refs(ev)) if r in cg.vars]
    cands = []
    for c, contrast in spec.get("candidates") or []:
        if c not in cg.vars:
            lines.append(f"  {c}: unknown variable{_did_you_mean(c, cg.vars)}")
            continue
        val, err = _coerce(contrast, cg.vars[c]["type"])
        if err:
            lines.append(f"  {c}: contrast {contrast!r} does not fit type {cg.vars[c]['type']} ({err})")
            continue
        cands.append((c, val))
    timed_out = []
    info = {}
    for c, cv in cands:
        do_c = {c: {"value": cv}}
        endo = c in cg.eqs
        act = []
        for i, h in enumerate(W.base):
            hs = h[c]
            if endo:
                act.append(any(x is not None and x != cv for x in hs))
            else:
                act.append(bool(hs) and hs[0] is not None and hs[0] != cv)
        try:
            alt = W.world(frozenset(), do=do_c)
        except OutOfTime:
            timed_out.append(c)
            continue
        phi_alt = [_ev_at(ev, h, at) for h in alt]
        # channel variables for contingencies: mediators from c to the event's variables, nearest the event first
        cv_vars = []
        for y in ev_vars:
            for m in reversed(mediators(cg, [c], y, ch_all)):
                if m not in cv_vars:
                    cv_vars.append(m)
        cv_vars = cv_vars[:MAX_CONTINGENCY_VARS]
        Ws = [(w,) for w in cv_vars] + list(itertools.combinations(cv_vars, 2))
        Ws = Ws[:MAX_CONTINGENCY_RUNS - 1]
        kinds = []
        for i in range(W.members):
            if not phi_actual[i]:
                kinds.append(("no event", None))
            elif not act[i]:
                kinds.append(("not active", None))
            elif phi_alt[i] is False:
                kinds.append(("but-for", None))
            else:
                kinds.append(("pending", None))
        if any(kd == "pending" for kd, _ in kinds) and Ws:
            for Wset in Ws:
                if not any(kd == "pending" for kd, _ in kinds):
                    break
                if time.perf_counter() > deadline:
                    timed_out.append(c)
                    break
                for i in range(W.members):
                    if kinds[i][0] != "pending":
                        continue
                    fz = {w: W.base[i][w] for w in Wset}
                    W.runs += 1
                    h = simulate_cf(cg, steps=W.steps, do=do_c, frozen=fz, rng=W._rng(i))[0]
                    if _ev_at(ev, h, at) is False:
                        kinds[i] = ("contingency", Wset)
        info[c] = {"cv": cv, "act": act, "alt": alt, "phi_alt": phi_alt, "kinds": kinds, "Wn": len(cv_vars)}
    # joint causes over pairs of candidates that are not causes alone
    pairs = [p for p in itertools.combinations([c for c, _ in cands if c in info], 2)][:MAX_JOINT_PAIRS]
    joint = {c: {} for c in info}
    for c, d in pairs:
        need = [i for i in range(W.members) if info[c]["kinds"][i][0] == "pending" and info[d]["kinds"][i][0] == "pending"]
        if not need:
            continue
        do_cd = {c: {"value": info[c]["cv"]}, d: {"value": info[d]["cv"]}}
        if time.perf_counter() > deadline:
            timed_out.append(f"{c}+{d}")
            continue
        for i in need:
            W.runs += 1
            h = simulate_cf(cg, steps=W.steps, do=do_cd, rng=W._rng(i))[0]
            if _ev_at(ev, h, at) is False:
                joint[c].setdefault(d, set()).add(i)
                joint[d].setdefault(c, set()).add(i)
    # per candidate text
    for c, cv in cands:
        if c not in info:
            continue
        x = info[c]
        a0 = W.base[0][c]
        firsts = [h[c][0] for h in W.base if h[c]]
        if c in cg.eqs:
            a_txt = "trajectory"
        elif not a0:
            a_txt = "-"
        elif len(set(map(repr, firsts))) > 1:
            nums = [_num(x) for x in firsts if _num(x) is not None]
            a_txt = "varies by member" + (f", mean {_fmt(sum(nums) / len(nums))}" if nums else "")
        else:
            a_txt = _fmt(a0[0])
        happened = [i for i in range(W.members) if phi_actual[i]]
        p_alt = _frac(x["phi_alt"])
        lbl = f"  {c} (actual {a_txt}, contrast {_fmt(cv)}): "
        if not happened:
            lines.append(lbl + "event did not happen in the actual world")
            continue
        nh = len(happened)
        cls = []                                          # per member: (kind, contingency set or joint partners)
        for i in happened:
            kd, Wset = x["kinds"][i]
            if kd == "pending":
                partners = sorted(d for d, s in joint[c].items() if i in s)
                cls.append(("joint", tuple(partners)) if partners else ("not a cause", None))
            else:
                cls.append((kd, Wset))
        parts = []
        for kd in ("but-for", "contingency", "joint", "not a cause", "not active"):
            sel = [w for k2, w in cls if k2 == kd]
            if not sel:
                continue
            fr = len(sel) / nh
            if kd == "but-for":
                txt = f"but-for cause (P(event) {_p(p_act)} -> {_p(p_alt)})"
            elif kd == "contingency":
                common = max(set(sel), key=sel.count)
                txt = "cause under contingency W={" + ", ".join(f"{w} held at actual" for w in common) + "}"
            elif kd == "joint":
                common = max(set(sel), key=sel.count)
                txt = "joint cause with " + ", ".join(common) + " (neither alone; both contrasts remove the event)"
            elif kd == "not active":
                txt = "not active (its actual value equals the contrast)"
            else:
                txt = "not a cause" + (" (time ran out before every test)" if c in timed_out else "")
            parts.append(txt + (f" in {fr:.2f} of members" if W.mc else ""))
        lines.append(lbl + "; ".join(parts))
        # the cause's mechanism: light channels of do(c = contrast) on the event's first variable
        if any(k2 in ("but-for", "contingency") for k2, _ in cls) and ev_vars and W.time_left() > 0:
            y = next((v for v in ev_vars if v in cg.eqs), None)
            if y is not None and y != c:
                chans, _ = channels_for(cg, doc.mechanisms, [c], y, ch_all)
                if len(chans) > MAX_CHANNELS:
                    chans = chans[:MAX_CHANNELS - 1] + [{"name": "other", "declared": False,
                                                         "vars": [v for q in chans[MAX_CHANNELS - 1:] for v in q["vars"]]}]
                Wc = _SubWorlds(W, {c: {"value": cv}})
                fits = (2 ** len(chans)) * W.t_run * W.members < W.time_left() / 4
                r = decompose(Wc, y, chans, "exact" if fits else "light", at if isinstance(at, int) else "final")
                if r["mode"] not in ("none",) and "T" in r:
                    shares, flags, ref = _flags(r)
                    order = sorted(range(len(chans)), key=lambda j: -abs(shares[j]))[:4]
                    if r.get("phi") is not None:
                        parts = "".join(f" + {chans[j]['name']} {_sg(shares[j], ref)}" for j in order)
                    else:
                        parts = " + channels (alone/but-for): " + ", ".join(
                            f"{chans[j]['name']} {_sg(r['alone'][j], ref)}/{_sg(r['butfor'][j], ref)}" for j in order)
                    lines.append(f"     under the contrast: {y} {_sg(r['T'], ref)} = direct {_sg(r['D'], ref)}"
                                 + parts)
    # observation table
    if obs:
        lines.append("  OBSERVATIONS (holds in the actual world; accounted for by = fails under that candidate's contrast)")
        for (osrc, oat, oast), xs in zip(obs, obs_actual):
            f = _frac(xs)
            acc = []
            for c, _ in cands:
                if c not in info:
                    continue
                alt = [_ev_at(oast, h, oat) for h in info[c]["alt"]]
                held = [i for i in range(W.members) if xs[i]]
                if held:
                    fr = sum(1 for i in held if alt[i] is False) / len(held)
                    if fr > 0:
                        acc.append(c + (f" ({fr:.2f})" if W.mc else ""))
            lines.append(f"  - {osrc} ({_at_text(oat)}): " + ("reproduced" if f else "NOT reproduced")
                         + f" (P {_p(f)}); accounted for by: " + (", ".join(acc) if acc else "none"))
    lines.append(f"  note: contingencies searched: |W| <= 2 among the channel variables (<= {MAX_CONTINGENCY_VARS}) "
                 "held at their actual values; a cause that needs a larger contingency is reported 'not a cause'")
    if timed_out:
        lines.append(f"  note: time budget ({budget_s:g} s) ran out; tests incomplete for: " + ", ".join(timed_out))
    return _cut_block(lines, EXPLAIN_CHARS)


class _SubWorlds:
    """A view of Worlds under another intervention, sharing the baseline members (for the cause's channels)."""

    def __init__(self, W, do):
        self.W, self.do = W, do
        self.base, self.members, self.mc = W.base, W.members, W.mc

    def world(self, frozen_vars, do=None):
        return self.W.world(frozen_vars, do=self.do if do is None else do)

    def time_left(self):
        return self.W.time_left()


def _cut_block(lines, chars):
    out, n = [], 0
    for i, ln in enumerate(lines):
        if n + len(ln) + 1 > chars - 50:
            out.append(f"  [... {len(lines) - i} line(s) cut to the bound]")
            break
        out.append(ln)
        n += len(ln) + 1
    return "\n".join(out)


# ============================================================================ output bound with priorities
def _strip_series(text, keep_shape):
    """Remove SERIES blocks (rows; SHAPE lines too unless keep_shape) from a G1C3 result."""
    out, skip_indent = [], None
    for ln in text.split("\n"):
        ind = len(ln) - len(ln.lstrip())
        if skip_indent is not None:
            if ln.strip() and ind > skip_indent:
                if keep_shape and ln.strip().startswith("SHAPE"):
                    out.append(ln)
                continue
            skip_indent = None
        if ln.strip().startswith("SERIES ("):
            skip_indent = ind
            if keep_shape:
                out.append(" " * ind + "SHAPES (series rows cut to make room for CHANNELS; evaluate at=all shows them):")
            continue
        out.append(ln)
    return "\n".join(out)


def merge_bounded(base, blocks, limit=MAX_OUT):
    """base (coverage, targets, WHATIF deltas, series) + blocks [(kind, text)] in priority order (CHANNELS before
    EXPLAIN). Series are cut first, then the lowest block, and every cut is stated."""
    blocks = [(k, t) for k, t in blocks if t]
    if not blocks:
        return base
    def total(b, bl):
        return len(b) + sum(len(t) + 1 for _, t in bl)
    if total(base, blocks) <= limit:
        return base + "".join("\n" + t for _, t in blocks)
    for keep in (True, False):
        b2 = _strip_series(base, keep)
        if b2 != base:
            b2 += "" if keep else "\n(series cut to make room for CHANNELS/EXPLAIN; evaluate at=all shows them)"
        if total(b2, blocks) <= limit:
            return b2 + "".join("\n" + t for _, t in blocks)
    base = b2
    notes = {k: f"[{k} cut to the output bound; call {'intervene' if k == 'CHANNELS' else 'explain'} for the rest]"
             for k, _ in blocks}
    reserve = sum(len(n) + 2 for n in notes.values())
    if len(base) + reserve + 10 > limit:                     # WHATIF deltas fill the bound: keep them, cut the base
        base = _cut_lines(base, limit - reserve - 10)
    room = limit - len(base) - reserve - 10
    out = []
    for kind, t in blocks:
        if len(t) + 1 <= room:
            out.append(t)
            room -= len(t) + 1
        else:
            cut = _cut_lines(t, room) if room > 150 else ""
            out.append((cut + "\n" if cut else "") + notes[kind])
            room = max(0, room - len(cut) - 1)
    txt = base + "".join("\n" + t for t in out)
    return txt if len(txt) <= limit else txt[:limit - 40] + "\n[... output cut to the bound]"


# ============================================================================ upsert patching
def line_key(line):
    """The name a document line declares: ('var', x) | ('edge', a, b) | ('whatif', w) | ('mechanism', m) |
    ('explain', e) | ('time',) | ('targets',); None for blank/comment lines; ValueError if unrecognised."""
    code, _ = _split_comment(line)
    s = code.strip()
    if not s:
        return None
    low = s.lower()
    if re.match(r"^targets?\s*:", low):
        return ("targets",)
    if re.match(r"^time\s*:?\s*(steps|dt)\b", low):
        return ("time",)
    m = MECH_RE.match(s)
    if m:
        return ("mechanism", m.group(1))
    m = EXPL_RE.match(s)
    if m:
        return ("explain", m.group(1))
    if low.startswith("edge ") or low.startswith("edge\t"):
        m = EDGE_RE.match(s)
        if not m:
            raise ValueError(f"bad edge line {s!r} (edge a -> b + | - | ? [lag k])")
        return ("edge", m.group(1), m.group(2))
    if low.startswith("whatif"):
        m = re.match(r"^whatif\s+([A-Za-z_]\w*)\s*:", s)
        if not m:
            raise ValueError(f"bad whatif line {s!r} (whatif name: do x = 5)")
        return ("whatif", m.group(1))
    m = VAR_RE.match(s)
    if m:
        return ("var", m.group(2))
    raise ValueError(f"unrecognised line {s!r} (a variable, edge, whatif, mechanism, explain, time or targets line)")


def _key_text(k):
    return k[0] if len(k) == 1 else f"edge {k[1]}->{k[2]}" if k[0] == "edge" else (
        k[1] if k[0] == "var" else f"{k[0]} {k[1]}")


def apply_upsert(source, upsert, delete):
    """(new source, summary) or raise ValueError (nothing applied)."""
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    keys = []
    for ln in lines:
        try:
            keys.append(line_key(ln))
        except ValueError:
            keys.append(None)
    rep, app, dele = 0, 0, 0
    entries, seen = [], set()
    for u in upsert:
        if not isinstance(u, str):
            raise ValueError(f"upsert entries must be strings (document lines), got {type(u).__name__}")
        for ln in u.replace("\r\n", "\n").split("\n"):
            if ln.strip().startswith("```") or ln.strip().startswith("~~~"):
                continue
            k = line_key(ln)
            if k is None:
                continue
            if k in seen:
                raise ValueError(f"two upsert lines declare {_key_text(k)!r}")
            seen.add(k)
            entries.append((k, ln.strip()))
    dkeys = []
    for d in delete:
        if not isinstance(d, str) or not d.strip():
            raise ValueError("delete entries must be names, e.g. \"x\", \"whatif w\", \"edge a->b\", \"mechanism m\"")
        s = d.strip()
        m = re.fullmatch(r"edge\s+([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)", s)
        m2 = re.fullmatch(r"(whatif|mechanism|explain)\s+([A-Za-z_]\w*)", s)
        if m:
            k = ("edge", m.group(1), m.group(2))
        elif m2:
            k = (m2.group(1), m2.group(2))
        elif s in ("time", "targets"):
            k = (s,)
        elif NAME_RE.fullmatch(s):
            cands = [x for x in set(keys) if x and x[-1] == s and x[0] in ("var", "whatif", "mechanism", "explain")]
            if ("var", s) in cands:
                k = ("var", s)
            elif len(cands) == 1:
                k = cands[0]
            elif not cands:
                raise ValueError(f"delete: no line declares {s!r}"
                                 + _did_you_mean(s, [x[-1] for x in keys if x and len(x) == 2]))
            else:
                raise ValueError(f"delete: {s!r} is ambiguous ({', '.join(_key_text(x) for x in cands)}); "
                                 "write 'whatif name', 'mechanism name' or 'explain name'")
        else:
            raise ValueError(f"delete: bad name {s!r}")
        if k not in keys:
            raise ValueError(f"delete: no line declares {_key_text(k)!r}")
        dkeys.append(k)
    for k, ln in entries:
        if k in dkeys:
            raise ValueError(f"{_key_text(k)!r} is both upserted and deleted")
    for k in dkeys:
        for i, kk in enumerate(keys):
            if kk == k:
                lines[i], keys[i] = None, None
                dele += 1
    for k, ln in entries:
        if k in keys:
            i = keys.index(k)
            lines[i] = ln
            rep += 1
        else:
            lines.append(ln)
            keys.append(k)
            app += 1
    new = "\n".join(x for x in lines if x is not None)
    return new, f"upsert: {rep} replaced, {app} appended; delete: {dele}"


# ============================================================================ the arm
_DO_DESC = ("do = {var: value | {value|scale|add: x, from?: step} | '+'|'-' (qualitative)}.")


class G3C2State(G1C3State):
    pass


class G3C2Arm(G1C3Arm):
    name = "G3C2"
    buildable = True
    executable = True
    guide_file = "g3c2_guide.md"
    mutating_tools = MUTATING
    custom_episode = True

    def tools(self):
        return [
            _tool("submit_graph", "Compile the whole graph document and replace the representation. Takes `source` "
                  "(document text) or `graph` (A2 JSON); with neither, the last ```graph block of this response. "
                  "Returns the coverage report, the baseline of the targets, every stored whatif with light CHANNELS, "
                  "and every stored explain.",
                  {"source": {"type": "string"}, "graph": {"type": "object"}}),
            _tool("patch_graph", "Edit the document, then recompile and rerun as submit_graph. Either `upsert`: whole "
                  "document lines, each replacing the line that declares the same name (variable, edge a->b, whatif, "
                  "mechanism, explain, time, targets:) or appended if new; and/or `delete`: names to remove "
                  "(\"x\", \"whatif w\", \"edge a->b\"). All entries apply in one call, or none. Or `old`/`new`: "
                  "exact unique text replace.",
                  {"upsert": {"type": "array", "items": {"type": "string"}},
                   "delete": {"type": "array", "items": {"type": "string"}},
                   "old": {"type": "string"}, "new": {"type": "string"}}),
            _tool("evaluate", "Compute variable values (static, over time, or Monte Carlo).", {"query": _QUERY}),
            _tool("intervene", "Compare interventions with the baseline (paired), with full CHANNELS (the share of "
                  "each mechanism in the effect). Either `do` + `query`, or `whatifs`: up to 6 of {name?, do, query?} "
                  "or names of stored whatif lines, run on one shared baseline. " + _DO_DESC
                  + " channels: false skips CHANNELS; at: 'final' | step | 'peak' is the CHANNELS step.",
                  {"do": {"type": "object"}, "query": _QUERY,
                   "whatifs": {"type": "array", "items": {}, "description": "what-ifs or stored whatif names"},
                   "channels": {"type": "boolean"}, "at": {"description": "'final' | step | 'peak'"}}),
            _tool("explain", "Actual causes of an event: for each candidate (with its contrast = its normal or working "
                  "value), whether it is a but-for cause, a cause under a contingency, a joint cause or not a cause, "
                  "its mechanism, and which observed events it accounts for. Either `name` of a stored explain line, "
                  "or `event` + `candidates` {name: contrast} [+ `observed` [events], `at` final|any|step, `samples`].",
                  {"name": {"type": "string"}, "event": {"type": "string"},
                   "candidates": {"type": "object"}, "observed": {"type": "array", "items": {"type": "string"}},
                   "at": {"description": "'final' | 'any' | step"}, "samples": {"type": "integer"}}),
            _tool("show", "Show the document with line numbers, mechanisms (declared and automatic), stored explains, "
                  "the compiled graph and its holes.", {}),
        ]

    def probe_tools(self):
        return [t for t in self.tools() if t["function"]["name"] in ("evaluate", "intervene", "show")]

    # ---------------------------------------------------------------- state
    def new_state(self):
        st = G3C2State()
        self._set_doc(st, G3Doc(graph={"variables": [], "edges": [], "equations": {}}))
        return st

    def load(self, rep, given_gold=False):
        st = G3C2State(given_gold=given_gold)
        if isinstance(rep, dict) and isinstance(rep.get("source"), str):
            self._set_doc(st, G3Doc(source=rep["source"]))
        elif isinstance(rep, dict):
            self._set_doc(st, G3Doc(graph=rep))
        else:
            self._set_doc(st, G3Doc(graph={"variables": [], "edges": [], "equations": {}}))
            st.log.append(("load", None, f"representation was {type(rep).__name__}, not an object; started empty"))
        return st

    def _doc(self, state):
        if getattr(state, "doc", None) is None or not isinstance(state.doc, G3Doc):
            d = state.rep if isinstance(state.rep, dict) else {}
            state.doc = G3Doc(source=d["source"]) if isinstance(d.get("source"), str) else G3Doc(graph=d)
        return state.doc

    def handle(self, state, name, args):
        if state.given_gold and name in MUTATING:
            out = "ERROR: the representation is read-only in this mode (hint: use evaluate, intervene, explain, show)"
            state.log.append((name, args, out[:200]))
            return out
        return G1C3Arm.handle(self, state, name, args)

    # ---------------------------------------------------------------- building
    def _t_submit_graph(self, state, args):
        src = args.get("source")
        g = _maybe_json(args.get("graph"))
        if isinstance(src, str) and src.strip():
            inner = fenced_block(src)
            doc = G3Doc(source=inner if inner is not None else src)
            how = "source argument"
        elif isinstance(g, dict):
            doc = G3Doc(graph=g)
            how = "graph argument (A2 JSON)"
        elif getattr(state, "pending_block", None):
            doc = G3Doc(source=state.pending_block)
            how = "```graph block"
        else:
            return ("ERROR: no document (hint: put the whole graph in ONE fenced block labelled graph in this response, "
                    "or pass source=<document text>)")
        self._set_doc(state, doc)
        return self._report(doc, f"graph submitted from the {how}")

    def _t_patch_graph(self, state, args):
        doc = self._doc(state)
        if doc.source is None:
            return "ERROR: no document to patch (hint: submit_graph a document first)"
        old, new = args.get("old"), args.get("new")
        ups, dels = _maybe_json(args.get("upsert")), _maybe_json(args.get("delete"))
        if isinstance(ups, str):
            ups = [ups]
        if isinstance(dels, str):
            dels = [x.strip() for x in dels.split(",")]
        if ups is None and dels is None and old is None:
            return ("ERROR: patch_graph needs `upsert` (whole lines) and/or `delete` (names), or `old` + `new` "
                    "(hint: upsert=[\"price : real EUR = 12   # stated ...\"])")
        if (ups is not None and not isinstance(ups, list)) or (dels is not None and not isinstance(dels, list)):
            return "ERROR: upsert and delete must be lists of strings (hint: upsert=[\"x : real = 1\"], delete=[\"y\"])"
        src, notes = doc.source, []
        if old is not None:
            if not isinstance(old, str) or not old or not isinstance(new, str):
                return "ERROR: old/new: 'old' must be non-empty text and 'new' text (hint: copy 'old' exactly from show)"
            k = src.count(old)
            if k == 0:
                near = difflib.get_close_matches(old.strip().split("\n")[0], [x.strip() for x in src.split("\n")],
                                                 n=1, cutoff=0.4)
                return ("ERROR: 'old' text not found in the document" + (f" (closest line: {near[0]!r})" if near else "")
                        + " (hint: copy it exactly, or use upsert with the whole line)")
            if k > 1:
                return f"ERROR: 'old' text occurs {k} times (hint: use upsert with the whole line instead)"
            src = src.replace(old, new, 1)
            notes.append("old/new replaced")
        if ups is not None or dels is not None:
            try:
                src, summ = apply_upsert(src, ups or [], dels or [])
            except ValueError as e:
                return f"ERROR: {e}; nothing was applied (hint: send each line whole; delete takes names)"
            notes.append(summ)
        nd = G3Doc(source=src)
        self._set_doc(state, nd)
        return self._report(nd, "patch applied (" + "; ".join(notes) + ")")

    def _report(self, doc, head):
        base = G1C3Arm._report(self, doc, head)
        # mechanism / explain section right after the head line (coverage first)
        sec = []
        if doc.mechanisms:
            sec.append("MECHANISMS: " + "; ".join(f"{m['name']} ({', '.join(m['vars'])})" for m in doc.mechanisms))
        if doc.explains:
            sec.append("EXPLAINS: " + ", ".join(x["name"] for x in doc.explains))
        if doc.notes:
            sec.append(f"MECHANISM/EXPLAIN NOTES ({len(doc.notes)}; notes only, nothing is rejected):\n"
                       + "\n".join(f"  - {n}" for n in doc.notes[:10])
                       + (f"\n  ... {len(doc.notes) - 10} more" if len(doc.notes) > 10 else ""))
        if sec:
            h, _, rest = base.partition("\n")
            base = h + "\n" + "\n".join(sec) + ("\n" + rest if rest else "")
        cg = doc.cg
        blocks = []
        if cg.vars and doc.whatifs:
            specs = self._specs(cg, doc, [{"name": w["name"]} for w in doc.whatifs[:MAX_BATCH]], None)
            blocks.append(("CHANNELS", self._channels(cg, doc, specs, LIGHT_BUDGET_S, light=True)))
        if cg.vars and doc.explains:
            t_end = time.perf_counter() + SUBMIT_EXPLAIN_BUDGET_S
            ex = []
            for x in doc.explains[:3]:
                left = max(0.05, t_end - time.perf_counter())
                ex.append(self._explain_spec(doc, x, left))
            if len(doc.explains) > 3:
                ex.append(f"(only the first 3 stored explains were run; run others with explain name=...)")
            blocks.append(("EXPLAIN", "\n".join(ex)))
        return merge_bounded(base, blocks)

    # ---------------------------------------------------------------- what-if specs and CHANNELS
    def _norm_do(self, cg, do_raw):
        """G1C3's do normalisation: (do, qualitative, errors)."""
        do, qual, errs = {}, {}, []
        for var, val in (do_raw or {}).items():
            if var not in cg.vars:
                errs.append(var)
                continue
            typ = cg.vars[var]["type"]
            val = _maybe_json(val)
            if isinstance(val, str) and typ != "category" and val.strip().lower() in QUAL_WORDS:
                qual[var] = QUAL_WORDS[val.strip().lower()]
                continue
            if isinstance(val, dict):
                iv = {k: val[k] for k in ("value", "scale", "add") if k in val}
                if not iv:
                    continue
                if "value" in iv:
                    c, err = _coerce(iv["value"], typ)
                    if err:
                        continue
                    iv["value"] = c
                if any(k in iv and (not isinstance(iv[k], (int, float)) or isinstance(iv[k], bool))
                       for k in ("scale", "add")):
                    continue
                if val.get("from") is not None:
                    iv["from"] = int(val["from"])
                do[var] = iv
            else:
                c, err = _coerce(val, typ)
                if err:
                    continue
                do[var] = {"value": c}
        return do, qual, errs

    def _specs(self, cg, doc, wis, q_default):
        """Resolved what-ifs [(name, do, q)] (as G1C3's _batch resolves them); unresolvable ones are skipped (the
        WHATIF section already reports them)."""
        out = []
        for w in wis:
            r = self._resolve(cg, doc, w, q_default)
            if isinstance(r, str):
                continue
            name, do_raw, qraw = r
            if isinstance(qraw, dict) and not qraw.get("targets") and not qraw.get("target"):
                qraw = dict(qraw, targets=None)
            try:
                q = self._parse_query(cg, qraw)
            except (ValueError, TypeError):
                continue
            do, _, _ = self._norm_do(cg, _maybe_json(do_raw) if not isinstance(do_raw, dict) else do_raw)
            if do:
                out.append((name, do, q))
        return out

    def _channels(self, cg, doc, specs, budget_s, light=False, at=None):
        """CHANNELS text for resolved what-ifs, sharing one work budget."""
        if not specs:
            return ""
        t_end = time.perf_counter() + budget_s
        ch = _children(cg)
        out = []
        for j, (name, do, q) in enumerate(specs):
            share_end = time.perf_counter() + max(0.0, t_end - time.perf_counter()) / (len(specs) - j)
            targets = [t for t in q["targets"] if t in cg.vars and t not in do
                       and cg.vars[t]["type"] != "category"]
            if not targets:
                continue
            q_at = at if at is not None else q.get("at", "final")
            if q_at not in ("final", "peak") and not (isinstance(q_at, int) and not isinstance(q_at, bool)):
                q_at = "final"
            n_max = min(q["samples"], MEMBERS_MAX) if q["samples"] else 0
            W = Worlds(cg, q["steps"], do, 1 if n_max else 0, q["seed"], share_end + 3600)
            # plan: channels per target, then the member count that fits the budget
            plans = []
            for y in targets:
                chans, notes = channels_for(cg, doc.mechanisms, list(do), y, ch)
                plans.append([y, chans, notes])
            need = sum((2 * len(c) + 2) if light else 2 ** min(len(c), MAX_CHANNELS) for _, c, _ in plans)
            need = max(1, need) + sum(len(c) + 1 for _, c, _ in plans if len(c) > MAX_CHANNELS)
            left = max(0.0, share_end - time.perf_counter())
            if n_max:
                fit = int(left / max(1e-6, need * W.t_run))
                n_d = max(min(n_max, MEMBERS_MIN), min(n_max, fit))
                W = Worlds(cg, q["steps"], do, n_d, q["seed"], share_end)
            else:
                W.deadline = share_end
            for ti, (y, chans, notes) in enumerate(plans):
                label = name
                try:
                    if len(chans) > MAX_CHANNELS:           # keep the 5 largest |but-for|, merge the rest
                        chans, n2 = self._merge_tail(W, y, chans, q_at)
                        notes = notes + [n2]
                except OutOfTime:
                    out.append(f"CHANNELS [{label}] {y}: not computed (time)")
                    continue
                k = len(chans)
                tl = max(0.0, W.deadline - time.perf_counter()) / (len(plans) - ti)
                cost = W.t_run * W.members
                if light:
                    mode = "light"
                elif 2 ** k * cost <= tl:
                    mode = "exact"
                elif (4 * k + 2) * cost <= tl:
                    mode = "perm"
                else:
                    mode = "light"                          # minimal: alone and but-for only
                W.deadline = min(share_end, time.perf_counter() + tl)
                res = decompose(W, y, chans, mode, q_at)
                W.deadline = share_end
                if res["mode"] == "light" and not light:
                    res["mode"] = "minimal"
                res["notes"] = notes + res["notes"]
                if n_max and W.members < n_max:
                    res["notes"].append(f"{W.members} of {q['samples']} members used (time budget)")
                elif n_max and q["samples"] > W.members:
                    res["notes"].append(f"first {W.members} of the {q['samples']} WHATIF samples")
                out.append(channels_text(res, label))
        return "\n".join(out)

    def _merge_tail(self, W, y, chans, at):
        """k > 6: rank channels by |but-for| (k + 1 worlds), keep the 5 largest, merge the rest into `other`."""
        allv = frozenset(v for c in chans for v in c["vars"])
        full = [_num(_pick_step(h[y], at if isinstance(at, int) else "final")) for h in W.world(frozenset())]
        bf = []
        for c in chans:
            hs = W.world(frozenset(c["vars"]))
            xs = [_num(_pick_step(h[y], at if isinstance(at, int) else "final")) for h in hs]
            d = [a - b for a, b in zip(full, xs) if a is not None and b is not None]
            bf.append(abs(sum(d) / len(d)) if d else 0.0)
        order = sorted(range(len(chans)), key=lambda j: -bf[j])
        keep = [chans[j] for j in order[:MAX_CHANNELS - 1]]
        rest = [chans[j] for j in order[MAX_CHANNELS - 1:]]
        other = {"name": "other", "vars": [v for c in rest for v in c["vars"]], "declared": False}
        return keep + [other], (f"{len(chans)} channels: the {MAX_CHANNELS - 1} largest |but-for| kept, "
                                f"{len(rest)} merged into other ({', '.join(c['name'] for c in rest[:6])})")

    def _t_intervene(self, state, args):
        base = G1C3Arm._t_intervene(self, state, args)
        if base.startswith("ERROR") or args.get("channels") in (False, "false", 0):
            return base
        doc = self._doc(state)
        cg = doc.cg
        wis = _maybe_json(args.get("whatifs"))
        if wis is None and args.get("names") is not None:
            wis = _maybe_json(args.get("names"))
        if isinstance(wis, (str, dict)):
            wis = [wis]
        if wis is None:
            wis = [{"name": "do", "do": _maybe_json(args.get("do")), "query": args.get("query")}]
        if not isinstance(wis, list):
            return base
        at = args.get("at")
        if isinstance(at, str) and at.strip().isdigit():
            at = int(at)
        specs = self._specs(cg, doc, wis[:MAX_BATCH], args.get("query"))
        return merge_bounded(base, [("CHANNELS", self._channels(cg, doc, specs, INTERVENE_BUDGET_S, at=at))])

    # ---------------------------------------------------------------- explain
    def _explain_spec(self, doc, x, budget_s):
        cg = doc.cg
        has_dist = any("distribution" in s for s in cg.vars.values())
        samples = x.get("samples")
        n = min(MEMBERS_MAX, samples if samples is not None else (MEMBERS_MAX if has_dist else 0))
        n = max(0, int(n)) if has_dist else 0
        steps = cg.steps if cg.timed else 0
        return explain_run(doc, x, n, 0, steps, budget_s)

    def _t_explain(self, state, args):
        doc = self._doc(state)
        cg = doc.cg
        if not cg.vars:
            return "ERROR: no model to explain (hint: submit_graph first)"
        nm = args.get("name")
        stored = {x["name"]: x for x in doc.explains}
        if nm and not args.get("event"):
            if nm not in stored:
                return (f"ERROR: explain '{nm}' is not stored{_did_you_mean(str(nm), stored)} (stored: "
                        f"{', '.join(stored) or 'none'}; hint: or pass event + candidates)")
            x = dict(stored[nm])
            for k in ("at", "samples"):
                if args.get(k) is not None:
                    x[k] = args[k]
            return _cut_lines(self._explain_spec(doc, x, EXPLAIN_BUDGET_S), MAX_OUT)
        ev = args.get("event")
        cands = _maybe_json(args.get("candidates"))
        if not isinstance(ev, str) or not ev.strip():
            return "ERROR: explain needs `event` (an expression, e.g. \"revenue < 500\") or the `name` of a stored explain"
        if isinstance(cands, list):
            try:
                cands = {c.split("=")[0].strip(): c.split("=", 1)[1].strip() for c in cands}
            except (AttributeError, IndexError):
                cands = None
        if not isinstance(cands, dict) or not cands:
            return "ERROR: explain needs `candidates` {name: contrast} (hint: {\"breaker_ok\": true, \"price\": 12})"
        try:
            _expr.parse(_split_event_at(ev)[0])
        except _expr.ExprError as e:
            return f"ERROR: bad event {ev!r}: {e} (hint: an expression over variables, e.g. temp < 5)"
        clist = []
        for c, v in cands.items():
            if isinstance(v, str):
                vs = v.strip()
                try:
                    v = _const(vs, "category" if NAME_RE.fullmatch(vs) and vs not in ("true", "false") else "real")
                except ValueError:
                    pass
            clist.append([c, v])
        obs = _maybe_json(args.get("observed")) or []
        if isinstance(obs, str):
            obs = [o.strip() for o in _split_top(obs, ",") if o.strip()]
        at = args.get("at")
        if isinstance(at, str) and at.strip().isdigit():
            at = int(at)
        x = {"name": nm or "explain", "event": ev, "candidates": clist, "observed": list(obs), "at": at,
             "samples": args.get("samples")}
        return _cut_lines(self._explain_spec(doc, x, EXPLAIN_BUDGET_S), MAX_OUT)

    # ---------------------------------------------------------------- show / render / validity / export
    def render(self, state):
        doc = self._doc(state)
        r = G1C3Arm.render(self, state)
        extra = []
        if doc.mechanisms:
            extra.append("MECHANISMS (declared): " + "; ".join(f"{m['name']} ({', '.join(m['vars'])})"
                                                              for m in doc.mechanisms))
        if doc.explains:
            extra.append("EXPLAINS (stored): " + "; ".join(
                f"{x['name']}: {x['event']} | candidates " + ", ".join(f"{c}={_fmt(v)}" for c, v in x["candidates"])
                for x in doc.explains))
        cg = doc.cg
        if cg.vars and doc.whatifs:
            ch = _children(cg)
            auto = []
            for w in doc.whatifs[:MAX_BATCH]:
                xs = [v for v in w.get("do", {}) if v in cg.vars]
                for y in (w.get("targets") or doc.targets)[:4]:
                    if y not in cg.vars or y in xs:
                        continue
                    chans, _ = channels_for(cg, doc.mechanisms, xs, y, ch)
                    if chans:
                        auto.append(f"  [{w['name']}] {y}: " + "; ".join(
                            f"{c['name']}{'' if c['declared'] else '*'} ({', '.join(c['vars'][:5])}"
                            + (f" +{len(c['vars']) - 5}" if len(c['vars']) > 5 else "") + ")" for c in chans))
            if auto:
                extra.append("CHANNELS per stored whatif and target (* = automatic, by path equivalence):")
                extra += auto[:24]
        if doc.notes:
            extra.append("MECHANISM/EXPLAIN NOTES:")
            extra += [f"  - {n}" for n in doc.notes[:20]]
        if not extra:
            return r
        add = "\n".join(extra)
        i = r.find("\nCOMPILED GRAPH")
        return (r[:i] + "\n" + add + r[i:]) if i >= 0 else (r + "\n" + add)

    def validity(self, state):
        v = G1C3Arm.validity(self, state)
        doc = self._doc(state)
        v["notes"] = list(doc.notes)
        v["mechanisms"] = [m["name"] for m in doc.mechanisms]
        v["explains"] = [x["name"] for x in doc.explains]
        return v


G3C2 = G3C2Arm
