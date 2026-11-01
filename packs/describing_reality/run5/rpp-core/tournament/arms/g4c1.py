"""G4C1: counterfactual channel model with a query interface that reaches the layer (G3C2 modified).

Spec: runs/discovery/gen4/specs/G4C1.md. The representation, the document syntax and every computed functional
(baseline, WHATIF, CHANNELS, EXPLAIN) are G3C2's and are imported unchanged: `g3c2.py`, `g1c3.py`, `a2_causal.py`
and `_expr.py` are not edited. Every change is an override in `G4C1Arm(G3C2Arm)`:

  * §3.1 query resolution honours the call. Each what-if entry's effective query is merged field by field (entry
    `query` > call `query` > stored what-if fields > the document's `targets:`), and its effective `at` is the
    entry's `at`/`query.at` > the call's top-level `at` > the call's `query.at` > the stored `at` > final. WHATIF
    numbers and CHANNELS are both read at it (`peak`: CHANNELS only, WHATIF at final; `all`: CHANNELS at final).
    Every WHATIF header shows the effective query. An exact repeat of an `intervene` entry or an `explain` call in
    the same episode (same document hash) returns a one-line REPEAT reference instead of the numbers.
  * §3.2 bounded output with the totals first: each what-if block is assembled in tiers (1 header, targets and
    events; 2 one compact CHANNELS line per target; 3 CHANNELS detail; 4 QUALITATIVE; 5 SERIES) and the bound
    (7,800 characters, unchanged) cuts the lowest tier first, across the whole result. Cut notes name the narrower
    call that shows what was cut. EXPLAIN blocks keep verdicts before the observation table before mechanism lines.
  * §3.3 per-name patching (`apply_upsert_per_name`): each upsert/delete entry applies on its own; the result lists
    what was and was not applied. G3C2's all-or-nothing `apply_upsert` is untouched.
  * §3.4 full CHANNELS at submit and patch for every stored what-if (≤ MAX_BATCH), sharing
    SUBMIT_CHANNELS_BUDGET_S, with a per-episode cache keyed by the what-if's dependence (ancestors of its targets
    plus its do variables) so an unrelated patch prints "(CHANNELS unchanged since the last submit)".
  * §3.5 an empty submit_graph (no `source`, no `graph`, no ```graph block in the response) returns a skeleton the
    first time in an episode and one short line afterwards.

The arm runs in process and launches no subprocess; its run-time imports are declared in RUNTIME_DEPS (D-060).

Build notes (details the spec left to Build):
  * Time budgets (§3.4): the runner (tournament/episode.py `execute_tool_call`; tournament/probes.py through
    `tool_loop`) calls `arm.handle` synchronously in process and sets no per-tool-call time limit, so
    SUBMIT_CHANNELS_BUDGET_S = 15 s and SUBMIT_EXPLAIN_BUDGET_S = 6 s stand unlowered (at most about 21 s of
    counterfactual work per submit or patch, close to G3C2's 20 s intervene budget).
  * EXPLAIN runs through G3C2's `explain_run` unchanged; only its final 2,000-character cut is replaced (the
    function's code is re-bound to a copy of g3c2's globals whose `_cut_block` keeps every line), so G4C1 computes
    exactly G3C2's lines and then cuts them by priority to the same 2,000-character block bound.
  * Cut notes: a what-if with several targets gets the narrowing call by target (spec); a single-target what-if in
    a call with several what-ifs gets the call for that what-if alone; otherwise the one-target note.
  * REPEAT ordinals (#k) count the tool calls that reach the arm in the episode, the implicit submit of a fenced
    block included. Probes load a fresh state per probe, so each probe starts with an empty repeat map, and a
    REPEAT line does not start with ERROR (probes.py counts any non-ERROR result as a successful run).
  * Golds (§5): tournament/gold/g4c1/<ID>.json started as byte-identical copies of the audited
    tournament/gold/g1c3/<ID>.json; where the gold builder added `mechanism` / `explain` lines they are appended at
    the end of the document and every original line is unchanged (see each <ID>.notes.md).
  * The effort target (≤ 1.3x A0+ output tokens) cannot be measured with fakes; it is measured at evaluation.
"""
import copy
import hashlib
import json
import re
import time
import types

from tournament.arms import _expr
from tournament.arms import g3c2 as G3
from tournament.arms.a2_causal import _holes_text, _maybe_json
from tournament.arms.g1c3 import (MAX_BATCH, MAX_OUT, NAME_RE, G1C3Arm, _bound, _cap_samples, _const, _cut_lines,
                                  _did_you_mean, _split_comment, _split_top, _tool, _QUERY, fenced_block)
from tournament.arms.g3c2 import (CAVEAT, EXPLAIN_CHARS, MAX_CHANNELS, MEMBERS_MAX, MEMBERS_MIN, G3C2Arm, G3C2State,
                                  G3Doc, OutOfTime, Worlds, _children, _flags, _key_text, _reach, _sg, _split_event_at,
                                  channels_for, channels_text, decompose, line_key)

RUNTIME_DEPS = ["tournament/arms/_expr.py", "tournament/arms/g1c3.py", "tournament/arms/a2_causal.py",
                "tournament/arms/g3c2.py"]

MUTATING = ("submit_graph", "patch_graph")
SUBMIT_CHANNELS_BUDGET_S = 15.0   # full CHANNELS work per submit/patch (shared by the stored what-ifs)
SUBMIT_EXPLAIN_BUDGET_S = 6.0     # stored explains at submit/patch (G3C2's value)
INTERVENE_BUDGET_S = 20.0         # CHANNELS work per intervene call (G3C2's value)
EXPLAIN_BUDGET_S = 20.0           # EXPLAIN work per explain call (G3C2's value)
COMPACT_TOP = 3                   # channels named on the compact CHANNELS line
MAX_EXPLAINS_AT_SUBMIT = 3
UNCHANGED = "(CHANNELS unchanged since the last submit)"
REPEAT_TAIL = "Nothing new to compute. For more detail, ask for fewer targets, another step (at), or other what-ifs."
EXPLAIN_REPEAT_TAIL = ("Nothing new to compute. For another view, change at or samples, the candidates or the "
                       "observations.")
EQUAL_SHARE_NOTE = ("[the what-ifs' totals alone exceed the output bound: each what-if got an equal share; ask for "
                    "fewer what-ifs or fewer targets]")
_MARK = "\x00G4C1-WHATIFS\x00"
AT_WORDS = ("final", "all", "peak")

SKELETON = """```graph
time steps=<N> dt=1                     # unit: <day|hour|...>
targets: <every quantity the question asks about>
<name> : real <unit> = <number>         # stated "<short quote>"
<name> : real <unit> := <expression>    # derived from ...
<stock> : real <unit> = <initial> := <stock>[t-1] + <inflow> - <outflow>
mechanism <process>: <var>, <var>
whatif <name>: do <var> = <value> ; targets <var>, <var>
```"""
EMPTY_FIRST = ("ERROR: this submit_graph call was empty: no `source` argument and no ```graph block in your response. "
               "Nothing was stored.\nWrite the WHOLE document in ONE fenced block labelled graph in your response "
               "text, then call submit_graph with no\narguments (or pass the text as `source`). Skeleton:\n" + SKELETON)
EMPTY_AGAIN = "ERROR: empty submit_graph again (no source, no graph block); see the skeleton in the earlier result."


# ============================================================================ per-name patching (§3.3)
def _short(x, n=60):
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[:n - 3] + "..."


def _delete_key(s, keys):
    """A delete entry -> (key, None, how) or (None, reason, None); how is 'name' or 'line'."""
    m = re.fullmatch(r"edge\s+([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)", s)
    m2 = re.fullmatch(r"(whatif|mechanism|explain)\s+([A-Za-z_]\w*)", s)
    how = "name"
    if m:
        k = ("edge", m.group(1), m.group(2))
    elif m2:
        k = (m2.group(1), m2.group(2))
    elif s in ("time", "targets"):
        k = (s,)
    elif NAME_RE.fullmatch(s):
        cands = sorted({x for x in keys if x and x[-1] == s and x[0] in ("var", "whatif", "mechanism", "explain")})
        if ("var", s) in cands:
            k = ("var", s)
        elif len(cands) == 1:
            k = cands[0]
        elif not cands:
            return None, (f"delete: no line declares {s!r}"
                          + _did_you_mean(s, [x[-1] for x in keys if x and len(x) == 2])), None
        else:
            return None, (f"delete: {s!r} is ambiguous ({', '.join(_key_text(x) for x in cands)}); "
                          "write 'whatif name', 'mechanism name' or 'explain name'"), None
    else:
        try:
            k = line_key(s)
        except ValueError:
            k = None
        if k is None:
            return None, (f"delete: bad name {_short(s)!r} (forms: \"x\", \"whatif w\", \"edge a->b\", "
                          "\"mechanism m\", or a whole document line)"), None
        how = "line"
    if k not in keys:
        names = [x[-1] for x in keys if x and len(x) == 2]
        return None, f"delete: no line declares {_key_text(k)!r}" + _did_you_mean(str(k[-1]), names), None
    return k, None, how


def apply_upsert_per_name(source, upsert, delete):
    """Spec §3.3. Returns (new source, applied, not_applied, notes) where applied = {"replaced", "appended",
    "deleted"} lists of names and not_applied lists each refused entry with its reason. Never raises."""
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    keys = []
    for ln in lines:
        try:
            keys.append(line_key(ln))
        except ValueError:
            keys.append(None)
    not_applied, notes = [], []
    ups = {}                                                  # key -> line; the last upsert of a name wins
    for u in upsert:
        if not isinstance(u, str):
            not_applied.append(f"{_short(u)} not a document line (upsert entries are strings)")
            continue
        for ln in u.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if ln.strip().startswith("```") or ln.strip().startswith("~~~"):
                continue
            try:
                k = line_key(ln)
            except ValueError as e:
                code = _split_comment(ln)[0].strip()
                reason = str(e).replace(" " + repr(code), "", 1)
                not_applied.append(f'"{_short(code)}" {reason}')
                continue
            if k is None:
                continue
            if k in ups:
                notes.append(f"earlier upsert of {_key_text(k)} superseded in the same call")
                ups.pop(k)
            ups[k] = ln.strip()
    dels = {}
    for d in delete:
        if not isinstance(d, str) or not d.strip():
            not_applied.append(f"delete {_short(d) or 'empty'}: entries must be names, e.g. \"x\", \"whatif w\", "
                               "\"edge a->b\", \"mechanism m\"")
            continue
        k, err, how = _delete_key(d.strip(), keys)
        if err:
            not_applied.append(err)
            continue
        if how == "line":
            notes.append(f"delete given a whole line; resolved to {_key_text(k)}")
        dels[k] = d.strip()
    for k in [k for k in ups if k in dels]:
        not_applied.append(f"{_key_text(k)} is both upserted and deleted; send it in only one list")
        ups.pop(k)
        dels.pop(k)
    applied = {"replaced": [], "appended": [], "deleted": []}
    for k in dels:
        hit = False
        for i, kk in enumerate(keys):
            if kk == k:
                lines[i], keys[i] = None, None
                hit = True
        if hit:
            applied["deleted"].append(_key_text(k))
    for k, ln in ups.items():
        if k in keys:
            lines[keys.index(k)] = ln
            applied["replaced"].append(_key_text(k))
        else:
            lines.append(ln)
            keys.append(k)
            applied["appended"].append(_key_text(k))
    return "\n".join(x for x in lines if x is not None), applied, not_applied, notes


def _not_applied_text(not_applied, cap=12):
    shown = not_applied[:cap]
    return (f"NOT applied {len(not_applied)}: " + "; ".join(shown)
            + (f"; ... {len(not_applied) - cap} more" if len(not_applied) > cap else ""))


# ============================================================================ query resolution (§3.1)
def _qdict(q):
    """A query argument -> dict of the fields it sets ({} when absent); ValueError if it is not a query."""
    q = _maybe_json(q)
    if q is None:
        return {}
    if isinstance(q, str):
        tg = [x.strip() for x in q.split(",") if x.strip()]
        return {"targets": tg} if tg else {}
    if isinstance(q, list):
        return {"targets": list(q)} if q else {}
    if not isinstance(q, dict):
        raise ValueError("query must be a name, a list of names or an object")
    d = {k: v for k, v in q.items() if v is not None}
    tg = d.pop("targets", None)
    t1 = d.pop("target", None)
    tg = tg if tg not in (None, [], "") else t1
    if isinstance(tg, str):
        tg = [x.strip() for x in tg.split(",") if x.strip()]
    if tg:
        d["targets"] = list(tg)
    ev = d.pop("events", None)
    if isinstance(ev, str):
        ev = [ev]
    if ev:
        d["events"] = list(ev)
    return d


def _norm_at(at):
    """-> (at, note): an int step, 'final', 'all' or 'peak'."""
    bad = (f"at {at!r} not understood (final | all | peak | step); final used")
    if isinstance(at, bool):
        return "final", bad
    if isinstance(at, int):
        return (at, None) if at >= 0 else ("final", bad)
    if isinstance(at, float) and at >= 0 and at == int(at):
        return int(at), None
    if isinstance(at, str):
        s = at.strip().lower()
        if s.isdigit():
            return int(s), None
        if s in AT_WORDS:
            return s, None
        m = re.fullmatch(r"(?:step|t)\s*[=:]?\s*(\d+)", s)
        if m:
            return int(m.group(1)), None
    return "final", bad


def _at_label(at):
    return f"step {at}" if isinstance(at, int) else str(at)


def _eff_text(cg, q, at):
    """The effective query of a WHATIF header: [targets ...; events ...; at ...; samples]."""
    tg = list(q["targets"])
    parts = ["targets " + (", ".join(tg[:8]) + (f" +{len(tg) - 8} more" if len(tg) > 8 else "") if tg else "none")]
    if q["events"]:
        parts.append("events " + ", ".join(q["events"][:4]) + (f" +{len(q['events']) - 4}" if len(q["events"]) > 4
                                                                 else ""))
    if isinstance(at, int):
        parts.append(f"at step {at}")
    elif cg.timed:
        parts.append(f"at final step {q['steps']}" + (" (peak: CHANNELS only)" if at == "peak" else
                                                       " (all: series below)" if at == "all" else ""))
    else:
        parts.append("static")
    parts.append(f"{q['samples']} paired samples, seed {q['seed']}" if q["samples"] else "deterministic")
    return "[" + "; ".join(parts) + "]"


def doc_hash(doc):
    h = getattr(doc, "_g4_hash", None)
    if h is None:
        s = doc.source if doc.source is not None else json.dumps(doc.export(), sort_keys=True, default=str)
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
        doc._g4_hash = h
    return h


def dep_signature(doc, do_vars, targets):
    """What a what-if's numbers depend on: the definitions of the targets' ancestors and the do variables, the
    stochastic inputs in sampling order, the declared mechanisms over those variables, and the time line."""
    cg = doc.cg
    par = {n: set() for n in cg.vars}
    for v, ast in cg.eqs.items():
        for r, _ in _expr.refs(ast):
            if r in cg.vars and r != v:
                par[v].add(r)
    need = _reach([t for t in targets if t in cg.vars] + [x for x in do_vars if x in cg.vars], par)
    eqsrc = doc.graph.get("equations") or {}
    items = []
    for n in sorted(need):
        spec = {k: v for k, v in cg.vars[n].items() if k != "provenance"}
        items.append((n, repr(sorted(spec.items(), key=lambda kv: kv[0])), str(eqsrc.get(n)), str(cg.broken.get(n))))
    dists = [(n, repr(s.get("distribution"))) for n, s in cg.vars.items() if s.get("exogenous") and "distribution" in s]
    mech = [(m["name"], tuple(v for v in m["vars"] if v in need)) for m in doc.mechanisms]
    mech = [m for m in mech if m[1]]
    return hashlib.sha1(repr((items, dists, mech, cg.steps, cg.dt, cg.timed)).encode("utf-8")).hexdigest()[:16]


# ============================================================================ CHANNELS text tiers (§3.2)
def compact_line(res):
    """Tier 2: one line per target -- total = direct + the largest shares (flags), the rest netted."""
    y = res["target"]
    step = res.get("step", "final")
    st = (f" (peak step {step})" if "peak_step" in res else
          f" (step {step})" if isinstance(step, int) and not isinstance(step, bool) else "")
    if res["mode"] == "none":
        return f"  CHANNELS {y}{st}: " + "; ".join(res.get("notes") or ["not computed"])
    chans, T, D = res["channels"], res["T"], res["D"]
    k = len(chans)
    mem = f", {res.get('n_used', res['members'])} members" if res.get("mc") else ""
    if k == 0:
        return f"  CHANNELS {y}{st}: total {_sg(T)} = direct {_sg(D, T)} [no mediators{mem}]"
    shares, flags, ref = _flags(res)
    plural = "s" if k > 1 else ""
    if res.get("phi") is not None:
        order = sorted(range(k), key=lambda c: -abs(shares[c]))
        top, rest = order[:COMPACT_TOP], order[COMPACT_TOP:]
        parts = [f"{chans[c]['name']} {_sg(shares[c], ref)}" + (f" ({' '.join(flags[c])})" if flags[c] else "")
                 for c in top]
        if rest:
            parts.append(f"{len(rest)} other{'s' if len(rest) > 1 else ''} "
                         f"(net {_sg(sum(shares[c] for c in rest), ref)})")
        how = "exact" if res["mode"] == "exact" else f"approximate, {res.get('perms', 0)} permutations"
        return (f"  CHANNELS {y}{st}: total {_sg(T, ref)} = direct {_sg(D, ref)} + " + " + ".join(parts)
                + f" [{how}, {k} channel{plural}{mem}]")
    return (f"  CHANNELS {y}{st}: total {_sg(T, ref)} = direct {_sg(D, ref)} + channels {_sg(T - D, ref)} "
            f"[shares not computed (time); alone and but-for in the detail; {k} channel{plural}{mem}]")


def detail_block(res, label):
    """Tier 3: G3C2's CHANNELS block without its total line (tier 2 has it) and without the caveat (once per
    result)."""
    if res["mode"] == "none" or not res.get("channels"):
        return ""
    lines = channels_text(res, label, chars=10 ** 7).split("\n")
    keep = [lines[0]] + [x for x in lines[2:] if x.strip() != CAVEAT]
    return "\n".join("  " + x for x in keep)


# ============================================================================ EXPLAIN priority (§3.2)
def explain_lines(doc, spec, n, seed, steps, budget_s):
    """G3C2's explain_run, computed unchanged, with its final 2,000-character cut replaced by 'keep every line'."""
    g = dict(vars(G3))
    g["_cut_block"] = lambda lines, chars: "\n".join(lines)
    f = G3.explain_run
    fn = types.FunctionType(f.__code__, g, f.__name__, f.__defaults__, f.__closure__)
    return fn(doc, spec, n, seed, steps, budget_s).split("\n")


def rank_explain(lines):
    """[(rank, line)]: 1 event status and per-candidate verdicts (and warnings); 2 the observation table;
    3 the per-candidate channel lines and the contingency boilerplate."""
    out, in_obs = [], False
    for ln in lines:
        if ln.startswith("  OBSERVATIONS"):
            in_obs = True
            out.append((2, ln))
            continue
        if in_obs and ln.startswith("  - "):
            out.append((2, ln))
            continue
        in_obs = False
        if ln.startswith("     under the contrast:") or ln.strip().startswith("note: contingencies searched"):
            out.append((3, ln))
        else:
            out.append((1, ln))
    return out


def cut_explain(ranked, chars=EXPLAIN_CHARS):
    """Cut a ranked EXPLAIN block to `chars`: channel lines first, then observation rows, then verdicts (never the
    header). Returns the kept [(rank, line)] plus a stated cut note."""
    keep = [True] * len(ranked)
    size = sum(len(ln) + 1 for _, ln in ranked)
    cut = {1: 0, 2: 0, 3: 0}
    for rank in (3, 2, 1):
        for i in reversed(range(1, len(ranked))):
            if size <= chars - 130:
                break
            if ranked[i][0] == rank and keep[i]:
                keep[i] = False
                size -= len(ranked[i][1]) + 1
                cut[rank] += 1
    out = [x for x, k in zip(ranked, keep) if k]
    if any(cut.values()):
        what = ", ".join(f"{n} {lab}" for n, lab in ((cut[3], "mechanism/note line(s)"),
                                                      (cut[2], "observation row(s)"), (cut[1], "verdict line(s)")) if n)
        out.append((1, f"  [cut to the bound: {what}; ask about fewer candidates or observations to see them]"))
    return out


# ============================================================================ the tiered assembler
class _U:
    """One output unit: tier 1 is never cut (except the equal-share fallback); higher tiers are cut first."""
    __slots__ = ("tier", "text", "kind", "target", "kept", "pos")

    def __init__(self, tier, text, kind="", target=None):
        self.tier, self.text, self.kind, self.target, self.kept, self.pos = tier, text, kind, target, True, 0


def assemble(secs, limit=MAX_OUT):
    """secs: [{"units": [_U], "note": f(dropped units) -> [lines] | None, "whatif": bool}] in output order.
    Cuts the highest tier first (latest unit first within a tier) until the text fits; every cut is stated."""
    pos = 0
    for s in secs:
        for u in s["units"]:
            u.pos = pos
            pos += 1
    last_wi = max((i for i, s in enumerate(secs) if s.get("whatif")), default=-1)

    def render():
        out = []
        detail = any(u.kept and u.kind == "detail" for s in secs for u in s["units"])
        for i, s in enumerate(secs):
            out += [u.text for u in s["units"] if u.kept and u.text]
            dropped = [u for u in s["units"] if not u.kept]
            if dropped and s.get("note"):
                out += s["note"](dropped)
            if i == last_wi and detail:
                out.append("  " + CAVEAT)
        return "\n".join(out)
    text = render()
    if len(text) <= limit:
        return text
    for u in sorted((u for s in secs for u in s["units"] if u.tier > 1), key=lambda u: (-u.tier, -u.pos)):
        u.kept = False
        text = render()
        if len(text) <= limit:
            return text
    eq = [s for s in secs if s.get("whatif")]
    if eq:                                    # tier 1 alone exceeds the bound: equal share per what-if
        t1 = [u for s in eq for u in s["units"] if u.kept and u.tier == 1]
        other = len(text) - sum(len(u.text) + 1 for u in t1)
        share = max(200, (limit - other - len(EQUAL_SHARE_NOTE) - 40) // len(eq))
        for s in eq:
            us = [u for u in s["units"] if u.kept and u.tier == 1]
            if not us:
                continue
            joined = "\n".join(u.text for u in us if u.text)
            us[0].text = _cut_lines(joined, share)
            for u in us[1:]:
                u.text = ""
        text = render() + "\n" + EQUAL_SHARE_NOTE
    return _bound(text, limit)


# ============================================================================ the arm
class G4C1State(G3C2State):
    def __init__(self, rep=None, given_gold=False):
        super().__init__(rep=rep, given_gold=given_gold)
        _ensure(self)


def _ensure(st):
    """Episode-scoped G4C1 fields (also on a plain G3C2/G1C3 state handed to this arm)."""
    for k, f in (("n_calls", int), ("answered", dict), ("empty_submits", int), ("ch_cache", dict),
                 ("last_shown", set)):
        if not hasattr(st, k):
            setattr(st, k, f())
    return st


_DO_DESC = "do = {var: value | {value|scale|add: x, from?: step} | '+'|'-' (qualitative)}."


class G4C1Arm(G3C2Arm):
    name = "G4C1"
    buildable = True
    executable = True
    guide_file = "g4c1_guide.md"
    mutating_tools = MUTATING
    custom_episode = True

    def tools(self):
        return [
            _tool("submit_graph", "Compile the whole graph document and replace the representation. Takes `source` "
                  "(document text) or `graph` (A2 JSON); with neither, the last ```graph block of this response "
                  "(a call with none of these stores nothing). Returns the coverage report, the baseline of the "
                  "targets, every stored whatif with its totals and full CHANNELS, and every stored explain.",
                  {"source": {"type": "string"}, "graph": {"type": "object"}}),
            _tool("patch_graph", "Edit the document, then recompile and rerun as submit_graph. `upsert`: whole "
                  "document lines, each replacing the line that declares the same name (variable, edge a->b, whatif, "
                  "mechanism, explain, time, targets:) or appended if new; and/or `delete`: names to remove "
                  "(\"x\", \"whatif w\", \"edge a->b\") or whole lines. Applied per name: each entry on its own; the "
                  "result lists what was applied and every entry NOT applied with its reason. Never put a name in "
                  "both lists. Or `old`/`new`: exact unique text replace.",
                  {"upsert": {"type": "array", "items": {"type": "string"}},
                   "delete": {"type": "array", "items": {"type": "string"}},
                   "old": {"type": "string"}, "new": {"type": "string"}}),
            _tool("evaluate", "Compute variable values (static, over time, or Monte Carlo).", {"query": _QUERY}),
            _tool("intervene", "Compare interventions with the baseline (paired), with full CHANNELS (the share of "
                  "each mechanism in the effect). Either `do` + `query`, or `whatifs`: up to 6 of "
                  "{name?, do?, query?, at?} or names of stored whatif lines, run on one shared baseline. A stored "
                  "name takes the call's `query` (field by field: targets, events, samples, steps, seed) and `at`. "
                  + _DO_DESC + " at: 'final' | step | 'peak' (CHANNELS only) | 'all'. channels: false skips "
                  "CHANNELS. Output order: totals first, then one CHANNELS line per target, then CHANNELS detail, "
                  "qualitative signs, series (cut in reverse order at the bound). An identical repeat returns a "
                  "reference to the earlier result.",
                  {"do": {"type": "object"}, "query": _QUERY,
                   "whatifs": {"type": "array", "items": {}, "description": "what-ifs or stored whatif names"},
                   "channels": {"type": "boolean"}, "at": {"description": "'final' | step | 'peak' | 'all'"}}),
            _tool("explain", "Actual causes of an event: for each candidate (with its contrast = its normal or working "
                  "value), whether it is a but-for cause, a cause under a contingency, a joint cause or not a cause, "
                  "its mechanism, and which observed events it accounts for. Either `name` of a stored explain line, "
                  "or `event` + `candidates` {name: contrast} [+ `observed` [events], `at` final|any|step, `samples`]. "
                  "An identical repeat returns a reference to the earlier result.",
                  {"name": {"type": "string"}, "event": {"type": "string"},
                   "candidates": {"type": "object"}, "observed": {"type": "array", "items": {"type": "string"}},
                   "at": {"description": "'final' | 'any' | step"}, "samples": {"type": "integer"}}),
            _tool("show", "Show the document with line numbers, mechanisms (declared and automatic), stored explains, "
                  "the compiled graph and its holes.", {}),
        ]

    # ---------------------------------------------------------------- state
    def new_state(self):
        st = G4C1State()
        self._set_doc(st, G3Doc(graph={"variables": [], "edges": [], "equations": {}}))
        return st

    def load(self, rep, given_gold=False):
        st = G4C1State(given_gold=given_gold)
        if isinstance(rep, dict) and isinstance(rep.get("source"), str):
            self._set_doc(st, G3Doc(source=rep["source"]))
        elif isinstance(rep, dict):
            self._set_doc(st, G3Doc(graph=rep))
        else:
            self._set_doc(st, G3Doc(graph={"variables": [], "edges": [], "equations": {}}))
            st.log.append(("load", None, f"representation was {type(rep).__name__}, not an object; started empty"))
        return st

    def handle(self, state, name, args):
        _ensure(state).n_calls += 1
        return G3C2Arm.handle(self, state, name, args)

    # ---------------------------------------------------------------- building
    def _t_submit_graph(self, state, args):
        st = _ensure(state)
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
            st.empty_submits += 1
            return EMPTY_FIRST if st.empty_submits == 1 else EMPTY_AGAIN
        self._set_doc(state, doc)
        return self._report4(state, doc, f"graph submitted from the {how}")

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
        src, parts = doc.source, []
        if old is not None:                                  # unchanged: exact unique replacement, before upserts
            if not isinstance(old, str) or not old or not isinstance(new, str):
                return "ERROR: old/new: 'old' must be non-empty text and 'new' text (hint: copy 'old' exactly from show)"
            k = src.count(old)
            if k == 0:
                import difflib
                near = difflib.get_close_matches(old.strip().split("\n")[0], [x.strip() for x in src.split("\n")],
                                                 n=1, cutoff=0.4)
                return ("ERROR: 'old' text not found in the document" + (f" (closest line: {near[0]!r})" if near else "")
                        + " (hint: copy it exactly, or use upsert with the whole line)")
            if k > 1:
                return f"ERROR: 'old' text occurs {k} times (hint: use upsert with the whole line instead)"
            src = src.replace(old, new, 1)
            parts.append("old/new replaced")
        not_applied, notes = [], []
        if ups is not None or dels is not None:
            src, applied, not_applied, notes = apply_upsert_per_name(src, ups or [], dels or [])
            for kind in ("replaced", "appended", "deleted"):
                if applied[kind]:
                    parts.append(f"{kind} {len(applied[kind])} ({', '.join(applied[kind])})")
            if not parts:
                return ("ERROR: nothing was applied. " + (_not_applied_text(not_applied) if not_applied else
                                                          "upsert and delete are empty")
                        + " (hint: send each line whole in upsert; delete takes names)")
        head = "patch applied: " + ", ".join(parts)
        if not_applied:
            head += "; " + _not_applied_text(not_applied)
        if notes:
            head += "; notes: " + "; ".join(notes[:8])
        nd = G3Doc(source=src)
        self._set_doc(state, nd)
        return self._report4(state, nd, head + "\ncompiled")

    # ---------------------------------------------------------------- the submit/patch report (§3.4)
    def _report(self, doc, head):
        return self._report4(None, doc, head)

    def _report4(self, state, doc, head):
        doc._g4_in_report, doc._g4_pending = True, None
        try:
            base = G1C3Arm._report(self, doc, head)
        finally:
            doc._g4_in_report = False
        blocks = doc._g4_pending or []
        doc._g4_pending = None
        sec = []
        if doc.mechanisms:
            sec.append("MECHANISMS: " + "; ".join(f"{m['name']} ({', '.join(m['vars'])})" for m in doc.mechanisms))
        if doc.explains:
            sec.append("EXPLAINS: " + ", ".join(x["name"] for x in doc.explains))
        if doc.notes:
            sec.append(f"MECHANISM/EXPLAIN NOTES ({len(doc.notes)}; notes only, nothing is rejected):\n"
                       + "\n".join(f"  - {n}" for n in doc.notes[:10])
                       + (f"\n  ... {len(doc.notes) - 10} more" if len(doc.notes) > 10 else ""))
        pre, _, post = base.partition(_MARK)
        post = post.replace(_MARK, "")
        lines = pre.rstrip("\n").split("\n")
        nh = head.count("\n") + 1
        if sec:
            lines = lines[:nh] + "\n".join(sec).split("\n") + lines[nh:]
        # the prefix: coverage and baseline (tier 1), the baseline series (tier 5, cut last within tier 5)
        i = next((k for k, ln in enumerate(lines) if ln.startswith("SERIES (")), None)
        pu = []
        if i is None:
            pu.append(_U(1, "\n".join(lines)))
        else:
            j = i + 1
            while j < len(lines) and lines[j].startswith(" "):
                j += 1
            pu += [_U(1, "\n".join(lines[:i])), _U(5, "\n".join(lines[i:j]), "bseries"),
                   _U(1, "\n".join(lines[j:]))]
        secs = [{"units": pu, "note": lambda d: ["(baseline series cut to the output bound; evaluate with "
                                                 "query {\"at\": \"all\"} shows them)"]}]
        # stored what-ifs with full CHANNELS (cached by dependence)
        st = _ensure(state) if state is not None else None
        cg = doc.cg
        run = [b for b in blocks if not b.get("error") and b.get("do")]
        rows, unchanged, ckeys = {}, set(), {}
        todo = []
        for b in run:
            ck = (dep_signature(doc, list(b["do"]), b["q_pre"]["targets"]), b["name"],
                  json.dumps(b["do"], sort_keys=True, default=str), tuple(b["q_pre"]["targets"]), b["at_ch"],
                  b["q_pre"]["samples"], b["q_pre"]["steps"], b["q_pre"]["seed"])
            ckeys[id(b)] = ck
            if st is not None and ck in st.ch_cache:
                rows[id(b)] = st.ch_cache[ck]
                if ck in st.last_shown:
                    unchanged.add(id(b))
            else:
                todo.append(b)
        if todo:
            res = self._channel_results(cg, doc, [(b["name"], b["do"], b["q_pre"], b["at_ch"]) for b in todo],
                                        SUBMIT_CHANNELS_BUDGET_S)
            for b, r in zip(todo, res):
                rows[id(b)] = r
                if st is not None and not any(x["mode"] == "none" for _, x in r):
                    st.ch_cache[ckeys[id(b)]] = r
        wsecs = []
        for b in blocks:
            s = self._whatif_section(b, rows.get(id(b)), multi=len(blocks) > 1, unchanged=id(b) in unchanged)
            wsecs.append((b, s))
            secs.append(s)
        # stored explains (the first 3, sharing 6 s), each cut by its own priority
        if cg.vars and doc.explains:
            t_end = time.perf_counter() + SUBMIT_EXPLAIN_BUDGET_S
            for x in doc.explains[:MAX_EXPLAINS_AT_SUBMIT]:
                left = max(0.05, t_end - time.perf_counter())
                secs.append(self._explain_section(doc, x, left))
            if len(doc.explains) > MAX_EXPLAINS_AT_SUBMIT:
                secs.append({"units": [_U(1, f"(only the first {MAX_EXPLAINS_AT_SUBMIT} stored explains were run; "
                                             "run others with explain name=...)")]})
        if post.strip():
            secs.append({"units": [_U(1, post.strip("\n"))]})
        text = assemble(secs, MAX_OUT)
        if st is not None:
            shown = set()
            for b, s in wsecs:
                if id(b) in ckeys and any(u.kept and u.kind == "compact" for u in s["units"]):
                    shown.add(ckeys[id(b)])
            st.last_shown = shown
        return text

    def _batch(self, cg, doc, wis, q_default, room=MAX_OUT, baseline=None):
        """G1C3's batch with the effective queries of §3.1; inside the submit report it hands the blocks to
        _report4 (which adds CHANNELS and applies the tiers)."""
        blocks = self._prepare(cg, doc, [self._resolve_eff(cg, doc, w, q_default, None) for w in wis])
        self._run_blocks(cg, doc, blocks, baseline)
        if getattr(doc, "_g4_in_report", False):
            doc._g4_pending = blocks
            return _MARK
        return assemble([self._whatif_section(b, None, multi=len(blocks) > 1) for b in blocks], room)

    # ---------------------------------------------------------------- what-if resolution (§3.1)
    def _resolve(self, cg, doc, w, q_default):
        """G1C3's interface: (name, do_raw, query raw) with the effective query, or an error string."""
        r = self._resolve_eff(cg, doc, w, q_default, None)
        if isinstance(r, str):
            return r
        return r["name"], r["do_raw"], dict(r["qraw"], at=r["at"])

    def _resolve_eff(self, cg, doc, w, call_q, call_at):
        """A what-if entry -> {name, do_raw, qraw, at, notes, narrow} or an error string."""
        stored = {x["name"]: x for x in doc.whatifs}
        if isinstance(w, str):
            if w not in stored:
                return f"whatif '{w}': not stored{_did_you_mean(w, stored)} (stored: {', '.join(stored) or 'none'})"
            w = {"name": w}
        if not isinstance(w, dict):
            return "whatif: must be a name or an object {name?, do?, query?, at?}"
        name = w.get("name") or "whatif"
        s = stored.get(name) if isinstance(name, str) else None
        notes = []
        do_given = w.get("do") is not None
        if do_given:
            do = _maybe_json(w.get("do"))
            if not isinstance(do, dict) or not do:
                return f"whatif '{name}': do must be a non-empty object"
            if s is not None:
                notes.append("stored do replaced by the call's do")
        elif s is not None:
            do = s["do"]
        else:
            return f"whatif '{name}': do must be a non-empty object"
        try:
            q_entry = _qdict(w.get("query"))
            q_call = _qdict(call_q)
        except ValueError as e:
            return f"whatif '{name}': bad query: {e} (hint: query={{\"targets\": [\"y\"]}})"
        q_stored = {k: s[k] for k in ("targets", "events", "samples", "steps", "seed")
                    if s is not None and s.get(k) not in (None, [])}
        q = {}
        for lvl in (q_stored, q_call, q_entry):              # lowest precedence first; field by field
            for k, v in lvl.items():
                if k != "at":
                    q[k] = v
        if not q.get("targets"):
            q["targets"] = list(doc.targets)
        at_src = None
        for cand in (w.get("at"), q_entry.get("at"), call_at, q_call.get("at"), s.get("at") if s else None):
            if cand is not None and cand != "":
                at_src = cand
                break
        at, note = _norm_at(at_src) if at_src is not None else ("final", None)
        if note:
            notes.append(note)
        if at == "peak":
            notes.append("peak applies to CHANNELS; WHATIF at final")
        elif at == "all":
            notes.append("at all: WHATIF at final with its series; CHANNELS at final")
        narrow = name if (s is not None and not do_given) else {"name": name, "do": do}
        return {"name": name, "do_raw": do, "qraw": q, "at": at, "notes": notes, "narrow": narrow}

    def _prepare(self, cg, doc, entries):
        """Parse each resolved entry's effective query (before the batch's sample cap)."""
        out = []
        for e in entries:
            if isinstance(e, str):
                out.append({"error": "ERROR " + e})
                continue
            qraw = dict(e["qraw"])
            qraw["at"] = e["at"] if isinstance(e["at"], int) else "final"
            if not qraw.get("targets"):
                qraw["targets"] = None
            try:
                q = self._parse_query(cg, qraw)
            except (ValueError, TypeError) as ex:
                out.append({"error": f"ERROR whatif '{e['name']}': bad query: {ex} "
                                     "(hint: query={\"targets\": [\"y\"]})"})
                continue
            q["notes"] = list(e["notes"]) + q["notes"]
            do_raw = e["do_raw"] if isinstance(e["do_raw"], dict) else _maybe_json(e["do_raw"])
            do, qual, _ = self._norm_do(cg, do_raw if isinstance(do_raw, dict) else {})
            at_ch = e["at"] if isinstance(e["at"], int) else ("peak" if e["at"] == "peak" else "final")
            out.append(dict(e, error=None, q=q, q_pre=copy.deepcopy(q), do=do, qual=qual, at_ch=at_ch))
        return out

    def _run_blocks(self, cg, doc, blocks, baseline=None):
        """G1C3's batch run (one shared baseline per steps/samples/seed; the same sample cap) per block, with the
        result split into tiers: t1 header + targets + events, t4 QUALITATIVE, t5 SERIES."""
        specs = [b for b in blocks if not b.get("error") and not b.get("repeat")]
        n_runs = len({(b["q"]["steps"], b["q"]["samples"], b["q"]["seed"]) for b in specs}) + len(specs)
        cache = {}
        if baseline and baseline[0] is not None:
            bq, bruns = baseline
            cache[(bq["steps"], bq["samples"], bq["seed"])] = bruns
            n_runs = 1 + len(specs)
        for b in specs:
            q = b["q"]
            _cap_samples(cg, q, n_runs)
            key = (q["steps"], q["samples"], q["seed"])
            if key not in cache:
                cache[key] = self._baseline(cg, q)
            head, body, series = self._one_whatif(cg, b["name"], b["do_raw"], q, cache[key])
            if head.startswith("ERROR"):
                b["error"] = "\n".join([head] + body)
                continue
            i = head.rfind("  [")
            if i >= 0 and head.endswith("]"):
                head = head[:i] + "  " + _eff_text(cg, q, b["at"])
            qi = next((k for k, ln in enumerate(body) if ln.startswith("  QUALITATIVE")), len(body))
            b["t1"], b["t4"], b["t5"] = [head] + body[:qi], body[qi:], series
        return blocks

    def _whatif_section(self, b, rows, multi, unchanged=False):
        if b.get("error"):
            return {"units": [_U(1, b["error"], "error")], "whatif": True}
        if b.get("repeat"):
            return {"units": [_U(1, b["repeat"], "repeat")], "whatif": True}
        units = [_U(1, "\n".join(b["t1"]), "t1")]
        rows = rows or []
        if unchanged:
            units.append(_U(2, "  " + UNCHANGED, "compact"))
        else:
            for y, res in rows:
                units.append(_U(2, compact_line(res), "compact", y))
            for y, res in rows:
                d = detail_block(res, b["name"])
                if d:
                    units.append(_U(3, d, "detail", y))
        if b.get("t4"):
            units.append(_U(4, "\n".join(b["t4"]), "qual"))
        if b.get("t5"):
            units.append(_U(5, "\n".join(b["t5"]), "series"))
        n_targets = len(rows)
        ref = json.dumps(b["narrow"], default=str)
        at_arg = "" if b["at"] == "final" else f", at={json.dumps(b['at'])}"

        def note(dropped):
            kinds = {u.kind for u in dropped}
            out = []
            low = [lab for k, lab in (("qual", "QUALITATIVE"), ("series", "SERIES")) if k in kinds]
            if low:
                out.append("  (cut to the output bound: " + ", ".join(low) + ")")
            comp = [u.target for u in dropped if u.kind == "compact" and u.target is not None]
            det = [u.target for u in dropped if u.kind == "detail" and u.target not in comp]
            for lab, ys in (("CHANNELS", comp), ("CHANNELS detail", det)):
                if not ys:
                    continue
                if n_targets > 1:
                    out.append(f"  {lab} cut for {', '.join(ys)}: call intervene with whatifs=[{ref}], "
                               f"query={{\"targets\": [\"{ys[0]}\"]}}{at_arg}")
                elif multi:
                    out.append(f"  {lab} cut for {ys[0]}: call intervene with whatifs=[{ref}]{at_arg}")
                else:
                    out.append(f"  {lab} for {ys[0]} cut to the output bound (one target; no narrower call shows "
                               "more)")
            return out
        return {"units": units, "note": note, "whatif": True}

    # ---------------------------------------------------------------- CHANNELS (G3C2's computation, structured)
    def _channel_results(self, cg, doc, specs, budget_s):
        """G3C2's _channels in full mode, returning per spec [(target, decompose result)] instead of text."""
        out = []
        if not specs:
            return out
        t_end = time.perf_counter() + budget_s
        ch = _children(cg)
        for j, (name, do, q, q_at) in enumerate(specs):
            share_end = time.perf_counter() + max(0.0, t_end - time.perf_counter()) / (len(specs) - j)
            rows = []
            out.append(rows)
            targets = [t for t in q["targets"] if t in cg.vars and t not in do and cg.vars[t]["type"] != "category"]
            if not targets or not do:
                continue
            if q_at not in ("final", "peak") and not (isinstance(q_at, int) and not isinstance(q_at, bool)):
                q_at = "final"
            n_max = min(q["samples"], MEMBERS_MAX) if q["samples"] else 0
            W = Worlds(cg, q["steps"], do, 1 if n_max else 0, q["seed"], share_end + 3600)
            plans = []
            for y in targets:
                chans, notes = channels_for(cg, doc.mechanisms, list(do), y, ch)
                plans.append([y, chans, notes])
            need = sum(2 ** min(len(c), MAX_CHANNELS) for _, c, _ in plans)
            need = max(1, need) + sum(len(c) + 1 for _, c, _ in plans if len(c) > MAX_CHANNELS)
            left = max(0.0, share_end - time.perf_counter())
            if n_max:
                fit = int(left / max(1e-6, need * W.t_run))
                n_d = max(min(n_max, MEMBERS_MIN), min(n_max, fit))
                W = Worlds(cg, q["steps"], do, n_d, q["seed"], share_end)
            else:
                W.deadline = share_end
            for ti, (y, chans, notes) in enumerate(plans):
                try:
                    if len(chans) > MAX_CHANNELS:
                        chans, n2 = self._merge_tail(W, y, chans, q_at)
                        notes = notes + [n2]
                except OutOfTime:
                    rows.append((y, {"target": y, "mode": "none", "channels": [], "notes": ["not computed (time)"]}))
                    continue
                k = len(chans)
                tl = max(0.0, W.deadline - time.perf_counter()) / (len(plans) - ti)
                cost = W.t_run * W.members
                if 2 ** k * cost <= tl:
                    mode = "exact"
                elif (4 * k + 2) * cost <= tl:
                    mode = "perm"
                else:
                    mode = "light"                              # minimal: alone and but-for only
                W.deadline = min(share_end, time.perf_counter() + tl)
                res = decompose(W, y, chans, mode, q_at)
                W.deadline = share_end
                if res["mode"] == "light":
                    res["mode"] = "minimal"
                res["notes"] = notes + res["notes"]
                if n_max and W.members < n_max:
                    res["notes"].append(f"{W.members} of {q['samples']} members used (time budget)")
                elif n_max and q["samples"] > W.members:
                    res["notes"].append(f"first {W.members} of the {q['samples']} WHATIF samples")
                rows.append((y, res))
        return out

    # ---------------------------------------------------------------- intervene (§3.1, §3.2)
    def _ikey(self, doc, b, chan_on):
        q = b["q_pre"]
        return ("intervene", doc_hash(doc), json.dumps([b["do"], b["qual"]], sort_keys=True, default=str),
                tuple(q["targets"]), tuple(q["events"]), b["at"], q["samples"], q["steps"], q["seed"], bool(chan_on))

    @staticmethod
    def _ilabel(b):
        tg = list(b["q_pre"]["targets"])
        return (f"intervene {b['name']}; targets " + (", ".join(tg[:4]) + (f" +{len(tg) - 4}" if len(tg) > 4 else "")
                                                      if tg else "none") + f"; at {_at_label(b['at'])}")

    def _t_intervene(self, state, args):
        st = _ensure(state)
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
            wis = [{"name": "do", "do": do}]
        if not isinstance(wis, list) or not wis:
            return "ERROR: whatifs must be a non-empty list (hint: whatifs=[\"stored_name\", {\"do\": {...}}])"
        note = ""
        if len(wis) > MAX_BATCH:
            note = f"note: only the first {MAX_BATCH} of {len(wis)} what-ifs were run (limit per call)"
            wis = wis[:MAX_BATCH]
        chan_on = args.get("channels") not in (False, "false", 0)
        call_at = args.get("at")
        blocks = self._prepare(cg, doc, [self._resolve_eff(cg, doc, w, args.get("query"), call_at) for w in wis])
        for b in blocks:
            if not b.get("error"):
                b["key"] = self._ikey(doc, b, chan_on)
        rep = [b for b in blocks if not b.get("error") and b["key"] in st.answered]
        if rep and len(rep) == len(blocks):
            refs = " and ".join(f"call #{st.answered[b['key']][0]} ({st.answered[b['key']][1]})" for b in rep)
            return f"REPEAT: identical to the result of {refs}. {REPEAT_TAIL}"
        for b in rep:
            k, lab = st.answered[b["key"]]
            b["repeat"] = f"WHATIF {b['name']}: REPEAT: identical to the result of call #{k} ({lab})."
        self._run_blocks(cg, doc, blocks, None)
        run = [b for b in blocks if not b.get("error") and not b.get("repeat")]
        rows = {}
        if chan_on:
            ok = [b for b in run if b["do"]]
            res = self._channel_results(cg, doc, [(b["name"], b["do"], b["q_pre"], b["at_ch"]) for b in ok],
                                        INTERVENE_BUDGET_S)
            rows = {id(b): r for b, r in zip(ok, res)}
        secs = []
        if note:
            secs.append({"units": [_U(1, note)]})
        for b in blocks:
            secs.append(self._whatif_section(b, rows.get(id(b)), multi=len(blocks) > 1))
        holes = _holes_text(cg, 6).strip("\n")
        if holes:
            secs.append({"units": [_U(1, holes)]})
        text = assemble(secs, MAX_OUT)
        for b in run:
            st.answered.setdefault(b["key"], (st.n_calls, self._ilabel(b)))
        return text

    # ---------------------------------------------------------------- explain (§3.1 repeats, §3.2 priority)
    def _explain_ranked(self, doc, x, budget_s):
        cg = doc.cg
        has_dist = any("distribution" in s for s in cg.vars.values())
        samples = x.get("samples")
        n = min(MEMBERS_MAX, samples if samples is not None else (MEMBERS_MAX if has_dist else 0))
        n = max(0, int(n)) if has_dist else 0
        steps = cg.steps if cg.timed else 0
        return cut_explain(rank_explain(explain_lines(doc, x, n, 0, steps, budget_s)), EXPLAIN_CHARS)

    def _explain_section(self, doc, x, budget_s):
        tiers = {1: 2, 2: 3, 3: 3.5}
        units = [_U(tiers[r], ln, "explain") for r, ln in self._explain_ranked(doc, x, budget_s)]
        nm = x.get("name") or "explain"
        return {"units": units,
                "note": lambda d: [f"  [EXPLAIN {nm}: {len(d)} line(s) cut to the output bound; call explain with "
                                   f"name=\"{nm}\" to see it alone]"]}

    def _t_explain(self, state, args):
        st = _ensure(state)
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
        else:
            ev = args.get("event")
            cands = _maybe_json(args.get("candidates"))
            if not isinstance(ev, str) or not ev.strip():
                return ("ERROR: explain needs `event` (an expression, e.g. \"revenue < 500\") or the `name` of a "
                        "stored explain")
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
            x = {"name": nm or "explain", "event": ev, "candidates": clist, "observed": list(obs), "at": None,
                 "samples": args.get("samples")}
            if args.get("at") is not None:
                x["at"] = args.get("at")
        if isinstance(x.get("at"), str) and x["at"].strip().isdigit():
            x["at"] = int(x["at"])
        key = ("explain", doc_hash(doc), str(x.get("event")), json.dumps(x.get("candidates"), default=str),
               tuple(x.get("observed") or []), str(x.get("at")), str(x.get("samples")))
        if key in st.answered:
            k, lab = st.answered[key]
            return f"REPEAT: identical to the result of call #{k} ({lab}). {EXPLAIN_REPEAT_TAIL}"
        text = _cut_lines("\n".join(ln for _, ln in self._explain_ranked(doc, x, EXPLAIN_BUDGET_S)), MAX_OUT)
        if not text.startswith("ERROR") and "]: ERROR" not in text.split("\n")[0]:
            at = x.get("at") or _split_event_at(str(x.get("event")))[1] or "final"
            st.answered[key] = (st.n_calls, f"explain {x.get('name') or 'explain'}; at {_at_label(at)}")
        return text


G4C1 = G4C1Arm
