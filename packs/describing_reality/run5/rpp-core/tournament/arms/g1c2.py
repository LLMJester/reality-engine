"""G1C2: execute-then-explain (A3 + A1 recombination; runs/discovery/gen1/specs/G1C2.md).

One document with two linked parts:
  * the mechanism LEDGER (fenced block `ledger`): A1-style tagged prose reduced to QUESTION TARGETS, ENTITIES,
    MECHANISMS, COUPLINGS, BINDING CONSTRAINTS, DISTRACTORS and TIMING. Every element is one line with an id and
    exactly one provenance tag ([stated: "quote"], [derived: from ids], [assumed: reason; range lo-hi]);
    Q lines are answer obligations (an observed key in backticks, or [qualitative]);
  * the SIMULATION (fenced block `model`): the A3 API (init_state / step / observe) with declared literal PARAMS,
    RULES and HORIZON (G1C1 conventions). Each mechanism's code starts with a `# M<k>` tag and calls the injected
    `fire("M<k>", amount)` when it acts. `step(state, t, rng)` switches to seeded runs (default 50 seeds).

The executor checks the ledger, maps ledger ids to code regions via the `# Mk` tags (coverage), runs a baseline plus
up to 4 named runs (overrides on PARAMS/RULES, per-run ephemeral `what_if_patch`) in one sandboxed process with
common seeds, and reports CHECK, FINAL, EFFECT vs baseline, SHAPE, MECHANISM ACTIVITY and SERIES within 8,000
characters. The arm drives its own episode (document channel, implicit submit, reserved answer call).
The A3 sandbox and driver are copied here (a3_sim.py is not edited; other arms import it).
"""
import ast
import copy
import difflib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from tournament.arms.a1_prose import TAG_RE, ANSWER_HEADING
from tournament.arms.a3_sim import HINTS, fmt_err, parse_lines, static_check
from tournament.arms.a4_prob import _quant, summarize as _summarize_a4
from tournament.arms.base import Arm, ArmState, ERROR_PREFIX, strip_answer_fields

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = ROOT / "harness" / "sandbox_runner.py"
# Runtime execution dependency (D-060/D-061): models run in the A3 sandbox subprocess; the final lock covers it.
RUNTIME_DEPS = ["harness/sandbox_runner.py"]
MAX_CODE = 60_000
MAX_LEDGER = 20_000
MAX_OUT = 7_800            # leaves room for the budget status line within the 8,000-character tool result
MAX_RUNS = 4               # named runs per call (the baseline is always added)
MAX_STEPS = 1000
DEFAULT_HORIZON = 30
DET_SEEDS, PROB_SEEDS, MAX_SEEDS = 1, 50, 200
WORK_CAP = 400_000         # runs x seeds x steps per call; seeds are reduced proportionally above it
TIMEOUT = 60
SERIES_KEYS, SERIES_POINTS = 10, 60
MAX_SUM_KEYS = 60          # numeric keys whose mean trajectory the driver keeps

# ---------------------------------------------------------------------------------------------------------------------
# Sandbox driver (copy of the A3 driver, extended): injected fire(), PARAMS/RULES merge and rebind, per-run source
# (ephemeral patch), several named runs with common seeds, aggregation inside the process (finals per seed, mean
# trajectories, per-seed mechanism activity).
DRIVER = r'''
import copy, io, json, math, random, sys, traceback, inspect
P = json.load(open("params.json", encoding="utf-8"))
SRC = open("model.py", encoding="utf-8").read()
OUT = sys.stdout
CAP = io.StringIO()
sys.stdout = CAP
LINES = SRC.splitlines()
MAXK = P.get("max_sum_keys", 60)

def _j(x, d=0):
    if d > 12:
        return repr(x)[:80]
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return x if abs(x) < 2**53 else float(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, dict):
        return {str(k): _j(v, d + 1) for k, v in list(x.items())[:500]}
    if isinstance(x, (list, tuple, set, frozenset)):
        xs = list(x) if not isinstance(x, (set, frozenset)) else sorted(x, key=repr)
        return [_j(v, d + 1) for v in xs[:500]]
    try:
        f = float(x)
        return f if math.isfinite(f) else str(f)
    except Exception:
        return repr(x)[:120]

def emit(tag, obj):
    OUT.write("@@" + tag + " " + json.dumps(obj) + "\n")
    OUT.flush()

def where(e, phase, t=None):
    d = {"phase": phase, "type": type(e).__name__, "msg": str(e)[:300], "t": t}
    if isinstance(e, SyntaxError) and (e.filename in ("model.py", "<unknown>", None)):
        d["line"] = e.lineno
        d["col"] = e.offset
        d["code"] = (e.text or "").strip()[:200]
        return d
    fr = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "model.py"]
    if fr:
        f = fr[-1]
        d["line"] = f.lineno
        d["func"] = f.name
        d["code"] = LINES[f.lineno - 1].strip()[:200] if 0 < f.lineno <= len(LINES) else ""
    return d

def nargs(fn):
    try:
        return len([p for p in inspect.signature(fn).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except Exception:
        return -1

def flat(x, pre="", out=None):
    out = {} if out is None else out
    if len(out) > 300:
        return out
    if isinstance(x, dict):
        for k, v in x.items():
            flat(v, f"{pre}.{k}" if pre else str(k), out)
    elif isinstance(x, list):
        if len(x) <= 20:
            for i, v in enumerate(x):
                flat(v, f"{pre}.{i}" if pre else str(i), out)
        else:
            out[(pre + ".len") if pre else "len"] = len(x)
    elif isinstance(x, (bool, int, float, str)) or x is None:
        out[pre or "value"] = x
    return out

ACT = {"cur": {}, "calls": 0}

def fire(mid, amount=1):
    """Record that mechanism `mid` acted in the current step, moving `amount`."""
    try:
        a = float(amount)
        if not math.isfinite(a):
            a = 1.0
    except Exception:
        a = 1.0
    k = str(mid)
    ACT["cur"][k] = ACT["cur"].get(k, 0.0) + a
    ACT["calls"] += 1

def load(src):
    global LINES
    LINES = src.splitlines()
    ns = {"__name__": "model", "fire": fire}
    try:
        exec(compile(src, "model.py", "exec"), ns)
    except BaseException as e:
        return None, where(e, "load")
    if ns.get("Model") is None:
        return None, {"phase": "load", "type": "MissingModel", "msg": "no class named Model"}
    return ns, None

def trajectory(m, scen, steps, rng):
    """-> (observations, activity {mid: [n_steps, first_t, last_t, total]}, error_or_None). t=-1 means init."""
    obs, act = [], {}
    def note(t):
        for k, a in ACT["cur"].items():
            r = act.setdefault(k, [0, t, t, 0.0])
            r[0] += 1
            r[2] = t
            r[3] += a
        ACT["cur"] = {}
    obs_fn = getattr(m, "observe", None)
    def observe(s):
        o = copy.deepcopy(s) if obs_fn is None else obs_fn(s)
        if not isinstance(o, dict):
            raise TypeError(f"observe must return a dict, got {type(o).__name__}")
        return flat(_j(o))
    ACT["cur"] = {}
    try:
        init = getattr(m, "init_state", None)
        if init is None:
            state = copy.deepcopy(scen)
        elif rng is not None and nargs(init) >= 2:
            state = init(copy.deepcopy(scen), rng)
        else:
            state = init(copy.deepcopy(scen))
        if not isinstance(state, dict):
            raise TypeError(f"init_state must return a dict, got {type(state).__name__}")
    except BaseException as e:
        return obs, act, where(e, "init_state", 0)
    note(-1)
    try:
        obs.append(observe(state))
    except BaseException as e:
        return obs, act, where(e, "observe", 0)
    step = getattr(m, "step", None)
    if step is None:
        return obs, act, {"phase": "step", "type": "MissingStep", "msg": "Model has no step method", "t": 0}
    for t in range(steps):
        try:
            new = step(state, t, rng) if rng is not None else step(state, t)
            if new is None:
                new = state
            if not isinstance(new, dict):
                raise TypeError(f"step must return a dict (or None after mutating state), got {type(new).__name__}")
            state = new
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            ACT["cur"] = {}
            return obs, act, where(e, "step", t)
        note(t)
        try:
            obs.append(observe(state))
        except BaseException as e:
            return obs, act, where(e, "observe", t + 1)
    return obs, act, None

for run in P["runs"]:
    src = run.get("src") or SRC
    ns, err = load(src)
    if err:
        emit("RUN", {"name": run["name"], "load_err": err})
        continue
    p0 = ns.get("PARAMS") if isinstance(ns.get("PARAMS"), dict) else {}
    r0 = ns.get("RULES") if isinstance(ns.get("RULES"), dict) else {}
    p1, r1, extra = dict(p0), dict(r0), {}
    for k, v in (run.get("overrides") or {}).items():
        if k in p1:
            p1[k] = v
        elif k in r1:
            r1[k] = v
        else:
            extra[k] = v
    ns["PARAMS"], ns["RULES"] = p1, r1
    scen = {**p1, **r1, **extra}
    try:
        probe = ns["Model"]()
    except BaseException as e:
        emit("RUN", {"name": run["name"], "load_err": where(e, "Model()")})
        continue
    st = getattr(probe, "step", None)
    prob = st is not None and nargs(st) >= 3
    n = run["n_seeds"] if prob else 1
    finals, lens, errs, acts = [], [], [], []
    sums, cnts, bools = {}, {}, {}
    for i in range(n):
        s = run.get("seed", 0) * 1000003 + i
        random.seed(s)
        try:
            m = probe if i == 0 else ns["Model"]()
        except BaseException as e:
            errs.append([i, where(e, "Model()")])
            continue
        obs, act, e = trajectory(m, scen, run["steps"], random.Random(s) if prob else None)
        if e:
            errs.append([i, e])
        lens.append(len(obs))
        finals.append(obs[-1] if obs else None)
        acts.append(act)
        for t, o in enumerate(obs):
            for k, v in o.items():
                if isinstance(v, (int, float)) and (k in sums or len(sums) < MAXK):
                    if isinstance(v, float) and not math.isfinite(v):
                        continue
                    a = sums.setdefault(k, [])
                    c = cnts.setdefault(k, [])
                    while len(a) <= t:
                        a.append(0.0)
                        c.append(0)
                    a[t] += float(v)
                    c[t] += 1
                    if isinstance(v, bool):
                        bools[k] = True
    mean = {k: [(a[t] / cnts[k][t]) if cnts[k][t] else None for t in range(len(a))] for k, a in sums.items()}
    emit("RUN", {"name": run["name"], "prob": prob, "n_seeds": n, "finals": finals, "lens": lens,
                 "errs": errs[:5], "n_err": len(errs), "mean": mean, "bools": sorted(bools), "acts": acts,
                 "fire_calls": ACT["calls"]})
    ACT["calls"] = 0
txt = CAP.getvalue()
if txt:
    emit("PRINT", txt[:600])
emit("DONE", {})
'''


def run_in_sandbox(files, timeout=TIMEOUT):
    """Run DRIVER with the given files in a fresh sandboxed process (as A3). Returns (stdout, stderr, timed_out, rc)."""
    with tempfile.TemporaryDirectory(prefix="rpp_g1c2_") as d:
        for name, content in files.items():
            Path(d, name).write_text(content, encoding="utf-8")
        main = Path(d, "main.py")
        main.write_text(DRIVER, encoding="utf-8")
        env = {"PYTHONIOENCODING": "utf-8", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        p = subprocess.Popen([sys.executable, "-I", str(SANDBOX), d, str(main)], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=d, env=env, text=True, encoding="utf-8", errors="replace")
        try:
            out, err = p.communicate(timeout=timeout)
            return out or "", err or "", False, p.returncode
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            return out or "", err or "", True, p.returncode


# ---------------------------------------------------------------------------------------------------------------------
# Fenced blocks
FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*([A-Za-z_][\w-]*)?\s*$")


def fenced_blocks(text):
    """[(label, body)] for fenced blocks; tolerates CRLF, ~~~ fences and a missing closing fence."""
    out, cur, fence, label = [], None, None, None
    for ln in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = FENCE_RE.match(ln)
        if cur is None:
            if m:
                fence, label, cur = m.group(1)[:3], (m.group(2) or "").lower(), []
            continue
        if m and m.group(1)[:3] == fence and not m.group(2):
            out.append((label, "\n".join(cur)))
            cur = None
            continue
        cur.append(ln)
    if cur is not None:
        out.append((label, "\n".join(cur)))
    return out


def last_block(text, label):
    bs = [b for lab, b in fenced_blocks(text) if lab == label and b.strip()]
    return bs[-1] if bs else None


# ---------------------------------------------------------------------------------------------------------------------
# Ledger checker (A1 parsing approach)
ELEMENT_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(?:\*\*)?(?:([A-Z])(\d+))?\b")
HEAD_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$")
REF_RE = re.compile(r"\b([QEMCBDW])(\d+)\b")
STATED_Q_RE = re.compile(r"\[\s*stated\s*:[^\]]*\]", re.I)
DERIVED_RE = re.compile(r"\[\s*derived\s*:([^\]]*)\]", re.I)
NOT_SIM_RE = re.compile(r"\[\s*not\s+simulated\b[^\]]*\]", re.I)
QUAL_RE = re.compile(r"\[\s*qualitative\b[^\]]*\]", re.I)
BACKTICK_RE = re.compile(r"`([^`\s]+)`")
SECTIONS = ["QUESTION TARGETS", "ENTITIES", "MECHANISMS", "COUPLINGS", "BINDING CONSTRAINTS", "DISTRACTORS", "TIMING"]
CODE_TAG_RE = re.compile(r"#\s*(M\d+)\b")
FIRE_RE = re.compile(r"""\bfire\(\s*["'](M\d+)["']""")


def parse_ledger(text):
    """-> dict(elements=[{id, kind, line, text, tags, not_sim, qualitative}], holes=[...], sections=[...], ...)."""
    lines = (text or "").replace("\r\n", "\n").split("\n")
    els, holes, sections, seen = [], [], [], {}
    counts = {"stated": 0, "derived": 0, "assumed": 0}
    for i, ln in enumerate(lines, 1):
        h = HEAD_RE.match(ln)
        if h and not ELEMENT_RE.match(ln):
            sections.append(h.group(1).strip().upper())
            continue
        m = ELEMENT_RE.match(ln)
        if not m:
            continue
        body = ln.strip()[:160]
        if not m.group(1):
            holes.append(f"ledger line {i}: element without an id (start it with Q1, E1, M1, C1, B1, D1 or W1): {body}")
            continue
        eid = m.group(1) + m.group(2)
        tags = [t.lower() for t in TAG_RE.findall(ln)]
        el = {"id": eid, "kind": m.group(1), "line": i, "text": ln.strip(), "tags": tags,
              "not_sim": bool(NOT_SIM_RE.search(ln)), "qualitative": bool(QUAL_RE.search(ln))}
        if eid in seen:
            holes.append(f"ledger line {i}: duplicate id {eid} (first on line {seen[eid]})")
        else:
            seen[eid] = i
        if el["kind"] != "Q":
            if not tags:
                holes.append(f"ledger line {i}: {eid} has no provenance tag ([stated: \"quote\"], [derived: from ids] "
                             f"or [assumed: reason; range lo-hi])")
            elif len(tags) > 1:
                holes.append(f"ledger line {i}: {eid} has {len(tags)} provenance tags (exactly one is required)")
        for t in tags[:1]:
            counts[t] += 1
        els.append(el)
    ids = {e["id"] for e in els}
    for e in els:
        scan = []
        if e["kind"] in ("C", "B"):
            scan.append(STATED_Q_RE.sub(" ", e["text"]))
        scan += DERIVED_RE.findall(e["text"])
        refs = sorted({a + b for s in scan for a, b in REF_RE.findall(s)} - {e["id"]})
        for r in refs:
            if r not in ids:
                holes.append(f"ledger line {e['line']}: {e['id']} refers to {r}, which is not in the ledger")
    q = [e for e in els if e["kind"] == "Q"]
    mech = [e for e in els if e["kind"] == "M"]
    if not q:
        holes.append("ledger: no QUESTION TARGETS (add Q1, Q2 ... lines naming what the question asks for)")
    if not mech:
        holes.append("ledger: no MECHANISMS (M1, M2 ... lines)")
    missing = [s for s in SECTIONS if not any(s in x for x in sections)]
    return {"elements": els, "holes": holes, "sections": sections, "missing_sections": missing, "counts": counts,
            "ids": ids, "q": q, "mech": mech, "untagged": any("no provenance tag" in h for h in holes)}


def code_tags(code):
    """{M id: [code line numbers]} from `# Mk` comments; plus the set of ids with a literal fire("Mk") call."""
    tags = {}
    for i, ln in enumerate((code or "").splitlines(), 1):
        for mid in CODE_TAG_RE.findall(ln):
            tags.setdefault(mid, []).append(i)
    return tags, set(FIRE_RE.findall(code or ""))


def declarations(code):
    """Top-level literal PARAMS / RULES / HORIZON -> (params or None, rules or None, horizon or None, warnings)."""
    out = {"PARAMS": None, "RULES": None, "HORIZON": None}
    warns = []
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return None, None, None, []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
            if nm in out:
                try:
                    out[nm] = ast.literal_eval(node.value)
                except Exception:
                    warns.append(f"{nm} is not a literal (only literal values can be listed and overridden)")
                    out[nm] = False
    p, r, h = out["PARAMS"], out["RULES"], out["HORIZON"]
    if p is None:
        warns.append("no PARAMS dict declared: parameter overrides are impossible")
    elif p is not False and not isinstance(p, dict):
        warns.append("PARAMS must be a dict literal")
        p = False
    if r is None:
        warns.append("no RULES dict declared: rule overrides are impossible")
    elif r is not False and not isinstance(r, dict):
        warns.append("RULES must be a dict literal")
        r = False
    if h is None:
        warns.append(f"no HORIZON declared: default steps = {DEFAULT_HORIZON}")
    elif h is not False and (not isinstance(h, int) or isinstance(h, bool) or h < 0):
        warns.append("HORIZON must be a non-negative int literal")
        h = False
    return (p if isinstance(p, dict) else None, r if isinstance(r, dict) else None,
            h if isinstance(h, int) and not isinstance(h, bool) else None, warns)


def uses_rng(code):
    """Static guess (the driver decides by argument count): does Model.step take an rng argument?"""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == "Model":
            for f in n.body:
                if isinstance(f, ast.FunctionDef) and f.name == "step":
                    return len(f.args.args) >= 4
    return False


# ---------------------------------------------------------------------------------------------------------------------
# Formatting helpers
def _f(v, sig=3):
    """Compact number: integers as such, otherwise `sig` significant figures (3 in SERIES, 6 in FINAL/EFFECT)."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            return str(v)
        if float(v) == int(v) and abs(v) < 1e9:
            return str(int(v))
        s = f"{v:.{sig}g}"
        if "e" in s and abs(v) < 1e9:          # avoid 1.78e+03 for ordinary magnitudes
            s = f"{round(v, sig - len(str(int(abs(v))))):.0f}" if abs(v) >= 1 else s
        return s
    s = json.dumps(v)
    return s if len(s) <= 240 else s[:240] + '..."'


def _F(v):
    return _f(v, 6)


def _S(v):
    return _f(v, 4)


def summarize(vals):
    """A4's seeded summary (mean, sd, quantiles; P(true) for bools; shares for categories), without e-notation."""
    ok = [v for v in vals if v is not None]
    nums = [float(v) for v in ok if _num(v)]
    if not ok or len(nums) != len(ok) or all(isinstance(v, bool) for v in ok):
        return _summarize_a4(vals)
    xs = sorted(nums)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
    q = lambda p: _quant(xs, p)
    return (f"mean {_S(m)} sd {_S(sd)} | p05 {_S(q(.05))} p25 {_S(q(.25))} p50 {_S(q(.5))} p75 {_S(q(.75))} "
            f"p95 {_S(q(.95))} | min {_S(xs[0])} max {_S(xs[-1])} (n={len(xs)})")


def _short(o, limit=300):
    s = json.dumps(o, separators=(",", ":"), default=str)
    return s if len(s) <= limit else s[:limit] + "..."


def _rle(vals):
    out, i = [], 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        n = j - i + 1
        out.append(f"{vals[i]}x{n}" if n >= 3 else " ".join([vals[i]] * n))
        i = j + 1
    return " ".join(out)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def shape(xs, smooth=False):
    """Shape summary of a mean trajectory (list of floats or None)."""
    pts = [(t, v) for t, v in enumerate(xs) if v is not None]
    if not pts:
        return None
    vs = [v for _, v in pts]
    v0 = vs[0]
    rng = max(vs) - min(vs)
    tol = 1e-9 * max(1.0, abs(rng), abs(v0))
    tmx = max(pts, key=lambda p: p[1])
    tmn = min(pts, key=lambda p: p[1])
    first = next((t for t, v in pts if abs(v - v0) > tol), None)
    ys = vs
    if smooth and len(vs) >= 5:
        ys = [sum(vs[max(0, i - 1):i + 2]) / len(vs[max(0, i - 1):i + 2]) for i in range(len(vs))]
    signs = []
    for a, b in zip(ys, ys[1:]):
        d = b - a
        if abs(d) > max(tol, 1e-6 * rng):
            s = 1 if d > 0 else -1
            if not signs or signs[-1] != s:
                signs.append(s)
    if not signs:
        cls = "constant"
    elif len(signs) == 1:
        cls = "monotone rising" if signs[0] > 0 else "monotone falling"
    elif len(signs) == 2:
        cls = "single peak" if signs[0] > 0 else "single trough"
    else:
        cls = f"oscillating ({len(signs) - 1} reversals)"
    return {"start": v0, "end": vs[-1], "max": tmx[1], "t_max": tmx[0], "min": tmn[1], "t_min": tmn[0],
            "class": cls, "first_change": first}


def _pick_points(n, keys_series):
    """Indices for SERIES: all when n <= 61, else 60 points incl. each shown key's extremes and the final step."""
    if n <= SERIES_POINTS + 1:
        return list(range(n))
    must = {0, n - 1}
    for xs in keys_series:
        pts = [(t, v) for t, v in enumerate(xs) if v is not None]
        if pts:
            must.add(max(pts, key=lambda p: p[1])[0])
            must.add(min(pts, key=lambda p: p[1])[0])
    rest = max(0, SERIES_POINTS - len(must))
    uni = {round(i * (n - 1) / max(1, rest - 1)) for i in range(rest)} if rest else set()
    idx = sorted(must | uni)
    while len(idx) > SERIES_POINTS:   # drop uniform points (never the extremes or the final step)
        for k in range(1, len(idx) - 1):
            if idx[k] not in must:
                idx.pop(k)
                break
        else:
            break
    return idx


def _tool(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


RUN_PROPS = {
    "name": {"type": "string", "description": "label of this run"},
    "overrides": {"type": "object", "description": "values for declared PARAMS/RULES names, e.g. {\"capacity\": 5}"},
    "steps": {"type": "integer", "description": f"number of steps (default HORIZON, max {MAX_STEPS})"},
    "n_seeds": {"type": "integer", "description": f"seeded runs if step takes rng (default {PROB_SEEDS}, max {MAX_SEEDS})"},
    "seed": {"type": "integer", "description": "base seed (default 0)"},
    "keys": {"type": "array", "items": {"type": "string"}, "description": "observation keys to show first"},
    "compare": {"type": "boolean", "description": "report differences against the baseline (default true)"},
    "what_if_patch": {"type": "object", "description": "{old, new}: a code change for this run only (the stored "
                      "model is unchanged)", "properties": {"old": {"type": "string"}, "new": {"type": "string"}}},
}
RUN_SCHEMA = {"type": "object", "properties": RUN_PROPS}
RUNS_PROP = {"type": "array", "items": RUN_SCHEMA, "description": f"up to {MAX_RUNS} named runs (what-ifs); "
             "each is compared with the baseline"}

TOOLS = [
    _tool("submit_model", "Submit the ledger and/or the simulation. Takes `ledger` and `source` from the arguments, "
          "or else from the last ```ledger and ```model fenced blocks of this response (an omitted part keeps the "
          "stored one). Checks both, then runs the baseline plus `runs` and returns CHECK, FINAL, EFFECT, SHAPE, "
          "MECHANISM ACTIVITY and SERIES in one result. A ledger-only submission is allowed for qualitative questions.",
          {"ledger": {"type": "string"}, "source": {"type": "string"}, "runs": RUNS_PROP}),
    _tool("patch_model", "Replace one exact, unique occurrence of `old` with `new` in one part (`ledger` or `model`); "
          "re-runs the baseline and `runs` if the model changed.",
          {"part": {"type": "string", "enum": ["ledger", "model"]}, "old": {"type": "string"},
           "new": {"type": "string"}, "runs": RUNS_PROP}, ["part", "old", "new"]),
    _tool("run_scenario", "Run the stored model: one run (its fields) or `runs` (up to 4), each with differences "
          "against the baseline. Does not change the model.", {**RUN_PROPS, "runs": RUNS_PROP}),
    _tool("show_model", "Show the ledger and the code with line numbers, plus the coverage table.",
          {"part": {"type": "string", "enum": ["ledger", "model", "both"]}}),
]
MUTATING = ("submit_model", "patch_model")
BUDGET_LINE = "[budget: {k} of {n} responses used; your last response must be the FINAL ANSWER without tool calls]"


class G1C2Arm(Arm):
    name = "G1C2"
    buildable = True
    executable = True
    guide_file = "g1c2_guide.md"
    mutating_tools = MUTATING
    custom_episode = True

    def guide(self):
        p = ROOT / "tournament" / "prompts" / self.guide_file
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def tools(self):
        return copy.deepcopy(TOOLS)

    def probe_tools(self):
        return [t for t in self.tools() if t["function"]["name"] in ("run_scenario", "show_model")]

    def current_probe_tools(self, state):
        # without a simulation the probe reads the ledger (model-mediated, as A1)
        return self.probe_tools() if self._code(state).strip() else []

    def gold_tools(self, state=None):
        ts = [t for t in self.tools() if not self.is_mutating(t["function"]["name"])]
        if state is not None and not self._code(state).strip():
            ts = [t for t in ts if t["function"]["name"] != "run_scenario"]
        return ts

    # ------------------------------------------------------------------ state
    def new_state(self):
        return ArmState(rep={"ledger": "", "source": ""})

    def load(self, rep, given_gold=False):
        st = self.new_state()
        if isinstance(rep, str):
            rep = {"source": rep} if "class Model" in rep else {"ledger": rep}
        if isinstance(rep, dict):
            led = rep.get("ledger")
            src = rep.get("source", rep.get("code"))
            st.rep = {"ledger": led if isinstance(led, str) else "", "source": src if isinstance(src, str) else ""}
        st.given_gold = given_gold
        return st

    def export(self, state):
        code = self._code(state)
        p, r, h, _ = declarations(code)
        return {"ledger": self._ledger(state), "source": code, "params": p, "rules": r, "horizon": h}

    def strip_answers(self, rep):
        rep = strip_answer_fields(copy.deepcopy(rep if isinstance(rep, dict) else {"source": rep}))
        led = rep.get("ledger")
        if isinstance(led, str):
            out, skip = [], False
            for ln in led.splitlines():
                h = HEAD_RE.match(ln)
                if h and not ELEMENT_RE.match(ln):
                    skip = bool(ANSWER_HEADING.search(h.group(1)))
                if not skip:
                    out.append(ln)
            rep["ledger"] = "\n".join(out)
        return rep

    def _code(self, state):
        c = state.rep.get("source")
        return c if isinstance(c, str) else ""

    def _ledger(self, state):
        c = state.rep.get("ledger")
        return c if isinstance(c, str) else ""

    def _hash(self, state):
        return hashlib.sha1(self._code(state).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ dispatch
    def handle(self, state, name, args):
        try:
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except Exception:
                    return f"{ERROR_PREFIX}: arguments must be a JSON object (hint: see the {name} schema)"
            args = {} if args is None else args
            if not isinstance(args, dict):
                return f"{ERROR_PREFIX}: arguments must be an object (hint: see the {name} schema)"
            fn = getattr(self, "_t_" + str(name), None)
            if fn is None:
                return (f"{ERROR_PREFIX}: unknown tool '{name}' (hint: available: "
                        + ", ".join(t["function"]["name"] for t in self.tools()) + ")")
            if state.given_gold and name in MUTATING:
                return f"{ERROR_PREFIX}: the model is read-only in this mode (hint: use run_scenario, show_model)"
            out = fn(state, args)
        except Exception as e:
            out = f"{ERROR_PREFIX}: internal {type(e).__name__}: {e} (hint: check the arguments)"
        state.log.append((name, {k: (v if k not in ("source", "ledger", "old", "new") else f"<{len(str(v))} chars>")
                                 for k, v in args.items()} if isinstance(args, dict) else args, out[:200]))
        return out

    # ------------------------------------------------------------------ tools
    def _t_submit_model(self, state, args):
        content = state.__dict__.get("_last_content") or ""
        led = args.get("ledger")
        src = args.get("source", args.get("code"))
        got = []
        if not (isinstance(led, str) and led.strip()):
            led = last_block(content, "ledger")
        if not (isinstance(src, str) and src.strip()):
            src = last_block(content, "model")
        if led is None and src is None:
            if not (self._ledger(state).strip() or self._code(state).strip()):
                return (f"{ERROR_PREFIX}: nothing to submit (hint: put the ledger in a ```ledger block and the program "
                        "in a ```model block in the same response, or pass `ledger` / `source`)")
        if isinstance(led, str):
            if len(led) > MAX_LEDGER:
                return f"{ERROR_PREFIX}: ledger too long ({len(led)} chars, max {MAX_LEDGER})"
            state.rep["ledger"] = led
            got.append("ledger")
        if isinstance(src, str):
            if len(src) > MAX_CODE:
                return f"{ERROR_PREFIX}: code too long ({len(src)} chars, max {MAX_CODE})"
            state.rep["source"] = src
            got.append("model")
        head = f"submitted: {', '.join(got) or 'nothing new (stored parts kept)'}"
        return self._check_and_run(state, args.get("runs"), head)

    def _t_patch_model(self, state, args):
        part, old, new = args.get("part"), args.get("old"), args.get("new")
        if part not in ("ledger", "model"):
            return f"{ERROR_PREFIX}: part must be 'ledger' or 'model'"
        if not isinstance(old, str) or not old or not isinstance(new, str):
            return f"{ERROR_PREFIX}: patch_model needs non-empty string 'old' and string 'new'"
        key = "ledger" if part == "ledger" else "source"
        text = state.rep.get(key) or ""
        n = text.count(old)
        if n == 0:
            first = old.strip().splitlines()[0] if old.strip() else old
            close = difflib.get_close_matches(first.strip(), [l.strip() for l in text.splitlines()], n=3, cutoff=0.5)
            hint = ("closest lines: " + " | ".join(close)) if close else "use show_model to copy the exact text"
            return f"{ERROR_PREFIX}: `old` text not found in the {part} (hint: {hint})"
        if n > 1:
            return f"{ERROR_PREFIX}: `old` text occurs {n} times in the {part} (hint: include more surrounding text)"
        new_text = text.replace(old, new, 1)
        if len(new_text) > (MAX_LEDGER if part == "ledger" else MAX_CODE):
            return f"{ERROR_PREFIX}: patched {part} too long; patch not applied"
        state.rep[key] = new_text
        head = f"{part} patched"
        if part == "ledger" and not args.get("runs"):
            return self._check_and_run(state, None, head, run=False)
        return self._check_and_run(state, args.get("runs"), head)

    def _t_run_scenario(self, state, args):
        if not self._code(state).strip():
            return (f"{ERROR_PREFIX}: no simulation yet (the ledger is the model; read it with show_model)"
                    if self._ledger(state).strip() else
                    f"{ERROR_PREFIX}: no model yet (hint: submit_model with a ```model block)")
        runs = args.get("runs")
        if runs is None:
            single = {k: v for k, v in args.items() if k in RUN_PROPS or k == "scenario"}
            runs = [single] if single else []
        return self._check_and_run(state, runs, "run_scenario (stored model unchanged)", check=False)

    def _t_show_model(self, state, args):
        part = args.get("part") or "both"
        if part not in ("ledger", "model", "both"):
            return f"{ERROR_PREFIX}: part must be 'ledger', 'model' or 'both'"
        if not (self._ledger(state).strip() or self._code(state).strip()):
            return f"{ERROR_PREFIX}: nothing written yet (hint: submit_model)"
        return self.render(state, part)[:MAX_OUT]

    # ------------------------------------------------------------------ check
    def _coverage(self, state, led=None):
        led = led or parse_ledger(self._ledger(state))
        code = self._code(state)
        tags, fires = code_tags(code)
        rows, holes = [], []
        mids = [e["id"] for e in led["mech"]]
        for e in led["mech"]:
            if e["id"] in tags:
                st = "simulated" + ("" if e["id"] in fires else " (no fire call)")
            elif e["not_sim"]:
                st = "not simulated (declared)"
            else:
                st = "NOT SIMULATED"
                if code.strip():
                    holes.append(f"coverage: {e['id']} (ledger line {e['line']}) has no `# {e['id']}` tag in the code "
                                 f"(simulate it, or mark the ledger line [not simulated: reason])")
            rows.append((e["id"], e["line"], tags.get(e["id"], []), e["id"] in fires, st))
        for mid, ls in sorted(tags.items(), key=lambda x: int(x[0][1:])):
            if mid not in mids:
                holes.append(f"coverage: code tag # {mid} (model.py line {ls[0]}) has no ledger line")
                rows.append((mid, None, ls, mid in fires, "ORPHAN TAG"))
        return rows, holes, tags, fires

    def _q_holes(self, state, led, observed):
        holes = []
        for q in led["q"]:
            if q["qualitative"]:
                continue
            if observed is None:
                holes.append(f"target {q['id']}: not [qualitative] and no simulation has run (numbers must come from "
                             "a run; add a ```model block, or mark the line [qualitative])")
                continue
            names = BACKTICK_RE.findall(q["text"])
            words = set(re.findall(r"[A-Za-z_][\w.]*", QUAL_RE.sub(" ", q["text"])))
            if not any(n in observed for n in names) and not (words & set(observed)):
                holes.append(f"target {q['id']}: names no observed key (write the key in backticks, e.g. `served`; "
                             f"observed: {', '.join(sorted(observed)[:15])}) and is not [qualitative]")
        return holes

    def _check_lines(self, state, observed):
        led = parse_ledger(self._ledger(state))
        lines = []
        rows, cov_holes, tags, fires = self._coverage(state, led)
        code = self._code(state)
        ledger_holes = led["holes"] if self._ledger(state).strip() else ["ledger: none written (start with the ledger)"]
        q_holes = self._q_holes(state, led, observed) if self._ledger(state).strip() else []
        code_holes, code_errs = (static_check(code) if code.strip() else ([], []))
        _, _, _, warns = declarations(code) if code.strip() else (None, None, None, [])
        allh = ledger_holes + cov_holes + q_holes
        c = led["counts"]
        n_sim = sum(1 for r in rows if r[4].startswith("simulated"))
        summary = (f"ledger {len(led['elements'])} elements (stated {c['stated']}, derived {c['derived']}, "
                   f"assumed {c['assumed']}); targets {len(led['q'])}; mechanisms {len(led['mech'])}"
                   + (f", simulated {n_sim}/{len(led['mech'])}" if code.strip() else ", no simulation"))
        lines.append("CHECK: " + ("OK - " if not allh and not code_errs and not code_holes else "") + summary)
        for h in allh[:25]:
            lines.append("  hole: " + h)
        if len(allh) > 25:
            lines.append(f"  ... {len(allh) - 25} more holes")
        for e in code_errs:
            lines.append("  error: " + e + "\n    hint: fix it with patch_model(part='model')")
        for h in code_holes:
            lines.append("  code hole: " + h)
        for w in warns:
            lines.append("  warning: " + w)
        if led["missing_sections"] and self._ledger(state).strip():
            lines.append("  note: sections not found: " + ", ".join(led["missing_sections"]))
        return lines

    # ------------------------------------------------------------------ running
    def _norm_runs(self, state, runs, params, rules, horizon):
        """-> (list of run dicts for the driver, notes) or (None, error text)."""
        if runs is None:
            runs = []
        if isinstance(runs, str):
            try:
                runs = json.loads(runs)
            except Exception:
                return None, f"{ERROR_PREFIX}: runs must be a list of run objects"
        if isinstance(runs, dict):
            runs = [runs]
        if not isinstance(runs, list):
            return None, f"{ERROR_PREFIX}: runs must be a list of run objects"
        if len(runs) > MAX_RUNS:
            return None, f"{ERROR_PREFIX}: at most {MAX_RUNS} runs per call (got {len(runs)}); split them"
        declared = {**(params or {}), **(rules or {})}
        code = self._code(state)
        prob = uses_rng(code)
        out, notes, names = [], [], set()
        for i, r in enumerate([{"name": "baseline", "_base": True}] + list(runs)):
            if not isinstance(r, dict):
                return None, f"{ERROR_PREFIX}: run {i} must be an object"
            nm = str(r.get("name") or (f"run{i}" if i else "baseline"))[:40]
            if nm in names:
                nm = f"{nm}_{i}"
            names.add(nm)
            ov = r.get("overrides", r.get("scenario")) or {}
            if isinstance(ov, str):
                try:
                    ov = json.loads(ov)
                except Exception:
                    return None, f"{ERROR_PREFIX}: run '{nm}': overrides must be an object"
            if not isinstance(ov, dict):
                return None, f"{ERROR_PREFIX}: run '{nm}': overrides must be an object"
            if declared:
                bad = [k for k in ov if k not in declared]
                if bad:
                    close = {k: difflib.get_close_matches(k, list(declared), n=3, cutoff=0.5) for k in bad}
                    return None, (f"{ERROR_PREFIX}: run '{nm}': unknown override key(s) {bad} (declared PARAMS/RULES: "
                                  f"{', '.join(list(declared)[:60])}"
                                  + "".join(f"; did you mean {c} for {k!r}?" for k, c in close.items() if c) + ")")
            elif ov:
                notes.append(f"note: run '{nm}': no PARAMS/RULES declared; overrides passed into the scenario unchecked")
            try:
                json.dumps(ov)
            except (TypeError, ValueError) as e:
                return None, f"{ERROR_PREFIX}: run '{nm}': overrides are not JSON-serialisable: {e}"
            try:
                steps = int(r.get("steps", horizon if horizon is not None else DEFAULT_HORIZON))
                n_seeds = int(r.get("n_seeds", PROB_SEEDS if prob else DET_SEEDS) or 1)
                seed = int(r.get("seed", 0) or 0)
            except (TypeError, ValueError):
                return None, f"{ERROR_PREFIX}: run '{nm}': steps, n_seeds and seed must be integers"
            if steps < 0:
                return None, f"{ERROR_PREFIX}: run '{nm}': steps must be >= 0"
            if steps > MAX_STEPS:
                notes.append(f"note: run '{nm}': steps capped at {MAX_STEPS}")
                steps = MAX_STEPS
            if n_seeds > MAX_SEEDS:
                notes.append(f"note: run '{nm}': n_seeds capped at {MAX_SEEDS}")
                n_seeds = MAX_SEEDS
            d = {"name": nm, "overrides": ov, "steps": steps, "n_seeds": max(1, n_seeds), "seed": seed,
                 "keys": [str(k) for k in r.get("keys") or [] if isinstance(k, (str, int))][:20]
                 if isinstance(r.get("keys"), list) else [],
                 "compare": r.get("compare", True) is not False, "base": bool(r.get("_base"))}
            wp = r.get("what_if_patch")
            if wp:
                if isinstance(wp, str):
                    try:
                        wp = json.loads(wp)
                    except Exception:
                        wp = None
                if not isinstance(wp, dict) or not isinstance(wp.get("old"), str) or not wp.get("old") \
                        or not isinstance(wp.get("new"), str):
                    return None, f"{ERROR_PREFIX}: run '{nm}': what_if_patch must be {{old, new}} strings"
                c = code.count(wp["old"])
                if c != 1:
                    return None, (f"{ERROR_PREFIX}: run '{nm}': what_if_patch `old` occurs {c} times in the model "
                                  "(it must occur exactly once; use show_model to copy the exact text)")
                d["src"] = code.replace(wp["old"], wp["new"], 1)
                try:
                    ast.parse(d["src"])
                except SyntaxError as e:
                    return None, f"{ERROR_PREFIX}: run '{nm}': patched code does not parse (line {e.lineno}: {e.msg})"
                d["patched"] = True
            out.append(d)
        # the baseline takes its seeds/steps from the first run when unspecified, so effects are paired
        if len(out) > 1:
            b, r1 = out[0], out[1]
            b["steps"], b["n_seeds"], b["seed"] = r1["steps"], r1["n_seeds"], r1["seed"]
            if not b["keys"]:
                b["keys"] = list(dict.fromkeys(k for r in out[1:] for k in r["keys"]))[:20]
        work = sum(r["steps"] * r["n_seeds"] for r in out)
        if work > WORK_CAP:
            f = WORK_CAP / work
            for r in out:
                r["n_seeds"] = max(1, int(r["n_seeds"] * f)) if r["n_seeds"] > 1 else 1
            notes.append(f"note: seeds reduced to {out[0]['n_seeds']} per run to fit the time limit")
        if len({(r['steps'], r['n_seeds'], r['seed']) for r in out}) > 1:
            notes.append("note: runs differ in steps/seeds; effects compare final values of unequal runs")
        return out, notes

    def _check_and_run(self, state, runs, head, run=True, check=True):
        code = self._code(state)
        lines = [head]
        observed = state.__dict__.get("_observed", {}).get(self._hash(state)) if code.strip() else None
        if not code.strip() or not run:
            if check:
                lines += self._check_lines(state, observed)
            lines.append("no simulation yet (ledger only)" if not code.strip() else
                         "model unchanged; not re-run (use run_scenario for runs)")
            return self._assemble(lines, [], [], [], [], [])
        holes, errors = static_check(code)
        if errors or "no class Model" in holes:
            if check:
                lines += self._check_lines(state, None)
            else:
                lines += ["error: " + e for e in errors] + ["code hole: " + h for h in holes]
            lines.insert(0, f"{ERROR_PREFIX}: the model cannot run")
            return "\n".join(lines)[:MAX_OUT]
        params, rules, horizon, _ = declarations(code)
        rs, notes = self._norm_runs(state, runs, params, rules, horizon)
        if rs is None:
            if check:
                lines += self._check_lines(state, observed)
            return notes + ("\n" + "\n".join(lines) if check else "")
        res, exec_notes = self._execute(state, rs)
        base = res.get("baseline")
        ok_base = bool(base) and not base.get("load_err") and not base.get("n_err") \
            and all(n == rs[0]["steps"] + 1 for n in base.get("lens", [])) and bool(base.get("lens"))
        obs_keys = None
        if base and base.get("finals"):
            obs_keys = sorted({k for f in base["finals"] if f for k in f})
            if not any(r.get("patched") for r in rs[:1]) and not rs[0]["overrides"]:
                state.__dict__.setdefault("_observed", {})[self._hash(state)] = obs_keys
        state.__dict__.setdefault("_runs", {})[self._hash(state)] = {"ok": ok_base}
        if check:
            lines += self._check_lines(state, obs_keys)
        lines += notes + exec_notes
        blocks = self._report(state, rs, res)
        if not base or base.get("load_err") or not any(base.get("lens") or [0]):
            lines.insert(0, f"{ERROR_PREFIX}: the baseline produced no observations")
        return self._assemble(lines, *blocks)

    def _execute(self, state, rs):
        files = {"model.py": self._code(state), "params.json": json.dumps({"runs": rs, "max_sum_keys": MAX_SUM_KEYS})}
        out, err, timed_out, rc = run_in_sandbox(files, TIMEOUT)
        recs = parse_lines(out)
        done = any(t == "DONE" for t, _ in recs)
        notes = []
        if timed_out:
            notes.append(f"[timed out after {TIMEOUT}s; partial results shown]\n    hint: reduce steps, seeds or runs, "
                         "or remove unbounded loops")
        elif not done:
            tail = err.strip().splitlines()[-3:] if err.strip() else []
            msg = "memory limit exceeded" if "memory limit" in err else f"process ended unexpectedly (exit code {rc})"
            notes.append(f"[{msg}; partial results shown]" + ("\n    " + "\n    ".join(tail) if tail else ""))
        prints = [b for t, b in recs if t == "PRINT"]
        if prints:
            notes.append("model printed: " + prints[0][:300].replace("\n", " | "))
        res = {b["name"]: b for t, b in recs if t == "RUN"}
        return res, notes

    # ------------------------------------------------------------------ report blocks
    def _report(self, state, rs, res):
        led = parse_ledger(self._ledger(state))
        base = res.get("baseline") or {}
        final, effect, shp, act, ser = [], [], [], [], []
        base_mean = base.get("mean") or {}
        for r in rs:
            x = res.get(r["name"])
            hdr = (f"[{r['name']}] overrides {_short(r['overrides'], 200)}" + (" + ephemeral patch applied"
                   if r.get("patched") else "") + f", steps {r['steps']}")
            if x is None:
                final.append(hdr + ": no result (the process stopped before this run)")
                continue
            if x.get("load_err"):
                final.append(hdr + f"\n  ERROR: {fmt_err(x['load_err'])}")
                continue
            prob = x.get("prob")
            n = x.get("n_seeds", 1)
            lens = x.get("lens") or []
            full = [f for f, ln in zip(x.get("finals") or [], lens) if f and ln == r["steps"] + 1]
            hdr += f", seeds {n} (completed {len(full)})" if prob else ", deterministic"
            final.append(hdr)
            for i, e in (x.get("errs") or [])[:1]:
                who = f" (seed {i}; {x.get('n_err', 1)} of {n} seeds failed)" if prob else ""
                final.append(f"  ERROR{who}: " + fmt_err(e)
                             + f"\n    (observations up to t={max(lens) - 1 if lens else 0})")
            finals = full or [f for f in (x.get("finals") or []) if f]
            keys = list(dict.fromkeys(r["keys"] + sorted({k for f in finals for k in f})))
            if r["keys"]:
                miss = [k for k in r["keys"] if not any(k in f for f in finals)]
                if miss:
                    final.append(f"  note: keys never observed: {miss}")
            shown = keys[:40]
            if prob:
                final += [f"  {k}: {summarize([f.get(k) for f in finals])}" for k in shown]
            else:
                f0 = finals[-1] if finals else {}
                final.append("  " + ", ".join(f"{k}={_F(f0.get(k))}" for k in shown))
            if len(keys) > 40:
                final.append(f"  ... {len(keys) - 40} more keys (use keys=[...])")
            # EFFECT vs baseline
            if not r["base"] and r["compare"] and base.get("finals"):
                effect += self._effect(r, x, base, keys)
            # SHAPE
            mean = x.get("mean") or {}
            nk = [k for k in keys if k in mean]
            if not r["base"]:
                nk = [k for k in nk if mean.get(k) != base_mean.get(k)]
            const = []
            for k in nk[:30 if r["base"] else 10]:
                s = shape(mean[k], smooth=bool(prob))
                if s and s["class"] == "constant":
                    const.append((k, s["start"]))
                elif s:
                    shp.append(f"  [{r['name']}] {k}: {s['class']}; start {_S(s['start'])}, end {_S(s['end'])}, "
                               f"max {_S(s['max'])} at t={s['t_max']}, min {_S(s['min'])} at t={s['t_min']}, "
                               + (f"first change t={s['first_change']}" if s["first_change"] is not None
                                  else "never changes"))
            if const:
                shp.append(f"  [{r['name']}] constant: " + ", ".join(f"{k}={_S(v)}" for k, v in const))
            nk = [k for k in nk if k not in {c for c, _ in const}]       # SERIES: constant keys add nothing
            # ACTIVITY
            act += self._activity(state, led, r, x)
            # SERIES
            sk = (nk[:SERIES_KEYS] if r["base"] else nk[:5])
            if sk:
                n_t = max(len(mean[k]) for k in sk)
                idx = _pick_points(n_t, [mean[k] for k in sk])
                ser.append(f"  [{r['name']}] t: " + (f"0..{n_t - 1} (every step)" if len(idx) == n_t
                                                      else " ".join(str(i) for i in idx)))
                for k in sk:
                    xs = mean[k]
                    ser.append(f"  {k}: " + _rle([_f(xs[i]) if i < len(xs) else "-" for i in idx]))
        return final, effect, shp, act, ser

    def _effect(self, r, x, base, keys):
        out = [f"  [{r['name']}] vs baseline:"]
        bf, rf = base.get("finals") or [], x.get("finals") or []
        changed, same = [], []
        for k in keys[:60]:
            pairs = [(b.get(k), v.get(k)) for b, v in zip(bf, rf) if b and v and k in b and k in v]
            if not pairs:
                continue
            if all(_num(a) and _num(b) for a, b in pairs):
                d = sorted(float(b) - float(a) for a, b in pairs)
                if all(abs(v) < 1e-12 for v in d):
                    same.append(k)
                    continue
                if len(d) == 1:
                    changed.append(f"    {k}: {_F(pairs[0][0])} -> {_F(pairs[0][1])} (diff {_F(d[0])})")
                else:
                    m = sum(d) / len(d)
                    q = lambda p: d[min(len(d) - 1, max(0, round(p * (len(d) - 1))))]
                    changed.append(f"    {k}: mean diff {_F(m)} (p05 {_F(q(.05))}, p95 {_F(q(.95))}, "
                                   f"P(diff>0)={sum(1 for v in d if v > 0) / len(d):.2f}, paired n={len(d)})")
            elif all(isinstance(a, bool) and isinstance(b, bool) for a, b in pairs):
                pa = sum(a for a, _ in pairs) / len(pairs)
                pb = sum(b for _, b in pairs) / len(pairs)
                if pa == pb and all(a == b for a, b in pairs):
                    same.append(k)
                elif len(pairs) == 1:
                    changed.append(f"    {k}: {_F(pairs[0][0])} -> {_F(pairs[0][1])}")
                else:
                    changed.append(f"    {k}: P(true) {pa:.2f} -> {pb:.2f} (diff {pb - pa:+.2f})")
            else:
                if all(a == b for a, b in pairs):
                    same.append(k)
                else:
                    changed.append(f"    {k}: {_F(pairs[0][0])} -> {_F(pairs[0][1])}"
                                   + (f" (seed 0; {sum(1 for a, b in pairs if a != b)}/{len(pairs)} seeds differ)"
                                      if len(pairs) > 1 else ""))
        if not changed:
            out.append("    no key changed (all paired differences 0)")
        out += changed
        if same and changed:
            out.append("    unchanged: " + ", ".join(same[:30]) + (" ..." if len(same) > 30 else ""))
        return out

    def _activity(self, state, led, r, x):
        acts = x.get("acts") or []
        tags, fires = code_tags(r.get("src") or self._code(state))
        out = [f"  [{r['name']}]"]
        if not fires and not x.get("fire_calls"):
            out[0] += " no fire calls (activity unavailable: call fire(\"M<k>\", amount) when a mechanism acts)"
            return out
        mids = sorted({k for a in acts for k in a}, key=lambda m: (len(m), m))
        ns = len(acts)
        steps = r["steps"]
        for mid in mids:
            rows = [a[mid] for a in acts if mid in a]
            first = min(rw[1] for rw in rows)
            last = max(rw[2] for rw in rows)
            ft = "init" if first < 0 else f"t={first}"
            lt = "init" if last < 0 else f"t={last}"
            if ns <= 1:
                rw = rows[0]
                out.append(f"    {mid}: fired at {rw[0]} of {steps} steps, first {ft}, last {lt}, total {_F(rw[3])}")
            else:
                mn = sum(rw[0] for rw in rows) / ns
                tot = sum(rw[3] for rw in rows) / ns
                out.append(f"    {mid}: fired in {len(rows)}/{ns} seeds; mean {_S(mn)} of {steps} steps, earliest "
                           f"{ft}, latest {lt}, mean total {_F(tot)}")
        ledger_m = [e["id"] for e in led["mech"]]
        never = [m for m in sorted(set(tags) | set(ledger_m), key=lambda m: (len(m), m))
                 if m in tags and m not in mids]
        if never:
            out.append("    never fired: " + ", ".join(never))
        nofire = [m for m in tags if m not in fires and m not in mids]
        if nofire:
            out.append("    tagged but no fire(\"Mk\") call in the code: " + ", ".join(sorted(nofire)))
        unk = [m for m in mids if ledger_m and m not in ledger_m]
        if unk:
            out.append("    fired but not in the ledger: " + ", ".join(unk))
        return out

    def _assemble(self, head, final, effect, shp, act, ser):
        """CHECK/header, FINAL and EFFECT first; SHAPE, ACTIVITY, SERIES lines are dropped from the end if needed."""
        must = list(head)
        if final:
            must += ["FINAL (last step):"] + final
        if effect:
            must += ["EFFECT vs baseline:"] + effect
        text = "\n".join(must)
        if len(text) > MAX_OUT - 200:
            text = text[:MAX_OUT - 260] + "\n[... FINAL/EFFECT shortened: request fewer keys]"
        omitted = []
        for title, blk in (("SHAPE", shp), ("MECHANISM ACTIVITY", act), ("SERIES", ser)):
            if not blk:
                continue
            add = "\n" + title + ":"
            if len(text) + len(add) + sum(len(x) + 1 for x in blk[:2]) + 120 > MAX_OUT:
                omitted.append(title)
                continue
            text += add
            for i, ln in enumerate(blk):
                if len(text) + len(ln) + 1 + 120 > MAX_OUT:
                    omitted.append(f"{title} ({len(blk) - i} of {len(blk)} lines)")
                    break
                text += "\n" + ln
        if omitted:
            text += "\n[output limit: omitted " + "; ".join(omitted) + "; use keys=[...] or fewer runs]"
        return text

    # ------------------------------------------------------------------ render / validity
    def render(self, state, part="both"):
        out = []
        led, code = self._ledger(state), self._code(state)
        if part in ("ledger", "both"):
            out.append("LEDGER" + ("" if led.strip() else ": (none)"))
            out += [f"{i:3d}| {ln}" for i, ln in enumerate(led.splitlines(), 1)]
        if part in ("model", "both"):
            out.append("MODEL (model.py)" + ("" if code.strip() else ": (no simulation)"))
            out += [f"{i:4d}| {ln}" for i, ln in enumerate(code.splitlines(), 1)]
        if part == "both" or (part == "model" and code.strip()):
            rows, holes, _, _ = self._coverage(state)
            if rows:
                out.append("COVERAGE (ledger mechanism -> code):")
                for mid, ll, cl, fi, st in rows:
                    out.append(f"  {mid}: ledger line {ll if ll else '-'}; code lines "
                               f"{', '.join(map(str, cl)) or '-'}; fire call {'yes' if fi else 'no'}; {st}")
        return "\n".join(out)[:60000]

    def validity(self, state):
        led_text, code = self._ledger(state), self._code(state)
        led = parse_ledger(led_text)
        holes = list(led["holes"]) if led_text.strip() else ["no ledger"]
        _, cov_holes, _, _ = self._coverage(state, led)
        holes += cov_holes
        errors = []
        ledger_ok = bool(led["mech"]) and not led["untagged"]
        code_holes, code_errs = static_check(code) if code.strip() else (["no simulation"], [])
        holes += code_holes
        errors += code_errs
        code_usable = bool(code.strip()) and not code_errs and "no class Model" not in code_holes
        executable = False
        h = self._hash(state)
        if code_usable:
            runs = state.__dict__.get("_runs", {})
            if h in runs and runs[h]["ok"] and h in state.__dict__.get("_observed", {}):
                executable = True
            else:
                executable, err = self._smoke(state)
                if err:
                    errors.append(err)
        if led_text.strip():
            holes += self._q_holes(state, led, state.__dict__.get("_observed", {}).get(h) if code_usable else None)
        return {"usable": ledger_ok or code_usable, "executable": executable, "holes": holes, "errors": errors,
                "stats": {"elements": len(led["elements"]), **led["counts"], "mechanisms": len(led["mech"]),
                          "targets": len(led["q"])}}

    def _smoke(self, state):
        rs = [{"name": "baseline", "overrides": {}, "steps": 2, "n_seeds": 2, "seed": 0, "keys": [],
               "compare": False, "base": True}]
        res, notes = self._execute(state, rs)
        b = res.get("baseline")
        if not b:
            return False, (notes[0] if notes else "smoke run produced no result")
        if b.get("load_err"):
            return False, fmt_err(b["load_err"])
        if b.get("errs"):
            return False, fmt_err(b["errs"][0][1])
        if notes and any(n.startswith("[") for n in notes):
            return False, notes[0]
        keys = sorted({k for f in b.get("finals") or [] if f for k in f})
        if keys:
            state.__dict__.setdefault("_observed", {})[self._hash(state)] = keys
        return bool(b.get("lens")), None

    # ------------------------------------------------------------------ episode (document channel)
    def run_episode(self, ep, state, messages):
        from tournament.episode import FORCE_FINAL_MSG, MAX_TOOL_CALLS, execute_tool_call
        from tournament.run import system_prompt
        messages[0]["content"] = system_prompt(self, True, ep.budget_calls, ep.budget_tokens)
        tools_fn = self.gold_tools if state.given_gold else self.current_tools
        trace, n_tools = [], 0

        def budget():
            return BUDGET_LINE.format(k=len(ep.calls), n=ep.budget_calls)

        while True:
            tools = tools_fn(state) or None
            if tools and not ep.can_continue():
                break
            m = ep.call(messages, tools=tools, label=f"turn{len(ep.calls)}", final=not tools)
            content = m.get("content") or ""
            state.__dict__["_last_content"] = content
            tcs = (m.get("tool_calls") or []) if tools else []
            am = {"role": "assistant", "content": content}
            if not tcs:
                messages.append(am)
                led_b, mod_b = last_block(content, "ledger"), last_block(content, "model")
                new = ((led_b is not None and led_b != self._ledger(state))
                       or (mod_b is not None and mod_b != self._code(state)))
                if tools and not state.given_gold and new and n_tools < MAX_TOOL_CALLS:
                    res = self.handle(state, "submit_model", {})
                    if len(res) > 8000:
                        res = res[:8000] + "\n[... output truncated]"
                    n_tools += 1
                    chars = len(led_b or "") + len(mod_b or "")
                    trace.append({"turn": len(ep.calls), "tool": "submit_model",
                                  "args": {"source": "fenced block", "chars": chars},
                                  "result": res[:3000], "error": res.startswith(ERROR_PREFIX)})
                    messages.append({"role": "user", "content": f"EXECUTOR RESULT (submit_model)\n{res}\n{budget()}"})
                    continue
                state.__dict__.pop("_last_content", None)
                return content, messages, trace
            am["tool_calls"] = m["tool_calls"]
            messages.append(am)
            offered = {t["function"]["name"] for t in tools}
            for tc in tcs:
                name, args, res = execute_tool_call(self, state, tc, offered, n_tools, MAX_TOOL_CALLS)
                n_tools += 1
                trace.append({"turn": len(ep.calls), "tool": name, "args": args, "result": res[:3000],
                              "error": res.startswith(ERROR_PREFIX)})
                messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call{len(trace)}",
                                 "content": res + "\n" + budget()})
        state.__dict__.pop("_last_content", None)
        ep.forced_final = True
        messages.append({"role": "user", "content": FORCE_FINAL_MSG})
        m = ep.call(messages, tools=None, label="forced_final", final=True)
        messages.append({"role": "assistant", "content": m.get("content") or ""})
        return m.get("content") or "", messages, trace


G1C2 = G1C2Arm
