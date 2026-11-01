"""Subprocess helper for A5: run with cwd = candidates/gen1/2 (the language under test).

Reads a JSON job from stdin, writes a JSON result to stdout. Never imported by the harness.
  {"mode": "check", "ast": {...}, "sensitivity": bool}
      -> {"verdict": <full checker verdict>, "usable", "executable", "holes", "reduced": <ast or null>,
          "reduced_verdict": <verdict or null>}
  {"mode": "run", "ast": {...}, "scenario": {...}, "claims": [{"id","time","expr"}], "show": [...]}
      -> {"trajectory": {t: {term: value}}, "claims": {id: holds}, "determinate", "holes", "error"?}
Graceful degradation: when the full model cannot be evaluated, elements named in schema /
well-formedness / compile errors are removed one round at a time (up to MAX_ROUNDS); the removed
elements are reported as holes and the rest is evaluated.
"""
import copy
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())

from checker import check as CK  # noqa: E402
from compiler import expr as X  # noqa: E402
from compiler.to_asp import CompileError  # noqa: E402

MAX_ROUNDS = 6
LISTS = ("sorts", "statics", "fluents", "initial", "actions", "laws", "derived", "constraints", "scenarios", "claims")
ELEM_RE = re.compile(r"\b(law|derived|constraint|action|claim)\s+'?([a-z][a-z0-9_]*)'?")
DECL_RE = re.compile(r"\b(fluent|static|sort)\s+'?([a-z][a-z0-9_]*)'?")
SCEN_RE = re.compile(r"^scenario ([a-z][a-z0-9_]*):")
ANS_RE = re.compile(r"answer\.(\w+) references unknown id '([^']+)'")
KEY = {"law": ("laws", "id"), "derived": ("derived", "id"), "constraint": ("constraints", "id"),
       "action": ("actions", "name"), "claim": ("claims", "id"), "fluent": ("fluents", "name"),
       "static": ("statics", "name"), "sort": ("sorts", "name")}


def normalise(ast):
    ast = copy.deepcopy(ast) if isinstance(ast, dict) else {}
    filled = []
    for k, v in (("rpp_core", "0.1"), ("system", "model"), ("sorts", []), ("fluents", []), ("claims", []),
                 ("answer", {})):
        if k not in ast:
            ast[k] = v
            filled.append(k)
    return ast, filled


def _remove(ast, kind, ident):
    sec, key = KEY[kind]
    before = len(ast.get(sec, []))
    ast[sec] = [e for e in ast.get(sec, []) if not (isinstance(e, dict) and e.get(key) == ident)]
    if kind == "fluent":   # initial entries of a removed fluent
        ast["initial"] = [e for e in ast.get("initial", []) if e.get("fluent") != ident]
    return len(ast[sec]) < before


def _clean_answer(ast):
    ids = {i for i, _, _ in CK.element_ids(ast) if i} | {s.get("id") for s in ast.get("scenarios", [])}
    ans = ast.get("answer") or {}
    for k in ("conclusions", "assumptions", "unexplained"):
        if k in ans:
            ans[k] = [i for i in ans[k] if i in ids]
    if "conflicts" in ans:
        ans["conflicts"] = [[i for i in g if i in ids] for g in ans["conflicts"]]
    ast["answer"] = ans


def degrade_round(ast, verdict):
    """Remove the elements blamed by the verdict's blocking failures. Returns list of holes removed."""
    holes = []
    schema_idx = {}
    for f in verdict.get("failures", []):
        ob, ex = f["obligation"], f["explanation"]
        if ob == "schema":
            path = ex.split(":", 1)[0].split("/")
            if len(path) >= 2 and path[0] in LISTS and path[1].isdigit():
                schema_idx.setdefault(path[0], set()).add(int(path[1]))
            elif path and path[0] and path[0] not in LISTS and "additional" in ex.lower():
                pass
            elif "Additional properties are not allowed" in ex:
                bad = re.findall(r"'([^']+)' was unexpected", ex) or re.findall(r"\('([^']+)'", ex)
                for b in bad:
                    if b in ast and path == [""]:
                        ast.pop(b, None)
                        holes.append({"element": f"top-level field {b}", "problem": ex[:200]})
            continue
        if ob != "well_formed":
            continue
        m = ANS_RE.search(ex)
        if m:
            continue  # handled by _clean_answer
        found = False
        for kind, ident in ELEM_RE.findall(ex):
            if _remove(ast, kind, ident):
                holes.append({"element": f"{kind} {ident}", "problem": ex[:300]})
                found = True
                break
        if found:
            continue
        if "derived fluent" in ex and "no default" in ex:
            n = ex.split("derived fluent", 1)[1].split()[0]
            for fl in ast.get("fluents", []):
                if fl.get("name") == n:
                    fl["default"] = False if fl.get("type") == "bool" else 0
                    holes.append({"element": f"fluent {n}", "problem": "no default; filled with false/0"})
            continue
        if "has no initial value" in ex:
            continue  # evaluated anyway; reported, not removed
        sm = SCEN_RE.match(ex)
        for kind, ident in DECL_RE.findall(ex):
            if kind in ("fluent", "static") and "unknown" in ex:
                continue   # a reference to something undeclared: blame the referencing scenario instead
            if _remove(ast, kind, ident):
                holes.append({"element": f"{kind} {ident}", "problem": ex[:300]})
                found = True
                break
        if not found and sm and sm.group(1) != "base":
            sid = sm.group(1)
            ast["scenarios"] = [s for s in ast.get("scenarios", []) if s.get("id") != sid]
            ast["claims"] = [c for c in ast.get("claims", []) if c.get("about", {}).get("scenario") != sid]
            holes.append({"element": f"scenario {sid}", "problem": ex[:300]})
    for sec, idxs in schema_idx.items():
        lst = ast.get(sec, [])
        for i in sorted(idxs, reverse=True):
            if i < len(lst):
                e = lst.pop(i)
                ident = e.get("id") or e.get("name") or e.get("fluent") if isinstance(e, dict) else None
                holes.append({"element": f"{sec}[{i}]" + (f" ({ident})" if ident else ""),
                              "problem": "does not match the language structure (schema)"})
    _clean_answer(ast)
    return holes


def evaluable(v):
    return bool(v.get("info", {}).get("scenarios"))


def blocking(v):
    return any(f["obligation"] in ("schema", "well_formed") for f in v.get("failures", [])) and not evaluable(v)


def do_check(ast, sensitivity=True):
    ast, filled = normalise(ast)
    v = CK.check(ast, sensitivity=sensitivity)
    out = {"verdict": v, "filled": filled, "holes": [], "reduced": None, "reduced_verdict": None}
    holes = []
    # holes: compile errors confined to some scenarios also make those scenarios holes
    red, rv = ast, v
    rounds = 0
    while rounds < MAX_ROUNDS and (blocking(rv) or any(
            f["obligation"] == "well_formed" and SCEN_RE.match(f["explanation"]) for f in rv.get("failures", []))):
        red = copy.deepcopy(red)
        h = degrade_round(red, rv)
        if not h:
            break
        holes += h
        rv = CK.check(red, sensitivity=False)
        rounds += 1
    if holes:
        out["reduced"], out["reduced_verdict"] = red, rv
    out["holes"] = holes
    out["executable"] = evaluable(rv)
    out["usable"] = out["executable"] or (not blocking(rv) and bool(red.get("fluents")))
    return out


def do_run(ast, sc, claims, show):
    ast, _ = normalise(ast)
    sc = dict(sc or {})
    sc.setdefault("id", "probe_run")
    rclaims = [{"id": c.get("id") or f"q{i}", "status": "assumed",
                "about": {"scenario": sc["id"], "time": c.get("time", 0), "expr": str(c.get("expr", ""))}}
               for i, c in enumerate(claims or [])]
    holes = []
    model = ast
    try:
        models, h = CK.run_scenario(model, sc, rclaims)
    except (CompileError, X.ExprError) as e:
        first = str(e)
        if first.startswith("claim "):
            return {"error": f"query {first}"}
        chk = do_check(ast, sensitivity=False)
        if not chk["reduced"]:
            return {"error": f"the model does not compile: {first}"}
        model, holes = chk["reduced"], chk["holes"]
        try:
            models, h = CK.run_scenario(model, sc, rclaims)
        except Exception as e2:
            return {"error": f"the model does not compile: {first}; after removing broken parts: {e2}",
                    "holes": holes}
    except TimeoutError as e:
        return {"error": f"{e}. Reduce the horizon or value ranges."}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    if not models:
        return {"error": "the model has no consistent trajectory for this scenario (no answer set)", "holes": holes}
    r = CK.parse_atoms(models[0])
    fl = {f["name"]: f for f in model.get("fluents", [])}
    traj = {}
    for t in sorted(r["val"]):
        row = {}
        for term, vals in sorted(r["val"][t].items()):
            base = term.split("(")[0]
            if show and base not in show and term not in show:
                continue
            vs = [CK.display_value(v, fl.get(base)) for v in vals]
            row[term] = vs[0] if len(vs) == 1 else vs
        traj[str(t)] = row
    return {"trajectory": traj, "horizon": h, "determinate": len(models) == 1,
            "claims": {c["id"]: (c["id"] in r["cholds"]) for c in rclaims},
            "violations": [{"constraint": a[0], "t": a[2]} for a in r["viol"]][:20], "holes": holes}


def main():
    job = json.loads(sys.stdin.read())
    real, sys.stdout = sys.stdout, sys.stderr   # library prints must not corrupt the JSON result
    try:
        if job.get("mode") == "run":
            res = do_run(job["ast"], job.get("scenario"), job.get("claims"), job.get("show"))
        else:
            res = do_check(job["ast"], sensitivity=job.get("sensitivity", True))
    finally:
        sys.stdout = real
    sys.stdout.write(json.dumps(res, default=str))


if __name__ == "__main__":
    main()
