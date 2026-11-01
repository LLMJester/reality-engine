"""R++ Core checker: compiles an AST, runs clingo per scenario, discharges the
verification obligations and emits a structured verdict. Deterministic; never
calls an LLM.

Verdict: {pass, failures: [{obligation, claim_ids, timestep, explanation}],
          claims: {id: {scenario, holds, model_value}}, sensitivity: [...], info: {...}}
"""
import json
from decimal import Decimal
import os
import pathlib
import pickle
import queue
import subprocess
import sys
import threading
import time

import clingo
import jsonschema

from compiler import expr as X
from compiler.to_asp import CompileError, compile_parts, decimals_of

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "lang" / "schema.json").read_text(encoding="utf-8"))
STATUSES = {"observed", "asserted", "inferred", "assumed"}
MAX_HORIZON = 200
MAX_RANGE = 2000
SOLVE_TIMEOUT_S = 20
SCENARIO_TIMEOUT_S = float(os.environ.get("RPP_SCENARIO_TIMEOUT", "60"))
MAX_STEP_MODELS = 8
MAX_BRANCH_SOLVES = 2000

OBLIGATIONS = [
    "schema", "well_formed", "determinate", "integrity", "state_consistency",
    "preconditions", "derivation", "prediction", "intervention",
    "assumptions_listed", "observations_explained", "conflicts_surfaced",
]


def _fail(obligation, explanation, claim_ids=(), timestep=None):
    d = {"obligation": obligation, "claim_ids": list(claim_ids), "explanation": explanation}
    if timestep is not None:
        d["timestep"] = timestep
    return d


def _ground_solve(program, facts, limit, deadline):
    """Ground program + facts and return up to `limit` answer sets (0 = all) as symbol lists."""
    ctl = clingo.Control([str(limit), "--warn=none"])
    ctl.add("base", [], program)
    if facts:
        ctl.add("base", [], facts)
    ctl.ground([("base", [])])
    models = []
    with ctl.solve(yield_=True, async_=True) as h:
        while True:
            h.resume()
            if not h.wait(max(0.01, min(SOLVE_TIMEOUT_S, deadline - time.monotonic()))):
                h.cancel()
                raise TimeoutError("clingo solve timeout")
            m = h.model()
            if m is None:
                break
            models.append(list(m.symbols(atoms=True)))
    return models


def solve(program, deadline=None):
    """Monolithic evaluation (the reference semantics); shown atoms of up to 2 answer sets."""
    deadline = deadline or time.monotonic() + SOLVE_TIMEOUT_S
    return [[str(s) for s in m if s.name in SHOWN] for m in _ground_solve(program, "", 2, deadline)]


def solve_stepwise(parts, deadline, progress=None):
    """Evaluate the scenario program along its splitting sequence (SPEC 5.4): ground and solve
    step 0, 1, ..., H, each with the atoms of the previous max-delay steps as facts, then the
    final layer (constraints, claims) over the whole trajectory. Steps with several answer sets
    are branched, so the number of answer sets (capped at 2) equals the monolithic program's."""
    prog = parts.step_program()
    H, D, preds = parts.horizon, parts.max_delay, parts.dyn_preds
    found = []
    budget = [MAX_BRANCH_SOLVES]

    def step_atoms(model, t):
        out = []
        for sym in model:
            if sym.name in preds and sym.arguments:
                last = sym.arguments[-1]
                if last.type == clingo.SymbolType.Number and last.number == t:
                    out.append(f"{sym}.")
        return out

    def rec(t, history):
        if progress:
            progress(t)
        window = [f for st in history[max(0, t - D):t] for f in st]
        budget[0] -= 1
        if budget[0] < 0:
            raise TimeoutError(f"more than {MAX_BRANCH_SOLVES} branch evaluations")
        models = _ground_solve(prog, "\n".join(window) + f"\nnow({t}).", MAX_STEP_MODELS, deadline)
        for mdl in models:
            nh = history + [step_atoms(mdl, t)]
            if t == H:
                found.append(nh)
            else:
                rec(t + 1, nh)
            if len(found) >= 2:
                return

    rec(0, [])
    if not found:
        return []
    facts = "\n".join(f for st in found[0] for f in st)
    fin = _ground_solve(parts.final_program(), facts, 2, deadline)
    if len(fin) != 1:
        raise RuntimeError("final layer is not stratified")
    shown = [str(s) for s in fin[0] if s.name in SHOWN]
    return [shown] + [None] * (len(found) - 1)


SHOWN = {"val", "viol", "multi", "setadd", "badocc", "badval", "cholds", "noinit"}


def parse_atoms(atoms):
    out = {"val": {}, "viol": [], "multi": [], "setadd": [], "badocc": [], "badval": [], "noinit": [],
           "cholds": set()}
    for a in atoms:
        sym = clingo.parse_term(a)
        n, args = sym.name, [str(x) for x in sym.arguments]
        if n == "val":
            out["val"].setdefault(int(args[2]), {}).setdefault(args[0], []).append(args[1])
        elif n == "cholds":
            out["cholds"].add(args[0])
        elif n in out:
            out[n].append(args)
    return out


def element_ids(ast):
    """All (id, kind, status) of elements that can carry status."""
    out = []
    for c in ast.get("claims", []):
        out.append((c.get("id"), "claim", c.get("status")))
    for key in ("laws", "derived", "constraints"):
        for r in ast.get(key, []):
            out.append((r.get("id"), key[:-1] if key != "derived" else "derived", r.get("status")))
    for a in ast.get("actions", []):
        out.append((a.get("name"), "action", a.get("status")))
    for s in ast.get("statics", []):
        for f in s.get("facts", []):
            out.append((f.get("id"), "static_fact", f.get("status")))
    for e in ast.get("initial", []):
        out.append((e.get("id"), "initial", e.get("status")))
    return out


def well_formed(ast):
    fails = []
    ids = [i for i, _, _ in element_ids(ast) if i]
    ids += [s["id"] for s in ast.get("scenarios", [])]
    seen = set()
    for i in ids:
        if i in seen:
            fails.append(_fail("well_formed", f"duplicate id {i!r}"))
        seen.add(i)
    for i, kind, status in element_ids(ast):
        if status == "assumed" and not i:
            fails.append(_fail("well_formed", f"an assumed {kind} has no id, so it cannot be listed"))
    scen_ids = {s["id"] for s in ast.get("scenarios", [])} | {"base"}
    rule_ids = ({r["id"] for k in ("laws", "derived", "constraints") for r in ast.get(k, [])}
                | {a["name"] for a in ast.get("actions", [])} | scen_ids | {"scenario"})
    claim_ids = {c["id"] for c in ast.get("claims", [])}
    for c in ast.get("claims", []):
        if c.get("status") not in STATUSES:
            fails.append(_fail("well_formed", f"claim {c['id']} has invalid status", [c["id"]]))
        sc = c.get("about", {}).get("scenario", "base")
        if sc not in scen_ids:
            fails.append(_fail("well_formed", f"claim {c['id']} refers to unknown scenario {sc!r}", [c["id"]]))
        try:
            X.parse(c.get("about", {}).get("expr", ""))
        except X.ExprError as e:
            fails.append(_fail("well_formed", f"claim {c['id']}: {e}", [c["id"]]))
    ans = ast.get("answer", {}) or {}
    all_ids = set(ids)
    for key in ("conclusions", "assumptions", "unexplained"):
        for i in ans.get(key, []) or []:
            if i not in all_ids:
                fails.append(_fail("well_formed", f"answer.{key} references unknown id {i!r}"))
    for grp in ans.get("conflicts", []) or []:
        for i in grp:
            if i not in all_ids:
                fails.append(_fail("well_formed", f"answer.conflicts references unknown id {i!r}"))
    declared = ([("sort", x.get("name")) for x in ast.get("sorts", [])]
                + [("instance", i) for x in ast.get("sorts", []) for i in x.get("instances", [])]
                + [("fluent", x.get("name")) for x in ast.get("fluents", [])]
                + [("static", x.get("name")) for x in ast.get("statics", [])]
                + [("action", x.get("name")) for x in ast.get("actions", [])])
    for kind, n in declared:
        if n in X.RESERVED:
            what = "the current step" if n in X.TIME_NAMES else "an expression keyword"
            fails.append(_fail("well_formed", f"{kind} name {n!r} is reserved ({what}); rename it"))
    for f in ast.get("fluents", []):
        if f.get("derived") and "default" not in f:
            fails.append(_fail("well_formed", f"derived fluent {f['name']} has no default value"))
    for s in ast.get("sorts", []):
        if "range" in s and s["range"][1] - s["range"][0] > MAX_RANGE:
            fails.append(_fail("well_formed", f"sort {s['name']} range larger than {MAX_RANGE}"))
    for s in ast.get("scenarios", []):
        if int(s.get("horizon", ast.get("horizon", 10))) > MAX_HORIZON:
            fails.append(_fail("well_formed", f"scenario {s['id']} horizon > {MAX_HORIZON}"))
    return fails, rule_ids, claim_ids


def scenarios_of(ast):
    scs = {s["id"]: s for s in ast.get("scenarios", [])}
    scs.setdefault("base", {"id": "base"})
    return scs


def display_value(v, decl):
    """Show a scaled decimal value in real units."""
    d = decimals_of(decl)
    if d and isinstance(v, str) and v.lstrip("-").isdigit():
        return str(Decimal(int(v)).scaleb(-d))
    return v


def claim_value_probe(c):
    """If the claim is 'fluent(args) = literal', return the fluent term string."""
    try:
        e = X.parse(c["about"]["expr"])
    except Exception:
        return None
    if e[0] == "cmp" and e[1] == "=":
        lhs = e[2]
        if lhs[0] == "call" and all(a[0] in ("name", "int") for a in lhs[2]):
            return f"{lhs[1]}({','.join(str(a[1]) for a in lhs[2])})"
        if lhs[0] == "name":
            return lhs[1]
    return None


def _run_inline(ast, sc, claims, mode, timeout, progress=None):
    parts = compile_parts(ast, sc, claims)
    deadline = time.monotonic() + timeout
    if mode == "monolithic" or (mode == "auto" and parts.horizon <= 1):
        # both evaluations give the same answer sets (SPEC 5.4); with at most one transition
        # nothing can accumulate, so the single grounding is cheaper
        return solve(parts.monolithic(), deadline), parts.horizon
    return solve_stepwise(parts, deadline, progress), parts.horizon


def worker_main():
    """Worker loop (python -m checker.worker): pickled jobs on stdin, pickled messages on stdout."""
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything printed by libraries goes to stderr, not into the message stream
    inp = sys.stdin.buffer

    def send(msg):
        pickle.dump(msg, out)
        out.flush()

    while True:
        try:
            job = pickle.load(inp)
        except EOFError:
            return
        ast, sc, claims, mode, timeout = job
        try:
            send(("ok", _run_inline(ast, sc, claims, mode, timeout, lambda t: send(("progress", t)))))
        except (CompileError, X.ExprError) as e:
            send(("compile", str(e)))
        except TimeoutError as e:
            send(("timeout", str(e)))
        except Exception as e:  # clingo errors
            send(("error", f"{type(e).__name__}: {e}"))


class _Pool:
    """One persistent worker process; killed and restarted when a scenario exceeds its budget."""
    proc = None
    queue = None

    @classmethod
    def get(cls):
        if cls.proc is None or cls.proc.poll() is not None:
            cls.proc = subprocess.Popen([sys.executable, "-m", "checker.worker"], cwd=str(ROOT),
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            cls.queue = q = queue.Queue()
            stream = cls.proc.stdout

            def pump():
                while True:
                    try:
                        q.put(pickle.load(stream))
                    except Exception:
                        q.put(("dead", None))
                        return
            threading.Thread(target=pump, daemon=True).start()
        return cls.proc, cls.queue

    @classmethod
    def kill(cls):
        if cls.proc is not None:
            cls.proc.kill()
            cls.proc.wait(5)
        cls.proc = cls.queue = None


def run_scenario(ast, sc, claims, mode="auto", timeout=None, isolate=None):
    """Compile and evaluate one scenario. Returns (models, horizon); models holds the shown atoms
    of the first answer set followed by None for a second one (at most 2 in all).
    With isolation (default), evaluation runs in a worker process that is killed when the
    scenario exceeds its time budget, so grounding blow-ups give a verdict instead of hanging."""
    timeout = timeout or SCENARIO_TIMEOUT_S
    if isolate is None:
        isolate = os.environ.get("RPP_ISOLATE", "1") != "0"
    if not isolate:
        return _run_inline(ast, sc, claims, mode, timeout)
    proc, q = _Pool.get()
    pickle.dump((ast, sc, claims, mode, timeout), proc.stdin)
    proc.stdin.flush()
    end = time.monotonic() + timeout + 5
    step = 0
    while True:
        try:
            kind, payload = q.get(timeout=max(0.01, end - time.monotonic()))
        except queue.Empty:
            _Pool.kill()
            raise TimeoutError(f"exceeded the {timeout:g} s grounding/solving budget at step {step}")
        if kind == "progress":
            step = payload
        elif kind == "ok":
            return payload
        elif kind == "compile":
            raise CompileError(payload)
        elif kind == "timeout":
            raise TimeoutError(f"exceeded the {timeout:g} s grounding/solving budget at step {step} ({payload})")
        elif kind == "dead":
            _Pool.kill()
            raise RuntimeError(f"the solver process died at step {step} (probably out of memory)")
        else:
            raise RuntimeError(payload)


def check(ast, sensitivity=True, timeout=None):
    failures, info, claims_out, sens = [], {}, {}, []
    errs = sorted(jsonschema.Draft202012Validator(SCHEMA).iter_errors(ast), key=lambda e: list(e.path))
    if errs:
        for e in errs[:10]:
            failures.append(_fail("schema", f"{'/'.join(str(p) for p in e.path)}: {e.message[:300]}"))
        return {"pass": False, "failures": failures, "claims": {}, "sensitivity": [], "info": {}}
    wf, rule_ids, claim_ids = well_formed(ast)
    failures += wf
    if wf:
        return {"pass": False, "failures": failures, "claims": {}, "sensitivity": [], "info": {}}

    claims = ast.get("claims", [])
    by_sc = {}
    for c in claims:
        by_sc.setdefault(c.get("about", {}).get("scenario", "base"), []).append(c)
    scs = scenarios_of(ast)
    results = {}
    noinit_seen = set()
    compile_errs = {}
    for sid, sc in scs.items():
        if sid == "base" and sid not in by_sc and len(scs) > 1:
            continue
        try:
            models, h = run_scenario(ast, sc, by_sc.get(sid, []), timeout=timeout)
        except (CompileError, X.ExprError) as e:
            msg = str(e)
            if msg in compile_errs:
                compile_errs[msg].append(sid)
                continue
            compile_errs[msg] = [sid]
            failures.append(_fail("well_formed", f"scenario {sid}: {e}"))
            continue
        except TimeoutError as e:
            failures.append(_fail("well_formed", f"scenario {sid}: {str(e)[:300]}. The model is too large to "
                                                 f"evaluate; reduce the horizon, value ranges or the number of "
                                                 f"fluent references per expression"))
            continue
        except RuntimeError as e:
            failures.append(_fail("well_formed", f"scenario {sid}: clingo error: {str(e)[:300]}"))
            continue
        if len(models) != 1:
            failures.append(_fail("determinate",
                                  f"scenario {sid}: the model has {'no' if not models else 'several'} "
                                  f"answer sets; its rules do not determine a unique trajectory"))
            if not models:
                continue
        r = parse_atoms(models[0])
        results[sid] = (r, h)
        for cid, inst, t in r["viol"]:
            failures.append(_fail("integrity", f"scenario {sid}: constraint {cid} {inst if inst != 'inst' else ''} "
                                               f"violated at t={t}", [], int(t)))
        for f, t in r["multi"]:
            vals = r["val"].get(int(t), {}).get(f, [])
            failures.append(_fail("state_consistency", f"scenario {sid}: {f} has several values {vals} at t={t}",
                                  [], int(t)))
        for f, t in r["setadd"]:
            failures.append(_fail("state_consistency",
                                  f"scenario {sid}: {f} is both set and incremented at t={t}", [], int(t)))
        for f, v, t in r["badval"]:
            failures.append(_fail("state_consistency", f"scenario {sid}: {f} = {v} is outside its type at t={t}",
                                  [], int(t)))
        for (f,) in r["noinit"]:
            if f not in noinit_seen:
                noinit_seen.add(f)
                failures.append(_fail("well_formed", f"fluent {f} has no initial value (conditions on it silently "
                                                     f"fail and adds to it have no effect)"))
        for a, t in r["badocc"]:
            failures.append(_fail("preconditions", f"scenario {sid}: event {a} at t={t} occurs while its "
                                                   f"precondition is false (its effects were not applied)",
                                  [], int(t)))
    for f in failures:
        for msg, sids in compile_errs.items():
            if len(sids) > 1 and f["explanation"] == f"scenario {sids[0]}: {msg}":
                f["explanation"] += f" (same error in scenarios {', '.join(sids[1:])})"
    # claims
    fluents = {f["name"]: f for f in ast.get("fluents", [])}
    for c in claims:
        sid = c.get("about", {}).get("scenario", "base")
        if sid not in results:
            continue
        r, h = results[sid]
        holds = c["id"] in r["cholds"]
        entry = {"scenario": sid, "holds": holds}
        probe = claim_value_probe(c)
        t = c["about"].get("time", 0)
        if probe and isinstance(t, int):
            vals = r["val"].get(t, {}).get(probe)
            decl = fluents.get(probe.split("(")[0])
            vals = [display_value(v, decl) for v in vals] if vals else vals
            entry["model_value"] = (vals[0] if len(vals) == 1 else vals) if vals else None
        claims_out[c["id"]] = entry
    ans = ast.get("answer", {}) or {}
    listed_assumptions = set(ans.get("assumptions", []) or [])
    listed_conflicts = {i for g in (ans.get("conflicts", []) or []) for i in g}
    listed_unexplained = set(ans.get("unexplained", []) or [])
    intervened = {s["id"] for s in ast.get("scenarios", []) if s.get("interventions")}
    candidates = {s["id"] for s in ast.get("scenarios", []) if s.get("candidate")}
    cand_report = {sid: {"reproduces_all_observations": True, "not_reproduced": []} for sid in candidates}
    for c in claims:
        cid, st = c["id"], c.get("status")
        entry = claims_out.get(cid)
        if st == "inferred":
            d = c.get("derivation")
            if not d or not d.get("rule"):
                failures.append(_fail("derivation", f"inferred claim {cid} has no derivation", [cid]))
            else:
                if d["rule"] not in rule_ids:
                    failures.append(_fail("derivation", f"claim {cid}: derivation names unknown rule {d['rule']!r}",
                                          [cid]))
                bad = [p for p in d.get("premises", []) if p not in claim_ids]
                if bad:
                    failures.append(_fail("derivation", f"claim {cid}: unknown premises {bad}", [cid]))
            if entry is not None and not entry["holds"]:
                ob = "intervention" if entry["scenario"] in intervened else "prediction"
                mv = f" (model computes {entry['model_value']})" if entry.get("model_value") else ""
                failures.append(_fail(ob, f"inferred claim {cid} does not follow from the model in scenario "
                                          f"{entry['scenario']}{mv}", [cid]))
        elif st == "observed" and entry is not None and entry["scenario"] in candidates:
            if not entry["holds"]:
                cand_report[entry["scenario"]]["reproduces_all_observations"] = False
                cand_report[entry["scenario"]]["not_reproduced"].append(cid)
        elif st == "observed":
            if entry is not None and not entry["holds"] and cid not in listed_unexplained:
                mv = f" (model computes {entry['model_value']})" if entry.get("model_value") else ""
                failures.append(_fail("observations_explained",
                                      f"observed claim {cid} is not reproduced by the model in scenario "
                                      f"{entry['scenario']}{mv} and is not listed as unexplained", [cid]))
        elif st == "asserted":
            if entry is not None and not entry["holds"] and cid not in listed_conflicts | listed_unexplained:
                mv = f" (model computes {entry['model_value']})" if entry.get("model_value") else ""
                failures.append(_fail("conflicts_surfaced",
                                      f"asserted claim {cid} contradicts the model in scenario "
                                      f"{entry['scenario']}{mv} and is not surfaced as a conflict", [cid]))
    # pairwise direct conflicts between observed/asserted claims
    direct = {}
    for c in claims:
        if c.get("status") not in ("observed", "asserted"):
            continue
        try:
            e = X.parse(c["about"]["expr"])
        except X.ExprError:
            continue
        if e[0] == "cmp" and e[1] == "=" and e[3][0] in ("int", "name", "dec"):
            key = (c["about"].get("scenario", "base"), str(c["about"].get("time", 0)), X.to_text(e[2]))
            lit = str(Decimal(e[3][1]).scaleb(-e[3][2]).normalize()) if e[3][0] == "dec" else str(e[3][1])
            direct.setdefault(key, []).append((c["id"], lit))
    for key, lst in direct.items():
        if len({v for _, v in lst}) > 1:
            ids = [i for i, _ in lst]
            if not any(set(ids) <= set(g) for g in (ans.get("conflicts", []) or [])):
                failures.append(_fail("conflicts_surfaced",
                                      f"claims {ids} assign different values to {key[2]} at t={key[1]} in "
                                      f"scenario {key[0]} and are not listed together as a conflict", ids))
    for i, kind, status in element_ids(ast):
        if status == "assumed" and i and i not in listed_assumptions:
            failures.append(_fail("assumptions_listed", f"assumed {kind} {i} is not listed in answer.assumptions",
                                  [i] if kind == "claim" else []))
    if sensitivity:
        sens = sensitivity_analysis(ast, by_sc, scs, claims_out, timeout=timeout)
    info["scenarios"] = {sid: {"horizon": h} for sid, (r, h) in results.items()}
    if cand_report:
        info["diagnosis_candidates"] = cand_report
    # a fluent with two values makes later claim truth meaningless: mark them indeterminate
    multi_from = {}
    for sid, (r, h) in results.items():
        ts = [int(t) for _, t in r["multi"]]
        if ts:
            multi_from[sid] = min(ts)
    for cid, e in claims_out.items():
        c = next(x for x in claims if x["id"] == cid)
        t = c["about"].get("time", 0)
        if e["scenario"] in multi_from and (not isinstance(t, int) or t >= multi_from[e["scenario"]]):
            e["holds"] = None
    return {"pass": not failures, "failures": failures, "claims": claims_out, "sensitivity": sens, "info": info}


def sensitivity_analysis(ast, by_sc, scs, claims_out, max_facts=6, timeout=None):
    """For assumed static facts with a plausible_range, re-run scenarios at the
    range ends and report conclusion claims whose truth changes."""
    concl = set((ast.get("answer", {}) or {}).get("conclusions", []) or [])
    out = []
    facts = []
    for s in ast.get("statics", []):
        for f in s.get("facts", []):
            if f.get("status") == "assumed" and f.get("plausible_range"):
                facts.append((s["name"], f))
    for name, f in facts[:max_facts]:
        flips = set()
        for val in f["plausible_range"]:
            for sid, cl in by_sc.items():
                targets = [c for c in cl if c["id"] in concl]
                if not targets or sid not in scs:
                    continue
                sc = dict(scs[sid])
                sc["interventions"] = list(sc.get("interventions", [])) + [
                    {"op": "set_static", "static": name, "args": f.get("args", []), "value": val}]
                try:
                    models, _ = run_scenario(ast, sc, targets, timeout=timeout)
                except Exception:
                    continue
                if len(models) != 1:
                    continue
                r = parse_atoms(models[0])
                for c in targets:
                    if (c["id"] in r["cholds"]) != claims_out.get(c["id"], {}).get("holds"):
                        flips.add(c["id"])
        if flips:
            out.append({"assumption": f.get("id"), "static": name, "args": f.get("args", []),
                        "range": f["plausible_range"], "conclusions_that_change": sorted(flips)})
    return out


def main():
    import sys
    ast = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(json.dumps(check(ast), indent=2))


if __name__ == "__main__":
    main()
