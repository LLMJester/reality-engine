"""Tournament runner: arms × models × items, shared effort budget, tool loop, grading, records, resume.

python -m tournament.run --phase P3 --arms A0,A0+,A1,A5,C2,A7 --models small,large,reasoning --split dev
                         [--mode e2e|gold|both] [--double-grade-frac 0.2] [--workers 8] [--phase-cap 5.00]
                         [--reasoning-model deepseek/deepseek-v4-pro] [--dry-run]

Records: runs/tournament/<phase>/<model>/<arm>/<item>.json (given-gold mode: <arm>__gold).
Re-running resumes: graded records are kept, ungraded ones are graded, errored ones re-run.
--phase-cap USD: OpenRouter usage is recorded at start (harness.llm.true_spent); the run stops cleanly
(no new jobs, no new model calls) once spend since start reaches the cap. The global cap in
harness.llm still applies. With several workers, calls already in flight may finish after the
cap is reached (overshoot is bounded by workers × one call).
"""
import argparse
import hashlib
import inspect
import json
import os
import pathlib
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from tournament import records as REC
from tournament.arms import get_arm
from tournament.arms.base import BUDGET_CALLS, BUDGET_OUTPUT_TOKENS
from tournament.episode import Episode, PhaseCapReached, tool_loop

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "tournament"
PROMPTS = ROOT / "harness" / "conditions" / "prompts"
SYSTEM_COMMON = (PROMPTS / "system_common.txt").read_text(encoding="utf-8")
USER_TMPL = (PROMPTS / "c1_user.txt").read_text(encoding="utf-8")   # identical user prompt for every arm (= C1)
MODEL_MAP = {
    "small": "qwen/qwen3-30b-a3b-instruct-2507",
    "large": "qwen/qwen3-235b-a22b-2507",
    "reasoning": os.environ.get("RPP_REASONING_MODEL", "deepseek/deepseek-v4-pro"),
}
SEED = 11
GOLD_SUFFIX = "__gold"
_print = threading.Lock()


def log(*a):
    with _print:
        print(*a, flush=True)


def resolve_model(key, reasoning=None):
    if key == "reasoning" and reasoning:
        return reasoning
    return MODEL_MAP.get(key, key)


def fill(t, **kw):
    for k, v in kw.items():
        t = t.replace("{" + k + "}", str(v))
    return t


def system_prompt(arm, with_tools, budget_calls=BUDGET_CALLS, budget_tokens=BUDGET_OUTPUT_TOKENS):
    s = SYSTEM_COMMON.rstrip()
    g = (arm.guide() or "").strip()
    if g:
        s += "\n\n" + g
    if with_tools:
        s += (f"\n\nWORK BUDGET: at most {budget_calls} responses from you in total (each response may contain "
              f"several tool calls) and {budget_tokens} output tokens. When the budget is used up you will be asked "
              f"for your FINAL ANSWER.")
    return s


def user_prompt(item, ctx, given_text=None):
    u = fill(USER_TMPL, context=ctx, question=item["question"]).rstrip()
    if given_text is None:
        return u
    head, sep, last = u.rpartition("\n\n")
    block = ("GIVEN REPRESENTATION\n====================\n"
             "This representation of the system has already been built and is loaded in your tools. You may query "
             "it with the tools; you cannot rebuild or modify it.\n\n" + given_text)
    return f"{head}\n\n{block}\n\n{last}" if sep else f"{u}\n\n{block}"


class SpendGuard:
    """Phase spend cap measured from the spend function's value at start."""

    def __init__(self, cap, spent_fn):
        self.cap, self.spent_fn = cap, spent_fn
        self.baseline = spent_fn() if spent_fn else 0.0
        self.stopped = False
        self._lock = threading.Lock()

    def phase_spent(self):
        return (self.spent_fn() - self.baseline) if self.spent_fn else 0.0

    def check(self):
        if self.stopped:
            raise PhaseCapReached("phase cap reached")
        if self.cap is None:
            return
        s = self.phase_spent()
        if s >= self.cap:
            with self._lock:
                self.stopped = True
            raise PhaseCapReached(f"phase spend ${s:.4f} reached the cap ${self.cap:.2f}")


# Per-model effort budgets (DISCOVERY D-034): identical for every arm with a given model. Reasoning models get
# a larger token allowance because their hidden reasoning counts as output; calls stay at 6 for everyone.
MODEL_BUDGETS = {"reasoning": {"budget_tokens": 72000, "per_call_max": 24000}}


def run_episode(arm, item, ctx, model_id, client, *, given_rep=None, cap_check=None, purpose="tournament",
                seed=SEED, budget_calls=BUDGET_CALLS, budget_tokens=BUDGET_OUTPUT_TOKENS, per_call_max=None):
    """One item, one arm, one model. Returns the record (without grade)."""
    t0 = time.time()
    given = given_rep is not None
    state = arm.load(given_rep, given_gold=True) if given else arm.new_state()
    if given:
        tools_fn = arm.gold_tools
    else:
        tools_fn = arm.current_tools
    with_tools = (not arm.custom_episode) and bool(tools_fn(state))
    messages = [{"role": "system", "content": system_prompt(arm, with_tools, budget_calls, budget_tokens)},
                {"role": "user", "content": user_prompt(item, ctx, arm.render(state) if given else None)}]
    ekw = {"per_call_max": per_call_max} if per_call_max else {}
    ep = Episode(client, model_id, purpose=purpose, seed=seed, budget_calls=budget_calls,
                 budget_tokens=budget_tokens, cap_check=cap_check, **ekw)
    if arm.custom_episode:
        final, messages, trace = arm.run_episode(ep, state, messages)
    else:
        final, messages, trace = tool_loop(ep, arm, state, messages, tools_fn)
    rec = {"arm": arm.name, "model": model_id, "item": item["id"],
           "system": item.get("system_id") or item["id"], "bank": item.get("bank"),
           "mode": "gold" if given else "e2e", "final_raw": final, "messages": messages, "tool_trace": trace,
           "calls": ep.calls, **ep.summary(), "error": None}
    try:
        rec["rep"] = arm.export(state)
        rec["validity"] = arm.validity(state)
    except Exception as e:
        rec["rep"], rec["validity"] = None, {"usable": False, "executable": False, "holes": [],
                                             "errors": [f"export/validity failed: {e}"]}
    rec["wall_s"] = max(rec.get("wall_s", 0.0), time.time() - t0)
    return rec


def record_path(base, phase, model_key, arm_name, item, given=False):
    return pathlib.Path(base) / phase / model_key / (arm_name + (GOLD_SUFFIX if given else "")) / f"{item}.json"


def wants_double(item, arm, model_key, frac):
    if frac >= 1:
        return True
    if frac <= 0:
        return False
    h = int(hashlib.sha256(f"{item}|{arm}|{model_key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def _is_budget_exceeded(e):
    return type(e).__name__ == "BudgetExceeded"


RESUME_VERSION = "resume-2 (recheck item 3: shared record validation, quarantine of invalid records)"


def quarantine_record(base, phase, path, kind, why):
    """Move an invalid record to <base>/_superseded/invalid/<phase>/<model>/<arm dir>/<item>.<time>.json with a
    .reason.json beside it (never deleted). Returns the new path."""
    path = pathlib.Path(path)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
    dest = pathlib.Path(base) / "_superseded" / "invalid" / phase / path.parent.parent.name / path.parent.name
    dest.mkdir(parents=True, exist_ok=True)
    q = dest / f"{path.stem}.{stamp}.json"
    os.replace(path, q)
    (dest / f"{path.stem}.{stamp}.reason.json").write_text(json.dumps(
        {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "original": str(path), "kind": kind, "reason": why,
         "rule": REC.RECORD_RULES_VERSION, "resume": RESUME_VERSION}, indent=1), encoding="utf-8")
    log(f"QUARANTINED invalid record {path} ({kind}: {why}) -> {q}; regenerating")
    return q


def run_one(job, cfg):
    """job = (item_id, arm_name, model_key, given). Returns the record, or None if skipped."""
    iid, arm_name, model_key, given = job
    guard = cfg["guard"]
    out = record_path(cfg["base"], cfg["phase"], model_key, arm_name, iid, given)
    double = wants_double(iid, arm_name, model_key, cfg.get("double_frac", 0.0))
    rec = None
    if out.exists():
        # RESUME_VERSION: the record is judged by the same rule as the completion check (tournament/records.py).
        # Valid -> kept unchanged; grading-only problem -> answer kept, regraded below; unreadable / identity mismatch /
        # generation error -> the file is quarantined (moved, never deleted) and the job regenerated.
        rec, kind, why = REC.check_file(out, (iid, arm_name, model_key, given), cfg["phase"], cfg.get("double_frac", 0.0))
        action = REC.resume_action(kind, rec)
        if action == "keep":
            return rec
        if action == "regenerate":
            quarantine_record(cfg["base"], cfg["phase"], out, kind, why)
            rec = None
    guard.check()
    item, ctx = cfg["load_item"](iid)
    model_id = resolve_model(model_key, cfg.get("reasoning"))
    if rec is None or rec.get("error"):
        arm = get_arm(arm_name)
        given_rep = None
        if given:
            given_rep = cfg["load_gold"](arm_name, iid)
        try:
            rec = run_episode(arm, item, ctx, model_id, cfg["client"], given_rep=given_rep, cap_check=guard.check,
                              purpose=f"tournament:{cfg['phase']}:{arm_name}{GOLD_SUFFIX if given else ''}:{model_key}",
                              seed=cfg.get("seed", SEED), **MODEL_BUDGETS.get(model_key, {}))
        except PhaseCapReached:
            raise
        except Exception as e:
            if _is_budget_exceeded(e):
                raise
            rec = {"arm": arm_name, "model": model_id, "item": iid, "mode": "gold" if given else "e2e",
                   "error": f"{e}\n{traceback.format_exc()[-1500:]}", "final_raw": "", "cost": 0.0}
        rec.update({"phase": cfg["phase"], "model_key": model_key, "arm_dir": out.parent.name})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    if rec.get("error"):
        return rec
    guard.check()
    grade_fn = cfg["grade"]
    gkw = {"cap_check": guard.check} if _accepts_kw(grade_fn, "cap_check") else {}   # guard before every attempt
    # grading spend on this record before this pass (earlier failed / interrupted / superseded gradings)
    prior = float(rec.get("grade_cost_cumulative", rec.get("grade_cost", 0.0)) or 0.0)
    if "grade_interrupted_cost" in rec:
        prior = float(rec.pop("grade_interrupted_cost") or 0.0)
    try:
        g = grade_fn(item, ctx, rec["final_raw"], double=double, **gkw)
    except Exception as e:
        interrupted = isinstance(e, PhaseCapReached) or _is_budget_exceeded(e)
        spent = float(getattr(e, "grader_cost", 0.0) or 0.0)
        if interrupted:
            if spent:      # record the grading spend of the interrupted pass; graded state is left unchanged
                rec["grade_interrupted_cost"] = prior + spent
                out.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
            raise
        rec["graded"] = False
        rec["grade_status"] = "grader_error"
        rec["grade_error"] = str(e)[:500]
        rec["grade_cost"] = spent
        rec["grade_cost_cumulative"] = prior + spent
    else:
        cost = g.get("cost")
        if cost is None:
            cost = sum((x or {}).get("cost", 0.0) or 0.0 for x in g.get("grades", []))
        status = g.get("status", "ok" if g.get("score") is not None else "grader_failed")
        rec["grade"] = g
        rec["grade_cost"] = cost
        rec["grade_cost_cumulative"] = prior + cost
        rec["grade_status"] = status
        rec["grader_impl_version"] = g.get("grader_impl")     # None: a grade function without versioning
        rec["graded"] = status == "ok" and g.get("score") is not None
        if rec["graded"]:
            rec.pop("grade_error", None)
        else:          # explicit failure: no score, stays resumable (the next run regrades it)
            rec["grade_status"] = "grader_failed"
            rec["grade_error"] = "grader failed: no valid grade after retries"
    out.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return rec


def _accepts_kw(fn, name):
    try:
        ps = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in ps or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ps.values())


def build_jobs(items, arms, models, mode="e2e", gold_available=None, seed=5):
    """Shuffled job list. gold_available(arm, item) -> bool limits given-gold jobs to existing gold."""
    rng = random.Random(seed)
    jobs = []
    for i in items:
        cell = []
        for m in models:
            for a in arms:
                if mode in ("e2e", "both"):
                    cell.append((i, a, m, False))
                if mode in ("gold", "both") and gold_available and gold_available(a, i):
                    cell.append((i, a, m, True))
        rng.shuffle(cell)
        jobs += cell
    return jobs


def run_all(jobs, cfg, workers=8):
    """Run jobs in a thread pool; stop cleanly on the phase cap or the global budget (pending jobs are
    cancelled; jobs already running stop at their next model call because the guard is tripped).
    Returns {"done": n, "stopped": reason|None, "records": [...]}."""
    done, records, stopped = 0, [], None
    with ThreadPoolExecutor(max(1, workers)) as ex:
        futs = {ex.submit(run_one, j, cfg): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                rec = f.result()
                done += 1
                if rec is not None:
                    records.append(rec)
                    s = (rec.get("grade") or {}).get("score")
                    log(f"[{done}/{len(jobs)}] {j[2]} {j[1]}{GOLD_SUFFIX if j[3] else ''} {j[0]} "
                        f"score={None if s is None else round(s, 2)} calls={rec.get('n_calls')} "
                        f"out_tok={rec.get('output_tokens')} cost=${rec.get('cost', 0) or 0:.4f} "
                        f"err={'Y' if rec.get('error') else 'n'} phase=${cfg['guard'].phase_spent():.3f}")
            except PhaseCapReached as e:
                if not stopped:
                    stopped = f"phase cap: {e}"
                    log(f"PHASE CAP REACHED: {e}. Stopping cleanly; unfinished jobs resume on the next run.")
                    cfg["guard"].stopped = True
                    for x in futs:
                        x.cancel()
                    break
            except Exception as e:
                if _is_budget_exceeded(e):
                    if not stopped:
                        stopped = f"global budget: {e}"
                        log(f"GLOBAL BUDGET EXCEEDED: {e}")
                        cfg["guard"].stopped = True
                        for x in futs:
                            x.cancel()
                        break
                else:
                    log(f"{j} FAILED {e}")
    return {"done": done, "stopped": stopped, "records": records}


def _heldout_ids():
    p = ROOT / "data" / "heldout" / "items.json"
    return set(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else set()


def _sealed(ids):
    """Requested ids that are sealed: the held-out id list plus every item of the held-out and generalisation
    systems in data/split.json (the generalisation set is sealed too, D-043). Fails closed if split.json is
    unreadable. Decided from ids alone, before any item file is read."""
    try:
        sp = json.loads((ROOT / "data" / "split.json").read_text(encoding="utf-8"))
    except Exception:
        return set(ids)
    systems = set(sp.get("heldout_systems", [])) | set(sp.get("generalisation_systems", [])) | set(sp.get("heldout_real", []))
    return {i for i in ids if i in _heldout_ids() or i.split("-")[0] in systems}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--arms", default="A0,A0+,A1,A5,C2,A7")
    ap.add_argument("--models", default="small,large")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--mode", default="e2e", choices=["e2e", "gold", "both"])
    ap.add_argument("--double-grade-frac", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--phase-cap", type=float, default=None, help="USD; stop when phase spend reaches it")
    ap.add_argument("--reasoning-model", default=None)
    ap.add_argument("--gold-dir", default=None)
    ap.add_argument("--allow-heldout", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list jobs; no API calls")
    a = ap.parse_args(argv)

    from harness.run import load_item, split_ids
    from tournament import gold as G
    ids = split_ids(a.split)
    if not a.allow_heldout and (a.split in ("heldout", "generalisation") or _sealed(ids)):
        sys.exit("ABORT: held-out items requested; the tournament runs on dev items until P6 (--allow-heldout).")
    gold_dir = a.gold_dir or str(G.GOLD_DIR)
    arms = a.arms.split(",")
    models = a.models.split(",")
    jobs = build_jobs(ids, arms, models, a.mode, gold_available=lambda arm, i: G.has_gold(arm, i, gold_dir))
    if a.dry_run:
        for j in jobs:
            print(j)
        print(f"{len(jobs)} jobs")
        return {"jobs": jobs}
    if a.phase_cap is None:
        sys.exit("ABORT: --phase-cap USD is required for a paid run.")
    from harness.verify_lock import verify
    bad = verify()
    if bad:
        sys.exit(f"ABORT: held-out lock violated: {bad}")
    from harness.judge.grade import grade
    from harness.llm import chat, true_spent

    def load_gold(arm_name, iid):
        return G.load_gold(arm_name, iid, gold_dir)[2]

    guard = SpendGuard(a.phase_cap, lambda: true_spent(max_age=10))
    cfg = {"base": RUNS, "phase": a.phase, "client": chat, "grade": grade, "load_item": load_item,
           "load_gold": load_gold, "guard": guard, "double_frac": a.double_grade_frac, "reasoning": a.reasoning_model}
    log(f"tournament {a.phase}: {len(jobs)} jobs; phase cap ${a.phase_cap:.2f}; usage at start ${guard.baseline:.4f}")
    res = run_all(jobs, cfg, a.workers)
    log(f"done {res['done']} jobs; phase spend ${guard.phase_spent():.4f}; stopped: {res['stopped']}")
    if res["stopped"]:
        sys.exit(4)
    return res


if __name__ == "__main__":
    main()
