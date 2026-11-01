"""Grader test (spec §9.2, milestone M4): choose the cheapest AI grader that agrees with reference marks.

Sample: answers from v2 gen-0 dev records (all 30 dev items have prose key-point answer keys), spread over arms and
items. Reference:
either human marks (a CSV/JSON the human fills in: {record_key: score 0..1}) or the consensus (median) of several
strong graders. Each candidate grades every answer twice (two seeds). Reported per candidate:
  agreement  = share of answers whose score is within one rubric level (0.25) of the reference
  consistency = share of answers whose two passes agree within one level
  mae, cost per answer, failures.
The candidate is eligible if agreement >= the configured threshold; the cheapest eligible one is recommended.
"""
import argparse
import json
import pathlib
import random
import statistics
import time

from v3.config import CFG, RUNS, V2

OUT = RUNS / "grader_test"


def sample(n=100, seed=0):
    """(key, item_id, final_raw) from v2 gen-0 dev e2e records; balanced across items."""
    from v3 import items as I
    recs = []
    for p in sorted((V2 / "runs" / "tournament" / "gen0_e2e").glob("*/*/*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if r.get("final_raw"):
            recs.append((f"{p.parent.parent.name}/{p.parent.name}/{p.stem}", r["item"], r["final_raw"]))
    rng = random.Random(seed)
    by = {}
    for x in recs:
        by.setdefault(x[1], []).append(x)
    for v in by.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < n and any(by.values()):
        for k in sorted(by):
            if by[k] and len(out) < n:
                out.append(by[k].pop())
        i += 1
    dev = set(I.v2_dev_ids())
    return [x for x in out if x[1] in dev]


def run(candidates, n=100, reference=None, seeds=(7, 107)):
    from v3 import items as I
    from v3.grading.grade import grade
    OUT.mkdir(parents=True, exist_ok=True)
    smp = sample(n)
    marks = {}
    from concurrent.futures import ThreadPoolExecutor
    items = {iid: I.v2_item(iid) for _, iid, _ in smp}

    def one(job):
        model, key, iid, raw, s = job
        try:
            g = grade(items[iid], raw, model=model, seed=s)
        except Exception as e:          # a failed call is a grader failure, never a crash of the test
            g = {"score": None, "cost": 0.0, "status": f"error: {type(e).__name__}"}
        return model, key, {"score": g.get("score"), "cost": g.get("cost", 0.0), "status": g.get("status")}

    jobs = [(m, k, i, r, s) for m in candidates for k, i, r in smp for s in seeds]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for model, key, mark in pool.map(one, jobs):
            marks.setdefault(model, {}).setdefault(key, []).append(mark)
    (OUT / "marks.json").write_text(json.dumps(marks, indent=1), encoding="utf-8")
    ref = reference or consensus(marks)
    report = summarise(marks, ref)
    (OUT / "report.json").write_text(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(smp),
                                                 "reference": "human" if reference else "consensus",
                                                 "candidates": report}, indent=1), encoding="utf-8")
    return report


def consensus(marks):
    keys = set().union(*[set(v) for v in marks.values()])
    ref = {}
    for k in keys:
        vals = [m["score"] for model in marks for m in marks[model].get(k, []) if m["score"] is not None]
        if vals:
            ref[k] = statistics.median(vals)
    return ref


def summarise(marks, ref, threshold=None):
    thr = CFG["grading"]["agreement_threshold"] if threshold is None else threshold
    out = {}
    for model, by in marks.items():
        agree = cons = n = nc = 0
        err, cost, fails = [], 0.0, 0
        for k, passes in by.items():
            cost += sum(p["cost"] or 0 for p in passes)
            sc = [p["score"] for p in passes if p["score"] is not None]
            fails += sum(1 for p in passes if p["score"] is None)
            if len(sc) >= 2:
                nc += 1
                cons += abs(sc[0] - sc[1]) <= 0.25
            if sc and k in ref:
                n += 1
                agree += abs(sc[0] - ref[k]) <= 0.25
                err.append(abs(sc[0] - ref[k]))
        out[model] = {"agreement": agree / n if n else None, "consistency": cons / nc if nc else None,
                      "mae": statistics.mean(err) if err else None, "failures": fails,
                      "cost_per_answer": cost / max(1, sum(len(v) for v in by.values())),
                      "eligible": bool(n and agree / n >= thr)}
    elig = sorted((m for m in out if out[m]["eligible"]), key=lambda m: out[m]["cost_per_answer"])
    for m in out:
        out[m]["recommended"] = bool(elig) and m == elig[0]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="comma-separated OpenRouter model ids")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reference", help="JSON file {record_key: score} of human marks")
    a = ap.parse_args(argv)
    ref = json.loads(pathlib.Path(a.reference).read_text(encoding="utf-8")) if a.reference else None
    print(json.dumps(run(a.candidates.split(","), a.n, ref), indent=1))


if __name__ == "__main__":
    main()
