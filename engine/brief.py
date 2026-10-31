"""The designer's brief: what the AI that writes new versions of a part may know. Numbers only.

The engine builds it from scores and the pack's aggregate evidence: the current best's mean and range, its means by
difficulty level, how often each failure type occurred, the pack's own aggregates, and every version tried before with
its measured effect and what happened to it. It never contains an item, a question, an answer, an answer key, a
per-item score or anything from the sealed set; the designer cannot see what it is tested on.
"""
import json

import numpy as np


def build(stats, store, pack, candidate, folder, items, history):
    sc = {i: store.score(folder, candidate["id"], i) for i in items}
    vals = [v for v in sc.values() if v is not None]
    by = {}
    for i, v in sc.items():
        if v is not None and stats.level_of(i) is not None:
            by.setdefault(str(stats.level_of(i)), []).append(v)
    fails = {}
    recs = store.records(folder, candidate["id"], items)
    for r in recs.values():
        t = pack.failure_type(r) if r else None
        if t:
            fails[t] = fails.get(t, 0) + 1
    ev = {"n_items": len(vals), "mean": round(float(np.mean(vals)), 4) if vals else None,
          "by_level": {k: {"mean": round(float(np.mean(v)), 4), "n": len(v)} for k, v in sorted(by.items())},
          "failure_types": fails, "pack": pack.evidence(candidate, recs) or {}, "history": history}
    return {"evidence": ev, "text": render(ev)}


def render(ev):
    lines = ["WHAT EARLIER STAGES MEASURED (scores only; no test case is shown)",
             f"Current best: mean score {ev['mean']} on {ev['n_items']} test cases."]
    if ev["by_level"]:
        lines.append("By difficulty level: " + ", ".join(f"{k}: {v['mean']} (n={v['n']})"
                                                         for k, v in ev["by_level"].items()))
    if ev["failure_types"]:
        lines.append("How runs lost points: " + ", ".join(f"{k} {v}" for k, v in
                                                          sorted(ev["failure_types"].items(), key=lambda x: -x[1])))
    for k, v in (ev["pack"] or {}).items():
        lines.append(f"{k}: {json.dumps(v) if not isinstance(v, str) else v}")
    if ev["history"]:
        lines.append("\nVersions tried before (effect = change in mean score against the design it changed; a range "
                     "that includes 0 means no clear effect):")
        for h in ev["history"]:
            rng = h.get("range") or [None, None]
            lines.append(f"- {h['where']}, part {h['slot']}: effect {h.get('effect')} (range {rng[0]} to {rng[1]}); "
                         f"{h['verdict']}" + (f"\n  version: {h['value']}" if h.get("value") is not None else ""))
    return "\n".join(lines)
