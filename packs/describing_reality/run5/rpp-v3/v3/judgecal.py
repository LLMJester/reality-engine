"""Judge calibration (V3-044, pre-registered; reported, never gating). After stage 1, JUDGE_CAL_N hand-written answers,
spread evenly over the founders and chosen at random with a fixed seed, are graded again by Claude Opus with the same
rubric, answer key and answer text as the run's judge (DeepSeek v4.1 Flash). Reported: the mean absolute difference,
the correlation, and each founder's mean difference (judge minus Opus), to show whether the judge favours any
founder's answer style.
"""
import json
import random

import numpy as np

SEED = 20261029


def opus_client(model, messages, **kw):
    """grade_once's client interface on Claude Opus (Claude plan, headless, no tools)."""
    from v3.hand5.factory import claude_text
    r = claude_text(messages[0]["content"], messages[1]["content"] + "\n\nReturn the JSON object only.",
                    timeout=900, model="opus")
    from v3.jsonx import objects
    grades = [o for o in objects(r["final_raw"]) if "key_points" in o]     # the grade object, not quoted JSON
    return {"content": json.dumps(grades[-1]) if grades else r["final_raw"]}, {"cost": 0.0}


def sample(mode, designs, founders, items, n):
    from v3.run5 import record
    pool = {f: [i for i in items if (record(mode, designs[f]["id"], i) or {}).get("grade", {}).get("status") == "ok"]
            for f in founders}
    rng = random.Random(SEED)
    for f in pool:
        rng.shuffle(pool[f])
    out, k = [], 0
    while len(out) < n and any(pool.values()):
        f = founders[k % len(founders)]
        if pool[f]:
            out.append((f, pool[f].pop()))
        k += 1
    return out


def calibrate(mode, designs, founders, items, n=100, client=None):
    from concurrent.futures import ThreadPoolExecutor
    from harness.judge.grade import grade_once
    from harness.strip import grader_text
    from v3 import items as I
    from v3.run5 import record
    if client is None:
        from v3.hand5.factory import claude_guard
        claude_guard()                                    # stops at 85% of the weekly Claude allowance
    picks = sample(mode, designs, founders, items, n)

    def one(fi):
        f, iid = fi
        try:                                              # one failed grade loses that grade only (code review)
            rec = record(mode, designs[f]["id"], iid)
            it = I.load(iid)
            g = grade_once(it, it["context"], grader_text(rec.get("final_raw")), 7, purpose="v3_judgecal",
                           model="claude-opus", client=client or opus_client)
            return {"founder": f, "item": iid, "judge": rec["grade"].get("score"), "opus": g.get("score")}
        except Exception as e:
            return {"founder": f, "item": iid, "judge": None, "opus": None, "error": f"{type(e).__name__}"}
    with ThreadPoolExecutor(8) as pool:
        rows = [r for r in pool.map(one, picks) if r["opus"] is not None and r["judge"] is not None]
    if not rows:
        return {"n": 0}
    a = np.array([r["judge"] for r in rows])
    b = np.array([r["opus"] for r in rows])
    by = {}
    for r in rows:
        by.setdefault(r["founder"], []).append(r["judge"] - r["opus"])
    return {"n": len(rows), "mean_abs_diff": round(float(np.mean(np.abs(a - b))), 4),
            "correlation": round(float(np.corrcoef(a, b)[0, 1]), 4) if len(rows) > 2 else None,
            "mean_diff_judge_minus_opus": round(float(np.mean(a - b)), 4),
            "by_founder": {f: {"n": len(v), "mean_diff": round(float(np.mean(v)), 4)} for f, v in by.items()},
            "rows": rows}
