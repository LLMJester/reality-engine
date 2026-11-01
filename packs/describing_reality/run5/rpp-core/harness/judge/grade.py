"""Blind grader. Input per call: description, question, answer key, rubric and
one anonymised prose answer. Nothing else (no condition, model, or method)."""
import json
import pathlib

from harness.llm import chat, GRADER_MODEL
from harness.strip import grader_text

# Implementation version of the grading procedure (not of the rubric, prompts, seeds or model, which are
# unchanged). 1: original. 2: failed passes are explicit (score None, status "grader_failed"), cost summed over
# every attempt, grade structure validated (invalid structure = retry), cap_check before every attempt.
GRADER_IMPL_VERSION = 2
ATTEMPTS = 3
VERDICTS = ("hit", "partial", "miss")
RUBRIC = (pathlib.Path(__file__).parent / "rubric.md").read_text(encoding="utf-8")
FMT = {"type": "json_schema", "json_schema": {"name": "grade", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["key_points", "material_errors", "comment"],
    "properties": {
        "key_points": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                  "required": ["id", "verdict", "reason"],
                                                  "properties": {"id": {"type": "string"},
                                                                 "verdict": {"enum": ["hit", "partial", "miss"]},
                                                                 "reason": {"type": "string"}}}},
        "material_errors": {"type": "array", "items": {"type": "string"}},
        "comment": {"type": "string"}}}}}


def level_of(s):
    """Rubric level 0-4, round half up."""
    return int(s * 4 + 0.5)


def key_for_grader(item):
    k = item["answer_key"]
    return {"summary": k.get("summary", ""), "key_points": k.get("key_points", []),
            "known_wrong_answers": k.get("wrong_answers", [])}


def score(item, g):
    kps = item["answer_key"].get("key_points", [])
    w = {kp["id"]: float(kp.get("weight", 1.0)) for kp in kps}
    verdicts = {kp["id"]: kp["verdict"] for kp in g.get("key_points", [])}
    total = sum(w.values()) or 1.0
    got = sum(w[i] * {"hit": 1.0, "partial": 0.5}.get(verdicts.get(i, "miss"), 0.0) for i in w)
    pen = 0.1 * len(g.get("material_errors", []))
    return max(0.0, got / total - pen)


def validate_grade(item, g):
    """Structural check of one grader output (GRADER_IMPL_VERSION 2). Returns None if valid, else a reason.
    Valid: a JSON object; key_points a list of objects with a verdict in the enum; every answer-key key-point id
    judged exactly once and no unknown ids; material_errors a list of strings."""
    if not isinstance(g, dict):
        return "grade is not an object"
    kps = g.get("key_points")
    if not isinstance(kps, list):
        return "key_points is not a list"
    errs = g.get("material_errors")
    if not isinstance(errs, list) or not all(isinstance(e, str) for e in errs):
        return "material_errors is not a list of strings"
    expected = {kp["id"] for kp in item["answer_key"].get("key_points", [])}
    seen = []
    for kp in kps:
        if not isinstance(kp, dict):
            return "key point verdict is not an object"
        if kp.get("verdict") not in VERDICTS:
            return f"verdict outside {sorted(VERDICTS)}"
        seen.append(kp.get("id"))
    if len(seen) != len(set(seen)):
        return "a key point was judged more than once"
    if set(seen) - expected:
        return "unknown key point id"
    if expected - set(seen):
        return "a key point was not judged"
    return None


def _no_check():
    return None


def grade_once(item, ctx, answer_text, seed, purpose="grade", model=GRADER_MODEL, cap_check=None, client=None):
    """One grading pass with up to three attempts. cap_check() is called before EVERY attempt (it may raise to
    stop spending). Returns status "ok" with score, or status "grader_failed" with score None; cost is the sum
    over all attempts, including invalid ones."""
    cap_check = cap_check or _no_check
    client = client or chat
    user = (f"SYSTEM DESCRIPTION\n==================\n{ctx}\n\nQUESTION\n========\n{item['question']}\n\n"
            f"TASK TYPE: {item['task_type']}\n\nANSWER KEY\n==========\n{json.dumps(key_for_grader(item), indent=1)}\n\n"
            f"ANSWER TO GRADE\n===============\n{answer_text or '(empty answer)'}\n")
    msgs = [{"role": "system", "content": RUBRIC}, {"role": "user", "content": user}]
    cost, errors = 0.0, []
    for attempt in range(ATTEMPTS):
        try:
            cap_check()
            m, u = client(model, msgs, purpose=purpose, seed=seed + 100 * attempt, max_tokens=6000,
                          response_format=FMT, temperature=0.0)
        except BaseException as e:
            try:
                e.grader_cost = cost      # cost of the earlier attempts of this pass (callers record it)
            except Exception:
                pass
            raise
        cost += float((u or {}).get("cost") or 0.0)
        try:
            g = json.loads(m.get("content") or "")
            bad = validate_grade(item, g)
            if bad:
                errors.append(bad)
                continue
            s = score(item, g)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {str(e)[:100]}")
            continue
        return {"grade": g, "score": s, "level": level_of(s), "seed": seed, "cost": cost, "status": "ok",
                "attempts": attempt + 1, "grader_impl": GRADER_IMPL_VERSION}
    return {"grade": None, "score": None, "level": None, "seed": seed, "cost": cost, "status": "grader_failed",
            "attempts": ATTEMPTS, "error": "grader output invalid", "attempt_errors": errors,
            "grader_impl": GRADER_IMPL_VERSION}


def _failed(gs, text):
    return {"score": None, "status": "grader_failed", "grades": gs, "text": text,
            "cost": sum(g.get("cost", 0.0) for g in gs), "grader_impl": GRADER_IMPL_VERSION}


def grade(item, ctx, raw_final, double=False, cap_check=None, client=None):
    """Returns {score, status, grades:[...], text, cost}. double=True: two seeds, third on > 1 level
    disagreement, median. If any required pass fails, status is "grader_failed" and score is None (no median is
    taken over failures). cap_check is called before every grader call (default: no check)."""
    text = grader_text(raw_final)
    kw = {"cap_check": cap_check, "client": client}

    def pass_cost(e, gs):          # attach the cost of completed passes to an interrupting exception
        try:
            e.grader_cost = getattr(e, "grader_cost", 0.0) + sum(g.get("cost", 0.0) for g in gs)
        except Exception:
            pass

    gs = []
    try:
        g1 = grade_once(item, ctx, text, seed=1, **kw)
        gs.append(g1)
        if g1["status"] != "ok":
            return _failed(gs, text)
        if double:
            g2 = grade_once(item, ctx, text, seed=2, **kw)
            gs.append(g2)
            if g2["status"] != "ok":
                return _failed(gs, text)
            if abs(g1["level"] - g2["level"]) > 1:
                g3 = grade_once(item, ctx, text, seed=3, **kw)
                gs.append(g3)
                if g3["status"] != "ok":
                    return _failed(gs, text)
    except BaseException as e:
        pass_cost(e, gs)
        raise
    scores = sorted(g["score"] for g in gs)
    med = scores[len(scores) // 2] if len(scores) != 2 else sum(scores) / 2
    return {"score": med, "grades": gs, "text": text, "status": "ok", "cost": sum(g["cost"] for g in gs),
            "grader_impl": GRADER_IMPL_VERSION}
