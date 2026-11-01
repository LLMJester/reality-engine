"""Grading (spec §9): script first, an AI grader for prose answer keys, a frontier spot-check audit.

- Generated items: exact scripted scoring (v3.items.score_generated).
- v2 real items: the v2 blind grader procedure (harness/judge/grade.py: rubric, key points, validation, retries), with
  the grader model chosen by the grader test (v3.grading.grader_test) and configured in config/v3.json.
"""
import random

from v3 import items as I
from v3.config import CFG


def grader_model():
    m = CFG["models"]["grader"]
    if m.startswith("decide"):
        raise RuntimeError("no AI grader chosen yet: run the grader test (M4) and set models.grader")
    return m


def grade(item, final_raw, *, model=None, seed=7, client=None, cap_check=None, lenient=False):
    """Returns {score, status, grader, cost, ...}. Never raises for grader failures (status 'grader_failed')."""
    if item.get("grading") == "script":
        return I.score_generated(item, final_raw, lenient)
    from harness.judge.grade import grade_once
    from harness.strip import grader_text
    from v3.llm import chat
    g = grade_once(item, item["context"], grader_text(final_raw), seed, purpose="v3_grade",
                   model=model or grader_model(), cap_check=cap_check, client=client or chat)
    return dict(g, grader=model or grader_model())


def needs_audit(record_id, fraction=None):
    """Deterministic 5% (configurable) sample of final-evaluation answers for the frontier audit."""
    f = CFG["grading"]["frontier_audit_fraction"] if fraction is None else fraction
    return random.Random(f"audit|{record_id}").random() < f
