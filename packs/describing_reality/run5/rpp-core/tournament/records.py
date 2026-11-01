"""One validation of tournament records, shared by the orchestrator's completion check
(harness.orchestrator.validate_record) and tournament.run's resume (run_one) - recheck item 3, RECORD_RULES_VERSION.

check_record(r, job, phase, double_frac) classifies a parsed record of job = (item, arm, model_key, given):
  None                       complete: identity matches, no error, graded with a valid grade of the required
                             multiplicity (the rule of review fix 1, unchanged);
  ("unreadable", reason)     not a JSON object (or unreadable file: check_file)        -> quarantine + regenerate
  ("identity", reason)       item / model_key / phase / arm / mode do not match the job -> quarantine + regenerate
  ("errored", reason)        generation error                                          -> quarantine + regenerate
  ("grading", reason)        a valid generation whose grading is missing, failed, malformed or of insufficient
                             multiplicity                                                -> keep the answer, regrade
The reason strings (and the complete / incomplete decision) are exactly those validate_record has always returned.
resume_action() maps a classification to what tournament.run does on resume.
"""
import json
import pathlib

RECORD_RULES_VERSION = "records-1 (recheck item 3)"
GRADE_OK = (None, "ok")
GOLD_SUFFIX = "__gold"
REGENERATE = ("unreadable", "identity", "errored")


def _wants_double(iid, arm, model_key, frac):
    from tournament.run import wants_double
    return wants_double(iid, arm, model_key, frac)


def check_record(r, job, phase, double_frac=0.0):
    iid, arm, model_key, given = job
    if not isinstance(r, dict):
        return "unreadable", "not a JSON object"
    adir = arm + (GOLD_SUFFIX if given else "")
    if r.get("item") != iid or r.get("model_key") != model_key or r.get("phase") != phase:
        return "identity", f"identity mismatch (record says {r.get('phase')}/{r.get('model_key')}/{r.get('item')})"
    if r.get("arm_dir", adir) != adir or (r.get("arm") != arm and r.get("arm_dir") != adir):
        return "identity", f"arm mismatch (record says {r.get('arm')}/{r.get('arm_dir')})"
    if r.get("mode") is not None and r.get("mode") != ("gold" if given else "e2e"):
        return "identity", f"mode mismatch ({r.get('mode')})"
    if r.get("error"):
        return "errored", "errored"
    if r.get("grade_status") not in GRADE_OK:
        return "grading", f"grade_status {r.get('grade_status')}"
    if r.get("graded") is not True:
        return "grading", "not graded" + (f" ({str(r.get('grade_error'))[:80]})" if r.get("grade_error") else "")
    g = r.get("grade")
    if not isinstance(g, dict):
        return "grading", "grade missing"
    if g.get("grade_status", g.get("status")) not in GRADE_OK:
        return "grading", f"grade status {g.get('grade_status', g.get('status'))}"
    s = g.get("score")
    if isinstance(s, bool) or not isinstance(s, (int, float)) or not 0 <= s <= 1:
        return "grading", f"invalid score {s!r}"
    gs = g.get("grades")
    if not isinstance(gs, list) or not gs:
        return "grading", "grades list missing or empty"
    for x in gs:
        if not isinstance(x, dict) or x.get("grade_status", x.get("status")) not in GRADE_OK or x.get("error"):
            return "grading", "a grader call failed"
        xs = x.get("score")
        if isinstance(xs, bool) or not isinstance(xs, (int, float)) or not 0 <= xs <= 1:
            return "grading", f"invalid grader score {xs!r}"
    need = 2 if _wants_double(iid, arm, model_key, double_frac or 0.0) else 1
    if len(gs) < need:
        return "grading", f"{len(gs)} grade(s); {need} required (double grading)"
    return None


def resume_action(kind, rec):
    """-> "keep" | "regrade" | "regenerate" for a check_file classification. A grading problem is regraded only when
    the record holds an answer text to grade; otherwise the generation itself is unusable."""
    if kind is None:
        return "keep"
    if kind == "grading" and isinstance(rec, dict) and isinstance(rec.get("final_raw"), str):
        return "regrade"
    return "regenerate"


def check_file(path, job, phase, double_frac=0.0):
    """-> (record or None, kind or None, reason or None); kind "missing" when there is no file."""
    try:
        r = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing", "missing"
    except Exception as e:
        return None, "unreadable", f"unreadable JSON ({type(e).__name__})"
    res = check_record(r, job, phase, double_frac)
    return (r if isinstance(r, dict) else None), (res[0] if res else None), (res[1] if res else None)
