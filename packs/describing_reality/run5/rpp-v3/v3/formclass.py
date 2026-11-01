"""Which form did the builder choose for its notes? (run 5 founder OWN, V3-042). Code only, no AI.

classify(text) -> {"form": one of FORMS, "chars": n, "lines": n, "features": {...}}. The first rule that fits wins:
  json       the notes (a single surrounding ``` fence ignored) parse as a JSON object or array
  code       a fenced code block, or at least 3 lines of program syntax (def / class / import / return / for, while
             or if ending in a colon) making up at least 20% of the lines ("name = value" lines alone are not code)
  tables     at least one Markdown table (a row followed by a |---| separator row)
  headings   at least 2 section headings (`# ...` or an upper-case line ending in a colon)
  bullets    at least half of the non-blank lines are bullets or numbered lines
  prose      anything else
"""
import json
import re

FORMS = ("json", "code", "tables", "headings", "bullets", "prose")
FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*)\n\s*```\s*$", re.S)
CODE_FENCE_RE = re.compile(r"```(?:python|py|js|javascript)\b", re.I)
CODE_LINE_RE = re.compile(r"^\s*(?:def |class |import |from \w+ import |return\b|for .+:\s*$|while .+:\s*$|if .+:\s*$)")
ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s+\S|[A-Z][A-Z0-9 &/()-]{2,40}:\s*$)")
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")


def classify(text):
    # a prose build ends with "FINAL ANSWER" and the notes; reasoning before it is not part of the notes (code review)
    text = re.split(r"(?m)^\s*(?:\*\*)?FINAL ANSWER(?:\*\*)?\s*:?\s*", text or "")[-1].strip()
    lines = [ln for ln in text.replace("\r\n", "\n").split("\n")]
    body = [ln for ln in lines if ln.strip()]
    m = FENCE_RE.match(text)
    feats = {
        "tables": sum(1 for a, b in zip(lines, lines[1:]) if ROW_RE.match(a) and SEP_RE.match(b)),
        "headings": sum(1 for ln in body if HEADING_RE.match(ln)),
        "bullets": sum(1 for ln in body if BULLET_RE.match(ln)),
        "code_lines": sum(1 for ln in body if CODE_LINE_RE.match(ln)),
        "code_fence": bool(CODE_FENCE_RE.search(text)),
    }
    form = "prose"
    try:
        parsed = json.loads(m.group(1) if m else text)
        if isinstance(parsed, (dict, list)):
            form = "json"
    except ValueError:
        pass
    if form != "json":
        if feats["code_fence"] or (feats["code_lines"] >= 3 and feats["code_lines"] >= 0.2 * len(body)):
            form = "code"
        elif feats["tables"]:
            form = "tables"
        elif feats["headings"] >= 2:
            form = "headings"
        elif body and feats["bullets"] >= len(body) / 2:
            form = "bullets"
    return {"form": form, "chars": len(text), "lines": len(body), "features": feats}
