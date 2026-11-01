"""Core "sectioned": structured prose with evolvable section headers (v2 A1 generalised).

The model is a plain-text document of sections. genes.core.params:
  headers      ordered list of 3-10 required section headers (schema.HEADER_VOCAB or designer-invented)
  tags         true: every element carries a provenance tag [stated: ...] / [derived: ...] / [assumed: ...]
  granularity  "coarse": an element may run over indented continuation lines and hold several sentences;
               "fine": one fact per bullet (a single line with a single sentence)

Checker rules (every problem has a line number where one exists, and a fix hint):
  S1 the document is not empty
  S2 a header line names one of the genome's headers (`## NAME`, or a bare upper-case NAME line)
  S3 no text before the first header
  S4 every line under a header is an element: a bullet (`- `, `* `, `•`) or numbered line (`1.` / `1)`)
     (coarse: indented continuation lines of an element are allowed)
  S5 tags=true: every element has a provenance tag (anywhere in the element, continuation lines included)
  S6 fine: an element is one line with one sentence (no continuation lines, no second sentence)
  S7 every header in the genome is present and has at least one element (repeated headers merge)
  S8 no unfilled template placeholder (`<...>`) in an element

Tools are A1's (write_model, revise_model, show_model); the A1 class is reused by subclassing (rpp-core untouched).
"""
import re

from v3.cores.text import (ELEMENT_RE, heading_of, numbered, problem, section_problems, sort_problems)
from v3.genome import schema as S

from tournament.arms.a1_prose import A1, TAG_RE   # v2, read-only (sys.path set by v3.cores.text)

CONTINUATION_RE = re.compile(r"^(?:\t| {2,})\S")
# a template placeholder: <...> or <word ...> (starts and ends with a letter), so comparisons such as
# "T1 < 10% and P1 > 50" are not placeholders (code review, 2026-10-29)
PLACEHOLDER_RE = re.compile(r"<(?:\.\.\.|(?![^<>\n]*\b(?:and|or)\b)[A-Za-z_][A-Za-z_ /-]{0,38}[A-Za-z_])>")
SENTENCE_SPLIT =re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"(])")
MAX_FINE_CHARS = 300
TAG_HINT = "end the element with [stated: \"short quote\"], [derived: from ...] or [assumed: why]"


def _facts(text):
    body = TAG_RE.sub("", text).strip()
    body = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", body)
    return [s for s in SENTENCE_SPLIT.split(body) if s.strip()]


def check(text, headers, tags=True, granularity="coarse"):
    """Static checker for a sectioned document. Returns [problem] sorted by line."""
    if not isinstance(text, str) or not text.strip():
        return [problem("the document is empty", None, None, "write the sections: " + ", ".join(headers))]
    probs, seen, counts = [], {}, {}
    current, elem = None, None          # elem = [first line, text] of the element being read

    def close():
        if elem and current is not None:
            if tags and not TAG_RE.search(elem[1]):
                probs.append(problem("element has no provenance tag", elem[0], current, TAG_HINT))
    for no, raw in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
        if not raw.strip():
            continue
        h = heading_of(raw, headers)
        if h is not None:
            close()
            elem = None
            if h not in headers:
                probs.append(problem(f"unknown section header '{h}'", no, h,
                                     "use only these headers: " + ", ".join(headers)))
            seen.setdefault(h, no)
            current = h
            continue
        if ELEMENT_RE.match(raw):
            close()
            elem = None
            if current is None:
                probs.append(problem("element before the first header", no, None,
                                     "put a header line such as `## " + headers[0] + "` above it"))
                continue
            elem = [no, raw]
            counts[current] = counts.get(current, 0) + 1
            if PLACEHOLDER_RE.search(raw):
                probs.append(problem("unfilled template placeholder", no, current,
                                     "replace every <...> with content from the description"))
            if granularity == "fine":
                if len(_facts(raw)) > 1:
                    probs.append(problem("more than one fact in one element", no, current,
                                         "split it: one fact per bullet line"))
                elif len(raw.strip()) > MAX_FINE_CHARS:
                    probs.append(problem(f"element longer than {MAX_FINE_CHARS} characters", no, current,
                                         "split it: one short fact per bullet line"))
            continue
        if elem and CONTINUATION_RE.match(raw):
            if granularity == "fine":
                probs.append(problem("continuation line (fine granularity allows one line per element)", no,
                                     current, "make it its own bullet line, one fact per bullet"))
            else:
                elem[1] += " " + raw.strip()
            continue
        close()
        elem = None
        if current is None:
            probs.append(problem("text before the first header", no, None,
                                 "start the document with a header line such as `## " + headers[0] + "`"))
        else:
            probs.append(problem("line is not an element (bullet or numbered line)", no, current,
                                 "start the line with `- ` (one element per bullet)"))
    close()
    probs += section_problems(headers, seen, counts)
    return sort_problems(probs)


class SectionedCore(A1):
    name = "SEC"
    buildable = True
    executable = False
    guide_file = None

    def __init__(self, params=None):
        p = dict(S.SECTIONED_DEFAULT)
        p.update(params or {})
        self.headers = list(p["headers"])
        self.tags = bool(p["tags"])
        self.granularity = p["granularity"]

    def problems(self, text):
        return check(text, self.headers, self.tags, self.granularity)

    def heading_of(self, line):
        return heading_of(line, self.headers)

    def _report(self, st):
        text = st.rep.get("text", "")
        n = sum(1 for ln in text.splitlines() if ELEMENT_RE.match(ln))
        secs = {h for ln in text.splitlines() if (h := heading_of(ln, self.headers))}
        return f"Model saved (version {st.versions}): {n} elements in {len(secs)} section(s)."

    def handle(self, state, name, args):
        if name == "show_model":
            text = state.rep.get("text", "")
            return numbered(text) if text else "ERROR: no model has been written yet; use write_model."
        return super().handle(state, name, args)

    def validity(self, state):
        text = state.rep.get("text", "")
        probs = self.problems(text) if text.strip() else [problem("no model written")]
        ok = bool(text.strip()) and not probs
        return {"usable": ok, "executable": ok, "checked": ok,
                "holes": [{"line": p["line"], "problem": p["msg"]} for p in probs],
                "errors": [] if ok else [f"{len(probs)} checker problem(s)"],
                "stats": {"elements": sum(1 for ln in text.splitlines() if ELEMENT_RE.match(ln)),
                          "problems": len(probs)}}

    def template(self):
        tag = " [stated: \"<short quote>\" | derived: from <...> | assumed: <why>]" if self.tags else ""
        one = "- <one fact>" if self.granularity == "fine" else "- <element>"
        body = "\n\n".join(f"## {h}\n{one}{tag}\n{one}{tag}" for h in self.headers)
        return ("TEMPLATE (fill every <...>; keep every header; every line under a header is a bullet"
                + ("; one fact per bullet" if self.granularity == "fine" else "") + ")\n" + body)

    def guide(self):
        lines = ["# Sectioned plain-language model - guide", "",
                 "Describe the system as a plain-language document organised under fixed section headers. The "
                 "document is checked; it counts as built only when the check reports no problems.", "",
                 "## Sections", "", "Write every one of these headers, as `## NAME` lines, each followed by at least "
                 "one element:", ""]
        lines += [f"- `## {h}`" for h in self.headers]
        lines += ["", "## Elements", "",
                  "Every non-blank line under a header is an element: a bullet (`- ...`) or a numbered line "
                  "(`1. ...`). No other headers and no text outside the sections."]
        if self.granularity == "fine":
            lines.append("One fact per bullet: one line, one sentence. Split compound statements into several "
                         "bullets.")
        else:
            lines.append("An element may continue on indented lines below its bullet.")
        if self.tags:
            lines += ["", "Every element carries one provenance tag: `[stated: \"short exact quote\"]`, "
                      "`[derived: from <other elements>]` or `[assumed: <reason>]`."]
        lines += ["", "## Tools", "", "- `write_model(text)` - write the whole document (replaces any earlier "
                  "version); the result lists the check problems with line numbers.",
                  "- `revise_model(text)` - replace the document with a revised full text.",
                  "- `show_model()` - show the document with line numbers."]
        return "\n".join(lines)
