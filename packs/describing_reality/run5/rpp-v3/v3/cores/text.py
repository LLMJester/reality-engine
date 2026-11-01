"""Shared plain-text helpers for the v3 plain-language cores: problems and section headers.

Header lines follow v2 A1 (tournament.arms.a1_prose.HEADING_RE): a markdown heading (`## NAME`) is always a header;
a bare upper-case line (`NAME` or `NAME:`) is a header when NAME is a known header (the genome's headers or the
starter vocabulary) or ends with a colon, so an upper-case term such as `CPU` is not mistaken for a header.
"""
import re
import sys

from v3.config import V2
from v3.genome import schema as S

if str(V2) not in sys.path:
    sys.path.append(str(V2))     # appended, as in v3.genome.render

from tournament.arms.a1_prose import ELEMENT_RE, HEADING_RE   # noqa: E402  (v2, read-only)

MD_HEADING = re.compile(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$")


def problem(msg, line=None, where=None, hint=None):
    """Same shape as v3.genome.render._problem."""
    return {"msg": str(msg), "line": line, "where": where, "hint": hint}


def norm_header(name):
    s = re.sub(r"^\d+[.)]\s*", "", str(name).strip())
    s = s.rstrip(":").replace("_", " ")
    return " ".join(s.split()).upper()


def heading_of(line, known=()):
    """Normalised header name if the line is a section header, else None."""
    if ELEMENT_RE.match(line):
        return None
    m = MD_HEADING.match(line)
    if m:
        return norm_header(m.group(1))
    h = HEADING_RE.match(line)
    if h and h.group(2):
        name = norm_header(h.group(2))
        if name in set(known) | set(S.HEADER_VOCAB) or line.rstrip().endswith(":"):
            return name
    return None


def sort_problems(probs):
    return sorted(probs, key=lambda p: (p["line"] is None, p["line"] or 0))


def numbered(text):
    return "\n".join(f"{i:3d} | {ln}" for i, ln in enumerate((text or "").splitlines(), 1))


def section_problems(headers, seen, counts):
    """Missing / empty required sections. seen = {header: first line}, counts = {header: n elements}."""
    out = []
    for h in headers:
        if h not in seen:
            out.append(problem(f"missing section '{h}'", None, h,
                               f"add a header line `## {h}` followed by its lines"))
        elif not counts.get(h):
            out.append(problem(f"section '{h}' has no content", seen[h], h,
                               f"write at least one line under `## {h}`"))
    return out
