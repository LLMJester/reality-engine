"""A1: structured prose world model (not executable).

The model writes a sectioned prose model; every element (bullet or numbered line) must carry a
provenance tag [stated: ...], [derived: ...] or [assumed: ...]. Untagged elements are reported as
holes with their line number; the rest of the model stays usable."""
import copy
import re

from tournament.arms.base import Arm, ArmState, ERROR_PREFIX

TAG_RE = re.compile(r"\[\s*(stated|derived|assumed)\b[^\]]*\]", re.I)
ELEMENT_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s+(.+?)\s*#*\s*$|([A-Z][A-Z &/,()-]{2,}):?\s*$)")
SECTIONS = ["ENTITIES", "STATE", "MECHANISMS", "CAUSAL LINKS", "TIMING", "UNCERTAINTY"]
ANSWER_HEADING = re.compile(r"conclusion|answer|prediction|verdict|recommendation", re.I)
MAX_TEXT = 40000


def _tool(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


TOOLS = [
    _tool("write_model", "Write the prose world model (the whole text). Replaces any earlier version.",
          {"text": {"type": "string"}}, ["text"]),
    _tool("revise_model", "Replace the prose world model with a revised full text.",
          {"text": {"type": "string"}}, ["text"]),
    _tool("show_model", "Show the current prose world model with line numbers."),
]


def analyse(text):
    """Element/tag statistics: elements, tag counts, untagged lines, sections present."""
    lines = (text or "").splitlines()
    counts = {"stated": 0, "derived": 0, "assumed": 0}
    untagged, elements, sections = [], 0, []
    for i, ln in enumerate(lines, 1):
        h = HEADING_RE.match(ln)
        if h and not ELEMENT_RE.match(ln):
            sections.append((h.group(1) or h.group(2)).strip().upper())
            continue
        if ELEMENT_RE.match(ln):
            elements += 1
            tags = TAG_RE.findall(ln)
            if not tags:
                untagged.append((i, ln.strip()[:120]))
            for t in tags[:1]:
                counts[t.lower()] += 1
    missing = [s for s in SECTIONS if not any(s in x for x in sections)]
    return {"elements": elements, "tags": counts, "untagged": untagged, "sections": sections,
            "missing_sections": missing}


class A1(Arm):
    name = "A1"
    buildable = True
    executable = False
    guide_file = "a1_guide.md"
    mutating_tools = ("write_model", "revise_model")

    def tools(self):
        return TOOLS

    def new_state(self):
        st = ArmState()
        st.rep = {"text": ""}
        st.versions = 0
        return st

    def _report(self, st):
        a = analyse(st.rep.get("text", ""))
        out = [f"Model saved (version {st.versions}): {a['elements']} elements "
               f"(stated {a['tags']['stated']}, derived {a['tags']['derived']}, assumed {a['tags']['assumed']})."]
        if a["untagged"]:
            out.append("Untagged elements (each element needs [stated: \"quote\"], [derived: from ...] or "
                       "[assumed: ...]):")
            out += [f"  line {i}: {t}" for i, t in a["untagged"][:20]]
        if a["missing_sections"]:
            out.append("Sections not found: " + ", ".join(a["missing_sections"]))
        return "\n".join(out)

    def handle(self, state, name, args):
        if name in ("write_model", "revise_model"):
            text = args.get("text")
            if not isinstance(text, str) or not text.strip():
                return f"{ERROR_PREFIX}: {name} needs a non-empty 'text' string (the whole model)."
            if name == "revise_model" and not state.rep.get("text"):
                note = " (no earlier model existed; this is the first version)"
            else:
                note = ""
            state.rep = {"text": text[:MAX_TEXT]}
            state.versions = getattr(state, "versions", 0) + 1
            state.log.append((name, {"chars": len(text)}, "saved"))
            return self._report(state) + note + ("\n(text truncated to 40000 characters)" if len(text) > MAX_TEXT else "")
        if name == "show_model":
            text = state.rep.get("text", "")
            if not text:
                return f"{ERROR_PREFIX}: no model has been written yet; use write_model."
            return "\n".join(f"{i:3d} | {ln}" for i, ln in enumerate(text.splitlines(), 1))
        return f"{ERROR_PREFIX}: unknown tool {name!r}."

    def load(self, rep, given_gold=False):
        st = self.new_state()
        if isinstance(rep, str):
            rep = {"text": rep}
        st.rep = {"text": (rep or {}).get("text", "")}
        st.versions = 1 if st.rep["text"] else 0
        st.given_gold = given_gold
        return st

    def export(self, state):
        return {"text": state.rep.get("text", "")}

    def validity(self, state):
        a = analyse(state.rep.get("text", ""))
        tagged = sum(a["tags"].values())
        errors = [] if a["elements"] else ["no model elements written"]
        holes = [{"line": i, "element": t, "problem": "no provenance tag"} for i, t in a["untagged"]]
        return {"usable": tagged > 0, "executable": False, "holes": holes, "errors": errors,
                "stats": {"elements": a["elements"], **a["tags"], "missing_sections": a["missing_sections"]}}

    def render(self, state):
        return state.rep.get("text", "")

    def strip_answers(self, rep):
        """Drop any section whose heading is answer-bearing (conclusions, answer, predictions...)."""
        rep = copy.deepcopy(rep if isinstance(rep, dict) else {"text": rep})
        out, skip = [], False
        for ln in rep.get("text", "").splitlines():
            h = HEADING_RE.match(ln)
            if h and not ELEMENT_RE.match(ln):
                skip = bool(ANSWER_HEADING.search(h.group(1) or h.group(2) or ""))
            if not skip:
                out.append(ln)
        return {"text": "\n".join(out)}
