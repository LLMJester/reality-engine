"""A0: natural language, single call (identical to C1).
A0+: natural language, effort-matched: answer, then up to N review/revise calls within the shared
budget (same call and output-token budget as the build-and-use arms). No tools, nothing built."""
import re

from tournament.arms.base import Arm, ArmState

FINAL_RE = re.compile(r"FINAL ANSWER", re.I)
NO_CHANGE_RE = re.compile(r"^\W*NO CHANGE\W*$", re.I)

REVIEW_MSG = ("Review your answer above against the system description and the question. If anything in it is "
              "wrong, missing or unsupported by the description, write a corrected answer, ending with a line "
              "containing only the words FINAL ANSWER followed by your full revised answer in plain prose. If it "
              "needs no change, reply with the single line NO CHANGE.")


class A0(Arm):
    name = "A0"
    buildable = False
    executable = False
    guide_file = "a0_guide.md"
    custom_episode = True

    def handle(self, state, name, args):
        return "ERROR: this arm has no tools."

    def export(self, state):
        return {}

    def validity(self, state):
        return {"usable": False, "executable": False, "holes": [], "errors": ["no representation (natural language)"]}

    def run_episode(self, ep, state, messages):
        """Single answer call. Returns (final_text, messages, trace)."""
        m = ep.call(messages, tools=None, label="answer", final=True)
        text = m.get("content") or ""
        messages.append({"role": "assistant", "content": text})
        return text, messages, []


class A0Plus(A0):
    name = "A0+"
    max_reviews = 5   # N: review/revise calls after the first answer (still bounded by the shared budget)

    def run_episode(self, ep, state, messages):
        trace = []
        m = ep.call(messages, tools=None, label="answer", final=False)
        final = m.get("content") or ""
        messages.append({"role": "assistant", "content": final})
        for k in range(self.max_reviews):
            if not ep.can_answer():
                break
            messages.append({"role": "user", "content": REVIEW_MSG})
            m = ep.call(messages, tools=None, label=f"review{k + 1}", final=False)
            text = m.get("content") or ""
            messages.append({"role": "assistant", "content": text})
            if NO_CHANGE_RE.match(text.strip()) or not text.strip():
                trace.append({"review": k + 1, "outcome": "no change"})
                break
            if FINAL_RE.search(text):
                final = text
                trace.append({"review": k + 1, "outcome": "revised"})
            else:   # e.g. truncated: keep the previous complete answer
                trace.append({"review": k + 1, "outcome": "ignored (no FINAL ANSWER line)"})
        return final, messages, trace
