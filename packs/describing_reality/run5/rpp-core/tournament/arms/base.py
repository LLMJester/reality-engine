"""Common Arm interface for the representation tournament (see tournament/DESIGN.md).

Every arm exposes tools for incremental construction/use, executes tool calls without raising
(errors come back as text with a location and a hint), and supports graceful degradation: broken
parts of a representation become `holes` while the rest stays usable.
"""
import copy
import pathlib
from dataclasses import dataclass, field

PROMPTS = pathlib.Path(__file__).resolve().parent.parent / "prompts"
ERROR_PREFIX = "ERROR"    # tool results that report a failure start with this (probes count them as failed runs)
ANSWER_KEYS = ("claims", "answers", "answer", "conclusions", "conclusion")

BUDGET_CALLS = 6          # model calls per item (same for every arm, A0+ and C2 included)
BUDGET_OUTPUT_TOKENS = 24000


@dataclass
class ArmState:
    """Mutable per-episode state; arms may subclass or store their representation in `rep`."""
    rep: dict = field(default_factory=dict)
    log: list = field(default_factory=list)      # (tool, args, result-summary) for the record
    given_gold: bool = False                      # True in E mode: representation pre-loaded, read-only


def strip_answer_fields(obj):
    """Generic answer stripping: drop answer-bearing keys at any depth (used by gold.py)."""
    if isinstance(obj, dict):
        return {k: strip_answer_fields(v) for k, v in obj.items() if k not in ANSWER_KEYS}
    if isinstance(obj, list):
        return [strip_answer_fields(v) for v in obj]
    return obj


class Arm:
    name = "base"
    buildable = False
    executable = False
    guide_file = None             # default: prompts/<name lower>_guide.md
    mutating_tools = ()           # tool names that build/modify; refused by the runner in given-gold mode
    custom_episode = False        # True: the arm drives its own episode via run_episode(ep) (A0, A0+)

    def guide(self) -> str:
        """Representation guide shown to the model (prompts/<name>_guide.md)."""
        f = self.guide_file or f"{self.name.lower()}_guide.md"
        p = PROMPTS / f
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def is_mutating(self, tool_name) -> bool:
        return tool_name in self.mutating_tools

    def current_tools(self, state) -> list:
        """Tools offered at this point of the episode (A7 changes them after `choose`)."""
        return self.tools()

    def gold_tools(self, state=None) -> list:
        """Tools offered in given-gold mode: every tool that does not build or modify."""
        tools = self.current_tools(state) if state is not None else self.tools()
        return [t for t in tools if not self.is_mutating(t["function"]["name"])]

    def current_probe_tools(self, state) -> list:
        return self.probe_tools()

    def strip_answers(self, rep):
        """Remove answer-bearing content from a gold representation (gold.py). Override if needed."""
        return strip_answer_fields(copy.deepcopy(rep))

    def tools(self) -> list:
        """OpenAI-style tool schemas for construction and use."""
        return []

    def new_state(self) -> ArmState:
        return ArmState()

    def handle(self, state: ArmState, name: str, args: dict) -> str:
        """Execute one tool call; never raise. Return a concise text result."""
        raise NotImplementedError

    def export(self, state: ArmState) -> dict:
        """Serialisable built representation (for buildability, fidelity, gold comparison)."""
        return state.rep

    def load(self, rep: dict, given_gold: bool = False) -> ArmState:
        st = self.new_state()
        st.rep = rep
        st.given_gold = given_gold
        return st

    def validity(self, state: ArmState) -> dict:
        """{"usable": bool, "executable": bool, "holes": [...], "errors": [...]}"""
        return {"usable": bool(state.rep), "executable": False, "holes": [], "errors": []}

    def probe_tools(self) -> list:
        """Tools a probe translator may call to run a what-if scenario against the representation."""
        return []

    def render(self, state: ArmState) -> str:
        """Human/LLM-readable rendering of the representation (shown in 'given gold' mode)."""
        import json
        return json.dumps(state.rep, indent=1)[:60000]
