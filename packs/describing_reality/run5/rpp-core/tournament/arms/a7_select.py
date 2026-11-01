"""A7: selector. The model first calls choose(arms=[1-2 of A1..A6], reason); the episode then continues
with the chosen arms' tools, combined, each name prefixed by its arm ("A5_check"). Arms are loaded
lazily from the registry, so arms built later plug in without changes here. Arms that cannot be
loaded are left out of the menu. No system-type labels are used anywhere."""
import copy

from tournament.arms import BUILDABLE, ArmUnavailable, available, get_arm
from tournament.arms.base import Arm, ArmState, ERROR_PREFIX

SEP = "_"
# Arms loaded lazily from the registry at run time (the default menu); the final lock covers their code and
# guides (harness.orchestrator._import_closure).
RUNTIME_ARMS = list(BUILDABLE)


def _prefixed(arm_name, tools):
    out = []
    for t in tools:
        t = copy.deepcopy(t)
        f = t["function"]
        f["name"] = f"{arm_name}{SEP}{f['name']}"
        f["description"] = f"[{arm_name}] " + f.get("description", "")
        out.append(t)
    return out


def split_name(name):
    if SEP in name:
        a, t = name.split(SEP, 1)
        return a, t
    return None, name


class A7State(ArmState):
    def __init__(self):
        super().__init__()
        self.chosen = []
        self.reason = ""
        self.subs = {}


class A7(Arm):
    name = "A7"
    buildable = True
    executable = False
    guide_file = "a7_guide.md"

    def __init__(self, menu=None):
        self._menu = menu   # None: every available arm of BUILDABLE

    def menu(self):
        return available(self._menu or BUILDABLE)

    def choose_tool(self):
        return {"type": "function", "function": {
            "name": "choose",
            "description": "Choose the representation(s) to build: one, or two to combine. Their guides and tools "
                           "become available after this call.",
            "parameters": {"type": "object", "properties": {
                "arms": {"type": "array", "items": {"type": "string", "enum": self.menu()}, "minItems": 1,
                         "maxItems": 2},
                "reason": {"type": "string"}}, "required": ["arms", "reason"]}}}

    def tools(self):
        return [self.choose_tool()]

    def new_state(self):
        return A7State()

    def current_tools(self, state):
        if state is None or not getattr(state, "chosen", None):
            return [self.choose_tool()]
        out = []
        for a in state.chosen:
            out += _prefixed(a, get_arm(a).tools())
        return out

    def current_probe_tools(self, state):
        out = []
        for a in getattr(state, "chosen", []):
            out += _prefixed(a, get_arm(a).current_probe_tools(state.subs[a]))
        return out

    def probe_tools(self):
        return []

    def is_mutating(self, tool_name):
        if tool_name == "choose":
            return True
        a, t = split_name(tool_name)
        try:
            return a is not None and get_arm(a).is_mutating(t)
        except ArmUnavailable:
            return False

    def handle(self, state, name, args):
        if name == "choose":
            if state.chosen:
                return f"{ERROR_PREFIX}: already chose {', '.join(state.chosen)}; use their tools."
            arms = args.get("arms")
            if isinstance(arms, str):
                arms = [arms]
            if not isinstance(arms, list) or not 1 <= len(arms) <= 2 or len(set(arms)) != len(arms):
                return f"{ERROR_PREFIX}: 'arms' must list one or two different representations from: " \
                       f"{', '.join(self.menu())}."
            menu = self.menu()
            bad = [a for a in arms if a not in menu]
            if bad:
                return f"{ERROR_PREFIX}: not available: {', '.join(map(str, bad))}. Choose from: {', '.join(menu)}."
            state.chosen = list(arms)
            state.reason = str(args.get("reason", ""))[:2000]
            for a in arms:
                state.subs[a] = get_arm(a).new_state()
                state.subs[a].given_gold = state.given_gold
            state.log.append(("choose", {"arms": arms}, state.reason[:200]))
            parts = [f"Chosen: {', '.join(arms)}. Their tools are now available, named with the prefix "
                     f"{' / '.join(a + SEP for a in arms)}. Guides follow."]
            for a in arms:
                parts.append(f"\n=== {a} guide ===\n{get_arm(a).guide()}")
            return "\n".join(parts)
        a, t = split_name(name)
        if a not in getattr(state, "chosen", []):
            return f"{ERROR_PREFIX}: unknown tool {name!r}; call choose first." if not state.chosen else \
                f"{ERROR_PREFIX}: unknown tool {name!r}."
        try:
            return get_arm(a).handle(state.subs[a], t, args)
        except Exception as e:
            return f"{ERROR_PREFIX}: {name} failed: {type(e).__name__}: {str(e)[:300]}"

    def export(self, state):
        return {"chosen": list(state.chosen), "reason": state.reason,
                "reps": {a: get_arm(a).export(state.subs[a]) for a in state.chosen}}

    def load(self, rep, given_gold=False):
        st = self.new_state()
        st.given_gold = given_gold
        rep = rep or {}
        st.chosen = [a for a in rep.get("chosen", []) if a in self.menu()]
        st.reason = rep.get("reason", "")
        for a in st.chosen:
            st.subs[a] = get_arm(a).load(rep.get("reps", {}).get(a, {}), given_gold=given_gold)
        st.rep = rep
        return st

    def validity(self, state):
        if not state.chosen:
            return {"usable": False, "executable": False, "holes": [], "errors": ["no representation chosen"],
                    "per_arm": {}}
        per = {a: get_arm(a).validity(state.subs[a]) for a in state.chosen}
        return {"usable": any(v.get("usable") for v in per.values()),
                "executable": any(v.get("executable") for v in per.values()),
                "holes": [dict(h, arm=a) if isinstance(h, dict) else {"arm": a, "hole": h}
                          for a, v in per.items() for h in v.get("holes", [])],
                "errors": [f"{a}: {e}" for a, v in per.items() for e in v.get("errors", [])],
                "per_arm": per, "chosen": list(state.chosen)}

    def render(self, state):
        return "\n\n".join(f"=== {a} ===\n{get_arm(a).render(state.subs[a])}" for a in state.chosen)

    def strip_answers(self, rep):
        rep = copy.deepcopy(rep)
        rep["reps"] = {a: get_arm(a).strip_answers(r) for a, r in rep.get("reps", {}).items()}
        return rep
