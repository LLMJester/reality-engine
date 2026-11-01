"""C2: unrestricted Python tool (clingo, z3, numpy installed), under the shared effort budget.
Code runs in the existing C2 sandbox (harness.conditions.runners.run_python)."""
from tournament.arms.base import Arm, ArmState, ERROR_PREFIX

# Imported lazily by _default_runner; listed so the final lock covers it (harness.orchestrator._import_closure).
RUNTIME_IMPORTS = ["harness.conditions.runners"]
# run_python launches harness/sandbox_runner.py in a subprocess. runners.py is a PREREG.LOCK frozen file and cannot
# carry its own declaration, so its runtime dependency is declared here, by the module that imports it
# (kept in sync by tests/test_recheck_lock.py).
RUNTIME_DEPS = ["harness/sandbox_runner.py"]

PY_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run Python 3 code in a fresh process. Returns stdout and stderr (truncated to 6000 "
                       "characters). clingo, z3-solver and numpy are installed.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
}


def _default_runner(code):
    from harness.conditions.runners import run_python
    return run_python(code)


class C2(Arm):
    name = "C2"
    buildable = False
    executable = False
    guide_file = "c2_guide.md"

    def __init__(self, runner=None):
        self.runner = runner or _default_runner

    def tools(self):
        return [PY_TOOL]

    def new_state(self):
        st = ArmState()
        st.rep = {"cells": []}
        return st

    def handle(self, state, name, args):
        if name != "python":
            return f"{ERROR_PREFIX}: unknown tool {name!r}."
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return f"{ERROR_PREFIX}: python needs a non-empty 'code' string."
        out = self.runner(code)
        state.rep.setdefault("cells", []).append({"code": code, "output": out[:3000]})
        return out if out.strip() else "(no output)"

    def validity(self, state):
        cells = state.rep.get("cells", [])
        return {"usable": bool(cells), "executable": False, "holes": [], "errors": []}
