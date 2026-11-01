"""Information walls (spec §10): what each agent role may not see.

The v2 sealed held-out and generalisation sets were never opened; v3 uses them as its final test, so every v3 agent is
walled off from them (and from answer keys) exactly as in v2. Paths are given both relative to the v2 root and as
absolute paths, because v3 agents run with the v3 repository as their working directory.
"""
import re

from v3.config import V2

# v2-relative paths no agent may read before the final hash lock
SEALED = ["data/heldout", "data/generalisation", "data/validation/answers", "data/validation/adjudication",
          "data/validation/audit", "data/real", "runs/heldout1", "runs/tournament/final_"]
# answer-bearing material: builders and designers never see it (graders and the scripted checker do, in code)
KEYED = ["data/items", "data/simulators", "runs/tournament"]

KEY_BLIND_ROLES = {"builder", "designer", "operator", "qa_fix"}


def _abs(rel):
    return (V2 / rel).as_posix()


def deny_rules(role):
    """Claude Code permission deny rules for an agent of this role."""
    paths = list(SEALED) + (list(KEYED) if role in KEY_BLIND_ROLES else [])
    rules = []
    for rel in paths:
        for base in (_abs(rel), "../rpp-core/" + rel):
            glob = base + ("*/**" if rel.endswith("_") else "/**")
            rules += [f"Read({glob})", f"Edit({glob})", f"Write({glob})"]
    rules += ["WebFetch", "WebSearch", "Bash(git push:*)"]
    return rules


def forbidden_patterns(role):
    """Regexes for the transcript scan: a forbidden path in a tool call or listing means the walls leaked."""
    paths = list(SEALED) + (list(KEYED) if role in KEY_BLIND_ROLES else [])
    return [re.compile(re.escape(p).replace("/", r"[/\\]+"), re.I) for p in paths]
