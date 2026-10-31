"""Step 2, lock the rules: the plan file, its defaults, and the lock that fingerprints it.

A plan is a JSON file that says which pack to use and every rule of the run: which item set each step uses, the
control, the screen bar, the keep rule's test family, the final check's order of rivals, the bootstrap seed. Before
the first run, `python -m engine lock <plan> --reason "..."` writes <work>/PLAN.LOCK: the sha256 of the plan and of
every source file of the engine and the pack. `python -m engine run` refuses to start unless the lock still matches,
so a rule cannot change after results exist without a new lock, and every lock is logged (LOCKS.log) with its reason.
"""
import copy
import hashlib
import importlib
import json
import pathlib
import time

ENGINE_DIR = pathlib.Path(__file__).resolve().parent
REPO = ENGINE_DIR.parent

DEFAULTS = {
    "name": None,
    "pack": None,                                  # "module.path:ClassName"
    "pack_options": {},
    "work": None,                                  # the work folder (default: work/<name> next to the plan's repo)
    "control": None,                               # default: the pack's control
    "baselines": [],                               # founders reported beside the others, never chosen
    "founders": None,                              # default: all the pack's founders
    "sets": {"compare": None, "compare_also": [], "takeapart": None, "rounds": [], "sealed": None},
    "takeapart": {"slots": None, "family": None},  # family: the Holm family (default: every part and pair)
    "buildout": {"slots": None, "per_slot": 4, "screen_gate": 0.015, "max_advance": 4, "converge": 0.005},
    "final": {"rivals": ["control", "runner_up"], "also_on": []},
    "run": {"workers": 8, "retries": 2, "backoff": [60, 300]},
    "stats": {"boot": 10_000, "seed": 20261028, "alpha": 0.05},
    "approval": "human",                           # "human": a person approves every set of new versions; "auto"
}


class PlanError(ValueError):
    pass


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path):
    path = pathlib.Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    plan = _merge(DEFAULTS, raw)
    plan["_path"] = str(path)
    if not plan["name"] or not plan["pack"]:
        raise PlanError("a plan needs a name and a pack")
    if not plan["sets"]["compare"]:
        raise PlanError("a plan needs sets.compare")
    plan["_work"] = str((REPO / plan["work"]).resolve() if plan.get("work") else REPO / "work" / plan["name"])
    return plan


def make_pack(plan):
    mod, _, cls = plan["pack"].partition(":")
    return getattr(importlib.import_module(mod), cls)(**(plan.get("pack_options") or {}))


def _pack_dir(plan):
    mod = plan["pack"].partition(":")[0]
    return pathlib.Path(importlib.import_module(mod).__file__).resolve().parent


def fingerprint(plan):
    """sha256 over the plan file and every .py file of the engine and the pack (sorted by path)."""
    files = {"plan": pathlib.Path(plan["_path"])}
    for d, tag in ((ENGINE_DIR, "engine"), (_pack_dir(plan), "pack")):
        for p in sorted(d.rglob("*.py")):
            if "__pycache__" not in p.parts:
                files[f"{tag}/{p.relative_to(d).as_posix()}"] = p
    h = hashlib.sha256()
    each = {}
    for name in sorted(files):
        b = files[name].read_bytes().replace(b"\r\n", b"\n")
        each[name] = hashlib.sha256(b).hexdigest()[:16]
        h.update(name.encode() + b"\0" + b + b"\0")
    return h.hexdigest(), each


def lock(plan, reason):
    work = pathlib.Path(plan["_work"])
    work.mkdir(parents=True, exist_ok=True)
    sha, each = fingerprint(plan)
    doc = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason, "sha256": sha, "files": each}
    (work / "PLAN.LOCK").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    with open(work / "LOCKS.log", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": doc["t"], "reason": reason, "sha256": sha}) + "\n")
    return doc


def verify(plan):
    """(ok, message). The run refuses to start unless ok."""
    p = pathlib.Path(plan["_work"]) / "PLAN.LOCK"
    if not p.exists():
        return False, f"no lock: run `python -m engine lock {plan['_path']} --reason ...` first"
    doc = json.loads(p.read_text(encoding="utf-8"))
    sha, each = fingerprint(plan)
    if sha == doc["sha256"]:
        return True, f"lock ok ({doc['t']}, {doc['reason']})"
    changed = sorted(set(each) ^ set(doc["files"]) | {k for k in each if doc["files"].get(k) != each[k]})
    return False, f"the plan or the code changed since the lock ({doc['t']}): {changed[:8]}"
