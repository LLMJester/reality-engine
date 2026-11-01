"""Pre-registration lock (spec §8.5, M10): hash everything that decides the experiment before generation 1.

Locked: config (objectives, operators, stopping rule, budgets, grading threshold, models), every v3 Python module
except the dashboard (display only), seed genomes, the fixed item-set ids, PREREG.md, and the v2 files v3 executes
(arms, episode runner, grader, rubric, world generator). `verify()` lists every changed or missing file; the orchestrator
refuses to run if anything changed (a legitimate change needs a new, recorded lock: python -m v3.prereg lock --reason).
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time

from v3.config import ROOT, STATE_DIR, V2

LOCK = STATE_DIR / "PREREG.LOCK"
V2_FILES = ["tournament/arms/*.py", "tournament/arms/registry.d/*", "tournament/episode.py", "tournament/run.py",
            "tournament/prompts/*.md", "harness/judge/grade.py", "harness/judge/rubric.md", "harness/strip.py",
            "harness/llm.py", "discovery/world/**/*.py"]


def files():
    out = [ROOT / "config" / "v3.json", ROOT / "PREREG.md"]
    # dashboards only read records, so they are never locked (V3-054: "dashboard5" too, not only "dashboard")
    out += sorted(p for p in (ROOT / "v3").rglob("*.py")
                  if not any(part.startswith("dashboard") for part in p.parts) and "__pycache__" not in p.parts)
    out += sorted((ROOT / "v3" / "genome" / "seeds").glob("*.json"))
    out += [p for p in [ROOT / "docs" / "seeds_run4_carryover.json"] if p.exists()]     # V3-037: run 4 starting designs
    # V3-042 (run 5): the design, the hand-written items, the sandbox the own-code founders run in
    out += [p for p in [ROOT / "docs" / "RUN5_DESIGN.md", ROOT / "items" / "hand5" / "items.json"] if p.exists()]
    # V3-058: the stages design and the frozen stage item sets
    out += [p for p in [ROOT / "docs" / "RUN5_STAGES_DESIGN.md", ROOT / "items" / "run5_stages.json"] if p.exists()]
    out += sorted(p for p in (ROOT / "sandbox").glob("*") if p.is_file())
    for pat in V2_FILES:
        out += sorted(p for p in V2.glob(pat) if p.is_file() and "__pycache__" not in p.parts)
    return out


def _rel(p):
    p = pathlib.Path(p)
    try:
        return "v3:" + p.relative_to(ROOT).as_posix()
    except ValueError:
        return "v2:" + p.relative_to(V2).as_posix()


def digest(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def snapshot(item_sets):
    h = {_rel(p): digest(p) for p in files() if p.exists()}
    h["item_sets"] = hashlib.sha256(json.dumps(item_sets, sort_keys=True).encode()).hexdigest()
    return h


def lock(item_sets, reason):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    prev = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else None
    doc = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason, "hashes": snapshot(item_sets),
           "item_sets": item_sets, "previous": prev and {"t": prev["t"], "reason": prev["reason"]}}
    LOCK.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def verify():
    """[] if nothing changed since the lock, else a list of changed/missing/new files."""
    if not LOCK.exists():
        return ["no lock"]
    doc = json.loads(LOCK.read_text(encoding="utf-8"))
    now = snapshot(doc["item_sets"])
    old = doc["hashes"]
    return sorted([k for k in old if now.get(k) != old[k]] + [f"new: {k}" for k in now if k not in old])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lock", "verify"])
    ap.add_argument("--reason", default="")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        ch = verify()
        print(json.dumps(ch, indent=1) if ch else "lock intact")
        return 1 if ch else 0
    if not a.reason:
        print("a lock needs --reason")
        return 2
    from v3.config import CFG
    from v3.directed import directed, run3_sets
    from v3.evolve import EV, run_sets
    if CFG.get("run") == 5:                         # V3-042: run 5's sets and the selected hand-written items
        from v3 import items as I
        from v3 import run5_sets as R5
        sets = dict(R5.sets(), hand5=sorted(I.hand5_ids(False)))
    elif EV.get("mode") == "directed4":                # V3-037: run 4's own item sets
        from v3.directed4 import run4_sets
        sets = run4_sets()
    else:
        sets = run3_sets() if directed() else run_sets()
    doc = lock(sets, a.reason)
    print(f"locked {len(doc['hashes'])} entries at {doc['t']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
