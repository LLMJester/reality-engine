"""The Reality Engine.

    python -m engine lock   <plan.json> --reason "why"   fingerprint the plan, the engine and the pack (step 2)
    python -m engine verify <plan.json>                  check the lock still matches
    python -m engine run    <plan.json>                  run steps 3 to 7 until done, or until it waits for a person
    python -m engine status <plan.json>                  where the run is, and the last events
    python -m engine report <plan.json>                  rewrite the trust report from the analysis

To pause, create the file PAUSE in the work folder: running jobs finish, nothing is lost, and `run` resumes.
"""
import argparse
import json
import sys

from engine import plan as P


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m engine")
    ap.add_argument("command", choices=["lock", "verify", "run", "status", "report"])
    ap.add_argument("plan")
    ap.add_argument("--reason", default="")
    a = ap.parse_args(argv)
    plan = P.load(a.plan)
    if a.command == "lock":
        if not a.reason:
            raise SystemExit("give a --reason for the lock")
        doc = P.lock(plan, a.reason)
        print(f"locked {plan['name']}: {doc['sha256'][:16]} ({len(doc['files'])} files)")
        return 0
    if a.command == "verify":
        ok, msg = P.verify(plan)
        print(msg)
        return 0 if ok else 1
    from engine.stages import Engine
    eng = Engine(plan, check_lock=a.command == "run")
    if a.command == "run":
        at = eng.loop()
        print(f"stopped at: {at}" + (f" (waiting for {eng.st['waiting_for']})" if eng.st.get("waiting_for") else ""))
        return 0
    if a.command == "report":
        from engine import report
        s = report.write(eng)
        print(json.dumps({k: s[k] for k in ("right", "verdict")}, indent=1))
        return 0
    st = eng.st
    print(f"{plan['name']}: step {st['step']}" + (f", round {st.get('round')}" if st["step"].startswith("round") else ""))
    p = st.get("progress") or {}
    if p:
        print(f"  {p.get('stage')}: {p.get('done')}/{p.get('planned')} runs")
    for k in ("waiting_for", "paused", "halted"):
        if st.get(k):
            print(f"  {k}: {st[k]}")
    for e in st.get("events", [])[-8:]:
        print(f"  {e['t']}  {e['msg']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
