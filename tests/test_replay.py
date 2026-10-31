"""The replay test: run 5 rerun through the engine must reach run 5's numbers. Needs the run 5 records (rpp-v3, kept
private: they hold the sealed answers) and is skipped without them. With REPLAY_FRESH=1 it reruns the whole replay
into a temporary folder (a few minutes); otherwise it checks the replay already in work/run5-replay."""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN5 = pathlib.Path(os.environ.get("RPP_V3_HOME") or ROOT.parent / "rpp-v3") / "runs" / "run5" / "analysis.json"


@pytest.mark.skipif(not RUN5.exists(), reason="the run 5 records are not here")
def test_replay_matches_run5(tmp_path):
    work = ROOT / "work" / "run5-replay"
    if os.environ.get("REPLAY_FRESH") or not (work / "analysis.json").exists():
        raw = json.loads((ROOT / "plans" / "run5_replay.json").read_text(encoding="utf-8"))
        raw["work"] = str(tmp_path / "replay")
        plan = tmp_path / "plan.json"
        plan.write_text(json.dumps(raw), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "engine", "lock", str(plan), "--reason", "test"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "engine", "run", str(plan)], cwd=ROOT, check=True)
        work = tmp_path / "replay"
    r = subprocess.run([sys.executable, "scripts/compare_replay.py", "--work", str(work), "--run5", str(RUN5)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:]
