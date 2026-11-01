"""Frontier referee for the grader test (V3-004): Claude Opus, via the subscription, marks every sampled answer once
with the same v2 rubric and key points. It replaces the human spot check as the reference. No v3 Claude answers are
in the sample (they are v2 OpenRouter answers), so the referee never grades its own work.

    python -m v3.grading.referee            # writes runs/grader_test/referee.json and report_referee.json
"""
import json
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from v3.builders.claude_builder import claude_exe
from v3.config import CFG
from v3.grading.grader_test import OUT, sample, summarise


def claude_client(model):
    """A harness.judge client backed by headless Claude (no tools); cost is subscription, recorded as 0."""
    def client(_model, msgs, **kw):
        system = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        user = "\n".join(m["content"] for m in msgs if m["role"] == "user")
        user += "\n\nReply with the JSON object only, no prose and no code fences."
        work = tempfile.mkdtemp(prefix="v3r_")
        try:
            p = subprocess.run([claude_exe(), "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config",
                                "--setting-sources", "project", "--no-session-persistence", "--model", model,
                                "--system-prompt", system],
                               input=user, cwd=work, capture_output=True, text=True, encoding="utf-8", timeout=600)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        try:
            text = json.loads(p.stdout).get("result", "")
        except ValueError:
            text = p.stdout
        m = re.search(r"\{.*\}", text, re.S)
        return {"content": m.group(0) if m else text}, {"cost": 0.0}
    return client


def run(model=None, workers=4):
    from v3 import items as I
    from harness.judge.grade import grade_once
    from harness.strip import grader_text
    model = model or CFG["models"]["designer_claude"]
    smp = sample(100)
    items = {iid: I.v2_item(iid) for _, iid, _ in smp}
    client = claude_client(model)

    def one(x):
        key, iid, raw = x
        try:
            g = grade_once(items[iid], items[iid]["context"], grader_text(raw), 7, purpose="v3_referee",
                           model=model, client=client)
            return key, g.get("score"), g.get("status")
        except Exception as e:
            return key, None, f"error: {type(e).__name__}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(one, smp))
    ref = {k: s for k, s, _ in res if s is not None}
    (OUT / "referee.json").write_text(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model,
                                                  "marks": ref, "failed": [k for k, s, _ in res if s is None]},
                                                 indent=1), encoding="utf-8")
    marks = json.loads((OUT / "marks.json").read_text(encoding="utf-8"))
    rep = summarise(marks, ref)
    (OUT / "report_referee.json").write_text(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(ref),
                                                         "reference": f"referee:{model}", "candidates": rep},
                                                        indent=1), encoding="utf-8")
    return rep


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
