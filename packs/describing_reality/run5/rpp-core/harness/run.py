"""Run an evaluation: conditions × tiers × items, then blind grading.

python -m harness.run --run <name> --split dev|heldout|<comma ids> --conds C1,C2,C3,C4 --tiers small,medium,large
                      [--lang <dir with lang/ compiler/ checker/>] [--double-grade] [--workers 8]

Outputs runs/<name>/<tier>/<cond>/<item>.json (raw record + grade). Re-running resumes: finished records are kept.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from harness.conditions import runners as R
from harness.judge.grade import grade
from harness.llm import MODELS, BudgetExceeded, true_spent as spent

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEEDS = {"C1": 11, "C2": 11, "C3": 11, "C4": 11}
_print = threading.Lock()


def log(*a):
    with _print:
        print(*a, flush=True)


def load_item(iid):
    item = json.loads((DATA / "items" / f"{iid}.json").read_text(encoding="utf-8"))
    if item["bank"] == "real":
        ctx = (DATA / "real" / f"{iid}.description.md").read_text(encoding="utf-8")
    else:
        ctx = (DATA / "simulators" / item["system_id"] / "description.md").read_text(encoding="utf-8")
    return item, ctx


def split_ids(split):
    if split in ("dev", "heldout"):
        return json.loads((DATA / split / "items.json").read_text(encoding="utf-8"))
    return split.split(",")


def make_check_fn(lang_root):
    lang_root = pathlib.Path(lang_root)

    def check_fn(ast):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(ast, f)
            path = f.name
        try:
            r = subprocess.run([sys.executable, "-m", "checker.check", path], cwd=lang_root, capture_output=True,
                               text=True, timeout=180)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-500:])
            return json.loads(r.stdout)
        except subprocess.TimeoutExpired:
            return {"pass": False, "failures": [{"obligation": "well_formed", "claim_ids": [],
                                                 "explanation": "checker timed out (model too large to ground)"}],
                    "claims": {}, "sensitivity": [], "timeout": True}
        finally:
            pathlib.Path(path).unlink(missing_ok=True)
    return check_fn


def run_one(iid, cond, tier, out, lang_root, double, purpose):
    if out.exists():
        rec = json.loads(out.read_text(encoding="utf-8"))
        if rec.get("graded") and (not double or len(rec["grade"]["grades"]) >= 2):
            return rec
    else:
        rec = None
    item, ctx = load_item(iid)
    model = MODELS[tier]
    if rec is None or rec.get("error"):
        seed = SEEDS[cond]
        try:
            if cond == "C1":
                rec = R.c1(item, ctx, tier, model, seed, purpose)
            elif cond == "C2":
                rec = R.c2(item, ctx, tier, model, seed, purpose)
            else:
                rec = R.c34(item, ctx, tier, model, seed, purpose, pathlib.Path(lang_root) / "lang",
                            cond == "C4", make_check_fn(lang_root), out.parent.parent / "models" / f"{iid}.json")
        except BudgetExceeded:
            raise
        except Exception as e:
            rec = {"condition": cond, "tier": tier, "model": model, "error": f"{e}\n{traceback.format_exc()[-1500:]}",
                   "final_raw": "", "cost": 0.0}
        rec["item"] = iid
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if rec.get("error"):
        return rec
    rec["grade"] = grade(item, ctx, rec["final_raw"], double=double)
    rec["graded"] = True
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--conds", default="C1,C2,C3,C4")
    ap.add_argument("--tiers", default="small,medium,large")
    ap.add_argument("--lang", default=str(ROOT))
    ap.add_argument("--double-grade", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from harness.verify_lock import verify
    bad = verify()
    if bad:
        sys.exit(f"ABORT: held-out lock violated: {bad}")
    ids = split_ids(a.split)
    import random
    rng = random.Random(5)
    jobs = []
    for i in ids:
        cell = [(i, c, t) for t in a.tiers.split(",") for c in a.conds.split(",")]
        rng.shuffle(cell)
        jobs += cell
    base = ROOT / "runs" / a.run
    log(f"run {a.run}: {len(jobs)} jobs, spent so far ${spent():.2f}")
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_one, i, c, t, base / t / c / f"{i}.json", a.lang, a.double_grade, f"{a.run}:{c}:{t}"):
                (i, c, t) for i, c, t in jobs}
        for f in as_completed(futs):
            i, c, t = futs[f]
            done += 1
            try:
                rec = f.result()
                s = rec.get("grade", {}).get("score")
                log(f"[{done}/{len(jobs)}] {t} {c} {i} score={s if s is None else round(s, 2)} "
                    f"cost=${rec.get('cost', 0):.4f} err={'Y' if rec.get('error') else 'n'} total=${spent():.2f}")
            except BudgetExceeded as e:
                log(f"BUDGET EXCEEDED: {e}")
                ex.shutdown(cancel_futures=True)
                sys.exit(3)
            except Exception as e:
                log(f"[{done}/{len(jobs)}] {t} {c} {i} FAILED {e}")


if __name__ == "__main__":
    main()
