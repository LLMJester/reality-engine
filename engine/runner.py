"""Running (candidate, item) pairs: records on disk, parallel workers, retries and the failure rules.

Every pair runs once per record folder. A pair met before in the same folder is reused, never rerun; a failed run is
kept beside the new one (record.void<N>.json), never overwritten. The failure kinds come from the pack:
  design  scored 0 after the retries (zero.json), never dropped
  infra   retried with a backoff, then the whole run stops for a person (Halted); never scored
  config  the run stops at once
A PAUSE file in the work folder stops new runs; running ones finish.
"""
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from engine.pack import Paused


class Halted(RuntimeError):
    """A failure that a person must look at: the run stops, nothing is scored or dropped."""


class Store:
    def __init__(self, work):
        self.work = work
        self.root = work / "records"

    def dir(self, folder, cid, item):
        return self.root / folder / cid / item

    def get(self, folder, cid, item):
        try:
            return json.loads((self.dir(folder, cid, item) / "record.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def score(self, folder, cid, item):
        """The run's score; 0.0 after a design-induced failure; None if it has not run (or must run again)."""
        r = self.get(folder, cid, item)
        if r and not r.get("error") and r.get("score") is not None:
            return float(r["score"])
        if (self.dir(folder, cid, item) / "zero.json").exists():
            return 0.0
        return None

    def put(self, folder, cid, item, rec):
        d = self.dir(folder, cid, item)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "record.json"
        if p.exists():
            p.rename(d / f"record.void{sum(1 for _ in d.glob('record.void*.json'))}.json")
        p.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")

    def zero(self, folder, cid, item, error):
        d = self.dir(folder, cid, item)
        d.mkdir(parents=True, exist_ok=True)
        (d / "zero.json").write_text(json.dumps({"reason": "design-induced failure after retries",
                                                 "error": str(error)[:500]}), encoding="utf-8")

    def scores(self, folder, designs, items):
        """{label: {item: score}} for designs {label: candidate}."""
        return {lab: {i: self.score(folder, c["id"], i) for i in items} for lab, c in designs.items()}

    def records(self, folder, cid, items):
        return {i: self.get(folder, cid, i) for i in items}


def job_order(designs, items, seed="engine-order"):
    """Items in a fixed shuffled order, designs rotated per item: a paused stage has every design at about the same
    number of items done."""
    its = list(items)
    random.Random(seed).shuffle(its)
    labs = list(designs)
    return [(labs[(n + j) % len(labs)], i) for n, i in enumerate(its) for j in range(len(labs))]


def run_jobs(pack, store, folder, designs, items, stage, progress, workers=8, retries=2, backoff=(60, 300),
             pause_file=None, on_progress=None):
    """Run every missing (design, item) pair of `folder`. Raises Paused or Halted; returns the number done."""
    todo = [(lab, i) for lab, i in job_order(designs, items) if store.score(folder, designs[lab]["id"], i) is None]
    total = len(designs) * len(items)
    progress.update(stage=stage, planned=total, done=total - len(todo), t0=time.time())
    progress.pop("halt", None)
    stop, lock = threading.Event(), threading.Lock()
    why = {}

    def one(job):
        lab, iid = job
        cand = designs[lab]
        last, kind = None, None
        for attempt in range(1 + retries):
            if stop.is_set() or (pause_file and pause_file.exists()):
                stop.set()
                return
            try:
                rec = pack.run(cand, iid, {"stage": stage, "folder": folder,
                                           "dir": store.dir(folder, cand["id"], iid)})
            except Exception as e:                       # Paused from a pack's guard, or a crash
                if type(e).__name__ == "Paused":
                    with lock:
                        why.setdefault("pause", str(e))
                    stop.set()
                    return
                rec = {"error": f"{type(e).__name__}: {e}", "failure": "infra"}
            last = rec
            if not rec.get("error") and rec.get("score") is not None:
                kind = None
                break
            kind = rec.get("failure") or "infra"
            if kind == "config":
                break
            if kind == "infra" and attempt < retries:
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
        with lock:
            if kind is None:
                store.put(folder, cand["id"], iid, dict(last, design=lab, item=iid, stage=stage))
            elif kind == "design":
                store.put(folder, cand["id"], iid, dict(last, design=lab, item=iid, stage=stage))
                store.zero(folder, cand["id"], iid, last.get("error"))
            else:
                why.setdefault("halt", f"{kind} failure on {lab} x {iid}: {str(last.get('error'))[:300]}")
                stop.set()
                return
            progress["done"] += 1
            if on_progress and progress["done"] % 20 == 0:
                on_progress()

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo))
    if why.get("halt"):
        progress["halt"] = why["halt"]
        raise Halted(why["halt"])
    if stop.is_set():
        raise Paused(why.get("pause") or "PAUSE file present")
    missing = [(lab, i) for lab, i in job_order(designs, items) if store.score(folder, designs[lab]["id"], i) is None]
    if missing:                                           # never hand over with a hole in the data
        raise Halted(f"{len(missing)} pairs in {stage} have no score (e.g. {missing[0]})")
    return progress["done"]
