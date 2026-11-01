"""v3 orchestrator entry point (spec §11).

    python -m v3.run status
    python -m v3.run baseline            # generation 0: seeds × full set × both builders
    python -m v3.run loop                # generations until the stopping rule, PAUSE, or budget
    python -m v3.run pilot               # one small generation; results discarded unless the human decides otherwise

Run 3 (V3-034, config evolution.mode = "directed"): the loop runs v3.directed rounds, and when the stopping rule is
met it runs the built-in finals (v3.final3: final lock, accuracy, interaction, transfer, analysis) unless
evolution.directed.auto_final is false. The finals resume from their records after a pause, like a generation.

Single writer: the state store's lock refuses a second orchestrator. `state/PAUSE` stops the loop between episodes
and generations; the watchdog never restarts a paused programme. Claude builder runs are throttled to the configured
runs per hour.
"""
import argparse
import json
import sys
import threading
import time

from v3.config import CFG, STATE_DIR
from v3.state import Store

PAUSE = STATE_DIR / "PAUSE"
LOCK_FILE = STATE_DIR / "PREREG.LOCK"


class Paused(RuntimeError):
    pass


class Throttle:
    """At most `per_hour` starts in any rolling hour (shared by all builder threads)."""

    def __init__(self, per_hour):
        self.per_hour, self.starts, self.lock = per_hour, [], threading.Lock()

    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                self.starts = [t for t in self.starts if now - t < 3600]
                if len(self.starts) < self.per_hour:
                    self.starts.append(now)
                    return
                delay = 3600 - (now - self.starts[0]) + 1
            time.sleep(min(delay, 60))


PILOT_SEEDS = ("A0P", "G1C3", "SEC_A1", "ONT_BASIC")
PILOT_LADDER = {"quick": {1: 1, 2: 1, 4: 2, 8: 1}, "screen": {1: 1, 2: 2, 4: 2, 8: 1}, "full": {1: 2, 2: 2, 4: 2, 8: 2},
                "micro": {1: 3, 2: 3, 4: 2, 8: 2}}


PILOT3_SEEDS = ("A0P", "SEC_A1", "ONT_LOOSE", "SEC_FLOW", "G1C3", "A5", "FACTS", "STATES", "SPEC")
PILOT3_START = 1_930_000        # run 3 pilot items (validation split, inside the calibration range's spare bases)


def pilot_sets():
    """A shake-down sized slice: 5 quick, 6 screen, full = 2 v2 dev + 2 composite items per k. Run 3 adds two
    extension sets of 2 bases x 4 k."""
    from v3 import items as I
    from v3.directed import directed, shared_base_set
    from v3.evolve import item_sets
    if directed():
        bases = iter(I.ladder_seeds(5 + 6 + 8 + 4, start=PILOT3_START))
        lad = {s: v for s, v in PILOT_LADDER.items() if s != "micro"}
        s = {name: [f"K{k}x{next(bases)}" for k in sorted(lad[name]) for _ in range(lad[name][k])]
             for name in ("quick", "screen", "full")}
        rest = [next(bases) for _ in range(4)]
        return dict(s, full=I.v2_dev_ids()[:2] + s["full"], ext1=shared_base_set(rest[:2]),
                    ext2=shared_base_set(rest[2:]))
    s = item_sets(ladder=PILOT_LADDER, dev=False)
    return dict(s, full=I.v2_dev_ids()[:2] + s["full"])


def make_runner(store, throttle, prefix=""):
    from v3 import items as I
    from v3.episode import run as run_episode
    cache, cache_lock = {}, threading.Lock()

    def item(iid):
        with cache_lock:
            if iid not in cache:
                cache[iid] = I.load(iid)
            return cache[iid]

    def runner(g, iid, builder, stage):
        if PAUSE.exists():
            raise Paused("state/PAUSE present")
        return run_episode(g, item(iid), builder, f"{prefix}gen{store.data.get('generation', 0)}_{stage}")
    return runner


def run4():
    return CFG["evolution"].get("mode") == "directed4"


def carryover_seeds(path=None):
    """Run 4 (V3-037): run 3's near-winners as starting designs, genes unchanged; each keeps its run 3 family (the
    starting design it descends from) for the family rules. V3-039: dropped from the run (config `carryover` null;
    the file stays on record as `carryover_dropped`), so this returns {} unless a path is given."""
    from v3.config import ROOT
    from v3.genome.schema import make
    path = path or (CFG["evolution"].get("directed") or {}).get("carryover")
    if not path:
        return {}
    out = {}
    for name, x in json.loads((ROOT / path).read_text(encoding="utf-8")).items():
        g = make(x["genes"], [], "seed", 0, 0, notes=f"carried over from run 3 ({x['run3_id']}, {x['run3_operator']} "
                                                     f"child in the {x['run3_root_name']} family)")
        g.update(family=x["run3_root"], carryover={"run3_id": x["run3_id"], "run3_name": name})
        out[f"R3_{name.replace(' ', '_').upper()}"] = g
    return out


def evolution(store, pilot=False):
    from v3.directed import Directed, directed, run3_sets
    from v3.evolve import Evolution, run_sets
    sets = store.data.get("item_sets")
    if not sets:
        if run4():
            from v3.directed4 import pilot4_sets, run4_sets
            sets = store.data["item_sets"] = pilot4_sets() if pilot else run4_sets()
        else:
            sets = store.data["item_sets"] = pilot_sets() if pilot else run3_sets() if directed() else run_sets()
        store.save()
    elif directed() and not run4() and not pilot and "ext3" not in sets:      # V3-036: a run started before ext3 gains it, unchanged
        fixed = run3_sets()
        if any(sets.get(k) != fixed[k] for k in ("quick", "screen", "full", "ext1", "ext2")):
            raise RuntimeError("stored item sets differ from run3_sets(): refusing to add ext3")
        sets["ext3"] = fixed["ext3"]
        store.save()
    cls = Directed if directed() else Evolution
    if run4():
        from v3.directed4 import Directed4
        cls = Directed4
    return cls(store, make_runner(store, Throttle(CFG["budget"]["claude_runs_per_hour"]),
                                  "pilot_" if pilot else ""), sets, log=lambda m: store.event(m))


def status(store):
    d = store.data
    return {"generation": d.get("generation"), "genomes": len(d.get("genomes", {})), "front": d.get("front"),
            "streak": d.get("streak"), "paused": PAUSE.exists(), "prereg_locked": LOCK_FILE.exists(),
            "last_events": d.get("events", [])[-5:]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "baseline", "loop", "pilot"])
    a = ap.parse_args(argv)
    # the pilot keeps its own state (state/pilot/) and records (runs/pilot_*), so the real run starts clean
    store = Store(STATE_DIR / "pilot" / "state.json") if a.cmd == "pilot" else Store()
    if a.cmd == "status":
        print(json.dumps(status(store), indent=1))
        return 0
    if a.cmd in ("baseline", "loop"):
        from v3.prereg import verify
        changed = verify()
        if changed:
            print(f"refusing: pre-registration lock missing or broken (spec M10): {changed[:10]}")
            return 2
    store.acquire()
    try:
        from v3.genome.seeds import load_all
        seeds = load_all()
        if CFG["evolution"].get("seeds") and a.cmd != "pilot":
            seeds = {k: v for k, v in seeds.items() if k in CFG["evolution"]["seeds"]}
            seeds.update(carryover_seeds() if run4() else {})
        if a.cmd == "pilot":
            from v3.directed import directed
            CFG["evolution"].update(children_per_generation=10 if directed() else 6, screen_items=6, full_top_k=2)
            if directed():
                CFG["evolution"]["directed"].update(confirm_bases=2, confirm_start=900_000)  # not the real run's range
            if run4():
                from v3.directed4 import PILOT_STAGE_N
                CFG["evolution"].update(children_per_generation=8, full_top_k=2)
                CFG["evolution"]["directed"].update(stage_n=PILOT_STAGE_N, practice_bases=1)
            seeds = {k: v for k, v in seeds.items() if k in (PILOT3_SEEDS if directed() else PILOT_SEEDS)}
            if run4():          # the run 4 pilot: four starting designs plus two carried over from run 3
                seeds = {k: v for k, v in seeds.items() if k in ("A0P", "FACTS", "SEC_A1", "STATES")}
                dc = CFG["evolution"]["directed"]
                seeds.update(dict(list(carryover_seeds(dc.get("carryover") or dc.get("carryover_dropped")).items())[:2]))
            store.data["pilot"] = True
        ev = evolution(store, pilot=a.cmd == "pilot")
        if store.data.get("pilot") and a.cmd != "pilot":
            print("refusing: this state holds a pilot run; start the real run from a fresh state (spec §15.4)")
            return 2
        if a.cmd in ("baseline", "pilot") or not store.data.get("controls"):
            ev.baseline(seeds)
        if a.cmd == "pilot":
            ev.generation(1)
            return 0
        gen = store.data.get("generation", 0)
        while not ev.should_stop():
            if PAUSE.exists():
                store.event("paused by human (state/PAUSE)")
                break
            gen += 1
            ev.generation(gen)
        else:
            store.event("stopping rule met: evolution complete")
            from v3.directed import cfg as dcfg, directed
            if directed() and dcfg().get("auto_final", True):
                store.save()
                from v3 import final3
                final3.run_all(store)
    except Paused as e:
        why = "by human" if "state/PAUSE" in str(e) else f"({e})"   # an allowance guard says which and why
        store.event(f"paused {why} during a generation; resumes from stored records")
    finally:
        store.save()
        store.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
