"""Run 5 item sets (V3-042, docs/RUN5_DESIGN.md section 5) and the run 5 seal.

Generated sets are composite ladder items K<k>x<base>: every base appears at k = 1, 2, 4, 8 (shared bases), so items
cluster by base and every design meets every item. Bases are chosen by v3.items.ladder_seeds (round robin over
problem families) from ranges no earlier run used (scripts/check_item_ranges.py --run5 proves it).

  main         120 bases x 4 k = 480  validation split   stage 1 and stage 2 (V3-044: was 90 bases)
  sealed       180 bases x 4 k = 720  train split        the final check; sealed until opened once
  pilot        15 bases x 4 k = 60    train split        the format pilot (never in results)
  mini         2 bases: 6 items       train split        the mini end-to-end run (never in results)
  mini_sealed  1 base x 4 k = 4       train split        the mini run's sealed final check (tests the seal)

The seal: the sealed sets' bases (each base's window [base, base + 500)) cannot be loaded (v3.items raises Sealed) until `open_sealed(name)` records the opening in
state/run5/opened.json, which the run engine does once, at the final check, and never undoes. The test split is not
used: its test-only problem families would make the sealed set a different kind of test from the main set.

V3-058 (docs/RUN5_STAGES_DESIGN.md sections 8, 12, 12b): stages 3 to 5 and a fresh sealed set, chosen once by
allocate_stage_sets() and frozen in items/run5_stages.json (never rewritten; hashed into the pre-registration lock):

  s<n>_screen   80 bases x 4 k = 320   validation split   stage n's screen (n = 3, 4, 5)
  s<n>_confirm  120 bases x 4 k = 480  validation split   stage n's confirmation factorial
  sealed2       180 bases x 4 k = 720  train split        the final check, after the last stage; sealed until opened

The old sealed set was opened at 11:18 on 2026-10-30 and its final check stopped at 12:25 (V3-058): it is retired.
Its records stay on disk unchanged (manifest in state/run5/retired_V3_058.json) and no analysis may read them
(retired_items(); v3.analysis5 and v3.run5 raise Retired).
"""
import bisect
import functools
import hashlib
import json
import os
import re
import threading
import time

from v3.config import ROOT, STATE_DIR

KS = (1, 2, 4, 8)
STRIDE = 500
STARTS = {"main": 1_705_000, "sealed": 610_000, "pilot": 160_000, "mini": 320_000, "mini_sealed": 330_000}
N_BASES = {"main": 120, "sealed": 180, "pilot": 15, "mini": 2, "mini_sealed": 1}
SEALED = ("sealed", "mini_sealed", "sealed2")
OPENED = STATE_DIR / "run5" / "opened.json"
# V3-058: the stage sets and the fresh sealed set
STAGES = (3, 4, 5)
STAGE_BASES = {"screen": 80, "confirm": 120}
SEALED2_BASES = 180
STAGE_FILE = ROOT / "items" / "run5_stages.json"
RETIRED = ("sealed",)                      # opened, then retired unviewed (V3-058)
RETIRED_MANIFEST = STATE_DIR / "run5" / "retired_V3_058.json"
# never used for the new sets: calibration bases (records outside runs/), the hand-of-God side test's range, the
# seeds v2 (rpp-core) selected its designs on - micro panels from the start of the train and validation splits, meso
# from offset 500,000 with up to 50 x 1,000 skips (discovery/micro/panel.py, discovery/meso/panel.py) - and run 3's
# configured confirmation range (v3/directed.py CONFIRM_START, 30 bases scanned 4 deep; used by its tests)
EXCLUDED_RANGES = ((1_900_000, 2_000_000), (850_000, 907_000),
                   (0, 5_000), (1_000_000, 1_005_000), (500_000, 560_000), (1_500_000, 1_560_000),
                   (20_000, 80_500))


class Retired(RuntimeError):
    """An analysis tried to read a record of the retired sealed set (V3-058)."""


def stage_set_names():
    return [f"s{n}_{part}" for n in STAGES for part in STAGE_BASES] + ["sealed2"]


@functools.lru_cache(maxsize=None)
def _stage_file():
    try:
        return json.loads(STAGE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _range(name):
    return STARTS[name], STARTS[name] + 4 * N_BASES[name] * STRIDE + STRIDE   # ladder_seeds scans at most 4n bases


@functools.lru_cache(maxsize=None)
def sealed_windows():
    """{set name: sorted base seeds}; a sealed item's members come from [base, base + STRIDE)."""
    return {n: tuple(sorted(_bases(n))) for n in SEALED if n != "sealed2" or _stage_file()}


def opened():
    try:
        return json.loads(OPENED.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def is_sealed(seed):
    """True if the seed lies in a run 5 sealed range that has not been opened."""
    if _choosing.is_active():
        return False
    op = opened()
    for name, bases in sealed_windows().items():
        if name in op:
            continue
        j = bisect.bisect_right(bases, seed) - 1
        if j >= 0 and bases[j] <= seed < bases[j] + STRIDE:
            return True
    return False


def open_sealed(name, reason):
    """Open a sealed set once (recorded with time and reason; never undone)."""
    if name not in SEALED:
        raise ValueError(name)
    op = opened()
    if name in op:
        return op[name]
    op[name] = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason}
    OPENED.parent.mkdir(parents=True, exist_ok=True)
    OPENED.write_text(json.dumps(op, indent=1), encoding="utf-8")
    return op[name]


@functools.lru_cache(maxsize=None)
def _bases(name):
    """Base seeds of a set. Choosing them reads only each candidate's problem family (no item is built or shown)."""
    if name in stage_set_names():
        doc = _stage_file()
        if not doc:
            raise FileNotFoundError(f"{STAGE_FILE} is missing: run scripts/allocate_run5_stages.py once")
        return tuple(doc["sets"][name])
    from v3 import items as I
    with _choosing():
        return tuple(I.ladder_seeds(N_BASES[name], start=STARTS[name]))


# ---------------------------------------------------------------- V3-058: the stage sets, chosen once
def used_bases(root=ROOT):
    """Every base seed (K<k>x<base>) and single seed (W<seed>) that names a folder anywhere in rpp-v3 (records of
    runs 1-5, archives, side experiments)."""
    k_used, w_used = set(), set()
    for dp, dn, _ in os.walk(root):
        dn[:] = [d for d in dn if d not in (".git", "__pycache__", "node_modules")]
        for d in dn:
            m = re.match(r"^K\d+x(\d+)$", d)
            if m:
                k_used.add(int(m.group(1)))
            m = re.match(r"^W(\d+)$", d)
            if m:
                w_used.add(int(m.group(1)))
    return k_used, w_used


def free_windows(lo, hi, k_used, w_used, other_bases):
    """Base seeds b in [lo, hi) whose window [b, b + STRIDE) meets no used K window, no W seed, no base of another
    run 5 set and no excluded range; scanned upwards, each new window starting where the last one ended."""
    blocked = [(u - STRIDE + 1, u + STRIDE) for u in set(k_used) | set(other_bases)]
    blocked += [(s - STRIDE + 1, s + 1) for s in w_used]
    blocked += [(a - STRIDE + 1, b) for a, b in EXCLUDED_RANGES]
    blocked.sort()
    merged = []
    for a, b in blocked:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    starts = [m[0] for m in merged]
    out, b = [], lo
    while b + STRIDE <= hi:
        j = bisect.bisect_right(starts, b) - 1
        if j >= 0 and merged[j][0] <= b < merged[j][1]:
            b = merged[j][1]
            continue
        out.append(b)
        b += STRIDE
    return out


def round_robin(seeds, family_of):
    """Seeds in round-robin order over their problem families (as v3.items.ladder_seeds orders them)."""
    fams = {}
    for s in seeds:
        fams.setdefault(family_of(s), []).append(s)
    order, i = [], 0
    while any(i < len(v) for v in fams.values()):
        for f in sorted(fams):
            if i < len(fams[f]):
                order.append(fams[f][i])
        i += 1
    return order


def allocate_stage_sets(root=ROOT, family_of=None):
    """Choose the V3-058 sets: every free window of the validation split (stage sets) and of the train split
    (sealed2), in round-robin order over problem families, cut into the sets in a fixed order. Reads only each
    candidate's problem family. Returns the document written to STAGE_FILE."""
    import v3.genome.render                               # noqa: F401  (puts rpp-core's discovery on sys.path)
    from discovery.world import make_instance
    family_of = family_of or (lambda s: make_instance(s).family)
    k_used, w_used = used_bases(root)
    with _choosing():
        others = set()
        for name in N_BASES:
            others |= set(_bases(name))
    sets = {}
    val = round_robin(free_windows(1_000_000, 2_000_000, k_used, w_used, others), family_of)
    need = [(f"s{n}_{part}", nb) for n in STAGES for part, nb in STAGE_BASES.items()]
    if sum(nb for _, nb in need) > len(val):
        raise ValueError(f"only {len(val)} free validation windows for {sum(nb for _, nb in need)} stage bases")
    pos = 0
    for name, nb in need:
        sets[name] = sorted(val[pos:pos + nb])
        pos += nb
    train = round_robin(free_windows(0, 1_000_000, k_used, w_used, others), family_of)
    if len(train) < SEALED2_BASES:
        raise ValueError(f"only {len(train)} free train windows for {SEALED2_BASES} sealed bases")
    sets["sealed2"] = sorted(train[:SEALED2_BASES])
    body = json.dumps(sets, sort_keys=True)
    return {"made": time.strftime("%Y-%m-%dT%H:%M:%S"), "decision": "V3-058",
            "method": "free 500-seed windows (no folder K<k>x<base> or W<seed> anywhere in rpp-v3, no other run 5 "
                      "set, no excluded range), round robin over problem families, cut in the order "
                      + ", ".join(stage_set_names()),
            "excluded_ranges": [list(r) for r in EXCLUDED_RANGES],
            "free_windows": {"validation": len(val), "train": len(train)},
            "sets": sets, "sha256": hashlib.sha256(body.encode()).hexdigest()}


@functools.lru_cache(maxsize=None)
def retired_items():
    """Item ids of the retired sealed set (V3-058): no analysis may read their records."""
    return frozenset(i for name in RETIRED for i in interleaved(_bases(name)))


def check_not_retired(items):
    bad = [i for i in items if i in retired_items()]
    if bad:
        raise Retired(f"{len(bad)} items of the retired sealed set (V3-058), e.g. {bad[0]}")


_tls = threading.local()


class _choosing:
    """Lets ladder_seeds read families inside a sealed range while the set's ids are chosen (the ids go into the
    pre-registration before anything is run). Per thread and re-entrant: other threads still find the seal closed."""

    @staticmethod
    def is_active():
        return getattr(_tls, "depth", 0) > 0

    def __enter__(self):
        _tls.depth = getattr(_tls, "depth", 0) + 1

    def __exit__(self, *a):
        _tls.depth = max(0, getattr(_tls, "depth", 1) - 1)


def interleaved(bases):
    """All 4 k of every base, ordered so that any prefix is spread over bases and k (base-major rounds of k)."""
    return [f"K{k}x{b}" for b in bases for k in KS]


def sets():
    main, sealed = _bases("main"), _bases("sealed")
    mini = _bases("mini")
    out = {
        "main": interleaved(main),
        "sealed": interleaved(sealed),
        "pilot": interleaved(_bases("pilot")),
        "mini": [f"K{k}x{mini[0]}" for k in KS] + [f"K1x{mini[1]}", f"K8x{mini[1]}"],
        "mini_sealed": interleaved(_bases("mini_sealed")),
    }
    if _stage_file():                                     # V3-058
        out.update({name: interleaved(_bases(name)) for name in stage_set_names()})
    return out
